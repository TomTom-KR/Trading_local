"""In-process publish/subscribe bus for LIVE NOTIFICATIONS to the web GUI.

The bus is deliberately not used for core control flow: the order in which market data, signal
tracking, the broker and the engine run is decided explicitly by the runtime (deterministic and
testable). Components only *announce* what happened here; the WebSocket hub forwards the events
to connected browsers.

``publish`` may be called from any thread (worker threads use it); delivery happens on the
event loop the bus is bound to.
"""

import asyncio
import contextlib
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Topic names (the WebSocket message "type" is the topic).
TOPIC_QUOTES = "quotes"  # {"items": [QuoteOut, ...]}  (coalesced by the hub)
TOPIC_BAR = "bar"  # {"symbol": str, "bar": [ts, o, h, l, c, v], "source": str}
TOPIC_NEWS = "news"  # NewsOut
TOPIC_SIGNAL = "signal"  # SignalOut (created or updated)
TOPIC_ORDER = "order"  # OrderOut
TOPIC_PORTFOLIO = "portfolio"  # {"key": "ai" | "me"}  -> clients re-fetch the summary
TOPIC_SESSIONS = "sessions"  # {"exchange": str, "phase": str}  on phase changes
TOPIC_STATUS = "status"  # SystemStatus (periodic)
TOPIC_ACTIVITY = "activity"  # ActivityOut
TOPIC_CHART_ANALYSIS = "chart_analysis"  # ChartAnalysisOut
TOPIC_VALIDATION = "validation"  # {"id": int, "status": str}


@dataclass(frozen=True, slots=True)
class Event:
    topic: str
    data: dict[str, Any]


class Subscription:
    """Async iterator over events. Slow consumers lose the OLDEST events instead of blocking publishers."""

    def __init__(self, bus: "EventBus", maxsize: int) -> None:
        self._bus = bus
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def _offer(self, event: Event) -> None:
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self.dropped += 1
        self._queue.put_nowait(event)

    async def get(self) -> Event:
        return await self._queue.get()

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        return await self._queue.get()

    def close(self) -> None:
        self._bus._unsubscribe(self)

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class EventBus:
    def __init__(self) -> None:
        self._subs: set[Subscription] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Bind the bus to the event loop that delivers events (call once at start-up from that loop)."""
        self._loop = loop or asyncio.get_running_loop()

    def subscribe(self, maxsize: int = 2000) -> Subscription:
        sub = Subscription(self, maxsize)
        with self._lock:
            self._subs.add(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            self._subs.discard(sub)

    def publish(self, topic: str, data: dict[str, Any] | None = None) -> None:
        """Announce an event. Never raises and never blocks; safe from any thread.

        Without a bound loop (unit tests, command-line tools) the event is silently dropped.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        event = Event(topic, data or {})
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        try:
            if running is loop:
                self._deliver(event)
            else:
                loop.call_soon_threadsafe(self._deliver, event)
        except RuntimeError:  # loop shut down between the check and the call
            log.debug("event bus: loop closed, dropped %s", topic)

    def _deliver(self, event: Event) -> None:
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            try:
                sub._offer(event)
            except Exception:  # noqa: BLE001 - one broken subscriber must not affect the others
                log.exception("event bus: delivery failed for topic %s", event.topic)
