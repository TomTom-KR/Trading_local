"""WebSocket hub: forwards the EventBus to the connected browsers.

Wire format server -> client: ``{"type": <topic>, "data": {...}}`` for every bus topic, plus
``{"type": "hello", "data": {"server_time": ts, "version": ...}}`` on connect, ``{"type": "ping"}`` every
20 s and ``{"type": "status", "data": <SystemStatusOut>}`` every 5 s while at least one client is connected
(the hub is the only source of 'status').

Session phases: with every status push the hub also asks the trading calendar for the phase of each venue
that has an instrument and sends ``{"type": "sessions", "data": {"exchange": ..., "phase": ...}}`` for
every venue whose phase changed since the previous push (Xetra opens, Tokyo goes to lunch ...). The GUI
reloads its session overview and instrument list on such a message.

* ONE bus subscription is fanned out to all clients.
* Every client has a bounded outbox: a slow client loses its OLDEST messages and never blocks the hub;
  a client whose socket stays blocked for ``send_timeout_s`` is disconnected.
* 'quotes' are coalesced per client (latest quote per symbol) and flushed at most every ``quote_flush_s``.
* 'bar' goes only to clients that subscribed to the symbol with ``{"type": "subscribe", "symbols": [...]}``
  (replace semantics, at most 20 symbols). Unknown client messages are ignored.
"""

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import orjson
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.websockets import WebSocket, WebSocketDisconnect

import trading_local
from trading_local.dto import SystemStatusOut
from trading_local.events import TOPIC_BAR, TOPIC_QUOTES, TOPIC_SESSIONS, TOPIC_STATUS, Event, Subscription
from trading_local.services import Services
from trading_local.web.deps import MAX_SYMBOL_LENGTH, ApiContext, cancel_and_wait, normalize_symbol
from trading_local.web.status import compose_status, venue_phases

log = logging.getLogger(__name__)

MAX_SUBSCRIBED_SYMBOLS = 20
MAX_CLIENT_MESSAGE_CHARS = 4096
_JSON_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
_CLOSE_GOING_AWAY = 1001
_CLOSE_TRY_AGAIN_LATER = 1013
_CLOSE_TIMEOUT_S = 1.0

router = APIRouter()


@dataclass(frozen=True, slots=True)
class HubTiming:
    status_interval_s: float = 5.0
    ping_interval_s: float = 20.0
    quote_flush_s: float = 0.5
    send_timeout_s: float = 10.0
    outbox_size: int = 500


@dataclass(eq=False, slots=True)
class _Client:
    websocket: WebSocket
    outbox: deque[str]
    wakeup: asyncio.Event = field(default_factory=asyncio.Event)
    symbols: frozenset[str] = frozenset()
    pending_quotes: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    dropped: int = 0  # messages lost because the outbox was full
    closing: bool = False  # set by Hub.stop()
    blocked: bool = False  # a send did not complete within the timeout

    def offer(self, text: str) -> None:
        """Queue a serialised message; a full outbox drops its oldest message."""
        if len(self.outbox) == self.outbox.maxlen:
            self.dropped += 1
        self.outbox.append(text)
        self.wakeup.set()


def _json_default(value: Any) -> Any:
    """Serialise pydantic models that a publisher put into an event payload."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    raise TypeError(f"type {type(value).__name__} is not JSON serializable")


def encode_message(message_type: str, data: Any = None) -> str | None:
    """The JSON text of one message (None, logged, if the payload cannot be serialised)."""
    message: dict[str, Any] = {"type": message_type} if data is None else {"type": message_type, "data": data}
    try:
        return orjson.dumps(message, default=_json_default, option=_JSON_OPTIONS).decode()
    except TypeError:
        log.warning("dropped a %r message that cannot be serialised", message_type, exc_info=True)
        return None


class Hub:
    def __init__(self, ctx: ApiContext, timing: HubTiming | None = None) -> None:
        self.timing = timing or HubTiming()
        self._ctx = ctx
        self._clients: set[_Client] = set()
        self._subscription: Subscription | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._phases: dict[str, str] | None = None  # venue -> phase at the previous status push

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def running(self) -> bool:
        return self._subscription is not None

    # -- life cycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        """Subscribe to the bus (which must be bound to the running loop) and start the timers."""
        if self.running:
            return
        self._subscription = subscription = self._ctx.services.bus.subscribe()
        timing = self.timing
        self._tasks = [
            asyncio.create_task(self._pump(subscription), name="ws-hub-pump"),
            asyncio.create_task(self._every(timing.quote_flush_s, self._flush_quotes), name="ws-quotes"),
            asyncio.create_task(self._every(timing.ping_interval_s, self._ping), name="ws-ping"),
            asyncio.create_task(self._every(timing.status_interval_s, self._push_status), name="ws-status"),
        ]

    async def stop(self) -> None:
        """Stop the timers, leave the bus and close every client connection (code 1001)."""
        if self._subscription is not None:
            self._subscription.close()
            self._subscription = None
        await cancel_and_wait(self._tasks)
        self._tasks = []
        for client in list(self._clients):
            client.closing = True
            client.wakeup.set()
            await _close_quietly(client.websocket, _CLOSE_GOING_AWAY)

    # -- one connection -------------------------------------------------------------------------
    async def serve(self, websocket: WebSocket) -> None:
        """Run one client connection until it disconnects, is too slow or the hub stops."""
        if not self.running:
            await websocket.close(code=_CLOSE_TRY_AGAIN_LATER)
            return
        await websocket.accept()
        if not self.running:  # the hub stopped while the handshake completed: stop() did not see this client
            await _close_quietly(websocket, _CLOSE_GOING_AWAY)
            return
        client = _Client(websocket, deque(maxlen=self.timing.outbox_size))
        hello = {"server_time": self._ctx.services.clock.ts(), "version": trading_local.__version__}
        if (text := encode_message("hello", hello)) is not None:
            client.offer(text)
        self._clients.add(client)
        reader = asyncio.create_task(self._read(client), name="ws-read")
        writer = asyncio.create_task(self._write(client), name="ws-write")
        try:
            await asyncio.wait({reader, writer}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._clients.discard(client)
            await cancel_and_wait([reader, writer])
            for task in (reader, writer):
                if not task.cancelled() and (exc := task.exception()) is not None:
                    log.error("websocket connection failed", exc_info=exc)
            if client.blocked:
                await _close_quietly(websocket, _CLOSE_TRY_AGAIN_LATER)

    async def _read(self, client: _Client) -> None:
        while True:
            message = await client.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            if isinstance(text, str) and len(text) <= MAX_CLIENT_MESSAGE_CHARS:
                self._handle_client_message(client, text)

    async def _write(self, client: _Client) -> None:
        """Send queued messages until the connection is gone, the hub stops or a send stays blocked."""
        while not client.closing:
            await client.wakeup.wait()
            client.wakeup.clear()
            while client.outbox and not client.closing:
                text = client.outbox.popleft()
                try:
                    async with asyncio.timeout(self.timing.send_timeout_s):
                        await client.websocket.send_text(text)
                except TimeoutError:
                    log.warning("websocket client blocked for %.1f s: disconnecting", self.timing.send_timeout_s)
                    client.blocked = True
                    return
                except (WebSocketDisconnect, RuntimeError, OSError):
                    return  # the connection is gone already

    @staticmethod
    def _handle_client_message(client: _Client, text: str) -> None:
        """Apply a subscription; anything else (or malformed JSON) is ignored."""
        try:
            message = orjson.loads(text)
        except orjson.JSONDecodeError:
            return
        if not isinstance(message, dict) or message.get("type") != "subscribe":
            return
        symbols = message.get("symbols")
        if not isinstance(symbols, list):
            return
        wanted = [
            normalize_symbol(symbol)
            for symbol in symbols
            if isinstance(symbol, str) and 0 < len(symbol.strip()) <= MAX_SYMBOL_LENGTH
        ]
        client.symbols = frozenset(list(dict.fromkeys(wanted))[:MAX_SUBSCRIBED_SYMBOLS])

    # -- fan-out --------------------------------------------------------------------------------
    async def _pump(self, subscription: Subscription) -> None:
        async for event in subscription:
            try:
                self._dispatch(event)
            except Exception:
                log.exception("websocket hub: could not forward a %r event", event.topic)

    def _dispatch(self, event: Event) -> None:
        if event.topic == TOPIC_STATUS:
            return  # the hub is the only source of status messages
        if event.topic == TOPIC_QUOTES:
            quotes = _quotes_by_symbol(event.data)
            for client in self._clients:
                client.pending_quotes.update(quotes)
            return
        text = encode_message(event.topic, event.data)
        if text is None:
            return
        if event.topic == TOPIC_BAR:
            symbol = event.data.get("symbol")
            for client in self._clients:
                if symbol in client.symbols:
                    client.offer(text)
            return
        self._broadcast(text)

    def _broadcast(self, text: str) -> None:
        for client in self._clients:
            client.offer(text)

    async def _every(self, interval_s: float, action: Callable[[], Awaitable[None]]) -> None:
        while True:
            await asyncio.sleep(interval_s)
            try:
                await action()
            except Exception:
                log.exception("websocket hub: periodic task failed")

    async def _flush_quotes(self) -> None:
        for client in self._clients:
            if client.pending_quotes:
                items = list(client.pending_quotes.values())
                client.pending_quotes = {}
                text = encode_message(TOPIC_QUOTES, {"items": items})
                if text is not None:
                    client.offer(text)

    async def _ping(self) -> None:
        text = encode_message("ping")
        if text is not None:
            self._broadcast(text)

    async def _push_status(self) -> None:
        if not self._clients:
            self._phases = None  # nobody listened meanwhile: the next client loads fresh sessions anyway
            return
        status, phases = await self._ctx.run(_status_and_phases, self._ctx.services)
        text = encode_message(TOPIC_STATUS, status.model_dump(mode="json"))
        if text is not None:
            self._broadcast(text)
        self._announce_phase_changes(phases)

    def _announce_phase_changes(self, phases: dict[str, str]) -> None:
        """One 'sessions' message per venue whose phase differs from the previous push (none on the first).
        An empty result (the calendar failed) keeps the previous phases, so no change gets lost."""
        if not phases:
            return
        previous, self._phases = self._phases, phases
        if previous is None:
            return
        for exchange, phase in phases.items():
            if previous.get(exchange, phase) == phase:
                continue
            text = encode_message(TOPIC_SESSIONS, {"exchange": exchange, "phase": phase})
            if text is not None:
                self._broadcast(text)


def _status_and_phases(services: Services) -> tuple[SystemStatusOut, dict[str, str]]:
    """Blocking: the system status and the session phase of every venue, in one executor call."""
    return compose_status(services), venue_phases(services)


def _quotes_by_symbol(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """symbol -> quote of a 'quotes' payload; items may be dicts or ``QuoteOut`` models, broken ones are skipped."""
    items = data.get("items")
    if not isinstance(items, list):
        return {}
    quotes: dict[str, Mapping[str, Any]] = {}
    for item in items:
        quote = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
        if isinstance(quote, Mapping) and isinstance(quote.get("symbol"), str):
            quotes[quote["symbol"]] = quote
    return quotes


async def _close_quietly(websocket: WebSocket, code: int) -> None:
    """Close without waiting long: a blocked socket may never accept the close frame."""
    with contextlib.suppress(Exception):
        async with asyncio.timeout(_CLOSE_TIMEOUT_S):
            await websocket.close(code=code)


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    hub: Hub = websocket.app.state.hub
    await hub.serve(websocket)
