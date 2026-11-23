"""Kraken WebSocket v2 stream (public, no key): live quotes from ``ticker``, closed 1-minute candles from ``ohlc``.

Protocol (docs.kraken.com; verified in a live session on 2026-11-22)
* URL ``wss://ws.kraken.com/v2``. Requests ``{"method": "subscribe" | "unsubscribe", "params": {"channel": ...,
  "symbol": [...], ...}, "req_id": n}``; Kraken acknowledges every symbol separately (``"success": true``), a
  refused symbol with ``"success": false, "error": "Currency pair not supported NOPE/EUR", "symbol": ...``.
* ``status`` channel (``system`` 'online' | 'maintenance' | 'cancel_only' | 'post_only'), repeated now and then.
* ``{"channel": "heartbeat"}`` about once per second while no other update is sent - only after a subscription.
* Application ping ``{"method": "ping", "req_id": n}`` is answered by ``{"method": "pong", "req_id": n, ...}``.
* ``ohlc`` with ``interval`` 1: a snapshot of the newest 10 candles WITH trades (the last one still forming),
  then an update carrying the complete in-progress candle on every trade. Candle fields: symbol, open, high,
  low, close, vwap, trades, volume, interval_begin (RFC 3339 with nanoseconds), interval, timestamp (deprecated).
  There is no "closed" flag, and no update arrives after a minute has ended (checked live). A candle is
  therefore emitted as CLOSED bar (source 'stream') when a candle of a later minute arrives for the symbol or
  ``close_grace_s`` after its minute ended; a later update of an emitted minute is emitted again (correction).
  Unfinished candles are discarded when the connection drops: the snapshot after the reconnect repeats them.
* ``ticker`` (``event_trigger`` 'trades', the default): symbol, bid, bid_qty, ask, ask_qty, last, volume, vwap,
  low, high, change, change_pct, timestamp. Volume, high, low and change are ROLLING 24 h figures, whereas the
  quote book keeps the UTC-day values of the REST ticker - so a tick carries only last price, bid, ask and time.

Liveness: an application ping every ``ping_interval_s`` (a missing pong after ``pong_timeout_s`` reconnects),
the protocol-level ping of ``websockets``, and a watchdog that reconnects when no market-data frame (heartbeat,
status, ticker, ohlc - acknowledgements and pongs do not count) arrived for ``idle_timeout_s``. Reconnects use exponential backoff with jitter; ``close`` cancels
the connection task and waits a bounded time.
"""

import asyncio
import contextlib
import json
import logging
import math
import random
import re
import time
from collections.abc import Callable, Collection
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from websockets.asyncio.client import connect

from trading_local.clock import Clock
from trading_local.data.provider import BarHandler, StateHandler, Tick, TickHandler
from trading_local.db.repo_market import BAR_SECONDS
from trading_local.models import Bar, BarSource

log = logging.getLogger(__name__)

WS_URL: Final = "wss://ws.kraken.com/v2"
_RFC3339: Final = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$")


@dataclass(frozen=True, slots=True)
class KrakenStreamOptions:
    url: str = WS_URL
    ping_interval_s: float = 20.0  # application-level ping
    pong_timeout_s: float = 10.0
    idle_timeout_s: float = 15.0  # heartbeats arrive about once per second
    # A candle counts as closed this long after its minute ended (Kraken's feed latency is ~30 ms; the minute
    # cycle's refresh runs 3 s after the boundary and must find the closed candle).
    close_grace_s: float = 1.5
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    open_timeout_s: float = 15.0
    close_timeout_s: float = 3.0
    poll_s: float = 0.5  # granularity of the liveness checks and of the candle closing
    max_size: int = 4 * 1024 * 1024


type Connector = Callable[[KrakenStreamOptions], AbstractAsyncContextManager[Any]]


def default_connector(options: KrakenStreamOptions) -> AbstractAsyncContextManager[Any]:
    return connect(
        options.url,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=options.open_timeout_s,
        close_timeout=options.close_timeout_s,
        max_size=options.max_size,
    )


# ------------------------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------------------------
def rfc3339_ts(text: Any) -> float | None:
    """Epoch seconds of an RFC 3339 time with any number of fractional digits ('...T23:13:00.000000000Z')."""
    if not isinstance(text, str):
        return None
    match = _RFC3339.match(text.strip())
    if match is None:
        return None
    day, clock, fraction, zone = match.groups()
    try:
        base = datetime.fromisoformat(f"{day}T{clock}{'+00:00' if zone == 'Z' else zone}")
    except ValueError:
        return None
    return base.timestamp() + (float(f"0.{fraction}") if fraction else 0.0)


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def decode_candle(item: Any, symbol_of: Callable[[Any], str | None], source: BarSource = "stream") -> Bar | None:
    """One ``ohlc`` candle (interval 1) -> ``Bar`` at its interval start; None if unusable or without trades."""
    if not isinstance(item, dict) or item.get("interval", 1) != 1:
        return None
    symbol = symbol_of(item.get("symbol"))
    if symbol is None:
        return None
    begin = rfc3339_ts(item.get("interval_begin"))
    if begin is None:
        end = rfc3339_ts(item.get("timestamp"))  # deprecated field: the END of the interval
        begin = None if end is None else end - BAR_SECONDS
    trades = item.get("trades")
    if begin is None or int(begin) % BAR_SECONDS or (isinstance(trades, int) and trades <= 0):
        return None
    prices = [_positive(item.get(key)) for key in ("open", "high", "low", "close")]
    volume = item.get("volume")
    if any(price is None for price in prices) or isinstance(volume, bool) or not isinstance(volume, (int, float)):
        return None
    open_, high, low, close = prices  # type: ignore[misc]
    if not (low <= min(open_, close) and high >= max(open_, close)) or not math.isfinite(volume):  # type: ignore[operator]
        return None
    return Bar(symbol, int(begin), open_, high, low, close, max(0.0, float(volume)), source)  # type: ignore[arg-type]


def decode_ticker(item: Any, received_ts: float, symbol_of: Callable[[Any], str | None]) -> Tick | None:
    """One ``ticker`` entry -> ``Tick`` with last price, bid, ask and Kraken's time (see the module docstring)."""
    if not isinstance(item, dict):
        return None
    symbol = symbol_of(item.get("symbol"))
    price = _positive(item.get("last"))
    if symbol is None or price is None:
        return None
    ts = rfc3339_ts(item.get("timestamp"))
    return Tick(
        symbol=symbol,
        ts=ts if ts is not None else received_ts,
        price=price,
        received_ts=received_ts,
        bid=_positive(item.get("bid")),
        ask=_positive(item.get("ask")),
    )


def backoff_delay(attempt: int, initial_s: float, max_s: float, rng: random.Random) -> float:
    """Exponential backoff with jitter: between 50 and 100 % of ``min(max_s, initial_s * 2**attempt)``."""
    return min(max_s, initial_s * 2 ** min(attempt, 30)) * rng.uniform(0.5, 1.0)


class CandleTracker:
    """Turns the ``ohlc`` updates of in-progress candles into CLOSED bars (pure, one event loop).

    ``update`` returns the bars an update closed (the previous minute of the symbol) or corrected (a minute
    already emitted, or an older candle of a snapshot); ``due(now)`` the candles whose minute ended at least
    ``close_grace_s`` ago. An identical candle is never emitted twice; corrections are kept for
    ``retention_s`` (counted back from the symbol's newest emitted minute).
    """

    def __init__(self, *, close_grace_s: float = 2.0, retention_s: int = 900) -> None:
        self._grace = close_grace_s
        self._retention_s = retention_s
        self._open: dict[str, Bar] = {}  # newest candle per symbol, not emitted yet
        self._emitted: dict[str, dict[int, Bar]] = {}

    def update(self, bar: Bar, now: float) -> list[Bar]:
        closed: list[Bar] = []
        current = self._open.get(bar.symbol)
        if current is not None and bar.ts > current.ts:
            closed += self._emit(self._open.pop(bar.symbol))
            current = None
        if current is not None and bar.ts < current.ts:  # an older minute (snapshot fill-in or correction)
            return closed + self._emit(bar)
        if current is None and bar.ts in self._emitted.get(bar.symbol, {}):
            return closed + self._emit(bar)  # a late update of a minute that was already emitted
        self._open[bar.symbol] = bar
        return closed + self.due(now)

    def due(self, now: float) -> list[Bar]:
        ready = [bar for bar in self._open.values() if bar.ts + BAR_SECONDS + self._grace <= now]
        emitted: list[Bar] = []
        for bar in sorted(ready, key=lambda b: (b.ts, b.symbol)):
            del self._open[bar.symbol]
            emitted += self._emit(bar)
        return emitted

    def drop_open(self) -> None:
        """Forget the unfinished candles (the connection dropped; the next snapshot repeats them)."""
        self._open.clear()

    def _emit(self, bar: Bar) -> list[Bar]:
        emitted = self._emitted.setdefault(bar.symbol, {})
        if emitted.get(bar.ts) == bar:
            return []
        emitted[bar.ts] = bar
        newest = max(emitted)
        for minute in [minute for minute in emitted if minute < newest - self._retention_s]:
            del emitted[minute]
        return [bar]


class _Stale(Exception):
    """The connection is open but no longer delivers (raised to force a reconnect)."""


# ------------------------------------------------------------------------------------------------
# The stream
# ------------------------------------------------------------------------------------------------
class KrakenStream:
    """``TickStream`` over one Kraken WebSocket v2 connection (see the module docstring).

    ``resolve`` maps our symbol ('BTC-EUR') to the WS symbol ('BTC/EUR'); symbols it does not know are ignored.
    Without ``on_bar`` only the ticker is subscribed. Handlers run on the event loop; their exceptions are
    logged and never stop the stream.
    """

    def __init__(
        self,
        on_tick: TickHandler,
        on_bar: BarHandler | None,
        on_state: StateHandler,
        *,
        clock: Clock,
        resolve: Callable[[str], str | None],
        options: KrakenStreamOptions | None = None,
        connector: Connector | None = None,
        rng: random.Random | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_tick = on_tick
        self._on_bar = on_bar
        self._on_state = on_state
        self._clock = clock
        self._resolve = resolve
        self._options = options or KrakenStreamOptions()
        self._connector: Connector = connector or default_connector
        self._rng = rng or random.Random()
        self._monotonic = monotonic
        self._tracker = CandleTracker(close_grace_s=self._options.close_grace_s)
        self._wanted: dict[str, str] = {}  # WS symbol -> our symbol
        self._subscribed: set[str] = set()  # WS symbols subscribed on the live connection
        self._dirty = False
        self._have_symbols = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._connected = False
        self._closing = False
        self._attempt = 0  # failed connection attempts since the last data message
        self._req_id = 0
        self._ping_sent: float | None = None
        self._system = "online"
        self._warned: set[str] = set()
        self.connects = 0
        self.last_message_ts: float | None = None  # wall-clock time of the newest message (status, tests)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def channels(self) -> tuple[str, ...]:
        return ("ohlc", "ticker") if self._on_bar is not None else ("ticker",)

    async def connect(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._closing = False
        self._task = asyncio.create_task(self._run(), name="kraken-stream")

    async def subscribe(self, symbols: Collection[str]) -> None:
        wanted: dict[str, str] = {}
        for symbol in dict.fromkeys(symbols):
            ws_symbol = self._resolve(symbol)
            if ws_symbol is None:
                self._warn_once(f"unknown:{symbol}", "Kraken stream: no Kraken pair for %s", symbol)
                continue
            wanted[ws_symbol] = symbol
        self._wanted = wanted
        self._dirty = True
        if wanted:
            self._have_symbols.set()
        else:
            self._have_symbols.clear()

    async def close(self) -> None:
        self._closing = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), self._options.close_timeout_s + 2)
        self._connected = False

    # -- connection loop ------------------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            await self._have_symbols.wait()
            failed = True
            try:
                async with self._connector(self._options) as ws:
                    self.connects += 1
                    await self._session(ws)
                    failed = False  # the symbol set became empty: closed on purpose
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - ConnectionClosed, OSError, TimeoutError, _Stale ...
                log.warning("Kraken stream down (%s: %s)", type(exc).__name__, exc)
            finally:
                self._subscribed.clear()
                self._tracker.drop_open()
                self._ping_sent = None
                if self._connected:
                    self._set_connected(False)
            if not failed:
                continue
            delay = backoff_delay(self._attempt, self._options.backoff_initial_s, self._options.backoff_max_s, self._rng)
            self._attempt += 1
            await asyncio.sleep(delay)

    async def _session(self, ws: Any) -> None:
        """Serve one open connection; returns when no symbol is wanted any more, raises on any failure."""
        await self._sync(ws)
        log.info("Kraken stream connected (#%d), %d pairs, channels %s", self.connects, len(self._wanted),
                 "+".join(self.channels))  # fmt: skip
        self._set_connected(True)
        now = self._monotonic()
        last_message = last_ping = now
        while True:
            if not self._wanted:
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._options.poll_s)
            except TimeoutError:
                raw = None
            now = self._monotonic()
            if self._dirty:
                await self._sync(ws)
            if raw is not None and self._handle(raw):  # acks and pongs do not prove that data flows
                last_message = now
                self._attempt = 0
            self._deliver_bars(self._tracker.due(self._clock.now().timestamp()))
            self._check_liveness(now, last_message)
            if now - last_ping >= self._options.ping_interval_s:
                self._req_id += 1
                await ws.send(json.dumps({"method": "ping", "req_id": self._req_id}))
                last_ping = now
                if self._ping_sent is None:
                    self._ping_sent = now

    def _check_liveness(self, now: float, last_message: float) -> None:
        if now - last_message > self._options.idle_timeout_s:
            raise _Stale(f"no message for {self._options.idle_timeout_s:.0f} s")
        if self._ping_sent is not None and now - self._ping_sent > self._options.pong_timeout_s:
            raise _Stale(f"no pong within {self._options.pong_timeout_s:.0f} s")

    async def _sync(self, ws: Any) -> None:
        """Bring the live subscriptions in line with the wanted symbols (both channels)."""
        self._dirty = False
        wanted = set(self._wanted)
        removed = sorted(self._subscribed - wanted)
        added = sorted(wanted - self._subscribed)
        for method, symbols in (("unsubscribe", removed), ("subscribe", added)):
            if not symbols:
                continue
            for channel in self.channels:
                params: dict[str, Any] = {"channel": channel, "symbol": symbols}
                if channel == "ohlc":
                    params["interval"] = 1
                if method == "subscribe":
                    params["snapshot"] = True
                self._req_id += 1
                await ws.send(json.dumps({"method": method, "params": params, "req_id": self._req_id}))
        self._subscribed = wanted

    # -- messages -------------------------------------------------------------------------------
    def _symbol_of(self, ws_symbol: Any) -> str | None:
        return self._wanted.get(ws_symbol) if isinstance(ws_symbol, str) else None

    def _handle(self, raw: str | bytes) -> bool:
        """Process one frame; True for market-data frames (heartbeat, status, ohlc, ticker)."""
        received = self._clock.now().timestamp()
        self.last_message_ts = received
        try:
            message = json.loads(raw)
        except ValueError:
            return False
        if not isinstance(message, dict):
            return False
        channel = message.get("channel")
        data = message.get("data")
        items = data if isinstance(data, list) else []
        if channel == "ohlc":
            for item in items:
                bar = decode_candle(item, self._symbol_of)
                if bar is not None:
                    self._deliver_bars(self._tracker.update(bar, received))
            return True
        if channel == "ticker":
            for item in items:
                tick = decode_ticker(item, received, self._symbol_of)
                if tick is not None:
                    self._deliver_tick(tick)
            return True
        if channel == "status":
            system = items[0].get("system") if items and isinstance(items[0], dict) else None
            if isinstance(system, str) and system != self._system:
                log.log(logging.INFO if system == "online" else logging.WARNING, "Kraken system status: %s", system)
                self._system = system
            return True
        if channel == "heartbeat":
            return True
        method = message.get("method")
        if method == "pong":
            self._ping_sent = None
        elif method in ("subscribe", "unsubscribe") and message.get("success") is False:
            symbol = message.get("symbol") or (message.get("result") or {}).get("symbol")
            self._warn_once(
                f"{method}:{symbol}:{message.get('error')}", "Kraken stream %s refused for %s: %s",
                method, symbol, message.get("error"),
            )  # fmt: skip
        return False

    def _deliver_bars(self, bars: list[Bar]) -> None:
        if self._on_bar is None:
            return
        for bar in bars:
            try:
                self._on_bar(bar)
            except Exception:  # noqa: BLE001 - a handler bug must not look like a dead connection
                log.exception("candle handler failed for %s", bar.symbol)

    def _deliver_tick(self, tick: Tick) -> None:
        try:
            self._on_tick(tick)
        except Exception:  # noqa: BLE001
            log.exception("tick handler failed for %s", tick.symbol)

    def _set_connected(self, connected: bool) -> None:
        if connected == self._connected:
            return
        self._connected = connected
        if self._closing:
            return
        try:
            self._on_state(connected)
        except Exception:  # noqa: BLE001
            log.exception("stream state handler failed")

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning(message, *args)


__all__ = [
    "CandleTracker",
    "KrakenStream",
    "KrakenStreamOptions",
    "WS_URL",
    "backoff_delay",
    "decode_candle",
    "decode_ticker",
    "default_connector",
    "rfc3339_ts",
]
