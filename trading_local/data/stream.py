"""Yahoo Finance price stream (WebSocket), built on the probe's tested replacement for yfinance's client.

Why not ``yfinance.AsyncWebSocket`` (verified with yfinance 1.8.0): it busy-loops and blocks the event loop
when the server closes cleanly, never reconnects after an abnormal close, and ``close()`` hangs while
``listen()`` runs. This module speaks the same protocol with the same libraries (websockets + yfinance's
compiled ``PricingData`` protobuf) and owns reconnects, re-subscription and liveness checks.

Protocol: client text frames ``{"subscribe": [...]}`` / ``{"unsubscribe": [...]}``; server text frames
``{"type": "pricing", "message": "<base64 PricingData>"}``. No acknowledgements, no error frames.

Verified quirks honoured here
* A connection silently serves only the FIRST 100 symbols it subscribed: the symbols are sharded over
  several connections (at most ``shard_size`` each, default 70) and every shard carries a 24/7 crypto canary.
  A shard whose canary goes silent ON ITS OWN CONNECTION (or that receives nothing at all) is reconnected;
  the same canary arriving on another shard's connection does not count.
* 64-bit fields are integers, ``time`` is epoch MILLISECONDS, fields equal to the proto default (0, '') are
  absent - optional fields with value 0 are therefore reported as None.
* Closed markets replay old prices with their old timestamps and some FX crosses send current timestamps
  while closed: the consumer must filter ticks by time and trading calendar.
* ``price`` is a 32-bit float on the wire; it is rounded to the message's ``price_hint`` decimals.
"""

import asyncio
import base64
import contextlib
import json
import logging
import math
import random
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import NoReturn

from websockets.asyncio.client import ClientConnection, connect
from yfinance.pricing_pb2 import PricingData

from trading_local.clock import Clock
from trading_local.data.provider import StateHandler, Tick, TickHandler

log = logging.getLogger(__name__)

STREAM_URL = "wss://streamer.finance.yahoo.com/?version=2"
MAX_SYMBOLS_PER_CONNECTION = 100
CANARIES: tuple[str, ...] = ("BTC-USD", "ETH-USD", "SOL-USD")
MARKET_HOURS: dict[int, str] = {0: "PRE", 1: "REGULAR", 2: "POST", 3: "EXTENDED"}


@dataclass(frozen=True, slots=True)
class StreamOptions:
    url: str = STREAM_URL
    shard_size: int = 70  # symbols per connection without the canary (Yahoo serves only the first 100)
    canaries: tuple[str, ...] = CANARIES
    resubscribe_s: float = 15.0  # re-send the subscription periodically (harmless, as yfinance does)
    idle_timeout_s: float = 45.0  # reconnect if nothing at all arrived for this long
    canary_timeout_s: float = 90.0  # reconnect if the shard's canary was silent for this long
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    open_timeout_s: float = 15.0
    close_timeout_s: float = 3.0
    poll_s: float = 1.0  # granularity of the liveness checks

    def __post_init__(self) -> None:
        if not 1 <= self.shard_size < MAX_SYMBOLS_PER_CONNECTION:
            raise ValueError(f"shard_size must be 1..{MAX_SYMBOLS_PER_CONNECTION - 1}")
        if not self.canaries:
            raise ValueError("at least one canary symbol is needed")


# ------------------------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------------------------
def _round_price(value: float, hint: int) -> float | None:
    if not math.isfinite(value) or value <= 0:
        return None
    return round(value, hint) if hint > 0 else float(f"{value:.7g}")


def decode_message(raw: str | bytes, received_ts: float) -> Tick | None:
    """One server frame -> ``Tick``; None for frames that are not a usable pricing message."""
    try:
        envelope = json.loads(raw)
    except ValueError:
        return None
    payload = envelope.get("message") if isinstance(envelope, dict) else None
    if not isinstance(payload, str) or not payload:
        return None
    data = PricingData()
    try:
        data.ParseFromString(base64.b64decode(payload))
    except Exception:  # noqa: BLE001 - protobuf raises its own DecodeError, base64 binascii.Error
        return None
    hint = int(data.price_hint)
    price = _round_price(data.price, hint)
    if not data.id or price is None or data.time <= 0:
        return None
    return Tick(
        symbol=data.id,
        ts=data.time / 1000.0,
        price=price,
        received_ts=received_ts,
        day_volume=float(data.day_volume) if data.day_volume > 0 else None,
        change_pct=float(data.change_percent) if data.change_percent and math.isfinite(data.change_percent) else None,
        market_state=MARKET_HOURS.get(int(data.market_hours)),
        day_open=_round_price(data.open_price, hint),
        day_high=_round_price(data.day_high, hint),
        day_low=_round_price(data.day_low, hint),
        prev_close=_round_price(data.previous_close, hint),
        bid=_round_price(data.bid, hint),
        ask=_round_price(data.ask, hint),
    )


def shard_symbols(symbols: Sequence[str], shard_size: int, canaries: Sequence[str]) -> list[list[str]]:
    """Split ``symbols`` (order kept, duplicates removed) into balanced shards of at most ``shard_size``
    and append a canary to every shard that has none (shard i gets ``canaries[i % len(canaries)]``)."""
    unique = list(dict.fromkeys(symbols))
    if not unique:
        return [[canaries[0]]]
    count = math.ceil(len(unique) / shard_size)
    size = math.ceil(len(unique) / count)
    shards = [unique[start : start + size] for start in range(0, len(unique), size)]
    for index, shard in enumerate(shards):
        if not any(symbol in canaries for symbol in shard):
            shard.append(canaries[index % len(canaries)])
    return shards


def backoff_delay(attempt: int, initial_s: float, max_s: float, rng: random.Random) -> float:
    """Exponential backoff with jitter: between 50 and 100 % of ``min(max_s, initial_s * 2**attempt)``."""
    return min(max_s, initial_s * 2 ** min(attempt, 30)) * rng.uniform(0.5, 1.0)


class _Stale(Exception):
    """The connection is open but no longer delivers (raised to force a reconnect)."""


# ------------------------------------------------------------------------------------------------
# One connection
# ------------------------------------------------------------------------------------------------
@dataclass(eq=False)
class _Shard:
    index: int
    symbols: list[str]
    options: StreamOptions
    deliver: Callable[["_Shard", str | bytes], None]  # (receiving shard, frame)
    on_connected: Callable[[], None]
    rng: random.Random
    monotonic: Callable[[], float]
    connected: bool = False
    connects: int = 0
    _attempt: int = field(default=0, repr=False)  # failed connection attempts since the last frame received
    _removed: set[str] = field(default_factory=set, repr=False)
    _resubscribe: bool = field(default=False, repr=False)
    _last_canary: float = field(default=0.0, repr=False)

    @property
    def canary(self) -> str:
        return next(symbol for symbol in self.symbols if symbol in self.options.canaries)

    def set_symbols(self, symbols: list[str]) -> None:
        """Change the subscription; applied by the connection loop within ``poll_s``."""
        self._removed = (self._removed | set(self.symbols)) - set(symbols)
        self.symbols = symbols
        self._resubscribe = True

    def saw(self, symbol: str) -> None:
        """Note a decoded update received on THIS shard's connection (liveness of its canary)."""
        if symbol == self.canary:
            self._last_canary = self.monotonic()

    async def run(self) -> None:
        """Connect, subscribe and deliver frames until cancelled; reconnects with backoff on any failure."""
        while True:
            try:
                async with connect(
                    self.options.url,
                    ping_interval=20,
                    ping_timeout=20,
                    open_timeout=self.options.open_timeout_s,
                    close_timeout=self.options.close_timeout_s,
                ) as ws:
                    self.connects += 1
                    await self._session(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - ConnectionClosed, OSError, TimeoutError, _Stale ...
                log.warning("stream shard %d down (%s: %s)", self.index, type(exc).__name__, exc)
            finally:
                if self.connected:
                    self.connected = False
                    self.on_connected()
            delay = backoff_delay(self._attempt, self.options.backoff_initial_s, self.options.backoff_max_s, self.rng)
            self._attempt += 1
            await asyncio.sleep(delay)

    async def _session(self, ws: ClientConnection) -> NoReturn:
        """Serve one open connection until it fails (always ends by raising)."""
        self._removed.clear()
        await ws.send(json.dumps({"subscribe": self.symbols}))
        self._resubscribe = False
        log.info("stream shard %d connected (#%d), %d symbols", self.index, self.connects, len(self.symbols))
        self.connected = True
        self.on_connected()
        now = self.monotonic()
        last_message = last_subscribe = self._last_canary = now
        while True:
            try:
                raw: str | bytes | None = await asyncio.wait_for(ws.recv(), timeout=self.options.poll_s)
            except TimeoutError:
                raw = None
            now = self.monotonic()
            if self._resubscribe or now - last_subscribe >= self.options.resubscribe_s:
                await self._send_subscription(ws)
                last_subscribe = now
            if raw is not None:
                last_message = now
                self._attempt = 0
                self.deliver(self, raw)
            if now - last_message > self.options.idle_timeout_s:
                raise _Stale(f"no message for {self.options.idle_timeout_s:.0f} s")
            if now - self._last_canary > self.options.canary_timeout_s:
                raise _Stale(f"canary {self.canary} silent for {self.options.canary_timeout_s:.0f} s")

    async def _send_subscription(self, ws: ClientConnection) -> None:
        if self._removed:
            await ws.send(json.dumps({"unsubscribe": sorted(self._removed)}))
            self._removed.clear()
        await ws.send(json.dumps({"subscribe": self.symbols}))
        self._resubscribe = False


# ------------------------------------------------------------------------------------------------
# The sharded stream
# ------------------------------------------------------------------------------------------------
class YahooStream:
    """``TickStream`` over one or more sharded Yahoo connections (see the module docstring)."""

    def __init__(
        self,
        on_tick: TickHandler,
        on_state: StateHandler,
        *,
        clock: Clock,
        options: StreamOptions | None = None,
        rng: random.Random | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_tick = on_tick
        self._on_state = on_state
        self._clock = clock
        self._options = options or StreamOptions()
        self._rng = rng or random.Random()
        self._monotonic = monotonic
        self._symbols: list[str] = []
        self._shards: list[_Shard] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._lifecycle = asyncio.Lock()  # serialises connect / subscribe / close (re-sharding awaits)
        self._running = False
        self._closing = False
        self._connected = False
        self._last_time: dict[str, float] = {}  # newest tick time per symbol (drops canary duplicates)

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        async with self._lifecycle:
            if self._running:
                return
            self._running = True
            self._closing = False
            self._start_shards()

    async def subscribe(self, symbols: Collection[str]) -> None:
        """Replace the symbol set: applied on the live connections, or by re-sharding when the number of
        connections changes. Concurrent calls are applied one after the other (the last one wins)."""
        async with self._lifecycle:
            self._symbols = list(dict.fromkeys(symbols))
            if not self._running:
                return
            layout = shard_symbols(self._symbols, self._options.shard_size, self._options.canaries)
            if len(layout) == len(self._shards):
                for shard, shard_symbols_ in zip(self._shards, layout, strict=True):
                    if shard_symbols_ != shard.symbols:
                        shard.set_symbols(shard_symbols_)
                return
            await self._stop_shards()
            self._start_shards()

    async def close(self) -> None:
        async with self._lifecycle:
            self._closing = True
            self._running = False
            await self._stop_shards()
            self._connected = False

    # -- internals ------------------------------------------------------------------------------
    def _start_shards(self) -> None:
        layout = shard_symbols(self._symbols, self._options.shard_size, self._options.canaries)
        self._shards = [
            _Shard(
                index=index,
                symbols=symbols,
                options=self._options,
                deliver=self._deliver,
                on_connected=self._update_state,
                rng=self._rng,
                monotonic=self._monotonic,
            )
            for index, symbols in enumerate(layout)
        ]
        self._tasks = [
            asyncio.create_task(shard.run(), name=f"yahoo-stream-{shard.index}") for shard in self._shards
        ]

    async def _stop_shards(self) -> None:
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=self._options.close_timeout_s + 2
                )
        self._shards = []
        self._update_state()

    def _deliver(self, shard: _Shard, raw: str | bytes) -> None:
        """Decode a frame received by ``shard``; credit its canary, drop duplicates, hand the tick on."""
        tick = decode_message(raw, self._clock.now().timestamp())
        if tick is None:
            return
        shard.saw(tick.symbol)
        if self._last_time.get(tick.symbol) == tick.ts:
            return
        self._last_time[tick.symbol] = max(tick.ts, self._last_time.get(tick.symbol, tick.ts))
        try:
            self._on_tick(tick)
        except Exception:  # noqa: BLE001 - a handler bug must not look like a dead connection
            log.exception("tick handler failed for %s", tick.symbol)

    def _update_state(self) -> None:
        connected = bool(self._shards) and all(shard.connected for shard in self._shards)
        if connected == self._connected:
            return
        self._connected = connected
        if self._closing:
            return
        try:
            self._on_state(connected)
        except Exception:  # noqa: BLE001
            log.exception("stream state handler failed")


__all__ = ["CANARIES", "StreamOptions", "YahooStream", "backoff_delay", "decode_message", "shard_symbols"]
