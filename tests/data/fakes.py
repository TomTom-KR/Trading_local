"""Test doubles for the market-data module: a scriptable Provider and TickStream."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading_local.clock import Clock, floor_ts
from trading_local.data.provider import RateLimitError, StateHandler, Tick, TickHandler
from trading_local.db.repo_market import BAR_COLUMNS, BAR_SECONDS
from trading_local.models import Instrument, Quote


@dataclass(frozen=True, slots=True)
class Call:
    kind: str  # 'minutes' | 'daily' | 'quotes' | 'lookup'
    symbols: tuple[str, ...]
    at: int  # clock time of the call
    start_ts: int | None = None
    end_ts: int | None = None
    prepost: bool = False

    @property
    def requests(self) -> int:
        """HTTP requests the real provider would make: one per symbol for bars, one for a quote batch."""
        return 1 if self.kind in ("quotes", "lookup") else len(self.symbols)


def minute_frame(start_ts: int, end_ts: int, *, price: float = 100.0, volume: float = 10.0) -> pd.DataFrame:
    """Canonical 1-minute bars for every minute with open time in [start_ts, end_ts)."""
    ts = np.arange(floor_ts(start_ts) + (BAR_SECONDS if start_ts % BAR_SECONDS else 0), end_ts, BAR_SECONDS)
    base = price + (ts % 3600) / 1000.0
    data = {"open": base, "high": base + 0.5, "low": base - 0.5, "close": base + 0.1, "volume": np.full(len(ts), volume)}
    return pd.DataFrame(data, index=pd.Index(ts.astype(np.int64), name="ts"), columns=list(BAR_COLUMNS))


def daily_frame(days: Sequence[str], *, price: float = 100.0) -> pd.DataFrame:
    values = np.array([[price, price + 1, price - 1, price + 0.5, 1000.0] for _ in days]).reshape(-1, 5)
    return pd.DataFrame(values, index=pd.Index(list(days), dtype="str", name="day"), columns=list(BAR_COLUMNS))


class FakeStream:
    """``TickStream`` double: records subscriptions; ``emit`` delivers a tick to the service."""

    def __init__(self, on_tick: TickHandler, on_state: StateHandler) -> None:
        self.on_tick = on_tick
        self.on_state = on_state
        self.subscriptions: list[list[str]] = []
        self.connect_calls = 0
        self.closed = False
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self.connect_calls += 1

    async def subscribe(self, symbols: Collection[str]) -> None:
        self.subscriptions.append(list(symbols))

    async def close(self) -> None:
        self.closed = True

    def set_connected(self, connected: bool) -> None:
        self._connected = connected
        self.on_state(connected)

    def emit(self, tick: Tick) -> None:
        self.on_tick(tick)


class FakeProvider:
    """``Provider`` double.

    By default every minute request returns synthetic bars for each minute from ``start_ts`` up to and
    INCLUDING the current (still forming) minute of the clock, like Yahoo does for a live market.
    ``minutes_for`` / ``daily_for`` replace the generators per symbol; ``quotes`` is the snapshot answer;
    ``fail`` (an exception) is raised by the next REST call(s) while set.
    """

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.calls: list[Call] = []
        self.quotes: dict[str, Quote] = {}
        self.instruments: dict[str, Instrument] = {}
        self.minutes_for: Callable[[str, int, int | None, bool], pd.DataFrame | None] = self._default_minutes
        self.daily_for: Callable[[str, int], pd.DataFrame | None] = lambda symbol, start_ts: None
        self.fail: Exception | None = None
        self.paused: int | None = None
        self.trips: list[str] = []
        self.stream: FakeStream | None = None
        self.closed = False

    # -- helpers for tests ------------------------------------------------------------------------
    def calls_of(self, kind: str) -> list[Call]:
        return [call for call in self.calls if call.kind == kind]

    def requested(self, kind: str = "minutes") -> list[str]:
        return [symbol for call in self.calls_of(kind) for symbol in call.symbols]

    def _default_minutes(self, symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        end = end_ts if end_ts is not None else floor_ts(self.clock.ts()) + BAR_SECONDS
        return minute_frame(start_ts, end)

    def _check(self) -> None:
        if self.paused is not None and self.paused > self.clock.ts():
            raise RateLimitError("paused")
        if self.fail is not None:
            raise self.fail

    # -- Provider ---------------------------------------------------------------------------------
    async def fetch_minutes(
        self, symbols: Sequence[str], *, start_ts: int, end_ts: int | None = None, prepost: bool = False
    ) -> dict[str, pd.DataFrame]:
        self._check()
        self.calls.append(Call("minutes", tuple(symbols), self.clock.ts(), start_ts, end_ts, prepost))
        frames = {symbol: self.minutes_for(symbol, start_ts, end_ts, prepost) for symbol in symbols}
        return {symbol: frame for symbol, frame in frames.items() if frame is not None and not frame.empty}

    async def fetch_daily(self, symbols: Sequence[str], *, start_ts: int) -> dict[str, pd.DataFrame]:
        self._check()
        self.calls.append(Call("daily", tuple(symbols), self.clock.ts(), start_ts))
        frames = {symbol: self.daily_for(symbol, start_ts) for symbol in symbols}
        return {symbol: frame for symbol, frame in frames.items() if frame is not None and not frame.empty}

    async def fetch_quotes(self, symbols: Sequence[str]) -> list[Quote]:
        self._check()
        self.calls.append(Call("quotes", tuple(symbols), self.clock.ts()))
        return [self.quotes[symbol] for symbol in symbols if symbol in self.quotes]

    async def lookup(self, symbol: str) -> Instrument | None:
        self._check()
        self.calls.append(Call("lookup", (symbol,), self.clock.ts()))
        return self.instruments.get(symbol)

    def open_stream(self, on_tick: TickHandler, on_state: StateHandler) -> FakeStream:
        self.stream = FakeStream(on_tick, on_state)
        return self.stream

    def paused_until(self) -> int | None:
        return self.paused if self.paused is not None and self.paused > self.clock.ts() else None

    def trip(self, reason: str) -> None:
        self.trips.append(reason)
        self.paused = self.clock.ts() + 600

    async def close(self) -> None:
        self.closed = True
