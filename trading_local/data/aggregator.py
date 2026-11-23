"""Turns price ticks into 1-minute bars (pure, synchronous, not thread-safe: use it from one thread).

Rules
* A tick belongs to the minute of its SOURCE time (``Tick.ts``), never to its arrival time.
* open/high/low/close come from the tick prices of that minute in time order.
* Volume comes from the increase of the cumulative session volume (``Tick.day_volume``): the first value
  seen for a symbol only sets the baseline; a DECREASE means a new session started, the baseline resets to 0
  and the new value counts completely. Ticks without ``day_volume`` add no volume. Volume is never negative.
* A bar is emitted when a tick of a later minute arrives, or when ``flush(now_ts)`` finds its minute over.
  "Over" accounts for the feed delay: a symbol whose newest tick arrived ``lag`` seconds after its source
  time is flushed once ``now_ts - lag >= minute end + grace`` - a 15-minute delayed feed is not cut off
  after one minute.
* A tick with the same time as the symbol's newest tick is a duplicate (e.g. the canary symbol delivered on
  two stream connections) and ignored.
* A late tick (older than the newest tick of its symbol) updates the bar of its minute: high/low widen and,
  if it is the latest tick of that minute, it sets the close. If that bar was already emitted, the corrected
  bar is emitted again; a late tick for a minute without a bar creates one. Late ticks add no volume (the
  session-volume baseline only moves forward). Minutes older than ``retention_s`` before the symbol's
  newest minute are no longer corrected.
* Every emitted bar carries the ``source`` given to the constructor.
"""

import math
from dataclasses import dataclass

from trading_local.data.provider import Tick
from trading_local.db.repo_market import BAR_SECONDS
from trading_local.models import Bar, BarSource


@dataclass(slots=True)
class _Building:
    """Mutable state of one bar."""

    minute: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    last_tick_ts: float  # source time of the tick that set the close

    def add(self, ts: float, price: float, volume: float) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        if ts > self.last_tick_ts:
            self.close = price
            self.last_tick_ts = ts
        self.volume += volume


class MinuteAggregator:
    def __init__(self, source: BarSource, *, grace_s: float = 0.0, retention_s: int = 300) -> None:
        """``grace_s`` is extra waiting time after the delay-adjusted minute end (0: late ticks are corrected
        anyway, and the minute cycle runs only a few seconds after the boundary); ``retention_s`` how long
        emitted bars stay correctable."""
        self._source: BarSource = source
        self._grace_s = grace_s
        self._retention_s = retention_s
        self._open: dict[str, _Building] = {}  # bar of the newest minute per symbol, not yet emitted
        self._emitted: dict[str, dict[int, _Building]] = {}  # recently emitted bars (for corrections)
        self._last_ts: dict[str, float] = {}  # source time of the newest tick per symbol
        self._lag: dict[str, float] = {}  # arrival delay of the newest tick per symbol (seconds, >= 0)
        self._day_volume: dict[str, float] = {}  # newest cumulative session volume per symbol

    @property
    def source(self) -> BarSource:
        return self._source

    def add(self, tick: Tick) -> list[Bar]:
        """Process one tick; returns the bars it completed (an earlier minute) or corrected (late tick)."""
        if not (math.isfinite(tick.price) and tick.price > 0 and math.isfinite(tick.ts)):
            return []
        symbol = tick.symbol
        last = self._last_ts.get(symbol)
        if last is not None and tick.ts == last:
            return []
        minute = int(tick.ts // BAR_SECONDS) * BAR_SECONDS
        if last is not None and tick.ts < last:
            return self._late(symbol, minute, tick)
        self._last_ts[symbol] = tick.ts
        self._lag[symbol] = max(0.0, tick.received_ts - tick.ts)
        volume = self._volume_delta(symbol, tick.day_volume)
        bars: list[Bar] = []
        current = self._open.get(symbol)
        if current is not None and current.minute != minute:
            bars.append(self._emit(symbol, current))
            current = None
        if current is not None:
            current.add(tick.ts, tick.price, volume)
            return bars
        emitted = self._emitted.get(symbol, {}).get(minute)
        if emitted is not None:  # the minute was flushed early (the feed's delay grew): correct it
            emitted.add(tick.ts, tick.price, volume)
            bars.append(self._bar(symbol, emitted))
            return bars
        self._open[symbol] = _Building(minute, tick.price, tick.price, tick.price, tick.price, volume, tick.ts)
        return bars

    def flush(self, now_ts: float) -> list[Bar]:
        """Emit every open bar whose minute is over at ``now_ts`` (see the module rules), oldest first."""
        due = [
            symbol
            for symbol, building in self._open.items()
            if now_ts - self._lag.get(symbol, 0.0) >= building.minute + BAR_SECONDS + self._grace_s
        ]
        bars = [self._emit(symbol, self._open[symbol]) for symbol in due]
        bars.sort(key=lambda bar: (bar.ts, bar.symbol))
        return bars

    def forget(self, symbol: str) -> None:
        """Drop all state of ``symbol`` (an unfinished bar is discarded)."""
        for state in (self._open, self._emitted, self._last_ts, self._lag, self._day_volume):
            state.pop(symbol, None)

    # -- internals ------------------------------------------------------------------------------
    def _volume_delta(self, symbol: str, day_volume: float | None) -> float:
        if day_volume is None or not math.isfinite(day_volume) or day_volume < 0:
            return 0.0
        previous = self._day_volume.get(symbol)
        self._day_volume[symbol] = day_volume
        if previous is None:
            return 0.0
        if day_volume >= previous:
            return day_volume - previous
        return day_volume  # the counter went down: a new session started, the baseline is 0

    def _late(self, symbol: str, minute: int, tick: Tick) -> list[Bar]:
        newest_minute = int(self._last_ts[symbol] // BAR_SECONDS) * BAR_SECONDS
        if minute < newest_minute - self._retention_s:
            return []
        current = self._open.get(symbol)
        if current is not None and current.minute == minute:
            current.add(tick.ts, tick.price, 0.0)
            return []
        emitted = self._emitted.setdefault(symbol, {})
        building = emitted.get(minute)
        if building is None:
            building = _Building(minute, tick.price, tick.price, tick.price, tick.price, 0.0, tick.ts)
            emitted[minute] = building
        else:
            building.add(tick.ts, tick.price, 0.0)
        return [self._bar(symbol, building)]

    def _emit(self, symbol: str, building: _Building) -> Bar:
        del self._open[symbol]
        emitted = self._emitted.setdefault(symbol, {})
        emitted[building.minute] = building
        oldest_kept = building.minute - self._retention_s
        for minute in [minute for minute in emitted if minute < oldest_kept]:
            del emitted[minute]
        return self._bar(symbol, building)

    def _bar(self, symbol: str, building: _Building) -> Bar:
        return Bar(
            symbol=symbol,
            ts=building.minute,
            open=building.open,
            high=building.high,
            low=building.low,
            close=building.close,
            volume=building.volume,
            source=self._source,
        )


__all__ = ["MinuteAggregator"]
