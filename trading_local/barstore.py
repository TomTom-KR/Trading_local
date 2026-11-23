"""As-of bounded access to stored bars: the ONLY door through which decision code reads bars.

The signal engine, the trackers, the paper broker and the validation replay take a ``BarReader`` that
is created for one decision time (``asof_ts``). The reader returns nothing but data that was complete
at that time:

* 1-minute bars that are CLOSED at the cutoff: ``ts + 60 <= cutoff``
* resampled buckets whose END is ``<= cutoff``
* daily bars of sessions that had completely ended at the cutoff

Because of this the same decision code runs live and on history without look-ahead: data for times
after the cutoff cannot change any result of the reader, whether it is written later (live operation)
or already sits in the database (replay).

Settled bars (for consumers that act on every bar exactly once)
The paper broker (fills, stops, targets) and the signal tracker (entries, exits, MFE/MAE) process each
bar once and then move a high-water mark past it. Live, the stored bars are not final when they first
appear: stream/quote bars are provisional and are replaced by authoritative REST bars minutes later, and
minutes the stream missed (or the downtime after a restart) are inserted behind newer bars. Such
consumers read through ``settled_bars_after`` / ``first_settled_bar_at_or_after``, which only hand out a
bar once it can no longer change and no earlier traded minute can still arrive (see ``settle_horizon``).
"""

from collections.abc import Mapping
from datetime import date, timedelta
from typing import Final

import numpy as np
import pandas as pd

from trading_local.clock import from_ts
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS, InstrumentBook
from trading_local.models import Bar, Instrument

# Default settle time in seconds, counted from the moment a minute's data can first arrive: the REST
# reconciliation refetches every symbol of an open venue about every 5 minutes; the rest is margin.
SETTLE_S: Final = 600
SETTLED_SOURCE: Final = "rest"  # authoritative source: a bar stored with it never changes again
NOT_SETTLED_BY_AGE: Final = -(2**62)  # settle horizon while the live data service is warming up

# Start of the live data service in THIS process (epoch seconds), or None (tests, replay, tools). While
# less than a settle period has passed since then, age alone settles nothing (see settle_horizon).
_live_since_ts: int | None = None


def set_live_since(ts: int | None) -> None:
    """Record when the live data service started (called by the runtime; None switches the rule off)."""
    global _live_since_ts
    _live_since_ts = None if ts is None else int(ts)


def resample(frame: pd.DataFrame, minutes: int, *, complete_before_ts: int | None = None) -> pd.DataFrame:
    """Aggregate canonical 1-minute bars into buckets of ``minutes`` minutes.

    Buckets are anchored at the epoch: a bar belongs to the bucket that starts at
    ``ts // (minutes * 60) * (minutes * 60)``. Per bucket: open = first, high = max, low = min,
    close = last, volume = sum. A bucket exists only if it contains at least one bar. With
    ``complete_before_ts`` every bucket that ends after that time (start + minutes * 60 >
    complete_before_ts) is dropped because it is still forming.

    Returns a canonical frame whose index ``ts`` is the bucket start. Raises ``ValueError`` if the
    frame is not canonical (integer epoch index in ascending order, all bar columns).
    """
    if minutes < 1:
        raise ValueError(f"minutes must be at least 1, got {minutes}")
    ts, values = repo_market.bar_arrays(frame)
    if ts.size == 0:
        return repo_market.empty_bars_1m()
    if np.any(ts[1:] < ts[:-1]):
        raise ValueError("bar frame must be in ascending time order")
    step = minutes * BAR_SECONDS
    starts = ts // step * step
    first = np.flatnonzero(np.r_[True, starts[1:] != starts[:-1]])  # row of the first bar of every bucket
    last = np.r_[first[1:], ts.size] - 1
    buckets = pd.DataFrame(
        {
            "open": values[first, 0],
            "high": np.maximum.reduceat(values[:, 1], first),
            "low": np.minimum.reduceat(values[:, 2], first),
            "close": values[last, 3],
            "volume": np.add.reduceat(values[:, 4], first),
        },
        index=pd.Index(starts[first], name="ts"),
    )
    if complete_before_ts is not None:
        buckets = buckets[buckets.index + step <= complete_before_ts]
    return buckets


class BarReader:
    """Read-only view of the stored bars as they were known at ``asof_ts``.

    ``cutoff(symbol)`` is the time up to which data of a symbol counts as known. It equals ``asof_ts``
    in live operation (``apply_feed_delay=False``): the database only contains what has physically
    arrived. The validation replay passes ``apply_feed_delay=True`` to reproduce the arrival delay of
    delayed feeds on stored history: the cutoff then lies ``Instrument.delay_min`` minutes earlier.
    Symbols that are missing in ``instruments`` have no delay.

    No method ever returns a bar, bucket or daily row that was not complete at the cutoff.
    """

    def __init__(
        self,
        db: Database,
        instruments: Mapping[str, Instrument] | InstrumentBook,
        calendar: MarketCalendar | None,
        asof_ts: int,
        *,
        apply_feed_delay: bool = False,
    ) -> None:
        self._db = db
        self._instruments = instruments
        self._calendar = calendar
        self._asof_ts = int(asof_ts)
        self._apply_feed_delay = apply_feed_delay

    @property
    def asof_ts(self) -> int:
        """The decision time this reader was created for."""
        return self._asof_ts

    def cutoff(self, symbol: str) -> int:
        """Time up to which data of ``symbol`` is visible (``asof_ts`` minus the feed delay, if applied)."""
        if not self._apply_feed_delay:
            return self._asof_ts
        instrument = self._instruments.get(symbol)
        delay_min = 0 if instrument is None else instrument.delay_min
        return self._asof_ts - delay_min * 60

    def minutes(self, symbol: str, *, lookback_min: int | None = None, start_ts: int | None = None) -> pd.DataFrame:
        """Closed 1-minute bars as canonical frame.

        ``lookback_min`` keeps the bars that opened within the last ``lookback_min`` minutes before the
        cutoff (``ts >= cutoff - lookback_min * 60``), ``start_ts`` those with ``ts >= start_ts``; with
        both, both bounds apply.
        """
        lower = start_ts
        if lookback_min is not None:
            lower = _later(lower, self.cutoff(symbol) - lookback_min * 60)
        return repo_market.read_bars_1m(self._db, symbol, start_ts=lower, end_ts=self._closed_before(symbol))

    def resampled(self, symbol: str, minutes: int, *, lookback_bars: int = 300) -> pd.DataFrame:
        """The newest ``lookback_bars`` COMPLETE buckets of ``minutes`` minutes (see ``resample``).

        The history is read by row count, not by time span, so the requested number of buckets is
        delivered across overnight, weekend and holiday gaps as long as the bars exist.
        """
        if minutes < 1:
            raise ValueError(f"minutes must be at least 1, got {minutes}")
        if lookback_bars < 0:
            raise ValueError(f"lookback_bars must not be negative, got {lookback_bars}")
        cutoff = self.cutoff(symbol)
        step = minutes * BAR_SECONDS
        # Every bar that opens before the last bucket boundary at or before the cutoff belongs to a
        # complete bucket. `lookback_bars` buckets hold at most `lookback_bars * minutes` bars, so that
        # many rows always cover the requested buckets. If the row limit cuts a bucket in two, it is the
        # oldest of MORE than `lookback_bars` buckets and tail() removes it.
        frame = repo_market.read_bars_1m(
            self._db, symbol, end_ts=cutoff // step * step, limit=lookback_bars * minutes
        )
        return resample(frame, minutes, complete_before_ts=cutoff).tail(lookback_bars)

    def daily(self, symbol: str, *, limit: int = 300) -> pd.DataFrame:
        """The newest ``limit`` daily bars of COMPLETED sessions as canonical daily frame.

        With a calendar and a known instrument these are the days up to
        ``calendar.last_completed_session(instrument.exchange, cutoff)``. Without a calendar, or for a
        symbol that is missing in ``instruments``, only days strictly before the UTC date of the cutoff
        count as completed.
        """
        last_day = self._last_completed_day(symbol)
        if last_day is None:
            return repo_market.empty_bars_1d()
        return repo_market.read_bars_1d(self._db, symbol, end_day=last_day.isoformat(), limit=limit)

    def last_bar(self, symbol: str) -> Bar | None:
        """The newest closed 1-minute bar."""
        return repo_market.last_bar(self._db, symbol, before_ts=self._closed_before(symbol))

    def last_price(self, symbol: str) -> float | None:
        """Close of the newest closed 1-minute bar."""
        bar = self.last_bar(symbol)
        return None if bar is None else bar.close

    def bars_after(self, symbol: str, ts: int | None, *, start_ts: int | None = None) -> pd.DataFrame:
        """Closed 1-minute bars with an open time strictly greater than ``ts`` as canonical frame.

        For consumers that process every new bar exactly once: pass the open time of the last bar that
        was processed. ``ts=None`` (nothing processed yet) returns all closed bars; ``start_ts`` is an
        additional inclusive lower bound that keeps this first read from fetching the whole history.
        """
        lower = start_ts if ts is None else _later(start_ts, int(ts) + 1)
        return repo_market.read_bars_1m(self._db, symbol, start_ts=lower, end_ts=self._closed_before(symbol))

    def first_bar_at_or_after(self, symbol: str, ts: int) -> Bar | None:
        """The oldest closed 1-minute bar with an open time ``>= ts``.

        This is the bar on which a market order placed at ``ts`` fills; ``None`` until it has closed.
        """
        bar = repo_market.first_bar_at_or_after(self._db, symbol, ts)
        # If the first candidate is still open, every later bar is as well.
        if bar is None or bar.ts >= self._closed_before(symbol):
            return None
        return bar

    # -- settled bars ---------------------------------------------------------------------------
    def settle_horizon(self, symbol: str, *, settle_s: int | None = None) -> int:
        """Instant up to which every minute of ``symbol`` counts as settled by age.

        A minute with open time ``ts`` is settled by age when ``ts + 60 <= settle_horizon(symbol)``,
        whether a bar is stored for it (with any source) or not. The horizon is counted from the time a
        minute's data can first ARRIVE, ``settle_s`` (default ``SETTLE_S`` = 600) seconds earlier::

            horizon = asof_ts - Instrument.delay_min * 60 - settle_s

        In live operation (``apply_feed_delay=False``, cutoff = ``asof_ts``) this is
        ``cutoff - delay_min * 60 - settle_s``: a 15-minute delayed venue delivers minute ``ts`` at
        ``ts + 60 + 15 min`` and that minute settles ``settle_s`` later. With ``apply_feed_delay=True`` the
        cutoff already lies the delay earlier, so the horizon is ``cutoff - settle_s`` (the delay is never
        subtracted twice; a replay at ``asof_ts`` judges exactly like live operation at ``asof_ts``).
        Symbols missing in ``instruments`` have no delay. Raises ``ValueError`` for a negative ``settle_s``.
        """
        settle = SETTLE_S if settle_s is None else int(settle_s)
        if settle < 0:
            raise ValueError(f"settle_s must not be negative, got {settle_s}")
        instrument = self._instruments.get(symbol)
        delay_s = 0 if instrument is None else instrument.delay_min * 60
        live_since = _live_since_ts
        if live_since is not None and self._asof_ts - settle < live_since:
            # The live data service has not been running for a whole settle period yet (fresh start, restart
            # after an outage or sleep): the REST reconciliation has had no chance to confirm or repair the
            # minutes stored before / missing during the downtime, so nothing is settled by age yet.
            return NOT_SETTLED_BY_AGE
        return self._asof_ts - delay_s - settle

    def settled_bars_after(
        self, symbol: str, ts: int | None, *, start_ts: int | None = None, settle_s: int | None = None
    ) -> pd.DataFrame:
        """The SETTLED prefix of ``bars_after(symbol, ts, start_ts=start_ts)`` as canonical frame.

        For consumers that act on every bar exactly once (paper broker, signal tracker): pass the open
        time of the last bar processed (``None`` before the first one, then ``start_ts`` bounds the read)
        and move the high-water mark to the last returned bar. Every bar is then handed out exactly once,
        with its final values, and no bar is ever handed out before an earlier traded minute that may
        still arrive. Fills and exits keep using the bars' own times and prices, so a late hand-out only
        delays the reaction, never changes it.

        The bars are walked in ascending order, starting after the reference minute (the minute before the
        first bar that can be returned: the minute of ``ts``, or the minute before ``start_ts`` rounded up
        to a full minute when ``ts`` is ``None``; without both bounds the first bar has no reference). The
        walk stops BEFORE the first bar that

        * is not settled: a bar is settled when its source is 'rest' (authoritative REST data, never
          replaced again) or when it is settled by age (``ts + 60 <= settle_horizon(symbol)``); a younger
          'stream' or 'quote' bar may still be corrected by the REST reconciliation, or
        * follows a HOLE that is still young: missing minutes between the reference (or the previous bar)
          and this bar during which the venue traded, at least one of which is not settled by age yet.
          REST may still insert those minutes. Traded means: inside a regular segment of
          ``calendar.segments`` (24/7 venues trade every minute); nights, weekends, holidays and lunch
          breaks are no holes. Without a calendar, for a symbol missing in ``instruments`` or a venue the
          calendar does not know, every missing minute counts as traded. A hole whose missing minutes are
          all settled by age is accepted: illiquid symbols really have minutes without trades.

        The result is always a prefix of ``bars_after`` with the same arguments, so it never contains a
        bar that was not closed at the cutoff. ``settle_s`` overrides ``SETTLE_S`` (see ``settle_horizon``).

        The exactly-once-with-final-values promise assumes that a minute's final data is stored within
        ``settle_s`` after the minute could first arrive (the REST reconciliation refetches every symbol
        of an open venue about every 5 minutes). Data that comes even later (a REST pause longer than the
        horizon) lies behind the consumer's high-water mark and is not handed out again.
        """
        lower = start_ts if ts is None else _later(start_ts, int(ts) + 1)
        frame, sources = repo_market.read_bars_1m_with_sources(
            self._db, symbol, start_ts=lower, end_ts=self._closed_before(symbol)
        )
        reference = None if lower is None else _ceil_minute(int(lower)) - BAR_SECONDS
        horizon = self.settle_horizon(symbol, settle_s=settle_s)
        return frame.iloc[: self._settled_prefix(symbol, frame.index.to_numpy(), sources, reference, horizon)]

    def first_settled_bar_at_or_after(self, symbol: str, ts: int, *, settle_s: int | None = None) -> Bar | None:
        """``first_bar_at_or_after(symbol, ts)`` once it is settled, else ``None``.

        The fill bar of a market order placed at ``ts``: the oldest closed bar with open time ``>= ts``,
        returned only if (rules of ``settled_bars_after``) it is settled - source 'rest' or settled by age -
        and no young hole precedes it: no traded minute from ``ts`` (rounded up to a full minute) up to
        the bar is missing while it is not settled by age yet, because REST may still insert that minute,
        which would then be the real fill bar. While ``None`` is returned the order simply stays pending.
        Under the settle assumption of ``settled_bars_after`` a returned bar never changes afterwards.
        """
        bar = self.first_bar_at_or_after(symbol, ts)
        if bar is None:
            return None
        horizon = self.settle_horizon(symbol, settle_s=settle_s)
        if bar.source != SETTLED_SOURCE and bar.ts + BAR_SECONDS > horizon:
            return None
        unsettled_from = max(_ceil_minute(int(ts)), _first_unsettled_minute(horizon))
        if unsettled_from < bar.ts and self._traded(symbol, unsettled_from, bar.ts):
            return None
        return bar

    def _settled_prefix(
        self, symbol: str, ts: np.ndarray, sources: np.ndarray, reference: int | None, horizon: int
    ) -> int:
        """Number of leading bars that may be handed out (see ``settled_bars_after``)."""
        count = len(ts)
        if count == 0:
            return 0
        settled = (sources == SETTLED_SOURCE) | (ts + BAR_SECONDS <= horizon)
        end = count if bool(settled.all()) else int(np.argmin(settled))
        previous = np.empty(count, dtype=np.int64)
        previous[0] = ts[0] - BAR_SECONDS if reference is None else reference
        previous[1:] = ts[:-1]
        # Per bar: the start of the part of the preceding hole that is not settled by age yet. Only the
        # newest bars (within the horizon) can have one, so the calendar is asked a few times at most.
        unsettled_from = np.maximum(previous + BAR_SECONDS, _first_unsettled_minute(horizon))
        for index in np.flatnonzero(unsettled_from[:end] < ts[:end]).tolist():
            if self._traded(symbol, int(unsettled_from[index]), int(ts[index])):
                return index
        return end

    def _traded(self, symbol: str, start_ts: int, end_ts: int) -> bool:
        """True if the venue of ``symbol`` traded during [start_ts, end_ts) (unknown: True)."""
        instrument = self._instruments.get(symbol)
        if self._calendar is None or instrument is None:
            return True
        try:
            return bool(self._calendar.segments(instrument.exchange, from_ts(start_ts), from_ts(end_ts)))
        except KeyError:
            return True

    def _closed_before(self, symbol: str) -> int:
        """Exclusive upper bound for the open time of closed bars.

        ``ts + 60 <= cutoff`` is the same as ``ts < cutoff - 59`` for integer timestamps.
        """
        return self.cutoff(symbol) - BAR_SECONDS + 1

    def _last_completed_day(self, symbol: str) -> date | None:
        at = from_ts(self.cutoff(symbol))
        instrument = self._instruments.get(symbol)
        if self._calendar is None or instrument is None:
            return at.date() - timedelta(days=1)
        return self._calendar.last_completed_session(instrument.exchange, at)


def _later(bound: int | None, other: int) -> int:
    """The stricter of an optional lower bound and a second lower bound."""
    return other if bound is None else max(int(bound), other)


def _ceil_minute(ts: int) -> int:
    """``ts`` rounded up to a full minute."""
    return -(-ts // BAR_SECONDS) * BAR_SECONDS


def _first_unsettled_minute(horizon: int) -> int:
    """Open time of the oldest minute that is not settled by age (``ts + 60 > horizon``)."""
    return _ceil_minute(horizon - BAR_SECONDS + 1)
