"""Review tests for the as-of promise of ``trading_local.barstore``.

``test_barstore.py`` proves that data written for the future never changes a result. The tests here pin
the other half: every result is EXACTLY what a naive filter over the complete history yields (nothing
missing, nothing early), bars, buckets and daily rows appear at the very second they complete, and a
view only ever grows as time passes.
"""

import random
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_local.barstore import BarReader, resample
from trading_local.clock import from_ts, to_ts
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS, InstrumentBook
from trading_local.models import Bar
from trading_local.testing import (
    DEFAULT_TEST_WINDOWS,
    SAMPLE_INSTRUMENTS,
    FakeCalendar,
    make_bars,
    seed_bars,
    seed_instruments,
)

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
SYMBOLS = ("NVDA", "SAP.DE", "7203.T", "BTC-USD", "ORPHAN")  # ORPHAN has no instrument record
HISTORY_START = to_ts(datetime(2026, 11, 16, tzinfo=UTC))  # Monday 00:00 UTC
HISTORY_DAYS = 3  # Monday ... Wednesday
HISTORY_SECONDS = HISTORY_DAYS * 86_400
NOVEMBER = [f"2026-11-{day:02d}" for day in range(1, 31)]
BUCKET_SIZES = (5, 7, 60)
# Minutes of the UTC day at which a venue of DEFAULT_TEST_WINDOWS opens or closes (0 = UTC day change).
SESSION_EDGES = sorted({0, *(edge for windows in DEFAULT_TEST_WINDOWS.values() for pair in windows for edge in pair)})
FEED_DELAYS = sorted({0, *(instrument.delay_min * 60 for instrument in SAMPLE_INSTRUMENTS)})  # seconds


def gappy_history(number: int) -> pd.DataFrame:
    """Three days of minutes with random holes and one gap of 15 hours, different for every symbol."""
    frame = make_bars(HISTORY_START, HISTORY_DAYS * 1440, start_price=40.0 * (number + 1), seed=10 + number)
    keep = np.random.default_rng(number).random(len(frame)) < 0.6
    keep[1440 + 300 * number : 1440 + 300 * number + 900] = False
    return frame[keep]


def daily_frame(days: list[str], seed: int) -> pd.DataFrame:
    base = 100.0 + 10.0 * seed + np.arange(len(days), dtype=np.float64)
    return pd.DataFrame(
        {"open": base, "high": base + 2.0, "low": base - 2.0, "close": base + 1.0, "volume": 1000.0 + base},
        index=pd.Index(days, dtype="str", name="day"),
    )


@pytest.fixture
def histories(db: Database) -> dict[str, pd.DataFrame]:
    """The complete stored 1-minute history per symbol, kept in memory as the reference."""
    seed_instruments(db)
    frames: dict[str, pd.DataFrame] = {}
    for number, symbol in enumerate(SYMBOLS):
        frames[symbol] = gappy_history(number)
        assert seed_bars(db, symbol, frames[symbol]) == len(frames[symbol])
        repo_market.upsert_bars_1d(db, symbol, daily_frame(NOVEMBER, seed=number))
    return frames


# -- naive reference implementations ---------------------------------------------------------------
def naive_buckets(history: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """All buckets of a history, built bar by bar in plain Python."""
    step = minutes * 60
    buckets: dict[int, list[float]] = {}
    for ts, (open_, high, low, close, volume) in zip(history.index.tolist(), history.to_numpy().tolist()):
        bucket = buckets.setdefault(ts - ts % step, [open_, high, low, close, 0.0])
        bucket[1] = max(bucket[1], high)
        bucket[2] = min(bucket[2], low)
        bucket[3] = close
        bucket[4] += volume
    return pd.DataFrame(
        list(buckets.values()),
        columns=list(BAR_COLUMNS),
        dtype="float64",
        index=pd.Index(list(buckets), dtype="int64", name="ts"),
    )


def complete_at(buckets: pd.DataFrame, minutes: int, cutoff: int) -> pd.DataFrame:
    return buckets[[start + minutes * 60 <= cutoff for start in buckets.index]]


def naive_last_completed_day(calendar: FakeCalendar | None, exchange: str | None, cutoff: int) -> date:
    """Newest day that is over at the cutoff, derived from ``segments`` instead of ``last_completed_session``."""
    at = from_ts(cutoff)
    day = at.date()
    if calendar is None or exchange not in DEFAULT_TEST_WINDOWS:
        return day - timedelta(days=1)  # UTC-day rule; an always-open venue trades the whole UTC day
    while True:
        midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
        windows = calendar.segments(exchange, midnight, midnight + timedelta(days=1))
        if windows and windows[-1].end <= at:
            return day
        day -= timedelta(days=1)


def as_bar(symbol: str, frame: pd.DataFrame, position: int) -> Bar:
    row = frame.iloc[position]
    return Bar(symbol, int(frame.index[position]), row["open"], row["high"], row["low"], row["close"], row["volume"])


def assert_buckets_equal(actual: pd.DataFrame, expected: pd.DataFrame, what: str) -> None:
    """Prices must match exactly; a volume sum may differ in the last bit (pairwise versus running sum)."""
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, obj=what)
    pd.testing.assert_frame_equal(
        actual.drop(columns="volume"), expected.drop(columns="volume"), check_exact=True, obj=what
    )


# -- differential test -----------------------------------------------------------------------------
def test_reader_returns_exactly_what_a_naive_filter_over_the_history_yields(
    db: Database, histories: dict[str, pd.DataFrame]
) -> None:
    book = InstrumentBook(db)
    calendar = FakeCalendar()
    all_buckets = {
        (symbol, minutes): naive_buckets(history, minutes)
        for symbol, history in histories.items()
        for minutes in BUCKET_SIZES
    }
    rng = random.Random(4711)
    checked_bars = 0

    for case in range(30):
        symbol = SYMBOLS[case % len(SYMBOLS)]
        history = histories[symbol]
        instrument = INSTRUMENTS.get(symbol)
        if case % 3:
            asof = HISTORY_START + rng.randrange(-1800, HISTORY_SECONDS + 1800)
        else:  # around the instants at which a session, a bucket or a (delayed) bar completes
            edge = HISTORY_START + rng.randrange(1, HISTORY_DAYS + 1) * 86_400 + rng.choice(SESSION_EDGES) * 60
            asof = edge + rng.choice(FEED_DELAYS) + rng.choice([-1, 0, 1])
        apply_feed_delay = case % 2 == 0
        used_calendar = calendar if case % 4 else None
        cutoff = asof - (instrument.delay_min * 60 if instrument is not None and apply_feed_delay else 0)
        reader = BarReader(db, book, used_calendar, asof, apply_feed_delay=apply_feed_delay)
        visible = history[history.index + 60 <= cutoff]
        checked_bars += len(visible)
        what = f"case {case}: {symbol} at {asof}"

        assert reader.cutoff(symbol) == cutoff, what
        pd.testing.assert_frame_equal(reader.minutes(symbol), visible, check_exact=True, obj=what)

        lookback = rng.choice([0, 1, 7, 45, 600])
        if case % 2 and len(visible) > 30:  # probe exactly at the open time of stored bars ...
            start, position = (int(ts) for ts in rng.sample(visible.index[-30:].tolist(), 2))
        else:  # ... and at arbitrary seconds on both sides of the cutoff
            start, position = cutoff + rng.randrange(-7200, 600), cutoff + rng.randrange(-3600, 900)
        in_lookback = visible.index >= cutoff - lookback * 60
        from_start = visible.index >= start
        after = visible.index > position
        pd.testing.assert_frame_equal(
            reader.minutes(symbol, lookback_min=lookback), visible[in_lookback], check_exact=True, obj=what
        )
        pd.testing.assert_frame_equal(
            reader.minutes(symbol, start_ts=start, lookback_min=lookback),
            visible[in_lookback & from_start],
            check_exact=True,
            obj=what,
        )
        pd.testing.assert_frame_equal(
            reader.minutes(symbol, start_ts=start), visible[from_start], check_exact=True, obj=what
        )
        pd.testing.assert_frame_equal(reader.bars_after(symbol, position), visible[after], check_exact=True, obj=what)
        pd.testing.assert_frame_equal(
            reader.bars_after(symbol, position, start_ts=start), visible[after & from_start], check_exact=True, obj=what
        )
        pd.testing.assert_frame_equal(
            reader.bars_after(symbol, None, start_ts=start), visible[from_start], check_exact=True, obj=what
        )
        # the settled reads hand out a prefix of the same bars (all stored bars are REST bars here, so only
        # holes that are still young can cut them short)
        settled = reader.settled_bars_after(symbol, position, start_ts=start, settle_s=(0, 600, 3600)[case % 3])
        pd.testing.assert_frame_equal(
            settled, visible[after & from_start].iloc[: len(settled)], check_exact=True, obj=what
        )

        for minutes in BUCKET_SIZES:
            complete = complete_at(all_buckets[symbol, minutes], minutes, cutoff)
            lookback_bars = rng.choice([1, 2, 9, 300])
            assert_buckets_equal(
                reader.resampled(symbol, minutes, lookback_bars=lookback_bars),
                complete.tail(lookback_bars),
                f"{what}, {lookback_bars} buckets of {minutes} minutes",
            )

        newest = None if visible.empty else as_bar(symbol, visible, -1)
        assert reader.last_bar(symbol) == newest, what
        assert reader.last_price(symbol) == (None if newest is None else newest.close), what
        following = visible[visible.index >= position]
        first = None if following.empty else as_bar(symbol, following, 0)
        assert reader.first_bar_at_or_after(symbol, position) == first, what

        exchange = None if instrument is None else instrument.exchange
        last_day = naive_last_completed_day(used_calendar, exchange, cutoff).isoformat()
        assert reader.daily(symbol, limit=4).index.tolist() == [day for day in NOVEMBER if day <= last_day][-4:], what

    assert checked_bars > 20_000  # the cases were not trivially empty


def test_daily_bar_appears_at_the_exact_second_its_session_is_over(db: Database) -> None:
    """One second before, at and after every session edge, with and without calendar and feed delay."""
    for symbol in SYMBOLS:
        repo_market.upsert_bars_1d(db, symbol, daily_frame(NOVEMBER, seed=0))

    for symbol in SYMBOLS:
        instrument = INSTRUMENTS.get(symbol)
        exchange = None if instrument is None else instrument.exchange
        delay = 0 if instrument is None else instrument.delay_min * 60
        last_days: set[str] = set()
        for used_calendar in (FakeCalendar(), None):
            for apply_feed_delay in (False, True):
                for day in (4, 5, 7):  # Friday, Saturday and Monday
                    for edge in SESSION_EDGES:
                        for offset in sorted({-1, 0, 1, delay - 1, delay, delay + 1}):
                            asof = HISTORY_START + day * 86_400 + edge * 60 + offset
                            cutoff = asof - (delay if apply_feed_delay else 0)
                            expected = naive_last_completed_day(used_calendar, exchange, cutoff).isoformat()

                            reader = BarReader(db, INSTRUMENTS, used_calendar, asof, apply_feed_delay=apply_feed_delay)

                            assert reader.daily(symbol, limit=1).index.tolist() == [expected], (symbol, asof)
                            last_days.add(expected)
        assert len(last_days) >= 3, symbol  # the probes really crossed session ends


# -- every second around the boundaries ------------------------------------------------------------
def test_bars_and_buckets_appear_at_the_exact_second_they_complete(db: Database) -> None:
    start = to_ts(datetime(2026, 11, 20, 14, 30, tzinfo=UTC))
    seed_bars(db, "NVDA", make_bars(start, 30))  # 14:30 ... 14:59, more than the scanned window

    for second in range(-3, 6 * 60 + 3):
        asof = start + second
        reader = BarReader(db, INSTRUMENTS, None, asof)
        closed = [start + 60 * n for n in range(max(second // 60, 0))]
        closed_minutes = len(closed)
        closed_buckets = max(second // 300, 0)

        assert reader.minutes("NVDA").index.tolist() == closed, second
        assert reader.minutes("NVDA", lookback_min=2).index.tolist() == [ts for ts in closed if ts >= asof - 120], second
        assert reader.minutes("NVDA", start_ts=asof - 120).index.tolist() == [ts for ts in closed if ts >= asof - 120]
        assert reader.resampled("NVDA", 5).index.tolist() == [start + 300 * n for n in range(closed_buckets)], second
        newest = reader.last_bar("NVDA")
        newest_ts = start + 60 * (closed_minutes - 1) if closed_minutes else None
        assert (None if newest is None else newest.ts) == newest_ts, second
        # the bar that contains `asof` is stored but never visible; the one before it is closed
        current_minute = asof - asof % 60
        assert reader.first_bar_at_or_after("NVDA", current_minute) is None, second
        previous = reader.first_bar_at_or_after("NVDA", current_minute - 60)
        assert (None if previous is None else previous.ts) == (current_minute - 60 if second >= 60 else None), second
        new_bars = reader.bars_after("NVDA", current_minute - 61).index.tolist()
        assert new_bars == ([current_minute - 60] if second >= 60 else []), second


def test_delayed_feed_shifts_every_boundary_by_exactly_the_delay(db: Database) -> None:
    start = to_ts(datetime(2026, 11, 20, 14, 30, tzinfo=UTC))
    seed_bars(db, "SAP.DE", make_bars(start, 60))
    delay = INSTRUMENTS["SAP.DE"].delay_min * 60

    for second in (*range(295, 305), *range(0, 12 * 60, 37)):
        live = BarReader(db, INSTRUMENTS, None, start + second)
        replay = BarReader(db, INSTRUMENTS, None, start + second + delay, apply_feed_delay=True)

        pd.testing.assert_frame_equal(replay.minutes("SAP.DE"), live.minutes("SAP.DE"), check_exact=True)
        pd.testing.assert_frame_equal(replay.resampled("SAP.DE", 5), live.resampled("SAP.DE", 5), check_exact=True)
        pd.testing.assert_frame_equal(
            replay.minutes("SAP.DE", lookback_min=3), live.minutes("SAP.DE", lookback_min=3), check_exact=True
        )
        pd.testing.assert_frame_equal(
            replay.bars_after("SAP.DE", start + 120), live.bars_after("SAP.DE", start + 120), check_exact=True
        )
        assert replay.last_bar("SAP.DE") == live.last_bar("SAP.DE")
        assert replay.last_price("SAP.DE") == live.last_price("SAP.DE")
        assert replay.first_bar_at_or_after("SAP.DE", start + 300) == live.first_bar_at_or_after("SAP.DE", start + 300)


# -- a view only grows ---------------------------------------------------------------------------
def test_what_was_visible_once_stays_visible_and_unchanged(db: Database, histories: dict[str, pd.DataFrame]) -> None:
    """A decision taken at an earlier time can be reproduced at any later time: the earlier view is a
    prefix of every later view (bars, buckets, daily rows) and a fill bar never changes."""
    calendar = FakeCalendar()
    rng = random.Random(99)
    everything = 10**6  # more buckets than exist: no lookback cut

    for symbol in ("SAP.DE", "7203.T", "ORPHAN"):
        times = sorted(HISTORY_START + rng.randrange(0, HISTORY_SECONDS + 3600) for _ in range(8))
        order_ts = times[0] + 30
        earlier: dict[str, pd.DataFrame] = {}
        fill_bar: Bar | None = None
        for asof in times:
            reader = BarReader(db, INSTRUMENTS, calendar, asof, apply_feed_delay=True)
            current = {
                "minutes": reader.minutes(symbol),
                "5m": reader.resampled(symbol, 5, lookback_bars=everything),
                "1440m": reader.resampled(symbol, 1440, lookback_bars=everything),
                "daily": reader.daily(symbol),
            }
            for name, frame in earlier.items():
                assert len(current[name]) >= len(frame), (symbol, name, asof)
                pd.testing.assert_frame_equal(current[name].iloc[: len(frame)], frame, check_exact=True, obj=name)
            bar = reader.first_bar_at_or_after(symbol, order_ts)
            assert fill_bar is None or bar == fill_bar, (symbol, asof)
            earlier, fill_bar = current, bar
        assert fill_bar is not None and len(earlier["minutes"]) > 1000  # the views were not trivially empty


# -- resample ------------------------------------------------------------------------------------
@pytest.mark.parametrize("minutes", [1, 3, 5, 15, 60])
def test_resample_ignores_every_bar_that_was_not_closed_yet(minutes: int) -> None:
    """Resampling the full frame with ``complete_before_ts`` equals resampling only the bars that were
    closed by then: bars of a bucket that is still forming cannot influence the result."""
    frame = make_bars(HISTORY_START + 17 * 60, 400, seed=minutes)
    frame = frame[np.random.default_rng(minutes).random(len(frame)) < 0.7]
    reference = naive_buckets(frame, minutes)
    rng = random.Random(minutes)

    for _ in range(25):
        cut = HISTORY_START + rng.randrange(0, 440 * 60)
        closed = frame[frame.index + 60 <= cut]

        buckets = resample(frame, minutes, complete_before_ts=cut)

        pd.testing.assert_frame_equal(buckets, resample(closed, minutes, complete_before_ts=cut), check_exact=True)
        assert all(start + minutes * 60 <= cut for start in buckets.index)
        assert_buckets_equal(buckets, complete_at(reference, minutes, cut), f"cut at {cut}")


def test_resample_accepts_a_numpy_integer_as_completion_time() -> None:
    frame = make_bars(HISTORY_START, 20)

    buckets = resample(frame, 5, complete_before_ts=frame.index[12])  # numpy.int64 taken from a frame

    assert buckets.index.tolist() == [HISTORY_START, HISTORY_START + 300]
    assert buckets.index.dtype == np.int64 and buckets.index.name == "ts"
    assert all(dtype == np.float64 for dtype in buckets.dtypes)
