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
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_bars, seed_bars, seed_instruments

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}


def utc(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """Epoch seconds of a time in November 2026 (the 20th is a Friday)."""
    return to_ts(datetime(2026, 11, day, hour, minute, second, tzinfo=UTC))


FRIDAY_OPEN = utc(20, 14, 30)  # New York opens; aligned to 5, 15 and 30 minute buckets


def minute_frame(*rows: tuple[float, ...]) -> pd.DataFrame:
    """Canonical 1-minute frame from (ts, open, high, low, close, volume) rows."""
    return pd.DataFrame(
        [row[1:] for row in rows],
        columns=list(BAR_COLUMNS),
        dtype="float64",
        index=pd.Index([row[0] for row in rows], dtype="int64", name="ts"),
    )


def daily_frame(days: list[str], seed: int = 0) -> pd.DataFrame:
    """Canonical daily frame with distinct, consistent values per day."""
    base = 100.0 + 10.0 * seed + np.arange(len(days), dtype=np.float64)
    return pd.DataFrame(
        {"open": base, "high": base + 2.0, "low": base - 2.0, "close": base + 1.0, "volume": 1000.0 + base},
        index=pd.Index(days, dtype="str", name="day"),
    )


def with_gaps(frame: pd.DataFrame, keep_share: float, seed: int) -> pd.DataFrame:
    """Randomly drop rows so that buckets are only partly filled."""
    keep = np.random.default_rng(seed).random(len(frame)) < keep_share
    return frame[keep]


def reference_resample(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Independent implementation of the bucket rules with a pandas groupby."""
    step = minutes * 60
    buckets = frame.groupby(frame.index // step * step).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    buckets.index.name = "ts"
    return buckets


def assert_canonical_minutes(frame: pd.DataFrame) -> None:
    assert frame.index.name == "ts"
    assert frame.index.dtype == np.int64
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


# ------------------------------------------------------------------------------------------------
# resample
# ------------------------------------------------------------------------------------------------
def test_resample_takes_first_open_extreme_high_low_last_close_and_volume_sum() -> None:
    frame = minute_frame(
        (FRIDAY_OPEN, 100.0, 101.0, 99.5, 100.5, 10.0),
        (FRIDAY_OPEN + 60, 100.5, 103.0, 100.0, 102.0, 20.0),
        (FRIDAY_OPEN + 120, 102.0, 102.5, 98.0, 99.0, 30.0),
        (FRIDAY_OPEN + 180, 99.0, 100.0, 98.5, 99.5, 40.0),
        (FRIDAY_OPEN + 240, 99.5, 101.5, 99.0, 101.0, 50.0),
        (FRIDAY_OPEN + 300, 101.0, 104.0, 100.5, 103.5, 60.0),
        (FRIDAY_OPEN + 360, 103.5, 103.8, 97.0, 97.5, 70.0),
    )

    buckets = resample(frame, 5)

    expected = minute_frame(
        (FRIDAY_OPEN, 100.0, 103.0, 98.0, 101.0, 150.0),
        (FRIDAY_OPEN + 300, 101.0, 104.0, 97.0, 97.5, 130.0),
    )
    assert_canonical_minutes(buckets)
    pd.testing.assert_frame_equal(buckets, expected, check_exact=True)


def test_resample_anchors_buckets_at_the_epoch_not_at_the_first_bar() -> None:
    frame = make_bars(FRIDAY_OPEN + 120, 70)  # 14:32 ... 15:41

    quarter_hours = resample(frame, 15)
    hours = resample(frame, 60)

    assert quarter_hours.index.tolist() == [FRIDAY_OPEN + offset for offset in (0, 900, 1800, 2700, 3600)]
    assert hours.index.tolist() == [utc(20, 14), utc(20, 15)]
    assert (hours.index % 3600 == 0).all()
    first = frame.loc[: FRIDAY_OPEN + 899]  # 14:32 ... 14:44, a partial first bucket
    assert len(first) == 13
    expected = [
        first["open"].iloc[0],
        first["high"].max(),
        first["low"].min(),
        first["close"].iloc[-1],
        first["volume"].sum(),
    ]
    assert quarter_hours.iloc[0].tolist() == pytest.approx(expected, rel=1e-12)


def test_resample_has_no_rows_for_empty_buckets_and_keeps_partial_ones() -> None:
    frame = minute_frame(
        (FRIDAY_OPEN, 100.0, 101.0, 99.0, 100.5, 1.0),
        (FRIDAY_OPEN + 60, 100.5, 102.0, 100.0, 101.0, 2.0),
        (FRIDAY_OPEN + 17 * 60, 101.0, 101.5, 100.5, 101.2, 4.0),  # 14:47 after a gap
        (FRIDAY_OPEN + 3 * 86_400, 105.0, 106.0, 104.0, 105.5, 8.0),  # Monday 14:30 after the weekend
    )

    buckets = resample(frame, 5)

    expected = minute_frame(
        (FRIDAY_OPEN, 100.0, 102.0, 99.0, 101.0, 3.0),
        (FRIDAY_OPEN + 15 * 60, 101.0, 101.5, 100.5, 101.2, 4.0),
        (FRIDAY_OPEN + 3 * 86_400, 105.0, 106.0, 104.0, 105.5, 8.0),
    )
    pd.testing.assert_frame_equal(buckets, expected, check_exact=True)


def test_resample_drops_buckets_that_are_still_forming() -> None:
    frame = make_bars(FRIDAY_OPEN, 15)  # 14:30 ... 14:44: buckets 14:30, 14:35, 14:40
    complete = resample(frame, 5)
    assert complete.index.tolist() == [FRIDAY_OPEN, FRIDAY_OPEN + 300, FRIDAY_OPEN + 600]

    def starts(complete_before_ts: int) -> list[int]:
        return resample(frame, 5, complete_before_ts=complete_before_ts).index.tolist()

    assert starts(FRIDAY_OPEN + 900) == [FRIDAY_OPEN, FRIDAY_OPEN + 300, FRIDAY_OPEN + 600]  # ends exactly then
    assert starts(FRIDAY_OPEN + 899) == [FRIDAY_OPEN, FRIDAY_OPEN + 300]
    assert starts(FRIDAY_OPEN + 600) == [FRIDAY_OPEN, FRIDAY_OPEN + 300]
    assert starts(FRIDAY_OPEN + 599) == [FRIDAY_OPEN]
    assert starts(FRIDAY_OPEN + 86_400) == complete.index.tolist()
    pd.testing.assert_frame_equal(resample(frame, 5, complete_before_ts=FRIDAY_OPEN + 899), complete.iloc[:2])
    nothing = resample(frame, 5, complete_before_ts=FRIDAY_OPEN + 299)
    assert nothing.empty
    assert_canonical_minutes(nothing)


def test_resample_of_an_empty_frame_is_an_empty_canonical_frame() -> None:
    for minutes in (1, 5, 60):
        buckets = resample(repo_market.empty_bars_1m(), minutes, complete_before_ts=FRIDAY_OPEN)
        assert buckets.empty
        assert_canonical_minutes(buckets)


def test_resample_to_one_minute_returns_the_same_bars() -> None:
    frame = with_gaps(make_bars(FRIDAY_OPEN, 50), 0.7, seed=3)

    pd.testing.assert_frame_equal(resample(frame, 1), frame, check_exact=True)


@pytest.mark.parametrize("minutes", [1, 2, 5, 7, 15, 60, 240, 1440])
def test_resample_matches_an_independent_groupby_implementation(minutes: int) -> None:
    frame = with_gaps(make_bars(utc(18, 23, 17), 2000, seed=minutes, vol=0.002), 0.6, seed=minutes)

    buckets = resample(frame, minutes)

    assert_canonical_minutes(buckets)
    pd.testing.assert_frame_equal(buckets, reference_resample(frame, minutes), check_exact=False, rtol=1e-12)
    assert (buckets["low"] <= buckets[["open", "close"]].min(axis=1)).all()
    assert (buckets["high"] >= buckets[["open", "close"]].max(axis=1)).all()


def test_resample_rejects_input_that_is_not_canonical() -> None:
    frame = make_bars(FRIDAY_OPEN, 10)

    with pytest.raises(ValueError, match="minutes"):
        resample(frame, 0)
    with pytest.raises(ValueError, match="minutes"):
        resample(frame, -5)
    with pytest.raises(ValueError, match="ascending"):
        resample(frame.iloc[::-1], 5)
    with pytest.raises(ValueError, match="integer epoch-second index"):
        resample(frame.set_axis(pd.to_datetime(frame.index, unit="s", utc=True)), 5)
    with pytest.raises(ValueError, match="lacks the columns"):
        resample(frame.drop(columns="volume"), 5)


def test_resample_leaves_its_input_untouched() -> None:
    frame = make_bars(FRIDAY_OPEN, 30)
    before = frame.copy(deep=True)

    resample(frame, 5, complete_before_ts=FRIDAY_OPEN + 1200)

    pd.testing.assert_frame_equal(frame, before, check_exact=True)


# ------------------------------------------------------------------------------------------------
# BarReader: cutoff
# ------------------------------------------------------------------------------------------------
def test_cutoff_is_the_asof_time_in_live_operation(db: Database) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 15))

    assert reader.asof_ts == utc(20, 15)
    assert [reader.cutoff(symbol) for symbol in ("NVDA", "SAP.DE", "7203.T", "UNKNOWN")] == [utc(20, 15)] * 4


def test_cutoff_lies_the_feed_delay_earlier_when_the_delay_is_applied(db: Database) -> None:
    seed_instruments(db)
    asof = utc(20, 15)

    for instruments in (INSTRUMENTS, InstrumentBook(db)):
        reader = BarReader(db, instruments, None, asof, apply_feed_delay=True)
        assert reader.asof_ts == asof
        assert reader.cutoff("NVDA") == asof  # real-time feed
        assert reader.cutoff("SAP.DE") == asof - 15 * 60
        assert reader.cutoff("7203.T") == asof - 20 * 60
        assert reader.cutoff("UNKNOWN") == asof  # unknown symbols have no delay


# ------------------------------------------------------------------------------------------------
# BarReader: 1-minute bars
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def session(db: Database) -> pd.DataFrame:
    """NVDA bars 14:30 ... 15:29 on Friday; the same minutes for SAP.DE, whose feed is 15 minutes late."""
    seed_instruments(db)
    frame = make_bars(FRIDAY_OPEN, 60)
    seed_bars(db, "NVDA", frame)
    seed_bars(db, "SAP.DE", make_bars(FRIDAY_OPEN, 60, seed=2))
    return frame


@pytest.mark.parametrize(
    ("asof", "visible"),
    [
        (utc(20, 14, 30), 0),
        (utc(20, 14, 30, 59), 0),  # the first bar is still open
        (utc(20, 14, 31), 1),  # it closes exactly now
        (utc(20, 14, 44, 59), 14),
        (utc(20, 14, 45), 15),
        (utc(20, 14, 45, 1), 15),
        (utc(20, 14, 46), 16),
        (utc(20, 15, 29, 59), 59),
        (utc(20, 15, 30), 60),
        (utc(21, 9), 60),
    ],
)
def test_minutes_returns_exactly_the_bars_closed_at_the_cutoff(
    db: Database, session: pd.DataFrame, asof: int, visible: int
) -> None:
    frame = BarReader(db, INSTRUMENTS, FakeCalendar(), asof).minutes("NVDA")

    assert_canonical_minutes(frame)
    pd.testing.assert_frame_equal(frame, session.iloc[:visible], check_exact=True)
    assert all(ts + 60 <= asof for ts in frame.index)


def test_minutes_lookback_counts_back_from_the_cutoff(db: Database, session: pd.DataFrame) -> None:
    def minutes(asof: int, lookback_min: int) -> list[int]:
        frame = BarReader(db, INSTRUMENTS, None, asof).minutes("NVDA", lookback_min=lookback_min)
        return [(ts - FRIDAY_OPEN) // 60 for ts in frame.index]

    assert minutes(utc(20, 14, 45), 5) == [10, 11, 12, 13, 14]
    assert minutes(utc(20, 14, 45, 30), 5) == [11, 12, 13, 14]  # bars that opened at or after 14:40:30
    assert minutes(utc(20, 14, 45), 1) == [14]
    assert minutes(utc(20, 14, 45), 0) == []
    assert minutes(utc(20, 14, 45), 600) == list(range(15))


def test_minutes_start_is_inclusive_and_combines_with_the_lookback(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, None, utc(20, 14, 45))

    def minutes(**bounds: int) -> list[int]:
        return [(ts - FRIDAY_OPEN) // 60 for ts in reader.minutes("NVDA", **bounds).index]

    assert minutes(start_ts=FRIDAY_OPEN + 12 * 60) == [12, 13, 14]
    assert minutes(start_ts=FRIDAY_OPEN + 12 * 60 + 1) == [13, 14]
    assert minutes(start_ts=FRIDAY_OPEN + 12 * 60, lookback_min=5) == [12, 13, 14]  # start is stricter
    assert minutes(start_ts=FRIDAY_OPEN + 2 * 60, lookback_min=5) == [10, 11, 12, 13, 14]  # lookback is stricter
    assert minutes(start_ts=utc(20, 15)) == []  # a start in the future never reveals later bars
    assert reader.minutes("UNKNOWN").empty


def test_feed_delay_hides_the_bars_that_had_not_arrived_yet(db: Database, session: pd.DataFrame) -> None:
    asof = utc(20, 15)
    live = BarReader(db, INSTRUMENTS, None, asof)
    replay = BarReader(db, INSTRUMENTS, None, asof, apply_feed_delay=True)

    assert live.minutes("SAP.DE").index[-1] == utc(20, 14, 59)
    assert replay.minutes("SAP.DE").index[-1] == utc(20, 14, 44)  # 15 minutes behind
    assert replay.minutes("NVDA").index[-1] == utc(20, 14, 59)  # real-time feed
    assert replay.minutes("SAP.DE", lookback_min=5).index.tolist() == [utc(20, 14, 40 + n) for n in range(5)]
    last = replay.last_bar("SAP.DE")
    assert last is not None and last.ts == utc(20, 14, 44)
    assert replay.resampled("SAP.DE", 15).index.tolist() == [FRIDAY_OPEN]
    assert live.resampled("SAP.DE", 15).index.tolist() == [FRIDAY_OPEN, FRIDAY_OPEN + 900]
    assert replay.first_bar_at_or_after("SAP.DE", utc(20, 14, 45)) is None
    assert replay.bars_after("SAP.DE", utc(20, 14, 42)).index.tolist() == [utc(20, 14, 43), utc(20, 14, 44)]


def test_reader_accepts_a_numpy_integer_as_asof_time(db: Database, session: pd.DataFrame) -> None:
    """sqlite3 would bind a numpy integer as BLOB: every bar would pass a ``ts < ?`` bound."""
    asof = session.index[15]  # 14:45 as numpy.int64, e.g. taken from a frame by a replay loop
    assert isinstance(asof, np.integer)

    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), asof)

    assert isinstance(reader.asof_ts, int)
    pd.testing.assert_frame_equal(reader.minutes("NVDA"), session.iloc[:15], check_exact=True)
    assert reader.resampled("NVDA", 5).index.tolist() == [FRIDAY_OPEN, FRIDAY_OPEN + 300, FRIDAY_OPEN + 600]
    assert reader.bars_after("NVDA", session.index[12]).index.tolist() == [utc(20, 14, 43), utc(20, 14, 44)]
    assert reader.first_bar_at_or_after("NVDA", asof) is None
    last = reader.last_bar("NVDA")
    assert last is not None and last.ts == utc(20, 14, 44)


# ------------------------------------------------------------------------------------------------
# BarReader: resampled bars
# ------------------------------------------------------------------------------------------------
def test_resampled_leaves_out_the_bucket_that_is_still_forming(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 15, 7))

    five = reader.resampled("NVDA", 5)

    assert_canonical_minutes(five)
    assert five.index.tolist() == [FRIDAY_OPEN + 300 * n for n in range(7)]  # 14:30 ... 15:00
    assert reader.minutes("NVDA").index[-1] == utc(20, 15, 6)  # closed bars of the forming 15:05 bucket exist
    pd.testing.assert_frame_equal(five, reference_resample(session.iloc[:35], 5), check_exact=False, rtol=1e-12)
    assert reader.resampled("NVDA", 15).index.tolist() == [FRIDAY_OPEN, FRIDAY_OPEN + 900]
    assert reader.resampled("NVDA", 60).index.tolist() == [utc(20, 14)]  # 14:00 bucket, filled from 14:30
    assert reader.resampled("NVDA", 240).empty  # the 12:00-16:00 bucket has not ended


@pytest.mark.parametrize(
    ("asof", "buckets"),
    [
        (utc(20, 14, 44, 59), 0),
        (utc(20, 14, 45), 1),  # the 14:30 bucket ends exactly now
        (utc(20, 14, 59, 59), 1),
        (utc(20, 15, 0), 2),
        (utc(20, 15, 14, 59), 2),
        (utc(20, 15, 15), 3),
        (utc(20, 15, 30), 4),
    ],
)
def test_resampled_bucket_becomes_visible_when_it_ends(
    db: Database, session: pd.DataFrame, asof: int, buckets: int
) -> None:
    frame = BarReader(db, INSTRUMENTS, None, asof).resampled("NVDA", 15)

    assert frame.index.tolist() == [FRIDAY_OPEN + 900 * n for n in range(buckets)]
    assert all(start + 900 <= asof for start in frame.index)


def test_resampled_keeps_the_newest_lookback_bars(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, None, utc(20, 15, 30))
    everything = reader.resampled("NVDA", 5)
    assert len(everything) == 12

    pd.testing.assert_frame_equal(reader.resampled("NVDA", 5, lookback_bars=3), everything.tail(3), check_exact=True)
    pd.testing.assert_frame_equal(reader.resampled("NVDA", 5, lookback_bars=1), everything.tail(1), check_exact=True)
    assert reader.resampled("NVDA", 5, lookback_bars=0).empty
    with pytest.raises(ValueError, match="lookback_bars"):
        reader.resampled("NVDA", 5, lookback_bars=-1)
    with pytest.raises(ValueError, match="minutes"):
        reader.resampled("NVDA", 0)


def test_resampled_delivers_the_requested_buckets_across_night_and_weekend(db: Database) -> None:
    thursday = make_bars(utc(19, 14, 30), 390, seed=1)
    friday = make_bars(utc(20, 14, 30), 390, seed=2)
    monday = make_bars(utc(23, 14, 30), 20, seed=3)
    for frame in (thursday, friday, monday):
        seed_bars(db, "NVDA", frame)
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(23, 14, 50))

    buckets = reader.resampled("NVDA", 15, lookback_bars=30)

    # One complete bucket on Monday, the 26 buckets of Friday and the last three of Thursday.
    assert len(buckets) == 30
    assert buckets.index[-1] == utc(23, 14, 30)
    assert buckets.index[0] == utc(19, 20, 15)
    days = [from_ts(start).day for start in buckets.index]
    assert (days.count(19), days.count(20), days.count(23)) == (3, 26, 1)
    reference = reference_resample(pd.concat([thursday, friday, monday.iloc[:15]]), 15).tail(30)
    pd.testing.assert_frame_equal(buckets, reference, check_exact=False, rtol=1e-12)


@pytest.mark.parametrize("minutes", [2, 5, 15, 60])
@pytest.mark.parametrize("lookback_bars", [1, 2, 7, 50])
def test_resampled_never_returns_a_bucket_cut_off_by_the_row_limit(
    db: Database, minutes: int, lookback_bars: int
) -> None:
    history = with_gaps(make_bars(utc(19, 22, 3), 1500, seed=minutes), 0.5, seed=lookback_bars)
    seed_bars(db, "BTC-USD", history)
    asof = utc(20, 21, 11, 40)
    reader = BarReader(db, INSTRUMENTS, None, asof)

    buckets = reader.resampled("BTC-USD", minutes, lookback_bars=lookback_bars)

    complete = reference_resample(history[history.index + 60 <= asof], minutes)
    complete = complete[complete.index + minutes * 60 <= asof]
    assert len(buckets) == min(lookback_bars, len(complete))
    pd.testing.assert_frame_equal(buckets, complete.tail(lookback_bars), check_exact=False, rtol=1e-12)


# ------------------------------------------------------------------------------------------------
# BarReader: daily bars
# ------------------------------------------------------------------------------------------------
DAYS = ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23", "2026-11-24"]


@pytest.fixture
def daily(db: Database) -> pd.DataFrame:
    seed_instruments(db)
    frame = daily_frame(DAYS)
    for symbol in ("NVDA", "SAP.DE", "7203.T", "BTC-USD", "ORPHAN"):
        repo_market.upsert_bars_1d(db, symbol, frame)
    return frame


@pytest.mark.parametrize(
    ("asof", "last_day"),
    [
        (utc(20, 14, 0), "2026-11-19"),  # Friday before the open
        (utc(20, 15, 0), "2026-11-19"),  # session running
        (utc(20, 20, 59, 59), "2026-11-19"),
        (utc(20, 21, 0), "2026-11-20"),  # the session ends exactly now
        (utc(21, 12, 0), "2026-11-20"),  # Saturday
        (utc(23, 14, 29, 59), "2026-11-20"),  # Monday before the open
        (utc(23, 20, 0), "2026-11-20"),
        (utc(23, 21, 0), "2026-11-23"),
        (utc(27, 0, 0), "2026-11-24"),  # later days have no bar
    ],
)
def test_daily_ends_with_the_last_completed_session_of_the_calendar(
    db: Database, daily: pd.DataFrame, asof: int, last_day: str
) -> None:
    frame = BarReader(db, INSTRUMENTS, FakeCalendar(), asof).daily("NVDA")

    assert frame.index.name == "day"
    assert frame.index.tolist() == DAYS[: DAYS.index(last_day) + 1]
    pd.testing.assert_frame_equal(frame, daily.loc[:last_day], check_exact=True)


def test_daily_respects_lunch_breaks_and_holidays(db: Database, daily: pd.DataFrame) -> None:
    calendar = FakeCalendar(holidays={"XNAS": {date(2026, 11, 20)}})

    def last_day(symbol: str, asof: int) -> str:
        return BarReader(db, INSTRUMENTS, calendar, asof).daily(symbol).index[-1]

    assert last_day("7203.T", utc(20, 3, 0)) == "2026-11-19"  # Tokyo lunch break: the session is not over
    assert last_day("7203.T", utc(20, 6, 30)) == "2026-11-20"
    assert last_day("NVDA", utc(20, 22, 0)) == "2026-11-19"  # holiday: no session on the 20th
    assert last_day("NVDA", utc(23, 21, 0)) == "2026-11-23"


def test_daily_of_an_always_open_venue_ends_yesterday(db: Database, daily: pd.DataFrame) -> None:
    def last_day(asof: int) -> str:
        return BarReader(db, INSTRUMENTS, FakeCalendar(), asof).daily("BTC-USD").index[-1]

    assert last_day(utc(20, 23, 59, 59)) == "2026-11-19"
    assert last_day(utc(21, 0, 0)) == "2026-11-20"
    assert last_day(utc(23, 12, 0)) == "2026-11-20"  # no bars for the weekend in the fixture
    assert last_day(utc(24, 0, 0)) == "2026-11-23"


def test_daily_judges_completeness_at_the_delayed_cutoff(db: Database, daily: pd.DataFrame) -> None:
    asof = utc(20, 16, 40)  # Xetra closed at 16:30, but SAP.DE data is 15 minutes late
    live = BarReader(db, INSTRUMENTS, FakeCalendar(), asof)
    replay = BarReader(db, INSTRUMENTS, FakeCalendar(), asof, apply_feed_delay=True)

    assert live.daily("SAP.DE").index[-1] == "2026-11-20"
    assert replay.daily("SAP.DE").index[-1] == "2026-11-19"
    later = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 16, 45), apply_feed_delay=True)
    assert later.daily("SAP.DE").index[-1] == "2026-11-20"


def test_daily_asks_the_calendar_for_the_venue_at_the_cutoff(db: Database, daily: pd.DataFrame) -> None:
    calls: list[tuple[str, datetime]] = []

    class RecordingCalendar(FakeCalendar):
        def last_completed_session(self, exchange: str, at: datetime) -> date | None:
            calls.append((exchange, at))
            return super().last_completed_session(exchange, at)

    reader = BarReader(db, INSTRUMENTS, RecordingCalendar(), utc(20, 16, 40), apply_feed_delay=True)
    reader.daily("SAP.DE")
    reader.daily("7203.T")

    assert calls == [
        ("XETR", datetime(2026, 11, 20, 16, 25, tzinfo=UTC)),
        ("XTKS", datetime(2026, 11, 20, 16, 20, tzinfo=UTC)),
    ]
    assert all(at.tzinfo is UTC for _, at in calls)


@pytest.mark.parametrize(
    ("asof", "last_day"),
    [
        (utc(20, 0, 0), "2026-11-19"),
        (utc(20, 21, 30), "2026-11-19"),  # no calendar: the current UTC day never counts as completed
        (utc(20, 23, 59, 59), "2026-11-19"),
        (utc(21, 0, 0), "2026-11-20"),
        (utc(23, 23, 0), "2026-11-20"),
        (utc(24, 0, 0), "2026-11-23"),
    ],
)
def test_daily_without_calendar_ends_before_the_utc_date_of_the_cutoff(
    db: Database, daily: pd.DataFrame, asof: int, last_day: str
) -> None:
    frame = BarReader(db, INSTRUMENTS, None, asof).daily("NVDA")

    assert frame.index.tolist() == DAYS[: DAYS.index(last_day) + 1]


def test_daily_without_calendar_applies_the_feed_delay(db: Database, daily: pd.DataFrame) -> None:
    asof = utc(21, 0, 10)  # ten minutes into Saturday (UTC)

    assert BarReader(db, INSTRUMENTS, None, asof).daily("SAP.DE").index[-1] == "2026-11-20"
    assert BarReader(db, INSTRUMENTS, None, asof, apply_feed_delay=True).daily("SAP.DE").index[-1] == "2026-11-19"


def test_daily_of_a_symbol_without_instrument_uses_the_utc_date_rule(db: Database, daily: pd.DataFrame) -> None:
    calendar = FakeCalendar()

    assert BarReader(db, INSTRUMENTS, calendar, utc(20, 23, 0)).daily("ORPHAN").index[-1] == "2026-11-19"
    assert BarReader(db, INSTRUMENTS, calendar, utc(21, 0, 0)).daily("ORPHAN").index[-1] == "2026-11-20"


def test_daily_is_empty_while_the_calendar_knows_no_completed_session(db: Database, daily: pd.DataFrame) -> None:
    class NoHistoryCalendar(FakeCalendar):
        def last_completed_session(self, exchange: str, at: datetime) -> date | None:
            return None

    frame = BarReader(db, INSTRUMENTS, NoHistoryCalendar(), utc(27, 0)).daily("NVDA")

    assert frame.empty
    pd.testing.assert_frame_equal(frame, repo_market.empty_bars_1d())


def test_daily_limit_keeps_the_newest_completed_days(db: Database, daily: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(21, 12))

    pd.testing.assert_frame_equal(reader.daily("NVDA", limit=2), daily.loc["2026-11-19":"2026-11-20"], check_exact=True)
    assert reader.daily("NVDA", limit=1).index.tolist() == ["2026-11-20"]
    assert reader.daily("UNKNOWN").empty


# ------------------------------------------------------------------------------------------------
# BarReader: single bars and incremental reads
# ------------------------------------------------------------------------------------------------
def test_last_bar_and_last_price_are_the_newest_closed_bar(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 14, 45, 30))
    row = session.loc[utc(20, 14, 44)]

    assert reader.last_bar("NVDA") == Bar(
        "NVDA", utc(20, 14, 44), row["open"], row["high"], row["low"], row["close"], row["volume"], "rest"
    )
    assert reader.last_price("NVDA") == row["close"]
    assert isinstance(reader.last_price("NVDA"), float)

    early = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 14, 30, 59))
    assert early.last_bar("NVDA") is None and early.last_price("NVDA") is None
    assert reader.last_bar("UNKNOWN") is None and reader.last_price("UNKNOWN") is None


def test_bars_after_returns_the_closed_bars_that_follow(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 14, 45))

    def after(ts: int | None, **bounds: int) -> list[int]:
        frame = reader.bars_after("NVDA", ts, **bounds)
        assert_canonical_minutes(frame)
        return [(bar_ts - FRIDAY_OPEN) // 60 for bar_ts in frame.index]

    assert after(None) == list(range(15))  # nothing processed yet: all closed bars
    assert after(utc(20, 14, 41)) == [12, 13, 14]  # strictly after
    assert after(utc(20, 14, 41) - 1) == [11, 12, 13, 14]
    assert after(utc(20, 14, 44)) == []
    assert after(utc(20, 15, 10)) == []  # a position beyond the cutoff reveals nothing
    assert after(None, start_ts=utc(20, 14, 40)) == [10, 11, 12, 13, 14]
    assert after(utc(20, 14, 41), start_ts=utc(20, 14, 35)) == [12, 13, 14]
    assert after(utc(20, 14, 35), start_ts=utc(20, 14, 43)) == [13, 14]
    pd.testing.assert_frame_equal(reader.bars_after("NVDA", utc(20, 14, 41)), session.iloc[12:15], check_exact=True)


def test_bars_after_hands_every_bar_to_a_tracker_exactly_once(db: Database, session: pd.DataFrame) -> None:
    processed: list[int] = []
    checked_ts: int | None = None
    for asof in range(utc(20, 14, 29), utc(20, 15, 32), 45):  # a tracker that wakes every 45 seconds
        new = BarReader(db, INSTRUMENTS, None, asof).bars_after("NVDA", checked_ts, start_ts=FRIDAY_OPEN)
        assert all(ts + 60 <= asof for ts in new.index)
        if not new.empty:
            processed.extend(new.index.tolist())
            checked_ts = processed[-1]

    assert processed == session.index.tolist()


def test_first_bar_at_or_after_returns_closed_bars_only(db: Database, session: pd.DataFrame) -> None:
    reader = BarReader(db, INSTRUMENTS, FakeCalendar(), utc(20, 14, 45))

    def first(ts: int) -> int | None:
        bar = reader.first_bar_at_or_after("NVDA", ts)
        return None if bar is None else bar.ts

    assert first(utc(20, 14, 40)) == utc(20, 14, 40)
    assert first(utc(20, 14, 40) + 1) == utc(20, 14, 41)
    assert first(0) == FRIDAY_OPEN
    assert first(utc(20, 14, 44)) == utc(20, 14, 44)  # closes exactly at the cutoff
    assert first(utc(20, 14, 44) + 1) is None  # the 14:45 bar exists in the database but is still open
    assert first(utc(20, 14, 45)) is None
    assert first(utc(20, 16, 0)) is None
    assert reader.first_bar_at_or_after("UNKNOWN", 0) is None
    row = session.loc[utc(20, 14, 40)]
    assert reader.first_bar_at_or_after("NVDA", utc(20, 14, 40)) == Bar(
        "NVDA", utc(20, 14, 40), row["open"], row["high"], row["low"], row["close"], row["volume"], "rest"
    )


def test_market_order_fills_on_the_first_bar_once_that_bar_has_closed(db: Database, session: pd.DataFrame) -> None:
    placed = utc(20, 14, 40, 20)  # the fill bar is the one that opens at 14:41

    def fill_bar(asof: int) -> int | None:
        bar = BarReader(db, INSTRUMENTS, FakeCalendar(), asof).first_bar_at_or_after("NVDA", placed)
        return None if bar is None else bar.ts

    assert fill_bar(utc(20, 14, 41)) is None
    assert fill_bar(utc(20, 14, 41, 59)) is None
    assert fill_bar(utc(20, 14, 42)) == utc(20, 14, 41)
    assert fill_bar(utc(20, 15, 20)) == utc(20, 14, 41)  # later bars never change the fill bar


# ------------------------------------------------------------------------------------------------
# The as-of promise
# ------------------------------------------------------------------------------------------------
HISTORY_START = utc(16, 0)  # Monday 00:00 UTC
HISTORY_MINUTES = 4 * 1440
PROPERTY_SYMBOLS = ("NVDA", "SAP.DE", "7203.T", "BTC-USD", "ORPHAN")  # ORPHAN has no instrument record
NOVEMBER = [f"2026-11-{day:02d}" for day in range(1, 31)]


def reader_results(reader: BarReader, symbol: str, cutoff: int) -> dict[str, object]:
    """Everything a BarReader can return for one symbol, with probe times on both sides of the cutoff."""
    return {
        "cutoff": reader.cutoff(symbol),
        "minutes": reader.minutes(symbol),
        "minutes_lookback": reader.minutes(symbol, lookback_min=45),
        "minutes_start": reader.minutes(symbol, start_ts=cutoff - 7200),
        "minutes_future_start": reader.minutes(symbol, start_ts=cutoff + 120),
        "resampled_5": reader.resampled(symbol, 5),
        "resampled_7": reader.resampled(symbol, 7, lookback_bars=5),
        "resampled_15": reader.resampled(symbol, 15, lookback_bars=40),
        "resampled_60": reader.resampled(symbol, 60, lookback_bars=10),
        "resampled_1440": reader.resampled(symbol, 1440),
        "daily": reader.daily(symbol),
        "daily_limited": reader.daily(symbol, limit=3),
        "last_bar": reader.last_bar(symbol),
        "last_price": reader.last_price(symbol),
        "after_none": reader.bars_after(symbol, None),
        "after_none_bounded": reader.bars_after(symbol, None, start_ts=cutoff - 600),
        "after_recent": reader.bars_after(symbol, cutoff - 900),
        "after_future": reader.bars_after(symbol, cutoff + 600),
        "first_past": reader.first_bar_at_or_after(symbol, cutoff - 1800),
        "first_edge": reader.first_bar_at_or_after(symbol, cutoff - 60),
        "first_now": reader.first_bar_at_or_after(symbol, cutoff),
        "first_future": reader.first_bar_at_or_after(symbol, cutoff + 300),
        "settled_none_bounded": reader.settled_bars_after(symbol, None, start_ts=cutoff - 7200),
        "settled_recent": reader.settled_bars_after(symbol, cutoff - 3600),
        "settled_short": reader.settled_bars_after(symbol, cutoff - 3600, settle_s=0),
        "settled_future": reader.settled_bars_after(symbol, cutoff + 600),
        "settled_first_past": reader.first_settled_bar_at_or_after(symbol, cutoff - 1800),
        "settled_first_edge": reader.first_settled_bar_at_or_after(symbol, cutoff - 60, settle_s=0),
        "settled_first_now": reader.first_settled_bar_at_or_after(symbol, cutoff),
    }


def assert_same_results(before: dict[str, object], after: dict[str, object]) -> None:
    assert before.keys() == after.keys()
    for name, expected in before.items():
        if isinstance(expected, pd.DataFrame):
            pd.testing.assert_frame_equal(after[name], expected, check_exact=True, obj=name)
        else:
            assert after[name] == expected, name


def assert_complete_at(results: dict[str, object], cutoff: int, last_complete_day: date) -> None:
    """No returned bar, bucket or daily row may be newer than what was complete at the cutoff."""
    for name, value in results.items():
        if isinstance(value, Bar):
            assert value.ts + 60 <= cutoff, name
        elif isinstance(value, pd.DataFrame) and name.startswith("daily"):
            assert all(day <= last_complete_day.isoformat() for day in value.index), name
        elif isinstance(value, pd.DataFrame):
            # 'resampled_15' holds 15-minute buckets, every other frame 1-minute bars
            seconds = 60 * int(name.removeprefix("resampled_")) if name.startswith("resampled_") else 60
            assert all(start + seconds <= cutoff for start in value.index), name


def test_future_data_never_changes_any_reader_result(db: Database) -> None:
    """Property: for random as-of times, writing arbitrary bars that were not complete at the cutoff
    (new ones and overwritten ones, intraday and daily) leaves every BarReader result untouched."""
    seed_instruments(db)
    book = InstrumentBook(db)
    calendar = FakeCalendar()
    for number, symbol in enumerate(PROPERTY_SYMBOLS):
        history = make_bars(HISTORY_START, HISTORY_MINUTES, start_price=50.0 * (number + 1), seed=number)
        seed_bars(db, symbol, with_gaps(history, 0.7, seed=number))
        repo_market.upsert_bars_1d(db, symbol, daily_frame(NOVEMBER, seed=number))
    rng = random.Random(20261122)
    visible_bars = 0

    for case in range(40):
        symbol = rng.choice(PROPERTY_SYMBOLS)
        asof = HISTORY_START + rng.randrange(-3600, HISTORY_MINUTES * 60 + 3600)
        apply_feed_delay = rng.random() < 0.5
        used_calendar = calendar if rng.random() < 0.6 else None
        instrument = INSTRUMENTS.get(symbol)
        delay_s = instrument.delay_min * 60 if instrument is not None and apply_feed_delay else 0
        cutoff = asof - delay_s
        if used_calendar is None or instrument is None:
            last_complete_day = from_ts(cutoff).date() - timedelta(days=1)
        else:
            last_complete_day = calendar.last_completed_session(instrument.exchange, from_ts(cutoff))

        reader = BarReader(db, book, used_calendar, asof, apply_feed_delay=apply_feed_delay)
        before = reader_results(reader, symbol, cutoff)
        assert before["cutoff"] == cutoff
        assert_complete_at(before, cutoff, last_complete_day)
        visible_bars += len(before["minutes"])

        # Everything written here was not complete at the cutoff: the 1-minute bar that contains the
        # cutoff and later ones, daily rows after the last completed session.
        first_open_bar = cutoff // 60 * 60
        future = make_bars(
            first_open_bar, rng.randrange(1, 900), start_price=rng.uniform(1.0, 5000.0), seed=1000 + case, vol=0.01
        )
        assert seed_bars(db, symbol, future) == len(future)
        future_days = [(last_complete_day + timedelta(days=n)).isoformat() for n in range(1, 9)]
        assert repo_market.upsert_bars_1d(db, symbol, daily_frame(future_days, seed=1000 + case)) == len(future_days)
        newest = repo_market.last_bar(db, symbol)
        assert newest is not None and newest.ts >= first_open_bar  # the future really is in the database

        replayed = BarReader(db, book, used_calendar, asof, apply_feed_delay=apply_feed_delay)
        assert_same_results(before, reader_results(replayed, symbol, cutoff))

    assert visible_bars > 10_000  # the cases were not trivially empty
