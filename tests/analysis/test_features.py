"""Feature builder: hand-checked values on small synthetic frames, session handling, lags, speed, news context.

The synthetic-market helpers at the top are shared with ``test_no_lookahead``.
"""

import math
import time
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_local.analysis import indicators as ind
from trading_local.analysis.features import (
    FEATURE_COLUMNS,
    build_features,
    feature_float,
    news_age_ref_ts,
    news_context,
)
from trading_local.analysis.setups import detect
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.barstore import resample
from trading_local.clock import from_ts, to_ts
from trading_local.db.repo_market import BAR_COLUMNS
from trading_local.db.repo_news import SymbolNews
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_bars

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
NVDA = INSTRUMENTS["NVDA"]
TOYOTA = INSTRUMENTS["7203.T"]
BTC = INSTRUMENTS["BTC-USD"]
CALENDAR = FakeCalendar()


# ------------------------------------------------------------------------------------------------
# Synthetic market helpers (also used by test_no_lookahead)
# ------------------------------------------------------------------------------------------------
def utc(day: int, hour: int, minute: int = 0) -> int:
    """Epoch seconds of a time in November 2026 (the 20th is a Friday)."""
    return to_ts(datetime(2026, 11, day, hour, minute, tzinfo=UTC))


def weekdays(first: date, last: date) -> list[date]:
    days = []
    current = first
    while current <= last:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def session_minutes(
    exchange: str,
    days: list[date],
    *,
    seed: int = 1,
    start_price: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.0012,
    base_volume: float = 1000.0,
    calendar: FakeCalendar = CALENDAR,
) -> pd.DataFrame:
    """A continuous random walk of 1-minute bars over the regular segments of ``days``."""
    frames = []
    price = start_price
    for k, day in enumerate(days):
        midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for j, segment in enumerate(calendar.segments(exchange, midnight, midnight + timedelta(days=1))):
            count = int((segment.end - segment.start).total_seconds() // 60)
            frame = make_bars(
                to_ts(segment.start),
                count,
                start_price=price,
                seed=seed * 10_000 + k * 10 + j,
                drift=drift,
                vol=vol,
                base_volume=base_volume,
            )
            price = float(frame["close"].iloc[-1])
            frames.append(frame)
    return pd.concat(frames)


def daily_from_minutes(minutes: pd.DataFrame, history_days: int = 0, seed: int = 99) -> pd.DataFrame:
    """Daily bars per UTC date (the fake calendar's trading date), optionally preceded by older history."""
    days = pd.Index(np.datetime_as_string(minutes.index.to_numpy().astype("datetime64[s]"), unit="D"))
    daily = minutes.groupby(days.to_numpy()).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    if history_days:
        first = date.fromisoformat(str(daily.index[0]))
        rng = np.random.default_rng(seed)
        close = float(daily["open"].iloc[0]) * np.exp(np.cumsum(0.01 * rng.standard_normal(history_days))[::-1] * -1)
        old_days = [(first - timedelta(days=history_days - i)).isoformat() for i in range(history_days)]
        old = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=old_days
        )
        daily = pd.concat([old, daily])
    daily.index = pd.Index(daily.index.astype(str), dtype="str", name="day")
    return daily.astype("float64")


def bars_frame(ts: list[int], open_: list[float], high: list[float], low: list[float], close: list[float],
               volume: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume or [0.0] * len(ts)},
        index=pd.Index(ts, dtype="int64", name="ts"),
    ).astype("float64")


NOVEMBER = weekdays(date(2026, 11, 2), date(2026, 11, 20))  # 15 trading days


# ------------------------------------------------------------------------------------------------
# Structure
# ------------------------------------------------------------------------------------------------
def test_columns_dtypes_and_index() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-3:])
    features = build_features(minutes, daily_from_minutes(minutes), instrument=NVDA, calendar=CALENDAR)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.index.name == "ts" and features.index.dtype == np.int64
    assert features.index.is_monotonic_increasing and features.index.is_unique
    assert (features["data_ts"].to_numpy() == features.index.to_numpy()).all()
    assert features["session"].dtype == "str"
    assert features["in_session"].dtype == bool
    floats = [c for c in FEATURE_COLUMNS if c not in ("session", "in_session", "data_ts")]
    assert (features[floats].dtypes == np.float64).all()
    assert len(features) == 3 * 78


def test_empty_input_gives_empty_frame_with_all_columns() -> None:
    empty = pd.DataFrame(columns=list(BAR_COLUMNS), dtype="float64", index=pd.Index([], dtype="int64", name="ts"))
    features = build_features(empty, empty, instrument=NVDA, calendar=CALENDAR)
    assert features.empty and tuple(features.columns) == FEATURE_COLUMNS


def test_non_canonical_frames_are_rejected() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-1:])
    with pytest.raises(ValueError):
        build_features(minutes.iloc[::-1], daily_from_minutes(minutes), instrument=NVDA, calendar=CALENDAR)
    daily = daily_from_minutes(session_minutes("XNAS", NOVEMBER[-3:]))
    with pytest.raises(ValueError):
        build_features(minutes, daily.iloc[::-1], instrument=NVDA, calendar=CALENDAR)
    duplicated = pd.concat([minutes.iloc[:10], minutes.iloc[9:20]])  # one open time twice
    with pytest.raises(ValueError):
        build_features(duplicated, daily, instrument=NVDA, calendar=CALENDAR)
    with pytest.raises(ValueError):  # a datetime index is not reinterpreted as epoch seconds
        build_features(minutes.set_axis(pd.to_datetime(minutes.index, unit="s")), daily, instrument=NVDA,
                       calendar=CALENDAR)


# ------------------------------------------------------------------------------------------------
# Hand-checked bucket values
# ------------------------------------------------------------------------------------------------
def ten_bars() -> pd.DataFrame:
    start = utc(20, 14, 30)
    ts = [start + 60 * i for i in range(10)]
    open_ = [10.0, 10.2, 10.1, 10.4, 10.3, 10.5, 10.6, 10.4, 10.8, 10.7]
    close = [10.2, 10.1, 10.4, 10.3, 10.5, 10.6, 10.4, 10.8, 10.7, 10.9]
    high = [max(o, c) + 0.05 for o, c in zip(open_, close)]
    low = [min(o, c) - 0.05 for o, c in zip(open_, close)]
    return bars_frame(ts, open_, high, low, close, [100.0] * 10)


def test_bucket_values_are_hand_checked() -> None:
    features = build_features(ten_bars(), ten_bars().iloc[:0], instrument=NVDA, calendar=CALENDAR)
    assert features.index.tolist() == [utc(20, 14, 35), utc(20, 14, 40)]
    first, second = features.iloc[0], features.iloc[1]
    assert (first["open_f"], first["high_f"], first["low_f"], first["price"]) == pytest.approx((10.0, 10.55, 9.95, 10.5))
    assert (second["open_f"], second["high_f"], second["low_f"], second["price"]) == pytest.approx(
        (10.5, 10.95, 10.35, 10.9)
    )
    assert (second["prev_high_f"], second["prev_low_f"], second["prev_price_f"]) == pytest.approx((10.55, 9.95, 10.5))
    assert math.isnan(first["prev_price_f"])
    assert (first["minutes_since_open"], first["minutes_to_close"]) == (5.0, 385.0)
    assert (second["minutes_since_open"], second["minutes_to_close"]) == (10.0, 380.0)
    assert first["session"] == "2026-11-20" and bool(first["in_session"])
    assert second["day_high"] == pytest.approx(10.95) and second["day_low"] == pytest.approx(9.95)
    assert first["day_high"] == pytest.approx(10.55)
    assert math.isnan(first["or_high"])  # the opening range is not complete before 15:00
    assert math.isnan(first["atr_slow"]) and math.isnan(first["rsi_fast"])  # warming up


def test_only_complete_buckets_produce_rows() -> None:
    bars = ten_bars()
    assert len(build_features(bars.iloc[:8], bars.iloc[:0], instrument=NVDA, calendar=CALENDAR)) == 1
    assert len(build_features(bars.iloc[:9], bars.iloc[:0], instrument=NVDA, calendar=CALENDAR)) == 1
    assert len(build_features(bars, bars.iloc[:0], instrument=NVDA, calendar=CALENDAR)) == 2
    with_gap = bars.drop(index=bars.index[7])  # a missing bar inside the last bucket does not matter
    assert len(build_features(with_gap, bars.iloc[:0], instrument=NVDA, calendar=CALENDAR)) == 2


# ------------------------------------------------------------------------------------------------
# Sessions and trading minutes
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("exchange", "instrument"), [("XNAS", NVDA), ("XTKS", TOYOTA), ("CRYPTO", BTC)])
def test_trading_minutes_agree_with_the_calendar(exchange: str, instrument) -> None:  # noqa: ANN001
    days = NOVEMBER[-3:]
    minutes = session_minutes(exchange, days)
    features = build_features(minutes, minutes.iloc[:0], instrument=instrument, calendar=CALENDAR)
    assert features["in_session"].all()
    for decision, row in features.iloc[::7].iterrows():
        at = from_ts(int(decision))
        session = CALENDAR.session_date(exchange, from_ts(int(decision) - 300)).isoformat()
        assert row["session"] == session
        if CALENDAR.is_open(exchange, at) and CALENDAR.session_date(exchange, at).isoformat() == session:
            assert row["minutes_since_open"] == CALENDAR.minutes_since_open(exchange, at)
            assert row["minutes_to_close"] == CALENDAR.minutes_to_close(exchange, at)
        else:  # decision on a segment end: the count continues
            before = from_ts(int(decision) - 60)
            assert row["minutes_since_open"] == CALENDAR.minutes_since_open(exchange, before) + 1
            assert row["minutes_to_close"] == CALENDAR.minutes_to_close(exchange, before) - 1


def test_lunch_break_venue_counts_trading_minutes_across_the_break() -> None:
    minutes = session_minutes("XTKS", [date(2026, 11, 20)])
    features = build_features(minutes, minutes.iloc[:0], instrument=TOYOTA, calendar=CALENDAR)
    assert features.loc[utc(20, 2, 30), "minutes_since_open"] == 150.0  # last bucket before the break
    assert features.loc[utc(20, 2, 30), "minutes_to_close"] == 180.0
    assert features.loc[utc(20, 3, 35), "minutes_since_open"] == 155.0  # first bucket after the break
    assert features.loc[utc(20, 6, 30), "minutes_to_close"] == 0.0
    assert not ((features.index > utc(20, 2, 30)) & (features.index <= utc(20, 3, 30))).any()  # no bars in the break


def test_24_7_instrument_uses_the_utc_day_as_session() -> None:
    minutes = session_minutes("CRYPTO", [date(2026, 11, 19), date(2026, 11, 20)])
    features = build_features(minutes, minutes.iloc[:0], instrument=BTC, calendar=CALENDAR)
    assert len(features) == 2 * 288 and features["in_session"].all()
    assert features.loc[utc(19, 23, 55), "session"] == "2026-11-19"
    assert features.loc[utc(20, 0, 0), "session"] == "2026-11-19"  # bucket 23:55-24:00 belongs to the 19th
    assert features.loc[utc(20, 0, 0), "minutes_to_close"] == 0.0
    assert features.loc[utc(20, 0, 5), "session"] == "2026-11-20"
    assert features.loc[utc(20, 0, 5), "minutes_since_open"] == 5.0
    assert features.loc[utc(20, 15, 0), "minutes_to_close"] == 540.0
    assert math.isnan(features.loc[utc(20, 0, 25), "or_high"])
    window = minutes[(minutes.index >= utc(20, 0)) & (minutes.index < utc(20, 0, 30))]
    assert features.loc[utc(20, 0, 30), "or_high"] == pytest.approx(window["high"].max())
    assert features.loc[utc(20, 0, 30), "or_low"] == pytest.approx(window["low"].min())


def test_bars_outside_regular_sessions() -> None:
    regular = session_minutes("XNAS", [date(2026, 11, 20)])
    pre = make_bars(utc(20, 13, 0), 30, start_price=99.0, seed=5)  # 13:00-13:30 UTC pre-market
    features = build_features(pd.concat([pre, regular]), regular.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    early = features.loc[utc(20, 13, 5)]
    assert not early["in_session"]
    assert math.isnan(early["minutes_since_open"]) and math.isnan(early["minutes_to_close"])
    assert early["session"] == "2026-11-20"
    assert math.isnan(early["day_high"]) and math.isnan(early["vwap"]) and math.isnan(early["gap_pct"])
    first_regular = features.loc[utc(20, 14, 35)]
    assert first_regular["in_session"] and first_regular["minutes_since_open"] == 5.0
    assert first_regular["day_high"] == pytest.approx(regular["high"].iloc[:5].max())  # pre-market not included


# ------------------------------------------------------------------------------------------------
# Daily features: one-session lag
# ------------------------------------------------------------------------------------------------
def daily_rows(values: dict[str, float]) -> pd.DataFrame:
    days = list(values)
    close = np.asarray(list(values.values()), dtype=np.float64)
    return pd.DataFrame(
        {"open": close - 1.0, "high": close + 2.0, "low": close - 2.0, "close": close, "volume": 1e6},
        index=pd.Index(days, dtype="str", name="day"),
    )


def test_daily_features_use_only_earlier_sessions() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 19), date(2026, 11, 20)])
    daily = daily_rows({"2026-11-17": 90.0, "2026-11-18": 100.0, "2026-11-19": 110.0, "2026-11-20": 500.0})
    features = build_features(minutes, daily, instrument=NVDA, calendar=CALENDAR)
    thursday = features[features["session"] == "2026-11-19"]
    friday = features[features["session"] == "2026-11-20"]
    assert (thursday["prev_day_close"] == 100.0).all() and (thursday["prev_day_high"] == 102.0).all()
    assert (friday["prev_day_close"] == 110.0).all() and (friday["prev_day_low"] == 108.0).all()
    changed = daily.copy()
    changed.loc["2026-11-20", ["open", "high", "low", "close"]] = [1.0, 2.0, 0.5, 1.5]
    again = build_features(minutes, changed, instrument=NVDA, calendar=CALENDAR)
    pd.testing.assert_frame_equal(features, again)


def test_daily_indicators_are_lagged_one_session() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 20)])
    daily = daily_from_minutes(session_minutes("XNAS", NOVEMBER, seed=3), history_days=260)
    features = build_features(minutes, daily, instrument=NVDA, calendar=CALENDAR)
    row = features.iloc[-1]
    before = daily.index.get_loc("2026-11-19")
    assert row["ema_daily_fast"] == pytest.approx(ind.ema(daily["close"], 50).iloc[before], rel=1e-12)
    assert row["ema_daily_slow"] == pytest.approx(ind.ema(daily["close"], 200).iloc[before], rel=1e-12)
    assert row["ema_daily_pullback"] == pytest.approx(ind.ema(daily["close"], 20).iloc[before], rel=1e-12)
    assert row["atr_daily"] == pytest.approx(ind.atr(daily, 14).iloc[before], rel=1e-12)
    assert row["rsi_daily"] == pytest.approx(ind.rsi(daily["close"], 14).iloc[before], rel=1e-12)


def test_gap_pct_compares_the_session_open_with_the_previous_close() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 20)], start_price=105.0)
    daily = daily_rows({"2026-11-19": 100.0})
    features = build_features(minutes, daily, instrument=NVDA, calendar=CALENDAR)
    assert features["gap_pct"].to_numpy() == pytest.approx(np.full(len(features), 5.0))


DAILY_COLUMNS = [
    "prev_day_high", "prev_day_low", "prev_day_close", "gap_pct", "ema_daily_fast", "ema_daily_slow",
    "ema_daily_pullback", "atr_daily", "rsi_daily",
]  # fmt: skip


def long_daily(last_day: str, days: int = 300) -> pd.DataFrame:
    """``days`` consecutive calendar days of daily bars ending with ``last_day`` (enough for the EMA200)."""
    index = pd.date_range(end=last_day, periods=days, freq="D").strftime("%Y-%m-%d")
    return daily_rows(dict(zip(index, 100.0 + 0.1 * np.arange(days), strict=True)))


def test_daily_features_are_nan_while_the_previous_session_bar_is_missing() -> None:
    """INT-6: shortly after UTC midnight the crypto daily bar of the previous UTC day is not stored yet. The
    rows must not describe the day before yesterday as 'yesterday' - every daily column is NaN instead."""
    minutes = make_bars(utc(19, 0), 2 * 1440, start_price=100.0, seed=3)  # BTC: Thu 19th and Fri 20th (UTC)
    complete = long_daily("2026-11-19")
    late = complete.iloc[:-1]  # the bar of the 19th has not arrived yet
    with_bar = build_features(minutes, complete, instrument=BTC, calendar=CALENDAR)
    without = build_features(minutes, late, instrument=BTC, calendar=CALENDAR)
    friday = (without["session"] == "2026-11-20").to_numpy()
    assert friday.any() and (~friday).any()
    assert without.loc[friday, DAILY_COLUMNS].isna().all().all()
    assert np.isfinite(with_bar.loc[friday, DAILY_COLUMNS].to_numpy(dtype=np.float64)).all()
    assert (with_bar.loc[friday, "prev_day_close"] == complete["close"].iloc[-1]).all()
    # Thursday's previous session (the 18th) is stored in both frames: nothing changes there
    pd.testing.assert_frame_equal(with_bar[~friday], without[~friday])


def test_daily_trend_cannot_fire_on_a_stale_daily_bar() -> None:
    minutes = make_bars(utc(19, 0), 2 * 1440, start_price=100.0, seed=3)
    rows = build_features(minutes, long_daily("2026-11-18"), instrument=BTC, calendar=CALENDAR)
    last = rows.iloc[-1].to_dict()
    assert math.isnan(last["prev_day_high"]) and math.isnan(last["atr_daily"])
    # a bucket far above any daily high after a close below it: with the stale bar of the 18th as
    # 'yesterday' (rising daily trend, pullback to the EMA20) this would be a daily_trend LONG
    breaking = {**last, "price": 1e6, "prev_price_f": 0.0}
    assert [hit for hit in detect(breaking, instrument=BTC) if hit.setup == "daily_trend"] == []
    complete = build_features(minutes, long_daily("2026-11-19"), instrument=BTC, calendar=CALENDAR)
    with_yesterday = {**breaking, **complete.iloc[-1][DAILY_COLUMNS].to_dict()}
    assert [hit.setup for hit in detect(with_yesterday, instrument=BTC)].count("daily_trend") == 1  # rule works


def test_previous_session_check_skips_weekends_and_holidays_and_looks_before_the_first_bar() -> None:
    monday = session_minutes("XNAS", [date(2026, 11, 23)])  # the frame starts on Monday
    daily = daily_rows({"2026-11-19": 100.0, "2026-11-20": 110.0})
    features = build_features(monday, daily, instrument=NVDA, calendar=CALENDAR)
    assert (features["prev_day_close"] == 110.0).all()  # Friday is Monday's previous session
    stale = build_features(monday, daily.iloc[:1], instrument=NVDA, calendar=CALENDAR)
    assert stale["prev_day_close"].isna().all() and stale["gap_pct"].isna().all()

    holiday = FakeCalendar(holidays={"XNAS": [date(2026, 11, 19)]})  # no session on Thursday
    friday = session_minutes("XNAS", [date(2026, 11, 20)], calendar=holiday)
    wednesday_close = daily_rows({"2026-11-17": 90.0, "2026-11-18": 100.0})
    after_holiday = build_features(friday, wednesday_close, instrument=NVDA, calendar=holiday)
    assert (after_holiday["prev_day_close"] == 100.0).all()  # Wednesday is Friday's previous session


# ------------------------------------------------------------------------------------------------
# Intraday session features
# ------------------------------------------------------------------------------------------------
def test_opening_range_day_extremes_and_vwap() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 20)], seed=4)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    before = features[features.index < utc(20, 15, 0)]
    after = features[features.index >= utc(20, 15, 0)]
    assert before["or_high"].isna().all() and before["or_low"].isna().all()
    window = minutes[minutes.index < utc(20, 15, 0)]
    assert (after["or_high"] == window["high"].max()).all() and (after["or_low"] == window["low"].min()).all()
    vwap = ind.session_vwap(minutes, pd.Series(0, index=minutes.index))
    for decision in (utc(20, 14, 35), utc(20, 16, 10), utc(20, 21, 0)):
        known = minutes[minutes.index + 60 <= decision]
        assert features.loc[decision, "day_high"] == known["high"].max()
        assert features.loc[decision, "day_low"] == known["low"].min()
        assert features.loc[decision, "vwap"] == pytest.approx(vwap.loc[known.index[-1]], rel=1e-12)


def test_pullback_extremes_span_four_buckets() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 20)], seed=6)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    decision = utc(20, 16, 0)
    window = minutes[(minutes.index >= decision - 20 * 60) & (minutes.index < decision)]
    assert features.loc[decision, "pullback_low"] == window["low"].min()
    assert features.loc[decision, "pullback_high"] == window["high"].max()
    assert features["pullback_low"].iloc[:3].isna().all()


def test_ret_60() -> None:
    minutes = session_minutes("XNAS", [date(2026, 11, 20)], seed=8)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    decision = utc(20, 17, 0)
    expected = (minutes.loc[decision - 60, "close"] / minutes.loc[decision - 3660, "close"] - 1.0) * 100.0
    assert features.loc[decision, "ret_60"] == pytest.approx(expected, rel=1e-12)
    assert math.isnan(features.loc[utc(20, 15, 0), "ret_60"])  # no bar an hour earlier


# ------------------------------------------------------------------------------------------------
# Relative volume
# ------------------------------------------------------------------------------------------------
def profiled_volume(days: list[date], first_bucket_volume: dict[date, float]) -> pd.DataFrame:
    """Sessions whose first 5-minute bucket trades 2,000 per bar (10,000 per bucket), all others 200 per bar."""
    minutes = session_minutes("XNAS", days, seed=2)
    volume = np.full(len(minutes), 200.0)
    for day in days:
        open_ts = to_ts(datetime(day.year, day.month, day.day, 14, 30, tzinfo=UTC))
        first = (minutes.index >= open_ts) & (minutes.index < open_ts + 300)
        volume[first] = first_bucket_volume.get(day, 10_000.0) / 5.0
    return minutes.assign(volume=volume)


def test_rel_volume_compares_with_the_same_time_of_day() -> None:
    days = weekdays(date(2026, 11, 11), date(2026, 11, 18))  # six sessions
    minutes = profiled_volume(days, {days[-1]: 20_000.0})
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    last_open = utc(18, 14, 35)
    assert features.loc[last_open, "rel_volume"] == pytest.approx(2.0)
    assert features.loc[last_open + 300, "rel_volume"] == pytest.approx(1.0)
    assert features.loc[utc(16, 14, 35), "rel_volume"] == pytest.approx(1.0)  # 4th session: 3 prior sessions


def test_rel_volume_falls_back_to_recent_buckets_with_little_history() -> None:
    days = weekdays(date(2026, 11, 11), date(2026, 11, 18))
    features = build_features(profiled_volume(days, {}), profiled_volume(days, {}).iloc[:0], instrument=NVDA,
                              calendar=CALENDAR)
    # 3rd session (13th): two prior sessions only -> median of the previous 20 buckets (1,000 each)
    assert features.loc[utc(13, 14, 35), "rel_volume"] == pytest.approx(10.0)
    assert features["rel_volume"].iloc[:20].isna().all()  # first session: fewer than 20 earlier buckets


def test_rel_volume_is_nan_without_volume_data() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-6:]).assign(volume=0.0)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    assert features["rel_volume"].isna().all()


# ------------------------------------------------------------------------------------------------
# Slow timeframe
# ------------------------------------------------------------------------------------------------
def test_slow_features_come_from_the_last_complete_slow_bucket() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-4:], seed=11)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    for decision in (utc(20, 16, 5), utc(20, 16, 15), utc(19, 20, 50), utc(20, 14, 35)):
        known = minutes[minutes.index + 60 <= decision]
        slow = resample(known, 15, complete_before_ts=decision)
        row = features.loc[decision]
        assert row["atr_slow"] == pytest.approx(ind.atr(slow, 14).iloc[-1], rel=1e-12)
        assert row["ema_fast_s"] == pytest.approx(ind.ema(slow["close"], 20).iloc[-1], rel=1e-12)
        assert row["ema_slow_s"] == pytest.approx(ind.ema(slow["close"], 50).iloc[-1], rel=1e-12)
        assert row["adx_slow"] == pytest.approx(ind.adx(slow, 14).iloc[-1], rel=1e-12)
        assert row["bb_lower_s"] == pytest.approx(ind.bollinger(slow["close"]).iloc[-1]["lower"], rel=1e-12)
        ema20 = ind.ema(slow["close"], 20)
        assert row["ema_fast_s_slope"] == pytest.approx(
            (ema20.iloc[-1] - ema20.iloc[-4]) / ind.atr(slow, 14).iloc[-1], rel=1e-9
        )
        hist = ind.macd(slow["close"])["hist"].iloc[-1]
        assert row["macd_hist_s"] == pytest.approx(hist / ind.atr(slow, 14).iloc[-1], rel=1e-9)


def test_donchian_spans_the_slow_buckets_before_the_containing_one() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-3:], seed=12)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    slow = resample(minutes, 15)
    for decision in (utc(20, 16, 5), utc(20, 16, 10), utc(20, 16, 15)):
        containing = (decision - 300) // 900 * 900
        previous = slow[slow.index < containing].tail(20)
        assert features.loc[decision, "donchian_upper_s"] == previous["high"].max()
        assert features.loc[decision, "donchian_lower_s"] == previous["low"].min()
        assert features.loc[decision, "price"] <= features.loc[decision, "high_f"]


def test_atr_pct_rank_is_a_trailing_percentile() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-6:], seed=13)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    ranks = features["atr_pct_rank"].dropna()
    assert not ranks.empty and ranks.between(0.0, 1.0).all()
    decision = utc(20, 18, 0)
    slow = resample(minutes[minutes.index + 60 <= decision], 15, complete_before_ts=decision)
    atr = ind.atr(slow, 14).dropna().to_numpy()  # six sessions < 20: the whole history is the window
    current = atr[-1]
    expected = ((atr < current).sum() + 0.5 * (atr == current).sum()) / atr.size
    assert features.loc[decision, "atr_pct_rank"] == pytest.approx(expected)


def test_atr_pct_rank_window_is_cut_at_atr_rank_sessions() -> None:
    days = weekdays(date(2026, 10, 19), date(2026, 11, 20))  # 25 sessions: longer than the rank window
    minutes = session_minutes("XNAS", days, seed=14)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    for decision in (utc(20, 18, 0), utc(17, 14, 45), utc(13, 21, 0)):
        slow = resample(minutes[minutes.index + 60 <= decision], 15, complete_before_ts=decision)
        atr = ind.atr(slow, 14)
        dates = np.asarray(np.datetime_as_string(slow.index.to_numpy().astype("datetime64[s]"), unit="D"))
        window_dates = sorted(set(dates.tolist()))[-ENGINE_PARAMS.atr_rank_sessions :]
        values = atr[np.isin(dates, window_dates)].dropna().to_numpy()
        current = atr.iloc[-1]
        expected = ((values < current).sum() + 0.5 * (values == current).sum()) / values.size
        assert features.loc[decision, "atr_pct_rank"] == pytest.approx(expected, rel=1e-12), from_ts(decision)


def test_rel_volume_matches_a_brute_force_time_of_day_median() -> None:
    days = weekdays(date(2026, 11, 2), date(2026, 11, 18))  # 13 sessions: longer than the 10-session window
    minutes = session_minutes("XNAS", days, seed=15)
    rng = np.random.default_rng(15)
    minutes = minutes.assign(volume=minutes["volume"].to_numpy() * rng.lognormal(0.0, 1.0, len(minutes)))
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    for day, hour, minute in ((18, 14, 35), (18, 17, 40), (17, 21, 0), (13, 15, 5)):
        decision = utc(day, hour, minute)
        offset = (decision - 300) % 86_400  # time of day of the bucket start (all sessions open at 14:30 UTC)
        previous = [d for d in days if d < date(2026, 11, day)][-10:]
        baseline = np.median(
            [
                minutes.loc[(minutes.index >= start) & (minutes.index < start + 300), "volume"].sum()
                for start in (to_ts(datetime(d.year, d.month, d.day, tzinfo=UTC)) + offset for d in previous)
            ]
        )
        own = minutes.loc[(minutes.index >= decision - 300) & (minutes.index < decision), "volume"].sum()
        assert features.loc[decision, "rel_volume"] == pytest.approx(own / baseline, rel=1e-12), from_ts(decision)


# ------------------------------------------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------------------------------------------
def test_benchmark_alignment() -> None:
    days = NOVEMBER[-4:]
    minutes = session_minutes("XNAS", days, seed=21)
    bench = session_minutes("XNAS", days, seed=22, start_price=6000.0)
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR, benchmark_minutes=bench)
    for decision in (utc(20, 17, 5), utc(20, 18, 15)):
        expected_ret = (bench.loc[decision - 60, "close"] / bench.loc[decision - 3660, "close"] - 1.0) * 100.0
        assert features.loc[decision, "bench_ret_60"] == pytest.approx(expected_ret, rel=1e-12)
        slow = resample(bench[bench.index + 60 <= decision], 15, complete_before_ts=decision)
        trend = (ind.ema(slow["close"], 20) - ind.ema(slow["close"], 50)) / ind.atr(slow, 14)
        assert features.loc[decision, "bench_trend"] == pytest.approx(trend.iloc[-1], rel=1e-12)
        row = features.loc[decision]
        assert row["rel_strength_60"] == pytest.approx(row["ret_60"] - row["bench_ret_60"])
    # future benchmark bars are invisible; a benchmark that stops early is used as of its last bar
    cut = utc(20, 17, 0)
    shorter = build_features(
        minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR, benchmark_minutes=bench[bench.index < cut]
    )
    early = features.index <= cut
    pd.testing.assert_frame_equal(shorter[early], features[early])
    late = shorter.loc[utc(20, 19, 0)]
    expected = (bench.loc[cut - 60, "close"] / bench.loc[cut - 60, "close"] - 1.0) * 100.0
    assert late["bench_ret_60"] == pytest.approx(expected)


def test_without_benchmark_the_market_columns_are_nan() -> None:
    minutes = session_minutes("XNAS", NOVEMBER[-2:])
    features = build_features(minutes, minutes.iloc[:0], instrument=NVDA, calendar=CALENDAR)
    assert features[["bench_trend", "bench_ret_60", "rel_strength_60"]].isna().all().all()


# ------------------------------------------------------------------------------------------------
# Speed
# ------------------------------------------------------------------------------------------------
def best_of(runs: int, call) -> float:  # noqa: ANN001
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        timings.append(time.perf_counter() - start)
    return min(timings)


def test_ten_day_frames_are_fast() -> None:
    days = NOVEMBER[-10:]
    minutes = session_minutes("XNAS", days, seed=31)  # 3,900 bars
    bench = session_minutes("XNAS", days, seed=32)
    daily = daily_from_minutes(minutes, history_days=260)
    crypto = session_minutes("CRYPTO", days)  # 14,400 bars
    equity_s = best_of(3, lambda: build_features(minutes, daily, instrument=NVDA, calendar=CALENDAR,
                                                 benchmark_minutes=bench))
    crypto_s = best_of(3, lambda: build_features(crypto, daily, instrument=BTC, calendar=CALENDAR))
    assert equity_s < 0.1, equity_s
    assert crypto_s < 0.1, crypto_s


# ------------------------------------------------------------------------------------------------
# feature_float
# ------------------------------------------------------------------------------------------------
def test_feature_float_reads_numbers_only() -> None:
    row = {"a": 1, "b": np.float64(2.5), "c": "x", "d": None, "e": True, "f": np.int64(3)}
    assert feature_float(row, "a") == 1.0 and feature_float(row, "b") == 2.5 and feature_float(row, "f") == 3.0
    for key in ("c", "d", "e", "missing"):
        assert math.isnan(feature_float(row, key))
    assert feature_float(pd.Series({"x": 4.0}), "x") == 4.0


# ------------------------------------------------------------------------------------------------
# News context
# ------------------------------------------------------------------------------------------------
NOW = utc(20, 15, 0)


def news(news_id: int, age_min: float, relevance: int = 5, sentiment: float = 0.8) -> SymbolNews:
    return SymbolNews(news_id, int(NOW - age_min * 60), relevance, sentiment, f"t{news_id}", "", "intraday", "other")


def test_news_score_formula() -> None:
    rows = [news(1, 0, relevance=5, sentiment=0.8), news(2, 120, relevance=4, sentiment=-0.5)]
    context = news_context(rows, asof_ts=NOW)
    w1, w2 = 1.0, 0.8 * 0.5
    assert context.score == pytest.approx((w1 * 0.8 + w2 * -0.5) / (1.0 + w1 + w2))
    assert context.count == 2 and context.news_ids == (1, 2)
    assert context.fresh == rows[0]


def test_a_single_item_cannot_reach_full_sentiment() -> None:
    assert news_context([news(1, 0, sentiment=1.0)], asof_ts=NOW).score == pytest.approx(0.5)
    many = [news(i, 0, sentiment=1.0) for i in range(1, 200)]
    assert 0.99 < news_context(many, asof_ts=NOW).score <= 1.0


def test_news_filters_and_fresh_rules() -> None:
    future = news(9, -1)  # analysed after asof
    old = news(8, 361)  # outside the 360-minute window
    assert news_context([future, old], asof_ts=NOW) == news_context([], asof_ts=NOW)
    assert news_context([], asof_ts=NOW).score == 0.0 and news_context([], asof_ts=NOW).fresh is None
    assert news_context([news(1, 21)], asof_ts=NOW).fresh is None  # too old for the news setup
    assert news_context([news(1, 20)], asof_ts=NOW).fresh is not None
    assert news_context([news(1, 5, relevance=3)], asof_ts=NOW).fresh is None
    assert news_context([news(1, 5, sentiment=0.49)], asof_ts=NOW).fresh is None
    assert news_context([news(1, 5, sentiment=-0.5)], asof_ts=NOW).fresh is not None
    # only the NEWEST item can be fresh
    assert news_context([news(1, 2, relevance=2), news(2, 10)], asof_ts=NOW).fresh is None


def test_news_ids_newest_first_at_most_five() -> None:
    rows = [news(i, age_min=i) for i in range(1, 9)]
    context = news_context(list(reversed(rows)), asof_ts=NOW)
    assert context.news_ids == (1, 2, 3, 4, 5) and context.count == 8


def test_news_rows_without_a_finite_sentiment_are_ignored() -> None:
    broken = news(2, 1, sentiment=math.nan)
    good = news(1, 3)
    assert news_context([broken, good], asof_ts=NOW) == news_context([good], asof_ts=NOW)
    assert news_context([broken], asof_ts=NOW) == news_context([], asof_ts=NOW)


def test_news_window_boundary_is_exclusive_at_the_old_end() -> None:
    assert news_context([news(1, 360)], asof_ts=NOW).count == 0  # exactly 360 minutes old: outside
    assert news_context([news(1, 359.9)], asof_ts=NOW).count == 1
    assert news_context([news(1, 0)], asof_ts=NOW).count == 1  # analysed at the decision time: known


def test_news_params_are_respected() -> None:
    params = EngineParams(news_half_life_min=60)
    rows = [news(1, 60, relevance=5, sentiment=1.0)]
    assert news_context(rows, asof_ts=NOW, params=params).score == pytest.approx(0.5 / 1.5)
    assert news_context(rows, asof_ts=NOW, params=ENGINE_PARAMS).score == pytest.approx(0.5**0.5 / (1 + 0.5**0.5))


@dataclass(frozen=True, slots=True)
class _SeenNews:
    """``repo_news.SymbolNews`` with the time the item was first fetched (used while SymbolNews lacks it)."""

    news_id: int
    analyzed_ts: int
    relevance: int
    sentiment: float
    title: str
    summary_de: str
    horizon: str
    category: str
    fetched_ts: int


def seen_news(
    news_id: int, *, analyzed_ts: int, fetched_ts: int, relevance: int = 5, sentiment: float = 0.8
) -> SymbolNews:
    """A news row that was first seen at ``fetched_ts`` and whose analysis became known at ``analyzed_ts``."""
    values: dict[str, Any] = {
        "news_id": news_id, "analyzed_ts": analyzed_ts, "relevance": relevance, "sentiment": sentiment,
        "title": f"t{news_id}", "summary_de": "", "horizon": "intraday", "category": "other", "fetched_ts": fetched_ts,
    }  # fmt: skip
    if "fetched_ts" in {spec.name for spec in fields(SymbolNews)}:
        return SymbolNews(**values)
    return _SeenNews(**values)  # type: ignore[return-value]


def test_news_age_is_counted_from_the_first_sight_not_from_the_analysis() -> None:
    """CAUS-3: an item fetched two hours ago (recovered after a restart) and analysed a minute ago is two
    hours old: it is not fresh news and its weight has decayed accordingly."""
    recovered = seen_news(1, analyzed_ts=NOW - 60, fetched_ts=NOW - 7200)
    context = news_context([recovered], asof_ts=NOW)
    assert context.count == 1 and context.news_ids == (1,)  # known: analysed before the decision
    assert context.fresh is None
    weight = 1.0 * 0.5 ** (120 / ENGINE_PARAMS.news_half_life_min)
    assert context.score == pytest.approx(weight * 0.8 / (1.0 + weight))
    assert news_age_ref_ts(recovered) == NOW - 7200
    queued = seen_news(2, analyzed_ts=NOW - 60, fetched_ts=NOW - 25 * 60)  # waited 24 minutes in the LLM queue
    assert news_context([queued], asof_ts=NOW).fresh is None
    prompt = seen_news(3, analyzed_ts=NOW - 60, fetched_ts=NOW - 3 * 60)
    assert news_context([prompt], asof_ts=NOW).fresh == prompt


def test_news_age_reference_never_makes_an_item_younger() -> None:
    assert news_age_ref_ts(news(1, 5)) == NOW - 300  # without fetched_ts: the analysis time
    broken = seen_news(2, analyzed_ts=NOW - 300, fetched_ts=NOW + 600)  # fetched 'after' its analysis
    assert news_age_ref_ts(broken) == NOW - 300
    assert news_age_ref_ts(seen_news(3, analyzed_ts=NOW - 300, fetched_ts=0)) == NOW - 300
    # the as-of filter stays on analyzed_ts: an item fetched long ago but analysed after asof is unknown
    future = seen_news(4, analyzed_ts=NOW + 60, fetched_ts=NOW - 600)
    assert news_context([future], asof_ts=NOW) == news_context([], asof_ts=NOW)
