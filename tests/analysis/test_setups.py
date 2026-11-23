"""Setup detectors on constructed feature rows: each setup in each direction, every guard, NaN handling."""

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.analysis.test_features import seen_news
from trading_local.analysis import setups
from trading_local.analysis.features import NewsContext
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.analysis.setups import SETUPS, SetupHit, detect
from trading_local.db.repo_news import SymbolNews
from trading_local.testing import SAMPLE_INSTRUMENTS

NVDA = next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "NVDA")
NOW = 1_795_186_800  # 2026-11-20 15:00 UTC

# A row in session that triggers nothing: flat trend, neutral oscillators, all levels far away.
BASE: dict[str, Any] = {
    "price": 100.0, "data_ts": NOW, "session": "2026-11-20", "minutes_since_open": 60.0, "minutes_to_close": 300.0,
    "in_session": True, "atr_fast": 0.6, "atr_slow": 1.0, "atr_daily": 3.0,
    "ema_fast_f": 100.0, "ema_slow_f": 100.0, "ema_fast_s": 100.0, "ema_slow_s": 100.0, "ema_fast_s_slope": 0.0,
    "rsi_fast": 50.0, "rsi_slow": 50.0, "rsi_daily": 50.0, "macd_hist_s": 0.0, "adx_slow": 18.0,
    "bb_mid_s": 100.0, "bb_upper_s": 105.0, "bb_lower_s": 95.0, "vwap": 100.0,
    "prev_high_f": 100.2, "prev_low_f": 99.8, "prev_price_f": 100.0, "high_f": 100.2, "low_f": 99.8, "open_f": 100.0,
    "pullback_low": 99.5, "pullback_high": 100.5, "donchian_upper_s": 110.0, "donchian_lower_s": 90.0,
    "day_high": 101.0, "day_low": 99.0, "or_high": 101.0, "or_low": 99.0,
    "prev_day_high": 112.0, "prev_day_low": 88.0, "prev_day_close": 100.0, "gap_pct": 0.0,
    "ema_daily_fast": 100.0, "ema_daily_slow": 100.0, "ema_daily_pullback": 100.0,
    "rel_volume": 1.0, "atr_pct_rank": 0.5, "ret_60": 0.0, "bench_trend": 0.0, "bench_ret_60": 0.0,
    "rel_strength_60": 0.0,
}  # fmt: skip

ROWS: dict[tuple[str, str], dict[str, Any]] = {
    ("trend_pullback", "LONG"): {
        "ema_slow_s": 99.0, "ema_fast_s_slope": 0.2, "adx_slow": 25.0, "pullback_low": 100.1, "prev_high_f": 100.3,
        "price": 100.5, "ema_fast_f": 100.2, "open_f": 100.2, "high_f": 100.6, "low_f": 100.1,
    },
    ("trend_pullback", "SHORT"): {
        "ema_slow_s": 101.0, "ema_fast_s_slope": -0.2, "adx_slow": 25.0, "pullback_high": 99.9, "prev_low_f": 99.7,
        "price": 99.5, "ema_fast_f": 99.8, "open_f": 99.8, "high_f": 99.9, "low_f": 99.4,
    },
    ("breakout", "LONG"): {
        "donchian_upper_s": 100.2, "or_high": 100.1, "price": 100.5, "prev_price_f": 100.15, "rel_volume": 2.0,
        "ema_slow_s": 99.5, "open_f": 100.2, "high_f": 100.6, "low_f": 100.1,
    },
    ("breakout", "SHORT"): {
        "donchian_lower_s": 99.8, "or_low": 99.9, "price": 99.5, "prev_price_f": 99.85, "rel_volume": 2.0,
        "ema_slow_s": 100.5, "open_f": 99.8, "high_f": 99.9, "low_f": 99.4,
    },
    ("mean_reversion", "LONG"): {
        "adx_slow": 12.0, "rsi_fast": 20.0, "low_f": 94.5, "open_f": 95.0, "prev_price_f": 95.1, "price": 95.4,
        "high_f": 95.5,
    },
    ("mean_reversion", "SHORT"): {
        "adx_slow": 12.0, "rsi_fast": 80.0, "high_f": 105.5, "open_f": 105.0, "prev_price_f": 104.9, "price": 104.6,
        "low_f": 104.5,
    },
    ("news_momentum", "LONG"): {"price": 100.4, "open_f": 100.0, "prev_price_f": 100.0, "high_f": 100.5},
    ("news_momentum", "SHORT"): {"price": 99.6, "open_f": 100.0, "prev_price_f": 100.0, "low_f": 99.5},
    ("daily_trend", "LONG"): {
        "prev_day_close": 110.0, "ema_daily_fast": 105.0, "ema_daily_slow": 100.0, "prev_day_high": 112.0,
        "prev_day_low": 107.0, "price": 112.5, "prev_price_f": 111.8, "open_f": 111.8, "high_f": 112.6,
        "low_f": 111.7, "donchian_upper_s": 130.0, "or_high": 125.0,
    },
    ("daily_trend", "SHORT"): {
        "prev_day_close": 90.0, "ema_daily_fast": 95.0, "ema_daily_slow": 100.0, "prev_day_high": 93.0,
        "prev_day_low": 88.0, "price": 87.5, "prev_price_f": 88.2, "open_f": 88.2, "high_f": 88.3, "low_f": 87.4,
        "donchian_lower_s": 70.0, "or_low": 75.0,
    },
}  # fmt: skip
INTRADAY = ("trend_pullback", "breakout", "mean_reversion", "news_momentum")


def fresh_news(sentiment: float, age_min: float = 5.0, relevance: int = 5) -> NewsContext:
    item = SymbolNews(7, int(NOW - age_min * 60), relevance, sentiment, "Titel", "", "intraday", "earnings")
    return NewsContext(score=0.4 * np.sign(sentiment), count=1, news_ids=(7,), fresh=item)


def row_for(setup: str, direction: str, **overrides: Any) -> dict[str, Any]:
    return {**BASE, **ROWS[(setup, direction)], **overrides}


def news_for(setup: str, direction: str) -> NewsContext | None:
    if setup != "news_momentum":
        return None
    return fresh_news(0.8 if direction == "LONG" else -0.8)


def hits_of(row: dict[str, Any], setup: str, direction: str, **kwargs: Any) -> list[SetupHit]:
    kwargs.setdefault("news", news_for(setup, direction))
    return [hit for hit in detect(row, instrument=NVDA, **kwargs) if hit.setup == setup and hit.direction == direction]


def test_base_row_triggers_nothing() -> None:
    assert detect(BASE, instrument=NVDA) == []
    assert detect(BASE, instrument=NVDA, news=NewsContext(0.0, 0, (), None)) == []


@pytest.mark.parametrize(("setup", "direction"), list(ROWS), ids=[f"{s}-{d}" for s, d in ROWS])
def test_each_setup_triggers_in_each_direction(setup: str, direction: str) -> None:
    hits = detect(row_for(setup, direction), instrument=NVDA, news=news_for(setup, direction))
    assert len(hits) == 1, hits
    hit = hits[0]
    assert (hit.setup, hit.direction) == (setup, direction)
    assert hit.stop_dist > 0 and hit.target_dist > 0


def test_hit_parameters() -> None:
    p = ENGINE_PARAMS
    trend = hits_of(row_for("trend_pullback", "LONG"), "trend_pullback", "LONG")[0]
    assert trend == SetupHit("trend_pullback", "LONG", "15m", True, 240, 1.0, 1.5, 3.0)
    breakout = hits_of(row_for("breakout", "SHORT"), "breakout", "SHORT")[0]
    assert breakout == SetupHit("breakout", "SHORT", "15m", True, 180, 1.0, pytest.approx(1.2), pytest.approx(2.4))
    news = hits_of(row_for("news_momentum", "LONG"), "news_momentum", "LONG")[0]
    assert news == SetupHit("news_momentum", "LONG", "5m", True, 240, 1.0, 1.5, 3.0)
    reversion = hits_of(row_for("mean_reversion", "LONG"), "mean_reversion", "LONG")[0]
    assert reversion == SetupHit("mean_reversion", "LONG", "5m", True, 120, 1.0, 1.0, 2.0)  # capped at 2R
    daily = hits_of(row_for("daily_trend", "LONG"), "daily_trend", "LONG")[0]
    assert daily == SetupHit("daily_trend", "LONG", "1d", False, 5 * 390, 3.0, 6.0, 12.0)
    daily_xetra = hits_of(row_for("daily_trend", "LONG"), "daily_trend", "LONG", session_minutes=510)[0]
    assert daily_xetra.horizon_min == 5 * 510
    assert p.horizon_daily_days == 5


def test_mean_reversion_target_is_the_band_middle_when_closer() -> None:
    row = row_for("mean_reversion", "LONG", bb_mid_s=96.8)  # 1.4 to the middle: between 1.2 and 2 stops
    assert hits_of(row, "mean_reversion", "LONG")[0].target_dist == pytest.approx(1.4)
    too_close = row_for("mean_reversion", "LONG", bb_mid_s=96.5)  # 1.1 < 1.2 stops
    assert hits_of(too_close, "mean_reversion", "LONG") == []
    short = row_for("mean_reversion", "SHORT", bb_mid_s=103.0)
    assert hits_of(short, "mean_reversion", "SHORT")[0].target_dist == pytest.approx(1.6)


# ------------------------------------------------------------------------------------------------
# Guards and individual conditions
# ------------------------------------------------------------------------------------------------
GUARD_BREAKERS = [
    ("in_session", False),
    ("in_session", math.nan),
    ("minutes_since_open", 14.0),
    ("minutes_since_open", math.nan),
    ("minutes_to_close", 29.0),
    ("minutes_to_close", math.nan),
    ("atr_slow", 0.0),
    ("atr_slow", math.nan),
]


@pytest.mark.parametrize("setup", INTRADAY)
@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
@pytest.mark.parametrize(("key", "value"), GUARD_BREAKERS)
def test_common_intraday_guards_block(setup: str, direction: str, key: str, value: Any) -> None:
    assert hits_of(row_for(setup, direction, **{key: value}), setup, direction) == []


@pytest.mark.parametrize("setup", INTRADAY)
def test_guard_boundaries_are_inclusive(setup: str) -> None:
    row = row_for(setup, "LONG", minutes_since_open=15.0, minutes_to_close=30.0)
    assert len(hits_of(row, setup, "LONG")) == 1


CONDITION_BREAKERS: list[tuple[str, str, dict[str, Any]]] = [
    ("trend_pullback", "LONG", {"ema_slow_s": 100.0}),
    ("trend_pullback", "LONG", {"ema_fast_s_slope": 0.0}),
    ("trend_pullback", "LONG", {"adx_slow": 17.9}),
    ("trend_pullback", "LONG", {"pullback_low": 100.26}),
    ("trend_pullback", "LONG", {"price": 100.3}),  # not above the previous high
    ("trend_pullback", "LONG", {"ema_fast_f": 100.5}),
    ("trend_pullback", "LONG", {"ema_fast_s": 99.4, "ema_slow_s": 99.0, "pullback_low": 99.5}),  # extended > 1 ATR
    ("trend_pullback", "SHORT", {"ema_slow_s": 99.0}),
    ("trend_pullback", "SHORT", {"ema_fast_s_slope": 0.1}),
    ("trend_pullback", "SHORT", {"pullback_high": 99.7}),
    ("trend_pullback", "SHORT", {"price": 99.7}),
    ("breakout", "LONG", {"prev_price_f": 100.25}),  # not the first close beyond the level
    ("breakout", "LONG", {"price": 100.2}),
    ("breakout", "LONG", {"rel_volume": 1.49}),
    ("breakout", "LONG", {"ema_fast_s": 99.4}),  # against the slow trend
    ("breakout", "LONG", {"or_high": 100.6}),  # opening range high not yet taken out
    ("breakout", "SHORT", {"prev_price_f": 99.75}),
    ("breakout", "SHORT", {"rel_volume": 1.0}),
    ("breakout", "SHORT", {"ema_fast_s": 100.6}),
    ("breakout", "SHORT", {"or_low": 99.4}),
    ("mean_reversion", "LONG", {"adx_slow": 18.0}),
    ("mean_reversion", "LONG", {"rsi_fast": 25.0}),
    ("mean_reversion", "LONG", {"low_f": 95.0}),
    ("mean_reversion", "LONG", {"price": 95.0}),  # closes at its open: no reversal bucket
    ("mean_reversion", "LONG", {"prev_price_f": 95.5}),
    ("mean_reversion", "SHORT", {"rsi_fast": 75.0}),
    ("mean_reversion", "SHORT", {"high_f": 105.0}),
    ("mean_reversion", "SHORT", {"prev_price_f": 104.5}),
    ("news_momentum", "LONG", {"price": 99.9, "prev_price_f": 99.5}),  # bucket closes against the news
    ("news_momentum", "LONG", {"price": 100.2}),  # reaction 0.2 ATR and normal volume
    ("news_momentum", "LONG", {"vwap": 98.3}),  # 2.1 ATR away from VWAP: chased
    ("news_momentum", "SHORT", {"price": 100.1}),
    ("daily_trend", "LONG", {"prev_price_f": 112.1}),  # crossed earlier: not the first cross
    ("daily_trend", "LONG", {"price": 112.0}),
    ("daily_trend", "LONG", {"ema_daily_fast": 111.0}),
    ("daily_trend", "LONG", {"ema_daily_slow": 106.0}),
    ("daily_trend", "LONG", {"rsi_daily": 65.0}),  # no pullback: RSI too high and low far from EMA20
    ("daily_trend", "LONG", {"atr_daily": 0.0}),
    ("daily_trend", "LONG", {"in_session": False}),
    ("daily_trend", "LONG", {"minutes_since_open": 10.0}),
    ("daily_trend", "SHORT", {"prev_price_f": 87.9}),
    ("daily_trend", "SHORT", {"rsi_daily": 35.0}),
    ("daily_trend", "SHORT", {"ema_daily_fast": 89.0}),
]


@pytest.mark.parametrize(
    ("setup", "direction", "change"), CONDITION_BREAKERS, ids=[f"{s}-{d}-{c}" for s, d, c in CONDITION_BREAKERS]
)
def test_each_condition_blocks(setup: str, direction: str, change: dict[str, Any]) -> None:
    assert hits_of(row_for(setup, direction), setup, direction), "the unchanged row must trigger"
    assert hits_of(row_for(setup, direction, **change), setup, direction) == []


REQUIRED: dict[str, tuple[str, ...]] = {
    "trend_pullback": ("price", "ema_fast_s", "ema_slow_s", "ema_fast_s_slope", "adx_slow", "pullback_low",
                       "prev_high_f", "ema_fast_f"),
    "breakout": ("price", "prev_price_f", "donchian_upper_s", "donchian_lower_s", "ema_fast_s", "ema_slow_s"),
    "mean_reversion": ("price", "adx_slow", "rsi_fast", "low_f", "open_f", "prev_price_f", "bb_mid_s", "bb_lower_s"),
    "news_momentum": ("price", "open_f", "prev_price_f", "vwap"),
    "daily_trend": ("price", "prev_price_f", "prev_day_close", "prev_day_high", "prev_day_low", "atr_daily",
                    "ema_daily_fast", "ema_daily_slow", "ema_daily_pullback"),
}  # fmt: skip


@pytest.mark.parametrize(("setup", "key"), [(s, k) for s, keys in REQUIRED.items() for k in keys])
def test_nan_or_missing_required_feature_blocks(setup: str, key: str) -> None:
    assert hits_of(row_for(setup, "LONG", **{key: math.nan}), setup, "LONG") == []
    missing = row_for(setup, "LONG")
    del missing[key]
    assert hits_of(missing, setup, "LONG") == []


def test_optional_features() -> None:
    # breakout: an unknown opening range is ignored
    assert hits_of(row_for("breakout", "LONG", or_high=math.nan), "breakout", "LONG")
    # breakout without volume data: range expansion instead of relative volume
    assert hits_of(row_for("breakout", "LONG", rel_volume=math.nan, atr_fast=0.5), "breakout", "LONG")
    assert not hits_of(row_for("breakout", "LONG", rel_volume=math.nan, atr_fast=0.49), "breakout", "LONG")
    assert not hits_of(row_for("breakout", "LONG", rel_volume=math.nan, atr_fast=math.nan), "breakout", "LONG")
    # news: high relative volume confirms a small reaction; unknown volume does not
    assert hits_of(row_for("news_momentum", "LONG", price=100.1, rel_volume=1.3), "news_momentum", "LONG")
    assert not hits_of(row_for("news_momentum", "LONG", price=100.1, rel_volume=math.nan), "news_momentum", "LONG")
    # daily trend: the pullback may also be shown by yesterday's low near the daily EMA20
    pulled_back = row_for("daily_trend", "LONG", rsi_daily=65.0, ema_daily_pullback=106.0)  # low 107 <= 106 + 1.5
    assert hits_of(pulled_back, "daily_trend", "LONG")
    assert hits_of(row_for("daily_trend", "LONG", rsi_daily=math.nan, ema_daily_pullback=106.0), "daily_trend", "LONG")
    # daily trend has no last-minutes guard
    assert hits_of(row_for("daily_trend", "LONG", minutes_to_close=5.0), "daily_trend", "LONG")


def test_news_requirements() -> None:
    row = row_for("news_momentum", "LONG")
    assert hits_of(row, "news_momentum", "LONG", news=None) == []
    assert hits_of(row, "news_momentum", "LONG", news=NewsContext(0.5, 1, (1,), None)) == []
    assert hits_of(row, "news_momentum", "LONG", news=fresh_news(0.8, age_min=21)) == []  # stale for this row
    assert hits_of(row, "news_momentum", "LONG", news=fresh_news(0.8, relevance=3)) == []
    assert hits_of(row, "news_momentum", "LONG", news=fresh_news(0.45)) == []
    assert hits_of(row, "news_momentum", "LONG", news=fresh_news(0.8, age_min=20))
    # negative news never produces a LONG
    assert hits_of(row, "news_momentum", "LONG", news=fresh_news(-0.8)) == []


def known_news(*, analyzed_ts: int, fetched_ts: int, sentiment: float = 0.8) -> NewsContext:
    item = seen_news(7, analyzed_ts=analyzed_ts, fetched_ts=fetched_ts, sentiment=sentiment)
    return NewsContext(score=0.4 * np.sign(sentiment), count=1, news_ids=(7,), fresh=item)


def test_news_must_be_known_by_the_end_of_the_reacting_bucket() -> None:
    """CAUS-2: a delayed feed decides on a bucket that ended long before the decision. News that became
    known only after that bucket cannot be confirmed by its price move."""
    moutai = replace(NVDA, symbol="600519.SS", exchange="XSHG", delay_min=30)
    decision = NOW + (moutai.delay_min + 1) * 60  # the bucket ending at NOW is decided 31 minutes later
    late = known_news(analyzed_ts=decision - 120, fetched_ts=decision - 150)  # known 29 minutes after the bucket
    row = row_for("news_momentum", "LONG")  # data_ts = NOW
    assert hits_of(row, "news_momentum", "LONG", news=late) == []
    before = known_news(analyzed_ts=NOW - 60, fetched_ts=NOW - 120)  # known before the bucket ended
    assert hits_of(row, "news_momentum", "LONG", news=before)
    exactly = known_news(analyzed_ts=NOW, fetched_ts=NOW - 60)  # analysed at the bucket end: known
    assert hits_of(row, "news_momentum", "LONG", news=exactly)


def test_news_age_for_the_setup_counts_from_the_first_sight() -> None:
    """CAUS-3: analysed just now, but first seen 2 hours ago (restart recovery): not fresh for the tape."""
    row = row_for("news_momentum", "LONG")
    recovered = known_news(analyzed_ts=NOW - 60, fetched_ts=NOW - 7200)
    assert hits_of(row, "news_momentum", "LONG", news=recovered) == []
    edge = known_news(analyzed_ts=NOW - 60, fetched_ts=NOW - ENGINE_PARAMS.news_max_age_min * 60)
    assert hits_of(row, "news_momentum", "LONG", news=edge)
    over = known_news(analyzed_ts=NOW - 60, fetched_ts=NOW - ENGINE_PARAMS.news_max_age_min * 60 - 1)
    assert hits_of(row, "news_momentum", "LONG", news=over) == []


def test_params_are_used() -> None:
    row = row_for("breakout", "LONG", rel_volume=1.3)
    assert not hits_of(row, "breakout", "LONG")
    assert hits_of(row, "breakout", "LONG", params=EngineParams(breakout_relvol_min=1.2))


# ------------------------------------------------------------------------------------------------
# Selection rules
# ------------------------------------------------------------------------------------------------
def test_first_setup_in_order_wins_per_direction() -> None:
    both = {**row_for("breakout", "LONG"), "price": 100.5, "open_f": 100.0, "prev_price_f": 100.1}
    hits = detect(both, instrument=NVDA, news=fresh_news(0.8))
    assert [(h.setup, h.direction) for h in hits] == [("news_momentum", "LONG")]
    assert SETUPS == ("news_momentum", "breakout", "trend_pullback", "mean_reversion", "daily_trend")


def test_one_hit_per_direction_from_different_setups() -> None:
    row = row_for("trend_pullback", "LONG", open_f=100.6, prev_price_f=100.9, high_f=100.9)  # bucket closes down
    hits = detect(row, instrument=NVDA, news=fresh_news(-0.8))
    assert [(h.setup, h.direction) for h in hits] == [("trend_pullback", "LONG"), ("news_momentum", "SHORT")]


def test_a_setup_firing_both_ways_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    contradictory = [SetupHit("breakout", d, "15m", True, 180, 1.0, 1.2, 2.4) for d in ("LONG", "SHORT")]
    monkeypatch.setitem(setups._DETECTORS, "breakout", lambda *args: contradictory)
    hits = detect(row_for("trend_pullback", "LONG"), instrument=NVDA)
    assert [(h.setup, h.direction) for h in hits] == [("trend_pullback", "LONG")]


def test_random_rows_never_give_more_than_one_hit_per_direction() -> None:
    rng = np.random.default_rng(5)
    numeric = [k for k, v in BASE.items() if isinstance(v, float)]
    found = 0
    for _ in range(2000):
        row = dict(BASE)
        for key in numeric:
            row[key] = BASE[key] + rng.normal(0.0, 1.0) * (10.0 if "rsi" in key or "adx" in key else 0.6)
        row["minutes_since_open"], row["minutes_to_close"] = 60.0, 300.0
        row["atr_slow"], row["atr_daily"] = 1.0, 3.0
        hits = detect(row, instrument=NVDA, news=fresh_news(float(rng.choice([-0.8, 0.8]))))
        directions = [hit.direction for hit in hits]
        assert len(directions) == len(set(directions)) <= 2
        assert all(hit.stop_dist > 0 and hit.target_dist > 0 for hit in hits)
        found += len(hits)
    assert found > 0


def test_pandas_series_rows_with_numpy_scalars() -> None:
    row = pd.Series(row_for("trend_pullback", "LONG"), dtype=object)
    row["in_session"] = np.True_
    row["price"] = np.float64(100.5)
    assert [h.setup for h in detect(row, instrument=NVDA)] == ["trend_pullback"]
