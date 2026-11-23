"""Composite score: bounds, neutral inputs, monotonicity per input, fixed weights, special cases."""

import math
from typing import Any

import numpy as np
import pytest

from trading_local.analysis.features import NewsContext
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.analysis.scoring import score_hit
from trading_local.analysis.setups import SetupHit
from trading_local.testing import SAMPLE_INSTRUMENTS

NVDA = next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "NVDA")
NAMES = ("trend", "momentum", "volume", "structure", "market", "news", "volatility")


def hit(setup: str = "trend_pullback", direction: str = "LONG", target: float = 2.0) -> SetupHit:
    return SetupHit(setup, direction, "15m", True, 240, 1.0, 1.0, target)  # type: ignore[arg-type]


# Neutral row: trends flat, oscillators centred, normal volume, an opposing level at half the target in both
# directions, flat benchmark, volatility rank unknown.
NEUTRAL: dict[str, Any] = {
    "price": 100.0, "atr_fast": 0.5, "atr_slow": 1.0, "atr_daily": 3.0,
    "ema_fast_f": 100.0, "ema_slow_f": 100.0, "ema_fast_s": 100.0, "ema_slow_s": 100.0,
    "prev_day_close": 100.0, "ema_daily_fast": 100.0, "rsi_slow": 50.0, "rsi_fast": 50.0, "macd_hist_s": 0.0,
    "rel_volume": 1.0, "high_f": 100.1, "low_f": 99.9, "day_high": 101.0, "day_low": 99.0,
    "prev_day_high": 103.0, "prev_day_low": 97.0, "donchian_upper_s": 104.0, "donchian_lower_s": 96.0,
    "bb_upper_s": 102.5, "bb_lower_s": 97.5, "bench_trend": 0.0, "bench_ret_60": 0.0, "atr_pct_rank": math.nan,
}  # fmt: skip


def components(row: dict[str, Any], setup_hit: SetupHit | None = None, news: NewsContext | None = None) -> dict[str, float]:
    return score_hit(setup_hit or hit(), row, instrument=NVDA, news=news)[1]


def component(name: str, row: dict[str, Any], **kwargs: Any) -> float:
    return components(row, **kwargs)[name]


def test_neutral_inputs_give_neutral_components_and_score() -> None:
    for direction in ("LONG", "SHORT"):
        score, parts = score_hit(hit(direction=direction), NEUTRAL, instrument=NVDA)
        assert parts == {name: 0.5 for name in NAMES}
        assert score == 0.5


def test_missing_inputs_are_neutral() -> None:
    _, parts = score_hit(hit(), {}, instrument=NVDA)
    assert parts == {name: 0.5 for name in NAMES}


def test_score_is_the_weighted_sum_with_fixed_weights() -> None:
    row = {**NEUTRAL, "ema_fast_s": 101.0, "rel_volume": 1.8, "atr_pct_rank": 0.3, "bench_trend": 0.7}
    news = NewsContext(0.4, 2, (1, 2), None)
    score, parts = score_hit(hit(), row, instrument=NVDA, news=news)
    weights = ENGINE_PARAMS.weight_map
    assert sum(weights.values()) == pytest.approx(1.0)
    assert score == pytest.approx(sum(weights[name] * parts[name] for name in NAMES), abs=1e-3)
    assert score == round(score, 4) and all(value == round(value, 4) for value in parts.values())
    other = EngineParams(weights=(("trend", 1.0), ("momentum", 0.0), ("volume", 0.0), ("structure", 0.0),
                                  ("market", 0.0), ("news", 0.0), ("volatility", 0.0)))
    assert score_hit(hit(), row, instrument=NVDA, params=other)[0] == parts["trend"]


def test_scores_and_components_stay_within_bounds() -> None:
    rng = np.random.default_rng(3)
    for _ in range(500):
        row = {key: (value + rng.normal(0, 30.0) if isinstance(value, float) and not math.isnan(value) else value)
               for key, value in NEUTRAL.items()}
        row["rel_volume"] = abs(rng.normal(1.0, 5.0))
        row["atr_pct_rank"] = rng.uniform(-0.5, 1.5)
        setup = str(rng.choice(["trend_pullback", "breakout", "mean_reversion", "news_momentum", "daily_trend"]))
        direction = str(rng.choice(["LONG", "SHORT"]))
        news = NewsContext(float(rng.uniform(-1, 1)), 1, (1,), None)
        score, parts = score_hit(hit(setup, direction, target=float(rng.uniform(0.1, 5))), row, instrument=NVDA,
                                 news=news)
        assert 0.0 <= score <= 1.0
        assert all(0.0 <= value <= 1.0 for value in parts.values())


@pytest.mark.parametrize(
    ("name", "key", "values"),
    [
        ("trend", "ema_fast_f", [99.0, 99.5, 100.0, 100.5, 101.0]),
        ("trend", "ema_fast_s", [98.0, 99.0, 100.0, 101.0, 102.0]),
        ("trend", "prev_day_close", [94.0, 97.0, 100.0, 103.0, 106.0]),
        ("momentum", "rsi_slow", [20.0, 40.0, 50.0, 60.0, 80.0]),
        ("momentum", "macd_hist_s", [-1.0, -0.2, 0.0, 0.2, 1.0]),
        ("volume", "rel_volume", [0.2, 0.8, 1.0, 1.5, 2.5]),
        ("structure", "donchian_upper_s", [100.5, 100.9, 101.2, 101.6, 150.0]),
        ("market", "bench_trend", [-2.0, -0.5, 0.0, 0.5, 2.0]),
        ("market", "bench_ret_60", [-1.0, -0.2, 0.0, 0.2, 1.0]),
    ],
)
def test_components_are_monotone_in_their_inputs(name: str, key: str, values: list[float]) -> None:
    base = dict(NEUTRAL)
    if name == "structure":  # only the donchian level remains as resistance above
        base.update(day_high=100.0, prev_day_high=200.0)
    long_values = [component(name, {**base, key: value}) for value in values]
    short_values = [component(name, {**base, key: value}, setup_hit=hit(direction="SHORT")) for value in values]
    assert long_values == sorted(long_values) and long_values[0] < long_values[-1]
    if name not in ("volume", "structure"):  # direction-free or one-sided inputs
        assert short_values == sorted(short_values, reverse=True) and short_values[0] > short_values[-1]


def test_news_component() -> None:
    assert component("news", NEUTRAL, news=NewsContext(0.6, 1, (1,), None)) == pytest.approx(0.8)
    assert component("news", NEUTRAL, setup_hit=hit(direction="SHORT"), news=NewsContext(0.6, 1, (1,), None)) == 0.2
    assert component("news", NEUTRAL, news=None) == 0.5


def test_volatility_component_peaks_in_the_middle_with_a_floor() -> None:
    values = {rank: component("volatility", {**NEUTRAL, "atr_pct_rank": rank}) for rank in (0.0, 0.25, 0.5, 0.75, 1.0)}
    assert values[0.5] == 1.0
    assert values[0.25] == values[0.75] == 0.5
    assert values[0.0] == values[1.0] == 0.2  # floor
    assert component("volatility", {**NEUTRAL, "atr_pct_rank": 0.45}) == pytest.approx(0.9)


def test_volume_component_values() -> None:
    assert component("volume", {**NEUTRAL, "rel_volume": 2.0}) == 0.75
    assert component("volume", {**NEUTRAL, "rel_volume": 4.0}) == 1.0
    assert component("volume", {**NEUTRAL, "rel_volume": math.nan}) == 0.5


def test_trend_terms_are_skipped_when_missing() -> None:
    row = {**NEUTRAL, "ema_fast_s": 101.0, "ema_fast_f": math.nan, "prev_day_close": math.nan}
    assert component("trend", row) == pytest.approx(0.5 + 0.5 * math.tanh(1.0), abs=1e-4)
    zero_atr = {**NEUTRAL, "atr_slow": 0.0, "ema_fast_s": 101.0}
    assert component("trend", zero_atr) == 0.5


def test_mean_reversion_special_cases() -> None:
    row = {**NEUTRAL, "ema_fast_s": 90.0, "rsi_fast": 20.0, "rsi_slow": 10.0}
    parts = components(row, setup_hit=hit("mean_reversion"))
    assert parts["trend"] == 0.5  # neutral by definition
    assert parts["momentum"] == pytest.approx(0.6)  # (|20 - 50| - 15) / 25
    assert component("momentum", {**row, "rsi_fast": 10.0}, setup_hit=hit("mean_reversion")) == 1.0
    assert component("momentum", {**row, "rsi_fast": 40.0}, setup_hit=hit("mean_reversion")) == 0.0
    assert component("momentum", {**row, "rsi_fast": 85.0}, setup_hit=hit("mean_reversion", "SHORT")) == 0.8
    # the band on the far side counts as opposing level for mean reversion only
    level_row = {**NEUTRAL, "day_high": 100.0, "prev_day_high": 200.0, "donchian_upper_s": 200.0, "bb_upper_s": 101.0}
    assert component("structure", level_row, setup_hit=hit("mean_reversion")) == 0.5
    assert component("structure", level_row, setup_hit=hit("trend_pullback")) == 1.0


def test_structure_rules() -> None:
    blue_sky = {**NEUTRAL, "day_high": 100.1, "prev_day_high": 99.0, "donchian_upper_s": 98.0}
    assert component("structure", blue_sky) == 1.0  # nothing above (the session high is the current bucket)
    earlier_high = {**blue_sky, "day_high": 100.5}
    assert component("structure", earlier_high) == 0.25  # 0.5 room / 2.0 target
    assert component("structure", earlier_high, setup_hit=hit(target=0.4)) == 1.0
    # breakout ignores the level just broken, daily_trend ignores yesterday's high
    broken = {**blue_sky, "donchian_upper_s": 100.4}
    assert component("structure", broken) == 0.2
    assert component("structure", broken, setup_hit=hit("breakout")) == 1.0
    daily = {**blue_sky, "prev_day_high": 100.4}
    assert component("structure", daily, setup_hit=hit("trend_pullback")) == 0.2
    assert component("structure", daily, setup_hit=hit("daily_trend")) == 1.0
    # SHORT looks below
    below = {**NEUTRAL, "low_f": 99.9, "day_low": 99.0, "prev_day_low": 50.0, "donchian_lower_s": 50.0}
    assert component("structure", below, setup_hit=hit(direction="SHORT")) == 0.5


def test_market_component_formula() -> None:
    row = {**NEUTRAL, "bench_trend": 0.4, "bench_ret_60": 0.3}
    expected = 0.5 + 0.5 * math.tanh((0.4 + 0.3 / 0.5) / 2)
    assert component("market", row) == pytest.approx(expected, abs=1e-4)
    assert component("market", row, setup_hit=hit(direction="SHORT")) == pytest.approx(1 - expected, abs=1e-4)
    assert component("market", {**NEUTRAL, "bench_trend": math.nan, "bench_ret_60": math.nan}) == 0.5
    only_return = {**NEUTRAL, "bench_trend": math.nan, "bench_ret_60": 0.5}
    assert component("market", only_return) == pytest.approx(0.5 + 0.5 * math.tanh(0.5), abs=1e-4)


def test_scoring_is_deterministic() -> None:
    row = {**NEUTRAL, "ema_fast_s": 100.7, "rel_volume": 1.3, "atr_pct_rank": 0.61}
    assert score_hit(hit(), row, instrument=NVDA) == score_hit(hit(), dict(row), instrument=NVDA)
