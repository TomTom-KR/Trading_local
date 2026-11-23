"""Outcome statistics: Wilson and bootstrap intervals against known values, determinism, grouping."""

import math

import numpy as np
import pytest

from trading_local.analysis.stats import bootstrap_mean_ci, cumulative_r, group_stats, stat_block, wilson_interval
from trading_local.dto import StatBlock
from trading_local.models import Signal


@pytest.mark.parametrize(
    ("wins", "n", "expected"),
    [
        (8, 10, (0.4902, 0.9433)),  # textbook example (Newcombe 1998)
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (50, 100, (0.4038, 0.5962)),
        (1, 1, (0.2065, 1.0)),
    ],
)
def test_wilson_interval_known_values(wins: int, n: int, expected: tuple[float, float]) -> None:
    lo, hi = wilson_interval(wins, n)
    assert lo == pytest.approx(expected[0], abs=1e-4) and hi == pytest.approx(expected[1], abs=1e-4)


def test_wilson_interval_edge_cases() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo90, hi90 = wilson_interval(8, 10, z=1.645)
    lo95, hi95 = wilson_interval(8, 10)
    assert lo95 < lo90 < 0.8 < hi90 < hi95


def test_bootstrap_is_deterministic_and_seed_dependent() -> None:
    values = list(np.random.default_rng(1).normal(0.2, 1.0, 40))
    assert bootstrap_mean_ci(values) == bootstrap_mean_ci(values)
    assert bootstrap_mean_ci(values) != bootstrap_mean_ci(values, seed=1)
    lo, hi = bootstrap_mean_ci(values)
    assert lo < float(np.mean(values)) < hi


def test_bootstrap_matches_the_normal_approximation_for_large_samples() -> None:
    values = np.random.default_rng(2).normal(0.5, 2.0, 400)
    lo, hi = bootstrap_mean_ci(list(values), n_boot=4000)
    half = 1.96 * values.std(ddof=1) / math.sqrt(values.size)
    assert lo == pytest.approx(values.mean() - half, abs=0.03)
    assert hi == pytest.approx(values.mean() + half, abs=0.03)


def test_bootstrap_known_small_cases() -> None:
    assert bootstrap_mean_ci([1.5] * 7) == (1.5, 1.5)
    # resample means of [0, 1] are 0 (25 %), 0.5 (50 %) and 1 (25 %): the 2.5 % / 97.5 % quantiles are 0 and 1
    assert bootstrap_mean_ci([0.0, 1.0], n_boot=5000) == (0.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])


def test_bootstrap_large_inputs_are_chunked_but_stay_deterministic() -> None:
    values = np.random.default_rng(3).normal(0.0, 1.0, 5000)
    first = bootstrap_mean_ci(list(values), n_boot=1000)  # 5,000 values x 1,000 resamples: several blocks
    assert first == bootstrap_mean_ci(list(values), n_boot=1000)
    assert first[0] < values.mean() < first[1]
    assert first[1] - first[0] == pytest.approx(2 * 1.96 / math.sqrt(values.size), rel=0.15)


def test_stat_block() -> None:
    block = stat_block([2.0, -1.0, -1.0, 2.0, 0.5, -0.2, math.nan])
    assert (block.n, block.wins) == (6, 3)
    assert block.win_rate == pytest.approx(0.5)
    assert (block.win_rate_lo, block.win_rate_hi) == pytest.approx(wilson_interval(3, 6))
    assert block.avg_r == pytest.approx(2.3 / 6)
    assert block.total_r == pytest.approx(2.3)
    assert block.profit_factor == pytest.approx(4.5 / 2.2)
    assert (block.avg_r_lo, block.avg_r_hi) == bootstrap_mean_ci([2.0, -1.0, -1.0, 2.0, 0.5, -0.2])


def test_stat_block_small_and_empty() -> None:
    assert stat_block([]) == StatBlock()
    small = stat_block([1.0, 2.0, 0.0, 3.0])
    assert small.n == 4 and small.wins == 3  # 0.0 is not a win
    assert small.avg_r_lo is None and small.avg_r_hi is None
    assert small.profit_factor is None  # no losses


def signal(setup: str, outcome: float | None, exit_ts: int | None, signal_id: int, direction: str = "LONG") -> Signal:
    return Signal(
        created_ts=0, data_ts=0, symbol="NVDA", direction=direction, setup=setup, timeframe="15m",  # type: ignore[arg-type]
        horizon_min=180, ref_price=100.0, atr=1.0, stop_dist=1.0, target_dist=2.0, score=0.6, confidence=0.4,
        engine_version="e1-x", status="target" if outcome is not None else "cancelled", outcome_r=outcome,
        exit_ts=exit_ts, id=signal_id,
    )  # fmt: skip


SIGNALS = [
    signal("breakout", 2.0, 300, 1),
    signal("breakout", -1.0, 100, 2),
    signal("trend_pullback", 2.0, 200, 3, "SHORT"),
    signal("breakout", -1.0, 200, 4),
    signal("mean_reversion", None, None, 5),
    signal("trend_pullback", -1.0, 400, 6),
]


def test_group_stats_sorted_by_count_with_labels() -> None:
    groups = group_stats(SIGNALS, key=lambda s: s.setup, labels={"breakout": "Ausbruch"})
    assert [(g.key, g.label_de, g.stats.n) for g in groups] == [
        ("breakout", "Ausbruch", 3),
        ("trend_pullback", "trend_pullback", 2),
    ]
    assert groups[0].stats.total_r == pytest.approx(0.0)
    by_direction = group_stats(SIGNALS, key=lambda s: s.direction)
    assert [(g.key, g.stats.n) for g in by_direction] == [("LONG", 4), ("SHORT", 1)]
    assert group_stats([], key=lambda s: s.setup) == []


def test_cumulative_r_in_exit_order() -> None:
    assert cumulative_r(SIGNALS) == [(100, -1.0), (200, 1.0), (200, 0.0), (300, 2.0), (400, 1.0)]
    assert cumulative_r([]) == []


def test_non_finite_results_never_poison_the_statistics() -> None:
    broken = [*SIGNALS, signal("breakout", math.nan, 150, 7)]
    assert cumulative_r(broken) == cumulative_r(SIGNALS)
    breakout = next(g for g in group_stats(broken, key=lambda s: s.setup) if g.key == "breakout")
    assert breakout.stats.n == 3 and breakout.stats.total_r == pytest.approx(0.0)
