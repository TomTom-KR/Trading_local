"""Calibration: prior, shrinkage, monotonicity (PAVA), interpolation, reliability bins, no future outcomes."""

import math

import numpy as np
import pytest

from trading_local.analysis.calibration import Calibrator, breakeven, outcomes_before
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.models import Signal

SCORES = np.linspace(0.0, 1.0, 101)


def test_breakeven() -> None:
    assert breakeven(2.0) == pytest.approx(1.0 / 3.0)
    assert breakeven(1.0) == 0.5
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            breakeven(bad)


def test_prior_only_calibrator() -> None:
    prior = Calibrator()
    assert prior.lift(0.5) == pytest.approx(0.0)
    assert prior.lift(0.8) == pytest.approx(0.4 * 0.3)
    assert prior.probability(0.5, 2.0) == pytest.approx(1.0 / 3.0)
    assert prior.probability(0.75, 1.0) == pytest.approx(0.6)
    assert prior.probability(0.0, 0.01) == pytest.approx(1.0 / 1.01 - 0.2)
    assert prior.probability(1.0, 0.01) == 0.98  # clipped
    assert prior.probability(0.0, 100.0) == 0.02  # clipped
    with pytest.raises(ValueError):
        prior.probability(0.5, 0.0)


def test_expected_r() -> None:
    prior = Calibrator()
    p = prior.probability(0.7, 2.0)
    assert prior.expected_r(0.7, 2.0, 0.05) == pytest.approx(p * 2.0 - (1.0 - p) - 0.05)
    assert prior.expected_r(0.5, 2.0, 0.0) == pytest.approx(0.0)  # break-even at a neutral score


@pytest.mark.parametrize(("score", "rr", "cost_r"), [(0.7, 2.0, 0.05), (0.55, 1.2, 0.0), (0.9, 3.0, 0.3)])
def test_expected_r_without_outcomes_is_the_binary_prior_formula(score: float, rr: float, cost_r: float) -> None:
    for calibrator in (Calibrator(), Calibrator.fit([]), Calibrator.fit([(0.7, 0.0, 1.0)])):
        p = calibrator.probability(score, rr)
        assert calibrator.expected_r(score, rr, cost_r) == pytest.approx(p * rr - (1.0 - p) - cost_r, abs=1e-12)


def test_expected_r_uses_the_realised_payoffs_of_time_stop_exits() -> None:
    """MONEY-2: winners closed by the time stop at +0.2 R (a tenth of the 2 R target). The binary formula
    counts each of them as +2 R and calls the setup profitable; the payoff-aware value is negative."""
    outcomes = [(0.75, 2.0, 0.2)] * 300 + [(0.75, 2.0, -1.0)] * 300
    fitted = Calibrator.fit(outcomes)
    p = fitted.probability(0.75, 2.0)
    k = ENGINE_PARAMS.prior_strength
    cost = 0.05
    win = (2.0 * 300 * 0.1 + k * (2.0 - cost)) / (300 + k)
    loss = (300 * -1.0 + k * (-1.0 - cost)) / (300 + k)
    assert fitted.expected_r(0.75, 2.0, cost) == pytest.approx(p * win + (1.0 - p) * loss)
    binary = p * 2.0 - (1.0 - p) - cost
    assert binary > 0.0 > fitted.expected_r(0.75, 2.0, cost)


def test_realised_payoffs_scale_with_the_reward_to_risk_and_count_costs_once() -> None:
    # wins as a fraction of their own rr: half the target at rr 2 is worth half the target at rr 3 as well
    fitted = Calibrator.fit([(0.65, 2.0, 1.0)] * 500 + [(0.65, 2.0, -0.8)] * 500)
    k = ENGINE_PARAMS.prior_strength
    p3 = fitted.probability(0.65, 3.0)
    win3 = (3.0 * 500 * 0.5 + k * 3.0) / (500 + k)
    loss = (500 * -0.8 + k * -1.0) / (500 + k)
    assert fitted.expected_r(0.65, 3.0, 0.0) == pytest.approx(p3 * win3 + (1.0 - p3) * loss)
    # realised results already contain the costs: with many outcomes cost_r barely matters any more
    many = Calibrator.fit([(0.65, 2.0, 1.5)] * 5000 + [(0.65, 2.0, -1.05)] * 5000)
    drift = many.expected_r(0.65, 2.0, 0.0) - many.expected_r(0.65, 2.0, 0.3)
    assert 0.0 < drift < 0.3 * 0.01


def test_fit_without_outcomes_equals_the_prior_everywhere() -> None:
    prior, fitted = Calibrator(), Calibrator.fit([])
    for score in SCORES:
        assert fitted.lift(score) == pytest.approx(prior.lift(score), abs=1e-15)
        assert fitted.probability(score, 1.5) == pytest.approx(prior.probability(score, 1.5), abs=1e-15)


def test_invalid_outcomes_are_ignored() -> None:
    fitted = Calibrator.fit([(math.nan, 2.0, 1.0), (0.65, 0.0, 1.0), (0.65, 2.0, math.nan), (0.65, math.inf, 1.0)])
    assert fitted.probability(0.65, 2.0) == pytest.approx(Calibrator().probability(0.65, 2.0))


def test_few_outcomes_barely_move_the_prior() -> None:
    losers = [(0.65, 2.0, -1.0)] * 10
    prior, fitted = Calibrator(), Calibrator.fit(losers)
    shift = prior.probability(0.65, 2.0) - fitted.probability(0.65, 2.0)
    assert 0.0 < shift < 0.08


def test_posterior_formula_for_one_bin() -> None:
    # 60 winners and 40 losers at rr 1 in the bin 0.7..0.8 (centre 0.75, prior lift 0.1)
    outcomes = [(0.75, 1.0, 1.0)] * 60 + [(0.75, 1.0, -1.0)] * 40
    fitted = Calibrator.fit(outcomes)
    expected_lift = (60 * 0.5 - 40 * 0.5 + 40 * 0.1) / (100 + 40)
    assert fitted.lift(0.75) == pytest.approx(expected_lift)


def test_many_outcomes_dominate_the_prior() -> None:
    outcomes = [(0.75, 1.0, 1.0)] * 9_000 + [(0.75, 1.0, -1.0)] * 1_000
    assert Calibrator.fit(outcomes).probability(0.75, 1.0) == pytest.approx(0.9, abs=0.01)


def test_pava_makes_the_lift_monotone_with_adversarial_data() -> None:
    # high scores always lose, low scores always win: the raw posterior would decrease
    outcomes = [(0.2, 2.0, 2.0)] * 300 + [(0.85, 2.0, -1.0)] * 300 + [(0.55, 2.0, 2.0)] * 50
    fitted = Calibrator.fit(outcomes)
    lifts = [fitted.lift(score) for score in SCORES]
    assert all(b >= a - 1e-12 for a, b in zip(lifts, lifts[1:]))
    probabilities = [fitted.probability(score, 2.0) for score in SCORES]
    assert all(b >= a - 1e-12 for a, b in zip(probabilities, probabilities[1:]))


@pytest.mark.parametrize("seed", range(5))
def test_lift_is_monotone_for_random_outcomes(seed: int) -> None:
    rng = np.random.default_rng(seed)
    outcomes = [
        (float(rng.uniform(0, 1)), float(rng.choice([1.0, 1.5, 2.0])), float(rng.choice([-1.0, 1.8])))
        for _ in range(int(rng.integers(5, 400)))
    ]
    fitted = Calibrator.fit(outcomes)
    lifts = [fitted.lift(score) for score in SCORES]
    assert all(b >= a - 1e-12 for a, b in zip(lifts, lifts[1:]))


def test_interpolation_between_bin_centres() -> None:
    outcomes = [(0.65, 1.0, 1.0)] * 40 + [(0.55, 1.0, -1.0)] * 40
    fitted = Calibrator.fit(outcomes)
    left, right = fitted.lift(0.55), fitted.lift(0.65)
    assert fitted.lift(0.60) == pytest.approx((left + right) / 2.0)
    assert fitted.lift(0.58) == pytest.approx(left + 0.3 * (right - left))


def test_outer_regions_keep_the_deviation_from_the_prior() -> None:
    fitted = Calibrator.fit([(0.9, 2.0, -1.0)] * 30 + [(0.1, 2.0, 2.0)] * 30)
    prior = Calibrator()
    deviation_top = fitted.lift(0.9) - prior.lift(0.9)
    assert fitted.lift(0.97) - prior.lift(0.97) == pytest.approx(deviation_top)
    deviation_bottom = fitted.lift(0.25) - prior.lift(0.25)
    assert fitted.lift(0.05) - prior.lift(0.05) == pytest.approx(deviation_bottom)


def test_bins_report() -> None:
    outcomes = [(0.65, 2.0, 1.0)] * 3 + [(0.65, 2.0, -1.0)] * 7 + [(0.3, 1.0, 1.0)]
    fitted = Calibrator.fit(outcomes)
    bins = fitted.bins()
    assert [(b.lo, b.hi) for b in bins] == [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    assert [b.n for b in bins] == [1, 0, 10, 0, 0]
    assert bins[2].observed == pytest.approx(0.3)
    assert bins[2].predicted == pytest.approx(fitted.probability(0.65, 2.0))
    assert bins[0].observed == 1.0 and bins[0].predicted == pytest.approx(fitted.probability(0.3, 1.0))
    assert bins[1].observed is None
    assert bins[1].predicted == pytest.approx(fitted.probability(0.55, 2.0))
    prior_bins = Calibrator().bins()
    assert all(b.n == 0 and b.observed is None for b in prior_bins)


def test_fit_is_independent_of_the_order_of_outcomes() -> None:
    rng = np.random.default_rng(9)
    outcomes = [(float(rng.uniform(0, 1)), 2.0, float(rng.choice([-1.0, 2.0]))) for _ in range(200)]
    shuffled = [outcomes[i] for i in rng.permutation(len(outcomes))]
    first, second = Calibrator.fit(outcomes), Calibrator.fit(shuffled)
    assert [first.lift(s) for s in SCORES] == pytest.approx([second.lift(s) for s in SCORES], abs=1e-15)
    assert [(b.n, b.observed) for b in first.bins()] == [(b.n, b.observed) for b in second.bins()]


def test_probability_falls_with_the_reward_to_risk() -> None:
    fitted = Calibrator.fit([(0.65, 2.0, 2.0)] * 30 + [(0.65, 2.0, -1.0)] * 30)
    for score in (0.2, 0.55, 0.65, 0.9):
        values = [fitted.probability(score, rr) for rr in (0.5, 1.0, 1.5, 2.0, 3.0)]
        assert values == sorted(values, reverse=True)


def test_scores_outside_the_range_use_the_outer_bins() -> None:
    fitted = Calibrator.fit([(1.2, 1.0, 1.0)] * 20 + [(-0.3, 1.0, -1.0)] * 20)
    bins = fitted.bins()
    assert bins[-1].n == 20 and bins[0].n == 20


def test_params_drive_the_prior() -> None:
    strong = EngineParams(prior_slope=0.8, prior_strength=10.0)
    assert Calibrator(strong).lift(0.75) == pytest.approx(0.2)
    losers = [(0.65, 2.0, -1.0)] * 10
    assert Calibrator.fit(losers, strong).lift(0.65) < Calibrator.fit(losers, ENGINE_PARAMS).lift(0.65)


# ------------------------------------------------------------------------------------------------
# Only outcomes known before the decision time
# ------------------------------------------------------------------------------------------------
def closed_signal(exit_ts: int | None, outcome: float | None, *, status: str = "target", score: float = 0.7) -> Signal:
    return Signal(
        created_ts=0, data_ts=0, symbol="NVDA", direction="LONG", setup="breakout", timeframe="15m", horizon_min=180,
        ref_price=100.0, atr=1.0, stop_dist=1.0, target_dist=2.0, score=score, confidence=0.4, engine_version="e1-x",
        status=status, exit_ts=exit_ts, outcome_r=outcome,  # type: ignore[arg-type]
    )  # fmt: skip


def test_outcomes_before_uses_only_signals_closed_earlier() -> None:
    signals = [
        closed_signal(100, 2.0),
        closed_signal(199, -1.0, status="stopped"),
        closed_signal(200, 2.0),  # exits exactly at the decision time: not yet known
        closed_signal(300, -1.0, status="stopped"),
        closed_signal(None, None, status="active"),
        closed_signal(150, None, status="cancelled"),
    ]
    assert outcomes_before(signals, before_ts=200) == [(0.7, 2.0, 2.0), (0.7, 2.0, -1.0)]
    fitted = Calibrator.from_signals(signals, before_ts=200)
    assert fitted.lift(0.7) == pytest.approx(Calibrator.fit([(0.7, 2.0, 2.0), (0.7, 2.0, -1.0)]).lift(0.7))
