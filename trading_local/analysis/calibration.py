"""Score -> win probability and payoffs: the ONLY things the engine learns from data.

The probability is ``P(R > 0)``: a win is any closed trade with a positive result in R, whether it reached
its target or was closed in profit by the time stop or the session close.

A trade with reward-to-risk ``rr`` whose wins pay ``+rr`` and whose losses pay ``-1`` breaks even at a win
rate of ``1 / (1 + rr)``. The calibrator predicts ``breakeven(rr) + lift(score)``: the score only shifts the
probability around that break-even.

* The prior (no data): ``lift = prior_slope * (score - 0.5)`` - an a-priori belief that a better score is
  worth a little, never fitted.
* ``fit``: per score bin the observed excess win rate over break-even is shrunk towards the prior with
  ``prior_strength`` pseudo-observations, then forced to be monotone (pool-adjacent-violators). With few
  outcomes the prior dominates, so a handful of lucky or unlucky trades cannot swing the probabilities.

Payoffs are ESTIMATED, not assumed binary: time stops and session-close exits give fractional results, so
the average win is usually less than ``+rr`` and the average loss can differ from ``-1``. ``fit`` stores the
realised payoffs (wins as a fraction of their ``rr``, losses in R) and ``expected_r`` shrinks them towards
the binary prior payoffs with the same ``prior_strength``; without outcomes it is exactly
``p * rr - (1 - p) - cost_r``.

Outcomes must come from signals that were CLOSED before the decision time (``outcomes_before``,
``repo_signals.closed_outcomes(before_ts=...)``), otherwise the calibration would see the future.
"""

import math
from collections.abc import Sequence

import numpy as np

from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.dto import CalibrationBin
from trading_local.models import SIGNAL_OUTCOME_STATUSES, Signal

type Outcome = tuple[float, float, float]  # (score, rr, outcome_r) of one closed signal

_DISPLAY_RR = 2.0  # reward-to-risk used to show the prediction of an empty bin


def breakeven(rr: float) -> float:
    """Win rate (``P(R > 0)``) at which a trade with reward-to-risk ``rr`` (> 0) has zero expectancy before
    costs if every win paid ``+rr`` and every loss ``-1``.

    It is the reference point of the probability calibration only; the expected result uses estimated
    payoffs (``Calibrator.expected_r``), because real exits are not binary.
    """
    if not math.isfinite(rr) or rr <= 0.0:
        raise ValueError(f"reward-to-risk must be a positive number, got {rr}")
    return 1.0 / (1.0 + rr)


def outcomes_before(signals: Sequence[Signal], *, before_ts: int) -> list[Outcome]:
    """(score, rr, outcome_r) of the signals with a realised result that exited strictly before ``before_ts``.

    The in-memory twin of ``repo_signals.closed_outcomes(before_ts=...)`` for the validation replay.
    """
    return [
        (signal.score, signal.rr, signal.outcome_r)
        for signal in signals
        if signal.status in SIGNAL_OUTCOME_STATUSES
        and signal.outcome_r is not None
        and signal.exit_ts is not None
        and signal.exit_ts < before_ts
    ]


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool-adjacent-violators); a few bins, so a plain loop."""
    blocks: list[list[float]] = []  # [weighted mean, weight, number of bins]
    for value, weight in zip(values.tolist(), weights.tolist()):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks[-1]
            total = left[1] + right[1]
            left[0] = (left[0] * left[1] + right[0] * right[1]) / total
            left[1] = total
            left[2] += right[2]
    return np.repeat([block[0] for block in blocks], [int(block[2]) for block in blocks])


class Calibrator:
    """Maps a composite score to a win probability (win := outcome_r > 0) and to an expected result in R.

    ``Calibrator(params)`` is the prior-only calibrator. ``Calibrator.fit(outcomes, params)`` returns a
    calibrator with per-bin posterior lifts and the realised payoffs. Between bin centres the fitted lift is
    interpolated linearly; beyond the outer centres the fitted DEVIATION from the prior is held constant.
    Because the prior is linear in the score this is exactly linear interpolation of the bin lifts between
    the centres, and it makes ``fit([])`` identical to the prior everywhere. The lift is non-decreasing in
    the score.
    """

    def __init__(self, params: EngineParams = ENGINE_PARAMS) -> None:
        self._params = params
        edges = np.asarray(params.calibration_bins, dtype=np.float64)
        self._lo = edges[:-1]
        self._hi = edges[1:]
        self._centres = (self._lo + self._hi) / 2.0
        self._deviation = np.zeros(self._centres.size)  # fitted lift minus prior lift at each centre
        self._observations: list[Outcome] = []
        # Realised payoffs (all zero for the prior-only calibrator): number of wins and the sum of their
        # results as a fraction of their reward-to-risk; number of losses and the sum of their results in R.
        self._wins = 0
        self._win_fraction_sum = 0.0
        self._losses = 0
        self._loss_r_sum = 0.0

    # -- construction -----------------------------------------------------------------------------
    @classmethod
    def fit(cls, outcomes: Sequence[Outcome], params: EngineParams = ENGINE_PARAMS) -> "Calibrator":
        """Calibrator fitted on (score, rr, outcome_r) of CLOSED signals.

        Per bin: observed excess = sum(win - breakeven(rr)); posterior lift = (excess + prior_strength *
        prior lift at the bin centre) / (n + prior_strength); then PAVA with weights n + prior_strength.
        Payoffs (pooled over all bins): wins ``n_w`` and ``S_w = sum(outcome_r / rr)`` over the wins, losses
        ``n_l`` and ``S_l = sum(outcome_r)`` over the losses (see ``expected_r``).
        Outcomes with a non-finite value or rr <= 0 are ignored; scores outside the bin range count in the
        outer bins.
        """
        calibrator = cls(params)
        valid = [
            (float(score), float(rr), float(result))
            for score, rr, result in outcomes
            if math.isfinite(score) and math.isfinite(rr) and rr > 0.0 and math.isfinite(result)
        ]
        calibrator._observations = valid
        if not valid:
            return calibrator
        data = np.asarray(valid, dtype=np.float64)
        won = data[:, 2] > 0.0
        calibrator._wins = int(won.sum())
        calibrator._win_fraction_sum = float(np.sum(data[won, 2] / data[won, 1]))
        calibrator._losses = int((~won).sum())
        calibrator._loss_r_sum = float(np.sum(data[~won, 2]))
        bins = calibrator._bin_of(data[:, 0])
        excess = won - 1.0 / (1.0 + data[:, 1])
        counts = np.bincount(bins, minlength=calibrator._centres.size).astype(np.float64)
        sums = np.bincount(bins, weights=excess, minlength=calibrator._centres.size)
        prior = calibrator._prior_lift(calibrator._centres)
        strength = params.prior_strength
        posterior = (sums + strength * prior) / (counts + strength)
        calibrator._deviation = _pava(posterior, counts + strength) - prior
        return calibrator

    @classmethod
    def from_signals(
        cls, signals: Sequence[Signal], *, before_ts: int, params: EngineParams = ENGINE_PARAMS
    ) -> "Calibrator":
        """Calibrator fitted on the signals that had closed with a result before ``before_ts``."""
        return cls.fit(outcomes_before(signals, before_ts=before_ts), params)

    # -- prediction -------------------------------------------------------------------------------
    def lift(self, score: float) -> float:
        """Probability shift over break-even for a score."""
        return float(self._lifts(np.asarray([score], dtype=np.float64))[0])

    def probability(self, score: float, rr: float) -> float:
        """Win probability of a trade with this score and reward-to-risk (> 0), clipped to the configured bounds."""
        breakeven(rr)  # validates rr
        return float(self._probabilities(np.asarray([score], dtype=np.float64), np.asarray([rr]))[0])

    def expected_r(self, score: float, rr: float, cost_r: float) -> float:
        """Expected result in R after costs: ``p * win_payoff + (1 - p) * loss_payoff`` with ``p = P(R > 0)``.

        The payoffs are the realised ones, shrunk towards the binary prior payoffs (``rr - cost_r`` for a
        win, ``-1 - cost_r`` for a loss) with ``k = prior_strength`` pseudo-observations::

            win_payoff  = (rr * S_w + k * (rr - cost_r)) / (n_w + k)
            loss_payoff = (S_l + k * (-1 - cost_r)) / (n_l + k)

        Without outcomes this is exactly ``p * rr - (1 - p) - cost_r``. Realised results already contain
        slippage and costs, so ``cost_r`` only enters through the prior part and is never subtracted twice.
        """
        p = self.probability(score, rr)
        k = self._params.prior_strength
        win_payoff = rr - cost_r
        if self._wins + k > 0.0:
            win_payoff = (rr * self._win_fraction_sum + k * (rr - cost_r)) / (self._wins + k)
        loss_payoff = -1.0 - cost_r
        if self._losses + k > 0.0:
            loss_payoff = (self._loss_r_sum + k * (-1.0 - cost_r)) / (self._losses + k)
        return p * win_payoff + (1.0 - p) * loss_payoff

    def bins(self) -> list[CalibrationBin]:
        """Reliability table: per bin the number of fitted outcomes, the mean predicted probability (of the
        bin centre at rr = 2 when the bin is empty) and the observed win rate (None when empty)."""
        observations = np.asarray(self._observations, dtype=np.float64).reshape(-1, 3)
        bins = self._bin_of(observations[:, 0]) if observations.size else np.zeros(0, dtype=np.int64)
        table: list[CalibrationBin] = []
        for index, (lo, hi, centre) in enumerate(zip(self._lo, self._hi, self._centres)):
            members = observations[bins == index]
            if members.shape[0] == 0:
                predicted = self.probability(float(centre), _DISPLAY_RR)
                observed = None
            else:
                predicted = float(self._probabilities(members[:, 0], members[:, 1]).mean())
                observed = float(np.mean(members[:, 2] > 0.0))
            n = int(members.shape[0])
            table.append(
                CalibrationBin(lo=float(lo), hi=min(float(hi), 1.0), n=n, predicted=predicted, observed=observed)
            )
        return table

    # -- internals --------------------------------------------------------------------------------
    def _prior_lift(self, scores: np.ndarray) -> np.ndarray:
        return self._params.prior_slope * (scores - 0.5)

    def _lifts(self, scores: np.ndarray) -> np.ndarray:
        return self._prior_lift(scores) + np.interp(scores, self._centres, self._deviation)

    def _probabilities(self, scores: np.ndarray, rr: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + rr) + self._lifts(scores)
        return np.clip(p, self._params.probability_min, self._params.probability_max)

    def _bin_of(self, scores: np.ndarray) -> np.ndarray:
        """Bin index of each score (lo <= score < hi); scores outside the range fall into the outer bins."""
        return np.clip(np.searchsorted(self._hi, scores, side="right"), 0, self._centres.size - 1)
