"""Outcome statistics of closed signals (pure functions; results are in R multiples after costs)."""

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from trading_local.dto import GroupStat, StatBlock
from trading_local.models import Signal

_MIN_BOOTSTRAP_N = 5
_BOOTSTRAP_CHUNK_CELLS = 2_000_000  # bounds the memory of one block of resamples


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion; (0, 1) when there are no trials."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(
    values: Sequence[float], *, n_boot: int = 2000, seed: int = 12345, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean (deterministic for a given seed).

    Resamples are drawn in blocks whose size depends only on the number of values, so the result is
    reproducible while memory stays bounded. Raises ``ValueError`` for an empty sequence.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("bootstrap needs at least one value")
    rng = np.random.default_rng(seed)
    block = max(1, _BOOTSTRAP_CHUNK_CELLS // data.size)
    means = np.empty(n_boot)
    for start in range(0, n_boot, block):
        stop = min(n_boot, start + block)
        means[start:stop] = data[rng.integers(0, data.size, size=(stop - start, data.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def stat_block(outcomes_r: Sequence[float]) -> StatBlock:
    """Statistics of a set of results in R (non-finite values are ignored).

    wins := r > 0; win rate with its 95 % Wilson interval; mean R with a 95 % bootstrap interval from
    5 results on; profit factor = gross wins / |gross losses| (None without losses).
    """
    data = np.asarray([value for value in outcomes_r if math.isfinite(value)], dtype=np.float64)
    n = int(data.size)
    if n == 0:
        return StatBlock()
    wins = int((data > 0.0).sum())
    lo, hi = wilson_interval(wins, n)
    avg_lo, avg_hi = bootstrap_mean_ci(data.tolist()) if n >= _MIN_BOOTSTRAP_N else (None, None)
    losses = float(-data[data < 0.0].sum())
    return StatBlock(
        n=n,
        wins=wins,
        win_rate=wins / n,
        win_rate_lo=lo,
        win_rate_hi=hi,
        avg_r=float(data.mean()),
        avg_r_lo=avg_lo,
        avg_r_hi=avg_hi,
        total_r=float(data.sum()),
        profit_factor=float(data[data > 0.0].sum()) / losses if losses > 0.0 else None,
    )


def group_stats(
    signals: Sequence[Signal], key: Callable[[Signal], str], labels: Mapping[str, str] | None = None
) -> list[GroupStat]:
    """One ``StatBlock`` per group of the signals with a realised result, sorted by n descending (ties by key).

    ``labels`` maps a group key to its German label; unknown keys are shown as they are.
    """
    groups: dict[str, list[float]] = {}
    for signal in signals:
        if signal.outcome_r is not None:
            groups.setdefault(key(signal), []).append(signal.outcome_r)
    stats = [
        GroupStat(key=name, label_de=(labels or {}).get(name, name), stats=stat_block(values))
        for name, values in groups.items()
    ]
    return sorted(stats, key=lambda group: (-group.stats.n, group.key))


def cumulative_r(signals: Sequence[Signal]) -> list[tuple[int, float]]:
    """(exit_ts, cumulative R) after each signal with a realised result, in exit order (ties by id).

    Non-finite results are ignored, like in ``stat_block``.
    """
    closed = sorted(
        (int(signal.exit_ts), signal.id if signal.id is not None else -1, float(signal.outcome_r))
        for signal in signals
        if signal.outcome_r is not None and math.isfinite(signal.outcome_r) and signal.exit_ts is not None
    )
    totals = np.cumsum([outcome for _, _, outcome in closed]).tolist()
    return [(exit_ts, total) for (exit_ts, _, _), total in zip(closed, totals)]
