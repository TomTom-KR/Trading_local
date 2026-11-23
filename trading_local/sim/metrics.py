"""Portfolio performance metrics: pure functions without I/O, clock or state.

Conventions
* equity values and P&L are in the portfolio base currency
* percentages are percent (12.5 means 12.5 %), win rates are fractions 0..1
* equity points are ``(ts, equity)`` pairs with ``ts`` in epoch seconds UTC, ascending
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_local.models import Position

TRADING_DAYS_PER_YEAR = 252
MIN_DAILY_RETURNS = 10  # fewer daily returns make a Sharpe ratio meaningless
_ZERO_VARIANCE = 1e-12


@dataclass(frozen=True, slots=True)
class TradeStats:
    closed_trades: int = 0
    wins: int = 0
    win_rate: float | None = None  # fraction 0..1 of trades with a positive result
    profit_factor: float | None = None  # gross profit / gross loss; None without a losing trade
    avg_win: float | None = None  # mean result of the winning trades (positive)
    avg_loss: float | None = None  # mean result of the losing trades (negative)


@dataclass(frozen=True, slots=True)
class Drawdown:
    """Running state of ``max_drawdown_pct`` over the part of an equity series seen so far."""

    peak: float | None = None  # highest finite equity value so far; None before the first one
    max_drawdown_pct: float | None = None  # largest decline so far; None while no positive peak was seen


def extend_drawdown(state: Drawdown, values: Iterable[float]) -> Drawdown:
    """``state`` continued over further equity values (oldest first).

    Folding a series piece by piece gives exactly ``max_drawdown_pct`` of the whole series, so a long history
    can be summarised once and extended with the newest values only.
    """
    equity = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if equity.size == 0:
        return state
    peak = np.maximum.accumulate(equity)
    if state.peak is not None:
        peak = np.maximum(peak, state.peak)
    worst = state.max_drawdown_pct
    valid = peak > 0
    if valid.any():
        current = float(((peak[valid] - equity[valid]) / peak[valid]).max() * 100.0)
        worst = current if worst is None else max(worst, current)
    return Drawdown(peak=float(peak[-1]), max_drawdown_pct=worst)


def max_drawdown_pct(values: Iterable[float]) -> float | None:
    """Largest peak-to-trough decline of an equity series as a POSITIVE percentage of the peak.

    0.0 for a series that never falls below an earlier high; None for an empty series. Non-finite values
    are ignored, as are stretches where the running peak is not positive (a percentage is undefined there).
    """
    return extend_drawdown(Drawdown(), values).max_drawdown_pct


def sharpe_daily(points: Sequence[tuple[int, float]], tz: str) -> float | None:
    """Annualised Sharpe ratio (risk-free rate 0) of the daily returns of an equity series.

    The last equity value of every local calendar day in ``tz`` is the day's close. Only Monday to Friday
    count: the broker writes snapshots on every calendar day, and annualising with sqrt(252) TRADING days
    requires one return per trading day - Saturday and Sunday closes would add zero-return days and shrink
    the ratio by about sqrt(5/7). Weekend results (crypto) therefore fall into Monday's return. Returns are
    taken between consecutive weekdays that have data; the ratio is mean / sample standard deviation *
    sqrt(252). None with fewer than 10 daily returns, with (practically) zero variance, or when a day's close
    is not positive.
    """
    if len(points) < MIN_DAILY_RETURNS + 1:
        return None
    ts = np.fromiter((point[0] for point in points), dtype=np.int64, count=len(points))
    equity = np.fromiter((point[1] for point in points), dtype=np.float64, count=len(points))
    local_days = pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).normalize()
    day_closes = pd.Series(equity, index=local_days).groupby(level=0, sort=True).last()
    closes = day_closes[day_closes.index.day_of_week < 5].to_numpy()
    if closes.size < MIN_DAILY_RETURNS + 1 or not (np.isfinite(closes).all() and (closes > 0).all()):
        return None
    returns = closes[1:] / closes[:-1] - 1.0
    std = float(returns.std(ddof=1))
    if not math.isfinite(std) or std < _ZERO_VARIANCE:
        return None
    return float(returns.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR))


def trade_stats(positions: Iterable[Position]) -> TradeStats:
    """Statistics of closed positions by their ``realized_pnl`` (after fees).

    Positions without a realised result are ignored. A result of exactly zero counts as a trade but
    neither as a win nor as a loss.
    """
    pnls = np.asarray([p.realized_pnl for p in positions if p.realized_pnl is not None], dtype=np.float64)
    if pnls.size == 0:
        return TradeStats()
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_loss = float(-losses.sum())
    return TradeStats(
        closed_trades=int(pnls.size),
        wins=int(wins.size),
        win_rate=wins.size / pnls.size,
        profit_factor=float(wins.sum()) / gross_loss if gross_loss > 0 else None,
        avg_win=float(wins.mean()) if wins.size else None,
        avg_loss=float(losses.mean()) if losses.size else None,
    )


def local_midnight_ts(now_ts: int, tz: str) -> int:
    """Epoch seconds of the most recent local midnight (start of today) in time zone ``tz``."""
    zone = ZoneInfo(tz)
    today = datetime.fromtimestamp(now_ts, tz=zone).date()
    return int(datetime.combine(today, time(0), tzinfo=zone).timestamp())


def day_pnl(
    equity_now: float, series: Sequence[tuple[int, float]], tz: str, start_cash: float, *, now_ts: int
) -> tuple[float, float]:
    """Today's result ``(pnl, pct)`` versus the last equity point before today's local midnight in ``tz``.

    Without such a point (new portfolio, reset today) the reference is ``start_cash``. ``pct`` is percent
    of the reference and 0.0 when the reference is not positive.
    """
    midnight = local_midnight_ts(now_ts, tz)
    reference = start_cash
    for ts, equity in series:
        if ts >= midnight:
            break
        reference = equity
    pnl = equity_now - reference
    pct = pnl / reference * 100.0 if reference > 0 else 0.0
    return pnl, pct
