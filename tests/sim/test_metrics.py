import math
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from trading_local.models import Position
from trading_local.sim import metrics

BERLIN = "Europe/Berlin"


def utc(*args: int) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp())


def closed(pnl: float | None) -> Position:
    return Position(1, "SAP.DE", "LONG", 1.0, 100.0, 1.0, 0, status="closed", realized_pnl=pnl)


# -- drawdown ---------------------------------------------------------------------------------------
def test_max_drawdown_pct() -> None:
    # 120 -> 90 is -25 %; the later 130 -> 117 is only -10 %
    assert metrics.max_drawdown_pct([100, 120, 90, 130, 117]) == pytest.approx(25.0)


def test_max_drawdown_edge_cases() -> None:
    assert metrics.max_drawdown_pct([]) is None
    assert metrics.max_drawdown_pct([100, 101, 105]) == 0.0
    assert metrics.max_drawdown_pct([100, math.nan, 80]) == pytest.approx(20.0)
    assert metrics.max_drawdown_pct([-5.0, -3.0]) is None  # no positive peak: a percentage is undefined


def test_drawdown_folded_in_pieces_equals_the_whole_series() -> None:
    series = [100.0, 120.0, math.nan, 90.0, 130.0, 117.0, 140.0, 126.0]
    whole = metrics.max_drawdown_pct(series)
    for cut in range(len(series) + 1):
        state = metrics.extend_drawdown(metrics.Drawdown(), series[:cut])
        assert metrics.extend_drawdown(state, series[cut:]).max_drawdown_pct == pytest.approx(whole)
    assert metrics.extend_drawdown(metrics.Drawdown(), []) == metrics.Drawdown()


# -- Sharpe -----------------------------------------------------------------------------------------
def weekdays(count: int) -> list[tuple[int, int, int]]:
    """``count`` consecutive Monday-to-Friday dates from Thursday 2026-10-01 on, as (year, month, day)."""
    days: list[tuple[int, int, int]] = []
    day = datetime(2026, 10, 1, tzinfo=UTC)
    while len(days) < count:
        if day.weekday() < 5:
            days.append((day.year, day.month, day.day))
        day += timedelta(days=1)
    return days


def daily_points(closes: list[float]) -> list[tuple[int, float]]:
    """One close per Berlin weekday at 15:00 UTC, preceded by an intraday point that must not count."""
    points = []
    for (year, month, day), close in zip(weekdays(len(closes)), closes, strict=True):
        points.append((utc(year, month, day, 9, 0), close * 0.5))
        points.append((utc(year, month, day, 15, 0), close))
    return points


def test_sharpe_daily_matches_hand_computation() -> None:
    closes = [100, 101, 100.5, 102, 103, 102.5, 104, 103.5, 105, 106, 105.5, 107]
    returns = [b / a - 1 for a, b in zip(closes, closes[1:], strict=False)]
    expected = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
    assert metrics.sharpe_daily(daily_points(closes), BERLIN) == pytest.approx(expected)


def test_sharpe_daily_uses_the_local_day() -> None:
    closes = [100, 101, 100.5, 102, 103, 102.5, 104, 103.5, 105, 106, 105.5]
    points = daily_points(closes)
    year, month, day = weekdays(len(closes))[-1]  # Thursday 2026-10-15
    # 23:30 UTC is already Friday in Berlin: this point is the close of Friday, not of Thursday
    points.append((utc(year, month, day, 23, 30), 108.0))
    extended = [*closes, 108.0]
    returns = [b / a - 1 for a, b in zip(extended, extended[1:], strict=False)]
    expected = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
    assert metrics.sharpe_daily(points, BERLIN) == pytest.approx(expected)


def test_sharpe_daily_ignores_weekend_snapshots() -> None:
    # The broker snapshots every calendar day. Ten weeks of the same Monday-to-Friday returns with snapshots on
    # Saturday and Sunday too: the weekend days must not enter as zero returns annualised with sqrt(252).
    pattern = [0.010, -0.004, 0.006, -0.008, 0.005]
    start = datetime(2026, 9, 7, tzinfo=UTC)  # a Monday
    equity, points, returns = 10_000.0, [], []
    for offset in range(70):
        day = start + timedelta(days=offset)
        weekday = day.weekday()
        if weekday < 5:
            equity *= 1 + pattern[weekday]
            returns.append(pattern[weekday])
        for hour in (9, 15):  # also on Saturday and Sunday
            points.append((int((day + timedelta(hours=hour)).timestamp()), equity))
    trading = returns[1:]  # the first trading day has no previous close
    expected = statistics.mean(trading) / statistics.stdev(trading) * math.sqrt(252)
    assert metrics.sharpe_daily(points, BERLIN) == pytest.approx(expected)


def test_weekend_results_fall_into_mondays_return() -> None:
    closes = [100.0 + i for i in range(11)]
    points = daily_points(closes)
    saturday = utc(2026, 10, 17, 15, 0)  # after the last weekday close (Thursday 2026-10-15) and Friday
    points += [(utc(2026, 10, 16, 15, 0), 111.0), (saturday, 120.0), (utc(2026, 10, 19, 15, 0), 120.0)]
    extended = [*closes, 111.0, 120.0]  # Monday's close carries the Saturday move
    returns = [b / a - 1 for a, b in zip(extended, extended[1:], strict=False)]
    expected = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(252)
    assert metrics.sharpe_daily(points, BERLIN) == pytest.approx(expected)


def test_sharpe_daily_needs_ten_returns_and_variance() -> None:
    assert metrics.sharpe_daily(daily_points([100 + i for i in range(10)]), BERLIN) is None  # 9 returns
    assert metrics.sharpe_daily(daily_points([100.0] * 15), BERLIN) is None  # zero variance
    assert metrics.sharpe_daily([], BERLIN) is None


# -- trades -----------------------------------------------------------------------------------------
def test_trade_stats() -> None:
    stats = metrics.trade_stats([closed(100), closed(-50), closed(30), closed(-20), closed(0), closed(None)])
    assert stats.closed_trades == 5
    assert stats.wins == 2
    assert stats.win_rate == pytest.approx(0.4)
    assert stats.profit_factor == pytest.approx(130 / 70)
    assert stats.avg_win == pytest.approx(65.0)
    assert stats.avg_loss == pytest.approx(-35.0)


def test_trade_stats_without_trades_or_losses() -> None:
    assert metrics.trade_stats([]) == metrics.TradeStats()
    only_wins = metrics.trade_stats([closed(10), closed(20)])
    assert only_wins.profit_factor is None
    assert only_wins.avg_loss is None
    assert only_wins.win_rate == 1.0


# -- day P&L ----------------------------------------------------------------------------------------
def test_local_midnight_ts() -> None:
    # 16:00 in Berlin (CET) -> midnight was 23:00 UTC the day before
    assert metrics.local_midnight_ts(utc(2026, 11, 20, 15, 0), BERLIN) == utc(2026, 11, 19, 23, 0)
    # summer time (CEST): midnight is 22:00 UTC
    assert metrics.local_midnight_ts(utc(2026, 7, 1, 12, 0), BERLIN) == utc(2026, 6, 30, 22, 0)
    # the day the clocks change keeps its midnight in winter time
    assert metrics.local_midnight_ts(utc(2026, 3, 29, 12, 0), BERLIN) == utc(2026, 3, 28, 23, 0)


def test_day_pnl_against_the_last_point_before_local_midnight() -> None:
    series = [(utc(2026, 11, 19, 22, 55), 101_000.0), (utc(2026, 11, 19, 23, 5), 102_000.0)]
    pnl, pct = metrics.day_pnl(103_000.0, series, BERLIN, 100_000.0, now_ts=utc(2026, 11, 20, 15, 0))
    assert pnl == pytest.approx(2000.0)
    assert pct == pytest.approx(2000.0 / 101_000.0 * 100)


def test_day_pnl_falls_back_to_the_start_cash() -> None:
    series = [(utc(2026, 11, 20, 9, 0), 99_000.0)]
    pnl, pct = metrics.day_pnl(100_500.0, series, BERLIN, 100_000.0, now_ts=utc(2026, 11, 20, 15, 0))
    assert (pnl, pct) == pytest.approx((500.0, 0.5))
