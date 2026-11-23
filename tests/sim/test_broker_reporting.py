"""Reports (summary, detail, trades, orders, equity curves), snapshots, reset, restart, threads and events."""

import threading
from collections.abc import Callable
from datetime import timedelta

import pytest

from trading_local.clock import ManualClock
from trading_local.contracts import Broker, OrderRejected
from trading_local.db import repo_ops, repo_portfolio
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import OrderRequest
from trading_local.events import TOPIC_ACTIVITY, TOPIC_ORDER, TOPIC_PORTFOLIO
from trading_local.models import EquityPoint
from tests.sim.simkit import (
    FEE,
    MIN,
    MONDAY_0000,
    SLIP_EU,
    SLIP_US,
    T0,
    FakeMarketData,
    RecordingBus,
    add_bar,
    add_daily,
    add_flat_bars,
    check_accounting,
    open_position,
)

START = 100_000.0
DAY = 86_400


def test_factory_builds_a_broker(broker: Broker) -> None:
    assert isinstance(broker, Broker)


# -- summary, detail, trades, orders ----------------------------------------------------------------
def test_summary_of_a_fresh_portfolio(broker: Broker) -> None:
    summary = broker.summary("ai")
    assert (summary.key, summary.name, summary.base_currency) == ("ai", "KI", "EUR")
    assert summary.start_cash == summary.cash == summary.equity == summary.buying_power == START
    assert summary.closed_trades == 0 and summary.win_rate is None and summary.sharpe is None
    assert summary.max_drawdown_pct == 0.0
    assert (summary.reset_ts, summary.as_of_ts) == (T0, T0)
    assert broker.summary("me").name == "Ich"
    with pytest.raises(KeyError):
        broker.summary("xx")


def test_detail_marks_open_positions(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    open_position(broker, db, data, clock, "NVDA", qty=10, price=200.0)
    data.set_quote("NVDA", 210.0, clock.ts())
    detail = broker.detail("me")
    position = detail.positions[0]
    entry = 200.0 * (1 + SLIP_US)
    assert (position.last_price, position.last_ts) == (210.0, clock.ts())
    assert position.exposure_base == pytest.approx(10 * 210.0 * 0.8)
    assert position.unrealized_pnl == pytest.approx(10 * 0.8 * (210.0 - entry) - FEE)
    assert position.unrealized_pct == pytest.approx(position.unrealized_pnl / (10 * entry * 0.8) * 100)
    assert (position.name, position.currency, position.portfolio_key) == ("NVIDIA", "USD", "me")
    summary = detail.summary
    assert summary.long_exposure == pytest.approx(1680.0)
    assert summary.gross_leverage == pytest.approx(1680.0 / summary.equity)
    assert summary.buying_power == pytest.approx(summary.equity - 1680.0)
    check_accounting(broker, "me")


def test_mark_falls_back_to_bars_then_to_the_entry(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=10, price=100.0)
    data.drop_quote("SAP.DE")
    add_bar(db, "SAP.DE", clock.ts(), 102.0, close=103.0)
    assert broker.detail("me").positions[0].last_price == 100.0  # the new bar is still forming
    clock.set(clock.ts() + MIN)
    assert broker.detail("me").positions[0].last_price == 103.0
    db.execute("DELETE FROM bars_1m WHERE symbol = 'SAP.DE'")
    position = broker.detail("me").positions[0]
    assert position.last_price == pytest.approx(position.entry_price) and position.last_ts is None


def test_trade_statistics(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    results = []
    for symbol, exit_price in (("SAP.DE", 110.0), ("^GDAXI", 95.0)):
        position = open_position(broker, db, data, clock, symbol, qty=10, price=100.0)
        broker.close_position("me", position.id)
        add_bar(db, symbol, clock.ts(), exit_price)
        clock.set(clock.ts() + MIN)
        broker.process(clock.now())
        results.append(broker.trades("me")[0].realized_pnl)
    win, loss = results
    summary = broker.summary("me")
    assert (summary.closed_trades, summary.wins, summary.win_rate) == (2, 1, 0.5)
    assert summary.profit_factor == pytest.approx(win / -loss)
    assert (summary.avg_win, summary.avg_loss) == (pytest.approx(win), pytest.approx(loss))
    assert summary.realized_pnl == pytest.approx(win + loss)
    assert summary.total_return_pct == pytest.approx((win + loss) / START * 100)
    assert [t.symbol for t in broker.trades("me")] == ["^GDAXI", "SAP.DE"]  # newest first
    assert len(broker.trades("me", limit=1)) == 1
    check_accounting(broker, "me")


def test_orders_and_pending_orders_newest_first(broker: Broker, data: FakeMarketData, clock: ManualClock) -> None:
    for offset, symbol in enumerate(("SAP.DE", "NVDA", "BTC-USD")):
        data.set_quote(symbol, 100.0, T0)
        clock.set(T0 + offset)
        broker.place_order("me", OrderRequest(symbol=symbol, side="LONG", qty=1))
    assert [o.symbol for o in broker.orders("me")] == ["BTC-USD", "NVDA", "SAP.DE"]
    assert [o.symbol for o in broker.detail("me").pending_orders] == ["BTC-USD", "NVDA", "SAP.DE"]
    assert broker.summary("me").pending_orders == 3
    assert broker.orders("ai") == []


def test_day_pnl_against_the_snapshot_before_local_midnight(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=100, price=100.0)
    data.set_quote("SAP.DE", 101.0, clock.ts())
    clock.set(T0 + 7 * 3600 + 55 * MIN)  # 22:55 UTC = 23:55 in Berlin
    broker.process(clock.now())
    equity_before_midnight = broker.summary("me").equity
    data.set_quote("SAP.DE", 102.0, clock.ts())
    clock.set(T0 + 19 * 3600)  # Saturday 10:00 UTC
    summary = broker.summary("me")
    assert summary.day_pnl == pytest.approx(summary.equity - equity_before_midnight)
    assert summary.day_pnl == pytest.approx(100.0)
    assert summary.day_return_pct == pytest.approx(100.0 / equity_before_midnight * 100)


def test_day_pnl_falls_back_to_the_start_capital(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=100, price=100.0)
    data.set_quote("SAP.DE", 101.0, clock.ts())
    summary = broker.summary("me")
    assert summary.day_pnl == pytest.approx(summary.equity - START)


def test_drawdown_from_the_equity_snapshots(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=500, price=100.0)
    peak_ts = T0 + 5 * MIN
    data.set_quote("SAP.DE", 110.0, peak_ts)
    clock.set(peak_ts)
    broker.process(clock.now())
    peak = broker.summary("me").equity
    data.set_quote("SAP.DE", 90.0, peak_ts + 5 * MIN)
    clock.set(peak_ts + 5 * MIN)
    trough = broker.summary("me").equity
    assert broker.summary("me").max_drawdown_pct == pytest.approx((peak - trough) / peak * 100)


def save_snapshots(db: Database, key: str, points: list[tuple[int, float]]) -> None:
    pid = repo_portfolio.get_portfolio(db, key).id  # type: ignore[union-attr]
    with db.write():
        for ts, equity in points:
            repo_portfolio.save_equity_point(db, EquityPoint(pid, ts, equity, equity))  # type: ignore[arg-type]


def test_drawdown_sees_a_short_dip_in_a_long_history(broker: Broker, db: Database, clock: ManualClock) -> None:
    count = 2500  # more snapshots than the thinned series used for the Sharpe ratio keeps
    save_snapshots(db, "ai", [(T0 + 300 * i, START) for i in range(1, count + 1)])
    pid = repo_portfolio.get_portfolio(db, "ai").id  # type: ignore[union-attr]
    thinned_points = repo_portfolio.equity_series(db, pid, since_ts=T0, max_points=2000)  # type: ignore[arg-type]
    thinned = {p.ts for p in thinned_points}
    dip_ts = next(T0 + 300 * i for i in range(count // 2, count) if T0 + 300 * i not in thinned)
    save_snapshots(db, "ai", [(dip_ts, 92_000.0)])  # one 5-minute trough that a thinned series skips
    clock.set(T0 + 300 * (count + 1))
    assert broker.summary("ai").max_drawdown_pct == pytest.approx(8.0)


def test_drawdown_keeps_up_with_new_and_replaced_snapshots(
    broker: Broker, make_broker: Callable[[], Broker], db: Database, clock: ManualClock
) -> None:
    save_snapshots(db, "ai", [(T0 + 300 * i, START) for i in range(1, 6)])
    clock.set(T0 + 1800)
    assert broker.summary("ai").max_drawdown_pct == 0.0
    save_snapshots(db, "ai", [(T0 + 1800, 95_000.0), (T0 + 2100, START)])
    clock.set(T0 + 2400)
    assert broker.summary("ai").max_drawdown_pct == pytest.approx(5.0)
    save_snapshots(db, "ai", [(T0 + 2100, 90_000.0)])  # the newest snapshot is replaced at the same time
    assert broker.summary("ai").max_drawdown_pct == pytest.approx(10.0)
    assert make_broker().summary("ai").max_drawdown_pct == pytest.approx(10.0)  # a restart reads everything
    save_snapshots(db, "ai", [(T0 + 2400, START)])
    clock.set(T0 + 2700)
    assert broker.summary("ai").max_drawdown_pct == pytest.approx(10.0)
    broker.reset()
    assert broker.summary("ai").max_drawdown_pct == 0.0


# -- equity snapshots -------------------------------------------------------------------------------
def snapshot_times(db: Database, key: str) -> list[int]:
    pid = repo_portfolio.get_portfolio(db, key).id  # type: ignore[union-attr]
    return [p.ts for p in repo_portfolio.equity_series(db, pid, max_points=1000)]


def test_snapshots_every_five_minutes_and_after_fills(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    for ts in (T0 + 10, T0 + 200, T0 + 299):
        clock.set(ts)
        broker.process(clock.now())
    assert snapshot_times(db, "ai") == [T0] and snapshot_times(db, "me") == [T0]
    clock.set(T0 + 301)
    broker.process(clock.now())
    assert snapshot_times(db, "me") == [T0, T0 + 300]
    data.set_quote("SAP.DE", 100.0, clock.ts())
    broker.place_order("me", OrderRequest(symbol="SAP.DE", side="LONG", qty=10))
    add_bar(db, "SAP.DE", T0 + 360, 100.0)
    clock.set(T0 + 425)
    broker.process(clock.now())  # the fill writes a snapshot at once
    assert snapshot_times(db, "me") == [T0, T0 + 300, T0 + 425]
    assert snapshot_times(db, "ai") == [T0, T0 + 300]
    pid = repo_portfolio.get_portfolio(db, "me").id  # type: ignore[union-attr]
    last = repo_portfolio.last_equity_point(db, pid)
    assert last is not None and last.open_positions == 1
    assert last.long_exposure == pytest.approx(1000.0)


# -- equity curves ----------------------------------------------------------------------------------
@pytest.fixture
def benchmark_history(db: Database, data: FakeMarketData) -> int:
    """Prices of both benchmarks and of EUR/USD from the reset (Friday 15:00) to Tuesday 15:00 (returned)."""
    friday_2059, monday_2059 = T0 + 6 * 3600 - MIN, MONDAY_0000 + 21 * 3600 - MIN
    for ts, rate in ((T0 - MIN, 1.25), (friday_2059, 1.28), (monday_2059, 1.30)):
        add_bar(db, "EURUSD=X", ts, rate, close=rate)
    add_bar(db, "^GSPC", T0 - MIN, 5000.0)
    add_daily(db, "^GSPC", "2026-11-19", 4900.0)  # before the reset: not part of the curve
    add_daily(db, "^GSPC", "2026-11-20", 5100.0)
    add_daily(db, "^GSPC", "2026-11-23", 5200.0)
    add_bar(db, "^GDAXI", T0 - MIN, 20_000.0)
    add_daily(db, "^GDAXI", "2026-11-20", 20_200.0)
    add_daily(db, "^GDAXI", "2026-11-23", 20_400.0)
    tuesday = T0 + 4 * DAY
    data.set_quote("^GSPC", 5300.0, tuesday - MIN)
    data.set_quote("^GDAXI", 20_500.0, tuesday - MIN)
    data.set_quote("EURUSD=X", 4 / 3, tuesday - MIN)
    return tuesday


def test_equity_curves_with_benchmarks_in_base_currency(
    broker: Broker, clock: ManualClock, runtime_settings: SettingsStore, benchmark_history: int
) -> None:
    runtime_settings.update({"benchmarks": ["^GDAXI", "^GSPC", "NOPE"]}, T0)
    broker.process(clock.now())
    tuesday = benchmark_history
    clock.set(tuesday)
    broker.process(clock.now())
    curves = broker.equity_curves()
    assert (curves.since_ts, curves.base_currency, curves.start_cash) == (T0, "EUR", START)
    assert curves.ai == [(T0, START), (tuesday, START)] and curves.me == curves.ai
    assert set(curves.benchmarks) == {"^GDAXI", "^GSPC"}
    friday_close_xetr, friday_close_nyse = T0 + 90 * MIN, T0 + 6 * 3600
    monday_close_xetr, monday_close_nyse = MONDAY_0000 + 16 * 3600 + 30 * MIN, MONDAY_0000 + 21 * 3600
    dax = curves.benchmarks["^GDAXI"]
    assert [ts for ts, _ in dax] == [T0, friday_close_xetr, monday_close_xetr, tuesday - MIN]
    assert [v for _, v in dax] == pytest.approx([START, 101_000.0, 102_000.0, 102_500.0])
    # 100,000 EUR bought 25 S&P units at 5,000 USD and 1 EUR = 1.25 USD
    spx = curves.benchmarks["^GSPC"]
    assert [ts for ts, _ in spx] == [T0, friday_close_nyse, monday_close_nyse, tuesday - MIN]
    assert [v for _, v in spx] == pytest.approx([START, 5100 / 1.28 * 25, 5200 / 1.30 * 25, 5300 * 0.75 * 25])


def test_equity_curves_since_and_thinning(
    broker: Broker, clock: ManualClock, runtime_settings: SettingsStore, benchmark_history: int
) -> None:
    runtime_settings.update({"benchmarks": ["^GSPC"]}, T0)
    for minutes in range(0, 60, 5):
        clock.set(T0 + minutes * MIN)
        broker.process(clock.now())
    clock.set(benchmark_history)
    since = MONDAY_0000
    curves = broker.equity_curves(since_ts=since)
    assert curves.since_ts == since and curves.ai == []
    assert [ts for ts, _ in curves.benchmarks["^GSPC"]] == [MONDAY_0000 + 21 * 3600, benchmark_history - MIN]
    thinned = broker.equity_curves(max_points=4)
    assert len(thinned.ai) == 4 and thinned.ai[0][0] == T0 and thinned.ai[-1][0] == T0 + 55 * MIN
    assert broker.equity_curves(since_ts=T0 - DAY).since_ts == T0  # never before the reset


def test_benchmark_without_history_is_left_out(broker: Broker) -> None:
    assert broker.equity_curves().benchmarks == {}


# -- reset ------------------------------------------------------------------------------------------
def test_reset_restarts_both_portfolios(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, bus: RecordingBus
) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=10, price=100.0)
    data.set_quote("NVDA", 200.0, clock.ts())
    broker.place_order("ai", OrderRequest(symbol="NVDA", side="SHORT", qty=1))
    clock.set(T0 + DAY)
    bus.events.clear()
    broker.reset(50_000.0)
    for key in ("ai", "me"):
        summary = broker.summary(key)
        assert summary.start_cash == summary.cash == summary.equity == 50_000.0
        assert (summary.open_positions, summary.pending_orders, summary.closed_trades, summary.fees_paid) == (0, 0, 0, 0.0)
        assert summary.reset_ts == T0 + DAY
        assert broker.orders(key) == [] and broker.trades(key) == []
    curves = broker.equity_curves()
    assert curves.since_ts == T0 + DAY and curves.ai == curves.me == [(T0 + DAY, 50_000.0)]
    assert bus.of(TOPIC_PORTFOLIO) == [{"key": "ai"}, {"key": "me"}]
    assert "Beide Depots starten mit 50.000,00 EUR" in repo_ops.list_activity(db, category="sim")[0].message


def test_reset_defaults_to_the_configured_capital(broker: Broker) -> None:
    broker.reset()
    assert broker.summary("me").start_cash == START
    with pytest.raises(OrderRejected, match="Startkapital"):
        broker.reset(0.0)


# -- restart, threads, events -----------------------------------------------------------------------
def test_a_new_broker_continues_seamlessly(
    make_broker: Callable[[], Broker], db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    first = make_broker()
    data.set_quote("SAP.DE", 100.0, T0)
    first.place_order("me", OrderRequest(symbol="SAP.DE", side="LONG", qty=10, stop_price=98.0))
    add_bar(db, "SAP.DE", T0, 100.0)
    clock.set(T0 + MIN)
    first.process(clock.now())
    before = first.summary("me")

    second = make_broker()  # restart: same database, new object
    assert second.summary("me") == before
    add_flat_bars(db, "SAP.DE", T0 + MIN, 3, 99.0)
    add_bar(db, "SAP.DE", T0 + 4 * MIN, 99.0, low=97.5)
    clock.set(T0 + 10 * MIN)
    closed = second.process(clock.now())
    assert len(closed) == 1 and closed[0].exit_reason == "stop"
    assert closed[0].exit_price == pytest.approx(98.0 * (1 - SLIP_EU))
    assert first.summary("me") == second.summary("me")
    check_accounting(second, "me")


def test_orders_placed_while_process_runs(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    symbols = ["SAP.DE", "NVDA", "^GDAXI", "BTC-USD", "SHEL.L"]
    for symbol in symbols:
        data.set_quote(symbol, 100.0, T0)
        add_flat_bars(db, symbol, T0, 40, 100.0)
    data.set_quote("7203.T", 3000.0, T0)
    errors: list[BaseException] = []

    def guarded(work: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                work()
            except BaseException as exc:  # noqa: BLE001 - reported by the test
                errors.append(exc)

        return run

    def cycle() -> None:
        for minute in range(1, 31):
            broker.process(clock.now() + timedelta(minutes=minute))

    def trade() -> None:
        for symbol in symbols:
            broker.place_order("me", OrderRequest(symbol=symbol, side="LONG", amount_base=5000.0))
            broker.summary("me")
        tokyo = broker.place_order("me", OrderRequest(symbol="7203.T", side="LONG", qty=10))  # Tokyo is closed
        broker.cancel_order("me", tokyo.id)

    threads = [threading.Thread(target=guarded(cycle)), threading.Thread(target=guarded(trade))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert errors == [] and not any(thread.is_alive() for thread in threads)
    clock.set(T0 + 35 * MIN)
    broker.process(clock.now())
    statuses = sorted((o.symbol, o.status) for o in broker.orders("me"))
    assert statuses == [
        ("7203.T", "cancelled"),
        ("BTC-USD", "filled"),
        ("NVDA", "filled"),
        ("SAP.DE", "filled"),
        ("SHEL.L", "filled"),
        ("^GDAXI", "filled"),
    ]
    check_accounting(broker, "me")


def test_events_of_a_fill(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, bus: RecordingBus) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=10, price=100.0)
    assert [o["status"] for o in bus.of(TOPIC_ORDER)] == ["pending", "filled"]
    assert {"key": "me"} in bus.of(TOPIC_PORTFOLIO)
    fill_entry = bus.of(TOPIC_ACTIVITY)[0]
    assert fill_entry["category"] == "sim" and fill_entry["level"] == "info"
    assert fill_entry["message"].startswith("Depot Ich: Kauf 10 SAP zu 100,04 ausgeführt (Wert 1.000,40 EUR, Gebühr 1,00 EUR)")
    position = broker.detail("me").positions[0]
    bus.events.clear()
    broker.close_position("me", position.id)
    add_bar(db, "SAP.DE", clock.ts(), 101.0)
    clock.set(clock.ts() + MIN)
    broker.process(clock.now())
    assert [o["status"] for o in bus.of(TOPIC_ORDER)] == ["pending", "filled"]
    close_entry = bus.of(TOPIC_ACTIVITY)[0]
    assert close_entry["message"].startswith("Depot Ich: SAP Long geschlossen (manuell) zu 100,96 – Ergebnis 7,20 EUR")
