"""Edge cases of the paper broker: failure injection, idempotence, SHORT exits, 24/7 and lunch-break venues,
mixed currencies, benchmark close times and concurrency."""

import threading
from collections.abc import Callable
from datetime import timedelta

import pytest

from trading_local.clock import ManualClock
from trading_local.contracts import Broker
from trading_local.db import repo_ops, repo_portfolio
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import OrderRequest
from trading_local.events import TOPIC_ORDER, TOPIC_PORTFOLIO
from trading_local.sim.broker import PRIOR_LEVELS_KEY
from trading_local.testing import FakeCalendar
from tests.sim.simkit import (
    FEE,
    MIN,
    MONDAY_0000,
    SETTLE_S,
    SLIP_ASIA,
    SLIP_CRYPTO,
    SLIP_EU,
    T0,
    FakeMarketData,
    RecordingBus,
    add_bar,
    add_daily,
    add_flat_bars,
    check_accounting,
    make_signal,
    open_position,
)

START = 100_000.0
DAY = 86_400


def run(broker: Broker, clock: ManualClock, ts: int) -> list:
    clock.set(ts)
    return broker.process(clock.now())


def snapshot_times(db: Database, key: str) -> list[int]:
    pid = repo_portfolio.get_portfolio(db, key).id  # type: ignore[union-attr]
    return [p.ts for p in repo_portfolio.equity_series(db, pid, max_points=1000)]


# -- failure injection --------------------------------------------------------------------------------
def test_failed_fill_books_nothing_and_is_retried(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    order = broker.place_order("me", OrderRequest(symbol="SAP.DE", side="LONG", qty=10))
    add_bar(db, "SAP.DE", T0, 100.0)
    original = repo_portfolio.adjust_cash

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo_portfolio, "adjust_cash", broken)
    run(broker, clock, T0 + MIN)
    assert repo_portfolio.open_positions(db) == []  # the position insert was rolled back with the booking
    assert repo_portfolio.get_order(db, order.id).status == "pending"
    assert broker.summary("me").cash == START
    monkeypatch.setattr(repo_portfolio, "adjust_cash", original)
    run(broker, clock, T0 + 2 * MIN)
    assert repo_portfolio.get_order(db, order.id).status == "filled"
    assert len(repo_portfolio.open_positions(db)) == 1
    check_accounting(broker, "me")


def test_failed_exit_keeps_the_position_and_is_retried(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=10, stop=99.0)
    add_bar(db, "SAP.DE", T0 + MIN, 100.0, low=98.5)
    original = repo_portfolio.adjust_cash
    monkeypatch.setattr(repo_portfolio, "adjust_cash", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("locked")))
    assert run(broker, clock, T0 + 2 * MIN) == []
    stored = repo_portfolio.open_positions(db)
    assert len(stored) == 1 and stored[0].status == "open"
    monkeypatch.setattr(repo_portfolio, "adjust_cash", original)
    closed = run(broker, clock, T0 + 3 * MIN)
    assert len(closed) == 1 and closed[0].exit_reason == "stop" and closed[0].closed_ts == T0 + MIN
    assert len([o for o in broker.orders("me") if o.intent == "close"]) == 1  # no orphaned closing order
    check_accounting(broker, "me")


def test_one_broken_position_does_not_block_the_others(
    broker: Broker,
    db: Database,
    data: FakeMarketData,
    clock: ManualClock,
    calendar: FakeCalendar,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert broker.execute_signal(make_signal(db, intraday=True)) is not None  # SAP, flattened at the close
    add_bar(db, "SAP.DE", T0, 100.0)
    open_position(broker, db, data, clock, "NVDA", qty=5, price=200.0, stop=195.0)
    assert broker.summary("ai").open_positions == 1
    original = calendar.segments

    def segments(exchange: str, *args: object, **kwargs: object) -> list:
        if exchange == "XETR":
            raise KeyError(exchange)
        return original(exchange, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(calendar, "segments", segments)
    add_bar(db, "NVDA", clock.ts(), 196.0, low=194.0)
    closed = run(broker, clock, clock.ts() + MIN)
    assert [p.symbol for p in closed] == ["NVDA"]
    assert broker.summary("ai").open_positions == 1


# -- idempotence --------------------------------------------------------------------------------------
def test_repeated_passes_at_the_same_time_change_nothing(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, bus: RecordingBus
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    broker.place_order("me", OrderRequest(symbol="SAP.DE", side="LONG", qty=10))
    add_bar(db, "SAP.DE", T0, 100.0)
    run(broker, clock, T0 + MIN)
    run(broker, clock, T0 + MIN)  # examines the entry bar
    state = (broker.summary("me"), broker.orders("me"), snapshot_times(db, "me"))
    fills = len([e for e in bus.of(TOPIC_ORDER) if e["status"] == "filled"])
    run(broker, clock, T0 + MIN)
    assert (broker.summary("me"), broker.orders("me"), snapshot_times(db, "me")) == state
    assert len([e for e in bus.of(TOPIC_ORDER) if e["status"] == "filled"]) == fills == 1


# -- SHORT exits --------------------------------------------------------------------------------------
def short_with_levels(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, entry_bar: dict) -> None:
    """SHORT 10 SAP with stop 101 and target 97, filled on the bar T0 (described by ``entry_bar``)."""
    data.set_quote("SAP.DE", 100.0, T0)
    broker.place_order(
        "me", OrderRequest(symbol="SAP.DE", side="SHORT", qty=10, stop_price=101.0, target_price=97.0)
    )
    add_bar(db, "SAP.DE", T0, 100.0, **entry_bar)
    run(broker, clock, T0 + MIN)
    data.drop_quote("SAP.DE")


def test_short_stop_inside_the_entry_bar(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    short_with_levels(broker, db, data, clock, {"high": 101.2, "low": 99.8, "close": 100.5})
    run(broker, clock, T0 + 2 * MIN)
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("stop", T0)
    assert trade.exit_price == pytest.approx(101.0 * (1 + SLIP_EU))  # buying back costs slippage
    entry = 100.0 * (1 - SLIP_EU)
    assert trade.realized_pnl == pytest.approx(-10 * (trade.exit_price - entry) - 2 * FEE)
    check_accounting(broker, "me")


def test_short_gap_through_the_stop(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    short_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.8})
    add_bar(db, "SAP.DE", T0 + MIN, 102.0, high=102.5, low=101.5)
    run(broker, clock, T0 + 2 * MIN)
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("stop", T0 + MIN)
    assert trade.exit_price == pytest.approx(102.0 * (1 + SLIP_EU))


def test_short_target_is_a_limit_fill(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    short_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.8})
    add_bar(db, "SAP.DE", T0 + MIN, 98.0, high=98.2, low=96.8)
    run(broker, clock, T0 + 2 * MIN)
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.exit_price) == ("target", 97.0)
    check_accounting(broker, "me")


# -- level changes are never retroactive ------------------------------------------------------------
def delayed_position(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, **levels: float) -> int:
    """LONG 10 SAP (15-minute delayed feed) filled on the bar T0; bars from 15:01 on have not arrived yet."""
    data.set_quote("SAP.DE", 100.0, T0)
    broker.place_order("me", OrderRequest(symbol="SAP.DE", side="LONG", qty=10, **levels))
    add_bar(db, "SAP.DE", T0, 100.0)
    run(broker, clock, T0 + MIN)
    run(broker, clock, T0 + 2 * MIN)  # the entry bar is examined
    return broker.detail("me").positions[0].id


def test_a_new_target_does_not_fill_on_earlier_bars(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position_id = delayed_position(broker, db, data, clock)
    clock.set(T0 + 20 * MIN)
    out = broker.update_position("me", position_id, stop_price=None, target_price=104.5)
    assert out.target_price == 104.5
    add_flat_bars(db, "SAP.DE", T0 + MIN, 19, 100.0)
    add_bar(db, "SAP.DE", T0 + 12 * MIN, 100.0, high=105.0)  # the spike happened before the target was set
    assert run(broker, clock, T0 + 28 * MIN) == []
    add_bar(db, "SAP.DE", T0 + 20 * MIN, 100.0, high=104.6)  # first bar after the change
    closed = run(broker, clock, T0 + 36 * MIN)
    assert [(p.exit_reason, p.closed_ts, p.exit_price) for p in closed] == [("target", T0 + 20 * MIN, 104.5)]


def test_removing_a_stop_does_not_undo_an_earlier_hit(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position_id = delayed_position(broker, db, data, clock, stop_price=99.0)
    clock.set(T0 + 20 * MIN)
    broker.update_position("me", position_id, stop_price=None, target_price=None)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 10, 100.0)
    add_bar(db, "SAP.DE", T0 + 11 * MIN, 99.5, low=98.0)  # hit the stop that was in force at 15:11
    closed = run(broker, clock, T0 + 28 * MIN)
    assert [(p.exit_reason, p.closed_ts) for p in closed] == [("stop", T0 + 11 * MIN)]
    assert closed[0].exit_price == pytest.approx(99.0 * (1 - SLIP_EU))
    check_accounting(broker, "me")


def test_a_level_change_stays_non_retroactive_across_a_restart(
    make_broker: Callable[[], Broker], db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    # The bars before a change can arrive (and settle) minutes after it. A restart in between must not judge
    # them by the new levels: the spike to 105 happened before the target of 104.5 was set.
    broker = make_broker()
    position_id = delayed_position(broker, db, data, clock)
    clock.set(T0 + 20 * MIN)
    broker.update_position("me", position_id, stop_price=None, target_price=104.5)
    restarted = make_broker()
    add_flat_bars(db, "SAP.DE", T0 + MIN, 19, 100.0)
    add_bar(db, "SAP.DE", T0 + 12 * MIN, 100.0, high=105.0)
    assert run(restarted, clock, T0 + 28 * MIN) == []
    add_bar(db, "SAP.DE", T0 + 20 * MIN, 100.0, high=104.6)  # first bar after the change
    closed = run(make_broker(), clock, T0 + 36 * MIN)
    assert [(p.exit_reason, p.closed_ts, p.exit_price) for p in closed] == [("target", T0 + 20 * MIN, 104.5)]
    assert repo_ops.get_setting(db, PRIOR_LEVELS_KEY) == {}  # nothing left to remember


def test_level_history_is_forgotten_once_the_bars_before_the_change_are_checked(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position_id = delayed_position(broker, db, data, clock, stop_price=99.0)
    clock.set(T0 + 5 * MIN)
    broker.update_position("me", position_id, stop_price=98.0, target_price=None)
    clock.set(T0 + 6 * MIN)
    broker.update_position("me", position_id, stop_price=97.0, target_price=None)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 4, 100.0)
    add_bar(db, "SAP.DE", T0 + 5 * MIN, 99.0, low=98.5)  # between the changes the stop was 98: no exit
    add_bar(db, "SAP.DE", T0 + 6 * MIN, 99.0, low=97.5)  # from 15:06 on the stop is 97: no exit
    assert run(broker, clock, T0 + 22 * MIN) == []
    assert broker._prior_levels == {}  # type: ignore[attr-defined]
    add_bar(db, "SAP.DE", T0 + 7 * MIN, 98.0, low=96.9)
    closed = run(broker, clock, T0 + 23 * MIN)
    assert [(p.exit_reason, p.closed_ts) for p in closed] == [("stop", T0 + 7 * MIN)]
    assert closed[0].exit_price == pytest.approx(97.0 * (1 - SLIP_EU))


# -- overdue exits, flattening on special venues ------------------------------------------------------
def test_overdue_exit_replaces_a_closing_order_without_bars(broker: Broker, db: Database, clock: ManualClock) -> None:
    assert broker.execute_signal(make_signal(db, intraday=True)) is not None
    add_bar(db, "SAP.DE", T0, 100.0)
    run(broker, clock, T0 + MIN)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 70, 100.0)  # the feed stops at 16:10
    position = broker.detail("ai").positions[0]
    clock.set(T0 + 80 * MIN)
    close_order = broker.close_position("ai", position.id)  # no bar will ever arrive for it
    session_end = T0 + 90 * MIN
    assert run(broker, clock, session_end + 15 * MIN + SETTLE_S - 1) == []  # the last minutes may still arrive
    closed = run(broker, clock, session_end + 15 * MIN + SETTLE_S)
    assert [p.exit_reason for p in closed] == ["session_close"]
    out = next(o for o in broker.orders("ai") if o.id == close_order.id)
    assert (out.status, out.status_reason) == ("cancelled", "Position bereits geschlossen")
    check_accounting(broker, "ai")


def test_crypto_intraday_position_is_flattened_before_utc_midnight(
    broker: Broker, db: Database, clock: ManualClock
) -> None:
    signal = make_signal(db, "BTC-USD", ref_price=50_000.0, stop_dist=1000.0, target_dist=2000.0,
                         horizon_min=600, intraday=True)
    assert broker.execute_signal(signal) is not None
    add_flat_bars(db, "BTC-USD", T0, 9 * 60 + 1, 50_000.0)  # 15:00 .. 24:00 UTC
    run(broker, clock, T0 + MIN)
    closed = run(broker, clock, T0 + 9 * 60 * MIN)
    assert len(closed) == 1
    trade = broker.trades("ai")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("session_close", T0 + 8 * 3600 + 55 * MIN)  # 23:55 UTC
    assert trade.exit_price == pytest.approx(50_000.0 * (1 - SLIP_CRYPTO))
    check_accounting(broker, "ai")


def test_lunch_break_venue_is_flattened_before_the_last_segment_ends(
    broker: Broker, db: Database, clock: ManualClock
) -> None:
    created = MONDAY_0000 + 10 * MIN  # Tokyo 09:10
    clock.set(created)
    signal = make_signal(db, "7203.T", created_ts=created, ref_price=3000.0, stop_dist=30.0, target_dist=60.0,
                         horizon_min=600, intraday=True)
    assert broker.execute_signal(signal) is not None
    add_bar(db, "7203.T", created, 3000.0)
    run(broker, clock, created + MIN)
    add_flat_bars(db, "7203.T", created + MIN, 139, 3000.0)  # .. 02:29 UTC, then the lunch break
    run(broker, clock, MONDAY_0000 + 3 * 3600)  # inside the lunch break: nothing to do
    assert broker.summary("ai").open_positions == 1
    add_flat_bars(db, "7203.T", MONDAY_0000 + 3 * 3600 + 30 * MIN, 176, 3000.0)  # 03:30 .. 06:25 UTC
    closed = run(broker, clock, MONDAY_0000 + 6 * 3600 + 27 * MIN)
    assert len(closed) == 1
    trade = broker.trades("ai")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("session_close", MONDAY_0000 + 6 * 3600 + 25 * MIN)
    assert trade.exit_price == pytest.approx(3000.0 * (1 - SLIP_ASIA))


def test_closing_order_expires_after_a_week_without_bars(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, bus: RecordingBus
) -> None:
    position = open_position(broker, db, data, clock, "SAP.DE", qty=10)
    close_order = broker.close_position("me", position.id)
    run(broker, clock, clock.ts() + MIN)  # examines the entry bar
    bus.events.clear()
    assert close_order.expire_ts is not None
    run(broker, clock, close_order.expire_ts + 1)  # Xetra trades: its last minutes may still bring a bar
    assert next(o for o in broker.orders("me") if o.id == close_order.id).status == "pending"
    run(broker, clock, close_order.expire_ts + MIN + 15 * MIN + SETTLE_S)
    out = next(o for o in broker.orders("me") if o.id == close_order.id)
    assert (out.status, out.status_reason) == ("cancelled", "abgelaufen")
    assert broker.summary("me").open_positions == 1
    assert [e["status"] for e in bus.of(TOPIC_ORDER)] == ["cancelled"]
    assert {"key": "me"} in bus.of(TOPIC_PORTFOLIO)


# -- mixed currencies ---------------------------------------------------------------------------------
def test_accounting_identities_across_currencies_and_fx_moves(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    trades = [("SAP.DE", "LONG", 100.0, 20), ("NVDA", "SHORT", 200.0, 10), ("SHEL.L", "LONG", 2500.0, 40),
              ("7203.T", "LONG", 3000.0, 30), ("BTC-USD", "SHORT", 50_000.0, 0.05)]
    for symbol, side, price, qty in trades:
        open_position(broker, db, data, clock, symbol, side, qty=qty, price=price)  # type: ignore[arg-type]
        check_accounting(broker, "me")
    for symbol, rate in (("EURUSD=X", 1.10), ("EURGBP=X", 0.85), ("EURJPY=X", 170.0)):
        data.set_quote(symbol, rate, clock.ts())
        check_accounting(broker, "me")
    for symbol, _, price, _ in trades:
        data.set_quote(symbol, price * 1.03, clock.ts())
    check_accounting(broker, "me")
    data.drop_quote("EURJPY=X")  # no rate at all: the position is valued with its entry rate
    check_accounting(broker, "me")
    for position in broker.detail("me").positions[:3]:
        broker.close_position("me", position.id)
        # the minutes since the entry (all three venues trade): none is missing, so none can still arrive
        add_flat_bars(db, position.symbol, position.opened_ts + MIN, (clock.ts() - position.opened_ts) // MIN - 1,
                      position.entry_price, spread=0.0)
        add_bar(db, position.symbol, clock.ts(), (position.last_price or 0.0) * 0.99)
    run(broker, clock, clock.ts() + MIN)
    summary = broker.summary("me")
    assert (summary.open_positions, summary.closed_trades) == (2, 3)
    assert summary.fees_paid == pytest.approx(8 * FEE)
    check_accounting(broker, "me")


def test_crypto_signal_is_sized_to_the_crypto_lot(broker: Broker, db: Database) -> None:
    # risk 500 EUR / (3,000 USD * 0.8) = 0.2083333 BTC; the 10 % cap allows 0.2892856 BTC
    out = broker.execute_signal(make_signal(db, "BTC-USD", ref_price=43_210.0, stop_dist=3000.0))
    assert out is not None and out.qty == 0.208333


# -- snapshots and benchmarks -------------------------------------------------------------------------
def test_first_snapshot_is_never_before_the_portfolios_exist(
    make_broker: Callable[[], Broker], db: Database, clock: ManualClock
) -> None:
    clock.set(T0 + 100)  # created in the middle of a 5-minute slot
    broker = make_broker()
    run(broker, clock, T0 + 120)
    assert snapshot_times(db, "ai") == [T0 + 100]
    assert broker.equity_curves().ai == [(T0 + 100, START)]
    run(broker, clock, T0 + 300)
    assert snapshot_times(db, "ai") == [T0 + 100, T0 + 300]


def test_crypto_benchmark_closes_at_utc_midnight(
    broker: Broker, db: Database, clock: ManualClock, runtime_settings: SettingsStore
) -> None:
    runtime_settings.update({"benchmarks": ["BTC-USD"]}, T0)
    add_bar(db, "BTC-USD", T0 - MIN, 50_000.0)
    for day, close in (("2026-11-18", 49_000.0), ("2026-11-20", 51_000.0), ("2026-11-21", 52_000.0)):
        add_daily(db, "BTC-USD", day, close)
    clock.set(T0 + 2 * DAY)
    curve = broker.equity_curves().benchmarks["BTC-USD"]
    saturday, sunday = MONDAY_0000 - 2 * DAY, MONDAY_0000 - DAY
    # 100,000 EUR bought 2.5 BTC at 50,000 USD and 1 EUR = 1.25 USD; the close before the reset is left out
    assert curve == [(T0, START), (saturday, pytest.approx(102_000.0)), (sunday, pytest.approx(104_000.0))]


# -- concurrency --------------------------------------------------------------------------------------
def test_signals_orders_and_passes_from_three_threads(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    ai_symbols, me_symbols = ["SAP.DE", "NVDA", "BTC-USD"], ["^GDAXI", "SHEL.L"]
    for symbol in ai_symbols + me_symbols:
        data.set_quote(symbol, 100.0, T0)
        add_flat_bars(db, symbol, T0, 40, 100.0)
    signals = [make_signal(db, symbol, stop_dist=5.0, target_dist=10.0) for symbol in ai_symbols]
    errors: list[BaseException] = []

    def guarded(work: Callable[[], None]) -> threading.Thread:
        def target() -> None:
            try:
                work()
            except BaseException as exc:  # noqa: BLE001 - reported by the test
                errors.append(exc)

        return threading.Thread(target=target)

    def cycle() -> None:
        for minute in range(1, 21):
            broker.process(clock.now() + timedelta(minutes=minute))

    def engine() -> None:
        for signal in signals:
            assert broker.execute_signal(signal) is not None

    def user() -> None:
        for symbol in me_symbols:
            broker.place_order("me", OrderRequest(symbol=symbol, side="SHORT", amount_base=3000.0))
            broker.detail("me")

    threads = [guarded(cycle), guarded(engine), guarded(user)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert errors == [] and not any(thread.is_alive() for thread in threads)
    run(broker, clock, T0 + 30 * MIN)
    assert sorted(o.symbol for o in broker.orders("ai") if o.status == "filled") == sorted(ai_symbols)
    assert sorted(o.symbol for o in broker.orders("me") if o.status == "filled") == sorted(me_symbols)
    for key in ("ai", "me"):
        check_accounting(broker, key)
