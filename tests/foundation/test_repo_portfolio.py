import dataclasses
import logging
import sqlite3
import threading
from itertools import pairwise
from typing import Any

import pytest

from trading_local.db import repo_portfolio
from trading_local.db.database import Database
from trading_local.models import EquityPoint, Order, Portfolio, Position

T0 = 1_800_000_000
START_CASH = 100_000.0
AI, ME = 1, 2  # portfolio ids on a fresh database ('ai' is created first)


@pytest.fixture(autouse=True)
def instruments(db: Database) -> None:
    """orders.symbol and positions.symbol reference instruments."""
    db.executemany(
        "INSERT INTO instruments (symbol, name, asset_class, region, exchange, currency, timezone) "
        "VALUES (?, ?, 'equity', ?, ?, ?, ?)",
        [
            ("NVDA", "NVIDIA", "US", "XNAS", "USD", "America/New_York"),
            ("SAP.DE", "SAP", "EU", "XETR", "EUR", "Europe/Berlin"),
        ],
    )


@pytest.fixture
def portfolios(db: Database) -> dict[str, Portfolio]:
    return repo_portfolio.ensure_portfolios(db, base_currency="EUR", start_cash=START_CASH, now_ts=T0)


def add_signal(db: Database) -> int:
    """A stored signal for the signal_id foreign keys of orders and positions."""
    return db.execute(
        "INSERT INTO signals (created_ts, data_ts, symbol, direction, setup, timeframe, horizon_min, ref_price, atr, "
        "stop_dist, target_dist, score, confidence, status, engine_version) "
        "VALUES (?, ?, 'NVDA', 'LONG', 'breakout', '15m', 240, 100, 1.5, 2, 3, 0.7, 0.6, 'active', 'v1')",
        (T0, T0),
    )


def make_order(portfolio_id: int = AI, **fields: Any) -> Order:
    values: dict[str, Any] = {
        "created_ts": T0,
        "symbol": "NVDA",
        "intent": "open",
        "side": "LONG",
        "origin": "manual",
        "qty": 10.0,
    }
    values.update(fields)
    return Order(portfolio_id=portfolio_id, **values)


def make_position(portfolio_id: int = AI, symbol: str = "NVDA", **fields: Any) -> Position:
    values: dict[str, Any] = {"side": "LONG", "qty": 10.0, "entry_price": 100.0, "entry_fx": 0.92, "opened_ts": T0}
    values.update(fields)
    return Position(portfolio_id=portfolio_id, symbol=symbol, **values)


def assert_no_default_left(obj: Any) -> None:
    """Guard for the round-trip tests: a field added to the model must be added to the fixture as well."""
    for field in dataclasses.fields(obj):
        if field.name == "id":
            continue
        value = getattr(obj, field.name)
        if field.default is not dataclasses.MISSING:
            assert value != field.default, field.name


def cash(db: Database, key: str) -> tuple[float, float]:
    portfolio = repo_portfolio.get_portfolio(db, key)
    assert portfolio is not None
    return portfolio.cash, portfolio.fees_paid


def fill_equity(db: Database, portfolio_id: int, timestamps: list[int]) -> None:
    with db.write():
        for ts in timestamps:
            repo_portfolio.save_equity_point(db, EquityPoint(portfolio_id, ts, START_CASH + ts - T0, 40_000.0))


# ------------------------------------------------------------------------------------------------
# Portfolios
# ------------------------------------------------------------------------------------------------
def test_ensure_portfolios_creates_ai_and_me(db: Database) -> None:
    created = repo_portfolio.ensure_portfolios(db, base_currency="EUR", start_cash=START_CASH, now_ts=T0)

    assert created == {
        "ai": Portfolio("ai", "KI", "EUR", START_CASH, START_CASH, created_ts=T0, reset_ts=T0, fees_paid=0.0, id=AI),
        "me": Portfolio("me", "Ich", "EUR", START_CASH, START_CASH, created_ts=T0, reset_ts=T0, fees_paid=0.0, id=ME),
    }


def test_ensure_portfolios_is_idempotent(db: Database, portfolios: dict[str, Portfolio]) -> None:
    repo_portfolio.adjust_cash(db, AI, cash_delta=-2500.0, fee=1.0)

    again = repo_portfolio.ensure_portfolios(db, base_currency="USD", start_cash=5.0, now_ts=T0 + 999)

    assert again["me"] == portfolios["me"]
    assert again["ai"] == dataclasses.replace(portfolios["ai"], cash=START_CASH - 2501.0, fees_paid=1.0)
    assert db.scalar("SELECT COUNT(*) FROM portfolios") == 2


def test_ensure_portfolios_adds_only_the_missing_one(db: Database) -> None:
    db.execute(
        "INSERT INTO portfolios (key, name, base_currency, start_cash, cash, created_ts, reset_ts) "
        "VALUES ('me', 'Ich', 'EUR', 500, 400, 1, 2)"
    )

    created = repo_portfolio.ensure_portfolios(db, base_currency="EUR", start_cash=START_CASH, now_ts=T0)

    assert (created["me"].id, created["me"].cash, created["me"].created_ts) == (1, 400.0, 1)
    assert (created["ai"].id, created["ai"].cash) == (2, START_CASH)
    assert [portfolio.key for portfolio in repo_portfolio.list_portfolios(db)] == ["ai", "me"]  # 'ai' first


def test_get_portfolio_by_key_and_id(db: Database, portfolios: dict[str, Portfolio]) -> None:
    assert repo_portfolio.get_portfolio(db, "me") == portfolios["me"]
    assert repo_portfolio.get_portfolio_by_id(db, AI) == portfolios["ai"]
    assert repo_portfolio.get_portfolio(db, "nobody") is None
    assert repo_portfolio.get_portfolio_by_id(db, 99) is None
    assert repo_portfolio.list_portfolios(db) == [portfolios["ai"], portfolios["me"]]


@pytest.mark.usefixtures("portfolios")
def test_adjust_cash_books_the_delta_and_the_fee(db: Database) -> None:
    repo_portfolio.adjust_cash(db, AI, cash_delta=-2500.0, fee=1.0)  # buy
    repo_portfolio.adjust_cash(db, AI, cash_delta=300.5)  # no fee

    assert cash(db, "ai") == (START_CASH - 2501.0 + 300.5, 1.0)
    assert cash(db, "me") == (START_CASH, 0.0)


def test_adjust_cash_joins_the_callers_transaction(db: Database, portfolios: dict[str, Portfolio]) -> None:
    with pytest.raises(RuntimeError), db.write():
        repo_portfolio.adjust_cash(db, AI, cash_delta=-500.0, fee=1.0)
        assert cash(db, "ai") == (START_CASH - 501.0, 1.0)  # visible inside the transaction
        raise RuntimeError("the fill failed after the booking")
    assert repo_portfolio.get_portfolio(db, "ai") == portfolios["ai"]  # rolled back with the caller

    with db.write():
        repo_portfolio.adjust_cash(db, AI, cash_delta=-500.0, fee=1.0)
        repo_portfolio.insert_position(db, make_position())
    assert cash(db, "ai") == (START_CASH - 501.0, 1.0)
    assert len(repo_portfolio.open_positions(db, AI)) == 1


@pytest.mark.usefixtures("portfolios")
def test_adjust_cash_for_an_unknown_portfolio_rolls_everything_back(db: Database) -> None:
    with pytest.raises(LookupError, match="99"), db.write():
        repo_portfolio.adjust_cash(db, AI, cash_delta=-100.0)
        repo_portfolio.adjust_cash(db, 99, cash_delta=100.0)

    assert cash(db, "ai") == (START_CASH, 0.0)


@pytest.mark.usefixtures("portfolios")
def test_concurrent_cash_bookings_are_not_lost(db: Database) -> None:
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(50):
                repo_portfolio.adjust_cash(db, AI, cash_delta=1.0, fee=0.25)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert cash(db, "ai") == (START_CASH + 300.0, 100.0)


def test_reset_portfolios_starts_a_new_comparison_period(db: Database, portfolios: dict[str, Portfolio]) -> None:
    signal_id = add_signal(db)
    for portfolio_id in (AI, ME):
        repo_portfolio.adjust_cash(db, portfolio_id, cash_delta=-1000.0, fee=2.0)
        repo_portfolio.insert_order(db, make_order(portfolio_id, signal_id=signal_id))
        repo_portfolio.insert_position(db, make_position(portfolio_id, signal_id=signal_id))
        repo_portfolio.save_equity_point(db, EquityPoint(portfolio_id, T0, 99_000.0, 98_998.0))

    repo_portfolio.reset_portfolios(db, start_cash=50_000.0, now_ts=T0 + 3600)

    for key in ("ai", "me"):
        assert repo_portfolio.get_portfolio(db, key) == dataclasses.replace(
            portfolios[key], start_cash=50_000.0, cash=50_000.0, fees_paid=0.0, reset_ts=T0 + 3600
        )
    for table in ("orders", "positions", "equity_snapshots"):
        assert db.scalar(f"SELECT COUNT(*) FROM {table}") == 0
    assert db.scalar("SELECT COUNT(*) FROM signals") == 1  # the signal journal is not part of the reset


# ------------------------------------------------------------------------------------------------
# Orders
# ------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("portfolios")
def test_order_round_trip_with_every_field(db: Database) -> None:
    order = make_order(
        intent="close",
        side="SHORT",
        origin="stop",
        status="filled",
        notional_base=2500.0,
        ref_price=101.5,
        stop_price=99.0,
        target_price=106.0,
        stop_dist=2.5,
        target_dist=4.5,
        max_drift=0.75,
        time_stop_ts=T0 + 14_400,
        flatten_at_close=True,
        position_id=17,
        signal_id=add_signal(db),
        status_reason="Stopp ausgelöst",
        expire_ts=T0 + 900,
        filled_ts=T0 + 60,
        filled_qty=10.0,
        fill_price=101.52,
        fill_fx=0.9213,
        fee_base=1.0,
    )
    assert_no_default_left(order)

    stored = repo_portfolio.insert_order(db, order)

    assert stored is order and order.id == 1
    loaded = repo_portfolio.get_order(db, 1)
    assert loaded == order
    assert loaded is not None and loaded.flatten_at_close is True
    assert repo_portfolio.get_order(db, 2) is None


@pytest.mark.usefixtures("portfolios")
def test_order_defaults_and_update(db: Database) -> None:
    order = repo_portfolio.insert_order(db, make_order())
    untouched = repo_portfolio.insert_order(db, make_order())
    loaded = repo_portfolio.get_order(db, 1)
    assert loaded == order and loaded is not None and loaded.flatten_at_close is False

    order.status = "filled"
    order.filled_ts, order.filled_qty, order.fill_price, order.fill_fx, order.fee_base = T0 + 60, 10.0, 100.2, 0.92, 1.0
    repo_portfolio.update_order(db, order)

    assert repo_portfolio.get_order(db, 1) == order
    assert repo_portfolio.get_order(db, 2) == untouched


@pytest.mark.usefixtures("portfolios")
def test_order_insert_and_update_guard_the_id(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    order = repo_portfolio.insert_order(db, make_order())

    with pytest.raises(ValueError, match="already stored"):
        repo_portfolio.insert_order(db, order)
    with pytest.raises(ValueError, match="without an id"):
        repo_portfolio.update_order(db, make_order())
    with caplog.at_level(logging.WARNING, logger="trading_local.db.repo_portfolio"):
        repo_portfolio.update_order(db, make_order(id=55))

    assert "order 55" in caplog.text
    assert db.scalar("SELECT COUNT(*) FROM orders") == 1


@pytest.mark.usefixtures("portfolios")
def test_list_orders_is_newest_first(db: Database) -> None:
    for fields in [
        {"created_ts": T0 + 60, "status": "filled"},  # id 1
        {"created_ts": T0, "status": "pending"},  # id 2: stored later but created first
        {"created_ts": T0 + 120, "status": "pending"},  # id 3
        {"created_ts": T0 + 120, "status": "cancelled"},  # id 4: same time as 3, stored later
    ]:
        repo_portfolio.insert_order(db, make_order(**fields))
    repo_portfolio.insert_order(db, make_order(ME, created_ts=T0 + 500))  # id 5, other portfolio

    def order_ids(**filters: Any) -> list[int | None]:
        return [order.id for order in repo_portfolio.list_orders(db, AI, **filters)]

    assert order_ids() == [4, 3, 1, 2]
    assert order_ids(limit=2) == [4, 3]
    assert order_ids(status="pending") == [3, 2]
    assert order_ids(status="rejected") == []
    assert [order.id for order in repo_portfolio.list_orders(db, ME)] == [5]


@pytest.mark.usefixtures("portfolios")
def test_pending_orders_are_oldest_first(db: Database) -> None:
    repo_portfolio.insert_order(db, make_order(AI, created_ts=T0 + 60))  # id 1
    repo_portfolio.insert_order(db, make_order(ME, created_ts=T0))  # id 2
    repo_portfolio.insert_order(db, make_order(AI, created_ts=T0 + 30))  # id 3
    repo_portfolio.insert_order(db, make_order(AI, created_ts=T0 + 10, status="filled"))  # id 4
    repo_portfolio.insert_order(db, make_order(AI, created_ts=T0 + 60))  # id 5: same time as 1

    assert [order.id for order in repo_portfolio.pending_orders(db)] == [2, 3, 1, 5]
    assert [order.id for order in repo_portfolio.pending_orders(db, AI)] == [3, 1, 5]
    assert [order.id for order in repo_portfolio.pending_orders(db, ME)] == [2]


# ------------------------------------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("portfolios")
def test_position_round_trip_with_every_field(db: Database) -> None:
    position = make_position(
        side="SHORT",
        status="closed",
        stop_price=103.0,
        target_price=95.5,
        time_stop_ts=T0 + 14_400,
        flatten_at_close=True,
        signal_id=add_signal(db),
        checked_ts=T0 + 3540,
        closed_ts=T0 + 3600,
        exit_price=95.4,
        exit_fx=0.9187,
        exit_reason="target",
        fees_base=2.0,
        realized_pnl=40.31,
    )
    assert_no_default_left(position)

    stored = repo_portfolio.insert_position(db, position)

    assert stored is position and position.id == 1
    loaded = repo_portfolio.get_position(db, 1)
    assert loaded == position
    assert loaded is not None and loaded.flatten_at_close is True
    assert repo_portfolio.get_position(db, 2) is None


@pytest.mark.usefixtures("portfolios")
def test_position_defaults_and_update(db: Database) -> None:
    position = repo_portfolio.insert_position(db, make_position())
    untouched = repo_portfolio.insert_position(db, make_position(symbol="SAP.DE"))
    loaded = repo_portfolio.get_position(db, 1)
    assert loaded == position and loaded is not None and loaded.flatten_at_close is False

    position.status, position.closed_ts, position.exit_price, position.exit_fx = "closed", T0 + 600, 104.0, 0.93
    position.exit_reason, position.fees_base, position.realized_pnl = "manual", 2.0, 35.2
    repo_portfolio.update_position(db, position)

    assert repo_portfolio.get_position(db, 1) == position
    assert repo_portfolio.get_position(db, 2) == untouched


@pytest.mark.usefixtures("portfolios")
def test_position_insert_and_update_guard_the_id(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    position = repo_portfolio.insert_position(db, make_position())

    with pytest.raises(ValueError, match="already stored"):
        repo_portfolio.insert_position(db, position)
    with pytest.raises(ValueError, match="without an id"):
        repo_portfolio.update_position(db, make_position())
    with caplog.at_level(logging.WARNING, logger="trading_local.db.repo_portfolio"):
        repo_portfolio.update_position(db, make_position(id=55))

    assert "position 55" in caplog.text
    assert db.scalar("SELECT COUNT(*) FROM positions") == 1


@pytest.mark.usefixtures("portfolios")
def test_only_one_open_position_per_portfolio_and_symbol(db: Database) -> None:
    first = repo_portfolio.insert_position(db, make_position())
    second = make_position(side="SHORT")

    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.insert_position(db, second)
    assert second.id is None

    repo_portfolio.insert_position(db, make_position(symbol="SAP.DE"))  # another symbol is fine
    repo_portfolio.insert_position(db, make_position(ME))  # another portfolio is fine

    first.status, first.closed_ts = "closed", T0 + 600
    repo_portfolio.update_position(db, first)
    repo_portfolio.insert_position(db, second)  # the symbol is free again once the first position is closed
    assert second.id == 4

    reopened = dataclasses.replace(first, status="open", closed_ts=None)
    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.update_position(db, reopened)


@pytest.mark.usefixtures("portfolios")
def test_open_positions_are_oldest_first(db: Database) -> None:
    repo_portfolio.insert_position(db, make_position(AI, "NVDA", opened_ts=T0 + 60))  # id 1
    repo_portfolio.insert_position(db, make_position(ME, "NVDA", opened_ts=T0 + 30))  # id 2
    repo_portfolio.insert_position(db, make_position(AI, "SAP.DE", opened_ts=T0))  # id 3
    repo_portfolio.insert_position(db, make_position(ME, "SAP.DE", opened_ts=T0 - 60, status="closed"))  # id 4

    assert [position.id for position in repo_portfolio.open_positions(db)] == [3, 2, 1]
    assert [position.id for position in repo_portfolio.open_positions(db, AI)] == [3, 1]
    assert [position.id for position in repo_portfolio.open_positions(db, ME)] == [2]


@pytest.mark.usefixtures("portfolios")
def test_open_position_for_symbol(db: Database) -> None:
    repo_portfolio.insert_position(db, make_position(AI, "NVDA", status="closed", closed_ts=T0 - 60))
    current = repo_portfolio.insert_position(db, make_position(AI, "NVDA"))
    repo_portfolio.insert_position(db, make_position(ME, "SAP.DE"))

    assert repo_portfolio.open_position_for_symbol(db, AI, "NVDA") == current
    assert repo_portfolio.open_position_for_symbol(db, AI, "SAP.DE") is None
    assert repo_portfolio.open_position_for_symbol(db, ME, "NVDA") is None


@pytest.mark.usefixtures("portfolios")
def test_closed_positions_are_most_recently_closed_first(db: Database) -> None:
    for portfolio_id, closed_ts in [(AI, T0 + 200), (AI, T0 + 100), (AI, T0 + 300), (ME, T0 + 400), (AI, T0 + 300)]:
        repo_portfolio.insert_position(db, make_position(portfolio_id, status="closed", closed_ts=closed_ts))
    repo_portfolio.insert_position(db, make_position(AI))  # id 6, still open

    def closed_ids(**filters: Any) -> list[int | None]:
        return [position.id for position in repo_portfolio.closed_positions(db, AI, **filters)]

    assert closed_ids() == [5, 3, 1, 2]
    assert closed_ids(limit=2) == [5, 3]
    assert closed_ids(since_ts=T0 + 200) == [5, 3, 1]
    assert closed_ids(since_ts=T0 + 301) == []
    assert [position.id for position in repo_portfolio.closed_positions(db, ME)] == [4]


# ------------------------------------------------------------------------------------------------
# Equity
# ------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("portfolios")
def test_save_equity_point_replaces_the_point_of_the_same_time(db: Database) -> None:
    first = EquityPoint(AI, T0, 100_250.0, 60_000.0, long_exposure=30_000.0, short_exposure=10_250.0, open_positions=3)
    corrected = EquityPoint(AI, T0, 100_300.0, 60_000.0)

    repo_portfolio.save_equity_point(db, first)
    assert repo_portfolio.equity_series(db, AI) == [first]

    repo_portfolio.save_equity_point(db, corrected)
    repo_portfolio.save_equity_point(db, EquityPoint(ME, T0, 99_000.0, 99_000.0))
    assert repo_portfolio.equity_series(db, AI) == [corrected]
    assert db.scalar("SELECT COUNT(*) FROM equity_snapshots") == 2


@pytest.mark.usefixtures("portfolios")
def test_equity_series_returns_a_short_series_completely(db: Database) -> None:
    timestamps = [T0, T0 + 60, T0 + 61, T0 + 5000, T0 + 5001, T0 + 90_000]  # irregular on purpose
    fill_equity(db, AI, list(reversed(timestamps)))
    fill_equity(db, ME, [T0 + 7])

    assert [point.ts for point in repo_portfolio.equity_series(db, AI)] == timestamps
    assert [point.ts for point in repo_portfolio.equity_series(db, AI, max_points=6)] == timestamps
    assert [point.ts for point in repo_portfolio.equity_series(db, AI, since_ts=T0 + 61)] == timestamps[2:]
    assert repo_portfolio.equity_series(db, AI, since_ts=T0 + 90_001) == []
    assert repo_portfolio.equity_series(db, 99) == []


@pytest.mark.usefixtures("portfolios")
def test_equity_series_thins_evenly_and_keeps_the_first_and_the_last_point(db: Database) -> None:
    fill_equity(db, AI, [T0 + 60 * minute for minute in range(1000)])
    fill_equity(db, ME, [T0 + 60 * minute + 30 for minute in range(50)])

    series = repo_portfolio.equity_series(db, AI, max_points=100)

    minutes = [(point.ts - T0) // 60 for point in series]
    assert len(series) == 100
    assert (minutes[0], minutes[-1]) == (0, 999)
    assert {b - a for a, b in pairwise(minutes)} == {10, 11}  # 999 steps spread over 99 gaps
    assert all(point.portfolio_id == AI and point.equity == START_CASH + point.ts - T0 for point in series)
    assert len(repo_portfolio.equity_series(db, AI)) == 600  # default limit
    assert [(point.ts - T0) // 60 for point in repo_portfolio.equity_series(db, AI, max_points=2)] == [0, 999]


@pytest.mark.usefixtures("portfolios")
def test_equity_series_thins_only_the_requested_range(db: Database) -> None:
    fill_equity(db, AI, [T0 + 60 * minute for minute in range(1000)])

    series = repo_portfolio.equity_series(db, AI, since_ts=T0 + 60 * 400, max_points=7)

    minutes = [(point.ts - T0) // 60 for point in series]
    assert len(series) == 7
    assert (minutes[0], minutes[-1]) == (400, 999)
    assert {b - a for a, b in pairwise(minutes)} <= {99, 100}


@pytest.mark.usefixtures("portfolios")
def test_equity_series_with_gaps_stays_within_the_limit(db: Database) -> None:
    # two busy sessions with a long pause in between
    timestamps = [T0 + n for n in range(300)] + [T0 + 500_000 + n for n in range(300)]
    fill_equity(db, AI, timestamps)

    series = repo_portfolio.equity_series(db, AI, max_points=50)

    picked = [point.ts for point in series]
    assert 2 <= len(picked) <= 50
    assert (picked[0], picked[-1]) == (timestamps[0], timestamps[-1])
    assert picked == sorted(set(picked)) and set(picked) <= set(timestamps)


@pytest.mark.usefixtures("portfolios")
@pytest.mark.parametrize("max_points", [1, 0, -5])
def test_equity_series_needs_room_for_both_ends(db: Database, max_points: int) -> None:
    with pytest.raises(ValueError, match="max_points"):
        repo_portfolio.equity_series(db, AI, max_points=max_points)


@pytest.mark.usefixtures("portfolios")
def test_last_equity_point(db: Database) -> None:
    assert repo_portfolio.last_equity_point(db, AI) is None
    fill_equity(db, AI, [T0, T0 + 60, T0 + 120])
    fill_equity(db, ME, [T0 + 600])

    def last_ts(**filters: Any) -> int | None:
        point = repo_portfolio.last_equity_point(db, AI, **filters)
        return None if point is None else point.ts

    assert last_ts() == T0 + 120
    assert last_ts(before_ts=T0 + 120) == T0 + 60  # strictly before
    assert last_ts(before_ts=T0 + 121) == T0 + 120
    assert last_ts(before_ts=T0) is None
    assert repo_portfolio.last_equity_point(db, AI) == EquityPoint(AI, T0 + 120, START_CASH + 120, 40_000.0)


# ------------------------------------------------------------------------------------------------
# Review additions: integrity, concurrency, hostile input, thinning properties, query cost
# ------------------------------------------------------------------------------------------------
def query_plan(db: Database, run: Any) -> list[str]:
    """The query plan of the single statement that ``run`` sends to the database."""
    statements: list[str] = []
    db.conn().set_trace_callback(statements.append)
    try:
        run()
    finally:
        db.conn().set_trace_callback(None)
    [statement] = statements
    return [row["detail"] for row in db.query("EXPLAIN QUERY PLAN " + statement)]


@pytest.mark.usefixtures("portfolios")
def test_rows_must_refer_to_stored_portfolios_instruments_and_signals(db: Database) -> None:
    orders = [make_order(99), make_order(symbol="TSLA"), make_order(signal_id=999)]
    positions = [make_position(99), make_position(symbol="TSLA"), make_position(signal_id=999)]

    for order in orders:
        with pytest.raises(sqlite3.IntegrityError):
            repo_portfolio.insert_order(db, order)
    for position in positions:
        with pytest.raises(sqlite3.IntegrityError):
            repo_portfolio.insert_position(db, position)
    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.save_equity_point(db, EquityPoint(99, T0, 1.0, 1.0))

    assert [row.id for row in (*orders, *positions)] == [None] * 6
    for table in ("orders", "positions", "equity_snapshots"):
        assert db.scalar(f"SELECT COUNT(*) FROM {table}") == 0


@pytest.mark.usefixtures("portfolios")
def test_invalid_rows_are_rejected_by_the_schema(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.insert_position(db, make_position(qty=0.0))  # a position has a positive size
    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.insert_order(db, make_order(status="open"))
    stored = repo_portfolio.insert_order(db, make_order())
    with pytest.raises(sqlite3.IntegrityError):
        repo_portfolio.update_order(db, dataclasses.replace(stored, origin="robot"))

    assert repo_portfolio.get_order(db, 1) == stored


@pytest.mark.usefixtures("portfolios")
def test_concurrent_opens_of_one_symbol_leave_a_single_open_position(db: Database) -> None:
    workers = 6
    barrier = threading.Barrier(workers)
    opened: list[Position] = []
    rejected: list[BaseException] = []

    def worker() -> None:
        barrier.wait(timeout=10)
        try:
            opened.append(repo_portfolio.insert_position(db, make_position()))
        except sqlite3.IntegrityError as exc:
            rejected.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert (len(opened), len(rejected)) == (1, workers - 1)
    assert repo_portfolio.open_positions(db, AI) == opened
    assert repo_portfolio.open_position_for_symbol(db, AI, "NVDA") == opened[0]


def test_reset_portfolios_joins_the_callers_transaction(db: Database, portfolios: dict[str, Portfolio]) -> None:
    repo_portfolio.adjust_cash(db, AI, cash_delta=-1000.0, fee=2.0)
    order = repo_portfolio.insert_order(db, make_order())
    position = repo_portfolio.insert_position(db, make_position())
    repo_portfolio.save_equity_point(db, EquityPoint(AI, T0, 99_000.0, 98_998.0))

    with pytest.raises(RuntimeError), db.write():
        repo_portfolio.reset_portfolios(db, start_cash=50_000.0, now_ts=T0 + 3600)
        assert repo_portfolio.open_positions(db) == []  # the reset is visible inside the transaction ...
        raise RuntimeError("closing the books failed after the reset")

    # ... and undone as a whole with it
    assert cash(db, "ai") == (START_CASH - 1002.0, 2.0)
    assert repo_portfolio.get_portfolio(db, "me") == portfolios["me"]
    assert repo_portfolio.list_orders(db, AI) == [order]
    assert repo_portfolio.open_positions(db, AI) == [position]
    assert len(repo_portfolio.equity_series(db, AI)) == 1


def test_reset_portfolios_without_portfolios_or_history_is_harmless(db: Database) -> None:
    repo_portfolio.reset_portfolios(db, start_cash=50_000.0, now_ts=T0)  # nothing to reset yet

    created = repo_portfolio.ensure_portfolios(db, base_currency="EUR", start_cash=START_CASH, now_ts=T0)
    repo_portfolio.reset_portfolios(db, start_cash=75_000.0, now_ts=T0 + 60)

    for key in ("ai", "me"):
        assert repo_portfolio.get_portfolio(db, key) == dataclasses.replace(
            created[key], start_cash=75_000.0, cash=75_000.0, reset_ts=T0 + 60
        )


@pytest.mark.usefixtures("portfolios")
@pytest.mark.parametrize("text", ["ai' OR '1'='1", "pending'; DROP TABLE orders; --", 'NV"DA', "NVD%", "%"])
def test_filters_take_their_values_literally(db: Database, text: str) -> None:
    repo_portfolio.insert_order(db, make_order())
    repo_portfolio.insert_position(db, make_position())

    assert repo_portfolio.get_portfolio(db, text) is None
    assert repo_portfolio.list_orders(db, AI, status=text) == []
    assert repo_portfolio.open_position_for_symbol(db, AI, text) is None
    assert db.scalar("SELECT COUNT(*) FROM orders") == 1  # and nothing was executed as SQL


@pytest.mark.usefixtures("portfolios")
def test_positions_opened_at_the_same_time_are_ordered_by_id(db: Database) -> None:
    repo_portfolio.insert_position(db, make_position(AI, "SAP.DE"))  # id 1
    repo_portfolio.insert_position(db, make_position(AI, "NVDA"))  # id 2
    repo_portfolio.insert_position(db, make_position(ME, "NVDA"))  # id 3

    assert [position.id for position in repo_portfolio.open_positions(db)] == [1, 2, 3]
    assert [position.id for position in repo_portfolio.open_positions(db, AI)] == [1, 2]


@pytest.mark.usefixtures("portfolios")
@pytest.mark.parametrize("max_points", [2, 3, 5, 50, 199, 200])
def test_thinned_equity_series_is_a_subset_within_the_limit_that_keeps_both_ends(db: Database, max_points: int) -> None:
    # bursts, single snapshots and long pauses, as left behind by a platform that does not run around the clock
    timestamps = sorted({T0 + (n * n * 37) % 100_000 for n in range(400)} | {T0, T0 + 100_000})
    assert len(timestamps) > max_points
    fill_equity(db, AI, timestamps)
    fill_equity(db, ME, [T0 - 5, T0 + 100_005])  # must never show up

    picked = [point.ts for point in repo_portfolio.equity_series(db, AI, max_points=max_points)]

    assert 2 <= len(picked) <= max_points
    assert (picked[0], picked[-1]) == (timestamps[0], timestamps[-1])
    assert picked == sorted(set(picked)) and set(picked) <= set(timestamps)


@pytest.mark.usefixtures("portfolios")
def test_equity_series_one_point_over_the_limit_is_thinned(db: Database) -> None:
    timestamps = [T0 + 60 * minute for minute in range(11)]
    fill_equity(db, AI, timestamps)

    assert [point.ts for point in repo_portfolio.equity_series(db, AI, max_points=11)] == timestamps
    thinned = [point.ts for point in repo_portfolio.equity_series(db, AI, max_points=10)]
    assert len(thinned) == 10 and (thinned[0], thinned[-1]) == (timestamps[0], timestamps[-1])
    assert set(thinned) <= set(timestamps)


@pytest.mark.usefixtures("portfolios")
def test_equity_series_since_ts_decides_what_the_first_point_is(db: Database) -> None:
    fill_equity(db, AI, [T0 + 60 * minute for minute in range(100)])

    # since_ts between two snapshots: the series starts with the first snapshot at or after it, also when thinned
    whole = repo_portfolio.equity_series(db, AI, since_ts=T0 + 60 * 90 + 1)
    thinned = repo_portfolio.equity_series(db, AI, since_ts=T0 + 60 * 40 + 1, max_points=5)

    assert [(point.ts - T0) // 60 for point in whole] == list(range(91, 100))
    assert [(point.ts - T0) // 60 for point in thinned][0] == 41
    assert [(point.ts - T0) // 60 for point in thinned][-1] == 99
    assert len(thinned) == 5


@pytest.mark.usefixtures("portfolios")
def test_hot_readers_use_the_partial_indexes(db: Database) -> None:
    def plan_of(run: Any) -> str:
        return " | ".join(query_plan(db, run))

    # A status bound as a parameter instead of written as a literal would silently lose these indexes.
    assert "orders_pending_idx" in plan_of(lambda: repo_portfolio.pending_orders(db))
    assert "positions_one_open_idx" in plan_of(lambda: repo_portfolio.open_positions(db))
    assert "positions_one_open_idx" in plan_of(lambda: repo_portfolio.open_position_for_symbol(db, AI, "NVDA"))
    assert "USING PRIMARY KEY (portfolio_id=? AND ts<?)" in plan_of(
        lambda: repo_portfolio.last_equity_point(db, AI, before_ts=T0)
    )
    assert "orders_portfolio_idx" in plan_of(lambda: repo_portfolio.list_orders(db, AI))


@pytest.mark.usefixtures("portfolios")
def test_thinning_reads_a_fixed_number_of_snapshots(db: Database) -> None:
    fill_equity(db, AI, [T0 + 60 * minute for minute in range(50)])
    statements: list[str] = []
    db.conn().set_trace_callback(statements.append)
    try:
        series = repo_portfolio.equity_series(db, AI, max_points=10)
    finally:
        db.conn().set_trace_callback(None)

    assert len(series) == 10
    assert len(statements) == 2  # the head of the range, then the thinned series - never one statement per point
    thinned_plan = [row["detail"] for row in db.query("EXPLAIN QUERY PLAN " + statements[1])]
    assert not [step for step in thinned_plan if step.startswith("SCAN equity_snapshots")]
