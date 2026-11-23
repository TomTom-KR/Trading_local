"""The fill rule, cash accounting in four currencies and every exit path of ``process``."""

import math

import pytest

from trading_local.clock import ManualClock, from_ts
from trading_local.config import DEFAULT_COSTS
from trading_local.contracts import Broker
from trading_local.db import repo_ops, repo_portfolio
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import OrderRequest
from trading_local.models import Instrument
from trading_local.testing import seed_instruments
from tests.sim.simkit import (
    FEE,
    MIN,
    MONDAY_0000,
    SETTLE_S,
    SLIP_ASIA,
    SLIP_CRYPTO,
    SLIP_EU,
    SLIP_INDEX,
    SLIP_US,
    T0,
    FakeMarketData,
    add_bar,
    add_flat_bars,
    check_accounting,
    make_signal,
    open_position,
)

START = 100_000.0


def run(broker: Broker, clock: ManualClock, ts: int) -> list:
    clock.set(ts)
    return broker.process(clock.now())


def place(broker: Broker, symbol: str, side: str = "LONG", key: str = "me", **fields: object):
    if "qty" not in fields and "amount_base" not in fields:
        fields["qty"] = 10
    return broker.place_order(key, OrderRequest(symbol=symbol, side=side, **fields))  # type: ignore[arg-type]


def order_by_id(broker: Broker, order_id: int, key: str = "me"):
    return next(o for o in broker.orders(key) if o.id == order_id)


# -- fill rule --------------------------------------------------------------------------------------
def test_fill_at_the_open_of_the_first_bar_after_the_order(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    clock.set(T0 + 30)
    order = place(broker, "SAP.DE")
    add_bar(db, "SAP.DE", T0, 99.0)  # opened before the order: never used
    add_bar(db, "SAP.DE", T0 + MIN, 105.0)
    run(broker, clock, T0 + MIN + 59)  # the 15:01 bar is still forming
    assert order_by_id(broker, order.id).status == "pending"
    run(broker, clock, T0 + 2 * MIN)
    filled = order_by_id(broker, order.id)
    assert filled.status == "filled"
    assert filled.filled_ts == T0 + MIN
    assert filled.fill_price == pytest.approx(105.0 * (1 + SLIP_EU))


def test_order_waits_for_a_delayed_feed(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    order = place(broker, "SAP.DE")
    for minutes in (1, 5, 15):
        run(broker, clock, T0 + minutes * MIN)
        assert order_by_id(broker, order.id).status == "pending"
    add_bar(db, "SAP.DE", T0, 100.5)  # 15 minutes late
    add_bar(db, "SAP.DE", T0 + MIN, 101.0)
    run(broker, clock, T0 + 16 * MIN)
    filled = order_by_id(broker, order.id)
    assert filled.filled_ts == T0 and filled.fill_price == pytest.approx(100.5 * (1 + SLIP_EU))


def test_order_waits_while_the_venue_is_closed(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    friday_night = T0 + 7 * 3600  # 22:00 UTC: Nasdaq closed until Monday 14:30
    clock.set(friday_night)
    data.set_quote("NVDA", 200.0, friday_night)
    order = place(broker, "NVDA")
    for ts in (friday_night + 3600, MONDAY_0000, MONDAY_0000 + 14 * 3600):
        run(broker, clock, ts)
        assert order_by_id(broker, order.id).status == "pending"
    monday_open = MONDAY_0000 + 14 * 3600 + 30 * MIN
    add_bar(db, "NVDA", monday_open, 202.0)
    run(broker, clock, monday_open + MIN)
    assert order_by_id(broker, order.id).filled_ts == monday_open


# -- round trips ------------------------------------------------------------------------------------
def test_long_round_trip_in_eur(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    place(broker, "SAP.DE")
    check_accounting(broker, "me")
    add_bar(db, "SAP.DE", T0, 101.0, high=102.0, low=100.5, close=101.5)
    run(broker, clock, T0 + MIN)
    entry = 101.0 * (1 + SLIP_EU)
    summary = broker.summary("me")
    assert summary.cash == pytest.approx(START - 10 * entry - FEE)
    assert summary.fees_paid == FEE
    position = broker.detail("me").positions[0]
    assert (position.entry_price, position.entry_fx, position.opened_ts) == (pytest.approx(entry), 1.0, T0)
    assert position.last_price == 100.0  # the quote is the mark
    check_accounting(broker, "me")

    data.set_quote("SAP.DE", 103.0, T0 + 2 * MIN)
    clock.set(T0 + 2 * MIN)
    broker.close_position("me", position.id)
    add_bar(db, "SAP.DE", T0 + MIN, 102.0)  # every traded minute has its bar: none can still be inserted
    add_bar(db, "SAP.DE", T0 + 2 * MIN, 103.0)
    closed = run(broker, clock, T0 + 3 * MIN)
    exit_ = 103.0 * (1 - SLIP_EU)
    assert [p.id for p in closed] == [position.id]
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("manual", T0 + 2 * MIN)
    assert trade.exit_price == pytest.approx(exit_)
    assert trade.realized_pnl == pytest.approx(10 * (exit_ - entry) - 2 * FEE)
    assert trade.realized_pct == pytest.approx(trade.realized_pnl / (10 * entry) * 100)
    summary = broker.summary("me")
    assert summary.cash == pytest.approx(START + trade.realized_pnl)
    assert summary.equity == pytest.approx(summary.cash)
    assert summary.open_positions == 0 and summary.closed_trades == 1
    check_accounting(broker, "me")


def test_short_round_trip_in_usd_with_fx_move(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("NVDA", 200.0, T0)
    place(broker, "NVDA", "SHORT")
    add_bar(db, "NVDA", T0, 200.0)
    run(broker, clock, T0 + MIN)
    entry, entry_fx = 200.0 * (1 - SLIP_US), 0.8
    assert broker.summary("me").cash == pytest.approx(START + 10 * entry * entry_fx - FEE)
    check_accounting(broker, "me")

    data.set_quote("EURUSD=X", 1.6, T0 + 2 * MIN)  # the dollar falls: 1 USD = 0.625 EUR
    data.set_quote("NVDA", 190.0, T0 + 2 * MIN)
    check_accounting(broker, "me")
    position = broker.detail("me").positions[0]
    assert position.exposure_base == pytest.approx(-10 * 190.0 * 0.625)
    # The proceeds stay in dollars: only the result of 10 * (entry - 190) USD is converted at the current rate.
    assert position.unrealized_pnl == pytest.approx(-10 * (190.0 - entry) * 0.625 - FEE)
    assert broker.summary("me").equity == pytest.approx(START + 10 * (entry - 190.0) * 0.625 - FEE)
    clock.set(T0 + 2 * MIN)
    broker.close_position("me", position.id)
    add_bar(db, "NVDA", T0 + MIN, 195.0)
    add_bar(db, "NVDA", T0 + 2 * MIN, 190.0)
    run(broker, clock, T0 + 3 * MIN)
    exit_ = 190.0 * (1 + SLIP_US)
    trade = broker.trades("me")[0]
    assert trade.realized_pnl == pytest.approx(-10 * (exit_ - entry) * 0.625 - 2 * FEE)
    assert broker.summary("me").cash == pytest.approx(START + trade.realized_pnl)
    check_accounting(broker, "me")


def test_a_stopped_out_short_loses_even_when_its_currency_falls(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    # Short 10 NVDA at 200 USD (1 USD = 0.80 EUR), stop 210. The dollar falls to 0.625 EUR before the stop is hit.
    # Booking the proceeds in euros would turn the stopped-out short into a profit of about 287 EUR (an FX bet
    # on the whole notional); in a real account the proceeds stay in dollars and only the loss is converted.
    data.set_quote("NVDA", 200.0, T0)
    place(broker, "NVDA", "SHORT", stop_price=210.0)
    add_bar(db, "NVDA", T0, 200.0)
    run(broker, clock, T0 + MIN)
    entry = 200.0 * (1 - SLIP_US)
    data.set_quote("EURUSD=X", 1.6, T0 + MIN)
    add_bar(db, "EURUSD=X", T0, 1.6)  # the rate known at the exit bar
    add_bar(db, "NVDA", T0 + MIN, 205.0, high=211.0, low=204.0)
    run(broker, clock, T0 + 2 * MIN)
    trade = broker.trades("me")[0]
    exit_ = 210.0 * (1 + SLIP_US)
    assert (trade.exit_reason, trade.closed_ts) == ("stop", T0 + MIN)
    assert trade.realized_pnl == pytest.approx(-10 * (exit_ - entry) * 0.625 - 2 * FEE)
    assert trade.realized_pnl < 0
    assert broker.summary("me").cash == pytest.approx(START + trade.realized_pnl)
    check_accounting(broker, "me")


def test_kraken_fills_move_only_by_the_spread_and_the_taker_fee_is_charged_as_money(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, instruments: InstrumentBook
) -> None:
    # A Kraken pair has no flat fee. Its proportional taker fee must NOT move the fill price (that would put a
    # tight stop right at the market price); it is charged as money on each order value.
    pair = Instrument("XBTEUR.K", "Bitcoin/Euro (Kraken)", "crypto", "GLOBAL", "KRAKEN", "EUR", timezone="UTC")
    seed_instruments(db, [pair])
    instruments.reload()
    position = open_position(broker, db, data, clock, pair.symbol, qty=0.5, price=50_000.0)
    spread = DEFAULT_COSTS.slippage_bps_kraken / 10_000
    fee = DEFAULT_COSTS.fee_bps_kraken / 10_000
    entry = 50_000.0 * (1 + spread)
    entry_fee = 0.5 * entry * fee
    assert position.entry_price == pytest.approx(entry)
    assert position.fees_base == pytest.approx(entry_fee)
    assert broker.summary("me").cash == pytest.approx(START - 0.5 * entry - entry_fee)
    broker.close_position("me", position.id)
    add_bar(db, pair.symbol, clock.ts(), 51_000.0)
    run(broker, clock, clock.ts() + MIN)
    trade = broker.trades("me")[0]
    exit_ = 51_000.0 * (1 - spread)
    fees = entry_fee + 0.5 * exit_ * fee
    assert trade.exit_price == pytest.approx(exit_)
    assert trade.fees_base == pytest.approx(fees)
    assert trade.realized_pnl == pytest.approx(0.5 * (exit_ - entry) - fees)
    summary = broker.summary("me")
    assert (summary.fees_paid, summary.cash) == (pytest.approx(fees), pytest.approx(START + trade.realized_pnl))


def test_long_round_trip_in_pence(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    position = open_position(broker, db, data, clock, "SHEL.L", qty=100, price=2500.0)
    entry = 2500.0 * (1 + SLIP_EU)
    assert position.entry_fx == pytest.approx(1.25)
    assert broker.summary("me").cash == pytest.approx(START - 100 * entry * 0.01 * 1.25 - FEE)
    check_accounting(broker, "me")
    data.set_quote("SHEL.L", 2600.0, clock.ts())
    assert broker.detail("me").positions[0].exposure_base == pytest.approx(100 * 2600.0 * 0.01 * 1.25)
    check_accounting(broker, "me")
    broker.close_position("me", position.id)
    add_bar(db, "SHEL.L", clock.ts(), 2600.0)
    run(broker, clock, clock.ts() + MIN)
    exit_ = 2600.0 * (1 - SLIP_EU)
    trade = broker.trades("me")[0]
    assert trade.realized_pnl == pytest.approx(100 * 0.01 * 1.25 * (exit_ - entry) - 2 * FEE)
    check_accounting(broker, "me")


def test_long_round_trip_in_jpy_across_the_weekend(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("7203.T", 3000.0, T0)
    place(broker, "7203.T", qty=100)
    run(broker, clock, T0 + 3600)  # Tokyo is closed: nothing happens
    assert broker.summary("me").open_positions == 0
    add_bar(db, "7203.T", MONDAY_0000, 3000.0)
    run(broker, clock, MONDAY_0000 + MIN)
    entry = 3000.0 * (1 + SLIP_ASIA)
    position = broker.detail("me").positions[0]
    assert position.entry_fx == pytest.approx(1 / 160)
    assert broker.summary("me").cash == pytest.approx(START - 100 * entry / 160 - FEE)
    check_accounting(broker, "me")
    data.set_quote("EURJPY=X", 150.0, MONDAY_0000 + 3600)  # the yen gains
    clock.set(MONDAY_0000 + 3600)
    broker.close_position("me", position.id)
    add_flat_bars(db, "7203.T", MONDAY_0000 + MIN, 59, 3050.0)  # Tokyo trades: no minute may be missing
    add_bar(db, "7203.T", MONDAY_0000 + 3600, 3100.0)
    run(broker, clock, MONDAY_0000 + 3600 + MIN)
    exit_ = 3100.0 * (1 - SLIP_ASIA)
    trade = broker.trades("me")[0]
    assert trade.realized_pnl == pytest.approx(100 * (exit_ / 150 - entry / 160) - 2 * FEE)
    check_accounting(broker, "me")


def test_fill_uses_the_fx_rate_known_at_the_bar(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    add_bar(db, "EURUSD=X", T0 - MIN, 1.25, close=1.28)  # closes at T0
    data.set_quote("EURUSD=X", 1.40, T0 + 20 * MIN)  # a later rate must not be used for the fill
    data.set_quote("NVDA", 200.0, T0)
    place(broker, "NVDA")
    add_bar(db, "NVDA", T0, 200.0)
    run(broker, clock, T0 + 20 * MIN)
    position = broker.detail("me").positions[0]
    assert position.entry_fx == pytest.approx(1 / 1.28)
    check_accounting(broker, "me")


# -- lots -------------------------------------------------------------------------------------------
def test_amount_orders_respect_the_lot_of_each_asset_class(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    btc = open_position(broker, db, data, clock, "BTC-USD", amount=1000.0, price=50_000.0)
    unit_value = 50_000.0 * (1 + SLIP_CRYPTO) * 0.8
    assert btc.qty == math.floor(1000.0 / unit_value * 1e6) / 1e6
    dax = open_position(broker, db, data, clock, "^GDAXI", amount=10_000.0, price=20_000.0)
    assert dax.qty == math.floor(10_000.0 / (20_000.0 * (1 + SLIP_INDEX)) * 100) / 100
    sap = open_position(broker, db, data, clock, "SAP.DE", amount=1000.0, price=99.0)
    assert sap.qty == 10.0
    check_accounting(broker, "me")


def test_amount_too_small_at_the_fill_is_rejected(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    order = place(broker, "SAP.DE", amount_base=100.5)
    add_bar(db, "SAP.DE", T0, 101.0)
    run(broker, clock, T0 + MIN)
    out = order_by_id(broker, order.id)
    assert (out.status, out.status_reason) == ("rejected", "Betrag reicht nicht für eine Einheit")
    assert broker.summary("me").cash == START
    entries = repo_ops.list_activity(db, category="sim")
    assert entries[0].level == "warning" and "abgelehnt – Betrag reicht nicht für eine Einheit" in entries[0].message


# -- buying power at the fill -----------------------------------------------------------------------
def test_buying_power_is_rechecked_at_the_fill(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    data.set_quote("BTC-USD", 50_000.0, T0)
    first = place(broker, "SAP.DE", amount_base=60_000.0)
    second = place(broker, "BTC-USD", amount_base=60_000.0)  # fits on its own, not after the first
    add_bar(db, "SAP.DE", T0, 100.0)
    add_bar(db, "BTC-USD", T0, 50_000.0)
    run(broker, clock, T0 + MIN)
    assert order_by_id(broker, first.id).status == "filled"
    out = order_by_id(broker, second.id)
    assert out.status == "rejected"
    assert out.status_reason.startswith("Kaufkraft reicht bei Ausführung nicht")
    summary = broker.summary("me")
    assert summary.gross_leverage <= 1.0
    check_accounting(broker, "me")


def test_short_exposure_uses_buying_power_too(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, runtime_settings: SettingsStore
) -> None:
    open_position(broker, db, data, clock, "SAP.DE", "SHORT", amount=60_000.0, price=100.0)
    summary = broker.summary("me")
    assert summary.short_exposure == pytest.approx(600 * 100.0)
    assert summary.buying_power == pytest.approx(summary.equity - summary.short_exposure)
    runtime_settings.update({"max_gross_leverage": 2.0}, T0)
    assert broker.summary("me").buying_power == pytest.approx(2 * summary.equity - summary.short_exposure)


# -- protective exits -------------------------------------------------------------------------------
def opened_with_levels(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, entry_bar: dict) -> None:
    """LONG 10 SAP with stop 99 and target 103, filled on the bar T0 (described by ``entry_bar``)."""
    data.set_quote("SAP.DE", 100.0, T0)
    place(broker, "SAP.DE", stop_price=99.0, target_price=103.0)
    add_bar(db, "SAP.DE", T0, 100.0, **entry_bar)
    run(broker, clock, T0 + MIN)
    data.drop_quote("SAP.DE")


def test_stop_inside_the_entry_bar(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 98.8, "close": 99.5})
    assert broker.summary("me").open_positions == 1  # the entry bar is examined in the next pass
    closed = run(broker, clock, T0 + 2 * MIN)
    assert len(closed) == 1
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("stop", T0)
    assert trade.exit_price == pytest.approx(99.0 * (1 - SLIP_EU))
    exit_order = broker.orders("me")[0]
    assert (exit_order.intent, exit_order.origin, exit_order.status, exit_order.filled_ts) == ("close", "stop", "filled", T0)
    check_accounting(broker, "me")


def test_gap_through_the_stop_fills_at_the_open(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.6})
    add_bar(db, "SAP.DE", T0 + MIN, 97.0, high=97.5, low=96.5)
    run(broker, clock, T0 + 2 * MIN)
    trade = broker.trades("me")[0]
    assert trade.exit_reason == "stop" and trade.closed_ts == T0 + MIN
    assert trade.exit_price == pytest.approx(97.0 * (1 - SLIP_EU))


def test_stop_wins_when_a_bar_touches_both(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.6})
    add_bar(db, "SAP.DE", T0 + MIN, 100.0, high=103.5, low=98.5)
    run(broker, clock, T0 + 2 * MIN)
    assert broker.trades("me")[0].exit_reason == "stop"


def test_target_is_a_limit_fill_without_slippage(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.6})
    add_flat_bars(db, "SAP.DE", T0 + MIN, 5, 100.5)
    add_bar(db, "SAP.DE", T0 + 6 * MIN, 100.5, high=103.2, low=100.2)
    closed = run(broker, clock, T0 + 10 * MIN)
    trade = broker.trades("me")[0]
    assert [p.id for p in closed] == [trade.id]
    assert (trade.exit_reason, trade.exit_price, trade.closed_ts) == ("target", 103.0, T0 + 6 * MIN)
    check_accounting(broker, "me")


def test_checked_bars_are_not_examined_twice(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.6})
    add_flat_bars(db, "SAP.DE", T0 + MIN, 3, 100.0)
    run(broker, clock, T0 + 4 * MIN)
    stored = repo_portfolio.open_positions(db)[0]
    assert stored.checked_ts == T0 + 3 * MIN
    run(broker, clock, T0 + 5 * MIN)
    assert repo_portfolio.open_positions(db)[0].checked_ts == T0 + 3 * MIN


def test_pending_manual_close_fills_before_a_later_stop(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 99.6})
    position = broker.detail("me").positions[0]
    clock.set(T0 + MIN + 30)
    broker.close_position("me", position.id)
    add_bar(db, "SAP.DE", T0 + MIN, 99.8, high=100.0, low=99.5)  # before the closing order: checked
    add_bar(db, "SAP.DE", T0 + 2 * MIN, 101.0, high=101.0, low=98.0)  # the close fills at its open first
    run(broker, clock, T0 + 3 * MIN)
    trade = broker.trades("me")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("manual", T0 + 2 * MIN)
    assert trade.exit_price == pytest.approx(101.0 * (1 - SLIP_EU))


def test_protective_exit_cancels_a_pending_close(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    opened_with_levels(broker, db, data, clock, {"high": 100.2, "low": 98.0})  # the entry bar hits the stop
    position = broker.detail("me").positions[0]
    clock.set(T0 + MIN + 30)
    close_order = broker.close_position("me", position.id)
    run(broker, clock, T0 + 2 * MIN)
    assert broker.trades("me")[0].exit_reason == "stop"
    out = order_by_id(broker, close_order.id)
    assert (out.status, out.status_reason) == ("cancelled", "Position bereits geschlossen")


# -- signal orders: time stop, flattening, drift, expiry ----------------------------------------------
def ai_position(broker: Broker, db: Database, clock: ManualClock, **signal_fields: object) -> None:
    """Fill a signal order for SAP (ref 100, stop distance 2, target distance 4) on the bar T0."""
    out = broker.execute_signal(make_signal(db, **signal_fields))  # type: ignore[arg-type]
    assert out is not None
    add_bar(db, "SAP.DE", T0, 100.0)
    run(broker, clock, T0 + MIN)
    assert broker.summary("ai").open_positions == 1


def test_signal_levels_are_relative_to_the_fill(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock)
    position = broker.detail("ai").positions[0]
    entry = 100.0 * (1 + SLIP_EU)
    assert position.stop_price == pytest.approx(entry - 2.0)
    assert position.target_price == pytest.approx(entry + 4.0)
    # 240 trading minutes: 90 until Friday's close at 16:30, the other 150 from Monday 08:00 on
    assert position.time_stop_ts == MONDAY_0000 + 10 * 3600 + 30 * MIN
    check_accounting(broker, "ai")


def test_time_stop(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock, horizon_min=30)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 29, 100.0)
    add_bar(db, "SAP.DE", T0 + 30 * MIN, 100.3)
    closed = run(broker, clock, T0 + 35 * MIN)
    assert len(closed) == 1
    trade = broker.trades("ai")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("time_stop", T0 + 30 * MIN)
    assert trade.exit_price == pytest.approx(100.3 * (1 - SLIP_EU))
    check_accounting(broker, "ai")


def test_session_flatten(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock, intraday=True)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 89, 100.0)  # until 16:29; Xetra closes at 16:30 in FakeCalendar
    run(broker, clock, T0 + 90 * MIN)
    trade = broker.trades("ai")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("session_close", T0 + 85 * MIN)  # 16:25 = close - 5 min
    assert trade.exit_price == pytest.approx(100.0 * (1 - SLIP_EU))


def test_overdue_flatten_without_bars(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock, intraday=True)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 70, 100.0, spread=0.2)  # the feed stops at 16:10
    session_end = T0 + 90 * MIN  # 16:30; the flatten deadline is 16:25
    # REST may deliver the minutes up to the close until the last one is settled: feed delay + settle time.
    settled = session_end + 15 * MIN + SETTLE_S
    run(broker, clock, settled - 1)
    assert broker.summary("ai").open_positions == 1
    closed = run(broker, clock, settled)  # 16:55, Xetra is closed and nothing can arrive any more
    assert len(closed) == 1
    trade = broker.trades("ai")[0]
    assert (trade.exit_reason, trade.closed_ts) == ("session_close", settled)
    assert trade.exit_price == pytest.approx(100.0 * (1 - SLIP_EU))  # close of the newest bar
    check_accounting(broker, "ai")


def test_overdue_flatten_waits_for_a_late_backfill(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock, intraday=True)
    add_flat_bars(db, "SAP.DE", T0 + MIN, 70, 100.0, spread=0.2)  # the feed stops at 16:10 (outage)
    run(broker, clock, T0 + 112 * MIN)  # 16:52: the old rule (deadline + delay + 10 min) would have flattened
    assert broker.summary("ai").open_positions == 1
    add_flat_bars(db, "SAP.DE", T0 + 71 * MIN, 19, 100.4, spread=0.2)  # REST fills 16:11 .. 16:29 late
    closed = run(broker, clock, T0 + 113 * MIN)
    assert len(closed) == 1
    trade = broker.trades("ai")[0]
    # flattened on the historical 16:25 bar at its open, exactly as if the data had arrived in time
    assert (trade.exit_reason, trade.closed_ts) == ("session_close", T0 + 85 * MIN)
    assert trade.exit_price == pytest.approx(100.4 * (1 - SLIP_EU))


def test_overdue_rule_waits_while_the_venue_is_open(broker: Broker, db: Database, clock: ManualClock) -> None:
    ai_position(broker, db, clock, horizon_min=10)  # time stop 15:10 while Xetra stays open until 16:30
    run(broker, clock, T0 + 60 * MIN)
    assert broker.summary("ai").open_positions == 1


def test_drift_rejection(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db))
    assert out is not None
    add_bar(db, "SAP.DE", T0, 101.5)  # more than half the stop distance (1.0) away from 100
    run(broker, clock, T0 + MIN)
    rejected = order_by_id(broker, out.id, "ai")
    assert rejected.status == "rejected"
    assert rejected.status_reason.startswith("Kurs zu weit gelaufen")
    assert broker.summary("ai").cash == START


def test_drift_within_the_limit_fills(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db))
    assert out is not None
    add_bar(db, "SAP.DE", T0, 100.9)
    run(broker, clock, T0 + MIN)
    assert order_by_id(broker, out.id, "ai").status == "filled"


def test_signal_order_expires(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db))
    deadline = T0 + 25 * MIN  # the signal's entry deadline: Xetra delay 15 + entry grace 10 minutes
    assert out is not None and out.expire_ts == deadline
    # No bar has arrived; one opening up to the deadline may still come until the deadline minute is settled.
    settled = deadline + MIN + 15 * MIN + SETTLE_S
    run(broker, clock, settled - 1)
    assert order_by_id(broker, out.id, "ai").status == "pending"
    run(broker, clock, settled)
    expired = order_by_id(broker, out.id, "ai")
    assert (expired.status, expired.status_reason) == ("cancelled", "abgelaufen")
    assert "abgelaufen" in repo_ops.list_activity(db, category="sim")[0].message


def test_bar_after_the_expiry_does_not_fill(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db))
    assert out is not None
    add_bar(db, "SAP.DE", T0 + 26 * MIN, 100.0)  # e.g. after a restart: the order was already dead
    run(broker, clock, T0 + 27 * MIN)
    assert order_by_id(broker, out.id, "ai").status == "pending"  # the minutes before it may still arrive
    run(broker, clock, T0 + 26 * MIN + 15 * MIN + SETTLE_S)
    assert order_by_id(broker, out.id, "ai").status_reason == "abgelaufen"
    assert broker.summary("ai").open_positions == 0


def test_manual_order_expires_after_a_week(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("SAP.DE", 100.0, T0)
    order = place(broker, "SAP.DE")
    expire_ts = T0 + 7 * 86_400  # Friday 15:00, Xetra trades: the last minutes may still bring a fill bar
    run(broker, clock, expire_ts + 1)
    assert order_by_id(broker, order.id).status == "pending"
    run(broker, clock, expire_ts + MIN + 15 * MIN + SETTLE_S)
    assert order_by_id(broker, order.id).status_reason == "abgelaufen"


def test_manual_order_expires_at_once_while_the_venue_is_closed(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    friday_night = T0 + 7 * 3600  # 22:00 UTC: the order's week ends on the next Friday night, Nasdaq closed
    clock.set(friday_night)
    data.set_quote("NVDA", 200.0, friday_night)
    order = place(broker, "NVDA")
    run(broker, clock, friday_night + 7 * 86_400 + 1)
    assert order_by_id(broker, order.id).status_reason == "abgelaufen"


def test_process_never_raises(broker: Broker, clock: ManualClock, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database gone")

    monkeypatch.setattr(repo_portfolio, "open_positions", broken)
    assert broker.process(from_ts(T0 + MIN)) == []
