"""AI portfolio: risk-based sizing, caps and every skip reason of ``execute_signal``."""

import pytest

from trading_local.analysis import tracking
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.clock import ManualClock
from trading_local.contracts import Broker
from trading_local.db import repo_ops, repo_portfolio
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import TOPIC_ACTIVITY, TOPIC_ORDER, TOPIC_PORTFOLIO
from trading_local.models import Instrument, Signal
from trading_local.sim.fx import FxTable
from trading_local.testing import FakeCalendar, seed_instruments
from tests.sim.simkit import MIN, T0, FakeMarketData, RecordingBus, add_bar, make_signal


def last_sim_message(db: Database) -> str:
    return repo_ops.list_activity(db, category="sim")[0].message


def assert_skipped(broker: Broker, db: Database, signal: Signal, reason: str) -> None:
    assert broker.execute_signal(signal) is None
    message = last_sim_message(db)
    assert message.startswith(f"Depot KI: Signal #{signal.id} ({signal.symbol} ")
    assert reason in message, message


# -- sizing -----------------------------------------------------------------------------------------
def test_order_is_sized_by_risk(broker: Broker, db: Database, bus: RecordingBus) -> None:
    signal = make_signal(db, stop_dist=10.0, target_dist=20.0, horizon_min=30, intraday=True)
    out = broker.execute_signal(signal)
    assert out is not None
    # 0.5 % of 100,000 EUR = 500 EUR risk / 10 EUR per share
    assert out.qty == 50.0
    assert (out.portfolio_key, out.origin, out.intent, out.side, out.status) == ("ai", "signal", "open", "LONG", "pending")
    assert (out.ref_price, out.signal_id, out.created_ts) == (100.0, signal.id, T0)
    assert out.expire_ts == T0 + (15 + 10) * MIN  # tracking.entry_deadline: feed delay + entry grace
    stored = next(o for o in broker.orders("ai") if o.id == out.id)
    assert stored == out
    assert bus.of(TOPIC_ORDER) == [out.model_dump()]
    assert {"key": "ai"} in bus.of(TOPIC_PORTFOLIO)
    activity = bus.of(TOPIC_ACTIVITY)[0]
    assert activity["category"] == "sim" and "Kauf 50 SAP" in activity["message"]
    assert "(Risiko 0,50 % des Depotwerts)" in activity["message"]


def test_order_carries_the_signal_rules(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db, stop_dist=10.0, target_dist=20.0, horizon_min=30, intraday=True))
    assert out is not None
    add_bar(db, "SAP.DE", T0, 100.0)
    clock.set(T0 + MIN)
    broker.process(clock.now())
    position = broker.detail("ai").positions[0]
    assert position.time_stop_ts == T0 + 30 * MIN  # add_trading_minutes(XETR, now, 30)
    assert position.flatten_at_close is True
    assert position.signal_id == out.signal_id


def test_reduce_verdict_halves_the_risk(broker: Broker, db: Database) -> None:
    out = broker.execute_signal(make_signal(db, stop_dist=10.0, llm_verdict="reduce"))
    assert out is not None and out.qty == 25.0
    message = last_sim_message(db)
    assert "(Risiko 0,25 % des Depotwerts, Positionsgröße nach KI-Prüfung auf 50 % reduziert)" in message


def test_reduce_verdict_also_halves_a_capped_position(broker: Broker, db: Database) -> None:
    # A typical intraday stop of 0.6 %: the risk alone would allow 833 SAP, the 10 % position cap allows 100.
    # 'reduce' must halve the size that is actually traded, not only the risk before the cap.
    full = broker.execute_signal(make_signal(db, stop_dist=0.6))
    assert full is not None and full.qty == 100.0
    assert "(Risiko 0,06 % des Depotwerts)" in last_sim_message(db)  # the risk really taken, not 0,50 %
    # NVDA at 100 USD = 80 EUR: the cap allows 125 shares, 'reduce' trades half of them
    half = broker.execute_signal(make_signal(db, "NVDA", stop_dist=0.6, llm_verdict="reduce"))
    assert half is not None and half.qty == 62.0
    assert "(Risiko 0,03 % des Depotwerts, Positionsgröße nach KI-Prüfung auf 50 % reduziert)" in last_sim_message(db)


def test_order_fees_that_eat_the_expected_value_skip_the_signal(broker: Broker, db: Database) -> None:
    broker.reset(10_000.0)
    # 10 % cap: 10 SAP for 1,000 EUR, planned risk 10 * 0.57 = 5.70 EUR; 2 * 1 EUR fees = 0.35 R
    signal = make_signal(db, stop_dist=0.57, target_dist=1.14, expected_r=0.30)
    assert_skipped(
        broker, db, signal, "die Ordergebühren für Ein- und Ausstieg (0,35 R) zehren den Erwartungswert (0,30 R) auf"
    )
    assert broker.orders("ai") == []
    out = broker.execute_signal(make_signal(db, stop_dist=0.57, target_dist=1.14, expected_r=0.40))
    assert out is not None and out.qty == 10.0


def test_kraken_signals_pay_no_flat_order_fee(broker: Broker, db: Database, instruments: InstrumentBook) -> None:
    """KRAKEN pairs carry their taker fee in the slippage: the flat-fee check of execute_signal never skips them."""
    pair = Instrument("ETH-EUR", "Ethereum", "crypto", "GLOBAL", "KRAKEN", "EUR", timezone="UTC")
    seed_instruments(db, [pair])
    instruments.reload()
    broker.reset(10_000.0)
    # the same small trade that the 2 x 1 EUR flat fee skips for SAP (0.35 R > 0.30 R) is placed for a Kraken pair
    out = broker.execute_signal(make_signal(db, "ETH-EUR", stop_dist=0.57, target_dist=1.14, expected_r=0.05))
    assert out is not None and out.status == "pending"


def test_signal_order_follows_the_tracked_signal(
    broker: Broker, db: Database, clock: ManualClock, instruments: InstrumentBook
) -> None:
    signal = make_signal(db, stop_dist=2.0)
    sap = instruments.get("SAP.DE")
    assert sap is not None
    tracking.issue(signal, issued_ts=T0 - 1, instrument=sap)  # issued a moment before the order is placed
    clock.set(T0 + 1)
    out = broker.execute_signal(signal)
    assert out is not None
    # same first bar, same entry deadline as the signal journal
    assert (out.created_ts, out.expire_ts) == (T0 - 1, signal.entry_deadline_ts)
    stored = repo_portfolio.get_order(db, out.id)
    assert stored is not None and stored.max_drift == pytest.approx(ENGINE_PARAMS.max_entry_drift * 2.0)


def test_usd_instrument_is_sized_in_base_currency(broker: Broker, db: Database) -> None:
    # 20 USD stop = 16 EUR per share -> 500 / 16 = 31.25 shares
    out = broker.execute_signal(make_signal(db, "NVDA", ref_price=200.0, stop_dist=20.0))
    assert out is not None and out.qty == 31.0


def test_position_value_cap(broker: Broker, db: Database) -> None:
    # risk alone: 500 shares; 10 % of the equity: 100 shares
    out = broker.execute_signal(make_signal(db, stop_dist=1.0))
    assert out is not None and out.qty == 100.0


def test_buying_power_cap(broker: Broker, db: Database, runtime_settings: SettingsStore) -> None:
    runtime_settings.update({"max_gross_leverage": 0.03}, T0)  # 3,000 EUR buying power
    out = broker.execute_signal(make_signal(db, stop_dist=10.0))
    assert out is not None and out.qty == 30.0


def test_short_signal(broker: Broker, db: Database, clock: ManualClock) -> None:
    out = broker.execute_signal(make_signal(db, direction="SHORT", stop_dist=10.0, target_dist=20.0))
    assert out is not None and out.side == "SHORT"
    add_bar(db, "SAP.DE", T0, 100.0)
    clock.set(T0 + MIN)
    broker.process(clock.now())
    position = broker.detail("ai").positions[0]
    assert position.stop_price == pytest.approx(position.entry_price + 10.0)
    assert position.target_price == pytest.approx(position.entry_price - 20.0)


# -- skip reasons -----------------------------------------------------------------------------------
def test_skip_not_actionable(broker: Broker, db: Database) -> None:
    assert_skipped(broker, db, make_signal(db, actionable=False), "nicht zum Handel freigegeben")


def test_skip_wrong_status(broker: Broker, db: Database) -> None:
    assert_skipped(broker, db, make_signal(db, status="review"), "nicht zum Handel freigegeben (Status: review)")


def test_skip_unknown_and_untradable_symbols(broker: Broker, db: Database) -> None:
    unknown = Signal(T0, T0, "XYZ", "LONG", "breakout", "15m", 60, 10.0, 1.0, 1.0, 2.0, 0.7, 0.6, "test",
                     status="pending_entry", actionable=True)
    assert broker.execute_signal(unknown) is None
    assert "XYZ ist unbekannt" in last_sim_message(db)
    assert_skipped(broker, db, make_signal(db, "EURUSD=X", ref_price=1.1, stop_dist=0.01), "EUR/USD ist nicht handelbar")


def test_skip_existing_position(broker: Broker, db: Database, clock: ManualClock) -> None:
    assert broker.execute_signal(make_signal(db)) is not None
    add_bar(db, "SAP.DE", T0, 100.0)
    clock.set(T0 + MIN)
    broker.process(clock.now())
    assert_skipped(broker, db, make_signal(db, direction="SHORT"), "für SAP besteht bereits eine Long-Position")


def test_skip_pending_order(broker: Broker, db: Database) -> None:
    assert broker.execute_signal(make_signal(db)) is not None
    assert_skipped(broker, db, make_signal(db), "für SAP liegt bereits eine offene Order vor")


def test_skip_when_max_open_positions_reached(broker: Broker, db: Database, runtime_settings: SettingsStore) -> None:
    runtime_settings.update({"max_open_positions": 1}, T0)
    assert broker.execute_signal(make_signal(db)) is not None  # a pending order occupies the slot
    assert_skipped(broker, db, make_signal(db, "NVDA", ref_price=200.0), "die Höchstzahl von 1 Positionen")


def test_skip_without_fx_rate(broker: Broker, db: Database, data: FakeMarketData) -> None:
    data.drop_quote("EURJPY=X")
    signal = make_signal(db, "7203.T", ref_price=3000.0, stop_dist=30.0)
    assert_skipped(broker, db, signal, "kein Wechselkurs JPY → EUR bekannt")


def test_skip_without_price(broker: Broker, db: Database) -> None:
    assert_skipped(broker, db, make_signal(db, ref_price=0.0), "kein gültiger Referenzkurs")


def test_skip_when_the_size_rounds_to_zero(broker: Broker, db: Database) -> None:
    # 10 % of the equity buys 0.01 shares at 1,000,000 EUR
    signal = make_signal(db, ref_price=1_000_000.0, stop_dist=1000.0)
    assert_skipped(broker, db, signal, "kleiner als eine handelbare Einheit")


def test_skip_unknown_trading_hours(
    broker: Broker, db: Database, calendar: FakeCalendar, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unknown(exchange: str, start: object, minutes: int) -> None:
        raise KeyError(exchange)

    monkeypatch.setattr(calendar, "add_trading_minutes", unknown)
    assert_skipped(broker, db, make_signal(db), "für den Handelsplatz XETR sind keine Handelszeiten bekannt")


def test_internal_errors_never_escape(broker: Broker, db: Database, fx: FxTable, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(fx, "rate", broken)
    signal = make_signal(db)
    assert broker.execute_signal(signal) is None
    entry = repo_ops.list_activity(db, category="sim")[0]
    assert entry.level == "error"
    assert "wegen eines internen Fehlers nicht gehandelt" in entry.message
    assert (entry.ref_type, entry.ref_id) == ("signal", signal.id)


def test_skips_store_no_order(broker: Broker, db: Database) -> None:
    broker.execute_signal(make_signal(db, actionable=False))
    assert broker.orders("ai") == []
