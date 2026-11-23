"""Manual orders: placement, validation (every OrderRejected branch), cancel, close and level updates."""

import pytest

from trading_local.clock import ManualClock
from trading_local.contracts import Broker, OrderRejected
from trading_local.db import repo_market, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import OrderRequest
from trading_local.events import TOPIC_ORDER, TOPIC_PORTFOLIO
from tests.sim.simkit import MIN, T0, FakeMarketData, RecordingBus, add_bar, make_signal, open_position


def request(symbol: str = "SAP.DE", side: str = "LONG", **fields: object) -> OrderRequest:
    if "qty" not in fields and "amount_base" not in fields:
        fields["qty"] = 10
    return OrderRequest(symbol=symbol, side=side, **fields)  # type: ignore[arg-type]


def rejected(broker: Broker, req: OrderRequest, key: str = "me") -> str:
    with pytest.raises(OrderRejected) as info:
        broker.place_order(key, req)
    return str(info.value)


@pytest.fixture(autouse=True)
def sap_quote(data: FakeMarketData) -> None:
    data.set_quote("SAP.DE", 100.0, T0)


# -- placement --------------------------------------------------------------------------------------
def test_place_order_stores_a_pending_manual_order(broker: Broker, bus: RecordingBus) -> None:
    out = broker.place_order("me", request(stop_price=95.0, target_price=110.0))
    assert (out.status, out.origin, out.intent, out.side) == ("pending", "manual", "open", "LONG")
    assert (out.portfolio_key, out.symbol, out.name) == ("me", "SAP.DE", "SAP")
    assert out.qty == 10 and out.notional_base is None
    assert out.ref_price == 100.0
    assert (out.stop_price, out.target_price) == (95.0, 110.0)
    assert out.created_ts == T0 and out.expire_ts == T0 + 7 * 86_400
    assert bus.of(TOPIC_ORDER) == [out.model_dump()]
    assert bus.of(TOPIC_PORTFOLIO) == [{"key": "me"}]
    assert broker.orders("me") == [out]
    assert broker.summary("me").pending_orders == 1


def test_amount_order_keeps_the_notional(broker: Broker) -> None:
    out = broker.place_order("me", request(amount_base=2500.0))
    assert out.qty is None and out.notional_base == 2500.0


def test_quantity_is_floored_to_the_lot(broker: Broker) -> None:
    assert broker.place_order("me", request(qty=7.9)).qty == 7.0


def test_last_price_falls_back_to_the_newest_closed_bar(broker: Broker, db: Database, data: FakeMarketData) -> None:
    data.drop_quote("SAP.DE")
    add_bar(db, "SAP.DE", T0 - 2 * MIN, 98.0, close=99.0)
    add_bar(db, "SAP.DE", T0 - MIN, 99.0, close=97.0)  # closes exactly at T0: known
    assert broker.place_order("me", request()).ref_price == 97.0


def test_signal_id_is_passed_through(broker: Broker, db: Database) -> None:
    signal = make_signal(db)
    assert broker.place_order("me", request(signal_id=signal.id)).signal_id == signal.id


# -- rejections -------------------------------------------------------------------------------------
def test_unknown_portfolio(broker: Broker) -> None:
    assert "Unbekanntes Depot 'xx'" in rejected(broker, request(), key="xx")


def test_unknown_symbol(broker: Broker) -> None:
    assert rejected(broker, request("XYZ")) == "Das Symbol XYZ ist unbekannt. Füge es zuerst der Watchlist hinzu."


def test_inactive_symbol(broker: Broker, db: Database, instruments: InstrumentBook, data: FakeMarketData) -> None:
    repo_market.set_instrument_flags(db, "NVDA", active=False)
    instruments.reload()
    data.set_quote("NVDA", 200.0, T0)
    assert "NVIDIA ist deaktiviert" in rejected(broker, request("NVDA"))


@pytest.mark.parametrize("symbol", ["EURUSD=X", "ES=F"])
def test_fx_pairs_and_futures_are_not_tradable(broker: Broker, data: FakeMarketData, symbol: str) -> None:
    data.set_quote(symbol, 1.1, T0)
    assert "Devisenpaare und Futures dienen nur als Marktinformation" in rejected(broker, request(symbol))


def test_symbol_marked_untradable(broker: Broker, db: Database, instruments: InstrumentBook) -> None:
    repo_market.set_instrument_flags(db, "SAP.DE", tradable=False)
    instruments.reload()
    assert "SAP ist als nicht handelbar markiert" in rejected(broker, request())


def test_exactly_one_of_amount_and_quantity(broker: Broker) -> None:
    message = "Gib entweder einen Betrag oder eine Stückzahl an – genau eines von beiden."
    assert rejected(broker, OrderRequest(symbol="SAP.DE", side="LONG")) == message
    assert rejected(broker, request(qty=1, amount_base=100.0)) == message


def test_open_position_blocks_a_new_order(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    open_position(broker, db, data, clock, "SAP.DE", qty=5)
    assert rejected(broker, request()) == "Für SAP besteht bereits eine Long-Position. Schließe sie zuerst."
    assert "Short-Position" not in rejected(broker, request(side="SHORT"))


def test_pending_order_blocks_a_second_one(broker: Broker) -> None:
    broker.place_order("me", request())
    assert rejected(broker, request(side="SHORT")).startswith("Für SAP liegt bereits eine offene Order vor.")
    broker.place_order("ai", request())  # the other portfolio is independent


@pytest.mark.parametrize(
    ("side", "levels", "fragment"),
    [
        ("LONG", {"stop_price": 100.0}, "Der Stopp (100,00) muss bei einer Long-Position unter dem aktuellen Kurs"),
        ("LONG", {"target_price": 99.0}, "Das Ziel (99,00) muss bei einer Long-Position über dem aktuellen Kurs"),
        ("SHORT", {"stop_price": 99.5}, "Der Stopp (99,50) muss bei einer Short-Position über dem aktuellen Kurs"),
        ("SHORT", {"target_price": 101.0}, "Das Ziel (101,00) muss bei einer Short-Position unter dem aktuellen Kurs"),
    ],
)
def test_levels_on_the_wrong_side(broker: Broker, side: str, levels: dict[str, float], fragment: str) -> None:
    message = rejected(broker, request(side=side, **levels))
    assert fragment in message and "von SAP (100,00)" in message


def test_levels_on_the_right_side_are_accepted(broker: Broker) -> None:
    out = broker.place_order("me", request(side="SHORT", stop_price=104.0, target_price=90.0))
    assert out.status == "pending"


def test_no_price_known(broker: Broker) -> None:
    message = rejected(broker, request("^GSPC"))
    assert message == "Für S&P 500 ist noch kein Kurs bekannt. Warte, bis Kursdaten eingetroffen sind."


def test_no_fx_rate(broker: Broker, data: FakeMarketData) -> None:
    data.set_quote("7203.T", 3000.0, T0)
    data.drop_quote("EURJPY=X")
    assert rejected(broker, request("7203.T")).startswith("Für JPY → EUR ist kein Wechselkurs bekannt")


def test_order_value_above_buying_power(broker: Broker) -> None:
    message = rejected(broker, request(amount_base=150_000.0))
    assert message.startswith("Der Auftragswert (150.000,00 EUR) übersteigt die verfügbare Kaufkraft (100.000,00 EUR).")
    rejected(broker, request(qty=1001))  # 100,100 EUR at 100 EUR per share
    broker.place_order("me", request(qty=1000))  # exactly the buying power


def test_buying_power_follows_the_leverage_setting(broker: Broker, runtime_settings: SettingsStore) -> None:
    runtime_settings.update({"max_gross_leverage": 2.0}, T0)
    assert broker.place_order("me", request(amount_base=150_000.0)).status == "pending"


def test_quantity_below_one_lot(broker: Broker) -> None:
    message = rejected(broker, request(qty=0.5))
    assert message == "Die Stückzahl 0,5 ist kleiner als die kleinste handelbare Einheit von SAP (1)."


def test_amount_below_one_unit(broker: Broker) -> None:
    assert rejected(broker, request(amount_base=50.0)).startswith("Der Betrag von 50,00 EUR reicht nicht für eine Einheit von SAP")


def test_unknown_signal(broker: Broker) -> None:
    assert rejected(broker, request(signal_id=4711)) == "Das Signal #4711 wurde nicht gefunden."


def test_rejections_store_nothing(broker: Broker, bus: RecordingBus) -> None:
    rejected(broker, request(amount_base=150_000.0))
    assert broker.orders("me") == [] and bus.events == []


# -- cancel -----------------------------------------------------------------------------------------
def test_cancel_order(broker: Broker, bus: RecordingBus) -> None:
    order = broker.place_order("me", request())
    bus.events.clear()
    out = broker.cancel_order("me", order.id)
    assert (out.status, out.status_reason) == ("cancelled", "storniert")
    assert bus.of(TOPIC_ORDER) == [out.model_dump()] and bus.of(TOPIC_PORTFOLIO) == [{"key": "me"}]
    with pytest.raises(OrderRejected, match="kann nicht mehr storniert werden \\(Status: storniert\\)"):
        broker.cancel_order("me", order.id)


def test_cancel_unknown_or_foreign_order(broker: Broker) -> None:
    order = broker.place_order("me", request())
    with pytest.raises(OrderRejected, match=f"Die Order #{order.id} wurde in diesem Depot nicht gefunden."):
        broker.cancel_order("ai", order.id)
    with pytest.raises(OrderRejected, match="Die Order #999 wurde in diesem Depot nicht gefunden."):
        broker.cancel_order("me", 999)


# -- close ------------------------------------------------------------------------------------------
def test_close_position_places_a_pending_closing_order(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position = open_position(broker, db, data, clock, "SAP.DE", qty=5)
    data.set_quote("SAP.DE", 101.0, clock.ts())
    out = broker.close_position("me", position.id)
    assert (out.intent, out.origin, out.status, out.side) == ("close", "manual", "pending", "LONG")
    assert (out.qty, out.position_id, out.ref_price) == (5.0, position.id, 101.0)
    with pytest.raises(OrderRejected, match="Für die Position in SAP liegt bereits eine Schließungs-Order vor."):
        broker.close_position("me", position.id)


def test_close_position_must_be_open_and_owned(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock) -> None:
    position = open_position(broker, db, data, clock, "SAP.DE", qty=5)
    with pytest.raises(OrderRejected, match=f"Die Position #{position.id} ist in diesem Depot nicht offen."):
        broker.close_position("ai", position.id)
    with pytest.raises(OrderRejected, match="Die Position #77 ist in diesem Depot nicht offen."):
        broker.close_position("me", 77)


# -- update levels ----------------------------------------------------------------------------------
def test_update_position_levels(broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock, bus: RecordingBus) -> None:
    position = open_position(broker, db, data, clock, "SAP.DE", "SHORT", qty=5)
    bus.events.clear()
    out = broker.update_position("me", position.id, stop_price=105.0, target_price=90.0)
    assert (out.stop_price, out.target_price, out.status) == (105.0, 90.0, "open")
    assert out.last_price == 100.0
    assert bus.of(TOPIC_PORTFOLIO) == [{"key": "me"}]
    removed = broker.update_position("me", position.id, stop_price=None, target_price=None)
    assert (removed.stop_price, removed.target_price) == (None, None)


def test_update_position_validates_against_the_last_price(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position = open_position(broker, db, data, clock, "SAP.DE", qty=5)
    data.set_quote("SAP.DE", 104.0, clock.ts())
    with pytest.raises(OrderRejected, match="Der Stopp \\(105,00\\) muss bei einer Long-Position unter"):
        broker.update_position("me", position.id, stop_price=105.0, target_price=None)
    with pytest.raises(OrderRejected, match="Der Stopp muss ein positiver Kurs sein."):
        broker.update_position("me", position.id, stop_price=-1.0, target_price=None)
    stored = broker.detail("me").positions[0]
    assert stored.stop_price is None


def test_activity_log_is_untouched_by_placements(broker: Broker, db: Database) -> None:
    broker.place_order("me", request())
    assert repo_ops.list_activity(db, category="sim") == []
