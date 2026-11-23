"""Watchlist endpoints."""

import time

from fastapi.testclient import TestClient

from trading_local.db import repo_market
from trading_local.models import Instrument
from trading_local.services import Services

from .conftest import NOW_TS, Fakes, insert_signal, quote


def test_list_shows_quote_session_phase_and_the_newest_open_actionable_signal(
    client: TestClient, services: Services, fakes: Fakes
) -> None:
    fakes.data.quote_book["NVDA"] = quote("NVDA", 181.5, NOW_TS - 5)
    db = services.db
    insert_signal(db, "NVDA", status="active", actionable=True, created_ts=NOW_TS - 900)
    newest_actionable = insert_signal(db, "NVDA", status="pending_entry", actionable=True, created_ts=NOW_TS - 600)
    insert_signal(db, "NVDA", status="active", actionable=False, created_ts=NOW_TS - 300)  # shadow
    insert_signal(db, "NVDA", status="target", actionable=True, created_ts=NOW_TS - 200)  # closed
    insert_signal(db, "SAP.DE", status="review", actionable=False, created_ts=NOW_TS - 100)

    body = client.get("/api/instruments").json()
    by_symbol = {entry["symbol"]: entry for entry in body}
    assert [entry["symbol"] for entry in body][:3] == ["NVDA", "SAP.DE", "SHEL.L"]  # sort order
    nvda = by_symbol["NVDA"]
    assert nvda["quote"]["price"] == 181.5
    assert (nvda["phase"], nvda["is_open"], nvda["exchange_name"]) == ("open", True, "Nasdaq")
    assert nvda["open_signal"]["id"] == newest_actionable.id
    assert by_symbol["SAP.DE"]["open_signal"] is None
    assert by_symbol["7203.T"]["phase"] == "closed" and by_symbol["7203.T"]["quote"] is None


def test_inactive_instruments_are_listed_on_request(client: TestClient) -> None:
    assert client.patch("/api/instruments/eurjpy=x", json={"active": False}).json()["active"] is False
    assert "EURJPY=X" not in {entry["symbol"] for entry in client.get("/api/instruments").json()}
    listed = client.get("/api/instruments", params={"include_inactive": "true"}).json()
    assert {entry["symbol"]: entry["active"] for entry in listed}["EURJPY=X"] is False
    assert client.get("/api/instruments", params={"include_inactive": "vielleicht"}).status_code == 422


def test_patch_changes_the_flags_and_reloads_the_book(client: TestClient, services: Services) -> None:
    response = client.patch("/api/instruments/NVDA", json={"tradable": False})
    assert response.status_code == 200
    assert (response.json()["tradable"], response.json()["active"]) == (False, True)
    assert services.instruments.require("NVDA").tradable is False
    assert client.patch("/api/instruments/NVDA", json={}).json()["tradable"] is False


def test_conversion_rates_and_futures_cannot_be_made_tradable(client: TestClient, services: Services) -> None:
    repo_market.upsert_instruments(
        services.db, [Instrument("ES=F", "E-mini S&P 500", "future", "US", "CME", "USD", tradable=False)]
    )
    services.instruments.reload()
    response = client.patch("/api/instruments/eurusd=x", json={"tradable": True})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("EURUSD=X (EUR/USD) ist ein Umrechnungskurs")
    response = client.patch("/api/instruments/ES=F", json={"tradable": True, "active": False})
    assert response.status_code == 400 and "ein Vergleichswert" in response.json()["detail"]
    for symbol in ("EURUSD=X", "ES=F"):  # nothing was stored, the engine never sees them as tradable
        assert repo_market.get_instrument(services.db, symbol).tradable is False  # type: ignore[union-attr]
        assert symbol not in {instrument.symbol for instrument in services.instruments.tradable()}
    assert repo_market.get_instrument(services.db, "ES=F").active is True  # type: ignore[union-attr]
    # switching off and the other flag stay possible
    assert client.patch("/api/instruments/EURUSD=X", json={"tradable": False, "active": False}).status_code == 200
    assert client.patch("/api/instruments/^GDAXI", json={"tradable": True}).status_code == 200  # indices trade


def test_patch_errors(client: TestClient) -> None:
    response = client.patch("/api/instruments/NOPE", json={"active": True})
    assert response.status_code == 404
    assert response.json()["detail"] == "Unbekanntes Symbol: NOPE"
    response = client.patch("/api/instruments/NVDA", json={"colour": "blue"})
    assert response.status_code == 422
    assert "Feld 'colour' ist kein bekanntes Feld" in response.json()["detail"]


def test_add_validates_at_the_data_source_and_downloads_the_history(
    client: TestClient, services: Services, fakes: Fakes
) -> None:
    fakes.data.lookups["ASML.AS"] = Instrument(
        "ASML.AS", "ASML Holding", "equity", "EU", "XAMS", "EUR", timezone="Europe/Amsterdam", sort_order=20
    )
    response = client.post("/api/instruments", json={"symbol": " asml.as "})
    assert response.status_code == 201
    body = response.json()
    assert (body["symbol"], body["name"], body["exchange"]) == ("ASML.AS", "ASML Holding", "XAMS")
    assert repo_market.get_instrument(services.db, "ASML.AS") is not None
    assert "ASML.AS" in services.instruments
    deadline = time.monotonic() + 2.0
    while not fakes.data.history_requests and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fakes.data.history_requests == ["ASML.AS"]


def test_add_rejects_known_and_unknown_symbols(client: TestClient, fakes: Fakes) -> None:
    response = client.post("/api/instruments", json={"symbol": "nvda"})
    assert response.status_code == 409
    assert response.json()["detail"] == "Das Symbol NVDA ist bereits in der Liste."
    response = client.post("/api/instruments", json={"symbol": "XYZ123"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Symbol bei der Datenquelle nicht gefunden"
    fakes.data.lookup_error = ConnectionError("network down")
    response = client.post("/api/instruments", json={"symbol": "XYZ123"})
    assert response.status_code == 502
    assert "Datenquelle" in response.json()["detail"]
    assert fakes.data.history_requests == []


def test_add_validates_the_body(client: TestClient) -> None:
    assert "Feld 'symbol' fehlt" in client.post("/api/instruments", json={}).json()["detail"]
    assert client.post("/api/instruments", json={"symbol": "   "}).status_code == 422
    response = client.post("/api/instruments", json={"symbol": "X" * 25})
    assert response.status_code == 422
    assert "zu lang (höchstens 24 Zeichen)" in response.json()["detail"]
