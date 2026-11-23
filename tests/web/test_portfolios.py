"""Paper portfolios: reading both, trading only the user's own, resetting with confirmation."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from .conftest import NOW_TS, ORIGIN, Fakes
from .fakes import REJECTED_SYMBOL, REJECTION_TEXT


def test_summaries_list_the_ai_portfolio_first(client: TestClient) -> None:
    assert [entry["key"] for entry in client.get("/api/portfolios").json()] == ["ai", "me"]


def test_equity_curves(client: TestClient, fakes: Fakes) -> None:
    body = client.get("/api/portfolios/equity", params={"since_ts": NOW_TS - 3600, "max_points": 50}).json()
    assert body["ai"] == [[NOW_TS, 100_500.0]]
    assert fakes.broker.calls[-1] == ("equity", NOW_TS - 3600, 50)
    client.get("/api/portfolios/equity")
    assert fakes.broker.calls[-1] == ("equity", None, 600)
    for params in ({"max_points": 1}, {"max_points": 5001}, {"since_ts": -1}):
        assert client.get("/api/portfolios/equity", params=params).status_code == 422


@pytest.mark.parametrize("key", ["ai", "me"])
def test_detail_trades_and_orders(client: TestClient, fakes: Fakes, key: str) -> None:
    detail = client.get(f"/api/portfolios/{key}").json()
    assert detail["summary"]["key"] == key and detail["positions"][0]["id"] == 7
    assert client.get(f"/api/portfolios/{key}/trades", params={"limit": 10}).json()[0]["status"] == "closed"
    assert client.get(f"/api/portfolios/{key}/orders").json()[0]["id"] == 5
    assert ("trades", key, 10) in fakes.broker.calls and ("orders", key, 200) in fakes.broker.calls
    assert client.get(f"/api/portfolios/{key}/orders", params={"limit": 1001}).status_code == 422


@pytest.mark.parametrize("path", ["/api/portfolios/xyz", "/api/portfolios/xyz/trades", "/api/portfolios/xyz/orders"])
def test_unknown_portfolios_are_404(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["detail"] == "Unbekanntes Portfolio: xyz"


def test_place_order_in_the_user_portfolio(client: TestClient, fakes: Fakes) -> None:
    response = client.post(
        "/api/portfolios/me/orders", json={"symbol": "sap.de", "side": "LONG", "amount_base": 5000.0},
        headers={"origin": ORIGIN},
    )
    assert response.status_code == 201
    assert response.json()["symbol"] == "SAP.DE"
    _, key, request = fakes.broker.calls[-1]
    assert (key, request.symbol, request.amount_base) == ("me", "SAP.DE", 5000.0)


def test_rejected_orders_are_400_with_the_broker_text(client: TestClient) -> None:
    response = client.post("/api/portfolios/me/orders", json={"symbol": REJECTED_SYMBOL, "side": "SHORT", "qty": 3})
    assert response.status_code == 400
    assert response.json() == {"detail": REJECTION_TEXT}
    response = client.delete("/api/portfolios/me/orders/404")
    assert response.status_code == 400
    assert response.json()["detail"] == "Die Order 404 ist nicht mehr offen."


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/portfolios/ai/orders", {"symbol": "NVDA", "side": "LONG", "qty": 1}),
        ("DELETE", "/api/portfolios/ai/orders/5", None),
        ("POST", "/api/portfolios/ai/positions/7/close", None),
        ("PATCH", "/api/portfolios/ai/positions/7", {"stop_price": 170.0}),
    ],
)
def test_the_ai_portfolio_cannot_be_traded(
    client: TestClient, fakes: Fakes, method: str, path: str, body: dict[str, Any] | None
) -> None:
    response = client.request(method, path, json=body)
    assert response.status_code == 403
    assert "KI-Portfolio" in response.json()["detail"]
    assert fakes.broker.calls == []
    assert client.request(method, path.replace("/ai/", "/xyz/"), json=body).status_code == 404


def test_cancel_close_and_update(client: TestClient, fakes: Fakes) -> None:
    assert client.delete("/api/portfolios/me/orders/5").json()["status"] == "cancelled"
    assert client.post("/api/portfolios/me/positions/7/close").json()["intent"] == "close"
    response = client.patch("/api/portfolios/me/positions/7", json={"stop_price": 170.0, "target_price": None})
    assert (response.json()["stop_price"], response.json()["target_price"]) == (170.0, None)
    assert fakes.broker.calls == [("cancel", "me", 5), ("close", "me", 7), ("update", "me", 7, 170.0, None)]
    response = client.patch("/api/portfolios/me/positions/7", json={"stop_price": 200.0, "target_price": 190.0})
    assert response.status_code == 400 and "Stopp" in response.json()["detail"]


def test_order_bodies_are_validated(client: TestClient) -> None:
    response = client.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "UP", "qty": 1})
    assert response.status_code == 422 and "Feld 'side'" in response.json()["detail"]
    response = client.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "LONG", "qty": -1})
    assert response.status_code == 422 and "muss größer als 0 sein" in response.json()["detail"]
    assert client.patch("/api/portfolios/me/positions/7", json={"stop": 1}).status_code == 422
    assert client.delete("/api/portfolios/me/orders/abc").status_code == 422


def test_reset_requires_confirmation(client: TestClient, fakes: Fakes) -> None:
    response = client.post("/api/portfolios/reset", json={"confirm": False})
    assert response.status_code == 400
    assert "bestätigen" in response.json()["detail"]
    assert client.post("/api/portfolios/reset", json={}).status_code == 422
    response = client.post("/api/portfolios/reset", json={"confirm": True, "start_capital": 50_000})
    assert response.status_code == 200
    assert [entry["key"] for entry in response.json()] == ["ai", "me"]
    assert fakes.broker.calls == [("reset", 50_000.0)]
