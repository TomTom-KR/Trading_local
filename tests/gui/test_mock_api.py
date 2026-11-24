"""The development mock server (tools/mock_api.py) implements the API contract the GUI is built against.

Every payload is validated against the pydantic models of trading_local.dto, so the mock (and with it the
GUI) cannot drift from the contract unnoticed.
"""

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import quote

import pytest
from pydantic import TypeAdapter
from starlette.testclient import TestClient

from trading_local import dto
from trading_local.config import RuntimeSettings

ROOT = Path(__file__).resolve().parents[2]
JSON = {"Content-Type": "application/json"}


def _load_mock() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mock_api", ROOT / "tools" / "mock_api.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["mock_api"] = module
    spec.loader.exec_module(module)
    return module


mock_api = _load_mock()


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Shared client for read-only checks (no background events)."""
    with TestClient(mock_api.create_app(live=False)) as c:
        yield c


@pytest.fixture
def fresh() -> Iterator[TestClient]:
    """Own client for tests that change state."""
    with TestClient(mock_api.create_app(live=False)) as c:
        yield c


READ_ENDPOINTS: list[tuple[str, Any]] = [
    ("/api/status", dto.SystemStatusOut),
    ("/api/sessions", dto.SessionsOut),
    ("/api/instruments?include_inactive=true", list[dto.InstrumentOut]),
    ("/api/quotes", list[dto.QuoteOut]),
    ("/api/sparklines", dto.SparklinesOut),
    ("/api/bars/NVDA?tf=1m&limit=50", dto.BarsOut),
    ("/api/bars/SAP.DE?tf=15m", dto.BarsOut),
    ("/api/bars/%5EGDAXI?tf=1h", dto.BarsOut),
    ("/api/bars/7203.T?tf=1d", dto.BarsOut),
    ("/api/analysis/NVDA", dto.AnalysisOut),
    ("/api/chart-analyses/NVDA?limit=2", list[dto.ChartAnalysisOut]),
    ("/api/news?limit=10&min_relevance=3", list[dto.NewsOut]),
    ("/api/agenda", list[dto.AgendaEventOut]),
    ("/api/agenda?min_importance=3&symbol=NVDA", list[dto.AgendaEventOut]),
    ("/api/signals?scope=open&actionable=true", list[dto.SignalOut]),
    ("/api/signals?scope=closed&actionable=false&limit=5", list[dto.SignalOut]),
    ("/api/signals/stats", dto.SignalStatsOut),
    ("/api/signals/101", dto.SignalDetailOut),
    ("/api/portfolios", list[dto.PortfolioSummaryOut]),
    ("/api/portfolios/equity?max_points=100", dto.EquityCurvesOut),
    ("/api/portfolios/ai", dto.PortfolioDetailOut),
    ("/api/portfolios/me/trades", list[dto.PositionOut]),
    ("/api/portfolios/me/orders", list[dto.OrderOut]),
    ("/api/settings", RuntimeSettings),
    ("/api/mail/log", list[dto.MailLogOut]),
    ("/api/validation", list[dto.ValidationRunOut]),
    ("/api/activity?limit=5&level=warning", list[dto.ActivityOut]),
    ("/api/system/llm-calls", list[dto.LlmCallOut]),
    ("/api/system/feeds", list[dto.FeedStateOut]),
    ("/api/system/jobs", list[dto.JobStatusOut]),
]


@pytest.mark.parametrize(("path", "model"), READ_ENDPOINTS, ids=[p for p, _ in READ_ENDPOINTS])
def test_read_endpoints_return_contract_shapes(client: TestClient, path: str, model: Any) -> None:
    response = client.get(path)
    assert response.status_code == 200, response.text
    TypeAdapter(model).validate_python(response.json())


def test_demo_data_covers_every_state_the_gui_shows(client: TestClient) -> None:
    signals = client.get("/api/signals?limit=500").json()
    statuses = {s["status"] for s in signals}
    assert {"review", "pending_entry", "active", "target", "stopped", "expired", "cancelled"} <= statuses
    assert any(not s["actionable"] for s in signals), "shadow candidates are missing"
    news = client.get("/api/news?limit=100").json()
    assert any(n["analysis"] is None and n["status"] == "queued" for n in news)
    assert any(n["analysis"] is not None for n in news)
    assert sum(f["fail_count"] > 0 for f in client.get("/api/system/feeds").json()) == 1
    instruments = client.get("/api/instruments?include_inactive=true").json()
    assert {i["region"] for i in instruments} == {"ASIA", "EU", "US", "GLOBAL"}
    assert any(i["price_scale"] == 0.01 for i in instruments), "a pence-quoted London share is missing"
    agenda = client.get("/api/agenda?min_importance=1").json()
    assert {e["kind"] for e in agenda} == {"macro", "central_bank", "earnings"}
    assert any(e["actual_text"] and e["surprise_label"] for e in agenda), "a released value is missing"
    assert any(e["blocks_entries"] for e in agenda), "an active entry block is missing"
    runs = client.get("/api/validation").json()
    assert any(r["status"] == "done" and r["result"] for r in runs) and any(r["status"] == "failed" for r in runs)


def test_bars_are_closed_ascending_and_page_backwards(client: TestClient) -> None:
    page = client.get("/api/bars/BTC-EUR?tf=5m&limit=100").json()
    ts = [b[0] for b in page["bars"]]
    assert ts == sorted(set(ts)) and len(ts) == 100
    older = client.get(f"/api/bars/BTC-EUR?tf=5m&limit=100&before={ts[0]}").json()
    assert older["bars"] and older["bars"][-1][0] < ts[0]
    status = client.get("/api/status").json()
    assert page["bars"][-1][0] + 60 <= status["server_time"]


def test_signal_details_use_the_engines_keys_and_bar_open_times(client: TestClient) -> None:
    """The demo must look like production: engine component/feature keys, entry/exit at bar OPEN times
    (the GUI snaps chart markers to the bar that opened at or before them)."""
    from trading_local.analysis.features import FEATURE_COLUMNS
    from trading_local.analysis.params import DEFAULT_WEIGHTS

    closed = [s for s in client.get("/api/signals?scope=closed&limit=200").json() if s["exit_ts"] and s["entry_ts"]]
    assert closed
    for sig in closed[:6]:
        detail = client.get(f"/api/signals/{sig['id']}").json()
        assert set(detail["components"]) == {name for name, _ in DEFAULT_WEIGHTS}
        assert set(detail["features"]) <= set(FEATURE_COLUMNS)
        for key in ("entry_ts", "exit_ts"):
            bars = client.get(f"/api/bars/{quote(sig['symbol'])}?tf=1m&limit=1&before={sig[key] + 1}").json()["bars"]
            assert bars[-1][0] == sig[key], f"signal {sig['id']}: {key} is not the open time of a bar"


def test_signal_reasons_are_the_engines_codes(client: TestClient) -> None:
    """Like production, the API carries machine codes; the GUI translates them (lib/labels.js)."""
    from typing import get_args

    from trading_local.analysis import engine, tracking
    from trading_local.execution import ExitReason

    shadow_codes = {value for name, value in vars(engine).items() if name.startswith("SHADOW_")}
    status_codes = {*get_args(ExitReason), tracking.REASON_NO_ENTRY, tracking.REASON_DRIFT}
    signals = client.get("/api/signals?limit=500").json()
    shadow = {s["shadow_reason"] for s in signals if not s["actionable"]}
    status = {s["status_reason"] for s in signals if s["status_reason"]}
    assert shadow and shadow <= shadow_codes
    assert status and status <= status_codes
    assert all(s["shadow_reason"] == "" for s in signals if s["actionable"])


def test_chart_analysis_request_explains_itself(fresh: TestClient) -> None:
    body = fresh.post("/api/analysis/NVDA/chart", json={}).json()
    assert body["queued"] is True and "angefordert" in body["detail_de"]
    assert fresh.post("/api/analysis/NVDA/chart", json={}).json()["queued"] is True  # already in progress


def test_fx_pairs_cannot_be_made_tradable_but_index_amounts_trade_in_lots(fresh: TestClient) -> None:
    refused = fresh.patch("/api/instruments/EURUSD=X", json={"tradable": True})
    assert refused.status_code == 400 and "Umrechnungskurs" in refused.json()["detail"]
    order = fresh.post("/api/portfolios/me/orders", json={"symbol": "^GDAXI", "side": "LONG", "amount_base": 2000})
    assert order.status_code == 201, order.text
    assert 0 < order.json()["qty"] < 1 and round(order.json()["qty"], 2) == order.json()["qty"]  # lots of 0.01


def test_state_changing_requests_require_json(client: TestClient) -> None:
    response = client.post("/api/mail/test", content=b"", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415
    assert "JSON" in response.json()["detail"]
    assert client.post("/api/mail/test", json={}).json()["ok"] is True


def test_order_lifecycle_and_german_rejections(fresh: TestClient) -> None:
    order = fresh.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "LONG", "amount_base": 1500})
    assert order.status_code == 201, order.text
    body = dto.OrderOut.model_validate(order.json())
    assert body.status == "pending" and body.qty and body.qty > 0
    cancelled = fresh.request("DELETE", f"/api/portfolios/me/orders/{body.id}", headers=JSON, content=b"{}")
    assert cancelled.json()["status"] == "cancelled"
    again = fresh.request("DELETE", f"/api/portfolios/me/orders/{body.id}", headers=JSON, content=b"{}")
    assert again.status_code == 400 and "nicht mehr offen" in again.json()["detail"]

    both = fresh.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "LONG", "amount_base": 100, "qty": 1})
    assert both.status_code == 400 and "entweder" in both.json()["detail"]
    wrong_stop = fresh.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "LONG", "amount_base": 100, "stop_price": 9999})
    assert wrong_stop.status_code == 400 and "Stopp" in wrong_stop.json()["detail"]
    fx = fresh.post("/api/portfolios/me/orders", json={"symbol": "EURUSD=X", "side": "LONG", "qty": 1})
    assert fx.status_code == 400 and "nicht handelbar" in fx.json()["detail"]
    invalid = fresh.post("/api/portfolios/me/orders", json={"symbol": "NVDA", "side": "UP", "qty": 1})
    assert invalid.status_code == 422 and invalid.json()["detail"].startswith("Ungültige Eingabe")


def test_position_close_and_levels(fresh: TestClient) -> None:
    position = fresh.get("/api/portfolios/me").json()["positions"][0]
    sign = 1 if position["side"] == "LONG" else -1
    stop = round(position["last_price"] * (1 - sign * 0.05), 2)
    updated = fresh.patch(f"/api/portfolios/me/positions/{position['id']}", json={"stop_price": stop, "target_price": None})
    assert updated.status_code == 200 and updated.json()["stop_price"] == stop
    bad = fresh.patch(f"/api/portfolios/me/positions/{position['id']}", json={"stop_price": position["last_price"] * (1 + sign * 0.5)})
    assert bad.status_code == 400 and "falschen Seite" in bad.json()["detail"]
    close = fresh.post(f"/api/portfolios/me/positions/{position['id']}/close", json={})
    assert close.status_code == 200 and close.json()["intent"] == "close"


def test_settings_validation_errors_are_german_and_name_the_field(fresh: TestClient) -> None:
    ok = fresh.put("/api/settings", json={"mail_max_per_hour": 5})
    assert ok.status_code == 200 and ok.json()["mail_max_per_hour"] == 5
    bad = fresh.put("/api/settings", json={"mail_max_per_hour": 500})
    assert bad.status_code == 422
    assert "mail_max_per_hour" in bad.json()["detail"] and "höchstens 120" in bad.json()["detail"]
    assert fresh.get("/api/settings").json()["mail_max_per_hour"] == 5


def test_reset_needs_confirmation_and_restarts_both_portfolios(fresh: TestClient) -> None:
    assert fresh.post("/api/portfolios/reset", json={"confirm": False}).status_code == 400
    summaries = fresh.post("/api/portfolios/reset", json={"confirm": True, "start_capital": 50_000}).json()
    assert [s["key"] for s in summaries] == ["ai", "me"]
    assert all(s["equity"] == 50_000 and s["open_positions"] == 0 for s in summaries)


def test_validation_run_conflicts_while_running(fresh: TestClient) -> None:
    first = fresh.post("/api/validation/run", json={"days": 10})
    assert first.status_code == 202 and first.json()["status"] == "running"
    second = fresh.post("/api/validation/run", json={"days": 10})
    assert second.status_code == 409 and "läuft bereits" in second.json()["detail"]


def test_unknown_resources_answer_in_german(client: TestClient) -> None:
    assert "nicht in der Beobachtungsliste" in client.get("/api/bars/XYZ").json()["detail"]
    assert client.get("/api/signals/999999").status_code == 404
    assert client.post("/api/instruments", json={"symbol": "NVDA"}).status_code == 409


def test_websocket_greets_with_server_time(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and isinstance(hello["data"]["server_time"], int)
        ws.send_json({"type": "subscribe", "symbols": ["NVDA"]})


def test_static_bundle_is_served_with_module_mime_type_and_strict_csp(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200 and "script-src 'self'" in index.headers["content-security-policy"]
    script = client.get("/vendor/lightweight-charts.mjs")
    assert script.headers["content-type"].startswith("text/javascript")
