"""Technical analysis, chart analyses, news and the signal journal."""

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_local.db import repo_signals
from trading_local.models import ChartAnalysis
from trading_local.services import Services
from trading_local.web.routes import analysis as analysis_routes

from .conftest import ORIGIN, NOW_TS, Fakes, insert_news, insert_signal, quote


# ------------------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------------------
def test_analysis_comes_from_the_engine(client: TestClient) -> None:
    body = client.get("/api/analysis/nvda").json()
    assert (body["symbol"], body["note_de"], body["features"]) == ("NVDA", "Aufwärtstrend", {"rsi14": 55.0})
    assert client.get("/api/analysis/NOPE").status_code == 404


def test_chart_analysis_request_is_queued(client: TestClient, fakes: Fakes) -> None:
    response = client.post("/api/analysis/sap.de/chart", headers={"origin": ORIGIN})
    assert response.status_code == 202
    assert response.json() == {"queued": True, "detail_de": analysis_routes.QUEUED_DE}
    assert fakes.engine.chart_requests == [("SAP.DE", "15m")]
    fakes.engine.chart_queue_accepts = False
    assert client.post("/api/analysis/SAP.DE/chart").json() == {"queued": False, "detail_de": analysis_routes.NOT_QUEUED_DE}
    assert client.post("/api/analysis/NOPE/chart").status_code == 404


def test_a_refused_chart_analysis_says_why(client: TestClient, fakes: Fakes, services: Services) -> None:
    fakes.engine.chart_queue_accepts = False
    services.settings = services.settings.model_copy(update={"enable_llm": False})
    body = client.post("/api/analysis/NVDA/chart").json()
    assert body == {"queued": False, "detail_de": analysis_routes.LLM_DISABLED_DE}
    assert "ausgeschaltet" in body["detail_de"]


def test_chart_analyses_newest_first(client: TestClient, services: Services) -> None:
    for offset in (300, 100, 200):
        repo_signals.insert_chart_analysis(services.db, ChartAnalysis(
            symbol="NVDA", ts=NOW_TS - offset, timeframe="15m", model="qwen3.5:9b", bias="bullish", strength=3,
            comment_de="Aufwärtstrend intakt",
        ))
    body = client.get("/api/chart-analyses/NVDA", params={"limit": 2}).json()
    assert [entry["ts"] for entry in body] == [NOW_TS - 100, NOW_TS - 200]
    assert body[0]["comment_de"] == "Aufwärtstrend intakt"
    assert client.get("/api/chart-analyses/NVDA", params={"limit": 201}).status_code == 422
    assert client.get("/api/chart-analyses/NOPE").status_code == 404


# ------------------------------------------------------------------------------------------------
# News
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def news_ids(services: Services) -> list[int]:
    db = services.db
    return [
        insert_news(db, "a", "NVIDIA übertrifft Erwartungen", NOW_TS - 400, symbols={"NVDA": 0.7}, relevance=5),
        insert_news(db, "b", "SAP senkt Prognose", NOW_TS - 300, symbols={"SAP.DE": -0.5}, relevance=2),
        insert_news(db, "c", "Ölpreis steigt", NOW_TS - 200, source="unknown_feed"),
        insert_news(db, "d", "Fed hält Zinsen", NOW_TS - 100, symbols={}, relevance=3),
    ]


def ids(response: Any) -> list[int]:
    assert response.status_code == 200, response.text
    return [entry["id"] for entry in response.json()]


def test_news_filters_and_paging(client: TestClient, news_ids: list[int]) -> None:
    a, b, c, d = news_ids
    assert ids(client.get("/api/news")) == [d, c, b, a]
    assert ids(client.get("/api/news", params={"limit": 2})) == [d, c]
    assert ids(client.get("/api/news", params={"before_id": c})) == [b, a]
    assert ids(client.get("/api/news", params={"symbol": "nvda"})) == [a]
    assert ids(client.get("/api/news", params={"min_relevance": 3})) == [d, a]
    assert ids(client.get("/api/news", params={"analyzed_only": "true"})) == [d, b, a]


def test_news_carry_their_analysis(client: TestClient, news_ids: list[int]) -> None:
    entry = client.get("/api/news", params={"symbol": "SAP.DE"}).json()[0]
    assert entry["analysis"]["relevance"] == 2
    assert entry["analysis"]["symbols"] == [{"symbol": "SAP.DE", "sentiment": -0.5}]


def test_news_source_names_come_from_the_feed_registry(
    client: TestClient, news_ids: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    feeds = types.ModuleType("trading_local.news.feeds")
    feeds.FEEDS = (SimpleNamespace(id="reuters_business", name="Reuters Wirtschaft"),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trading_local.news.feeds", feeds)
    names = {entry["source"]: entry["source_name"] for entry in client.get("/api/news").json()}
    assert names == {"reuters_business": "Reuters Wirtschaft", "unknown_feed": "unknown_feed"}
    for broken in (42, [{"id": "reuters_business"}]):  # unreadable registries fall back to the feed id
        feeds.FEEDS = broken  # type: ignore[attr-defined]
        assert client.get("/api/news").json()[-1]["source_name"] == "reuters_business"


def test_news_source_names_from_the_real_registry(client: TestClient, services: Services) -> None:
    feed = pytest.importorskip("trading_local.news.feeds").FEEDS[0]
    insert_news(services.db, "real", "Meldung", NOW_TS, source=feed.id)
    assert client.get("/api/news").json()[0]["source_name"] == feed.name


@pytest.mark.parametrize(
    ("params", "status"),
    [({"limit": 0}, 422), ({"limit": 201}, 422), ({"min_relevance": 6}, 422), ({"before_id": 0}, 422),
     ({"analyzed_only": "maybe"}, 422), ({"symbol": "NOPE"}, 404)],
)
def test_news_parameters_are_validated(client: TestClient, params: dict[str, Any], status: int) -> None:
    response = client.get("/api/news", params=params)
    assert response.status_code == status
    assert response.json()["detail"]


# ------------------------------------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------------------------------------
def test_signal_list_scopes_and_filters(client: TestClient, services: Services, fakes: Fakes) -> None:
    db = services.db
    fakes.data.quote_book["NVDA"] = quote("NVDA", 186.0, NOW_TS)
    active = insert_signal(db, "NVDA", status="active", actionable=True, created_ts=NOW_TS - 500, entry_price=180.0)
    shadow = insert_signal(db, "NVDA", status="review", actionable=False, created_ts=NOW_TS - 400)
    closed = insert_signal(db, "SAP.DE", status="stopped", actionable=True, created_ts=NOW_TS - 300)
    assert ids(client.get("/api/signals")) == [closed.id, shadow.id, active.id]
    assert ids(client.get("/api/signals", params={"scope": "open"})) == [shadow.id, active.id]
    assert ids(client.get("/api/signals", params={"scope": "closed"})) == [closed.id]
    assert ids(client.get("/api/signals", params={"actionable": "true"})) == [closed.id, active.id]
    assert ids(client.get("/api/signals", params={"actionable": "false"})) == [shadow.id]
    assert ids(client.get("/api/signals", params={"symbol": "sap.de"})) == [closed.id]
    assert ids(client.get("/api/signals", params={"before_id": closed.id, "limit": 1})) == [shadow.id]
    by_id = {entry["id"]: entry for entry in client.get("/api/signals").json()}
    assert by_id[active.id]["last_price"] == 186.0 and by_id[active.id]["open_r"] == pytest.approx(2.0)
    assert by_id[active.id]["name"] == "NVIDIA" and by_id[active.id]["setup_label_de"] == "Ausbruch"
    assert by_id[closed.id]["last_price"] is None  # closed signals show no live price


@pytest.mark.parametrize(
    ("params", "status"),
    [({"scope": "pending"}, 422), ({"actionable": "yes"}, 422), ({"limit": 501}, 422), ({"limit": 0}, 422),
     ({"symbol": "NOPE"}, 404)],
)
def test_signal_parameters_are_validated(client: TestClient, params: dict[str, Any], status: int) -> None:
    assert client.get("/api/signals", params=params).status_code == status


def test_signal_detail_links_its_news(client: TestClient, services: Services, news_ids: list[int]) -> None:
    signal = insert_signal(services.db, "NVDA", status="active", actionable=True, created_ts=NOW_TS - 50,
                           news_ids=[news_ids[0], 999])
    body = client.get(f"/api/signals/{signal.id}").json()
    assert body["id"] == signal.id
    assert [item["id"] for item in body["news"]] == [news_ids[0]]  # unknown news ids are skipped
    assert body["news"][0]["analysis"]["symbols"][0]["symbol"] == "NVDA"
    response = client.get("/api/signals/424242")
    assert response.status_code == 404
    assert response.json()["detail"] == "Signal 424242 nicht gefunden."
    assert client.get("/api/signals/abc").status_code == 422


def test_signal_stats_come_from_the_engine(client: TestClient) -> None:
    body = client.get("/api/signals/stats", params={"since_ts": NOW_TS - 86_400}).json()
    assert (body["since_ts"], body["actionable"]["n"], body["shadow"]["n"]) == (NOW_TS - 86_400, 3, 5)
    assert client.get("/api/signals/stats").json()["since_ts"] is None
    assert client.get("/api/signals/stats", params={"since_ts": -5}).status_code == 422
