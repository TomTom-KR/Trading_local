"""Status composition, trading sessions, activity log, LLM calls, feeds and jobs."""

import sys
import types
from datetime import datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

import trading_local
from trading_local.clock import ManualClock
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.dto import DbStatusOut, SessionsOut
from trading_local.models import Instrument
from trading_local.services import Services
from trading_local.web.status import compose_status

from .conftest import NOW_TS, Fakes, seed_market


def test_status_composes_all_components(client: TestClient, services: Services) -> None:
    seed_market(services.db)
    body = client.get("/api/status").json()
    assert body["server_time"] == NOW_TS
    assert body["started_ts"] == NOW_TS - 600
    assert body["version"] == trading_local.__version__
    assert body["engine_version"] == "engine-test-1"
    assert body["base_currency"] == "EUR"
    assert body["offline"] is True
    assert body["llm"]["model"] == "qwen3.5:9b" and body["llm"]["queue_depth"] == 2
    assert body["data"]["stream_connected"] is True
    assert body["news"]["items_last_hour"] == 7
    assert body["mail"]["recipient_masked"] == "e***@example.invalid"
    assert body["db"]["bars_1m"] > 1000 and body["db"]["last_bar_ts"] is not None
    assert body["jobs"][0]["job"] == "market_data"


def test_database_statistics_are_cached_for_30_seconds(
    services: Services, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Database] = []

    def counting_db_stats(db: Database) -> DbStatusOut:
        calls.append(db)
        return DbStatusOut(bars_1m=len(calls))

    monkeypatch.setattr(repo_ops, "db_stats", counting_db_stats)
    assert compose_status(services).db.bars_1m == 1
    clock.advance(29)
    assert compose_status(services).db.bars_1m == 1
    clock.advance(1)
    assert compose_status(services).db.bars_1m == 2
    assert len(calls) == 2


def test_a_failing_component_status_does_not_break_the_status(client: TestClient, fakes: Fakes) -> None:
    fakes.llm.status_error = RuntimeError("ollama down")
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["llm"] == {**response.json()["llm"], "model": "qwen3.5:9b", "available": False}


def test_sessions_come_from_the_markets_overview(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[Any, ...]] = []

    def sessions_overview(calendar: MarketCalendar, instruments: list[Instrument], now: datetime) -> SessionsOut:
        received.append((calendar, instruments, now))
        return SessionsOut(now=NOW_TS, regions=[], timeline_start=NOW_TS - 3600, timeline_end=NOW_TS, timeline=[])

    overview = types.ModuleType("trading_local.markets.overview")
    overview.sessions_overview = sessions_overview  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trading_local.markets.overview", overview)
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json()["timeline_start"] == NOW_TS - 3600
    calendar, instruments, now = received[0]
    assert isinstance(calendar, MarketCalendar)
    assert [instrument.symbol for instrument in instruments][:2] == ["NVDA", "SAP.DE"]
    assert int(now.timestamp()) == NOW_TS


def test_activity_log_with_filters_and_paging(client: TestClient, services: Services) -> None:
    entries = [("info", "data"), ("warning", "mail"), ("error", "mail"), ("info", "sim")]
    ids = [
        repo_ops.log_activity(services.db, ts=NOW_TS - 100 + i, level=level, category=category, message=f"Eintrag {i}")
        for i, (level, category) in enumerate(entries)
    ]

    def listed(**params: Any) -> list[int]:
        return [entry["id"] for entry in client.get("/api/activity", params=params).json()]

    assert listed() == ids[::-1]
    assert listed(category="mail") == [ids[2], ids[1]]
    assert listed(level="info") == [ids[3], ids[0]]
    assert listed(before_id=ids[2], limit=1) == [ids[1]]


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ({"limit": 0}, "Abfrageparameter 'limit' muss mindestens 1 sein"),
        ({"limit": 1001}, "Abfrageparameter 'limit' darf höchstens 1000 sein"),
        ({"level": "debug"}, "Abfrageparameter 'level' muss einer der Werte"),
        ({"before_id": "abc"}, "Abfrageparameter 'before_id' muss eine ganze Zahl sein"),
        ({"category": "x" * 33}, "Abfrageparameter 'category' ist zu lang"),
    ],
)
def test_activity_parameters_are_validated_in_german(
    client: TestClient, params: dict[str, Any], fragment: str
) -> None:
    response = client.get("/api/activity", params=params)
    assert response.status_code == 422
    assert fragment in response.json()["detail"]


def test_llm_calls_feeds_and_jobs(client: TestClient, services: Services) -> None:
    repo_ops.log_llm_call(services.db, ts=NOW_TS - 10, kind="news", ok=True, latency_ms=2300)
    repo_ops.log_llm_call(services.db, ts=NOW_TS - 5, kind="chart", ok=False, latency_ms=45_000, error="timeout")
    calls = client.get("/api/system/llm-calls", params={"limit": 1}).json()
    assert [(call["kind"], call["ok"], call["error"]) for call in calls] == [("chart", False, "timeout")]
    assert client.get("/api/system/llm-calls", params={"limit": 5000}).status_code == 422
    assert client.get("/api/system/feeds").json()[0]["name"] == "Reuters Wirtschaft"
    assert client.get("/api/system/jobs").json()[0]["label_de"] == "Kurse und Minutenbalken"
