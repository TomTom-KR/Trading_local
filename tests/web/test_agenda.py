"""GET /api/agenda: the calendar with expectation, actual value, surprise and entry block."""

from fastapi.testclient import TestClient

from trading_local.db import repo_agenda
from trading_local.db.database import Database
from trading_local.db.repo_agenda import AgendaEvent

from .conftest import NOW_TS


def event(**fields: object) -> AgendaEvent:
    values: dict[str, object] = {
        "kind": "macro", "source": "forexfactory", "title": "CPI m/m", "event_ts": NOW_TS + 600, "first_seen_ts": 0,
        "updated_ts": 0, "currency": "USD", "importance": 3, "forecast_text": "0.3%", "previous_text": "0.2%",
    }  # fmt: skip
    values.update(fields)
    values.setdefault("uid", f"{values['kind']}:{values['title']}:{values['event_ts']}:{values['currency']}")
    return AgendaEvent(**values)  # type: ignore[arg-type]


def seed(db: Database) -> None:
    repo_agenda.upsert_events(db, [
        event(),
        event(title="Ifo Business Climate", currency="EUR", importance=2, event_ts=NOW_TS - 3600, forecast_text="87.5",
              actual=88.1, actual_text="88.1", surprise=0.6, actual_ts=NOW_TS - 3550),
        event(kind="earnings", title="Quartalszahlen", currency="", symbol="NVDA", event_ts=NOW_TS + 3 * 86_400,
              forecast_text="1.25"),
        event(title="Redbook y/y", importance=1, event_ts=NOW_TS - 7200),
        event(title="Far away", event_ts=NOW_TS + 30 * 86_400),
    ])  # fmt: skip


def test_agenda_lists_the_coming_week_with_results(client: TestClient, db: Database) -> None:
    seed(db)
    rows = client.get("/api/agenda").json()
    assert [row["title"] for row in rows] == ["Redbook y/y", "Ifo Business Climate", "CPI m/m", "Quartalszahlen"]
    cpi = rows[2]
    assert cpi["kind_de"] == "Konjunkturdaten" and cpi["importance"] == 3 and cpi["affected"] == 3  # NVDA, ^GSPC, BTC
    assert cpi["block_from_ts"] == NOW_TS + 600 - 30 * 60 and cpi["blocks_entries"] is True
    ifo = rows[1]
    assert ifo["surprise_label"] == "above" and ifo["result_de"] == "88.1 statt 87.5 – über Erwartung"
    assert ifo["block_from_ts"] is None and ifo["blocks_entries"] is False
    earnings = rows[3]
    assert (earnings["kind"], earnings["symbol"], earnings["name"], earnings["affected"]) == ("earnings", "NVDA",
                                                                                             "NVIDIA", 1)  # fmt: skip


def test_agenda_filters(client: TestClient, db: Database) -> None:
    seed(db)
    titles = lambda **params: [row["title"] for row in client.get("/api/agenda", params=params).json()]  # noqa: E731
    assert titles(min_importance=3) == ["CPI m/m", "Quartalszahlen"]
    assert titles(kind="earnings") == ["Quartalszahlen"]
    assert titles(symbol="SAP.DE") == ["Ifo Business Climate"]  # euro data, no US events, no NVDA earnings
    assert titles(symbol="NVDA") == ["Redbook y/y", "CPI m/m", "Quartalszahlen"]
    assert titles(from_ts=NOW_TS + 20 * 86_400, to_ts=NOW_TS + 40 * 86_400) == ["Far away"]
    client.put("/api/settings", json={"event_block_enabled": False})
    assert all(not row["blocks_entries"] for row in client.get("/api/agenda").json())
    assert client.get("/api/agenda", params={"symbol": "NOPE"}).status_code == 404
    assert client.get("/api/agenda", params={"min_importance": 4}).status_code == 422
