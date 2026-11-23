"""Validation runs: process start with a fixed argument list, one run at a time, end of the process."""

import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trading_local.config import PROJECT_ROOT
from trading_local.db import repo_ops
from trading_local.services import Services
from trading_local.web.ws import HubTiming

from .conftest import BASE_URL, NOW_TS, ORIGIN, WS_URL


class FakePopen:
    """Records the start of a process; the test decides when it exits."""

    def __init__(self, args: list[str], **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def processes(monkeypatch: pytest.MonkeyPatch) -> list[FakePopen]:
    started: list[FakePopen] = []

    def popen(args: list[str], **kwargs: Any) -> FakePopen:
        process = FakePopen(args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    return started


@pytest.fixture
def live_client(app: FastAPI) -> Iterator[TestClient]:
    """A client whose app pings every 50 ms (WebSocket reads never wait long) and polls processes quickly."""
    app.state.hub.timing = HubTiming(status_interval_s=3600, ping_interval_s=0.05)
    app.state.validation.poll_interval_s = 0.01
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as test_client:
        yield test_client


def receive_until(websocket: Any, message_type: str, limit: int = 200) -> dict[str, Any]:
    for _ in range(limit):
        message: dict[str, Any] = websocket.receive_json()
        if message["type"] == message_type:
            return message
    raise AssertionError(f"no {message_type!r} message")


def test_start_spawns_the_process_with_a_fixed_argument_list(client: TestClient, processes: list[FakePopen]) -> None:
    response = client.post("/api/validation/run", json={"days": 10}, headers={"origin": ORIGIN})
    assert response.status_code == 202
    run = response.json()
    assert (run["status"], run["params"], run["engine_version"]) == ("running", {"days": 10}, "engine-test-1")
    assert len(processes) == 1
    process = processes[0]
    assert process.args == [sys.executable, "-m", "trading_local.validate", "--days", "10", "--run-id", str(run["id"])]
    assert not process.kwargs.get("shell")
    assert process.kwargs["cwd"] == PROJECT_ROOT
    assert process.kwargs["stdin"] == subprocess.DEVNULL


def test_a_second_run_is_refused_while_one_is_running(client: TestClient, processes: list[FakePopen]) -> None:
    assert client.post("/api/validation/run", json={"days": 5}).status_code == 202
    response = client.post("/api/validation/run", json={"days": 5})
    assert response.status_code == 409
    assert response.json()["detail"] == "Es läuft bereits eine Validierung. Bitte warten."
    assert len(processes) == 1


def test_an_exit_without_result_marks_the_run_failed_and_announces_it(
    live_client: TestClient, processes: list[FakePopen]
) -> None:
    with live_client.websocket_connect(WS_URL) as websocket:
        run_id = live_client.post("/api/validation/run", json={}).json()["id"]
        processes[0].returncode = 1
        assert receive_until(websocket, "validation")["data"] == {"id": run_id, "status": "failed"}
    run = live_client.get("/api/validation").json()[0]
    assert (run["id"], run["status"]) == (run_id, "failed")
    assert "Exit-Code 1" in run["message"]
    assert live_client.post("/api/validation/run", json={"days": 3}).status_code == 202  # free again


def test_a_result_written_by_the_process_is_kept(
    live_client: TestClient, services: Services, processes: list[FakePopen]
) -> None:
    with live_client.websocket_connect(WS_URL) as websocket:
        run_id = live_client.post("/api/validation/run", json={"days": 20}).json()["id"]
        repo_ops.finish_validation_run(services.db, run_id, status="done", message="fertig")
        processes[0].returncode = 0
        assert receive_until(websocket, "validation")["data"] == {"id": run_id, "status": "done"}
    assert live_client.get("/api/validation").json()[0]["message"] == "fertig"


def test_stale_running_rows_are_marked_failed(
    client: TestClient, services: Services, processes: list[FakePopen]
) -> None:
    stale = repo_ops.create_validation_run(services.db, ts=NOW_TS - 3 * 3600, engine_version="old", params={})
    runs = {run["id"]: run for run in client.get("/api/validation").json()}
    assert runs[stale]["status"] == "failed" and "zwei Stunden" in runs[stale]["message"]
    young = repo_ops.create_validation_run(services.db, ts=NOW_TS - 3600, engine_version="old", params={})
    assert client.post("/api/validation/run", json={"days": 5}).status_code == 409  # younger than two hours
    assert {run["id"]: run["status"] for run in client.get("/api/validation").json()}[young] == "running"
    assert processes == []


def test_a_process_that_cannot_start_fails_the_run(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_popen(args: list[str], **kwargs: Any) -> FakePopen:
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "Popen", broken_popen)
    response = client.post("/api/validation/run", json={"days": 5})
    assert response.status_code == 503
    assert response.json()["detail"] == "Die Validierung konnte nicht gestartet werden."
    assert client.get("/api/validation").json()[0]["status"] == "failed"


def test_the_latest_five_runs_are_listed(client: TestClient, services: Services) -> None:
    created = [
        repo_ops.create_validation_run(services.db, ts=NOW_TS - 1000 + i, engine_version="v", params={"days": i})
        for i in range(7)
    ]
    for run_id in created:
        repo_ops.finish_validation_run(services.db, run_id, status="done")
    assert [run["id"] for run in client.get("/api/validation").json()] == created[::-1][:5]


@pytest.mark.parametrize(("days", "fragment"), [(1, "mindestens 2"), (366, "höchstens 365"), ("x", "ganze Zahl")])
def test_days_are_validated(client: TestClient, processes: list[FakePopen], days: Any, fragment: str) -> None:
    response = client.post("/api/validation/run", json={"days": days})
    assert response.status_code == 422
    assert "Feld 'days'" in response.json()["detail"] and fragment in response.json()["detail"]
    assert processes == []


def test_watchers_stop_with_the_app(app: FastAPI, processes: list[FakePopen]) -> None:
    app.state.validation.poll_interval_s = 0.01
    with TestClient(app, base_url=BASE_URL) as client:
        assert client.post("/api/validation/run", json={"days": 5}).status_code == 202
        started = time.monotonic()
    assert time.monotonic() - started < 2.0  # the running watcher did not delay the shutdown
