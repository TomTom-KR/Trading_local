"""Reviewer's attack tests: parameter overflow, non-finite numbers, threading rule, hub races, validation
runs that outlive two hours, symbol canonicalisation and other boundaries the feature tests leave open."""

import asyncio
import re
import subprocess
import sys
import threading
import time
import types
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trading_local.clock import ManualClock
from trading_local.db import repo_ops
from trading_local.dto import QuoteOut, SessionsOut
from trading_local.events import TOPIC_QUOTES
from trading_local.models import Instrument
from trading_local.services import Services
from trading_local.web import create_app, ws
from trading_local.web.deps import MAX_ID, ApiContext
from trading_local.web.routes import market
from trading_local.web.routes.validation import STALE_AFTER_S
from trading_local.web.ws import Hub, HubTiming

from .conftest import BASE_URL, NOW_TS, Fakes, insert_signal, seed_market
from .test_validation import FakePopen
from .test_ws import QUIET, ScriptedSocket, connect, eventually

HUGE = str(10**20)  # beyond SQLite's 64-bit INTEGER


@pytest.fixture
def overview_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    overview = types.ModuleType("trading_local.markets.overview")
    overview.sessions_overview = lambda calendar, instruments, now: SessionsOut(  # type: ignore[attr-defined]
        now=NOW_TS, regions=[], timeline_start=NOW_TS, timeline_end=NOW_TS, timeline=[]
    )
    monkeypatch.setitem(sys.modules, "trading_local.markets.overview", overview)


def get_paths(app: FastAPI, values: dict[str, str]) -> list[str]:
    return sorted(
        re.sub(r"\{(\w+)\}", lambda match: values[match.group(1)], path)
        for path, operations in app.openapi()["paths"].items()
        if "get" in operations
    )


# ------------------------------------------------------------------------------------------------
# Parameter bounds: ids and times beyond the database range are 422, never 500
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", f"/api/signals/{HUGE}"),
        ("GET", "/api/signals/0"),
        ("GET", f"/api/signals?before_id={HUGE}"),
        ("GET", f"/api/signals/stats?since_ts={HUGE}"),
        ("GET", f"/api/news?before_id={HUGE}"),
        ("GET", f"/api/activity?before_id={HUGE}"),
        ("GET", f"/api/portfolios/equity?since_ts={HUGE}"),
        ("GET", f"/api/bars/NVDA?before={HUGE}"),
        ("DELETE", f"/api/portfolios/me/orders/{HUGE}"),
        ("POST", f"/api/portfolios/me/positions/{HUGE}/close"),
    ],
)
def test_out_of_range_ids_and_times_are_422(client: TestClient, fakes: Fakes, method: str, path: str) -> None:
    response = client.request(method, path)
    assert response.status_code == 422, response.text
    assert response.json()["detail"].startswith("Ungültige Eingabe: ")
    assert fakes.broker.calls == []


def test_the_largest_valid_id_is_accepted(client: TestClient) -> None:
    response = client.get(f"/api/signals/{MAX_ID}")
    assert response.status_code == 404
    assert client.get("/api/news", params={"before_id": MAX_ID}).status_code == 200


@pytest.mark.parametrize(
    ("method", "path", "body", "field"),
    [
        ("POST", "/api/portfolios/reset", b'{"confirm": true, "start_capital": Infinity}', "start_capital"),
        ("POST", "/api/portfolios/me/orders", b'{"symbol": "NVDA", "side": "LONG", "qty": Infinity}', "qty"),
        ("POST", "/api/portfolios/me/orders",
         b'{"symbol": "NVDA", "side": "LONG", "amount_base": 1e400}', "amount_base"),
        ("POST", "/api/portfolios/me/orders",
         b'{"symbol": "NVDA", "side": "LONG", "qty": 1, "signal_id": 99999999999999999999}', "signal_id"),
        ("PATCH", "/api/portfolios/me/positions/7", b'{"stop_price": Infinity}', "stop_price"),
    ],
)
def test_non_finite_or_oversized_numbers_in_bodies_are_422(
    client: TestClient, fakes: Fakes, method: str, path: str, body: bytes, field: str
) -> None:
    response = client.request(method, path, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert f"Feld '{field}'" in response.json()["detail"]
    assert fakes.broker.calls == []


def test_unreadable_json_bodies_get_german_details(client: TestClient) -> None:
    response = client.put("/api/settings", content=b"\xff\xfe{", headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json()["detail"].startswith("Der Anfrageinhalt konnte nicht gelesen werden")
    response = client.put("/api/settings", content=b'{"a": ', headers={"content-type": "application/json"})
    assert response.json()["detail"] == "Ungültige Eingabe: Der Anfrageinhalt ist kein gültiges JSON."


# ------------------------------------------------------------------------------------------------
# Threading rule: handlers never read the database on the event loop or in FastAPI's thread pool
# ------------------------------------------------------------------------------------------------
def test_database_reads_happen_only_on_the_two_api_threads(
    services: Services, static_dir: Path, monkeypatch: pytest.MonkeyPatch, overview_stub: None
) -> None:
    db = services.db
    seed_market(db)
    signal = insert_signal(db, "NVDA", status="active", actionable=True, created_ts=NOW_TS - 300)
    repo_ops.log_activity(db, ts=NOW_TS - 50, level="info", category="sim", message="Eintrag")
    threads: list[str] = []
    real_conn = db.conn

    def recording_conn() -> Any:
        threads.append(threading.current_thread().name)
        return real_conn()

    monkeypatch.setattr(db, "conn", recording_conn)
    pool_calls: list[object] = []

    async def forbidden_pool(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        pool_calls.append(func)
        return func(*args, **kwargs)

    import fastapi.dependencies.utils as fastapi_utils
    import fastapi.routing as fastapi_routing

    monkeypatch.setattr(fastapi_routing, "run_in_threadpool", forbidden_pool)
    monkeypatch.setattr(fastapi_utils, "run_in_threadpool", forbidden_pool)
    app = create_app(services, static_dir=static_dir)
    app.state.hub.timing = HubTiming(status_interval_s=3600, ping_interval_s=3600)
    with TestClient(app, base_url=BASE_URL) as client:
        assert threads and all(name.startswith("api-blocking") for name in threads)  # book loaded at start-up
        for path in get_paths(app, {"symbol": "NVDA", "key": "me", "signal_id": str(signal.id)}):
            assert client.get(path).status_code == 200, path
        assert client.put("/api/settings", json={"mail_enabled": False}).status_code == 200
        assert client.patch("/api/instruments/NVDA", json={"tradable": True}).status_code == 200
    assert pool_calls == []
    assert {name for name in threads if not name.startswith("api-blocking")} == set()
    assert len(set(threads)) <= 2


async def test_the_api_executor_runs_at_most_two_calls_at_once(services: Services) -> None:
    ctx = ApiContext(services)
    lock = threading.Lock()
    running = peak = 0

    def blocking() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.03)
        with lock:
            running -= 1

    try:
        await asyncio.gather(*(ctx.run(blocking) for _ in range(6)))
    finally:
        await ctx.aclose()
    assert peak == 2


async def test_closing_waits_for_running_calls_and_drops_queued_ones(services: Services) -> None:
    ctx = ApiContext(services)
    started = threading.Semaphore(0)
    finished: list[int] = []

    def blocking(n: int) -> None:
        started.release()
        time.sleep(0.05)
        finished.append(n)

    calls = [asyncio.ensure_future(ctx.run(blocking, n)) for n in range(4)]
    assert await asyncio.to_thread(lambda: started.acquire(timeout=1.0) and started.acquire(timeout=1.0))
    await ctx.aclose()
    assert sorted(finished) == [0, 1]  # both workers had started; the two queued calls never ran
    await asyncio.wait(calls)
    assert [call.cancelled() for call in calls] == [False, False, True, True]


def test_sparklines_read_all_symbols_with_one_statement(services: Services, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_market(services.db)
    services.instruments.all()  # load the book first: only the sparkline statements are counted
    statements: list[str] = []
    real_query = services.db.query

    def counting_query(sql: str, params: Any = ()) -> Any:
        statements.append(sql)
        return real_query(sql, params)

    monkeypatch.setattr(services.db, "query", counting_query)
    series = market.compute_sparklines(services).series
    assert len(statements) == 1
    assert series["NVDA"] and series["7203.T"]


# ------------------------------------------------------------------------------------------------
# GUI files
# ------------------------------------------------------------------------------------------------
def test_html_pages_are_never_cached_and_head_works(client: TestClient) -> None:
    assert client.get("/index.html").headers["cache-control"] == "no-cache"
    assert client.head("/").status_code == 200
    assert client.head("/assets/style.css").headers["cache-control"] == "public, max-age=60"


@pytest.mark.parametrize(
    "path",
    ["/CON", "/NUL", "/assets/com1.js", "/..%5c..%5cpyproject.toml", "/C:/Windows/win.ini",
     "/%2e%2e/%2e%2e/pyproject.toml", "/assets/..%5c..%5c..%5cconftest.py"],
)
def test_windows_device_names_and_traversal_variants_are_404(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert response.json() == {"detail": "Nicht gefunden."}


# ------------------------------------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------------------------------------
def test_a_failing_start_up_hook_stops_the_hub_and_skips_the_shutdown_hook(
    services: Services, static_dir: Path
) -> None:
    calls: list[str] = []

    async def on_startup() -> None:
        calls.append("start")
        raise RuntimeError("runtime could not start")

    async def on_shutdown() -> None:
        calls.append("stop")

    app = create_app(services, on_startup=on_startup, on_shutdown=on_shutdown, static_dir=static_dir)
    with pytest.raises(RuntimeError, match="runtime could not start"):
        with TestClient(app, base_url=BASE_URL):
            pass
    assert calls == ["start"]
    assert not app.state.hub.running


# ------------------------------------------------------------------------------------------------
# WebSocket hub
# ------------------------------------------------------------------------------------------------
@pytest.fixture
async def hub(services: Services) -> AsyncIterator[Hub]:
    services.bus.bind_loop()
    ctx = ApiContext(services)
    instance = Hub(ctx, QUIET)
    yield instance
    await instance.stop()
    await ctx.aclose()


async def test_quote_models_are_coalesced_like_dicts(hub: Hub, services: Services) -> None:
    socket = ScriptedSocket()
    await connect(hub, socket)
    services.bus.publish(TOPIC_QUOTES, {"items": [QuoteOut(symbol="NVDA", price=1.0, ts=NOW_TS)]})
    services.bus.publish(TOPIC_QUOTES, {"items": [QuoteOut(symbol="NVDA", price=2.0, ts=NOW_TS + 1)]})
    await eventually(lambda: bool(socket.of_type("quotes")))
    items = socket.of_type("quotes")[0]["data"]["items"]
    assert [(item["symbol"], item["price"], item["ts"]) for item in items] == [("NVDA", 2.0, NOW_TS + 1)]


class SlowHandshakeSocket(ScriptedSocket):
    def __init__(self) -> None:
        super().__init__()
        self.accepting = asyncio.Event()
        self.release = asyncio.Event()

    async def accept(self) -> None:
        self.accepting.set()
        await self.release.wait()


async def test_a_handshake_that_completes_after_stop_is_closed(hub: Hub) -> None:
    await hub.start()
    socket = SlowHandshakeSocket()
    task = asyncio.create_task(hub.serve(socket))  # type: ignore[arg-type]
    await socket.accepting.wait()
    await hub.stop()
    socket.release.set()
    await asyncio.wait_for(task, timeout=1.0)
    assert socket.close_codes == [1001] and socket.sent == []
    assert hub.client_count == 0


async def test_no_status_is_composed_while_nobody_listens(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Services] = []
    monkeypatch.setattr(ws, "compose_status", lambda s: calls.append(s))
    services.bus.bind_loop()
    ctx = ApiContext(services)
    idle = Hub(ctx, HubTiming(status_interval_s=0.01, ping_interval_s=3600))
    await idle.start()
    await asyncio.sleep(0.08)
    await idle.stop()
    await ctx.aclose()
    assert calls == []


async def test_messages_survive_a_round_trip_through_orjson(hub: Hub, services: Services) -> None:
    socket = ScriptedSocket()
    await connect(hub, socket)
    services.bus.publish("news", {"title": "Ölpreis \u2013 Rekord", "score": float("nan"), "n": 2**40})
    await eventually(lambda: bool(socket.of_type("news")))
    assert socket.of_type("news")[0]["data"] == {"title": "Ölpreis \u2013 Rekord", "score": None, "n": 2**40}
    assert orjson.loads(ws.encode_message("ping") or "") == {"type": "ping"}


# ------------------------------------------------------------------------------------------------
# Validation runs
# ------------------------------------------------------------------------------------------------
def test_a_watched_run_that_outlives_two_hours_still_blocks_new_runs(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, clock: ManualClock
) -> None:
    started: list[FakePopen] = []

    def popen(args: list[str], **kwargs: Any) -> FakePopen:
        started.append(FakePopen(args, **kwargs))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    run_id = client.post("/api/validation/run", json={"days": 300}).json()["id"]
    clock.advance(STALE_AFTER_S + 60)
    assert client.get("/api/validation").json()[0]["status"] == "running"  # its process is still alive
    response = client.post("/api/validation/run", json={"days": 5})
    assert response.status_code == 409
    assert len(started) == 1
    started[0].returncode = 0
    runner = client.app.state.validation  # type: ignore[attr-defined]
    assert runner._finish(run_id, 0) == "failed"  # exited without writing a result
    assert client.post("/api/validation/run", json={"days": 5}).status_code == 202


# ------------------------------------------------------------------------------------------------
# Instruments and news
# ------------------------------------------------------------------------------------------------
def test_a_lookup_that_resolves_to_a_listed_symbol_is_409(client: TestClient, fakes: Fakes) -> None:
    fakes.data.lookups["NVIDIA"] = Instrument("NVDA", "NVIDIA", "equity", "US", "XNAS", "USD")
    response = client.post("/api/instruments", json={"symbol": "nvidia"})
    assert response.status_code == 409
    assert response.json()["detail"] == "Das Symbol NVDA ist bereits in der Liste."
    assert fakes.data.history_requests == []


def test_a_broken_feed_registry_falls_back_to_feed_ids(
    client: TestClient, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .conftest import insert_news

    insert_news(services.db, "x", "Meldung", NOW_TS - 10, source="cnbc_top")

    class ExplodingFinder:
        def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
            if name == "trading_local.news.feeds":
                raise RuntimeError("feed ids must be unique")

    monkeypatch.delitem(sys.modules, "trading_local.news.feeds", raising=False)
    monkeypatch.setattr(sys, "meta_path", [ExplodingFinder(), *sys.meta_path])
    response = client.get("/api/news")
    assert response.status_code == 200
    assert response.json()[0]["source_name"] == "cnbc_top"


# ------------------------------------------------------------------------------------------------
# Bars
# ------------------------------------------------------------------------------------------------
def test_bars_never_include_a_bar_that_closes_after_now_or_before(client: TestClient, services: Services) -> None:
    seed_market(services.db)
    for tf, step in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
        body = client.get("/api/bars/NVDA", params={"tf": tf, "limit": 2000}).json()
        assert body["bars"] and all(bar[0] + step <= NOW_TS for bar in body["bars"]), tf
        cut = body["bars"][-3][0]
        older = client.get("/api/bars/NVDA", params={"tf": tf, "limit": 2000, "before": cut}).json()
        assert older["bars"] and all(bar[0] + step <= cut for bar in older["bars"]), tf
        assert all(value == value for line in older["overlays"].values() for _, value in line), tf


def test_daily_bars_without_history_are_an_empty_series(client: TestClient) -> None:
    body = client.get("/api/bars/NVDA", params={"tf": "1d"}).json()
    assert body["bars"] == [] and body["overlays"] == {"ema20": [], "ema50": [], "ema200": []}
