"""Host allow-list, same-origin rule, JSON-only writes, security headers and secret hygiene."""

import logging
import re
import socket
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket as StarletteWebSocket
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from trading_local.db import repo_ops
from trading_local.dto import SessionsOut
from trading_local.services import Services
from trading_local.testing import make_settings
from trading_local.web.security import (
    CONTENT_SECURITY_POLICY,
    WS_POLICY_VIOLATION,
    SecurityMiddleware,
    allowed_hosts,
)

from .conftest import MAIL_FROM, MAIL_TO, NOW_TS, ORIGIN, SMTP_PASSWORD, WS_URL, insert_news, insert_signal, seed_market

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "x-frame-options": "DENY",
    "content-security-policy": CONTENT_SECURITY_POLICY,
}


def assert_security_headers(response: object) -> None:
    headers = response.headers  # type: ignore[attr-defined]
    for name, value in SECURITY_HEADERS.items():
        assert headers.get(name) == value, name
    assert not any(name.lower().startswith("access-control-") for name in headers)


# ------------------------------------------------------------------------------------------------
# Host header
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("host", ["evil.example:8000", "127.0.0.1:9999", "127.0.0.1", "localhost.evil:8000", ""])
def test_foreign_host_headers_are_rejected(client: TestClient, host: str) -> None:
    response = client.get("/api/status", headers={"host": host})
    assert response.status_code == 400
    assert "Host" in response.json()["detail"]
    assert_security_headers(response)


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "LOCALHOST:8000", "[::1]:8000"])
def test_loopback_host_headers_pass(client: TestClient, host: str) -> None:
    assert client.get("/api/status", headers={"host": host}).status_code == 200


def test_allowed_hosts_add_a_deliberate_bind_address(tmp_path: Path) -> None:
    assert allowed_hosts(make_settings(tmp_path)) == {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"}
    lan = allowed_hosts(make_settings(tmp_path, host="192.168.1.20", port=8080))
    assert lan == {"127.0.0.1:8080", "localhost:8080", "[::1]:8080", "192.168.1.20:8080"}
    assert "[fd00::5]:8000" in allowed_hosts(make_settings(tmp_path, host="fd00::5"))
    assert {"localhost", "localhost:80"} <= allowed_hosts(make_settings(tmp_path, port=80))


# ------------------------------------------------------------------------------------------------
# Origin and content type of state-changing requests
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("origin", ["http://evil.example", "http://localhost:8000", "null", "http://127.0.0.1:8001"])
def test_cross_site_writes_are_rejected(client: TestClient, origin: str) -> None:
    response = client.post("/api/mail/test", headers={"origin": origin})
    assert response.status_code == 403
    assert response.json()["detail"] == "Anfrage von einer fremden Webseite abgelehnt."
    assert_security_headers(response)
    response = client.put("/api/settings", json={"mail_enabled": False}, headers={"origin": origin})
    assert response.status_code == 403


def test_same_origin_and_origin_less_writes_pass(client: TestClient) -> None:
    assert client.post("/api/mail/test", headers={"origin": ORIGIN}).status_code == 200
    assert client.post("/api/mail/test").status_code == 200


def test_reads_from_other_origins_get_no_cors_headers(client: TestClient) -> None:
    response = client.get("/api/status", headers={"origin": "http://evil.example"})
    assert response.status_code == 200
    assert_security_headers(response)
    preflight = client.options(
        "/api/settings", headers={"origin": "http://evil.example", "access-control-request-method": "PUT"}
    )
    assert preflight.status_code == 405
    assert_security_headers(preflight)


@pytest.mark.parametrize(
    "content_type", ["application/x-www-form-urlencoded", "text/plain", "multipart/form-data; boundary=x"]
)
def test_writes_must_be_json(client: TestClient, content_type: str) -> None:
    response = client.put("/api/settings", content=b"mail_enabled=false", headers={"content-type": content_type})
    assert response.status_code == 415
    assert "application/json" in response.json()["detail"]
    response = client.post("/api/mail/test", headers={"content-type": content_type})
    assert response.status_code == 415


def test_writes_with_a_body_but_no_content_type_are_rejected(client: TestClient) -> None:
    response = client.put("/api/settings", content=b'{"mail_enabled": false}', headers={"content-type": ""})
    assert response.status_code == 415


def test_json_with_charset_is_accepted(client: TestClient) -> None:
    response = client.put(
        "/api/settings", content=b'{"mail_enabled": false}', headers={"content-type": "application/json; charset=utf-8"}
    )
    assert response.status_code == 200
    assert response.json()["mail_enabled"] is False


# ------------------------------------------------------------------------------------------------
# WebSocket handshake
# ------------------------------------------------------------------------------------------------
def test_cross_site_websocket_handshakes_are_closed_before_they_are_accepted(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect(WS_URL, headers={"origin": "http://evil.example"}):
            pass
    assert denied.value.code == WS_POLICY_VIOLATION


def test_websocket_handshakes_with_a_foreign_host_are_closed_before_they_are_accepted(client: TestClient) -> None:
    with pytest.raises(WebSocketDisconnect) as denied:
        with client.websocket_connect("ws://evil.example:8000/ws"):
            pass
    assert denied.value.code == WS_POLICY_VIOLATION


@pytest.mark.parametrize(
    ("allowed_host", "origin"),
    [("{bound}", "http://evil.example"), ("trading.invalid:8000", None)],
    ids=["foreign-origin", "foreign-host"],
)
def test_uvicorn_answers_a_refused_handshake_with_403_and_logs_no_error(
    caplog: pytest.LogCaptureFixture, allowed_host: str, origin: str | None
) -> None:
    """Through a real uvicorn server: an explicit denial response made uvicorn's default WebSocket protocol log
    'ASGI callable returned without completing handshake.' as an ERROR for every refused handshake."""

    async def endpoint(websocket: StarletteWebSocket) -> None:  # pragma: no cover - must never be reached
        await websocket.accept()
        await websocket.close()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    bound = f"127.0.0.1:{sock.getsockname()[1]}"
    app = SecurityMiddleware(Starlette(routes=[WebSocketRoute("/ws", endpoint)]),
                             hosts=[allowed_host.format(bound=bound)])
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="off", ws="auto"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    caplog.set_level(logging.INFO, logger="uvicorn")
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "uvicorn did not start"
            time.sleep(0.02)
        with pytest.raises(InvalidStatus) as refused:
            with connect(f"ws://{bound}/ws", origin=origin, open_timeout=5):  # type: ignore[arg-type]
                pass
        assert refused.value.response.status_code == 403
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
    errors = [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]
    assert errors == []
    assert any("rejected websocket /ws" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("headers", [{"origin": ORIGIN}, {}])
def test_same_origin_and_origin_less_websockets_connect(client: TestClient, headers: dict[str, str]) -> None:
    with client.websocket_connect(WS_URL, headers=headers) as websocket:
        assert websocket.receive_json()["type"] == "hello"


# ------------------------------------------------------------------------------------------------
# Headers everywhere, secrets nowhere
# ------------------------------------------------------------------------------------------------
def test_every_kind_of_response_carries_the_security_headers(client: TestClient) -> None:
    for path in ("/", "/app.mjs", "/api/status", "/api/unknown", "/missing.css", "/api/bars/NOPE"):
        assert_security_headers(client.get(path))
    assert_security_headers(client.get("/api/news", params={"limit": 0}))


def test_no_get_endpoint_leaks_the_smtp_password_or_full_mail_addresses(
    client: TestClient, app: FastAPI, services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = services.db
    seed_market(db)
    news_id = insert_news(db, "n1", "NVIDIA meldet Rekordumsatz", NOW_TS - 600, symbols={"NVDA": 0.6})
    signal = insert_signal(db, "NVDA", status="active", actionable=True, created_ts=NOW_TS - 300, news_ids=[news_id])
    repo_ops.log_mail(db, ts=NOW_TS - 60, kind="signal", recipient=f"Max <{MAIL_TO}>", subject="LONG NVDA",
                      status="file", signal_id=signal.id)
    repo_ops.log_activity(db, ts=NOW_TS - 50, level="info", category="mail", message="Mail geschrieben.")
    overview = types.ModuleType("trading_local.markets.overview")
    overview.sessions_overview = lambda calendar, instruments, now: SessionsOut(  # type: ignore[attr-defined]
        now=NOW_TS, regions=[], timeline_start=NOW_TS, timeline_end=NOW_TS, timeline=[]
    )
    monkeypatch.setitem(sys.modules, "trading_local.markets.overview", overview)
    values = {"symbol": "NVDA", "key": "me", "signal_id": str(signal.id)}
    paths = sorted(
        re.sub(r"\{(\w+)\}", lambda match: values[match.group(1)], path)
        for path, operations in app.openapi()["paths"].items()
        if "get" in operations
    )
    assert len(paths) >= 24  # every GET endpoint of the API
    for path in paths:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text)
        for secret in (SMTP_PASSWORD, MAIL_TO, MAIL_FROM):
            assert secret not in response.text, (path, secret)
    assert client.get("/api/mail/log").json()[0]["recipient"] == "e***@example.invalid"
    assert client.get("/api/status").json()["mail"]["recipient_masked"] == "e***@example.invalid"
