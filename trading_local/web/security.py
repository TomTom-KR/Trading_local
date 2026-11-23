"""Protection of the login-free local service against other web pages open in the user's browser.

``SecurityMiddleware`` (pure ASGI, outermost) enforces for every HTTP request and WebSocket handshake:

1. Host allow-list (DNS-rebinding protection): 127.0.0.1, localhost and [::1] with the configured port,
   plus ``settings.host`` when the user deliberately binds elsewhere -> 400.
2. Same origin: state-changing requests (POST/PUT/PATCH/DELETE) and WebSocket handshakes that carry an
   ``Origin`` must come from the request's own host; requests without ``Origin`` (non-browser clients)
   pass -> 403.
3. State-changing requests with a body or a Content-Type must send ``application/json``, so "simple"
   cross-site form posts never reach a handler -> 415.
4. No CORS headers are ever added (cross-origin reads stay blocked by the browser).
5. Security headers on every HTTP response.

A WebSocket handshake that violates rule 1 or 2 is closed before it is accepted (code 1008); the server then
answers the handshake with HTTP 403. This is the one rejection every ASGI server completes cleanly: uvicorn's
default WebSocket protocol logs an explicit denial response ('websocket.http.response') as an error
("ASGI callable returned without completing handshake"), and browsers never show a handshake's response
body to the page anyway.
"""

import logging
from collections.abc import Collection, Mapping
from urllib.parse import urlsplit

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from trading_local.config import Settings

log = logging.getLogger(__name__)

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self'; connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)
SECURITY_HEADERS: Mapping[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
}
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
WS_POLICY_VIOLATION = 1008  # close code of a refused WebSocket handshake

_LOOPBACK_NAMES = ("127.0.0.1", "localhost", "[::1]")
_BAD_HOST_DE = "Ungültiger Host-Header: Die Oberfläche ist nur über die eingestellte Adresse erreichbar."
_FOREIGN_ORIGIN_DE = "Anfrage von einer fremden Webseite abgelehnt."
_NOT_JSON_DE = "Nur JSON-Anfragen werden angenommen (Content-Type: application/json)."


def allowed_hosts(settings: Settings) -> frozenset[str]:
    """Accepted ``Host`` header values (lower case). Without an explicit port browsers mean port 80."""
    names = list(_LOOPBACK_NAMES)
    if not settings.is_loopback:
        names.append(_host_name(settings.host))
    hosts = {f"{name}:{settings.port}" for name in names}
    if settings.port == 80:
        hosts.update(names)
    return frozenset(host.lower() for host in hosts)


def _host_name(host: str) -> str:
    """IPv6 literals appear in brackets in the Host header."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _same_origin(origin: str, host: str) -> bool:
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and parts.netloc.lower() == host


def _is_json(content_type: str | None) -> bool:
    return content_type is not None and content_type.split(";", 1)[0].strip().lower() == "application/json"


def _carries_body(headers: Headers) -> bool:
    length = headers.get("content-length")
    return (length is not None and length.strip() != "0") or "transfer-encoding" in headers


class SecurityMiddleware:
    def __init__(self, app: ASGIApp, *, hosts: Collection[str]) -> None:
        self.app = app
        self._hosts = frozenset(host.lower() for host in hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        rejection = self._check(scope)
        if rejection is not None:
            status, detail = rejection
            log.warning("rejected %s %s: %s", scope["type"], scope.get("path", ""), detail)
            await _reject(scope, receive, send, status, detail)
            return
        if scope["type"] == "websocket":
            await self.app(scope, receive, send)
            return
        await self.app(scope, receive, _with_security_headers(send))

    def _check(self, scope: Scope) -> tuple[int, str] | None:
        """(status, German detail) of the first violated rule, or None if the request may pass."""
        headers = Headers(scope=scope)
        host = headers.get("host", "").lower()
        if host not in self._hosts:
            return 400, _BAD_HOST_DE
        state_changing = scope["type"] == "http" and scope["method"] in STATE_CHANGING_METHODS
        if scope["type"] == "websocket" or state_changing:
            origin = headers.get("origin")
            if origin is not None and not _same_origin(origin, host):
                return 403, _FOREIGN_ORIGIN_DE
        if state_changing and ("content-type" in headers or _carries_body(headers)):
            if not _is_json(headers.get("content-type")):
                return 415, _NOT_JSON_DE
        return None


def _with_security_headers(send: Send) -> Send:
    async def send_with_headers(message: Message) -> None:
        if message["type"] == "http.response.start":
            headers = MutableHeaders(scope=message)
            for name, value in SECURITY_HEADERS.items():
                if name not in headers:
                    headers[name] = value
        await send(message)

    return send_with_headers


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, detail: str) -> None:
    """Answer an HTTP request with a JSON error; close a WebSocket handshake before accepting it (HTTP 403)."""
    if scope["type"] == "http":
        response = JSONResponse({"detail": detail}, status_code=status, headers=dict(SECURITY_HEADERS))
        await response(scope, receive, send)
        return
    await send({"type": "websocket.close", "code": WS_POLICY_VIOLATION, "reason": ""})
