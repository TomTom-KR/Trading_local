"""The FastAPI application: REST API under /api, the WebSocket hub at /ws, the GUI files at / , security
checks and German error responses."""

import asyncio
import contextlib
import logging
import mimetypes
from collections.abc import AsyncIterator, Awaitable, Callable
from http import HTTPStatus
from os import PathLike, stat_result
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.staticfiles import NotModifiedResponse, StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocketClose

import trading_local
from trading_local.services import Services
from trading_local.web.deps import ApiContext, validation_detail
from trading_local.web.routes import (
    analysis,
    instruments,
    mail,
    market,
    news,
    portfolios,
    settings,
    signals,
    system,
    validation,
)
from trading_local.web.security import SecurityMiddleware, allowed_hosts
from trading_local.web.ws import Hub
from trading_local.web.ws import router as ws_router

log = logging.getLogger(__name__)

# Windows takes MIME types from the registry, which maps .mjs (and sometimes .js) to text/plain: browsers
# then refuse to run the GUI's ES modules. Register the types the GUI needs explicitly.
for _extension, _media_type in {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".map": "application/json",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}.items():
    mimetypes.add_type(_media_type, _extension)

DEFAULT_STATIC_DIR = Path(__file__).with_name("static")
INDEX_CACHE_CONTROL = "no-cache"  # always revalidate the entry page
ASSET_CACHE_CONTROL = "public, max-age=60"  # file names carry no version: keep the lifetime short

INTERNAL_ERROR_DE = "Interner Fehler. Die Einzelheiten stehen im Protokoll des Servers."
_STATUS_DETAILS_DE: dict[int, str] = {
    400: "Ungültige Anfrage.",
    403: "Zugriff verweigert.",
    404: "Nicht gefunden.",
    405: "Diese Methode ist für diese Adresse nicht erlaubt.",
    409: "Konflikt mit dem aktuellen Zustand.",
    413: "Die Anfrage ist zu groß.",
    415: "Nicht unterstütztes Datenformat.",
    422: "Ungültige Eingabe.",
    500: INTERNAL_ERROR_DE,
    503: "Dienst vorübergehend nicht verfügbar.",
}
# English details that FastAPI itself puts into HTTP errors (e.g. a JSON body that is not valid UTF-8).
_FRAMEWORK_DETAILS_DE: dict[str, str] = {
    "There was an error parsing the body": "Der Anfrageinhalt konnte nicht gelesen werden (kein gültiges JSON in UTF-8).",
}
_UNKNOWN_API_PATH_DE = "Unbekannter API-Pfad."
_MISSING_GUI_HTML = """<!doctype html>
<html lang="de"><head><meta charset="utf-8"><title>Trading Local</title><link rel="icon" href="data:,"></head>
<body><h1>Trading Local</h1>
<p>Die Oberfläche ist nicht installiert (das Verzeichnis web/static oder dessen index.html fehlt).
Die Programmierschnittstelle ist unter /api erreichbar.</p></body></html>
"""

_API_ROUTERS = (
    system.router,
    instruments.router,
    market.router,
    analysis.router,
    news.router,
    signals.router,
    portfolios.router,
    settings.router,
    mail.router,
    validation.router,
)


def create_app(
    services: Services,
    *,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build the web application around the wired services.

    The lifespan binds ``services.bus`` to the running loop, starts the WebSocket hub, then awaits
    ``on_startup`` (the runtime starts the background services there) and loads the instrument book on the
    API executor, so that handlers never trigger its first database read on the event loop. On shutdown it
    awaits ``on_shutdown`` (only if the start-up hook succeeded), stops the hub, cancels the API's background
    tasks and releases its executor. ``static_dir`` defaults to ``trading_local/web/static``.
    """
    ctx = ApiContext(services)
    hub = Hub(ctx)
    gui_dir = DEFAULT_STATIC_DIR if static_dir is None else Path(static_dir)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        services.bus.bind_loop(asyncio.get_running_loop())
        await hub.start()
        try:
            if on_startup is not None:
                await on_startup()
            try:
                await ctx.run(services.instruments.all)
                yield
            finally:
                if on_shutdown is not None:
                    await on_shutdown()
        finally:
            await hub.stop()
            await ctx.aclose()

    app = FastAPI(
        title="Trading Local",
        version=trading_local.__version__,
        lifespan=lifespan,
        openapi_url="/api/openapi.json",
        docs_url=None,  # the documentation pages load scripts from a CDN, which the CSP forbids
        redoc_url=None,
        middleware=[
            Middleware(SecurityMiddleware, hosts=allowed_hosts(services.settings)),
            Middleware(ErrorMiddleware),
        ],
        exception_handlers={
            StarletteHTTPException: _http_error,
            RequestValidationError: _validation_error,
        },
    )
    app.state.ctx = ctx
    app.state.hub = hub
    app.state.sparklines = market.build_sparkline_cache(ctx)
    app.state.validation = validation.ValidationRunner(ctx)
    for router in _API_ROUTERS:
        app.include_router(router)
    app.include_router(ws_router)
    app.add_api_route("/", _index_endpoint(gui_dir), methods=["GET", "HEAD"], include_in_schema=False)
    # Paths no route matches (also no partial match, which yields 405) end up here.
    app.router.default = GuiFallback(gui_dir)
    return app


# ------------------------------------------------------------------------------------------------
# GUI files
# ------------------------------------------------------------------------------------------------
def _index_endpoint(directory: Path) -> Callable[[], Awaitable[Response]]:
    async def gui_index() -> Response:
        index = directory / "index.html"
        if index.is_file():
            return FileResponse(index, headers={"Cache-Control": INDEX_CACHE_CONTROL})
        return HTMLResponse(_MISSING_GUI_HTML, headers={"Cache-Control": INDEX_CACHE_CONTROL})

    return gui_index


class _GuiFiles(StaticFiles):
    """Static files with a short cache lifetime (ETag revalidation as in Starlette); HTML pages, like the
    entry page at '/', are always revalidated."""

    def file_response(
        self, full_path: PathLike[str] | str, stat_result: stat_result, scope: Scope, status_code: int = 200
    ) -> Response:
        is_page = str(full_path).lower().endswith((".html", ".htm"))
        cache_control = INDEX_CACHE_CONTROL if is_page else ASSET_CACHE_CONTROL
        response = FileResponse(
            full_path, status_code=status_code, stat_result=stat_result, headers={"Cache-Control": cache_control}
        )
        if self.is_not_modified(response.headers, Headers(scope=scope)):
            return NotModifiedResponse(response.headers)
        return response


class GuiFallback:
    """Router default: GUI assets for every other path; JSON 404 under /api and /ws, which never fall
    through to files. The GUI uses hash routing, so no SPA fallback beyond '/' is needed."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._files = _GuiFiles(directory=directory, check_dir=False)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            await WebSocketClose()(scope, receive, send)
            return
        path: str = scope["path"]
        if _is_reserved(path):
            await JSONResponse({"detail": _UNKNOWN_API_PATH_DE}, status_code=404)(scope, receive, send)
            return
        if not self._directory.is_dir():
            raise StarletteHTTPException(status_code=404)
        await self._files(scope, receive, send)


def _is_reserved(path: str) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in ("/api", "/ws"))


# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------
async def _http_error(request: Request, exc: Exception) -> Response:
    """HTTP errors as ``{"detail": "<German text>"}``; Starlette's English default phrases are translated."""
    if not isinstance(exc, StarletteHTTPException):  # registered for this type only
        raise exc
    detail = exc.detail
    if isinstance(detail, str) and detail in _FRAMEWORK_DETAILS_DE:
        detail = _FRAMEWORK_DETAILS_DE[detail]
    elif not isinstance(detail, str) or detail == _phrase(exc.status_code):
        detail = _STATUS_DETAILS_DE.get(exc.status_code, "Die Anfrage konnte nicht bearbeitet werden.")
    return JSONResponse({"detail": detail}, status_code=exc.status_code, headers=exc.headers)


async def _validation_error(request: Request, exc: Exception) -> Response:
    if not isinstance(exc, RequestValidationError):  # registered for this type only
        raise exc
    return JSONResponse({"detail": validation_detail(exc.errors())}, status_code=422)


def _phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""


class ErrorMiddleware:
    """Unhandled exceptions are logged with their traceback and answered with a generic German 500
    (no internals leave the process). Sits inside ``SecurityMiddleware``, so the answer carries its headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracking_send(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            log.exception("unhandled error in %s %s", scope["method"], scope["path"])
            if started:
                raise
            await JSONResponse({"detail": INTERNAL_ERROR_DE}, status_code=500)(scope, receive, send)
