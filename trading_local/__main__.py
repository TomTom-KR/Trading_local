"""Start the platform: ``python -m trading_local [--offline] [--port 8000] [--no-browser]``."""

import argparse
import asyncio
import logging
import sys
import threading
import webbrowser

from trading_local import __version__
from trading_local.config import Settings

log = logging.getLogger("trading_local")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m trading_local", description="Trading Local starten")
    parser.add_argument("--offline", action="store_true", help="keine Datenabrufe und Analysen, nur die Oberfläche")
    parser.add_argument("--port", type=int, default=None, help="Port des Webservice (Standard aus .env, sonst 8000)")
    parser.add_argument("--no-browser", action="store_true", help="den Browser nicht automatisch öffnen")
    return parser.parse_args(argv)


async def _serve(settings: Settings) -> None:
    import uvicorn

    from trading_local.runtime import Runtime, build_services
    from trading_local.web import create_app

    services = build_services(settings)
    runtime = services.runtime
    if not isinstance(runtime, Runtime):
        raise TypeError("build_services must return the background Runtime")
    app = create_app(services, on_startup=runtime.start, on_shutdown=runtime.stop)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
        log_config=None,  # keep our own logging configuration
    )
    server = uvicorn.Server(config)
    url = f"http://{'127.0.0.1' if settings.host in ('0.0.0.0', '::') else settings.host}:{settings.port}/"
    if settings.open_browser:
        threading.Timer(2.0, webbrowser.open, args=(url,)).start()
    log.info("Trading Local %s: %s", __version__, url)
    try:
        await server.serve()
    finally:
        services.db.close()


def main(argv: list[str] | None = None) -> int:
    from trading_local.logging_setup import setup_logging

    # sqlite3 releases the GIL once per fetched row; with the default 5 ms switch interval a busy Python
    # thread next to a database read makes the read thousands of times slower (measured on this stack).
    sys.setswitchinterval(0.0005)
    args = _parse_args(argv)
    overrides: dict[str, object] = {}
    if args.offline:
        overrides["offline"] = True
    if args.port is not None:
        overrides["port"] = args.port
    if args.no_browser:
        overrides["open_browser"] = False
    settings = Settings(**overrides)
    setup_logging(settings)
    if not settings.is_loopback:
        log.warning(
            "the web service listens on %s - it has no login; make sure only trusted devices can reach it",
            settings.host,
        )
    try:
        asyncio.run(_serve(settings))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
