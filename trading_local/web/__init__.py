"""Web layer: REST API, WebSocket hub and the browser GUI's files.

Public entry point for the runtime::

    create_app(services, *, on_startup=None, on_shutdown=None, static_dir=None) -> fastapi.FastAPI
"""

from trading_local.web.app import create_app

__all__ = ["create_app"]
