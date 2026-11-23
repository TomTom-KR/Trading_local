"""Process-wide logging: console output plus a rotating log file in ``settings.log_dir``.

``setup_logging`` may be called any number of times (start-up, tests, changed settings): it replaces
the two handlers it owns and leaves handlers installed by others (pytest, uvicorn) alone.

Secrets never reach the log. The settings object is not logged, and the value of every ``SecretStr``
setting (the SMTP password) is masked in the complete output of both handlers - message, arguments
and exception tracebacks - should a library ever echo it.
"""

import logging
import threading
from logging.handlers import RotatingFileHandler

from pydantic import SecretStr

from trading_local.config import Settings

log = logging.getLogger(__name__)

LOG_FILE_NAME = "trading_local.log"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5
CONSOLE_HANDLER_NAME = "trading_local.console"
FILE_HANDLER_NAME = "trading_local.file"
# Third-party loggers that flood the log below WARNING.
NOISY_LOGGERS: tuple[str, ...] = (
    "yfinance", "peewee", "httpx", "httpcore", "urllib3", "matplotlib", "PIL", "asyncio", "websockets", "curl_cffi"
)

_MASK = "***"
_setup_lock = threading.Lock()


class _RedactingFormatter(logging.Formatter):
    """Formatter that masks secret values in the finished log text."""

    def __init__(self, fmt: str, secrets: tuple[str, ...]) -> None:
        super().__init__(fmt)
        self._secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for secret in self._secrets:
            text = text.replace(secret, _MASK)
        return text


def _secret_values(settings: Settings) -> tuple[str, ...]:
    """Plain values of all non-empty ``SecretStr`` settings, longest first.

    Longest first, so that a secret which contains another one is masked as a whole.
    """
    field_values = (getattr(settings, name) for name in type(settings).model_fields)
    secrets = {value.get_secret_value() for value in field_values if isinstance(value, SecretStr)}
    return tuple(sorted(secrets - {""}, key=lambda secret: (-len(secret), secret)))


def _level(name: str) -> int | None:
    """Numeric level for a level name such as 'INFO' or 'debug'; ``None`` if the name is unknown."""
    return logging.getLevelNamesMapping().get(name.strip().upper())


def setup_logging(settings: Settings, *, file_name: str = LOG_FILE_NAME) -> None:
    """Configure the root logger: console plus ``<log_dir>/<file_name>`` (5 MB x 5 backups, UTF-8).

    The level comes from ``settings.log_level`` (INFO if that is not a level name); the chatty
    third-party loggers in ``NOISY_LOGGERS`` are limited to WARNING. Idempotent: a repeated call replaces
    the handlers of the previous call instead of adding more.

    Auxiliary processes (the validation replay) must pass their own ``file_name``: on Windows a log
    rollover fails while another process holds the same file open.
    """
    level = _level(settings.log_level)
    formatter = _RedactingFormatter(LOG_FORMAT, _secret_values(settings))
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    handlers: dict[str, logging.Handler] = {
        CONSOLE_HANDLER_NAME: logging.StreamHandler(),
        FILE_HANDLER_NAME: RotatingFileHandler(
            settings.log_dir / file_name, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
        ),
    }
    root = logging.getLogger()
    with _setup_lock:
        for handler in [handler for handler in root.handlers if handler.name in handlers]:
            root.removeHandler(handler)
            handler.close()
        for name, handler in handlers.items():
            handler.name = name  # named only now: closing the predecessor unregisters its name
            handler.setFormatter(formatter)
            root.addHandler(handler)
        root.setLevel(logging.INFO if level is None else level)
        for name in NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
    if level is None:
        log.warning("unknown log level %r in the settings, using INFO", settings.log_level)
