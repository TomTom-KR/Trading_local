"""Composition of the system status (``GET /api/status`` and the periodic WebSocket ``status`` push) and of the
session phase per venue (the hub announces phase changes as ``sessions`` messages)."""

import logging
import threading
import weakref
from collections.abc import Callable

import trading_local
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.dto import (
    DataStatusOut,
    DbStatusOut,
    LlmStatusOut,
    MailStatusOut,
    NewsStatusOut,
    SystemStatusOut,
)
from trading_local.presenters import mask_email
from trading_local.services import Services

log = logging.getLogger(__name__)

DB_STATS_TTL_S = 30  # db_stats counts every bar (linear in the table size): reuse a result for this long


class _DbStatsCache:
    """The last ``db_stats`` result of one database; thread-safe (status is composed on executor threads)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: DbStatusOut | None = None
        self.stored_ts = 0


_db_stats_caches: weakref.WeakKeyDictionary[Database, _DbStatsCache] = weakref.WeakKeyDictionary()
_caches_lock = threading.Lock()


def compose_status(services: Services) -> SystemStatusOut:
    """Blocking (database statistics): call it through the API executor, never on the event loop.

    A component whose ``status()`` fails is reported with its defaults instead of failing the whole
    status. The mail recipient is masked again here, so a full address can never leave the process.
    """
    settings = services.settings
    mail = _guarded("mail", services.mailer.status, MailStatusOut)
    return SystemStatusOut(
        server_time=services.clock.ts(),
        started_ts=services.runtime.started_ts,
        version=trading_local.__version__,
        engine_version=services.engine.version,
        base_currency=settings.base_currency,
        offline=services.runtime.offline,
        llm=_guarded("llm", services.llm.status, lambda: LlmStatusOut(model=settings.ollama_model)),
        data=_guarded("market data", services.data.status, DataStatusOut),
        news=_guarded("news", services.news.status, NewsStatusOut),
        mail=mail.model_copy(update={"recipient_masked": mask_email(mail.recipient_masked)}),
        db=_guarded("database", lambda: _cached_db_stats(services), DbStatusOut),
        jobs=_guarded("jobs", services.runtime.jobs, list),
    )


def venue_phases(services: Services) -> dict[str, str]:
    """Session phase ('open', 'break', 'pre', 'post', 'closed') of every venue that has an instrument, active or not.

    Blocking (instrument book, calendar): call it through the API executor. Venues the calendar does not know
    are left out; a failing calendar yields an empty result instead of an exception.
    """
    now = services.clock.now()
    phases: dict[str, str] = {}
    try:
        exchanges = sorted({instrument.exchange for instrument in services.instruments.all(active_only=False)})
        for exchange in exchanges:
            try:
                phases[exchange] = services.calendar.state(exchange, now).phase
            except KeyError:
                continue
    except Exception:
        log.exception("session phases of the venues are unavailable")
        return {}
    return phases


def _guarded[T](what: str, call: Callable[[], T], fallback: Callable[[], T]) -> T:
    try:
        return call()
    except Exception:
        log.exception("status of %s is unavailable", what)
        return fallback()


def _cached_db_stats(services: Services) -> DbStatusOut:
    """``repo_ops.db_stats`` at most once per ``DB_STATS_TTL_S`` seconds of the services' clock."""
    with _caches_lock:
        cache = _db_stats_caches.setdefault(services.db, _DbStatsCache())
    with cache.lock:
        now_ts = services.clock.ts()
        if cache.value is None or not 0 <= now_ts - cache.stored_ts < DB_STATS_TTL_S:
            cache.value = repo_ops.db_stats(services.db)
            cache.stored_ts = now_ts
        return cache.value
