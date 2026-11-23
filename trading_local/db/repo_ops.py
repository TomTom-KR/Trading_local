"""Operations repository: key/value settings and the runtime-settings store, the mail, LLM-call, job-run
and activity logs, validation runs, database statistics and log retention.

These tables have no domain dataclass; the readers return the API models of ``trading_local.dto`` (or
small typed dicts) directly. All functions are safe to call from several threads.
"""

import logging
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from pydantic import ValidationError

from trading_local.config import RUNTIME_SETTINGS_KEY, RuntimeSettings
from trading_local.db.database import Database, dumps, loads
from trading_local.dto import ActivityOut, DbStatusOut, LlmCallOut, MailLogOut, ValidationResultOut, ValidationRunOut
from trading_local.events import TOPIC_ACTIVITY, EventBus

log = logging.getLogger(__name__)

# Mail-log statuses that mean a mail really went out (SMTP) or was written to the outbox (file mode).
DELIVERED_MAIL_STATUSES: tuple[str, ...] = ("sent", "file")

_JOB_RUNS_KEPT = 500  # per job
_SECONDS_PER_DAY = 86_400
_BYTES_PER_MB = 1024 * 1024
_UNREADABLE = object()  # marks stored JSON that could not be parsed

_LLM_CALL_STATS = """
SELECT COUNT(*) AS calls,
       COUNT(*) FILTER (WHERE ok = 0) AS errors,
       AVG(latency_ms) FILTER (WHERE ok = 1) AS avg_latency_ms,
       (SELECT latency_ms FROM llm_calls WHERE ts >= :since_ts AND ok = 1
        ORDER BY ts DESC, id DESC LIMIT 1) AS last_latency_ms,
       (SELECT error FROM llm_calls WHERE ts >= :since_ts AND ok = 0
        ORDER BY ts DESC, id DESC LIMIT 1) AS last_error
FROM llm_calls
WHERE ts >= :since_ts
"""

_LATEST_JOB_RUNS = """
SELECT job, started_ts, duration_ms, ok, message
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY job ORDER BY started_ts DESC, id DESC) AS recency
    FROM job_runs
)
WHERE recency = 1
"""

_SELECT_VALIDATION_RUN = "SELECT id, ts, status, params_json, result_json, engine_version, message FROM validation_runs"

# The CTE hops from symbol to symbol with one primary-key seek each (SQLite would otherwise scan the whole
# table to find the distinct symbols); the first and the last bar of a symbol are one seek each as well.
_DB_STATS = """
WITH RECURSIVE symbols (symbol) AS (
    SELECT MIN(symbol) FROM bars_1m
    UNION ALL
    SELECT (SELECT MIN(symbol) FROM bars_1m WHERE symbol > symbols.symbol) FROM symbols WHERE symbol IS NOT NULL
)
SELECT (SELECT COUNT(*) FROM bars_1m) AS bars_1m,
       (SELECT MIN((SELECT MIN(ts) FROM bars_1m WHERE symbol = symbols.symbol)) FROM symbols) AS first_bar_ts,
       (SELECT MAX((SELECT MAX(ts) FROM bars_1m WHERE symbol = symbols.symbol)) FROM symbols) AS last_bar_ts,
       (SELECT COUNT(*) FROM news) AS news,
       (SELECT COUNT(*) FROM signals) AS signals
"""


class LlmCallStats(TypedDict):
    calls: int
    errors: int
    last_latency_ms: int | None  # newest successful call
    avg_latency_ms: float | None  # successful calls only
    last_error: str


class JobRun(TypedDict):
    job: str
    started_ts: int
    duration_ms: int
    ok: bool
    message: str


# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------
def get_setting(db: Database, key: str, default: Any = None) -> Any:
    """The JSON value stored under ``key``; ``default`` if the key is missing or does not hold valid JSON."""
    text = db.scalar("SELECT value FROM settings WHERE key = ?", (key,))
    if text is None:
        return default
    value = loads(text, _UNREADABLE)
    if value is _UNREADABLE:
        log.warning("setting %r does not hold valid JSON; using the default", key)
        return default
    return value


def set_setting(db: Database, key: str, value: Any, now_ts: int) -> None:
    """Store ``value`` as JSON under ``key`` (insert or replace)."""
    db.execute(
        "INSERT INTO settings (key, value, updated_ts) VALUES (?, ?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value, updated_ts = excluded.updated_ts",
        (key, dumps(value), now_ts),
    )


class SettingsStore:
    """Thread-safe, cached access to the user's ``RuntimeSettings`` (settings key ``runtime``).

    The stored row is the source of truth. The cache only ever holds committed data that was read from it,
    and ``update`` reads, validates and writes inside one write transaction, so concurrent partial updates
    from several threads cannot lose each other. ``get`` never waits for a writer.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = threading.Lock()  # guards _cached; never held while waiting for the database write lock
        self._cached: RuntimeSettings | None = None

    def get(self) -> RuntimeSettings:
        """The current settings as a private copy.

        Defaults apply to everything that is not stored. Stored values that are unknown or invalid are
        ignored field by field, so a damaged row can never stop the platform.
        """
        with self._lock:
            if self._cached is None:
                self._cached = self._load()
            return self._cached.model_copy(deep=True)

    def update(self, patch: Mapping[str, Any], now_ts: int) -> RuntimeSettings:
        """Merge ``patch`` over the current settings, validate, persist and return the new settings.

        Raises ``ValueError`` for keys that are no runtime settings and ``pydantic.ValidationError`` for
        invalid values; in both cases nothing is stored. (``ValidationError`` is a ``ValueError`` subclass:
        catch it first to tell the two apart.)

        Must not be called inside a caller's ``db.write()`` block (``RuntimeError``): the cache may only be
        refreshed once the new row is committed, and a surrounding transaction commits or rolls back later.
        """
        if self._db.conn().in_transaction:
            raise RuntimeError("SettingsStore.update must not be called inside a database transaction")
        unknown = sorted(set(patch) - set(RuntimeSettings.model_fields))
        if unknown:
            raise ValueError(f"unknown runtime settings: {', '.join(unknown)}")
        with self._db.write():
            updated = RuntimeSettings.model_validate({**self._load().model_dump(), **patch})
            set_setting(self._db, RUNTIME_SETTINGS_KEY, updated.model_dump(mode="json"), now_ts)
        with self._lock:
            self._cached = None  # the next get() reads the committed row
        return updated

    def _load(self) -> RuntimeSettings:
        stored = get_setting(self._db, RUNTIME_SETTINGS_KEY, {})
        if not isinstance(stored, dict):
            log.warning("stored runtime settings are not a JSON object; using the defaults")
            stored = {}
        try:
            return RuntimeSettings.model_validate(stored)
        except ValidationError as exc:
            invalid = {str(error["loc"][0]) for error in exc.errors()}
            log.warning("stored runtime settings: using the defaults for invalid %s", ", ".join(sorted(invalid)))
            return RuntimeSettings.model_validate({key: value for key, value in stored.items() if key not in invalid})


# ------------------------------------------------------------------------------------------------
# Mail log
# ------------------------------------------------------------------------------------------------
def log_mail(
    db: Database,
    *,
    ts: int,
    kind: str,
    recipient: str,
    subject: str,
    status: str,
    detail: str = "",
    signal_id: int | None = None,
) -> int:
    """Record one mail attempt (status 'sent', 'file', 'skipped' or 'failed') and return the id of the entry."""
    return db.execute(
        "INSERT INTO mail_log (ts, kind, signal_id, recipient, subject, status, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, kind, signal_id, recipient, subject, status, detail),
    )


def list_mail_log(db: Database, *, limit: int = 100) -> list[MailLogOut]:
    """Mail attempts, newest first. Recipients are returned as stored (unmasked)."""
    rows = db.query(
        "SELECT id, ts, kind, signal_id, recipient, subject, status, detail FROM mail_log "
        "ORDER BY ts DESC, id DESC LIMIT ?",
        (limit,),
    )
    return [MailLogOut.model_validate(dict(row)) for row in rows]


def count_mails(db: Database, *, since_ts: int, statuses: Sequence[str] = DELIVERED_MAIL_STATUSES) -> int:
    """Number of mail attempts since ``since_ts`` with one of ``statuses`` (default: the delivered ones)."""
    marks = ", ".join("?" * len(statuses))
    return int(db.scalar(f"SELECT COUNT(*) FROM mail_log WHERE ts >= ? AND status IN ({marks})", (since_ts, *statuses)))


def last_mail_ts_for_symbol(db: Database, symbol: str) -> int | None:
    """Time of the newest delivered 'signal' or 'exit' mail whose signal belongs to ``symbol`` (None if none)."""
    marks = ", ".join("?" * len(DELIVERED_MAIL_STATUSES))
    ts: int | None = db.scalar(
        "SELECT MAX(m.ts) FROM mail_log m JOIN signals s ON s.id = m.signal_id "
        f"WHERE s.symbol = ? AND m.kind IN ('signal', 'exit') AND m.status IN ({marks})",
        (symbol, *DELIVERED_MAIL_STATUSES),
    )
    return ts


def last_mail_error(db: Database) -> str:
    """Detail text of the newest failed mail - or '' if a mail was delivered after it (the problem is over)
    or nothing failed. Skipped mails are neither a failure nor a delivery and are ignored."""
    # Newest attempt that was a failure or a delivery. MAX(ts) walks mail_log_ts_idx down from the newest
    # entry and stops at the first match; the outer query then only looks at the attempts of that second.
    # (One ORDER BY ts DESC, id DESC over the filtered rows would read and sort the whole log on every call.)
    outcome = f"status IN ('failed', {', '.join('?' * len(DELIVERED_MAIL_STATUSES))})"
    row = db.query_one(
        f"SELECT status, detail FROM mail_log "
        f"WHERE {outcome} AND ts = (SELECT MAX(ts) FROM mail_log WHERE {outcome}) ORDER BY id DESC LIMIT 1",
        (*DELIVERED_MAIL_STATUSES, *DELIVERED_MAIL_STATUSES),
    )
    return str(row["detail"]) if row is not None and row["status"] == "failed" else ""


# ------------------------------------------------------------------------------------------------
# LLM calls
# ------------------------------------------------------------------------------------------------
def log_llm_call(
    db: Database,
    *,
    ts: int,
    kind: str,
    ok: bool,
    latency_ms: int,
    queue_wait_ms: int = 0,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    error: str = "",
) -> int:
    """Record one model call and return the id of the entry."""
    return db.execute(
        "INSERT INTO llm_calls (ts, kind, ok, latency_ms, queue_wait_ms, prompt_tokens, output_tokens, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, kind, ok, latency_ms, queue_wait_ms, prompt_tokens, output_tokens, error),
    )


def list_llm_calls(db: Database, *, limit: int = 100) -> list[LlmCallOut]:
    """Model calls, newest first."""
    rows = db.query(
        "SELECT id, ts, kind, ok, latency_ms, queue_wait_ms, prompt_tokens, output_tokens, error FROM llm_calls "
        "ORDER BY ts DESC, id DESC LIMIT ?",
        (limit,),
    )
    return [LlmCallOut.model_validate(dict(row)) for row in rows]


def llm_call_stats(db: Database, *, since_ts: int) -> LlmCallStats:
    """Statistics of the model calls since ``since_ts``.

    ``calls`` and ``errors`` count all and the failed calls. The latencies describe successful calls only
    (a timeout or a refused connection says nothing about the model's speed): ``last_latency_ms`` is the
    newest one, ``avg_latency_ms`` their mean; both are None without a successful call. ``last_error`` is
    the error text of the newest failed call ('' if none failed).
    """
    row = db.query(_LLM_CALL_STATS, {"since_ts": since_ts})[0]  # an aggregate without GROUP BY yields one row
    return LlmCallStats(
        calls=row["calls"],
        errors=row["errors"],
        last_latency_ms=row["last_latency_ms"],
        avg_latency_ms=row["avg_latency_ms"],
        last_error=row["last_error"] or "",
    )


# ------------------------------------------------------------------------------------------------
# Job runs
# ------------------------------------------------------------------------------------------------
def log_job_run(db: Database, *, job: str, started_ts: int, duration_ms: int, ok: bool, message: str = "") -> None:
    """Record one run of a background job. Only the newest 500 runs of each job are kept."""
    with db.write() as conn:
        conn.execute(
            "INSERT INTO job_runs (job, started_ts, duration_ms, ok, message) VALUES (?, ?, ?, ?, ?)",
            (job, started_ts, duration_ms, ok, message),
        )
        conn.execute(
            "DELETE FROM job_runs WHERE id IN ("
            "SELECT id FROM job_runs WHERE job = ? ORDER BY started_ts DESC, id DESC LIMIT -1 OFFSET ?)",
            (job, _JOB_RUNS_KEPT),
        )


def latest_job_runs(db: Database) -> dict[str, JobRun]:
    """The newest run of every job, by job name."""
    return {
        row["job"]: JobRun(
            job=row["job"],
            started_ts=row["started_ts"],
            duration_ms=row["duration_ms"],
            ok=bool(row["ok"]),
            message=row["message"],
        )
        for row in db.query(_LATEST_JOB_RUNS)
    }


# ------------------------------------------------------------------------------------------------
# Activity log
# ------------------------------------------------------------------------------------------------
def log_activity(
    db: Database,
    *,
    ts: int,
    level: str,
    category: str,
    message: str,
    ref_type: str | None = None,
    ref_id: int | None = None,
    bus: EventBus | None = None,
) -> int:
    """Append an entry to the activity log (``message`` is German and shown in the GUI) and return its id.

    ``level`` is 'info', 'warning' or 'error'; any other value raises ``sqlite3.IntegrityError``.
    With a ``bus`` the stored entry is also announced on ``TOPIC_ACTIVITY``; the payload is the
    ``ActivityOut`` as a dict. Inside a caller's ``db.write()`` block the announcement precedes the commit.
    """
    entry = {
        "ts": ts,
        "level": level,
        "category": category,
        "message": message,
        "ref_type": ref_type,
        "ref_id": ref_id,
    }
    activity_id = db.execute(
        "INSERT INTO activity_log (ts, level, category, message, ref_type, ref_id) "
        "VALUES (:ts, :level, :category, :message, :ref_type, :ref_id)",
        entry,
    )
    if bus is not None:
        bus.publish(TOPIC_ACTIVITY, ActivityOut.model_validate({"id": activity_id, **entry}).model_dump())
    return activity_id


def list_activity(
    db: Database,
    *,
    limit: int = 200,
    level: str | None = None,
    category: str | None = None,
    before_id: int | None = None,
) -> list[ActivityOut]:
    """Activity entries, newest first (id descending); page with ``before_id``."""
    conditions: list[str] = []
    params: list[Any] = []
    if level is not None:
        conditions.append("level = ?")
        params.append(level)
    if category is not None:
        conditions.append("category = ?")
        params.append(category)
    if before_id is not None:
        conditions.append("id < ?")
        params.append(before_id)
    rows = db.query(
        "SELECT id, ts, level, category, message, ref_type, ref_id FROM activity_log "
        f"WHERE {' AND '.join(conditions) or 'TRUE'} ORDER BY id DESC LIMIT ?",
        (*params, limit),
    )
    return [ActivityOut.model_validate(dict(row)) for row in rows]


# ------------------------------------------------------------------------------------------------
# Validation runs
# ------------------------------------------------------------------------------------------------
def create_validation_run(db: Database, *, ts: int, engine_version: str, params: dict[str, Any]) -> int:
    """Register a validation run that has just started (status 'running') and return its id."""
    return db.execute(
        "INSERT INTO validation_runs (ts, status, params_json, engine_version) VALUES (?, 'running', ?, ?)",
        (ts, dumps(params), engine_version),
    )


def finish_validation_run(
    db: Database, run_id: int, *, status: str, result: dict[str, Any] | None = None, message: str = ""
) -> None:
    """Close a validation run with status 'done' or 'failed', its result (if any) and a message.

    ``result`` must be JSON data with finite numbers: NaN or infinity (a profit factor without losses, for
    example) raise ``ValueError`` and leave the run untouched, i.e. still 'running'.
    """
    db.execute(
        "UPDATE validation_runs SET status = ?, result_json = ?, message = ? WHERE id = ?",
        (status, dumps(result or {}), message, run_id),
    )


def get_validation_run(db: Database, run_id: int) -> ValidationRunOut | None:
    row = db.query_one(f"{_SELECT_VALIDATION_RUN} WHERE id = ?", (run_id,))
    return _validation_run_from_row(row) if row is not None else None


def list_validation_runs(db: Database, *, limit: int = 5) -> list[ValidationRunOut]:
    """Validation runs, newest first. ``result`` is None while a run has no result or if the stored
    result does not fit ``ValidationResultOut`` (for example after the report format changed)."""
    rows = db.query(f"{_SELECT_VALIDATION_RUN} ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
    return [_validation_run_from_row(row) for row in rows]


def _validation_run_from_row(row: sqlite3.Row) -> ValidationRunOut:
    params = loads(row["params_json"], {})
    return ValidationRunOut(
        id=row["id"],
        ts=row["ts"],
        status=row["status"],
        engine_version=row["engine_version"],
        message=row["message"],
        params=params if isinstance(params, dict) else {},
        result=_validation_result(row["result_json"]),
    )


def _validation_result(text: str) -> ValidationResultOut | None:
    data = loads(text, {})
    if not data:
        return None
    try:
        return ValidationResultOut.model_validate(data)
    except ValidationError:
        return None


# ------------------------------------------------------------------------------------------------
# Database statistics and retention
# ------------------------------------------------------------------------------------------------
def db_stats(db: Database) -> DbStatusOut:
    """Size of the database (file plus write-ahead log) and the row counts shown on the status page.

    All counts come from one statement, i.e. one consistent snapshot. Cost:

    * first / last bar: a few primary-key seeks per symbol - milliseconds, independent of the number of
      bars (a plain ``MIN(ts)`` over the table would scan it completely).
    * news, signals: ``COUNT(*)`` over the smallest covering index - negligible.
    * bars: SQLite keeps no row counter, so ``COUNT(*)`` walks every leaf page of the primary-key b-tree,
      which for this WITHOUT ROWID table is the table itself. Linear in the table size: about 60 ms per
      million bars with the pages in the OS cache (measured: 3 million bars / 216 MB in 0.17 s), far more
      on a cold cache. Callers that poll should reuse a result for a minute or so instead of calling this
      every few seconds.
    """
    row = db.query(_DB_STATS)[0]  # scalar subqueries only: always exactly one row
    wal = db.path.with_name(db.path.name + "-wal")
    size_mb = round((_file_size(db.path) + _file_size(wal)) / _BYTES_PER_MB, 2)
    return DbStatusOut(size_mb=size_mb, **dict(row))


def prune_ops(db: Database, *, now_ts: int, keep_days: int = 30) -> None:
    """Retention for the operational logs: delete the LLM calls, job runs, activity entries and mail-log
    rows that are older than ``keep_days`` (one transaction)."""
    cutoff = now_ts - keep_days * _SECONDS_PER_DAY
    with db.write() as conn:
        conn.execute("DELETE FROM llm_calls WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM job_runs WHERE started_ts < ?", (cutoff,))
        conn.execute("DELETE FROM activity_log WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM mail_log WHERE ts < ?", (cutoff,))


def _file_size(path: Path) -> int:
    """Size of a file in bytes; 0 if it does not exist (the write-ahead log comes and goes)."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
