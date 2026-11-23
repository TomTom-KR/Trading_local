"""Signal repository: the journal of all trade candidates (actionable and shadow) and the LLM chart analyses.

Rows are mapped to ``models.Signal`` and ``models.ChartAnalysis``. The dataclass fields are named like the
columns, except that ``Signal.components`` / ``Signal.features`` live in ``components_json`` /
``features_json``.

All functions are safe to call from several threads.
"""

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from trading_local.db.database import Database, dumps, loads
from trading_local.models import SIGNAL_OPEN_STATUSES, SIGNAL_OUTCOME_STATUSES, ChartAnalysis, Signal

log = logging.getLogger(__name__)

# All columns except the generated id, in schema order.
_SIGNAL_COLUMNS = (
    "created_ts",
    "issued_ts",
    "data_ts",
    "symbol",
    "direction",
    "setup",
    "timeframe",
    "intraday",
    "horizon_min",
    "ref_price",
    "atr",
    "stop_dist",
    "target_dist",
    "score",
    "components_json",
    "features_json",
    "confidence",
    "expected_r",
    "llm_verdict",
    "llm_conviction",
    "llm_latency_ms",
    "rationale_de",
    "risks_de",
    "news_ids",
    "actionable",
    "shadow_reason",
    "status",
    "status_reason",
    "entry_deadline_ts",
    "checked_ts",
    "entry_ts",
    "entry_price",
    "stop_price",
    "target_price",
    "expires_ts",
    "exit_ts",
    "exit_price",
    "outcome_r",
    "mfe_r",
    "mae_r",
    "mailed_ts",
    "engine_version",
)
_CHART_COLUMNS = (
    "symbol",
    "ts",
    "timeframe",
    "model",
    "bias",
    "strength",
    "pattern",
    "support",
    "resistance",
    "comment_de",
    "latency_ms",
)


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    """INSERT statement with one named parameter per column."""
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})"


def _update_sql(table: str, columns: Sequence[str]) -> str:
    """UPDATE statement that writes all columns of the row with ``id = :id``."""
    return f"UPDATE {table} SET {', '.join(f'{c} = :{c}' for c in columns)} WHERE id = :id"


def _marks(values: Sequence[Any]) -> str:
    """Positional placeholders for an IN list with one entry per value."""
    return ", ".join("?" * len(values))


_SELECT_SIGNAL = f"SELECT id, {', '.join(_SIGNAL_COLUMNS)} FROM signals"
_INSERT_SIGNAL = _insert_sql("signals", _SIGNAL_COLUMNS)
_UPDATE_SIGNAL = _update_sql("signals", _SIGNAL_COLUMNS)
_SELECT_CHART = f"SELECT id, {', '.join(_CHART_COLUMNS)} FROM chart_analyses"
_INSERT_CHART = _insert_sql("chart_analyses", _CHART_COLUMNS)

# Lower end of the id walk for the ``since_ts`` filter of list_signals (parameters: probe, since_ts, probe);
# it only repeats what ``created_ts >= ?`` says and changes no result. The inner query takes up to ``probe``
# matching entries straight from signals_created_idx. Fewer than ``probe`` means that these are all there
# are: their smallest id is the exact end of the walk, so a page that the matches cannot fill ("the signals
# of today") does not read the whole journal. Otherwise there is no bound (0 is below every id) and the walk
# is what it would be without this term.
_SINCE_FLOOR = """(
    SELECT CASE WHEN COUNT(*) < ? THEN MIN(id) ELSE 0 END
    FROM (SELECT id FROM signals WHERE created_ts >= ? LIMIT ?)
)"""
# Smallest probe. The probe is always one entry larger than the page, so a page that the matches cannot fill
# always gets its bound, and at least this large: that also bounds ranges of up to this many matches when a
# second, rarely matching filter is what keeps the page from filling.
_SINCE_PROBE = 10_000


# ------------------------------------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------------------------------------
def insert_signal(db: Database, signal: Signal) -> Signal:
    """Store a new signal and return the same object with its ``id`` set.

    Raises ``ValueError`` if the signal already has an id: it is stored already and a second row would
    corrupt the journal (use ``update_signal``). ``components`` and ``features`` must be JSON data with
    finite numbers: NaN or infinity raise ``ValueError`` as well and nothing is stored.
    """
    if signal.id is not None:
        raise ValueError(f"signal {signal.id} is already stored; use update_signal")
    signal.id = db.execute(_INSERT_SIGNAL, _signal_params(signal))
    return signal


def update_signal(db: Database, signal: Signal) -> None:
    """Write all columns of a stored signal. Raises ``ValueError`` if the signal has no id.

    Full-row means that the object wins over everything stored: a column that was changed in the database
    after the object was read - ``mailed_ts`` set by ``mark_mailed``, for example - is overwritten with
    the object's older value. Carry such changes over to the object (or read it again) before updating.
    """
    if signal.id is None:
        raise ValueError("cannot update a signal without an id; use insert_signal")
    with db.write() as conn:
        if conn.execute(_UPDATE_SIGNAL, _signal_params(signal)).rowcount == 0:
            log.warning("update of signal %s: no such row, nothing was written", signal.id)


def get_signal(db: Database, signal_id: int) -> Signal | None:
    row = db.query_one(f"{_SELECT_SIGNAL} WHERE id = ?", (signal_id,))
    return _signal_from_row(row) if row is not None else None


def list_signals(
    db: Database,
    *,
    statuses: Sequence[str] | None = None,
    actionable: bool | None = None,
    symbol: str | None = None,
    since_ts: int | None = None,
    before_id: int | None = None,
    limit: int = 100,
) -> list[Signal]:
    """Signals, newest first (id descending); page with ``before_id``.

    ``statuses`` restricts the result to these statuses (an empty sequence matches nothing),
    ``actionable`` selects issued (True) or shadow (False) signals and ``since_ts`` refers to ``created_ts``
    (inclusive).
    """
    conditions: list[str] = []
    params: list[Any] = []
    if statuses is not None:
        conditions.append(f"status IN ({_marks(statuses)})")
        params += statuses
    if actionable is not None:
        conditions.append("actionable = ?")
        params.append(actionable)
    if symbol is not None:
        conditions.append("symbol = ?")
        params.append(symbol)
    if since_ts is not None:
        probe = max(limit, _SINCE_PROBE) + 1
        conditions.append(f"created_ts >= ? AND id >= {_SINCE_FLOOR}")
        params += [since_ts, probe, since_ts, probe]
    if before_id is not None:
        conditions.append("id < ?")
        params.append(before_id)
    rows = db.query(
        f"{_SELECT_SIGNAL} WHERE {' AND '.join(conditions) or 'TRUE'} ORDER BY id DESC LIMIT ?", (*params, limit)
    )
    return [_signal_from_row(row) for row in rows]


def open_signals(db: Database) -> list[Signal]:
    """Signals that still need tracking (review, pending_entry, active), oldest first."""
    rows = db.query(
        f"{_SELECT_SIGNAL} WHERE status IN ({_marks(SIGNAL_OPEN_STATUSES)}) ORDER BY created_ts, id",
        SIGNAL_OPEN_STATUSES,
    )
    return [_signal_from_row(row) for row in rows]


def last_signal_for_symbol(db: Database, symbol: str, *, actionable_only: bool = False) -> Signal | None:
    """The symbol's most recently created signal (optionally ignoring shadow signals)."""
    condition = "symbol = ? AND actionable = 1" if actionable_only else "symbol = ?"
    row = db.query_one(f"{_SELECT_SIGNAL} WHERE {condition} ORDER BY created_ts DESC, id DESC LIMIT 1", (symbol,))
    return _signal_from_row(row) if row is not None else None


def closed_outcomes(
    db: Database,
    *,
    engine_version: str | None = None,
    since_ts: int | None = None,
    before_ts: int | None = None,
) -> list[Signal]:
    """Signals with a realised result (target, stopped or expired with ``outcome_r`` set), by exit time ascending.

    ``since_ts`` and ``before_ts`` bound the exit time: ``since_ts <= exit_ts < before_ts``. Calibration
    passes its decision time as ``before_ts`` and therefore never sees an outcome that was not known yet.
    Actionable and shadow signals are returned alike.
    """
    conditions = [
        f"status IN ({_marks(SIGNAL_OUTCOME_STATUSES)})",
        "outcome_r IS NOT NULL",
        "exit_ts IS NOT NULL",
    ]
    params: list[Any] = [*SIGNAL_OUTCOME_STATUSES]
    if engine_version is not None:
        conditions.append("engine_version = ?")
        params.append(engine_version)
    if since_ts is not None:
        conditions.append("exit_ts >= ?")
        params.append(since_ts)
    if before_ts is not None:
        conditions.append("exit_ts < ?")
        params.append(before_ts)
    rows = db.query(f"{_SELECT_SIGNAL} WHERE {' AND '.join(conditions)} ORDER BY exit_ts, id", params)
    return [_signal_from_row(row) for row in rows]


def mark_mailed(db: Database, signal_id: int, ts: int) -> None:
    """Record that the entry recommendation of a signal was mailed at ``ts``."""
    db.execute("UPDATE signals SET mailed_ts = ? WHERE id = ?", (ts, signal_id))


# ------------------------------------------------------------------------------------------------
# Chart analyses
# ------------------------------------------------------------------------------------------------
def insert_chart_analysis(db: Database, analysis: ChartAnalysis) -> ChartAnalysis:
    """Store a new chart analysis and return the same object with its ``id`` set.

    Raises ``ValueError`` if the analysis already has an id.
    """
    if analysis.id is not None:
        raise ValueError(f"chart analysis {analysis.id} is already stored")
    analysis.id = db.execute(_INSERT_CHART, asdict(analysis))
    return analysis


def latest_chart_analysis(
    db: Database, symbol: str, *, timeframe: str | None = None, before_ts: int | None = None
) -> ChartAnalysis | None:
    """The symbol's newest chart analysis, optionally of one timeframe and made strictly before ``before_ts``."""
    conditions = ["symbol = ?"]
    params: list[Any] = [symbol]
    if timeframe is not None:
        conditions.append("timeframe = ?")
        params.append(timeframe)
    if before_ts is not None:
        conditions.append("ts < ?")
        params.append(before_ts)
    row = db.query_one(f"{_SELECT_CHART} WHERE {' AND '.join(conditions)} ORDER BY ts DESC, id DESC LIMIT 1", params)
    return ChartAnalysis(**dict(row)) if row is not None else None


def list_chart_analyses(db: Database, symbol: str, *, limit: int = 20) -> list[ChartAnalysis]:
    """The symbol's chart analyses, newest first."""
    rows = db.query(f"{_SELECT_CHART} WHERE symbol = ? ORDER BY ts DESC, id DESC LIMIT ?", (symbol, limit))
    return [ChartAnalysis(**dict(row)) for row in rows]


def last_chart_analysis_ts(db: Database) -> dict[str, int]:
    """Time of the newest chart analysis per symbol (symbols without an analysis are absent)."""
    rows = db.query("SELECT symbol, MAX(ts) AS ts FROM chart_analyses GROUP BY symbol")
    return {row["symbol"]: row["ts"] for row in rows}


# ------------------------------------------------------------------------------------------------
# Row mapping
# ------------------------------------------------------------------------------------------------
def _signal_from_row(row: sqlite3.Row) -> Signal:
    values = dict(row)
    values["components"] = loads(values.pop("components_json"), {})
    values["features"] = loads(values.pop("features_json"), {})
    values["news_ids"] = loads(values["news_ids"], [])
    values["intraday"] = bool(values["intraday"])
    values["actionable"] = bool(values["actionable"])
    return Signal(**values)


def _signal_params(signal: Signal) -> dict[str, Any]:
    """Statement parameters named like the columns (plus ``id``)."""
    params = asdict(signal)
    params["components_json"] = dumps(params.pop("components"))
    params["features_json"] = dumps(params.pop("features"))
    params["news_ids"] = dumps(params["news_ids"])
    return params
