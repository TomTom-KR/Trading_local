"""Agenda repository: macro data, central-bank decisions and earnings releases with expectation and actual.

Time semantics (no look-ahead)
* ``first_seen_ts``  when this platform first learned of an event (its schedule). Schedules of macro data,
                     central-bank meetings and earnings are published weeks ahead.
* ``actual_ts``      when the actual value became known here. Decision code that reads actual values or
                     surprises passes ``asof_ts``; values learned later are invisible to it.
"""

import sqlite3
from collections.abc import Collection, Iterable
from dataclasses import asdict, dataclass, fields

from trading_local.db.database import Database


@dataclass(slots=True)
class AgendaEvent:
    uid: str
    kind: str  # 'macro' | 'central_bank' | 'earnings'
    source: str  # 'forexfactory' | 'yahoo' | 'financialjuice'
    title: str
    event_ts: int
    first_seen_ts: int
    updated_ts: int
    currency: str = ""
    symbol: str | None = None
    time_known: bool = True
    importance: int = 1  # 1 low, 2 medium, 3 high
    unit: str = ""
    forecast: float | None = None
    previous: float | None = None
    actual: float | None = None
    forecast_text: str = ""
    previous_text: str = ""
    actual_text: str = ""
    surprise: float | None = None
    surprise_z: float | None = None
    actual_ts: int | None = None
    id: int | None = None


_COLUMNS = tuple(spec.name for spec in fields(AgendaEvent) if spec.name != "id")
_SELECT = f"SELECT id, {', '.join(_COLUMNS)} FROM agenda_events"
# A newer schedule replaces the schedule fields; an actual value, once known, is kept (first known wins).
_UPSERT = (
    f"INSERT INTO agenda_events ({', '.join(_COLUMNS)}) VALUES ({', '.join(':' + c for c in _COLUMNS)}) "
    "ON CONFLICT (uid) DO UPDATE SET kind = excluded.kind, source = excluded.source, title = excluded.title, "
    "event_ts = excluded.event_ts, time_known = excluded.time_known, importance = excluded.importance, "
    "currency = excluded.currency, symbol = excluded.symbol, "
    "unit = CASE WHEN excluded.unit != '' THEN excluded.unit ELSE unit END, "
    "forecast = COALESCE(excluded.forecast, forecast), "
    "forecast_text = CASE WHEN excluded.forecast_text != '' THEN excluded.forecast_text ELSE forecast_text END, "
    "previous = COALESCE(excluded.previous, previous), "
    "previous_text = CASE WHEN excluded.previous_text != '' THEN excluded.previous_text ELSE previous_text END, "
    "actual_ts = CASE WHEN actual IS NULL THEN excluded.actual_ts ELSE actual_ts END, "
    "actual_text = CASE WHEN actual IS NULL THEN excluded.actual_text ELSE actual_text END, "
    "surprise = CASE WHEN actual IS NULL THEN excluded.surprise ELSE surprise END, "
    "surprise_z = CASE WHEN actual IS NULL THEN excluded.surprise_z ELSE surprise_z END, "
    "actual = COALESCE(actual, excluded.actual), updated_ts = excluded.updated_ts"
)


def _params(event: AgendaEvent) -> dict[str, object]:
    values = asdict(event)
    values.pop("id")
    values["time_known"] = int(event.time_known)
    return values


def _from_row(row: sqlite3.Row) -> AgendaEvent:
    values = dict(row)
    values["time_known"] = bool(values["time_known"])
    return AgendaEvent(**values)


def upsert_events(db: Database, events: Iterable[AgendaEvent]) -> int:
    """Insert new events or update the schedule of known ones (by ``uid``; ``first_seen_ts`` and a known actual
    value are kept). Returns the number of rows written."""
    rows = [_params(event) for event in events]
    if not rows:
        return 0
    with db.write() as conn:
        before = conn.total_changes
        conn.executemany(_UPSERT, rows)
        return conn.total_changes - before


def set_actual(
    db: Database,
    event_id: int,
    *,
    actual: float,
    actual_text: str,
    actual_ts: int,
    surprise: float | None,
    surprise_z: float | None,
    unit: str = "",
) -> bool:
    """Record the actual value of an event that has none yet. Returns False when it already had one."""
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE agenda_events SET actual = ?, actual_text = ?, actual_ts = ?, surprise = ?, surprise_z = ?, "
            "unit = CASE WHEN unit = '' THEN ? ELSE unit END, updated_ts = ? WHERE id = ? AND actual IS NULL",
            (actual, actual_text, actual_ts, surprise, surprise_z, unit, actual_ts, event_id),
        )
        return cursor.rowcount == 1


def get_event(db: Database, event_id: int) -> AgendaEvent | None:
    row = db.query_one(f"{_SELECT} WHERE id = ?", (event_id,))
    return None if row is None else _from_row(row)


def events_between(
    db: Database,
    *,
    start_ts: int,
    end_ts: int,
    asof_ts: int | None = None,
    min_importance: int = 1,
    kinds: Collection[str] | None = None,
    symbol: str | None = None,
    currencies: Collection[str] | None = None,
    limit: int = 2000,
) -> list[AgendaEvent]:
    """Events with ``start_ts <= event_ts < end_ts``, oldest first.

    With ``asof_ts`` only events known at that time are returned, and actual values (with their surprises)
    learned later are blanked out. ``symbol`` keeps the earnings of that instrument and every non-earnings
    event (the caller decides which of those concern it).
    """
    conditions = ["event_ts >= ?", "event_ts < ?", "importance >= ?"]
    params: list[object] = [start_ts, end_ts, min_importance]
    if asof_ts is not None:
        conditions.append("first_seen_ts <= ?")
        params.append(asof_ts)
    if kinds:
        conditions.append(f"kind IN ({', '.join('?' * len(kinds))})")
        params.extend(kinds)
    if symbol is not None:
        conditions.append("(kind != 'earnings' OR symbol = ?)")
        params.append(symbol)
    if currencies:
        conditions.append(f"(kind = 'earnings' OR currency IN ({', '.join('?' * len(currencies))}))")
        params.extend(currencies)
    rows = db.query(f"{_SELECT} WHERE {' AND '.join(conditions)} ORDER BY event_ts, id LIMIT ?", (*params, limit))
    events = [_from_row(row) for row in rows]
    if asof_ts is not None:
        for event in events:
            if event.actual_ts is None or event.actual_ts > asof_ts:
                event.actual = event.surprise = event.surprise_z = event.actual_ts = None
                event.actual_text = ""
    return events


def earlier_surprises(db: Database, *, currency: str, title: str, before_ts: int, limit: int = 36) -> list[float]:
    """Surprises of the same event (currency and title) released before ``before_ts``, newest first."""
    rows = db.query(
        "SELECT surprise FROM agenda_events WHERE currency = ? AND title = ? AND event_ts < ? "
        "AND surprise IS NOT NULL AND kind != 'earnings' ORDER BY event_ts DESC LIMIT ?",
        (currency, title, before_ts, limit),
    )
    return [float(row[0]) for row in rows]


def delete_unlisted(
    db: Database, *, source: str, start_ts: int, end_ts: int, keep_uids: Collection[str], symbol: str | None = None
) -> int:
    """Delete events of ``source`` in ``[start_ts, end_ts)`` without an actual value whose uid is not in
    ``keep_uids`` (moved or cancelled in the source). Returns the number deleted."""
    conditions = ["source = ?", "event_ts >= ?", "event_ts < ?", "actual IS NULL"]
    params: list[object] = [source, start_ts, end_ts]
    if symbol is not None:
        conditions.append("symbol = ?")
        params.append(symbol)
    rows = db.query(f"SELECT id, uid FROM agenda_events WHERE {' AND '.join(conditions)}", params)
    doomed = [(row["id"],) for row in rows if row["uid"] not in keep_uids]
    if doomed:
        db.executemany("DELETE FROM agenda_events WHERE id = ?", doomed)
    return len(doomed)
