"""Paper-trading repository: the two portfolios ('ai' and 'me'), their orders, positions and equity snapshots.

Rows are mapped to the dataclasses of ``trading_local.models``, whose fields are named like the columns.
All functions are safe to call from several threads. Every function that writes may be called inside a
caller's ``db.write()`` block and then becomes part of that transaction - this is how the broker keeps a
fill (order, position and cash) atomic.
"""

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from trading_local.db.database import Database
from trading_local.models import EquityPoint, Order, Portfolio, Position

log = logging.getLogger(__name__)

# Portfolio key -> display name, in listing order.
_PORTFOLIO_NAMES = {"ai": "KI", "me": "Ich"}

# All columns except the generated id, in schema order.
_ORDER_COLUMNS = (
    "portfolio_id",
    "created_ts",
    "symbol",
    "intent",
    "side",
    "qty",
    "notional_base",
    "ref_price",
    "stop_price",
    "target_price",
    "stop_dist",
    "target_dist",
    "max_drift",
    "time_stop_ts",
    "flatten_at_close",
    "position_id",
    "signal_id",
    "origin",
    "status",
    "status_reason",
    "expire_ts",
    "filled_ts",
    "filled_qty",
    "fill_price",
    "fill_fx",
    "fee_base",
)
_POSITION_COLUMNS = (
    "portfolio_id",
    "symbol",
    "side",
    "qty",
    "entry_price",
    "entry_fx",
    "opened_ts",
    "stop_price",
    "target_price",
    "time_stop_ts",
    "flatten_at_close",
    "signal_id",
    "checked_ts",
    "status",
    "closed_ts",
    "exit_price",
    "exit_fx",
    "exit_reason",
    "fees_base",
    "realized_pnl",
)


def _insert_sql(table: str, columns: Sequence[str]) -> str:
    """INSERT statement with one named parameter per column."""
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(f':{c}' for c in columns)})"


def _update_sql(table: str, columns: Sequence[str]) -> str:
    """UPDATE statement that writes all columns of the row with ``id = :id``."""
    return f"UPDATE {table} SET {', '.join(f'{c} = :{c}' for c in columns)} WHERE id = :id"


_SELECT_PORTFOLIO = (
    "SELECT id, key, name, base_currency, start_cash, cash, fees_paid, created_ts, reset_ts FROM portfolios"
)
_SELECT_ORDER = f"SELECT id, {', '.join(_ORDER_COLUMNS)} FROM orders"
_INSERT_ORDER = _insert_sql("orders", _ORDER_COLUMNS)
_UPDATE_ORDER = _update_sql("orders", _ORDER_COLUMNS)
_SELECT_POSITION = f"SELECT id, {', '.join(_POSITION_COLUMNS)} FROM positions"
_INSERT_POSITION = _insert_sql("positions", _POSITION_COLUMNS)
_UPDATE_POSITION = _update_sql("positions", _POSITION_COLUMNS)
_SELECT_EQUITY = (
    "SELECT portfolio_id, ts, equity, cash, long_exposure, short_exposure, open_positions FROM equity_snapshots"
)

_UPSERT_EQUITY = """
INSERT INTO equity_snapshots (portfolio_id, ts, equity, cash, long_exposure, short_exposure, open_positions)
VALUES (:portfolio_id, :ts, :equity, :cash, :long_exposure, :short_exposure, :open_positions)
ON CONFLICT (portfolio_id, ts) DO UPDATE SET
    equity = excluded.equity,
    cash = excluded.cash,
    long_exposure = excluded.long_exposure,
    short_exposure = excluded.short_exposure,
    open_positions = excluded.open_positions
"""

# The first :max_points + 1 snapshots of the range: enough to tell whether the series must be thinned.
_EQUITY_HEAD = f"""
{_SELECT_EQUITY}
WHERE portfolio_id = :portfolio_id AND ts >= :since_ts
ORDER BY ts
LIMIT :max_points + 1
"""

# Thinning without reading the whole series: lay an even grid of :max_points instants over the time range
# and take, for each instant, the last snapshot at or before it. Grid step 0 is the first snapshot and the
# last step is the last snapshot, so both ends are always kept. Every lookup is a primary-key seek, which
# makes the cost independent of the number of stored snapshots.
_EQUITY_THINNED = f"""
WITH RECURSIVE
bounds AS (
    SELECT (SELECT MIN(ts) FROM equity_snapshots WHERE portfolio_id = :portfolio_id AND ts >= :since_ts) AS first_ts,
           (SELECT MAX(ts) FROM equity_snapshots WHERE portfolio_id = :portfolio_id AND ts >= :since_ts) AS last_ts
),
grid (step) AS (
    SELECT 0
    UNION ALL
    SELECT step + 1 FROM grid WHERE step + 1 < :max_points
),
picked AS (
    SELECT DISTINCT (
        SELECT MAX(ts) FROM equity_snapshots
        WHERE portfolio_id = :portfolio_id
          AND ts BETWEEN first_ts AND first_ts + (last_ts - first_ts) * step / (:max_points - 1)
    ) AS ts
    FROM bounds, grid
    WHERE first_ts IS NOT NULL
)
{_SELECT_EQUITY}
WHERE portfolio_id = :portfolio_id AND ts IN (SELECT ts FROM picked)
ORDER BY ts
"""

_NO_LOWER_BOUND = -(2**63)  # smallest SQLite integer: "since the beginning"


# ------------------------------------------------------------------------------------------------
# Portfolios
# ------------------------------------------------------------------------------------------------
def ensure_portfolios(db: Database, *, base_currency: str, start_cash: float, now_ts: int) -> dict[str, Portfolio]:
    """Create the portfolios 'ai' (named 'KI') and 'me' (named 'Ich') if they are missing.

    Idempotent: existing portfolios are returned unchanged, whatever arguments are passed.
    """
    rows = [
        (key, name, base_currency, start_cash, start_cash, now_ts, now_ts) for key, name in _PORTFOLIO_NAMES.items()
    ]
    with db.write() as conn:
        conn.executemany(
            "INSERT INTO portfolios (key, name, base_currency, start_cash, cash, created_ts, reset_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (key) DO NOTHING",
            rows,
        )
        return {portfolio.key: portfolio for portfolio in list_portfolios(db)}


def get_portfolio(db: Database, key: str) -> Portfolio | None:
    row = db.query_one(f"{_SELECT_PORTFOLIO} WHERE key = ?", (key,))
    return Portfolio(**dict(row)) if row is not None else None


def get_portfolio_by_id(db: Database, portfolio_id: int) -> Portfolio | None:
    row = db.query_one(f"{_SELECT_PORTFOLIO} WHERE id = ?", (portfolio_id,))
    return Portfolio(**dict(row)) if row is not None else None


def list_portfolios(db: Database) -> list[Portfolio]:
    """All portfolios, 'ai' first."""
    rows = db.query(f"{_SELECT_PORTFOLIO} ORDER BY key = 'ai' DESC, id")
    return [Portfolio(**dict(row)) for row in rows]


def adjust_cash(db: Database, portfolio_id: int, *, cash_delta: float, fee: float = 0.0) -> None:
    """Book a cash movement: ``cash += cash_delta - fee`` and ``fees_paid += fee``.

    The update is relative and atomic, so concurrent bookings never lose each other. Raises
    ``LookupError`` if the portfolio does not exist - a booking must not vanish silently; inside a
    caller's ``db.write()`` block this rolls the whole transaction back.
    """
    with db.write() as conn:
        cursor = conn.execute(
            "UPDATE portfolios SET cash = cash + ?, fees_paid = fees_paid + ? WHERE id = ?",
            (cash_delta - fee, fee, portfolio_id),
        )
        if cursor.rowcount == 0:
            raise LookupError(f"portfolio {portfolio_id} does not exist")


def reset_portfolios(db: Database, *, start_cash: float, now_ts: int) -> None:
    """Start a new comparison period: delete all orders, positions and equity snapshots and give every
    portfolio the same fresh capital (``cash = start_cash``, no fees, ``reset_ts = now_ts``). One transaction."""
    with db.write() as conn:
        conn.execute("DELETE FROM orders")
        conn.execute("DELETE FROM positions")
        conn.execute("DELETE FROM equity_snapshots")
        conn.execute(
            "UPDATE portfolios SET start_cash = ?, cash = ?, fees_paid = 0, reset_ts = ?",
            (start_cash, start_cash, now_ts),
        )


# ------------------------------------------------------------------------------------------------
# Orders
# ------------------------------------------------------------------------------------------------
def insert_order(db: Database, order: Order) -> Order:
    """Store a new order and return the same object with its ``id`` set. Raises ``ValueError`` if it has one already."""
    if order.id is not None:
        raise ValueError(f"order {order.id} is already stored; use update_order")
    order.id = db.execute(_INSERT_ORDER, asdict(order))
    return order


def update_order(db: Database, order: Order) -> None:
    """Write all columns of a stored order. Raises ``ValueError`` if the order has no id."""
    if order.id is None:
        raise ValueError("cannot update an order without an id; use insert_order")
    _update_row(db, _UPDATE_ORDER, asdict(order), "order")


def get_order(db: Database, order_id: int) -> Order | None:
    row = db.query_one(f"{_SELECT_ORDER} WHERE id = ?", (order_id,))
    return _order_from_row(row) if row is not None else None


def list_orders(db: Database, portfolio_id: int, *, status: str | None = None, limit: int = 200) -> list[Order]:
    """Orders of a portfolio, newest first, optionally only those with the given status."""
    conditions = ["portfolio_id = ?"]
    params: list[Any] = [portfolio_id]
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    rows = db.query(
        f"{_SELECT_ORDER} WHERE {' AND '.join(conditions)} ORDER BY created_ts DESC, id DESC LIMIT ?", (*params, limit)
    )
    return [_order_from_row(row) for row in rows]


def pending_orders(db: Database, portfolio_id: int | None = None) -> list[Order]:
    """Pending orders of one portfolio (or of all), oldest first - the order in which they are filled."""
    conditions = ["status = 'pending'"]
    params: list[Any] = []
    if portfolio_id is not None:
        conditions.append("portfolio_id = ?")
        params.append(portfolio_id)
    rows = db.query(f"{_SELECT_ORDER} WHERE {' AND '.join(conditions)} ORDER BY created_ts, id", params)
    return [_order_from_row(row) for row in rows]


# ------------------------------------------------------------------------------------------------
# Positions
# ------------------------------------------------------------------------------------------------
def insert_position(db: Database, position: Position) -> Position:
    """Store a new position and return the same object with its ``id`` set.

    Raises ``sqlite3.IntegrityError`` if the portfolio already holds an open position in the symbol and
    ``ValueError`` if the position has an id already.
    """
    if position.id is not None:
        raise ValueError(f"position {position.id} is already stored; use update_position")
    position.id = db.execute(_INSERT_POSITION, asdict(position))
    return position


def update_position(db: Database, position: Position) -> None:
    """Write all columns of a stored position. Raises ``ValueError`` if the position has no id."""
    if position.id is None:
        raise ValueError("cannot update a position without an id; use insert_position")
    _update_row(db, _UPDATE_POSITION, asdict(position), "position")


def get_position(db: Database, position_id: int) -> Position | None:
    row = db.query_one(f"{_SELECT_POSITION} WHERE id = ?", (position_id,))
    return _position_from_row(row) if row is not None else None


def open_positions(db: Database, portfolio_id: int | None = None) -> list[Position]:
    """Open positions of one portfolio (or of all), oldest first."""
    conditions = ["status = 'open'"]
    params: list[Any] = []
    if portfolio_id is not None:
        conditions.append("portfolio_id = ?")
        params.append(portfolio_id)
    rows = db.query(f"{_SELECT_POSITION} WHERE {' AND '.join(conditions)} ORDER BY opened_ts, id", params)
    return [_position_from_row(row) for row in rows]


def open_position_for_symbol(db: Database, portfolio_id: int, symbol: str) -> Position | None:
    """The portfolio's open position in ``symbol`` (the schema allows at most one)."""
    row = db.query_one(
        f"{_SELECT_POSITION} WHERE portfolio_id = ? AND symbol = ? AND status = 'open'", (portfolio_id, symbol)
    )
    return _position_from_row(row) if row is not None else None


def closed_positions(
    db: Database, portfolio_id: int, *, limit: int = 200, since_ts: int | None = None
) -> list[Position]:
    """Closed positions of a portfolio, most recently closed first; ``since_ts`` refers to ``closed_ts``."""
    conditions = ["portfolio_id = ?", "status = 'closed'"]
    params: list[Any] = [portfolio_id]
    if since_ts is not None:
        conditions.append("closed_ts >= ?")
        params.append(since_ts)
    rows = db.query(
        f"{_SELECT_POSITION} WHERE {' AND '.join(conditions)} ORDER BY closed_ts DESC, id DESC LIMIT ?",
        (*params, limit),
    )
    return [_position_from_row(row) for row in rows]


# ------------------------------------------------------------------------------------------------
# Equity
# ------------------------------------------------------------------------------------------------
def save_equity_point(db: Database, point: EquityPoint) -> None:
    """Insert the snapshot or replace the one stored for the same portfolio and time."""
    db.execute(_UPSERT_EQUITY, asdict(point))


def equity_series(
    db: Database, portfolio_id: int, *, since_ts: int | None = None, max_points: int = 600
) -> list[EquityPoint]:
    """Equity snapshots of a portfolio from ``since_ts`` on, ascending, at most ``max_points`` of them.

    A series that fits is returned completely. A longer one is thinned evenly over time (see
    ``_EQUITY_THINNED``): the result always contains the first and the last snapshot and consists of
    stored snapshots only - nothing is interpolated. For snapshots at regular intervals this yields
    exactly ``max_points`` evenly spaced points; gaps in the data yield fewer. Raises ``ValueError``
    for ``max_points < 2`` because the two ends could not both be kept.
    """
    if max_points < 2:
        raise ValueError("max_points must be at least 2 (the first and the last point are always kept)")
    params = {
        "portfolio_id": portfolio_id,
        "since_ts": _NO_LOWER_BOUND if since_ts is None else since_ts,
        "max_points": max_points,
    }
    rows = db.query(_EQUITY_HEAD, params)
    if len(rows) > max_points:
        rows = db.query(_EQUITY_THINNED, params)
    return [EquityPoint(**dict(row)) for row in rows]


def last_equity_point(db: Database, portfolio_id: int, *, before_ts: int | None = None) -> EquityPoint | None:
    """The portfolio's newest equity snapshot, optionally the newest one strictly before ``before_ts``."""
    conditions = ["portfolio_id = ?"]
    params: list[Any] = [portfolio_id]
    if before_ts is not None:
        conditions.append("ts < ?")
        params.append(before_ts)
    row = db.query_one(f"{_SELECT_EQUITY} WHERE {' AND '.join(conditions)} ORDER BY ts DESC LIMIT 1", params)
    return EquityPoint(**dict(row)) if row is not None else None


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
def _update_row(db: Database, sql: str, params: dict[str, Any], what: str) -> None:
    """Run a full-row UPDATE by id; a missing row is logged instead of being ignored silently."""
    with db.write() as conn:
        if conn.execute(sql, params).rowcount == 0:
            log.warning("update of %s %s: no such row, nothing was written", what, params["id"])


def _order_from_row(row: sqlite3.Row) -> Order:
    values = dict(row)
    values["flatten_at_close"] = bool(values["flatten_at_close"])
    return Order(**values)


def _position_from_row(row: sqlite3.Row) -> Position:
    values = dict(row)
    values["flatten_at_close"] = bool(values["flatten_at_close"])
    return Position(**values)
