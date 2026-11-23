"""Repository for market data: instruments, 1-minute bars, daily bars and the persisted quote book.

Canonical bar frames (used across the whole code base)
* 1-minute bars: index ``ts`` = int64 epoch seconds of the bar OPEN time (UTC), ascending and unique;
  float64 columns ``open, high, low, close, volume``.
* daily bars: index ``day`` = 'YYYY-MM-DD' strings (session date of the venue), same columns.
* Empty results are empty frames with exactly these columns and this index name.

Every function takes the ``Database`` as first argument and is safe to call from several threads.
Timestamp arguments are converted with ``int()``, so numpy integers taken from a frame index are fine.

Decision code (signal engine, trackers, paper broker, validation replay) must NOT read bars through
this module: ``trading_local.barstore.BarReader`` is the as-of bounded door that prevents look-ahead.
"""

import logging
import math
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import astuple, fields
from datetime import date
from itertools import compress, repeat
from typing import Any, TypedDict

import numpy as np
import pandas as pd

from trading_local.db.database import Database, dumps, loads
from trading_local.models import Bar, Instrument, Quote

log = logging.getLogger(__name__)

NON_TRADABLE_CLASSES: frozenset[str] = frozenset({"fx", "future"})  # see InstrumentBook.tradable

BAR_SECONDS = 60  # length of a stored intraday bar
BAR_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Precedence of the bar sources: REST history is authoritative, bars built from the trade stream beat
# bars sampled from quotes. A stored bar is only replaced by a source of the same or a higher rank.
SOURCE_RANK: dict[str, int] = {"quote": 0, "stream": 1, "rest": 2}


class BarStats(TypedDict):
    """Size and time span of the bars_1m table."""

    rows: int
    first_ts: int | None
    last_ts: int | None


# ------------------------------------------------------------------------------------------------
# Instruments
# ------------------------------------------------------------------------------------------------
_INSTRUMENT_COLUMNS = tuple(field.name for field in fields(Instrument))  # identical to the instruments columns
_USER_FLAGS: tuple[str, ...] = ("tradable", "active")  # set by the user in the GUI
_SELECT_INSTRUMENTS = f"SELECT {', '.join(_INSTRUMENT_COLUMNS)} FROM instruments"


def _instrument_upsert_sql(keep_user_flags: bool) -> str:
    updated = [
        column
        for column in _INSTRUMENT_COLUMNS
        if column != "symbol" and not (keep_user_flags and column in _USER_FLAGS)
    ]
    return (
        f"INSERT INTO instruments ({', '.join(_INSTRUMENT_COLUMNS)}) "
        f"VALUES ({', '.join('?' * len(_INSTRUMENT_COLUMNS))}) "
        f"ON CONFLICT (symbol) DO UPDATE SET {', '.join(f'{column} = excluded.{column}' for column in updated)}"
    )


def _instrument_params(instrument: Instrument) -> tuple[Any, ...]:
    """Column values in the order of ``_INSTRUMENT_COLUMNS`` (flags as 0/1, aliases as JSON text)."""
    return (
        instrument.symbol,
        instrument.name,
        instrument.asset_class,
        instrument.region,
        instrument.exchange,
        instrument.currency,
        instrument.price_scale,
        instrument.timezone,
        instrument.delay_min,
        int(instrument.tradable),
        int(instrument.active),
        instrument.benchmark,
        dumps(list(instrument.aliases)),
        instrument.sort_order,
    )


def _instrument(row: sqlite3.Row) -> Instrument:
    return Instrument(
        symbol=row["symbol"],
        name=row["name"],
        asset_class=row["asset_class"],
        region=row["region"],
        exchange=row["exchange"],
        currency=row["currency"],
        price_scale=row["price_scale"],
        timezone=row["timezone"],
        delay_min=row["delay_min"],
        tradable=bool(row["tradable"]),
        active=bool(row["active"]),
        benchmark=row["benchmark"],
        aliases=tuple(loads(row["aliases"], [])),
        sort_order=row["sort_order"],
    )


def upsert_instruments(db: Database, instruments: Iterable[Instrument], *, keep_user_flags: bool = True) -> int:
    """Insert new instruments and update the descriptive columns of existing ones.

    With ``keep_user_flags`` the flags the user controls (``active``, ``tradable``) keep their stored
    values on existing rows; new rows always take them from the given instrument.
    Returns the number of rows inserted or updated.
    """
    rows = [_instrument_params(instrument) for instrument in instruments]
    if not rows:
        return 0
    return db.executemany(_instrument_upsert_sql(keep_user_flags), rows)


def get_instrument(db: Database, symbol: str) -> Instrument | None:
    row = db.query_one(f"{_SELECT_INSTRUMENTS} WHERE symbol = ?", (symbol,))
    return None if row is None else _instrument(row)


def list_instruments(
    db: Database,
    *,
    active_only: bool = True,
    tradable_only: bool = False,
    region: str | None = None,
    asset_class: str | None = None,
    exchange: str | None = None,
) -> list[Instrument]:
    """Instruments matching all given filters, ordered by ``sort_order, symbol``."""
    conditions: list[str] = []
    params: list[str] = []
    if active_only:
        conditions.append("active = 1")
    if tradable_only:
        conditions.append("tradable = 1")
    for column, value in (("region", region), ("asset_class", asset_class), ("exchange", exchange)):
        if value is not None:
            conditions.append(f"{column} = ?")
            params.append(value)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = db.query(f"{_SELECT_INSTRUMENTS}{where} ORDER BY sort_order, symbol", params)
    return [_instrument(row) for row in rows]


def set_instrument_flags(
    db: Database, symbol: str, *, active: bool | None = None, tradable: bool | None = None
) -> Instrument | None:
    """Change the user-controlled flags that are not ``None``.

    Returns the instrument as stored afterwards, or ``None`` if the symbol is unknown.
    """
    changes = {
        column: int(value) for column, value in (("active", active), ("tradable", tradable)) if value is not None
    }
    with db.write() as conn:
        if changes:
            assignments = ", ".join(f"{column} = ?" for column in changes)
            conn.execute(f"UPDATE instruments SET {assignments} WHERE symbol = ?", (*changes.values(), symbol))
        return get_instrument(db, symbol)


class InstrumentBook:
    """Thread-safe in-memory view of the instruments table, shared by the services.

    Instruments are needed on every cycle and API call, so the services share one book instead of
    querying the table. The book loads lazily on first use and does not notice database changes by
    itself: whoever adds or changes instruments calls ``reload()`` once the change is committed.
    Readers always work on one complete snapshot; the returned lists are copies the caller may modify.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._lock = threading.Lock()
        # symbol -> instrument in list order (sort_order, symbol); replaced as a whole, never mutated
        self._snapshot: dict[str, Instrument] | None = None

    def reload(self) -> None:
        """Replace the view with the current content of the table."""
        with self._lock:
            self._snapshot = self._load()

    def get(self, symbol: str) -> Instrument | None:
        return self._current().get(symbol)

    def require(self, symbol: str) -> Instrument:
        """Like ``get`` but raises ``KeyError`` for an unknown symbol."""
        return self._current()[symbol]

    def all(self, *, active_only: bool = True) -> list[Instrument]:
        """Instruments ordered by ``sort_order, symbol``."""
        return [instrument for instrument in self._current().values() if instrument.active or not active_only]

    def tradable(self) -> list[Instrument]:
        """Instruments that are active and tradable.

        FX pairs and futures never count, even if an old row carries tradable=1: they are conversion rates
        and comparison values that the paper broker cannot trade (the web API refuses to switch them on).
        """
        return [
            instrument
            for instrument in self._current().values()
            if instrument.active and instrument.tradable and instrument.asset_class not in NON_TRADABLE_CLASSES
        ]

    def by_exchange(self) -> dict[str, list[Instrument]]:
        """Active instruments grouped by their calendar key."""
        groups: dict[str, list[Instrument]] = {}
        for instrument in self.all():
            groups.setdefault(instrument.exchange, []).append(instrument)
        return groups

    def symbols(self, *, active_only: bool = True) -> list[str]:
        return [instrument.symbol for instrument in self.all(active_only=active_only)]

    def __contains__(self, symbol: object) -> bool:
        return symbol in self._current()

    def _current(self) -> dict[str, Instrument]:
        snapshot = self._snapshot
        if snapshot is None:
            with self._lock:
                if self._snapshot is None:
                    self._snapshot = self._load()
                snapshot = self._snapshot
        return snapshot

    def _load(self) -> dict[str, Instrument]:
        return {instrument.symbol: instrument for instrument in list_instruments(self._db, active_only=False)}


# ------------------------------------------------------------------------------------------------
# Canonical frames
# ------------------------------------------------------------------------------------------------
_MINUTE_FIELDS: list[tuple[str, Any]] = [("ts", np.int64), *((column, np.float64) for column in BAR_COLUMNS)]
_MINUTE_DTYPE = np.dtype(_MINUTE_FIELDS)
# The same rows plus the bar source (fixed-width text wide enough for every name in SOURCE_RANK).
_MINUTE_SOURCE_DTYPE = np.dtype([*_MINUTE_FIELDS, ("source", f"<U{max(map(len, SOURCE_RANK))}")])
_SELECT_MINUTE_FRAME = f"SELECT ts, {', '.join(BAR_COLUMNS)} FROM bars_1m"
_SELECT_MINUTE_SOURCES = f"SELECT ts, {', '.join(BAR_COLUMNS)}, source FROM bars_1m"
_SELECT_DAILY_FRAME = f"SELECT day, {', '.join(BAR_COLUMNS)} FROM bars_1d"


def _minute_frame(data: np.ndarray) -> pd.DataFrame:
    """Canonical 1-minute frame from a structured array of ``_MINUTE_DTYPE``."""
    return pd.DataFrame({column: data[column] for column in BAR_COLUMNS}, index=pd.Index(data["ts"], name="ts"))


def _daily_frame(days: Sequence[str], values: np.ndarray) -> pd.DataFrame:
    """Canonical daily frame from the day keys and an (n, 5) float array in ``BAR_COLUMNS`` order."""
    return pd.DataFrame(values, index=pd.Index(days, dtype="str", name="day"), columns=list(BAR_COLUMNS))


def empty_bars_1m() -> pd.DataFrame:
    """Empty frame in the canonical 1-minute format."""
    return _minute_frame(np.empty(0, dtype=_MINUTE_DTYPE))


def empty_bars_1d() -> pd.DataFrame:
    """Empty frame in the canonical daily format."""
    return _daily_frame([], np.empty((0, len(BAR_COLUMNS))))


def _ohlcv(frame: pd.DataFrame) -> np.ndarray:
    """The price/volume columns of a bar frame as (n, 5) float64 array in ``BAR_COLUMNS`` order.

    The array may share memory with the frame and must not be modified.
    """
    missing = [column for column in BAR_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"bar frame lacks the columns {missing}")
    return frame[list(BAR_COLUMNS)].to_numpy(dtype=np.float64)


def bar_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a canonical 1-minute frame into ``(ts, values)`` for bulk processing.

    ``ts`` is the int64 epoch-second index, ``values`` an (n, 5) float64 array in ``BAR_COLUMNS`` order;
    both may share memory with the frame and must not be modified. Raises ``ValueError`` for a frame
    that is not in the canonical format; in particular a datetime index is rejected instead of being
    reinterpreted as (nano)seconds.
    """
    if not pd.api.types.is_integer_dtype(frame.index.dtype):
        raise ValueError(f"bar frame needs an integer epoch-second index, got dtype {frame.index.dtype}")
    return frame.index.to_numpy(dtype=np.int64), _ohlcv(frame)


def _clean_ohlcv(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the storage rules to an (n, 5) open/high/low/close/volume array.

    Returns ``(keep, cleaned)``. ``keep`` is the boolean mask of the rows that may be stored: all four
    prices finite and positive. ``cleaned`` is a new array with only those rows, high/low widened so
    that ``low <= min(open, close)`` and ``high >= max(open, close)``, and a missing, non-finite or
    negative volume replaced by 0.
    """
    prices = values[:, :4]
    keep = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    cleaned = values[keep]  # boolean indexing copies: the input stays untouched
    open_, high, low, close, volume = cleaned.T
    cleaned[:, 1] = np.maximum(high, np.maximum(open_, close))
    cleaned[:, 2] = np.minimum(low, np.minimum(open_, close))
    cleaned[:, 4] = np.where(np.isfinite(volume) & (volume > 0), volume, 0.0)
    return keep, cleaned


def _fetch_tuples(db: Database, sql: str, params: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Run a SELECT and return plain tuples instead of ``sqlite3.Row`` objects (numpy converts them in bulk)."""
    cursor = db.conn().cursor()
    cursor.row_factory = None
    try:
        return cursor.execute(sql, params).fetchall()
    finally:
        cursor.close()


def _with_limit(sql: str, params: list[Any], limit: int | None) -> tuple[str, list[Any]]:
    """Append an optional ``LIMIT`` to a statement and its parameters."""
    if limit is None:
        return sql, params
    if limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")
    return f"{sql} LIMIT ?", [*params, int(limit)]


# ------------------------------------------------------------------------------------------------
# Bars (1 minute)
# ------------------------------------------------------------------------------------------------
_BAR_FIELDS = ", ".join(field.name for field in fields(Bar))  # identical to the bars_1m columns
_SELECT_BAR = f"SELECT {_BAR_FIELDS} FROM bars_1m"


def _rank_sql(column: str) -> str:
    """SQL expression for the precedence rank of a ``source`` value."""
    cases = " ".join(f"WHEN '{source}' THEN {rank}" for source, rank in SOURCE_RANK.items())
    return f"CASE {column} {cases} END"


# The WHERE clause makes the precedence rule part of the statement itself: no read-modify-write race.
_UPSERT_BAR_1M = f"""
    INSERT INTO bars_1m ({_BAR_FIELDS})
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (symbol, ts) DO UPDATE SET
        open = excluded.open, high = excluded.high, low = excluded.low, close = excluded.close,
        volume = excluded.volume, source = excluded.source
    WHERE {_rank_sql("excluded.source")} >= {_rank_sql("bars_1m.source")}
"""

# The primary key is (symbol, ts) and SQLite has no loose index scan: GROUP BY symbol, MIN(ts)/MAX(ts)
# over all symbols or a bare "ts < ?" would read every row of the table. The recursive query jumps
# from symbol to symbol through the primary key instead (one index seek per symbol); the statements
# below combine it with per-symbol seeks, so their cost depends on the number of symbols, not of bars.
_KNOWN_SYMBOLS = """
    WITH RECURSIVE known (symbol) AS (
        SELECT MIN(symbol) FROM bars_1m
        UNION ALL
        SELECT (SELECT MIN(symbol) FROM bars_1m WHERE symbol > known.symbol)
        FROM known WHERE known.symbol IS NOT NULL
    )
"""
_LAST_BAR_PER_SYMBOL = f"""{_KNOWN_SYMBOLS}
    SELECT symbol, (SELECT MAX(ts) FROM bars_1m WHERE bars_1m.symbol = known.symbol)
    FROM known WHERE symbol IS NOT NULL
"""
# Only COUNT(*) still walks the table (SQLite counts b-tree entries without decoding the rows).
_BAR_STATS = f"""{_KNOWN_SYMBOLS}
    SELECT (SELECT COUNT(*) FROM bars_1m),
           MIN((SELECT MIN(ts) FROM bars_1m WHERE bars_1m.symbol = known.symbol)),
           MAX((SELECT MAX(ts) FROM bars_1m WHERE bars_1m.symbol = known.symbol))
    FROM known WHERE symbol IS NOT NULL
"""
_DELETE_BARS_BEFORE = f"""
    DELETE FROM bars_1m
    WHERE symbol IN ({_KNOWN_SYMBOLS} SELECT symbol FROM known WHERE symbol IS NOT NULL) AND ts < ?
"""


def _check_source(source: str) -> None:
    if source not in SOURCE_RANK:
        raise ValueError(f"unknown bar source {source!r}, expected one of {sorted(SOURCE_RANK)}")


def _floor_to_bar(ts: np.ndarray) -> np.ndarray:
    return ts // BAR_SECONDS * BAR_SECONDS


def _write_bars_1m(db: Database, rows: list[tuple[Any, ...]]) -> int:
    """Upsert prepared rows in one transaction; returns how many of them were actually written."""
    if not rows:
        return 0
    return db.executemany(_UPSERT_BAR_1M, rows)


def upsert_bars_1m(db: Database, symbol: str, frame: pd.DataFrame, source: str) -> int:
    """Store a canonical 1-minute frame under the precedence rule of ``SOURCE_RANK``.

    Rows with a non-finite or non-positive open/high/low/close are dropped, timestamps are floored to
    full minutes, high/low are widened to contain open and close, and a missing (NaN), non-finite or
    negative volume counts as 0. A row replaces a stored bar only if ``source`` ranks at least as high as
    the stored source. Rows are applied in frame order within one transaction, so of several rows for
    the same minute the last one wins. Returns the number of rows written (inserted or replaced).
    """
    _check_source(source)
    ts, values = bar_arrays(frame)
    keep, cleaned = _clean_ohlcv(values)
    if len(cleaned) < len(values):
        log.debug("%s: skipped %d of %d bars without valid prices", symbol, len(values) - len(cleaned), len(values))
    minutes = _floor_to_bar(ts[keep])
    return _write_bars_1m(db, list(zip(repeat(symbol), minutes.tolist(), *cleaned.T.tolist(), repeat(source))))


def upsert_bar_rows(db: Database, bars: Iterable[Bar]) -> int:
    """Store single bars of any symbols and sources with the rules of ``upsert_bars_1m``."""
    batch = list(bars)
    for bar in batch:
        _check_source(bar.source)
    values = np.array(
        [(bar.open, bar.high, bar.low, bar.close, bar.volume) for bar in batch], dtype=np.float64
    ).reshape(-1, len(BAR_COLUMNS))
    keep, cleaned = _clean_ohlcv(values)
    kept = list(compress(batch, keep.tolist()))
    if len(kept) < len(batch):
        log.debug("skipped %d of %d bars without valid prices", len(batch) - len(kept), len(batch))
    minutes = _floor_to_bar(np.array([bar.ts for bar in kept], dtype=np.int64))
    symbols = [bar.symbol for bar in kept]
    sources = [bar.source for bar in kept]
    return _write_bars_1m(db, list(zip(symbols, minutes.tolist(), *cleaned.T.tolist(), sources, strict=True)))


def _read_minute_rows(
    db: Database,
    select: str,
    dtype: np.dtype,
    conditions: Sequence[str],
    params: list[Any],
    *,
    limit: int | None,
    newest: bool,
) -> np.ndarray:
    """Read 1-minute rows into an ascending structured array; with a ``limit`` keep the newest or the oldest rows."""
    order = "DESC" if newest else "ASC"
    sql, params = _with_limit(f"{select} WHERE {' AND '.join(conditions)} ORDER BY ts {order}", params, limit)
    rows = _fetch_tuples(db, sql, params)
    data = np.fromiter(rows, dtype=dtype, count=len(rows))
    return data[::-1] if newest else data


def _read_minute_frame(
    db: Database, conditions: Sequence[str], params: list[Any], *, limit: int | None, newest: bool
) -> pd.DataFrame:
    """Read 1-minute bars into a canonical frame; with a ``limit`` keep the newest or the oldest rows."""
    data = _read_minute_rows(
        db, _SELECT_MINUTE_FRAME, _MINUTE_DTYPE, conditions, params, limit=limit, newest=newest
    )
    return _minute_frame(data)


def _range_conditions(symbol: str, start_ts: int | None, end_ts: int | None) -> tuple[list[str], list[Any]]:
    """WHERE conditions and parameters for the bars of ``symbol`` with ``start_ts <= open time < end_ts``."""
    conditions = ["symbol = ?"]
    params: list[Any] = [symbol]
    if start_ts is not None:
        conditions.append("ts >= ?")
        params.append(int(start_ts))
    if end_ts is not None:
        conditions.append("ts < ?")
        params.append(int(end_ts))
    return conditions, params


def read_bars_1m(
    db: Database, symbol: str, *, start_ts: int | None = None, end_ts: int | None = None, limit: int | None = None
) -> pd.DataFrame:
    """1-minute bars with ``start_ts <= open time < end_ts`` as canonical frame (ascending).

    ``limit`` keeps the NEWEST rows of the range.
    """
    conditions, params = _range_conditions(symbol, start_ts, end_ts)
    return _read_minute_frame(db, conditions, params, limit=limit, newest=True)


def read_bars_1m_with_sources(
    db: Database, symbol: str, *, start_ts: int | None = None, end_ts: int | None = None, limit: int | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """``read_bars_1m`` plus the stored source of every row.

    Returns ``(frame, sources)``: ``frame`` is exactly the canonical frame ``read_bars_1m`` returns for the
    same arguments (read in the same statement, so both parts belong to one consistent snapshot), and
    ``sources`` a numpy ``str`` array of the same length whose element ``i`` is the source of row ``i``
    ('rest', 'stream' or 'quote', see ``SOURCE_RANK``). ``limit`` keeps the NEWEST rows of the range.
    """
    conditions, params = _range_conditions(symbol, start_ts, end_ts)
    data = _read_minute_rows(
        db, _SELECT_MINUTE_SOURCES, _MINUTE_SOURCE_DTYPE, conditions, params, limit=limit, newest=True
    )
    return _minute_frame(data), data["source"].copy()


def bars_after(db: Database, symbol: str, ts: int, *, limit: int | None = None) -> pd.DataFrame:
    """1-minute bars with an open time strictly greater than ``ts`` as canonical frame (ascending).

    ``limit`` keeps the OLDEST rows, i.e. the bars that directly follow ``ts``: a consumer that works
    through the bars in chunks continues with the last ``ts`` it has seen and never skips a bar.
    """
    return _read_minute_frame(db, ["symbol = ?", "ts > ?"], [symbol, int(ts)], limit=limit, newest=False)


def first_bar_at_or_after(db: Database, symbol: str, ts: int) -> Bar | None:
    """The oldest bar with an open time ``>= ts``."""
    row = db.query_one(f"{_SELECT_BAR} WHERE symbol = ? AND ts >= ? ORDER BY ts LIMIT 1", (symbol, int(ts)))
    return None if row is None else Bar(*row)


def last_bar(db: Database, symbol: str, *, before_ts: int | None = None) -> Bar | None:
    """The newest bar; with ``before_ts`` the newest bar with an open time ``< before_ts``."""
    condition = "symbol = ?"
    params: list[Any] = [symbol]
    if before_ts is not None:
        condition += " AND ts < ?"
        params.append(int(before_ts))
    row = db.query_one(f"{_SELECT_BAR} WHERE {condition} ORDER BY ts DESC LIMIT 1", params)
    return None if row is None else Bar(*row)


def bar_range(db: Database, symbol: str) -> tuple[int, int] | None:
    """Open times of the first and the last stored bar of ``symbol``; ``None`` without bars."""
    first_ts, last_ts = db.query(
        "SELECT (SELECT MIN(ts) FROM bars_1m WHERE symbol = :symbol),"
        " (SELECT MAX(ts) FROM bars_1m WHERE symbol = :symbol)",
        {"symbol": symbol},
    )[0]
    return None if first_ts is None else (first_ts, last_ts)


def bar_stats(db: Database) -> BarStats:
    """Row count and time span of all 1-minute bars (one statement, i.e. one consistent snapshot).

    The time span comes straight from the primary key. Counting still has to walk the whole table
    (roughly 60 ms per million rows): meant for status pages, which should cache the result instead
    of asking on every request.
    """
    rows, first_ts, last_ts = db.query(_BAR_STATS)[0]
    return BarStats(rows=rows, first_ts=first_ts, last_ts=last_ts)


def symbols_with_bars(db: Database) -> dict[str, int]:
    """Every symbol that has 1-minute bars, mapped to the open time of its newest bar."""
    return {symbol: last_ts for symbol, last_ts in db.query(_LAST_BAR_PER_SYMBOL)}


def delete_bars_before(db: Database, ts: int) -> int:
    """Delete the 1-minute bars of all symbols with an open time ``< ts``; returns the number of deleted rows.

    Deletes one primary-key range per symbol, so the write lock is held only as long as it takes to
    remove the outdated rows - not for a scan of the whole table.
    """
    with db.write() as conn:
        return conn.execute(_DELETE_BARS_BEFORE, (int(ts),)).rowcount


# ------------------------------------------------------------------------------------------------
# Bars (daily)
# ------------------------------------------------------------------------------------------------
_REPLACE_BAR_1D = f"INSERT OR REPLACE INTO bars_1d (symbol, day, {', '.join(BAR_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?, ?)"


def _is_day_key(value: object) -> bool:
    """True for a calendar date written exactly as 'YYYY-MM-DD'."""
    if not isinstance(value, str):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _day_keys(frame: pd.DataFrame) -> list[str]:
    days = frame.index.tolist()
    invalid = [day for day in days if not _is_day_key(day)]
    if invalid:
        raise ValueError(f"daily bar frame needs 'YYYY-MM-DD' strings as index, got {invalid[0]!r}")
    return days


def upsert_bars_1d(db: Database, symbol: str, frame: pd.DataFrame) -> int:
    """Store a canonical daily frame; existing days are always replaced.

    Prices and volume are cleaned like 1-minute bars (see ``upsert_bars_1m``).
    Returns the number of rows written.
    """
    days = _day_keys(frame)
    keep, cleaned = _clean_ohlcv(_ohlcv(frame))
    kept_days = list(compress(days, keep.tolist()))
    if len(kept_days) < len(days):
        log.debug("%s: skipped %d of %d daily bars without valid prices", symbol, len(days) - len(kept_days), len(days))
    if not kept_days:
        return 0
    return db.executemany(_REPLACE_BAR_1D, zip(repeat(symbol), kept_days, *cleaned.T.tolist()))


def read_bars_1d(db: Database, symbol: str, *, end_day: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Daily bars up to and including ``end_day`` as canonical frame (ascending).

    ``limit`` keeps the NEWEST rows.
    """
    condition = "symbol = ?"
    params: list[Any] = [symbol]
    if end_day is not None:
        condition += " AND day <= ?"
        params.append(end_day)
    sql, params = _with_limit(f"{_SELECT_DAILY_FRAME} WHERE {condition} ORDER BY day DESC", params, limit)
    rows = _fetch_tuples(db, sql, params)[::-1]
    values = np.array([row[1:] for row in rows], dtype=np.float64).reshape(-1, len(BAR_COLUMNS))
    return _daily_frame([row[0] for row in rows], values)


def last_day(db: Database, symbol: str) -> str | None:
    """Session date of the newest stored daily bar."""
    return db.scalar("SELECT MAX(day) FROM bars_1d WHERE symbol = ?", (symbol,))


# ------------------------------------------------------------------------------------------------
# Quotes
# ------------------------------------------------------------------------------------------------
_QUOTE_FIELDS = ", ".join(field.name for field in fields(Quote))  # identical to the quotes columns
_REPLACE_QUOTE = f"INSERT OR REPLACE INTO quotes ({_QUOTE_FIELDS}) VALUES ({', '.join('?' * len(fields(Quote)))})"


def upsert_quotes(db: Database, quotes: Iterable[Quote]) -> int:
    """Persist the latest quote per symbol (replaces the stored one). Returns the number of rows written.

    Quotes without a finite, positive price are skipped (logged at debug level) so that one broken quote
    never rolls back the whole batch.
    """
    rows = []
    for quote in quotes:
        price = quote.price
        if price is None or not math.isfinite(price) or price <= 0:
            log.debug("skipping quote of %s without a valid price", quote.symbol)
            continue
        rows.append(astuple(quote))
    if not rows:
        return 0
    return db.executemany(_REPLACE_QUOTE, rows)


def load_quotes(db: Database) -> dict[str, Quote]:
    """All persisted quotes by symbol."""
    return {row["symbol"]: Quote(*row) for row in db.query(f"SELECT {_QUOTE_FIELDS} FROM quotes")}
