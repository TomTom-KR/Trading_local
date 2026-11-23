"""SQLite access: one database file in WAL mode, one connection per thread, serialised writers.

Usage
-----
    db = Database(settings.db_path)
    db.initialize()                                   # creates the file and applies schema.sql (idempotent)

    rows = db.query("SELECT * FROM instruments WHERE region = ?", ("US",))     # list[sqlite3.Row]
    with db.write() as conn:                          # one exclusive transaction (BEGIN IMMEDIATE ... COMMIT)
        conn.execute("UPDATE portfolios SET cash = cash - ? WHERE id = ?", (100.0, 1))
        conn.executemany("INSERT INTO ... VALUES (?, ?)", rows)

Rules
* Connections are in SQLite autocommit mode; all multi-statement changes go through ``write()``.
* ``write()`` is re-entrant per thread: a nested ``write()`` joins the outer transaction (no savepoints -
  an exception that escapes any level rolls back the whole transaction).
* Readers never block writers and vice versa (WAL). Always consume results completely (``fetchall``):
  an unfinished cursor keeps a read snapshot open.
* The class is thread-safe. Blocking calls belong in worker threads, not on the asyncio event loop
  (single fast statements from the loop are acceptable).
"""

import contextlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# numpy scalars would otherwise be bound as BLOBs: STRICT tables reject them on INSERT and, worse,
# comparisons such as "ts < ?" silently match every row. Bar frames hand out numpy ints via their index.
for _int_type in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
    sqlite3.register_adapter(_int_type, int)
for _float_type in (np.float16, np.float32, np.float64):
    sqlite3.register_adapter(_float_type, float)
sqlite3.register_adapter(np.bool_, int)

SCHEMA_VERSION = 1
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")

Params = Sequence[Any] | dict[str, Any]


def _json_default(value: Any) -> Any:
    """Serialise numpy scalars/arrays and other objects with ``item()`` / ``tolist()`` as plain Python values."""
    for attr in ("tolist", "item"):
        method = getattr(value, attr, None)
        if callable(method):
            return method()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def dumps(value: Any) -> str:
    """Compact JSON text for JSON columns (numpy scalars and arrays are converted to plain values).

    Non-finite floats (NaN, inf) are not valid JSON and are rejected with ``ValueError``.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False)


def loads(text: str | None, default: Any = None) -> Any:
    """Parse a JSON column; ``default`` for NULL / empty / broken content."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False

    # -- connections ----------------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, autocommit=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache per connection
        return conn

    def conn(self) -> sqlite3.Connection:
        """The calling thread's connection (created on first use)."""
        if self._closed:
            raise RuntimeError("database is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            self._local.depth = 0
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def initialize(self) -> None:
        """Create the database file if needed and apply the schema (idempotent)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        script = _SCHEMA_FILE.read_text(encoding="utf-8")
        with self.write() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema version {version} is newer than this program ({SCHEMA_VERSION})"
                )
            for statement in _split_statements(script):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        """Close all connections (call once at shutdown, after worker threads have finished)."""
        with self._conns_lock:
            conns, self._conns = self._conns, []
            self._closed = True
        for conn in conns:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("PRAGMA optimize")
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # -- writing --------------------------------------------------------------------------------
    @contextlib.contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """Exclusive write transaction; re-entrant within the same thread."""
        with self._write_lock:
            conn = self.conn()
            depth = self._local.depth
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE")
            self._local.depth = depth + 1
            try:
                yield conn
            except BaseException:
                self._local.depth = depth
                if depth == 0 and conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            else:
                self._local.depth = depth
                if depth == 0:
                    conn.execute("COMMIT")

    def execute(self, sql: str, params: Params = ()) -> int:
        """Run one modifying statement in its own transaction. Returns ``lastrowid`` for INSERTs."""
        with self.write() as conn:
            cur = conn.execute(sql, params)
            return cur.lastrowid or 0

    def executemany(self, sql: str, rows: Iterable[Params]) -> int:
        """Run one statement for many parameter sets in a single transaction. Returns the affected row count."""
        with self.write() as conn:
            cur = conn.executemany(sql, rows)
            return cur.rowcount

    # -- reading --------------------------------------------------------------------------------
    def query(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        rows = self.conn().execute(sql, params).fetchall()
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Params = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if row is None or row[0] is None:
            return default
        return row[0]


def _split_statements(script: str) -> list[str]:
    """Split an SQL script into single statements (comment lines removed).

    schema.sql contains no triggers and no semicolons inside string literals, so splitting on ';'
    is sufficient. ``executescript`` is avoided because it cannot run inside our explicit transaction
    in every Python version.
    """
    lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        # drop trailing comments
        cut = line.find("--")
        lines.append(line if cut < 0 else line[:cut])
    statements = [part.strip() for part in "\n".join(lines).split(";")]
    return [s for s in statements if s]
