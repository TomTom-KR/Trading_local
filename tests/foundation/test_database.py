import sqlite3
import threading

import pytest

from trading_local.db.database import SCHEMA_VERSION, Database, dumps, loads


def test_initialize_is_idempotent_and_sets_version(db: Database) -> None:
    db.initialize()
    assert db.scalar("PRAGMA user_version") == SCHEMA_VERSION
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"instruments", "bars_1m", "bars_1d", "news", "signals", "orders", "positions", "settings"} <= tables
    assert db.scalar("PRAGMA journal_mode") == "wal"
    assert db.scalar("PRAGMA foreign_keys") == 1


def test_write_commits_and_rolls_back(db: Database) -> None:
    with db.write() as conn:
        conn.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('a', '1', 0)")
    assert db.scalar("SELECT value FROM settings WHERE key = 'a'") == "1"

    with pytest.raises(RuntimeError), db.write() as conn:
        conn.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('b', '2', 0)")
        raise RuntimeError("boom")
    assert db.scalar("SELECT COUNT(*) FROM settings WHERE key = 'b'") == 0


def test_nested_write_joins_outer_transaction(db: Database) -> None:
    with pytest.raises(ValueError), db.write() as outer:
        outer.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('x', '1', 0)")
        with db.write() as inner:
            inner.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('y', '2', 0)")
        raise ValueError("outer fails after inner finished")
    assert db.scalar("SELECT COUNT(*) FROM settings") == 0

    with db.write(), db.write() as inner:
        inner.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('z', '3', 0)")
    assert db.scalar("SELECT COUNT(*) FROM settings") == 1


def test_strict_tables_and_checks_reject_bad_rows(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO instruments (symbol, name, asset_class, region, exchange, currency, timezone) "
            "VALUES ('X', 'x', 'bond', 'US', 'XNYS', 'USD', 'UTC')"
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO bars_1m (symbol, ts, open, high, low, close) VALUES ('X', 'abc', 1, 1, 1, 1)")


def test_concurrent_writers_and_readers(db: Database) -> None:
    errors: list[BaseException] = []

    def writer(offset: int) -> None:
        try:
            for i in range(200):
                db.execute(
                    "INSERT INTO bars_1m (symbol, ts, open, high, low, close, volume) VALUES (?, ?, 1, 1, 1, 1, 0)",
                    (f"S{offset}", i * 60),
                )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            for _ in range(300):
                db.scalar("SELECT COUNT(*) FROM bars_1m")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert db.scalar("SELECT COUNT(*) FROM bars_1m") == 800


def test_closed_database_refuses_work(settings) -> None:
    database = Database(settings.db_path)
    database.initialize()
    database.close()
    with pytest.raises(RuntimeError):
        database.query("SELECT 1")


def test_json_helpers() -> None:
    assert dumps({"a": [1, "ä"]}) == '{"a":[1,"ä"]}'
    assert loads('{"a": 1}') == {"a": 1}
    assert loads(None, []) == []
    assert loads("not json", {}) == {}
