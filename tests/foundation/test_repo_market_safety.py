"""Review tests for ``trading_local.db.repo_market``: properties the functional tests do not pin down.

* symbols and filters are bound as data, never interpreted as SQL
* a failing row rolls back the whole batch (one transaction per call)
* the precedence rule holds between the rows of ONE batch
* queries on the bar tables go through the primary key instead of walking the table
"""

import random
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
import pytest

from trading_local.clock import to_ts
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS, InstrumentBook
from trading_local.models import Bar, Instrument, Quote
from trading_local.testing import SAMPLE_INSTRUMENTS, make_bars, seed_bars, seed_instruments

T0 = to_ts(datetime(2026, 11, 20, 14, 30, tzinfo=UTC))

HOSTILE_SYMBOLS = [
    "X'; DROP TABLE bars_1m; --",
    'A" OR "1"="1',
    "B' OR '1'='1",
    "%",
    "_",
    "C;D",
    "",
    "Ünïcode-€",
    "?",
    ":symbol",
]


def table_names(db: Database) -> set[str]:
    return {row["name"] for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}


def vm_steps(db: Database, action: Callable[[], object]) -> int:
    """Number of SQLite virtual-machine instructions the calling thread's connection runs for ``action``.

    A statement that walks a table needs several instructions per row; one that seeks through an index
    needs a number that does not depend on the size of the table. Counting instructions makes that
    visible without measuring time.
    """
    steps = 0

    def count() -> int:
        nonlocal steps
        steps += 1
        return 0

    conn = db.conn()
    conn.set_progress_handler(count, 1)
    try:
        action()
    finally:
        conn.set_progress_handler(None, 1)
    return steps


# ------------------------------------------------------------------------------------------------
# SQL injection safety
# ------------------------------------------------------------------------------------------------
def test_symbols_are_data_in_every_statement(db: Database) -> None:
    """All hostile symbols live side by side: none is interpreted and none matches another one."""
    tables = table_names(db)
    seed_instruments(db)
    bars = make_bars(T0, 5)
    seed_bars(db, "NVDA", bars)
    daily = pd.DataFrame(
        [[1.0, 2.0, 0.5, 1.5, 10.0]], columns=list(BAR_COLUMNS), index=pd.Index(["2026-11-20"], dtype="str", name="day")
    )
    quotes = {symbol: Quote(symbol, T0, T0, 1.25, market_state=symbol, source=symbol) for symbol in HOSTILE_SYMBOLS}

    for symbol in HOSTILE_SYMBOLS:
        instrument = Instrument(symbol, symbol, "equity", "US", symbol, "USD", aliases=(symbol,), benchmark=symbol)
        assert repo_market.upsert_instruments(db, [instrument]) == 1
        assert repo_market.upsert_bars_1m(db, symbol, bars.head(3), "stream") == 3
        assert repo_market.upsert_bar_rows(db, [Bar(symbol, T0 + 180, 1.0, 1.0, 1.0, 1.0, 1.0, "quote")]) == 1
        assert repo_market.upsert_bars_1d(db, symbol, daily) == 1
        assert repo_market.upsert_quotes(db, [quotes[symbol]]) == 1

    for symbol in HOSTILE_SYMBOLS:
        instrument = Instrument(symbol, symbol, "equity", "US", symbol, "USD", aliases=(symbol,), benchmark=symbol)
        assert repo_market.get_instrument(db, symbol) == instrument
        assert repo_market.set_instrument_flags(db, symbol, tradable=False) == Instrument(
            symbol, symbol, "equity", "US", symbol, "USD", aliases=(symbol,), benchmark=symbol, tradable=False
        )
        assert [found.symbol for found in repo_market.list_instruments(db, exchange=symbol)] == [symbol]
        assert repo_market.read_bars_1m(db, symbol).index.tolist() == [T0, T0 + 60, T0 + 120, T0 + 180]
        assert repo_market.bars_after(db, symbol, T0 + 60).index.tolist() == [T0 + 120, T0 + 180]
        assert repo_market.bar_range(db, symbol) == (T0, T0 + 180)
        first = repo_market.first_bar_at_or_after(db, symbol, T0 + 1)
        last = repo_market.last_bar(db, symbol, before_ts=T0 + 180)
        assert first is not None and first.ts == T0 + 60 and first.symbol == symbol
        assert last is not None and last.ts == T0 + 120
        pd.testing.assert_frame_equal(repo_market.read_bars_1d(db, symbol, end_day="2026-11-20"), daily)
        assert repo_market.last_day(db, symbol) == "2026-11-20"

    assert repo_market.load_quotes(db) == quotes
    assert repo_market.symbols_with_bars(db) == {"NVDA": T0 + 240, **dict.fromkeys(HOSTILE_SYMBOLS, T0 + 180)}
    book = InstrumentBook(db)
    assert all(symbol in book and book.require(symbol).exchange == symbol for symbol in HOSTILE_SYMBOLS)
    assert [instrument.symbol for instrument in book.tradable()] == [
        instrument.symbol for instrument in SAMPLE_INSTRUMENTS if instrument.tradable
    ]
    # nothing else was touched and nothing was interpreted
    assert table_names(db) == tables
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), bars, check_exact=True)
    assert repo_market.list_instruments(db, active_only=False, exchange="XNAS") == [SAMPLE_INSTRUMENTS[0]]
    assert db.scalar("SELECT COUNT(*) FROM instruments") == len(SAMPLE_INSTRUMENTS) + len(HOSTILE_SYMBOLS)
    assert db.scalar("SELECT COUNT(*) FROM bars_1m") == 5 + 4 * len(HOSTILE_SYMBOLS)


def test_instrument_filters_are_compared_literally(db: Database) -> None:
    seed_instruments(db)

    for value in ("US' OR '1'='1", "US'--", "US; DELETE FROM instruments", "%", "US%", "_S", ""):
        assert repo_market.list_instruments(db, region=value) == []
        assert repo_market.list_instruments(db, asset_class=value) == []
        assert repo_market.list_instruments(db, exchange=value) == []
        assert repo_market.get_instrument(db, value) is None
        assert repo_market.set_instrument_flags(db, value, active=False) is None
    assert repo_market.list_instruments(db, active_only=False) == SAMPLE_INSTRUMENTS


def test_end_day_and_source_are_bound_as_data(db: Database) -> None:
    frame = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0, 1.0]], columns=list(BAR_COLUMNS), index=pd.Index(["2026-11-20"], dtype="str", name="day")
    )
    repo_market.upsert_bars_1d(db, "NVDA", frame)

    assert repo_market.read_bars_1d(db, "NVDA", end_day="2026-11-19' OR '1'='1").index.tolist() == []
    assert repo_market.read_bars_1d(db, "NVDA", end_day="9999' --").index.tolist() == ["2026-11-20"]  # plain text order
    with pytest.raises(ValueError, match="unknown bar source"):
        repo_market.upsert_bars_1m(db, "NVDA", make_bars(T0, 1), "rest'); DROP TABLE bars_1m; --")
    assert db.scalar("SELECT COUNT(*) FROM bars_1d") == 1


# ------------------------------------------------------------------------------------------------
# Transactions
# ------------------------------------------------------------------------------------------------
def test_a_failing_bar_row_rolls_back_the_whole_batch(db: Database) -> None:
    seed_bars(db, "NVDA", make_bars(T0, 2))
    before = repo_market.read_bars_1m(db, "NVDA")
    bars = [
        Bar("NVDA", T0 + 120, 5.0, 5.0, 5.0, 5.0, 1.0, "rest"),
        Bar("NVDA", T0, 9.0, 9.0, 9.0, 9.0, 1.0, "rest"),  # would replace a stored bar
        Bar(None, T0 + 180, 5.0, 5.0, 5.0, 5.0, 1.0, "rest"),  # type: ignore[arg-type]  # violates NOT NULL
    ]

    with pytest.raises(sqlite3.IntegrityError):
        repo_market.upsert_bar_rows(db, bars)

    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), before, check_exact=True)
    assert db.scalar("SELECT COUNT(*) FROM bars_1m") == 2


@pytest.mark.parametrize("bad_price", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_quotes_without_a_valid_price_are_skipped(db: Database, bad_price: float | None) -> None:
    batch = [
        Quote("NVDA", T0 + 60, T0 + 60, 150.0),
        Quote("BTC-USD", T0 + 60, T0 + 60, bad_price),  # type: ignore[arg-type]
        Quote("SAP.DE", T0 + 60, T0 + 60, 200.0),
    ]

    assert repo_market.upsert_quotes(db, batch) == 2
    assert set(repo_market.load_quotes(db)) == {"NVDA", "SAP.DE"}


def test_a_failing_quote_rolls_back_the_whole_batch(db: Database) -> None:
    stored = Quote("NVDA", T0, T0, 142.5, bid=142.4)
    repo_market.upsert_quotes(db, [stored])
    batch = [
        Quote("NVDA", T0 + 60, T0 + 60, 150.0),
        Quote("SAP.DE", T0 + 60, T0 + 60, 200.0),
        Quote(None, T0 + 60, T0 + 60, 1.0),  # type: ignore[arg-type]  # a quote needs a symbol
    ]

    with pytest.raises(sqlite3.IntegrityError):
        repo_market.upsert_quotes(db, batch)

    assert repo_market.load_quotes(db) == {"NVDA": stored}


def test_repository_writes_join_an_outer_transaction(db: Database) -> None:
    seed_instruments(db)
    seed_bars(db, "NVDA", make_bars(T0, 3))

    with pytest.raises(RuntimeError, match="abort"), db.write():
        repo_market.upsert_instruments(db, [Instrument("AAPL", "Apple", "equity", "US", "XNAS", "USD")])
        repo_market.set_instrument_flags(db, "NVDA", active=False)
        repo_market.upsert_bars_1m(db, "NVDA", make_bars(T0 + 180, 3, seed=5), "rest")
        repo_market.upsert_bar_rows(db, [Bar("SAP.DE", T0, 1.0, 1.0, 1.0, 1.0)])
        repo_market.upsert_quotes(db, [Quote("NVDA", T0, T0, 1.0)])
        assert repo_market.delete_bars_before(db, T0 + 120) == 3  # 2 of NVDA and the new SAP.DE bar
        assert repo_market.bar_range(db, "NVDA") == (T0 + 120, T0 + 300)  # the transaction sees its own writes
        raise RuntimeError("abort")

    assert repo_market.get_instrument(db, "AAPL") is None
    assert repo_market.get_instrument(db, "NVDA") == SAMPLE_INSTRUMENTS[0]
    assert repo_market.bar_range(db, "NVDA") == (T0, T0 + 120)
    assert repo_market.symbols_with_bars(db) == {"NVDA": T0 + 120}
    assert repo_market.load_quotes(db) == {}


# ------------------------------------------------------------------------------------------------
# Precedence inside one batch
# ------------------------------------------------------------------------------------------------
def test_rows_of_one_batch_compete_under_the_precedence_rule(db: Database) -> None:
    cases: list[tuple[tuple[str, ...], int, int]] = [  # sources in batch order, index of the winner, rows written
        (("rest", "quote"), 0, 1),  # a later row of a lower rank does not replace the earlier one
        (("rest", "stream", "quote"), 0, 1),
        (("stream", "quote", "stream"), 2, 2),
        (("quote", "stream", "rest"), 2, 3),
        (("quote", "quote"), 1, 2),  # equal rank: the last row wins
        (("stream", "rest", "rest", "stream"), 2, 3),
    ]

    for number, (sources, winner, written) in enumerate(cases):
        symbol = f"CASE{number}"
        bars = [
            Bar(symbol, T0 + n, 10.0 + n, 10.0 + n, 10.0 + n, 10.0 + n, float(n), source)  # all in the same minute
            for n, source in enumerate(sources)
        ]

        assert repo_market.upsert_bar_rows(db, bars) == written, sources

        expected = 10.0 + winner
        assert repo_market.last_bar(db, symbol) == Bar(
            symbol, T0, expected, expected, expected, expected, float(winner), sources[winner]
        ), sources
    assert db.scalar("SELECT COUNT(*) FROM bars_1m") == len(cases)  # one bar per case


def test_a_lower_ranked_source_fills_gaps_but_leaves_better_bars_alone(db: Database) -> None:
    rest = make_bars(T0, 10, seed=1)
    stream = make_bars(T0, 12, seed=2)
    quote = make_bars(T0 - 120, 16, seed=3)
    seed_bars(db, "NVDA", rest.iloc[[0, 1, 2, 5, 6]], "rest")
    assert seed_bars(db, "NVDA", stream, "stream") == 7  # minutes 3, 4, 7 ... 11

    assert seed_bars(db, "NVDA", quote, "quote") == 4  # two minutes before and two after everything known

    stored = {row["ts"]: row["source"] for row in db.query("SELECT ts, source FROM bars_1m WHERE symbol = 'NVDA'")}
    assert stored == {
        **{T0 + 60 * n: "quote" for n in (-2, -1, 12, 13)},
        **{T0 + 60 * n: "stream" for n in (3, 4, 7, 8, 9, 10, 11)},
        **{T0 + 60 * n: "rest" for n in (0, 1, 2, 5, 6)},
    }
    frame = repo_market.read_bars_1m(db, "NVDA")
    pd.testing.assert_frame_equal(frame.loc[T0 : T0 + 120], rest.iloc[:3], check_exact=True)
    pd.testing.assert_frame_equal(frame.loc[T0 + 180 : T0 + 240], stream.iloc[3:5], check_exact=True)
    assert seed_bars(db, "NVDA", rest, "rest") == 10  # the authoritative source replaces everything it covers
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA", start_ts=T0, end_ts=T0 + 600), rest)


# ------------------------------------------------------------------------------------------------
# Housekeeping statements against a full scan
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def many_symbols(db: Database) -> dict[str, pd.DataFrame]:
    """Twelve symbols with different, partly overlapping time ranges (6000 bars in total)."""
    rng = random.Random(5)
    frames: dict[str, pd.DataFrame] = {}
    for number in range(12):
        symbol = f"SYM{number:02d}"
        frames[symbol] = make_bars(T0 + 60 * rng.randrange(0, 2000), 500, seed=number)
        seed_bars(db, symbol, frames[symbol])
    return frames


def test_bar_stats_and_symbols_agree_with_a_full_scan(db: Database, many_symbols: dict[str, pd.DataFrame]) -> None:
    scan = db.query("SELECT COUNT(*), MIN(ts), MAX(ts) FROM bars_1m")[0]

    assert repo_market.bar_stats(db) == {"rows": scan[0], "first_ts": scan[1], "last_ts": scan[2]}
    assert repo_market.bar_stats(db) == {
        "rows": 6000,
        "first_ts": min(frame.index[0] for frame in many_symbols.values()),
        "last_ts": max(frame.index[-1] for frame in many_symbols.values()),
    }
    assert repo_market.symbols_with_bars(db) == {symbol: frame.index[-1] for symbol, frame in many_symbols.items()}
    assert list(repo_market.symbols_with_bars(db)) == sorted(many_symbols)
    assert all(
        repo_market.bar_range(db, symbol) == (frame.index[0], frame.index[-1]) for symbol, frame in many_symbols.items()
    )


def test_delete_bars_before_removes_exactly_the_older_bars_of_every_symbol(
    db: Database, many_symbols: dict[str, pd.DataFrame]
) -> None:
    remaining = dict(many_symbols)
    for threshold in (T0 - 60, T0, T0 + 60 * 700 + 1, T0 + 60 * 1400, T0 + 60 * 2200, T0 + 60 * 2600):
        expected = db.scalar("SELECT COUNT(*) FROM bars_1m WHERE ts < ?", (threshold,))

        assert repo_market.delete_bars_before(db, threshold) == expected

        remaining = {symbol: frame[frame.index >= threshold] for symbol, frame in remaining.items()}
        remaining = {symbol: frame for symbol, frame in remaining.items() if not frame.empty}
        assert repo_market.symbols_with_bars(db) == {symbol: frame.index[-1] for symbol, frame in remaining.items()}
        for symbol, frame in remaining.items():
            pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, symbol), frame, check_exact=True)
        assert repo_market.delete_bars_before(db, threshold) == 0  # a bar that opens exactly then is kept

    assert remaining == {}
    assert repo_market.bar_stats(db) == {"rows": 0, "first_ts": None, "last_ts": None}


def test_bar_queries_seek_through_the_primary_key(db: Database, many_symbols: dict[str, pd.DataFrame]) -> None:
    """Nothing but the row count itself may walk the table: the cost must not grow with the bar count."""
    newest = int(many_symbols["SYM05"].index[-1])
    full_scan = vm_steps(db, lambda: db.query("SELECT MAX(ts) FROM bars_1m"))  # reads all 6000 rows
    assert full_scan > 6000

    seeking: dict[str, Callable[[], object]] = {
        "bar_stats": lambda: repo_market.bar_stats(db),
        "symbols_with_bars": lambda: repo_market.symbols_with_bars(db),
        "bar_range": lambda: repo_market.bar_range(db, "SYM05"),
        "last_bar": lambda: repo_market.last_bar(db, "SYM05"),
        "last_bar before": lambda: repo_market.last_bar(db, "SYM05", before_ts=newest),
        "first_bar_at_or_after": lambda: repo_market.first_bar_at_or_after(db, "SYM05", T0),
        "read newest rows": lambda: repo_market.read_bars_1m(db, "SYM05", limit=20),
        "read range": lambda: repo_market.read_bars_1m(db, "SYM05", start_ts=newest - 600, end_ts=newest),
        "bars_after": lambda: repo_market.bars_after(db, "SYM05", T0, limit=20),
        "delete nothing": lambda: repo_market.delete_bars_before(db, T0 - 60),
        "delete a few": lambda: repo_market.delete_bars_before(db, T0 + 60),
    }
    for name, action in seeking.items():
        assert vm_steps(db, action) * 10 < full_scan, name
    assert repo_market.read_bars_1m(db, "SYM05", start_ts=newest - 600, end_ts=newest).shape == (10, 5)


# ------------------------------------------------------------------------------------------------
# Frames read back
# ------------------------------------------------------------------------------------------------
def test_integer_valued_prices_come_back_as_float64(db: Database) -> None:
    frame = pd.DataFrame(
        {"open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [5, 0]},
        index=pd.Index([T0, T0 + 60], dtype="int64", name="ts"),
    )

    assert repo_market.upsert_bars_1m(db, "NVDA", frame, "rest") == 2

    stored = repo_market.read_bars_1m(db, "NVDA")
    assert all(dtype == "float64" for dtype in stored.dtypes) and stored.index.dtype == "int64"
    pd.testing.assert_frame_equal(stored, frame.astype("float64"), check_exact=True)
    newest = repo_market.last_bar(db, "NVDA")
    assert newest == Bar("NVDA", T0 + 60, 101.0, 103.0, 100.0, 102.0, 0.0, "rest")
    assert newest is not None and all(isinstance(value, float) for value in (newest.open, newest.volume))


def test_frames_read_back_are_independent_of_the_database_and_of_each_other(db: Database) -> None:
    bars = make_bars(T0, 5)
    seed_bars(db, "NVDA", bars)

    first = repo_market.read_bars_1m(db, "NVDA", limit=3)
    second = repo_market.read_bars_1m(db, "NVDA", limit=3)
    first.loc[:, "close"] = 0.0
    first.loc[first.index[0], "volume"] = -1.0

    pd.testing.assert_frame_equal(second, bars.tail(3), check_exact=True)
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), bars, check_exact=True)
