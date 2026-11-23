import sqlite3
import threading
import time
from dataclasses import fields, replace
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_local.clock import to_ts
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS, SOURCE_RANK, InstrumentBook
from trading_local.models import Bar, Instrument, Quote
from trading_local.testing import SAMPLE_INSTRUMENTS, make_bars, seed_instruments

T0 = to_ts(datetime(2026, 11, 20, 14, 30, tzinfo=UTC))  # Friday, New York opens
SOURCES = sorted(SOURCE_RANK)
NAN = float("nan")
INF = float("inf")


def minute_frame(*rows: tuple[float, ...]) -> pd.DataFrame:
    """Canonical 1-minute frame from (ts, open, high, low, close, volume) rows."""
    return pd.DataFrame(
        [row[1:] for row in rows],
        columns=list(BAR_COLUMNS),
        dtype="float64",
        index=pd.Index([row[0] for row in rows], dtype="int64", name="ts"),
    )


def daily_frame(*rows: tuple[object, ...]) -> pd.DataFrame:
    """Canonical daily frame from (day, open, high, low, close, volume) rows."""
    return pd.DataFrame(
        [row[1:] for row in rows],
        columns=list(BAR_COLUMNS),
        dtype="float64",
        index=pd.Index([row[0] for row in rows], dtype="str", name="day"),
    )


def flat_bar(ts: int, price: float, volume: float = 10.0) -> tuple[float, ...]:
    return (ts, price, price, price, price, volume)


def stored_bars(db: Database, symbol: str) -> list[tuple[object, ...]]:
    """Raw table content of one symbol: (ts, open, high, low, close, volume, source)."""
    rows = db.query(
        "SELECT ts, open, high, low, close, volume, source FROM bars_1m WHERE symbol = ? ORDER BY ts", (symbol,)
    )
    return [tuple(row) for row in rows]


def assert_canonical_minutes(frame: pd.DataFrame) -> None:
    assert frame.index.name == "ts"
    assert frame.index.dtype == np.int64
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


def assert_canonical_daily(frame: pd.DataFrame) -> None:
    assert frame.index.name == "day"
    assert frame.index.dtype == "str"
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    assert frame.index.is_monotonic_increasing
    assert frame.index.is_unique


@pytest.mark.parametrize(("table", "model"), [("instruments", Instrument), ("bars_1m", Bar), ("quotes", Quote)])
def test_models_list_the_table_columns_in_schema_order(db: Database, table: str, model: type) -> None:
    """The repository derives its column lists from the dataclasses."""
    columns = [row["name"] for row in db.query(f"PRAGMA table_info({table})")]

    assert [field.name for field in fields(model)] == columns


# ------------------------------------------------------------------------------------------------
# Instruments
# ------------------------------------------------------------------------------------------------
def test_instruments_round_trip_with_json_and_flag_columns(db: Database) -> None:
    assert repo_market.upsert_instruments(db, SAMPLE_INSTRUMENTS) == len(SAMPLE_INSTRUMENTS)

    for instrument in SAMPLE_INSTRUMENTS:
        assert repo_market.get_instrument(db, instrument.symbol) == instrument

    shell = repo_market.get_instrument(db, "SHEL.L")
    assert shell is not None
    assert shell.aliases == ("Shell",)
    assert shell.price_scale == 0.01
    assert shell.active is True and shell.tradable is True
    fx = repo_market.get_instrument(db, "EURUSD=X")
    assert fx is not None and fx.tradable is False and fx.benchmark is None and fx.aliases == ()


def test_get_instrument_returns_none_for_unknown_symbol(db: Database) -> None:
    seed_instruments(db)
    assert repo_market.get_instrument(db, "UNKNOWN") is None


def test_upsert_instruments_returns_number_of_rows_inserted_or_updated(db: Database) -> None:
    assert repo_market.upsert_instruments(db, []) == 0
    assert repo_market.upsert_instruments(db, SAMPLE_INSTRUMENTS[:4]) == 4
    assert repo_market.upsert_instruments(db, iter(SAMPLE_INSTRUMENTS[2:7])) == 5  # 2 updated, 3 inserted
    assert db.scalar("SELECT COUNT(*) FROM instruments") == 7


def test_upsert_instruments_updates_all_descriptive_columns(db: Database) -> None:
    seed_instruments(db)
    changed = Instrument(
        symbol="NVDA",
        name="NVIDIA Corp.",
        asset_class="index",
        region="EU",
        exchange="XETR",
        currency="EUR",
        price_scale=0.5,
        timezone="Europe/Berlin",
        delay_min=20,
        benchmark="^GDAXI",
        aliases=("Chips", "Grafikkarten"),
        sort_order=42,
    )

    assert repo_market.upsert_instruments(db, [changed]) == 1

    assert repo_market.get_instrument(db, "NVDA") == changed


def test_upsert_instruments_keeps_user_flags_of_existing_rows_by_default(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "NVDA", active=False, tradable=False)
    repo_market.set_instrument_flags(db, "EURUSD=X", tradable=True)
    newcomer = Instrument("AAPL", "Apple", "equity", "US", "XNAS", "USD", active=False, tradable=False)
    renamed = [replace(instrument, name=instrument.name + " (neu)") for instrument in SAMPLE_INSTRUMENTS]

    repo_market.upsert_instruments(db, [*renamed, newcomer])

    nvda = repo_market.get_instrument(db, "NVDA")
    assert nvda == replace(SAMPLE_INSTRUMENTS[0], name="NVIDIA (neu)", active=False, tradable=False)
    fx = repo_market.get_instrument(db, "EURUSD=X")
    assert fx is not None and fx.tradable is True  # the stored flag wins over tradable=False of the sample
    assert repo_market.get_instrument(db, "AAPL") == newcomer  # a new row takes its flags from the instrument


def test_upsert_instruments_overwrites_user_flags_on_request(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "NVDA", active=False, tradable=False)

    repo_market.upsert_instruments(db, SAMPLE_INSTRUMENTS, keep_user_flags=False)

    assert repo_market.get_instrument(db, "NVDA") == SAMPLE_INSTRUMENTS[0]


def test_upsert_instruments_is_one_transaction(db: Database) -> None:
    broken = replace(SAMPLE_INSTRUMENTS[1], asset_class="bond")  # violates the CHECK constraint

    with pytest.raises(sqlite3.IntegrityError):
        repo_market.upsert_instruments(db, [SAMPLE_INSTRUMENTS[0], broken])

    assert db.scalar("SELECT COUNT(*) FROM instruments") == 0


def test_list_instruments_is_ordered_by_sort_order_then_symbol(db: Database) -> None:
    tied = [
        Instrument("ZZZ", "Z", "equity", "US", "XNYS", "USD", sort_order=1),
        Instrument("AAA", "A", "equity", "US", "XNYS", "USD", sort_order=1),
        Instrument("MMM", "M", "equity", "US", "XNYS", "USD", sort_order=0),
    ]
    seed_instruments(db, tied)

    assert [instrument.symbol for instrument in repo_market.list_instruments(db)] == ["MMM", "AAA", "ZZZ"]


def test_list_instruments_filters(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "SAP.DE", active=False)

    def symbols(**filters: object) -> list[str]:
        return [instrument.symbol for instrument in repo_market.list_instruments(db, **filters)]

    everything = [instrument.symbol for instrument in SAMPLE_INSTRUMENTS]
    assert symbols(active_only=False) == everything
    assert symbols() == [symbol for symbol in everything if symbol != "SAP.DE"]
    assert symbols(tradable_only=True) == ["NVDA", "SHEL.L", "7203.T", "BTC-USD", "^GDAXI", "^GSPC"]
    assert symbols(active_only=False, tradable_only=True)[:2] == ["NVDA", "SAP.DE"]
    assert symbols(region="EU") == ["SHEL.L", "^GDAXI"]
    assert symbols(asset_class="fx") == ["EURUSD=X", "EURGBP=X", "EURJPY=X"]
    assert symbols(exchange="XETR") == ["^GDAXI"]
    assert symbols(exchange="XETR", active_only=False) == ["SAP.DE", "^GDAXI"]
    assert symbols(region="EU", asset_class="equity", exchange="XLON", tradable_only=True) == ["SHEL.L"]
    assert symbols(region="US", asset_class="crypto") == []


def test_set_instrument_flags_changes_only_the_given_flags(db: Database) -> None:
    seed_instruments(db)
    nvda = SAMPLE_INSTRUMENTS[0]

    assert repo_market.set_instrument_flags(db, "NVDA", active=False) == replace(nvda, active=False)
    assert repo_market.set_instrument_flags(db, "NVDA", tradable=False) == replace(nvda, active=False, tradable=False)
    assert repo_market.set_instrument_flags(db, "NVDA") == replace(nvda, active=False, tradable=False)
    assert repo_market.set_instrument_flags(db, "NVDA", active=True, tradable=True) == nvda
    assert repo_market.get_instrument(db, "NVDA") == nvda
    assert repo_market.get_instrument(db, "SAP.DE") == SAMPLE_INSTRUMENTS[1]  # other rows untouched


def test_set_instrument_flags_returns_none_for_unknown_symbol(db: Database) -> None:
    seed_instruments(db)
    assert repo_market.set_instrument_flags(db, "UNKNOWN", active=False) is None


# ------------------------------------------------------------------------------------------------
# InstrumentBook
# ------------------------------------------------------------------------------------------------
def test_instrument_book_loads_lazily_and_only_once(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    statements: list[str] = []
    original = db.query

    def counting_query(sql: str, params: object = ()) -> list[sqlite3.Row]:
        statements.append(sql)
        return original(sql, params)

    monkeypatch.setattr(db, "query", counting_query)
    book = InstrumentBook(db)
    seed_instruments(db)  # stored after the book was created, before its first use
    assert statements == []

    assert book.get("NVDA") == SAMPLE_INSTRUMENTS[0]
    assert "SAP.DE" in book
    assert len(book.all()) == len(SAMPLE_INSTRUMENTS)
    assert len(statements) == 1


def test_instrument_book_loads_once_when_many_threads_use_it_first(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_instruments(db)
    loads = 0
    original = db.query

    def slow_query(sql: str, params: object = ()) -> list[sqlite3.Row]:
        nonlocal loads
        loads += 1
        time.sleep(0.05)  # long enough for the other threads to arrive while the first one loads
        return original(sql, params)

    monkeypatch.setattr(db, "query", slow_query)
    book = InstrumentBook(db)
    start = threading.Barrier(6)
    seen: list[Instrument | None] = []

    def first_use() -> None:
        start.wait()
        seen.append(book.get("NVDA"))

    threads = [threading.Thread(target=first_use) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen == [SAMPLE_INSTRUMENTS[0]] * 6
    assert loads == 1


def test_instrument_book_accessors(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "SAP.DE", active=False)
    book = InstrumentBook(db)
    everything = [instrument.symbol for instrument in SAMPLE_INSTRUMENTS]
    active = [symbol for symbol in everything if symbol != "SAP.DE"]

    assert book.get("NVDA") == SAMPLE_INSTRUMENTS[0]
    assert book.get("UNKNOWN") is None
    assert book.require("NVDA") == SAMPLE_INSTRUMENTS[0]
    with pytest.raises(KeyError):
        book.require("UNKNOWN")
    assert "NVDA" in book and "SAP.DE" in book and "UNKNOWN" not in book
    assert [instrument.symbol for instrument in book.all()] == active
    assert [instrument.symbol for instrument in book.all(active_only=False)] == everything
    assert book.symbols() == active
    assert book.symbols(active_only=False) == everything
    tradable = [instrument.symbol for instrument in book.tradable()]
    assert tradable == ["NVDA", "SHEL.L", "7203.T", "BTC-USD", "^GDAXI", "^GSPC"]
    inactive = book.get("SAP.DE")
    assert inactive is not None and inactive.active is False


def test_instrument_book_groups_active_instruments_by_exchange(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "SAP.DE", active=False)
    repo_market.set_instrument_flags(db, "SHEL.L", active=False)

    groups = InstrumentBook(db).by_exchange()

    assert {exchange: [instrument.symbol for instrument in group] for exchange, group in groups.items()} == {
        "XNAS": ["NVDA"],
        "XTKS": ["7203.T"],
        "CRYPTO": ["BTC-USD"],
        "XETR": ["^GDAXI"],
        "XNYS": ["^GSPC"],
        "FX": ["EURUSD=X", "EURGBP=X", "EURJPY=X"],
    }


def test_instrument_book_shows_changes_only_after_reload(db: Database) -> None:
    seed_instruments(db)
    book = InstrumentBook(db)
    assert "AAPL" not in book

    repo_market.upsert_instruments(db, [Instrument("AAPL", "Apple", "equity", "US", "XNAS", "USD", sort_order=0)])
    repo_market.set_instrument_flags(db, "NVDA", tradable=False)
    assert "AAPL" not in book
    assert book.require("NVDA").tradable is True

    book.reload()
    assert book.symbols()[0] == "AAPL"
    assert book.require("NVDA").tradable is False
    assert "NVDA" not in [instrument.symbol for instrument in book.tradable()]


def test_instrument_book_hands_out_copies(db: Database) -> None:
    seed_instruments(db)
    book = InstrumentBook(db)

    book.all().clear()
    book.tradable().clear()
    book.symbols().clear()
    groups = book.by_exchange()
    groups["FX"].clear()
    groups.clear()

    assert len(book.all()) == len(SAMPLE_INSTRUMENTS)
    assert len(book.tradable()) == 7
    assert len(book.symbols()) == len(SAMPLE_INSTRUMENTS)
    assert len(book.by_exchange()["FX"]) == 3


def test_instrument_book_is_consistent_while_reloaded_from_another_thread(db: Database) -> None:
    seed_instruments(db)
    repo_market.set_instrument_flags(db, "SAP.DE", active=False)
    book = InstrumentBook(db)  # not loaded yet: the first readers race for the lazy load
    start = threading.Barrier(5)
    read_pass = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def toggle() -> None:
        """Every committed state has exactly one of NVDA / SAP.DE active."""
        try:
            start.wait()
            for step in range(100):
                with db.write():
                    repo_market.set_instrument_flags(db, "NVDA", active=step % 2 == 1)
                    repo_market.set_instrument_flags(db, "SAP.DE", active=step % 2 == 0)
                book.reload()
                read_pass.clear()
                assert read_pass.wait(10), "no reader finished a pass on the reloaded book"
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    def read() -> None:
        try:
            start.wait()
            while not done.is_set():
                active = book.symbols()
                assert len({"NVDA", "SAP.DE"} & set(active)) == 1
                assert len(active) == len(SAMPLE_INSTRUMENTS) - 1
                assert len(book.all(active_only=False)) == len(SAMPLE_INSTRUMENTS)
                assert sum(len(group) for group in book.by_exchange().values()) == len(SAMPLE_INSTRUMENTS) - 1
                read_pass.set()
                time.sleep(0.0005)  # readers spinning without a pause would starve the writer of the GIL
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=toggle), *(threading.Thread(target=read) for _ in range(4))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert book.require("NVDA").active is True and book.require("SAP.DE").active is False


# ------------------------------------------------------------------------------------------------
# 1-minute bars: writing
# ------------------------------------------------------------------------------------------------
def test_source_rank_orders_quote_stream_rest() -> None:
    assert SOURCE_RANK == {"quote": 0, "stream": 1, "rest": 2}


@pytest.mark.parametrize("incoming", SOURCES)
@pytest.mark.parametrize("stored", SOURCES)
def test_bar_is_replaced_only_by_a_source_of_equal_or_higher_rank(db: Database, stored: str, incoming: str) -> None:
    repo_market.upsert_bars_1m(db, "NVDA", minute_frame((T0, 100.0, 101.0, 99.0, 100.5, 10.0)), stored)

    written = repo_market.upsert_bars_1m(db, "NVDA", minute_frame((T0, 200.0, 202.0, 198.0, 201.0, 20.0)), incoming)

    if SOURCE_RANK[incoming] >= SOURCE_RANK[stored]:
        assert written == 1
        assert stored_bars(db, "NVDA") == [(T0, 200.0, 202.0, 198.0, 201.0, 20.0, incoming)]
    else:
        assert written == 0
        assert stored_bars(db, "NVDA") == [(T0, 100.0, 101.0, 99.0, 100.5, 10.0, stored)]


@pytest.mark.parametrize("incoming", SOURCES)
@pytest.mark.parametrize("stored", SOURCES)
def test_bar_rows_follow_the_same_precedence_rule(db: Database, stored: str, incoming: str) -> None:
    repo_market.upsert_bar_rows(db, [Bar("NVDA", T0, 100.0, 101.0, 99.0, 100.5, 10.0, stored)])

    written = repo_market.upsert_bar_rows(db, [Bar("NVDA", T0, 200.0, 202.0, 198.0, 201.0, 20.0, incoming)])

    replaced = SOURCE_RANK[incoming] >= SOURCE_RANK[stored]
    assert written == int(replaced)
    assert repo_market.last_bar(db, "NVDA") == (
        Bar("NVDA", T0, 200.0, 202.0, 198.0, 201.0, 20.0, incoming)
        if replaced
        else Bar("NVDA", T0, 100.0, 101.0, 99.0, 100.5, 10.0, stored)
    )


def test_upsert_bars_1m_returns_only_the_rows_actually_written(db: Database) -> None:
    repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0, 100.0)), "rest")
    repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0 + 60, 100.0)), "stream")
    repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0 + 120, 100.0)), "quote")
    incoming = minute_frame(*(flat_bar(T0 + 60 * n, 200.0) for n in range(4)))

    written = repo_market.upsert_bars_1m(db, "NVDA", incoming, "stream")

    assert written == 3  # the REST bar stays, two bars are replaced, one is new
    assert [(row[1], row[6]) for row in stored_bars(db, "NVDA")] == [
        (100.0, "rest"),
        (200.0, "stream"),
        (200.0, "stream"),
        (200.0, "stream"),
    ]


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.0, -1.0])
def test_upsert_bars_1m_drops_rows_with_invalid_prices(db: Database, column: str, bad: float) -> None:
    frame = minute_frame(
        (T0, 100.0, 101.0, 99.0, 100.5, 10.0),
        (T0 + 60, 100.0, 101.0, 99.0, 100.5, 10.0),
        (T0 + 120, 100.0, 101.0, 99.0, 100.5, 10.0),
    )
    frame.loc[T0 + 60, column] = bad

    assert repo_market.upsert_bars_1m(db, "NVDA", frame, "rest") == 2

    assert [row[0] for row in stored_bars(db, "NVDA")] == [T0, T0 + 120]


def test_upsert_bars_1m_treats_missing_or_invalid_volume_as_zero(db: Database) -> None:
    frame = minute_frame(
        (T0, 100.0, 101.0, 99.0, 100.5, NAN),
        (T0 + 60, 100.0, 101.0, 99.0, 100.5, -5.0),
        (T0 + 120, 100.0, 101.0, 99.0, 100.5, INF),
        (T0 + 180, 100.0, 101.0, 99.0, 100.5, 0.0),
        (T0 + 240, 100.0, 101.0, 99.0, 100.5, 7.5),
    )

    assert repo_market.upsert_bars_1m(db, "NVDA", frame, "rest") == 5

    assert repo_market.read_bars_1m(db, "NVDA")["volume"].tolist() == [0.0, 0.0, 0.0, 0.0, 7.5]


def test_upsert_bars_1m_widens_high_and_low_to_contain_open_and_close(db: Database) -> None:
    frame = minute_frame(
        (T0, 100.0, 99.0, 98.0, 101.0, 1.0),  # high below open and close
        (T0 + 60, 100.0, 105.0, 102.0, 101.0, 1.0),  # low above open and close
        (T0 + 120, 103.0, 100.0, 104.0, 99.0, 1.0),  # both wrong (falling bar)
        (T0 + 180, 100.0, 105.0, 95.0, 101.0, 1.0),  # consistent: stays as it is
    )

    repo_market.upsert_bars_1m(db, "NVDA", frame, "rest")

    expected = minute_frame(
        (T0, 100.0, 101.0, 98.0, 101.0, 1.0),
        (T0 + 60, 100.0, 105.0, 100.0, 101.0, 1.0),
        (T0 + 120, 103.0, 103.0, 99.0, 99.0, 1.0),
        (T0 + 180, 100.0, 105.0, 95.0, 101.0, 1.0),
    )
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), expected)


def test_upsert_bars_1m_floors_timestamps_to_full_minutes(db: Database) -> None:
    frame = minute_frame(flat_bar(T0 + 1, 100.0), flat_bar(T0 + 59, 101.0), flat_bar(T0 + 125, 102.0))

    assert repo_market.upsert_bars_1m(db, "NVDA", frame, "stream") == 3  # two writes hit the same minute

    assert [(row[0], row[1]) for row in stored_bars(db, "NVDA")] == [(T0, 101.0), (T0 + 120, 102.0)]


def test_upsert_bars_1m_accepts_an_empty_frame(db: Database) -> None:
    assert repo_market.upsert_bars_1m(db, "NVDA", repo_market.empty_bars_1m(), "rest") == 0
    assert repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0, NAN)), "rest") == 0
    assert stored_bars(db, "NVDA") == []


def test_upsert_bars_1m_leaves_the_callers_frame_untouched(db: Database) -> None:
    frame = minute_frame((T0 + 7, 100.0, 99.0, 101.0, 100.5, NAN), (T0 + 60, NAN, 1.0, 1.0, 1.0, 1.0))
    before = frame.copy(deep=True)

    repo_market.upsert_bars_1m(db, "NVDA", frame, "rest")

    pd.testing.assert_frame_equal(frame, before)


def test_upsert_bars_1m_rejects_frames_that_are_not_canonical(db: Database) -> None:
    good = minute_frame(flat_bar(T0, 100.0))
    datetime_index = good.set_axis(pd.to_datetime(good.index, unit="s", utc=True))
    float_index = good.set_axis(good.index.astype("float64"))

    with pytest.raises(ValueError, match="integer epoch-second index"):
        repo_market.upsert_bars_1m(db, "NVDA", datetime_index, "rest")
    with pytest.raises(ValueError, match="integer epoch-second index"):
        repo_market.upsert_bars_1m(db, "NVDA", float_index, "rest")
    with pytest.raises(ValueError, match="lacks the columns"):
        repo_market.upsert_bars_1m(db, "NVDA", good.drop(columns=["volume", "low"]), "rest")
    with pytest.raises(ValueError, match="unknown bar source"):
        repo_market.upsert_bars_1m(db, "NVDA", good, "yahoo")
    assert stored_bars(db, "NVDA") == []


def test_upsert_bars_1m_ignores_extra_columns_and_column_order(db: Database) -> None:
    frame = minute_frame((T0, 100.0, 101.0, 99.0, 100.5, 10.0))
    shuffled = frame[["volume", "close", "low", "high", "open"]].assign(dividends=0.0)

    repo_market.upsert_bars_1m(db, "NVDA", shuffled, "rest")

    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), frame)


def test_upsert_bar_rows_stores_mixed_symbols_and_sources(db: Database) -> None:
    repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0, 100.0)), "rest")
    bars = [
        Bar("NVDA", T0, 1.0, 1.0, 1.0, 1.0, 1.0, "stream"),  # loses against the stored REST bar
        Bar("NVDA", T0 + 60, 101.0, 102.0, 100.0, 101.5, 5.0, "stream"),
        Bar("BTC-USD", T0, 90_000.0, 90_100.0, 89_900.0, 90_050.0, 0.25, "quote"),
        Bar("BTC-USD", T0, 90_001.0, 90_101.0, 89_901.0, 90_051.0, 0.5, "stream"),  # replaces the quote bar
    ]

    assert repo_market.upsert_bar_rows(db, iter(bars)) == 3

    assert stored_bars(db, "NVDA") == [
        (T0, 100.0, 100.0, 100.0, 100.0, 10.0, "rest"),
        (T0 + 60, 101.0, 102.0, 100.0, 101.5, 5.0, "stream"),
    ]
    assert stored_bars(db, "BTC-USD") == [(T0, 90_001.0, 90_101.0, 89_901.0, 90_051.0, 0.5, "stream")]


def test_upsert_bar_rows_applies_the_cleaning_rules(db: Database) -> None:
    bars = [
        Bar("NVDA", T0 + 31, 100.0, 99.0, 101.0, 100.5, NAN, "stream"),  # floored, repaired, volume 0
        Bar("NVDA", T0 + 60, NAN, 101.0, 99.0, 100.5, 1.0, "stream"),
        Bar("NVDA", T0 + 120, 100.0, INF, 99.0, 100.5, 1.0, "stream"),
        Bar("NVDA", T0 + 180, 100.0, 101.0, 0.0, 100.5, 1.0, "stream"),
        Bar("NVDA", T0 + 240, 100.0, 101.0, 99.0, -100.5, 1.0, "stream"),
    ]

    assert repo_market.upsert_bar_rows(db, bars) == 1

    assert stored_bars(db, "NVDA") == [(T0, 100.0, 100.5, 100.0, 100.5, 0.0, "stream")]
    assert repo_market.upsert_bar_rows(db, []) == 0


def test_upsert_bar_rows_rejects_an_unknown_source_before_writing(db: Database) -> None:
    bars = [
        Bar("NVDA", T0, 1.0, 1.0, 1.0, 1.0),
        Bar("NVDA", T0 + 60, 1.0, 1.0, 1.0, 1.0, source="yahoo"),  # type: ignore[arg-type]
    ]

    with pytest.raises(ValueError, match="unknown bar source"):
        repo_market.upsert_bar_rows(db, bars)

    assert stored_bars(db, "NVDA") == []


def test_concurrent_upserts_end_with_the_highest_ranked_source(db: Database) -> None:
    minutes = 120
    prices = {"quote": 1.0, "stream": 2.0, "rest": 3.0}
    start = threading.Barrier(len(prices) * 2)
    errors: list[BaseException] = []

    def write(source: str, chunk: int) -> None:
        frame = minute_frame(*(flat_bar(T0 + 60 * n, prices[source]) for n in range(minutes)))
        try:
            start.wait()
            for offset in range(0, minutes, chunk):
                repo_market.upsert_bars_1m(db, "NVDA", frame.iloc[offset : offset + chunk], source)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(source, chunk)) for source in prices for chunk in (7, 20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert [(row[1], row[6]) for row in stored_bars(db, "NVDA")] == [(3.0, "rest")] * minutes


def test_readers_see_whole_batches_while_another_thread_writes(db: Database) -> None:
    batches, batch_size = 30, 25
    frame = make_bars(T0, batches * batch_size)
    start = threading.Barrier(4)
    read_pass = threading.Event()
    done = threading.Event()
    errors: list[BaseException] = []

    def write() -> None:
        try:
            start.wait()
            for offset in range(0, len(frame), batch_size):
                repo_market.upsert_bars_1m(db, "NVDA", frame.iloc[offset : offset + batch_size], "stream")
                read_pass.clear()
                assert read_pass.wait(10), "no reader finished a pass after this batch"
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    def read() -> None:
        try:
            start.wait()
            while not done.is_set():
                stored = repo_market.read_bars_1m(db, "NVDA")
                assert len(stored) % batch_size == 0  # a batch is one transaction: visible completely or not at all
                pd.testing.assert_frame_equal(stored, frame.iloc[: len(stored)], check_exact=True)
                newest = repo_market.last_bar(db, "NVDA")
                assert newest is None or (newest.ts - T0) // 60 % batch_size == batch_size - 1
                read_pass.set()
                time.sleep(0.0005)  # readers spinning without a pause would starve the writer of the GIL
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write), *(threading.Thread(target=read) for _ in range(3))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), frame, check_exact=True)


# ------------------------------------------------------------------------------------------------
# 1-minute bars: reading
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def ten_bars(db: Database) -> pd.DataFrame:
    """Ten consecutive NVDA bars from T0 plus bars of another symbol that must never show up."""
    frame = make_bars(T0, 10)
    repo_market.upsert_bars_1m(db, "NVDA", frame, "rest")
    repo_market.upsert_bars_1m(db, "SAP.DE", make_bars(T0 - 600, 40, seed=2), "rest")
    return frame


def test_read_bars_1m_returns_the_stored_frame(db: Database, ten_bars: pd.DataFrame) -> None:
    frame = repo_market.read_bars_1m(db, "NVDA")

    assert_canonical_minutes(frame)
    pd.testing.assert_frame_equal(frame, ten_bars, check_exact=True)


def test_read_bars_1m_start_is_inclusive_and_end_exclusive(db: Database, ten_bars: pd.DataFrame) -> None:
    def read(**bounds: int) -> list[int]:
        return [(ts - T0) // 60 for ts in repo_market.read_bars_1m(db, "NVDA", **bounds).index]

    assert read(start_ts=T0 + 180) == [3, 4, 5, 6, 7, 8, 9]
    assert read(start_ts=T0 + 181) == [4, 5, 6, 7, 8, 9]
    assert read(end_ts=T0 + 180) == [0, 1, 2]
    assert read(end_ts=T0 + 181) == [0, 1, 2, 3]
    assert read(start_ts=T0 + 120, end_ts=T0 + 300) == [2, 3, 4]
    assert read(start_ts=T0 + 300, end_ts=T0 + 300) == []
    assert read(start_ts=T0 + 6000) == []


def test_read_bars_1m_limit_keeps_the_newest_rows_in_ascending_order(db: Database, ten_bars: pd.DataFrame) -> None:
    newest = repo_market.read_bars_1m(db, "NVDA", limit=3)
    assert_canonical_minutes(newest)
    pd.testing.assert_frame_equal(newest, ten_bars.tail(3), check_exact=True)

    bounded = repo_market.read_bars_1m(db, "NVDA", start_ts=T0 + 60, end_ts=T0 + 360, limit=2)
    pd.testing.assert_frame_equal(bounded, ten_bars.iloc[4:6], check_exact=True)

    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA", limit=100), ten_bars, check_exact=True)
    assert repo_market.read_bars_1m(db, "NVDA", limit=0).empty
    with pytest.raises(ValueError, match="limit"):
        repo_market.read_bars_1m(db, "NVDA", limit=-1)


def test_read_bars_1m_empty_result_is_a_canonical_frame(db: Database, ten_bars: pd.DataFrame) -> None:
    for frame in (
        repo_market.read_bars_1m(db, "UNKNOWN"),
        repo_market.read_bars_1m(db, "NVDA", start_ts=T0 + 6000),
        repo_market.bars_after(db, "NVDA", T0 + 6000),
        repo_market.empty_bars_1m(),
    ):
        assert frame.empty
        assert_canonical_minutes(frame)
        pd.testing.assert_frame_equal(frame, ten_bars.iloc[:0])


def test_read_bars_1m_with_sources_adds_the_source_of_every_row(db: Database, ten_bars: pd.DataFrame) -> None:
    repo_market.upsert_bar_rows(
        db,
        [
            Bar("NVDA", T0 + 600, 101.0, 101.5, 100.5, 101.2, 5.0, "stream"),
            Bar("NVDA", T0 + 660, 101.2, 101.4, 101.0, 101.3, 0.0, "quote"),
            Bar("NVDA", T0 + 120, 1.0, 1.0, 1.0, 1.0, 0.0, "stream"),  # lower rank than the stored REST bar
        ],
    )

    frame, sources = repo_market.read_bars_1m_with_sources(db, "NVDA")

    assert_canonical_minutes(frame)
    pd.testing.assert_frame_equal(frame, repo_market.read_bars_1m(db, "NVDA"), check_exact=True)
    assert isinstance(sources, np.ndarray) and sources.dtype.kind == "U"
    assert sources.tolist() == ["rest"] * 10 + ["stream", "quote"]
    assert sources.tolist() == [row[6] for row in stored_bars(db, "NVDA")]

    repo_market.upsert_bars_1m(db, "NVDA", minute_frame(flat_bar(T0 + 600, 102.0)), "rest")  # REST replaces it
    frame, sources = repo_market.read_bars_1m_with_sources(db, "NVDA", start_ts=T0 + 540)
    assert frame.index.tolist() == [T0 + 540, T0 + 600, T0 + 660]
    assert sources.tolist() == ["rest", "rest", "quote"]
    assert frame.loc[T0 + 600, "open"] == 102.0


def test_read_bars_1m_with_sources_takes_the_bounds_and_limit_of_read_bars_1m(
    db: Database, ten_bars: pd.DataFrame
) -> None:
    repo_market.upsert_bar_rows(db, [Bar("NVDA", T0 + 600, 101.0, 101.0, 101.0, 101.0, 0.0, "stream")])
    cases: list[dict[str, Any]] = [
        {},
        {"start_ts": T0 + 180},
        {"end_ts": T0 + 181},
        {"start_ts": T0 + 60, "end_ts": T0 + 360, "limit": 2},
        {"limit": 3},
        {"limit": 0},
        {"start_ts": T0 + 6000},
        {"start_ts": np.int64(T0 + 120), "end_ts": np.int64(T0 + 300)},
    ]
    for bounds in cases:
        frame, sources = repo_market.read_bars_1m_with_sources(db, "NVDA", **bounds)
        pd.testing.assert_frame_equal(frame, repo_market.read_bars_1m(db, "NVDA", **bounds), check_exact=True)
        assert len(sources) == len(frame), bounds
        assert sources.tolist() == ["stream" if ts == T0 + 600 else "rest" for ts in frame.index], bounds

    empty, no_sources = repo_market.read_bars_1m_with_sources(db, "UNKNOWN")
    assert_canonical_minutes(empty)
    pd.testing.assert_frame_equal(empty, ten_bars.iloc[:0])
    assert no_sources.shape == (0,)
    with pytest.raises(ValueError, match="limit"):
        repo_market.read_bars_1m_with_sources(db, "NVDA", limit=-1)


def test_bar_queries_accept_numpy_integers(db: Database, ten_bars: pd.DataFrame) -> None:
    """sqlite3 would bind a numpy integer as BLOB, which compares greater than every number."""
    third, sixth = ten_bars.index[2], ten_bars.index[5]
    assert isinstance(third, np.integer)

    frame = repo_market.read_bars_1m(db, "NVDA", start_ts=third, end_ts=sixth, limit=np.int64(2))
    assert frame.index.tolist() == [T0 + 180, T0 + 240]
    assert repo_market.bars_after(db, "NVDA", sixth, limit=np.int64(1)).index.tolist() == [T0 + 360]
    first = repo_market.first_bar_at_or_after(db, "NVDA", third)
    assert first is not None and first.ts == T0 + 120
    last = repo_market.last_bar(db, "NVDA", before_ts=third)
    assert last is not None and last.ts == T0 + 60
    assert repo_market.delete_bars_before(db, np.int64(T0 + 60)) == 12  # 1 NVDA bar and 11 older SAP.DE bars


def test_bars_after_is_strict_and_limit_keeps_the_oldest_rows(db: Database, ten_bars: pd.DataFrame) -> None:
    after = repo_market.bars_after(db, "NVDA", T0 + 420)
    assert_canonical_minutes(after)
    pd.testing.assert_frame_equal(after, ten_bars.iloc[8:], check_exact=True)

    assert repo_market.bars_after(db, "NVDA", T0 + 419).index.tolist() == [T0 + 420, T0 + 480, T0 + 540]
    assert repo_market.bars_after(db, "NVDA", T0 + 540).empty
    pd.testing.assert_frame_equal(repo_market.bars_after(db, "NVDA", T0 - 1), ten_bars, check_exact=True)

    chunk = repo_market.bars_after(db, "NVDA", T0 + 60, limit=3)
    pd.testing.assert_frame_equal(chunk, ten_bars.iloc[2:5], check_exact=True)
    with pytest.raises(ValueError, match="limit"):
        repo_market.bars_after(db, "NVDA", T0, limit=-3)


def test_bars_after_in_chunks_visits_every_bar_exactly_once(db: Database, ten_bars: pd.DataFrame) -> None:
    seen: list[int] = []
    cursor = T0 - 1
    while not (chunk := repo_market.bars_after(db, "NVDA", cursor, limit=4)).empty:
        seen.extend(chunk.index.tolist())
        cursor = seen[-1]

    assert seen == ten_bars.index.tolist()


def test_first_bar_at_or_after(db: Database, ten_bars: pd.DataFrame) -> None:
    row = ten_bars.loc[T0 + 120]
    expected = Bar("NVDA", T0 + 120, row["open"], row["high"], row["low"], row["close"], row["volume"], "rest")

    assert repo_market.first_bar_at_or_after(db, "NVDA", T0 + 120) == expected
    assert repo_market.first_bar_at_or_after(db, "NVDA", T0 + 61) == expected
    first = repo_market.first_bar_at_or_after(db, "NVDA", 0)
    assert first is not None and first.ts == T0
    assert repo_market.first_bar_at_or_after(db, "NVDA", T0 + 541) is None
    assert repo_market.first_bar_at_or_after(db, "UNKNOWN", 0) is None


def test_last_bar_optionally_before_a_time(db: Database, ten_bars: pd.DataFrame) -> None:
    newest = repo_market.last_bar(db, "NVDA")
    assert newest is not None and newest.ts == T0 + 540
    assert isinstance(newest.ts, int) and isinstance(newest.close, float) and newest.source == "rest"
    assert newest.close == ten_bars["close"].iloc[-1]

    before = repo_market.last_bar(db, "NVDA", before_ts=T0 + 540)
    assert before is not None and before.ts == T0 + 480
    inside = repo_market.last_bar(db, "NVDA", before_ts=T0 + 541)
    assert inside is not None and inside.ts == T0 + 540
    assert repo_market.last_bar(db, "NVDA", before_ts=T0) is None
    assert repo_market.last_bar(db, "UNKNOWN") is None


def test_bar_range_stats_and_symbols(db: Database, ten_bars: pd.DataFrame) -> None:
    assert repo_market.bar_range(db, "NVDA") == (T0, T0 + 540)
    assert repo_market.bar_range(db, "SAP.DE") == (T0 - 600, T0 + 1740)
    assert repo_market.bar_range(db, "UNKNOWN") is None
    assert repo_market.bar_stats(db) == {"rows": 50, "first_ts": T0 - 600, "last_ts": T0 + 1740}
    assert repo_market.symbols_with_bars(db) == {"NVDA": T0 + 540, "SAP.DE": T0 + 1740}


def test_bar_statistics_of_an_empty_table(db: Database) -> None:
    assert repo_market.bar_stats(db) == {"rows": 0, "first_ts": None, "last_ts": None}
    assert repo_market.symbols_with_bars(db) == {}


def test_symbols_with_bars_lists_every_symbol_once(db: Database) -> None:
    expected: dict[str, int] = {}
    for n, instrument in enumerate(SAMPLE_INSTRUMENTS):
        repo_market.upsert_bars_1m(db, instrument.symbol, make_bars(T0, 3 + n, seed=n), "rest")
        expected[instrument.symbol] = T0 + 60 * (2 + n)

    assert repo_market.symbols_with_bars(db) == expected


def test_delete_bars_before_removes_older_bars_of_all_symbols(db: Database, ten_bars: pd.DataFrame) -> None:
    assert repo_market.delete_bars_before(db, T0 + 300) == 5 + 15

    assert repo_market.bar_range(db, "NVDA") == (T0 + 300, T0 + 540)
    assert repo_market.bar_range(db, "SAP.DE") == (T0 + 300, T0 + 1740)
    assert repo_market.delete_bars_before(db, T0 + 300) == 0


def test_ten_trading_days_of_minutes_round_trip_exactly(db: Database) -> None:
    sessions = [make_bars(T0 + day * 86_400, 390, seed=day) for day in range(10)]
    frame = pd.concat(sessions)

    assert repo_market.upsert_bars_1m(db, "NVDA", frame, "rest") == 3900

    stored = repo_market.read_bars_1m(db, "NVDA")
    assert_canonical_minutes(stored)
    pd.testing.assert_frame_equal(stored, frame, check_exact=True)
    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA", limit=390), sessions[-1], check_exact=True)


# ------------------------------------------------------------------------------------------------
# Daily bars
# ------------------------------------------------------------------------------------------------
def test_daily_bars_round_trip(db: Database) -> None:
    frame = daily_frame(
        ("2026-11-18", 100.0, 102.0, 99.0, 101.0, 1000.0),
        ("2026-11-19", 101.0, 103.0, 100.0, 102.0, 1100.0),
        ("2026-11-20", 102.0, 104.0, 101.0, 103.0, 1200.0),
    )

    assert repo_market.upsert_bars_1d(db, "NVDA", frame) == 3
    repo_market.upsert_bars_1d(db, "SAP.DE", daily_frame(("2026-11-23", 1.0, 1.0, 1.0, 1.0, 1.0)))

    stored = repo_market.read_bars_1d(db, "NVDA")
    assert_canonical_daily(stored)
    pd.testing.assert_frame_equal(stored, frame, check_exact=True)
    assert repo_market.last_day(db, "NVDA") == "2026-11-20"
    assert repo_market.last_day(db, "UNKNOWN") is None


def test_daily_bars_are_always_replaced(db: Database) -> None:
    repo_market.upsert_bars_1d(db, "NVDA", daily_frame(("2026-11-20", 102.0, 104.0, 101.0, 103.0, 1200.0)))
    update = daily_frame(
        ("2026-11-20", 102.0, 108.0, 100.0, 107.0, 2400.0),
        ("2026-11-23", 107.0, 109.0, 106.0, 108.0, 5.0),
    )

    assert repo_market.upsert_bars_1d(db, "NVDA", update) == 2

    pd.testing.assert_frame_equal(repo_market.read_bars_1d(db, "NVDA"), update, check_exact=True)
    assert db.scalar("SELECT COUNT(*) FROM bars_1d") == 2


def test_read_bars_1d_end_day_is_inclusive_and_limit_keeps_the_newest_days(db: Database) -> None:
    days = [f"2026-11-{day:02d}" for day in range(9, 21)]
    frame = daily_frame(*((day, 100.0 + n, 101.0 + n, 99.0 + n, 100.5 + n, 10.0 * n) for n, day in enumerate(days)))
    repo_market.upsert_bars_1d(db, "NVDA", frame)

    assert repo_market.read_bars_1d(db, "NVDA", end_day="2026-11-17").index.tolist() == days[:9]
    newest_two = repo_market.read_bars_1d(db, "NVDA", end_day="2026-11-17", limit=2)
    assert newest_two.index.tolist() == ["2026-11-16", "2026-11-17"]
    assert repo_market.read_bars_1d(db, "NVDA", end_day="2026-11-08").empty
    limited = repo_market.read_bars_1d(db, "NVDA", limit=3)
    assert_canonical_daily(limited)
    pd.testing.assert_frame_equal(limited, frame.tail(3), check_exact=True)
    assert repo_market.read_bars_1d(db, "NVDA", limit=0).empty
    with pytest.raises(ValueError, match="limit"):
        repo_market.read_bars_1d(db, "NVDA", limit=-1)


def test_read_bars_1d_empty_result_is_a_canonical_frame(db: Database) -> None:
    reference = daily_frame(("2026-11-20", 1.0, 1.0, 1.0, 1.0, 1.0)).iloc[:0]

    for frame in (repo_market.read_bars_1d(db, "UNKNOWN"), repo_market.empty_bars_1d()):
        assert frame.empty
        assert_canonical_daily(frame)
        pd.testing.assert_frame_equal(frame, reference)


def test_upsert_bars_1d_applies_the_cleaning_rules(db: Database) -> None:
    frame = daily_frame(
        ("2026-11-17", NAN, 102.0, 99.0, 101.0, 1000.0),
        ("2026-11-18", 100.0, 102.0, 0.0, 101.0, 1000.0),
        ("2026-11-19", 100.0, 99.0, 101.0, 102.0, NAN),
        ("2026-11-20", 100.0, 104.0, 98.0, 103.0, 1200.0),
    )

    assert repo_market.upsert_bars_1d(db, "NVDA", frame) == 2

    expected = daily_frame(
        ("2026-11-19", 100.0, 102.0, 100.0, 102.0, 0.0),
        ("2026-11-20", 100.0, 104.0, 98.0, 103.0, 1200.0),
    )
    pd.testing.assert_frame_equal(repo_market.read_bars_1d(db, "NVDA"), expected, check_exact=True)
    assert repo_market.upsert_bars_1d(db, "NVDA", repo_market.empty_bars_1d()) == 0


@pytest.mark.parametrize(
    "bad_day",
    ["20261120", "2026-11-20 00:00:00", "2026-13-01", "2026-W47-5", "20.11.2026", "", date(2026, 11, 20), 20261120],
)
def test_upsert_bars_1d_rejects_day_keys_that_are_not_iso_dates(db: Database, bad_day: object) -> None:
    frame = pd.DataFrame(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]],
        columns=list(BAR_COLUMNS),
        index=pd.Index(["2026-11-19", bad_day], dtype="object", name="day"),
    )

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        repo_market.upsert_bars_1d(db, "NVDA", frame)

    assert db.scalar("SELECT COUNT(*) FROM bars_1d") == 0


def test_upsert_bars_1d_rejects_a_datetime_index(db: Database) -> None:
    frame = daily_frame(("2026-11-20", 1.0, 1.0, 1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        repo_market.upsert_bars_1d(db, "NVDA", frame.set_axis(pd.to_datetime(frame.index)))


# ------------------------------------------------------------------------------------------------
# Quotes
# ------------------------------------------------------------------------------------------------
def test_quotes_round_trip_including_missing_fields(db: Database) -> None:
    full = Quote(
        symbol="NVDA",
        ts=T0,
        received_ts=T0 + 2,
        price=142.5,
        bid=142.4,
        ask=142.6,
        prev_close=140.0,
        day_open=141.0,
        day_high=143.0,
        day_low=140.5,
        day_volume=1.5e6,
        change_pct=1.79,
        market_state="REGULAR",
        source="stream",
    )
    sparse = Quote("BTC-USD", ts=T0 - 30, received_ts=T0, price=90_000.0)

    assert repo_market.upsert_quotes(db, [full, sparse]) == 2

    quotes = repo_market.load_quotes(db)
    assert quotes == {"NVDA": full, "BTC-USD": sparse}
    loaded = quotes["BTC-USD"]
    assert loaded.bid is None and loaded.day_volume is None and loaded.market_state is None
    assert loaded.source == "quote"
    assert isinstance(loaded.ts, int) and isinstance(loaded.price, float)


def test_upsert_quotes_replaces_the_stored_quote(db: Database) -> None:
    repo_market.upsert_quotes(db, [Quote("NVDA", T0, T0, 142.5, bid=142.4, market_state="REGULAR", source="stream")])
    newer = Quote("NVDA", T0 + 60, T0 + 61, 143.0)

    assert repo_market.upsert_quotes(db, iter([newer])) == 1

    assert repo_market.load_quotes(db) == {"NVDA": newer}  # fields missing in the new quote are cleared


def test_quotes_of_an_empty_book(db: Database) -> None:
    assert repo_market.upsert_quotes(db, []) == 0
    assert repo_market.load_quotes(db) == {}
