import dataclasses
import logging
import sqlite3
from typing import Any

import pytest

from trading_local.db import repo_signals
from trading_local.db.database import Database
from trading_local.models import ChartAnalysis, Signal

T0 = 1_800_000_000


@pytest.fixture(autouse=True)
def instruments(db: Database) -> None:
    """signals.symbol references instruments."""
    db.executemany(
        "INSERT INTO instruments (symbol, name, asset_class, region, exchange, currency, timezone) "
        "VALUES (?, ?, 'equity', ?, ?, ?, ?)",
        [
            ("NVDA", "NVIDIA", "US", "XNAS", "USD", "America/New_York"),
            ("SAP.DE", "SAP", "EU", "XETR", "EUR", "Europe/Berlin"),
        ],
    )


def make_signal(**fields: Any) -> Signal:
    values: dict[str, Any] = {
        "created_ts": T0,
        "data_ts": T0,
        "symbol": "NVDA",
        "direction": "LONG",
        "setup": "breakout",
        "timeframe": "15m",
        "horizon_min": 240,
        "ref_price": 100.0,
        "atr": 1.5,
        "stop_dist": 2.0,
        "target_dist": 3.0,
        "score": 0.66,
        "confidence": 0.58,
        "engine_version": "v1",
    }
    values.update(fields)
    return Signal(**values)


def full_signal() -> Signal:
    """A closed signal in which every optional field differs from its default."""
    return make_signal(
        symbol="SAP.DE",
        direction="SHORT",
        status="target",
        intraday=False,
        components={"trend": 0.8, "volume": 0.4},
        features={"rsi": 61.5, "ema": [101.25, 99.5], "muster": "Rücksetzer", "gap": None, "nested": {"n": 3}},
        expected_r=0.21,
        issued_ts=T0 + 30,
        llm_verdict="reduce",
        llm_conviction=0.55,
        llm_latency_ms=8123,
        rationale_de="Abwärtstrend intakt, Rücksetzer an die EMA20",
        risks_de="Quartalszahlen am Abend",
        news_ids=[3, 7],
        actionable=True,
        shadow_reason="llm_reduce",
        status_reason="Ziel erreicht",
        entry_deadline_ts=T0 + 900,
        checked_ts=T0 + 3540,
        entry_ts=T0 + 60,
        entry_price=99.9,
        stop_price=101.9,
        target_price=96.9,
        expires_ts=T0 + 14_460,
        exit_ts=T0 + 3600,
        exit_price=96.85,
        outcome_r=1.47,
        mfe_r=1.6,
        mae_r=-0.35,
        mailed_ts=T0 + 31,
    )


def full_chart_analysis() -> ChartAnalysis:
    return ChartAnalysis(
        symbol="NVDA",
        ts=T0,
        timeframe="15m",
        model="vision-model",
        bias="bearish",
        strength=4,
        comment_de="Doppeltop unter dem Widerstand",
        pattern="double_top",
        support=97.5,
        resistance=103.25,
        latency_ms=15_400,
    )


def assert_no_default_left(obj: Any) -> None:
    """Guard for the round-trip tests: a field added to the model must be added to the fixture as well."""
    for field in dataclasses.fields(obj):
        if field.name == "id":
            continue
        value = getattr(obj, field.name)
        if field.default is not dataclasses.MISSING:
            assert value != field.default, field.name
        elif field.default_factory is not dataclasses.MISSING:
            assert value != field.default_factory(), field.name


def ids(signals: list[Signal]) -> list[int | None]:
    return [signal.id for signal in signals]


# ------------------------------------------------------------------------------------------------
# Insert, update, get
# ------------------------------------------------------------------------------------------------
def test_insert_sets_the_id_and_round_trips_a_minimal_signal(db: Database) -> None:
    signal = make_signal()

    stored = repo_signals.insert_signal(db, signal)

    assert stored is signal and signal.id == 1
    loaded = repo_signals.get_signal(db, 1)
    assert loaded == signal
    assert loaded is not None and loaded.intraday is True and loaded.actionable is False
    assert loaded.components == {} and loaded.features == {} and loaded.news_ids == []
    assert repo_signals.get_signal(db, 2) is None


def test_round_trip_keeps_every_optional_field_and_json_column(db: Database) -> None:
    signal = full_signal()
    assert_no_default_left(signal)

    repo_signals.insert_signal(db, signal)

    loaded = repo_signals.get_signal(db, 1)
    assert loaded == signal
    assert loaded is not None and loaded.intraday is False and loaded.actionable is True
    assert loaded.features["ema"] == [101.25, 99.5] and loaded.features["gap"] is None


def test_insert_rejects_a_signal_that_is_stored_already(db: Database) -> None:
    signal = repo_signals.insert_signal(db, make_signal())

    with pytest.raises(ValueError, match="already stored"):
        repo_signals.insert_signal(db, signal)

    assert db.scalar("SELECT COUNT(*) FROM signals") == 1


def test_insert_for_an_unknown_instrument_fails(db: Database) -> None:
    signal = make_signal(symbol="TSLA")

    with pytest.raises(sqlite3.IntegrityError):
        repo_signals.insert_signal(db, signal)

    assert signal.id is None


def test_update_signal_writes_the_full_row(db: Database) -> None:
    repo_signals.insert_signal(db, make_signal())
    untouched = repo_signals.insert_signal(db, make_signal())
    changed = dataclasses.replace(full_signal(), id=1)

    repo_signals.update_signal(db, changed)

    assert repo_signals.get_signal(db, 1) == changed
    assert repo_signals.get_signal(db, 2) == untouched


def test_update_signal_needs_an_id(db: Database) -> None:
    with pytest.raises(ValueError, match="without an id"):
        repo_signals.update_signal(db, make_signal())


def test_update_of_a_missing_signal_is_logged(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="trading_local.db.repo_signals"):
        repo_signals.update_signal(db, make_signal(id=77))

    assert "77" in caplog.text
    assert db.scalar("SELECT COUNT(*) FROM signals") == 0


def test_mark_mailed(db: Database) -> None:
    repo_signals.insert_signal(db, make_signal())
    repo_signals.insert_signal(db, make_signal())

    repo_signals.mark_mailed(db, 2, T0 + 45)

    assert [row["mailed_ts"] for row in db.query("SELECT mailed_ts FROM signals ORDER BY id")] == [None, T0 + 45]


# ------------------------------------------------------------------------------------------------
# Queries
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def journal(db: Database) -> None:
    """Five signals, created one minute apart (ids 1..5)."""
    rows: list[dict[str, Any]] = [
        {"symbol": "NVDA", "status": "review", "actionable": False},
        {"symbol": "SAP.DE", "status": "active", "actionable": True},
        {"symbol": "NVDA", "status": "target", "actionable": True},
        {"symbol": "NVDA", "status": "cancelled", "actionable": False},
        {"symbol": "SAP.DE", "status": "pending_entry", "actionable": True},
    ]
    for minute, fields in enumerate(rows):
        repo_signals.insert_signal(db, make_signal(created_ts=T0 + 60 * minute, **fields))


@pytest.mark.usefixtures("journal")
def test_list_signals_is_newest_first_and_paginates(db: Database) -> None:
    assert ids(repo_signals.list_signals(db)) == [5, 4, 3, 2, 1]
    assert ids(repo_signals.list_signals(db, limit=2)) == [5, 4]
    assert ids(repo_signals.list_signals(db, limit=2, before_id=4)) == [3, 2]
    assert ids(repo_signals.list_signals(db, before_id=1)) == []


@pytest.mark.usefixtures("journal")
def test_list_signals_filters(db: Database) -> None:
    assert ids(repo_signals.list_signals(db, statuses=["active", "pending_entry"])) == [5, 2]
    assert ids(repo_signals.list_signals(db, statuses=("target",))) == [3]
    assert ids(repo_signals.list_signals(db, statuses=[])) == []
    assert ids(repo_signals.list_signals(db, actionable=True)) == [5, 3, 2]
    assert ids(repo_signals.list_signals(db, actionable=False)) == [4, 1]
    assert ids(repo_signals.list_signals(db, symbol="NVDA")) == [4, 3, 1]
    assert ids(repo_signals.list_signals(db, since_ts=T0 + 120)) == [5, 4, 3]
    assert ids(repo_signals.list_signals(db, symbol="NVDA", actionable=True, statuses=["target", "stopped"])) == [3]
    assert ids(repo_signals.list_signals(db, symbol="SAP.DE", since_ts=T0 + 61, before_id=5)) == []


@pytest.mark.usefixtures("journal")
def test_open_signals_are_oldest_first(db: Database) -> None:
    # stored last but created first: the creation time decides, not the id
    repo_signals.insert_signal(db, make_signal(created_ts=T0 - 60, status="active"))

    assert ids(repo_signals.open_signals(db)) == [6, 1, 2, 5]


@pytest.mark.usefixtures("journal")
def test_last_signal_for_symbol(db: Database) -> None:
    repo_signals.insert_signal(db, make_signal(created_ts=T0 - 60, actionable=True))  # id 6, but the oldest

    newest = repo_signals.last_signal_for_symbol(db, "NVDA")
    newest_actionable = repo_signals.last_signal_for_symbol(db, "NVDA", actionable_only=True)

    assert newest is not None and newest.id == 4
    assert newest_actionable is not None and newest_actionable.id == 3
    assert repo_signals.last_signal_for_symbol(db, "TSLA") is None


def test_closed_outcomes_are_realised_results_in_exit_order(db: Database) -> None:
    rows: list[dict[str, Any]] = [
        {"status": "target", "outcome_r": 1.5, "exit_ts": T0 + 300, "actionable": True},
        {"status": "stopped", "outcome_r": -1.0, "exit_ts": T0 + 100},
        {"status": "expired", "outcome_r": 0.2, "exit_ts": T0 + 200, "engine_version": "v2"},
        {"status": "cancelled"},
        {"status": "active", "entry_ts": T0 + 10},
        {"status": "target", "exit_ts": T0 + 50},  # no result recorded
        {"status": "cancelled", "outcome_r": 0.4, "exit_ts": T0 + 60},  # cancelled signals never count
        {"status": "stopped", "outcome_r": -1.0},  # no exit time: cannot be placed in time
    ]
    for fields in rows:
        repo_signals.insert_signal(db, make_signal(**fields))

    assert ids(repo_signals.closed_outcomes(db)) == [2, 3, 1]  # actionable and shadow signals alike
    assert ids(repo_signals.closed_outcomes(db, engine_version="v1")) == [2, 1]
    assert ids(repo_signals.closed_outcomes(db, engine_version="v3")) == []
    assert ids(repo_signals.closed_outcomes(db, since_ts=T0 + 200)) == [3, 1]
    assert ids(repo_signals.closed_outcomes(db, since_ts=T0 + 100, before_ts=T0 + 300)) == [2, 3]


def test_closed_outcomes_before_ts_hides_what_was_not_known_yet(db: Database) -> None:
    repo_signals.insert_signal(db, make_signal(status="target", outcome_r=1.5, exit_ts=T0 + 300))

    assert ids(repo_signals.closed_outcomes(db, before_ts=T0 + 300)) == []  # an exit at the decision time is too late
    assert ids(repo_signals.closed_outcomes(db, before_ts=T0 + 301)) == [1]


# ------------------------------------------------------------------------------------------------
# Chart analyses
# ------------------------------------------------------------------------------------------------
def test_chart_analysis_round_trip(db: Database) -> None:
    analysis = full_chart_analysis()
    assert_no_default_left(analysis)

    stored = repo_signals.insert_chart_analysis(db, analysis)

    assert stored is analysis and analysis.id == 1
    assert repo_signals.latest_chart_analysis(db, "NVDA") == analysis
    with pytest.raises(ValueError, match="already stored"):
        repo_signals.insert_chart_analysis(db, analysis)


def test_chart_analysis_with_defaults(db: Database) -> None:
    analysis = ChartAnalysis(
        symbol="NVDA", ts=T0, timeframe="1d", model="vision-model", bias="neutral", strength=1, comment_de="Seitwärts"
    )

    repo_signals.insert_chart_analysis(db, analysis)

    assert repo_signals.list_chart_analyses(db, "NVDA") == [analysis]


@pytest.fixture
def chart_history(db: Database) -> None:
    """NVDA: 15m at T0, 1d at T0 + 60, 15m at T0 + 120 (ids 1..3); SAP.DE: 15m at T0 + 30 (id 4)."""
    for symbol, ts, timeframe in [
        ("NVDA", T0, "15m"),
        ("NVDA", T0 + 60, "1d"),
        ("NVDA", T0 + 120, "15m"),
        ("SAP.DE", T0 + 30, "15m"),
    ]:
        analysis = dataclasses.replace(full_chart_analysis(), symbol=symbol, ts=ts, timeframe=timeframe)
        repo_signals.insert_chart_analysis(db, analysis)


@pytest.mark.usefixtures("chart_history")
def test_latest_chart_analysis(db: Database) -> None:
    def latest_id(symbol: str, **filters: Any) -> int | None:
        analysis = repo_signals.latest_chart_analysis(db, symbol, **filters)
        return None if analysis is None else analysis.id

    assert latest_id("NVDA") == 3
    assert latest_id("NVDA", timeframe="1d") == 2
    assert latest_id("NVDA", before_ts=T0 + 120) == 2  # strictly before
    assert latest_id("NVDA", before_ts=T0 + 121) == 3
    assert latest_id("NVDA", timeframe="15m", before_ts=T0 + 120) == 1
    assert latest_id("NVDA", before_ts=T0) is None
    assert latest_id("SAP.DE") == 4
    assert latest_id("TSLA") is None


@pytest.mark.usefixtures("chart_history")
def test_list_chart_analyses_is_newest_first(db: Database) -> None:
    assert [analysis.id for analysis in repo_signals.list_chart_analyses(db, "NVDA")] == [3, 2, 1]
    assert [analysis.id for analysis in repo_signals.list_chart_analyses(db, "NVDA", limit=2)] == [3, 2]
    assert repo_signals.list_chart_analyses(db, "TSLA") == []


@pytest.mark.usefixtures("chart_history")
def test_last_chart_analysis_ts_per_symbol(db: Database) -> None:
    assert repo_signals.last_chart_analysis_ts(db) == {"NVDA": T0 + 120, "SAP.DE": T0 + 30}


def test_last_chart_analysis_ts_without_analyses(db: Database) -> None:
    assert repo_signals.last_chart_analysis_ts(db) == {}


# ------------------------------------------------------------------------------------------------
# Review additions: boundaries, ties, hostile input, JSON columns, query cost
# ------------------------------------------------------------------------------------------------
def query_plan(db: Database, run: Any) -> list[str]:
    """The query plan of the single statement that ``run`` sends to the database."""
    statements: list[str] = []
    db.conn().set_trace_callback(statements.append)
    try:
        run()
    finally:
        db.conn().set_trace_callback(None)
    [statement] = statements
    return [row["detail"] for row in db.query("EXPLAIN QUERY PLAN " + statement)]


def test_list_signals_since_ts_is_exact_for_signals_stored_out_of_order(db: Database) -> None:
    # the ids ascend in storage order, the decision times do not (signal n has id n + 1)
    created = [T0 + 50, T0 + 10, T0 + 40, T0, T0 + 30, T0 + 20]
    for created_ts in created:
        repo_signals.insert_signal(db, make_signal(created_ts=created_ts))

    for since_ts in (T0 - 1, T0, T0 + 1, T0 + 30, T0 + 50, T0 + 51):
        expected = [n + 1 for n in reversed(range(len(created))) if created[n] >= since_ts]  # newest id first
        assert ids(repo_signals.list_signals(db, since_ts=since_ts)) == expected, since_ts
        assert ids(repo_signals.list_signals(db, since_ts=since_ts, limit=2)) == expected[:2]
        assert ids(repo_signals.list_signals(db, since_ts=since_ts, before_id=4)) == [i for i in expected if i < 4]


def test_list_signals_since_ts_reads_only_the_recent_part_of_the_journal(db: Database) -> None:
    plan = query_plan(db, lambda: repo_signals.list_signals(db, since_ts=T0))

    assert "SCAN signals" not in plan  # no walk over the whole journal ...
    assert "SEARCH signals USING INTEGER PRIMARY KEY (rowid>?)" in plan  # ... but one with a lower end,
    assert any("COVERING INDEX signals_created_idx" in step for step in plan)  # which the decision-time index provides


def vm_steps(db: Database, run: Any) -> int:
    """Work that ``run`` causes on this thread's connection, in units of ten SQLite virtual-machine steps."""
    ticks = 0

    def tick() -> int:
        nonlocal ticks
        ticks += 1
        return 0

    db.conn().set_progress_handler(tick, 10)
    try:
        run()
    finally:
        db.conn().set_progress_handler(None, 0)
    return ticks


def test_list_signals_since_ts_costs_what_is_new_and_not_what_is_stored(db: Database) -> None:
    stored = 3000
    db.executemany(
        "INSERT INTO signals (created_ts, data_ts, symbol, direction, setup, timeframe, horizon_min, ref_price, atr, "
        "stop_dist, target_dist, score, confidence, status, engine_version) "
        "VALUES (?, ?, 'NVDA', 'LONG', 'breakout', '15m', 240, 100, 1.5, 2, 3, 0.7, 0.6, 'cancelled', 'v1')",
        ((T0 + n, T0 + n) for n in range(1, stored + 1)),
    )

    whole_journal = vm_steps(db, lambda: repo_signals.list_signals(db, actionable=True))  # no match: reads everything
    nothing_new = vm_steps(db, lambda: repo_signals.list_signals(db, since_ts=T0 + stored + 1))
    five_new = vm_steps(db, lambda: repo_signals.list_signals(db, since_ts=T0 + stored - 4))
    five_new_filtered = vm_steps(db, lambda: repo_signals.list_signals(db, since_ts=T0 + stored - 4, actionable=True))

    assert len(repo_signals.list_signals(db, since_ts=T0 + stored - 4)) == 5
    assert max(nothing_new, five_new, five_new_filtered) * 5 < whole_journal


def test_list_signals_since_ts_is_exact_beyond_the_probed_range(db: Database) -> None:
    # More matches than the lower-end probe looks at: the walk must stay unbounded, down to the oldest match.
    total = repo_signals._SINCE_PROBE + 50
    db.executemany(
        "INSERT INTO signals (created_ts, data_ts, symbol, direction, setup, timeframe, horizon_min, ref_price, atr, "
        "stop_dist, target_dist, score, confidence, status, engine_version) "
        "VALUES (?, ?, 'NVDA', 'LONG', 'breakout', '15m', 240, 100, 1.5, 2, 3, 0.7, 0.6, ?, 'v1')",
        ((T0 + n, T0 + n, "active" if n % 1000 == 0 else "cancelled") for n in range(1, total + 1)),  # id n
    )

    def found(**filters: Any) -> list[int | None]:
        return ids(repo_signals.list_signals(db, **filters))

    assert found(since_ts=T0, limit=3) == [total, total - 1, total - 2]
    assert found(since_ts=T0, before_id=4) == [3, 2, 1]  # the oldest matches are not cut off
    assert found(since_ts=T0 + 2, before_id=4) == [3, 2]
    assert found(since_ts=T0, statuses=["active"]) == list(range(total // 1000 * 1000, 0, -1000))
    # as many matches as the probe covers, and just below and above that
    for matches in (repo_signals._SINCE_PROBE + step for step in (-1, 0, 1, 2)):
        oldest = total - matches + 1
        assert found(since_ts=T0 + oldest, before_id=oldest + 2) == [oldest + 1, oldest], matches
        assert found(since_ts=T0 + oldest, limit=1) == [total]


@pytest.mark.parametrize(
    "run",
    [
        lambda db: repo_signals.open_signals(db),
        lambda db: repo_signals.last_signal_for_symbol(db, "NVDA", actionable_only=True),
        lambda db: repo_signals.closed_outcomes(db, before_ts=T0),
        lambda db: repo_signals.latest_chart_analysis(db, "NVDA", timeframe="15m", before_ts=T0),
        lambda db: repo_signals.list_chart_analyses(db, "NVDA"),
        lambda db: repo_signals.get_signal(db, 1),
    ],
    ids=["open_signals", "last_signal", "closed_outcomes", "latest_chart", "list_charts", "get_signal"],
)
def test_frequent_readers_never_scan_a_whole_table(db: Database, run: Any) -> None:
    assert not [step for step in query_plan(db, lambda: run(db)) if step.startswith("SCAN")]


@pytest.mark.parametrize("text", ["NVDA' OR '1'='1", "NVDA'; DROP TABLE signals; --", 'NV"DA', "NVD%", "NVD_"])
def test_filters_take_their_values_literally(db: Database, text: str) -> None:
    repo_signals.insert_signal(db, make_signal(status="target", outcome_r=1.0, exit_ts=T0 + 60))
    repo_signals.insert_chart_analysis(db, full_chart_analysis())

    assert repo_signals.list_signals(db, symbol=text) == []
    assert repo_signals.list_signals(db, statuses=[text]) == []
    assert ids(repo_signals.list_signals(db, statuses=[text, "target"])) == [1]
    assert repo_signals.last_signal_for_symbol(db, text) is None
    assert repo_signals.closed_outcomes(db, engine_version=text) == []
    assert repo_signals.latest_chart_analysis(db, text) is None
    assert repo_signals.latest_chart_analysis(db, "NVDA", timeframe=text) is None
    assert repo_signals.list_chart_analyses(db, text) == []
    assert db.scalar("SELECT COUNT(*) FROM signals") == 1  # and nothing was executed as SQL


def test_equal_times_are_ordered_by_id(db: Database) -> None:
    for status in ("active", "review", "target", "stopped", "active"):
        closed = status in ("target", "stopped")
        repo_signals.insert_signal(
            db, make_signal(status=status, exit_ts=T0 + 600 if closed else None, outcome_r=0.5 if closed else None)
        )

    assert ids(repo_signals.open_signals(db)) == [1, 2, 5]  # oldest first: the lower id
    assert ids(repo_signals.closed_outcomes(db)) == [3, 4]
    newest = repo_signals.last_signal_for_symbol(db, "NVDA")
    assert newest is not None and newest.id == 5  # newest: the higher id


def test_closed_outcomes_time_bounds(db: Database) -> None:
    for exit_ts in (T0 + 100, T0 + 200, T0 + 300):
        repo_signals.insert_signal(db, make_signal(status="stopped", outcome_r=-1.0, exit_ts=exit_ts))

    def exits(**bounds: int) -> list[int | None]:
        return [signal.exit_ts for signal in repo_signals.closed_outcomes(db, **bounds)]

    assert exits(since_ts=T0 + 200) == [T0 + 200, T0 + 300]  # the lower bound is inclusive ...
    assert exits(before_ts=T0 + 200) == [T0 + 100]  # ... the upper one exclusive: nothing from the decision time on
    assert exits(since_ts=T0 + 200, before_ts=T0 + 200) == []
    assert exits(since_ts=T0 + 100, before_ts=T0 + 301) == [T0 + 100, T0 + 200, T0 + 300]


def test_closed_outcomes_ignore_signals_that_are_still_open(db: Database) -> None:
    # An open signal may carry provisional numbers; only a closed state makes the result a known outcome.
    for status in ("review", "pending_entry", "active"):
        repo_signals.insert_signal(db, make_signal(status=status, outcome_r=0.3, exit_ts=T0 + 10))

    assert repo_signals.closed_outcomes(db) == []
    assert repo_signals.closed_outcomes(db, before_ts=T0 + 3600) == []


def test_numpy_values_in_json_columns_are_stored_as_plain_numbers(db: Database) -> None:
    np = pytest.importorskip("numpy")
    signal = make_signal(
        components={"trend": np.float64(0.75)},
        features={"rsi": np.float64(61.5), "bars": np.int64(120), "ema": np.array([101.25, 99.5]), "up": np.bool_(1)},
    )

    repo_signals.insert_signal(db, signal)

    loaded = repo_signals.get_signal(db, 1)
    assert loaded is not None
    assert loaded.components == {"trend": 0.75}
    assert loaded.features == {"rsi": 61.5, "bars": 120, "ema": [101.25, 99.5], "up": True}
    assert db.scalar("SELECT features_json FROM signals") == '{"rsi":61.5,"bars":120,"ema":[101.25,99.5],"up":true}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_feature_values_are_rejected_before_anything_is_stored(db: Database, value: float) -> None:
    signal = make_signal(features={"rsi": value})
    stored = repo_signals.insert_signal(db, make_signal())

    with pytest.raises(ValueError, match="JSON"):
        repo_signals.insert_signal(db, signal)
    with pytest.raises(ValueError, match="JSON"):
        repo_signals.update_signal(db, dataclasses.replace(stored, components={"trend": value}, score=0.99))

    assert signal.id is None
    assert repo_signals.get_signal(db, 1) == stored
    assert db.scalar("SELECT COUNT(*) FROM signals") == 1


def test_update_signal_overwrites_what_was_stored_after_the_object_was_read(db: Database) -> None:
    signal = repo_signals.insert_signal(db, make_signal(status="active"))
    repo_signals.mark_mailed(db, 1, T0 + 45)  # the mailer works with its own copy

    signal.checked_ts = T0 + 60
    repo_signals.update_signal(db, signal)  # full row: the tracker's stale copy wins
    assert db.scalar("SELECT mailed_ts FROM signals") is None

    signal.mailed_ts = T0 + 45  # carrying the change over keeps it
    repo_signals.update_signal(db, signal)
    assert repo_signals.get_signal(db, 1) == signal


def test_signal_writes_join_the_callers_transaction(db: Database) -> None:
    stored = repo_signals.insert_signal(db, make_signal())

    with pytest.raises(RuntimeError), db.write():
        repo_signals.insert_signal(db, make_signal())
        repo_signals.update_signal(db, dataclasses.replace(stored, status="active", entry_price=100.5))
        repo_signals.mark_mailed(db, 1, T0 + 45)
        repo_signals.insert_chart_analysis(db, full_chart_analysis())
        raise RuntimeError("the minute cycle failed after the writes")

    assert ids(repo_signals.list_signals(db)) == [1]
    assert repo_signals.get_signal(db, 1) == stored
    assert repo_signals.last_chart_analysis_ts(db) == {}


def test_latest_chart_analysis_of_equal_time_is_the_one_stored_last(db: Database) -> None:
    for comment in ("erste", "zweite"):
        repo_signals.insert_chart_analysis(db, dataclasses.replace(full_chart_analysis(), comment_de=comment))

    latest = repo_signals.latest_chart_analysis(db, "NVDA")

    assert latest is not None and (latest.id, latest.comment_de) == (2, "zweite")
    assert [analysis.id for analysis in repo_signals.list_chart_analyses(db, "NVDA")] == [2, 1]
