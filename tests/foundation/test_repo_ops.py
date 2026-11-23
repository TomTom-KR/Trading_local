import asyncio
import logging
import sqlite3
import threading
from typing import Any

import pytest
from pydantic import ValidationError

from trading_local.config import RUNTIME_SETTINGS_KEY, RuntimeSettings
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import (
    ActivityOut,
    BaselineOut,
    DbStatusOut,
    FoldOut,
    LlmCallOut,
    MailLogOut,
    StatBlock,
    ValidationResultOut,
    ValidationRunOut,
)
from trading_local.events import TOPIC_ACTIVITY, EventBus

T0 = 1_800_000_000
HOUR = 3600
DAY = 86_400


def stored_runtime_settings(db: Database) -> Any:
    return repo_ops.get_setting(db, RUNTIME_SETTINGS_KEY)


def store_raw_runtime_settings(db: Database, text: str) -> None:
    db.execute("INSERT INTO settings (key, value, updated_ts) VALUES (?, ?, 0)", (RUNTIME_SETTINGS_KEY, text))


def add_signal(db: Database, symbol: str) -> int:
    """A stored signal (with its instrument) for mail-log entries that refer to one."""
    db.execute(
        "INSERT OR IGNORE INTO instruments (symbol, name, asset_class, region, exchange, currency, timezone) "
        "VALUES (?, ?, 'equity', 'US', 'XNAS', 'USD', 'America/New_York')",
        (symbol, symbol),
    )
    return db.execute(
        "INSERT INTO signals (created_ts, data_ts, symbol, direction, setup, timeframe, horizon_min, ref_price, atr, "
        "stop_dist, target_dist, score, confidence, status, engine_version) "
        "VALUES (?, ?, ?, 'LONG', 'breakout', '15m', 240, 100, 1.5, 2, 3, 0.7, 0.6, 'active', 'v1')",
        (T0, T0, symbol),
    )


def log_mail(db: Database, ts: int, status: str, **fields: Any) -> int:
    values: dict[str, Any] = {"kind": "signal", "recipient": "max@example.org", "subject": "LONG NVDA"}
    values.update(fields)
    return repo_ops.log_mail(db, ts=ts, status=status, **values)


# ------------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [{"a": [1, 2.5, "ä"], "b": None}, [1, 2], "text", 7, 0, False, None])
def test_setting_round_trip(db: Database, value: Any) -> None:
    repo_ops.set_setting(db, "some.key", value, T0)

    assert repo_ops.get_setting(db, "some.key", "fallback") == value


def test_get_setting_default_and_overwrite(db: Database) -> None:
    assert repo_ops.get_setting(db, "missing") is None
    assert repo_ops.get_setting(db, "missing", 42) == 42

    repo_ops.set_setting(db, "k", {"v": 1}, T0)
    repo_ops.set_setting(db, "k", {"v": 2}, T0 + 5)

    assert repo_ops.get_setting(db, "k") == {"v": 2}
    assert [tuple(row) for row in db.query("SELECT key, updated_ts FROM settings")] == [("k", T0 + 5)]


def test_get_setting_with_broken_json_returns_the_default(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    db.execute("INSERT INTO settings (key, value, updated_ts) VALUES ('broken', '{not json', 0)")

    with caplog.at_level(logging.WARNING, logger="trading_local.db.repo_ops"):
        assert repo_ops.get_setting(db, "broken", "fallback") == "fallback"

    assert "broken" in caplog.text


# ------------------------------------------------------------------------------------------------
# SettingsStore
# ------------------------------------------------------------------------------------------------
def test_store_returns_defaults_when_nothing_is_stored(db: Database) -> None:
    assert SettingsStore(db).get() == RuntimeSettings()
    assert stored_runtime_settings(db) is None  # reading never writes


def test_store_hands_out_private_copies(db: Database) -> None:
    store = SettingsStore(db)

    first = store.get()
    first.mail_enabled = False
    first.benchmarks.append("^N225")

    assert store.get() == RuntimeSettings()
    assert store.get() is not store.get()


def test_update_merges_a_partial_patch_and_persists(db: Database) -> None:
    store = SettingsStore(db)
    assert store.get() == RuntimeSettings()  # fills the cache: the updates below must refresh it

    updated = store.update({"mail_max_per_hour": 3, "mail_regions": ["EU"]}, T0)
    assert store.get() == updated == RuntimeSettings(mail_max_per_hour=3, mail_regions=["EU"])
    again = store.update({"allow_short": False}, T0 + 60)

    expected = RuntimeSettings(mail_max_per_hour=3, mail_regions=["EU"], allow_short=False)
    assert again == expected and store.get() == expected
    assert SettingsStore(db).get() == expected  # a new store reads the persisted row
    assert stored_runtime_settings(db) == expected.model_dump(mode="json")
    assert db.scalar("SELECT updated_ts FROM settings WHERE key = ?", (RUNTIME_SETTINGS_KEY,)) == T0 + 60


def test_update_result_is_a_private_copy(db: Database) -> None:
    store = SettingsStore(db)

    updated = store.update({"benchmarks": ["^GSPC"]}, T0)
    updated.benchmarks.append("^N225")

    assert store.get().benchmarks == ["^GSPC"]


def test_update_validates_and_coerces_like_the_model(db: Database) -> None:
    store = SettingsStore(db)

    updated = store.update({"mail_max_per_hour": "15", "min_score_actionable": 1}, T0)

    assert (updated.mail_max_per_hour, updated.min_score_actionable) == (15, 1.0)
    assert stored_runtime_settings(db)["mail_max_per_hour"] == 15


@pytest.mark.parametrize(
    "patch",
    [
        {"mail_max_per_hour": 500},  # above the allowed range
        {"risk_per_trade_pct": 0},  # must be positive
        {"mail_regions": ["EU", "MARS"]},  # unknown region
        {"allow_long": "vielleicht"},
        {"min_score_review": None},
        {"mail_max_per_hour": 5, "benchmarks": "DAX"},  # one bad value rejects the whole patch
    ],
)
def test_update_with_invalid_values_changes_nothing(db: Database, patch: dict[str, Any]) -> None:
    store = SettingsStore(db)
    before = store.update({"mail_max_per_hour": 20}, T0)

    with pytest.raises(ValidationError):
        store.update(patch, T0 + 60)

    assert store.get() == before
    assert stored_runtime_settings(db) == before.model_dump(mode="json")
    assert db.scalar("SELECT updated_ts FROM settings") == T0


def test_update_with_unknown_keys_raises_a_plain_value_error(db: Database) -> None:
    store = SettingsStore(db)

    with pytest.raises(ValueError, match="no_such_setting") as caught:
        store.update({"mail_max_per_hour": 5, "no_such_setting": 1}, T0)

    assert not isinstance(caught.value, ValidationError)
    assert store.get() == RuntimeSettings()
    assert stored_runtime_settings(db) is None


def test_update_refuses_to_run_inside_a_callers_transaction(db: Database) -> None:
    store = SettingsStore(db)

    with db.write(), pytest.raises(RuntimeError, match="transaction"):
        store.update({"allow_short": False}, T0)

    assert store.get() == RuntimeSettings()
    assert stored_runtime_settings(db) is None


@pytest.mark.parametrize("text", ["{not json", "", "[1, 2, 3]", '"text"', "null", "17"])
def test_store_survives_a_corrupted_row(db: Database, text: str) -> None:
    store_raw_runtime_settings(db, text)
    store = SettingsStore(db)

    assert store.get() == RuntimeSettings()

    updated = store.update({"allow_short": False}, T0)  # and the next update repairs the row
    assert updated == RuntimeSettings(allow_short=False)
    assert store.get() == updated
    assert stored_runtime_settings(db) == updated.model_dump(mode="json")


def test_invalid_stored_values_fall_back_field_by_field(db: Database, caplog: pytest.LogCaptureFixture) -> None:
    store_raw_runtime_settings(
        db,
        '{"mail_max_per_hour": 999, "mail_regions": ["EU", "MARS"], "min_score_review": "hoch", '
        '"allow_short": false, "max_open_positions": 4, "setting_of_an_older_version": true}',
    )
    store = SettingsStore(db)

    with caplog.at_level(logging.WARNING, logger="trading_local.db.repo_ops"):
        settings = store.get()

    assert settings == RuntimeSettings(allow_short=False, max_open_positions=4)  # the valid values survive
    assert "mail_max_per_hour, mail_regions, min_score_review" in caplog.text
    assert store.update({"llm_review": False}, T0) == RuntimeSettings(
        allow_short=False, max_open_positions=4, llm_review=False
    )


def test_concurrent_partial_updates_are_all_kept(db: Database) -> None:
    store = SettingsStore(db)
    fields = [
        "mail_max_per_hour",
        "mail_symbol_cooldown_min",
        "llm_review_timeout_s",
        "signal_cooldown_min",
        "max_open_positions",
        "chart_scan_interval_min",
    ]
    errors: list[BaseException] = []

    def run(action: Any, *args: Any) -> None:
        try:
            action(*args)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def read_and_scribble() -> None:
        for _ in range(50):
            store.get().benchmarks.append("scribble")  # copies may be changed freely

    for round_no in range(5):
        value = 20 + round_no
        barrier = threading.Barrier(len(fields))

        def write(field: str, barrier: threading.Barrier = barrier, value: int = value) -> None:
            barrier.wait(timeout=10)
            store.update({field: value}, T0 + value)

        threads = [threading.Thread(target=run, args=(write, field)) for field in fields]
        threads += [threading.Thread(target=run, args=(read_and_scribble,)) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors
        assert store.get() == RuntimeSettings(**dict.fromkeys(fields, value))  # no update was lost
    assert SettingsStore(db).get() == store.get()


def test_readers_do_not_wait_for_a_blocked_update(db: Database) -> None:
    store = SettingsStore(db)
    seen: list[RuntimeSettings] = []
    updater = threading.Thread(target=store.update, args=({"allow_short": False}, T0))
    reader = threading.Thread(target=lambda: seen.append(store.get()))

    with db.write():  # another writer is busy: the update has to wait for the database
        updater.start()
        reader.start()
        reader.join(timeout=10)
        assert not reader.is_alive(), "get() waited for the pending update"
        assert seen == [RuntimeSettings()]
        assert store.get() == RuntimeSettings()  # also from the thread that holds the write lock
    updater.join(timeout=10)

    assert not updater.is_alive()
    assert store.get() == RuntimeSettings(allow_short=False)


# ------------------------------------------------------------------------------------------------
# Mail log
# ------------------------------------------------------------------------------------------------
def test_mail_log_is_listed_newest_first(db: Database) -> None:
    signal_id = add_signal(db, "NVDA")
    first = log_mail(db, T0 + 60, "sent", signal_id=signal_id)
    second = log_mail(db, T0, "skipped", kind="test", subject="Test", detail="Limit erreicht")  # older, stored later
    third = log_mail(db, T0 + 60, "failed", kind="exit", detail="SMTP 535")

    entries = repo_ops.list_mail_log(db)

    assert (first, second, third) == (1, 2, 3)
    assert [entry.id for entry in entries] == [3, 1, 2]
    assert entries[1] == MailLogOut(
        id=1,
        ts=T0 + 60,
        kind="signal",
        signal_id=signal_id,
        recipient="max@example.org",
        subject="LONG NVDA",
        status="sent",
    )
    assert (entries[2].kind, entries[2].signal_id, entries[2].detail) == ("test", None, "Limit erreicht")
    assert [entry.id for entry in repo_ops.list_mail_log(db, limit=1)] == [3]


def test_log_mail_rejects_an_unknown_status(db: Database) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        log_mail(db, T0, "bounced")


def test_count_mails_counts_delivered_mails_by_default(db: Database) -> None:
    for ts, status in [
        (T0 - HOUR - 1, "sent"),
        (T0 - HOUR, "sent"),
        (T0 - 10, "file"),
        (T0 - 5, "skipped"),
        (T0 - 4, "failed"),
        (T0, "sent"),
    ]:
        log_mail(db, ts, status)

    assert repo_ops.count_mails(db, since_ts=T0 - HOUR) == 3
    assert repo_ops.count_mails(db, since_ts=T0 - HOUR, statuses=["failed", "skipped"]) == 2
    assert repo_ops.count_mails(db, since_ts=T0 - HOUR, statuses=[]) == 0
    assert repo_ops.count_mails(db, since_ts=T0 + 1) == 0


def test_last_mail_ts_for_symbol(db: Database) -> None:
    nvda, sap = add_signal(db, "NVDA"), add_signal(db, "SAP.DE")
    older_nvda = add_signal(db, "NVDA")
    log_mail(db, T0 + 10, "sent", signal_id=older_nvda)
    log_mail(db, T0 + 50, "file", kind="exit", signal_id=nvda)
    log_mail(db, T0 + 60, "failed", signal_id=nvda)  # not delivered
    log_mail(db, T0 + 70, "skipped", signal_id=nvda)  # not delivered
    log_mail(db, T0 + 80, "sent", kind="digest", signal_id=nvda)  # not a signal or exit mail
    log_mail(db, T0 + 90, "sent", signal_id=sap)
    log_mail(db, T0 + 95, "sent", kind="test")  # no signal

    assert repo_ops.last_mail_ts_for_symbol(db, "NVDA") == T0 + 50
    assert repo_ops.last_mail_ts_for_symbol(db, "SAP.DE") == T0 + 90
    assert repo_ops.last_mail_ts_for_symbol(db, "TSLA") is None


def test_last_mail_error_is_cleared_by_a_later_delivery(db: Database) -> None:
    assert repo_ops.last_mail_error(db) == ""

    log_mail(db, T0, "sent")
    log_mail(db, T0 + 10, "failed", detail="SMTP 535 Authentication failed")
    log_mail(db, T0 + 20, "skipped", detail="Limit erreicht")  # neither a failure nor a delivery
    assert repo_ops.last_mail_error(db) == "SMTP 535 Authentication failed"

    log_mail(db, T0 + 30, "failed", detail="Zeitüberschreitung")
    assert repo_ops.last_mail_error(db) == "Zeitüberschreitung"

    log_mail(db, T0 + 40, "file")
    assert repo_ops.last_mail_error(db) == ""


# ------------------------------------------------------------------------------------------------
# LLM calls
# ------------------------------------------------------------------------------------------------
def test_llm_calls_are_listed_newest_first(db: Database) -> None:
    first = repo_ops.log_llm_call(db, ts=T0 + 60, kind="news", ok=True, latency_ms=2100)
    second = repo_ops.log_llm_call(
        db,
        ts=T0 + 120,
        kind="trade_review",
        ok=False,
        latency_ms=45_000,
        queue_wait_ms=350,
        prompt_tokens=1800,
        output_tokens=0,
        error="timeout",
    )
    third = repo_ops.log_llm_call(db, ts=T0, kind="chart", ok=True, latency_ms=9000)  # older, stored later

    calls = repo_ops.list_llm_calls(db)

    assert (first, second, third) == (1, 2, 3)
    assert [call.id for call in calls] == [2, 1, 3]
    assert calls[0] == LlmCallOut(
        id=2,
        ts=T0 + 120,
        kind="trade_review",
        ok=False,
        latency_ms=45_000,
        queue_wait_ms=350,
        prompt_tokens=1800,
        output_tokens=0,
        error="timeout",
    )
    assert calls[1] == LlmCallOut(id=1, ts=T0 + 60, kind="news", ok=True, latency_ms=2100)
    assert calls[1].ok is True and calls[0].ok is False
    assert [call.id for call in repo_ops.list_llm_calls(db, limit=2)] == [2, 1]


def test_llm_call_stats(db: Database) -> None:
    repo_ops.log_llm_call(db, ts=T0 - HOUR - 1, kind="news", ok=False, latency_ms=10, error="too old to count")
    repo_ops.log_llm_call(db, ts=T0 - HOUR, kind="news", ok=True, latency_ms=1000)
    repo_ops.log_llm_call(db, ts=T0 - 600, kind="news", ok=False, latency_ms=45_000, error="timeout")
    repo_ops.log_llm_call(db, ts=T0 - 300, kind="chart", ok=True, latency_ms=4000)
    repo_ops.log_llm_call(db, ts=T0 - 60, kind="news", ok=False, latency_ms=5, error="connection refused")

    stats = repo_ops.llm_call_stats(db, since_ts=T0 - HOUR)

    assert stats == {
        "calls": 4,
        "errors": 2,
        "last_latency_ms": 4000,  # the failed calls say nothing about the model's speed
        "avg_latency_ms": 2500.0,
        "last_error": "connection refused",
    }


def test_llm_call_stats_without_calls_or_without_errors(db: Database) -> None:
    empty = {"calls": 0, "errors": 0, "last_latency_ms": None, "avg_latency_ms": None, "last_error": ""}
    assert repo_ops.llm_call_stats(db, since_ts=T0) == empty

    repo_ops.log_llm_call(db, ts=T0, kind="news", ok=True, latency_ms=1500)

    assert repo_ops.llm_call_stats(db, since_ts=T0) == {
        "calls": 1,
        "errors": 0,
        "last_latency_ms": 1500,
        "avg_latency_ms": 1500.0,
        "last_error": "",
    }
    assert repo_ops.llm_call_stats(db, since_ts=T0 + 1) == empty


# ------------------------------------------------------------------------------------------------
# Job runs
# ------------------------------------------------------------------------------------------------
def test_latest_job_runs_returns_the_newest_run_of_every_job(db: Database) -> None:
    assert repo_ops.latest_job_runs(db) == {}

    repo_ops.log_job_run(db, job="minute", started_ts=T0, duration_ms=120, ok=True)
    repo_ops.log_job_run(db, job="minute", started_ts=T0 + 60, duration_ms=95, ok=False, message="Quelle langsam")
    repo_ops.log_job_run(db, job="news", started_ts=T0 + 30, duration_ms=800, ok=True)
    repo_ops.log_job_run(db, job="news", started_ts=T0 - 300, duration_ms=700, ok=False)  # older, stored later

    assert repo_ops.latest_job_runs(db) == {
        "minute": {"job": "minute", "started_ts": T0 + 60, "duration_ms": 95, "ok": False, "message": "Quelle langsam"},
        "news": {"job": "news", "started_ts": T0 + 30, "duration_ms": 800, "ok": True, "message": ""},
    }
    assert repo_ops.latest_job_runs(db)["news"]["ok"] is True


def test_log_job_run_keeps_the_newest_500_runs_per_job(db: Database) -> None:
    with db.write():
        for n in range(505):
            repo_ops.log_job_run(db, job="minute", started_ts=T0 + 60 * n, duration_ms=n, ok=True)
        for n in range(3):
            repo_ops.log_job_run(db, job="news", started_ts=T0 + 300 * n, duration_ms=n, ok=True)

    [minute] = db.query("SELECT COUNT(*), MIN(started_ts), MAX(started_ts) FROM job_runs WHERE job = 'minute'")
    assert tuple(minute) == (500, T0 + 60 * 5, T0 + 60 * 504)
    assert db.scalar("SELECT COUNT(*) FROM job_runs WHERE job = 'news'") == 3


# ------------------------------------------------------------------------------------------------
# Activity log
# ------------------------------------------------------------------------------------------------
def test_activity_log_round_trip_and_filters(db: Database) -> None:
    first = repo_ops.log_activity(db, ts=T0, level="info", category="data", message="Kurse aktualisiert")
    second = repo_ops.log_activity(
        db, ts=T0 + 5, level="warning", category="signal", message="Signal verworfen", ref_type="signal", ref_id=12
    )
    third = repo_ops.log_activity(db, ts=T0 + 9, level="error", category="data", message="Datenquelle nicht erreichbar")
    fourth = repo_ops.log_activity(db, ts=T0 + 9, level="info", category="signal", message="Signal erstellt")

    entries = repo_ops.list_activity(db)

    assert (first, second, third, fourth) == (1, 2, 3, 4)
    assert [entry.id for entry in entries] == [4, 3, 2, 1]
    assert entries[2] == ActivityOut(
        id=2, ts=T0 + 5, level="warning", category="signal", message="Signal verworfen", ref_type="signal", ref_id=12
    )
    assert (entries[3].ref_type, entries[3].ref_id) == (None, None)
    assert [entry.id for entry in repo_ops.list_activity(db, level="info")] == [4, 1]
    assert [entry.id for entry in repo_ops.list_activity(db, category="data")] == [3, 1]
    assert [entry.id for entry in repo_ops.list_activity(db, level="info", category="data")] == [1]
    assert [entry.id for entry in repo_ops.list_activity(db, limit=2)] == [4, 3]
    assert [entry.id for entry in repo_ops.list_activity(db, limit=2, before_id=3)] == [2, 1]
    assert repo_ops.list_activity(db, before_id=1) == []


async def test_log_activity_publishes_the_stored_entry(db: Database, bus: EventBus) -> None:
    bus.bind_loop()
    with bus.subscribe() as subscription:
        activity_id = repo_ops.log_activity(
            db,
            ts=T0,
            level="warning",
            category="mail",
            message="Mail-Limit erreicht",
            ref_type="signal",
            ref_id=7,
            bus=bus,
        )
        event = await asyncio.wait_for(subscription.get(), timeout=5)

    assert event.topic == TOPIC_ACTIVITY
    assert event.data == {
        "id": activity_id,
        "ts": T0,
        "level": "warning",
        "category": "mail",
        "message": "Mail-Limit erreicht",
        "ref_type": "signal",
        "ref_id": 7,
    }
    assert ActivityOut.model_validate(event.data) == repo_ops.list_activity(db)[0]


async def test_log_activity_publishes_from_a_worker_thread(db: Database, bus: EventBus) -> None:
    bus.bind_loop()
    with bus.subscribe() as subscription:
        activity_id = await asyncio.to_thread(
            repo_ops.log_activity, db, ts=T0, level="info", category="sim", message="Order ausgeführt", bus=bus
        )
        event = await asyncio.wait_for(subscription.get(), timeout=5)

    assert (event.topic, event.data["id"], event.data["message"]) == (TOPIC_ACTIVITY, activity_id, "Order ausgeführt")


async def test_log_activity_without_a_bus_publishes_nothing(db: Database, bus: EventBus) -> None:
    bus.bind_loop()
    with bus.subscribe() as subscription:
        repo_ops.log_activity(db, ts=T0, level="info", category="system", message="Gestartet")
        bus.publish("probe")
        event = await asyncio.wait_for(subscription.get(), timeout=5)

    assert event.topic == "probe"  # nothing was announced before the probe
    assert len(repo_ops.list_activity(db)) == 1


def test_log_activity_with_an_unbound_bus_still_stores_the_entry(db: Database, bus: EventBus) -> None:
    activity_id = repo_ops.log_activity(db, ts=T0, level="info", category="system", message="Gestartet", bus=bus)

    assert [entry.id for entry in repo_ops.list_activity(db)] == [activity_id]


async def test_rejected_activity_is_neither_stored_nor_published(db: Database, bus: EventBus) -> None:
    bus.bind_loop()
    with bus.subscribe() as subscription:
        with pytest.raises(sqlite3.IntegrityError):
            repo_ops.log_activity(db, ts=T0, level="debug", category="system", message="x", bus=bus)
        bus.publish("probe")
        event = await asyncio.wait_for(subscription.get(), timeout=5)

    assert event.topic == "probe"  # nothing was announced before the probe
    assert repo_ops.list_activity(db) == []


# ------------------------------------------------------------------------------------------------
# Validation runs
# ------------------------------------------------------------------------------------------------
def make_result() -> ValidationResultOut:
    return ValidationResultOut(
        period_start_ts=T0 - 30 * DAY,
        period_end_ts=T0,
        trading_days=21,
        symbols=8,
        decisions=4200,
        overall=StatBlock(n=40, wins=22, win_rate=0.55, avg_r=0.12, total_r=4.8, profit_factor=1.3),
        folds=[FoldOut(index=0, start_ts=T0 - 30 * DAY, end_ts=T0 - 15 * DAY, stats=StatBlock(n=18, wins=9))],
        baseline=BaselineOut(n_sims=500, mean_avg_r=-0.05, p95_avg_r=0.09, p_value=0.04),
        equity_r=[(T0 - DAY, 0.5), (T0, 4.8)],
        warnings_de=["Wenige Trades im zweiten Abschnitt"],
        verdict_de="Leichter Vorteil gegenüber Zufallseinstiegen",
    )


def test_validation_run_lifecycle(db: Database) -> None:
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={"days": 30, "symbols": ["NVDA"]})

    running = repo_ops.get_validation_run(db, run_id)
    assert running == ValidationRunOut(
        id=1, ts=T0, status="running", engine_version="v1", params={"days": 30, "symbols": ["NVDA"]}, result=None
    )

    result = make_result()
    repo_ops.finish_validation_run(db, run_id, status="done", result=result.model_dump(), message="fertig")

    finished = repo_ops.get_validation_run(db, run_id)
    assert finished is not None
    assert (finished.status, finished.message, finished.params) == ("done", "fertig", {"days": 30, "symbols": ["NVDA"]})
    assert finished.result == result
    assert repo_ops.get_validation_run(db, 99) is None


def test_failed_validation_run_has_no_result(db: Database) -> None:
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={})

    repo_ops.finish_validation_run(db, run_id, status="failed", message="Zu wenig Daten")

    failed = repo_ops.get_validation_run(db, run_id)
    assert failed is not None
    assert (failed.status, failed.message, failed.result, failed.params) == ("failed", "Zu wenig Daten", None, {})


@pytest.mark.parametrize("result_json", ['{"unexpected": 1}', "{broken", "[1, 2]", '"text"', "", "{}"])
def test_unusable_stored_results_are_read_as_none(db: Database, result_json: str) -> None:
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={"days": 5})
    db.execute("UPDATE validation_runs SET status = 'done', result_json = ?, params_json = '[]'", (result_json,))

    run = repo_ops.get_validation_run(db, run_id)

    assert run is not None
    assert (run.status, run.result, run.params) == ("done", None, {})


def test_validation_runs_are_listed_newest_first(db: Database) -> None:
    for ts in (T0 + 100, T0, T0 + 200, T0 + 200):
        repo_ops.create_validation_run(db, ts=ts, engine_version="v1", params={})
    repo_ops.finish_validation_run(db, 3, status="done", result=make_result().model_dump())

    runs = repo_ops.list_validation_runs(db, limit=3)

    assert [run.id for run in runs] == [4, 3, 1]
    assert runs[1].result == make_result() and runs[0].result is None
    assert [run.id for run in repo_ops.list_validation_runs(db, limit=10)] == [4, 3, 1, 2]
    assert len(repo_ops.list_validation_runs(db)) == 4  # default limit 5


# ------------------------------------------------------------------------------------------------
# Database statistics and retention
# ------------------------------------------------------------------------------------------------
def test_db_stats_of_an_empty_database(db: Database) -> None:
    stats = repo_ops.db_stats(db)

    assert isinstance(stats, DbStatusOut)
    assert (stats.bars_1m, stats.first_bar_ts, stats.last_bar_ts, stats.news, stats.signals) == (0, None, None, 0, 0)
    assert stats.size_mb > 0


def test_db_stats_counts_rows_and_finds_the_bar_range_across_symbols(db: Database) -> None:
    bars = [("AAA", 600), ("AAA", 660), ("MMM", 300), ("MMM", 9000), ("ZZZ", 60), ("ZZZ", 120), ("ZZZ", 180)]
    db.executemany(
        "INSERT INTO bars_1m (symbol, ts, open, high, low, close) VALUES (?, ?, 1, 1, 1, 1)", bars
    )
    db.executemany(
        "INSERT INTO news (uid, title_key, source, url, title, fetched_ts) VALUES (?, ?, 'feed', 'https://x', 't', 0)",
        [("u1", "k1"), ("u2", "k2")],
    )
    add_signal(db, "NVDA")

    stats = repo_ops.db_stats(db)

    # the earliest bar belongs to the alphabetically last symbol, the latest one to the middle symbol
    assert (stats.bars_1m, stats.first_bar_ts, stats.last_bar_ts) == (7, 60, 9000)
    assert (stats.news, stats.signals) == (2, 1)


def test_db_stats_size_includes_the_write_ahead_log(db: Database) -> None:
    db.conn().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = repo_ops.db_stats(db).size_mb
    payload = "x" * 1000
    db.executemany(
        "INSERT INTO activity_log (ts, level, category, message) VALUES (0, 'info', 'system', ?)",
        [(payload,) for _ in range(2000)],
    )  # about 2 MB that sit in the write-ahead log until the next checkpoint

    wal_bytes = db.path.with_name(db.path.name + "-wal").stat().st_size
    assert wal_bytes > 1_000_000
    assert repo_ops.db_stats(db).size_mb >= before + 1.0


def test_db_stats_without_a_write_ahead_log_file(db: Database) -> None:
    db.conn().execute("PRAGMA journal_mode=DELETE")  # leaving WAL mode removes the -wal file

    assert not db.path.with_name(db.path.name + "-wal").exists()
    assert repo_ops.db_stats(db).size_mb > 0


def test_prune_ops_deletes_log_rows_older_than_the_retention(db: Database) -> None:
    old, cutoff = T0 - 30 * DAY - 1, T0 - 30 * DAY
    for ts in (old, cutoff, T0):
        repo_ops.log_llm_call(db, ts=ts, kind="news", ok=True, latency_ms=1)
        repo_ops.log_job_run(db, job="minute", started_ts=ts, duration_ms=1, ok=True)
        repo_ops.log_activity(db, ts=ts, level="info", category="system", message="x")
        log_mail(db, ts, "sent")
    repo_ops.create_validation_run(db, ts=old, engine_version="v1", params={})
    repo_ops.set_setting(db, "k", 1, old)

    repo_ops.prune_ops(db, now_ts=T0)

    for table, column in [("llm_calls", "ts"), ("job_runs", "started_ts"), ("activity_log", "ts"), ("mail_log", "ts")]:
        assert [row[0] for row in db.query(f"SELECT {column} FROM {table} ORDER BY {column}")] == [cutoff, T0], table
    assert db.scalar("SELECT COUNT(*) FROM validation_runs") == 1  # not a log: kept
    assert repo_ops.get_setting(db, "k") == 1

    repo_ops.prune_ops(db, now_ts=T0, keep_days=0)

    for table in ("llm_calls", "job_runs", "activity_log", "mail_log"):
        assert db.scalar(f"SELECT COUNT(*) FROM {table}") == 1, table


# ------------------------------------------------------------------------------------------------
# Review additions: hostile input, transactions, retention details, query cost
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


@pytest.mark.parametrize("text", ["info' OR '1'='1", "sent'; DROP TABLE mail_log; --", 'NV"DA', "NVD%", "%", "_"])
def test_filters_take_their_values_literally(db: Database, text: str) -> None:
    signal_id = add_signal(db, "NVDA")
    log_mail(db, T0, "sent", signal_id=signal_id)
    repo_ops.log_activity(db, ts=T0, level="info", category="system", message="Gestartet")
    repo_ops.set_setting(db, "k", 1, T0)

    assert repo_ops.list_activity(db, level=text) == []
    assert repo_ops.list_activity(db, category=text) == []
    assert repo_ops.count_mails(db, since_ts=0, statuses=[text]) == 0
    assert repo_ops.count_mails(db, since_ts=0, statuses=[text, "sent"]) == 1
    assert repo_ops.last_mail_ts_for_symbol(db, text) is None
    assert repo_ops.get_setting(db, text, "fallback") == "fallback"
    assert db.scalar("SELECT COUNT(*) FROM mail_log") == 1  # and nothing was executed as SQL


def test_texts_are_stored_verbatim(db: Database) -> None:
    text = "Kurs > 100 €; 'Zitat' \"doppelt\" -- kein Kommentar\nzweite Zeile"

    repo_ops.log_activity(db, ts=T0, level="info", category=text, message=text, ref_type=text)
    log_mail(db, T0, "failed", recipient=text, subject=text, detail=text)
    repo_ops.log_llm_call(db, ts=T0, kind="news", ok=False, latency_ms=1, error=text)
    repo_ops.log_job_run(db, job=text, started_ts=T0, duration_ms=1, ok=False, message=text)
    repo_ops.set_setting(db, text, {"text": text}, T0)

    [activity] = repo_ops.list_activity(db, category=text)
    assert (activity.message, activity.ref_type) == (text, text)
    [mail] = repo_ops.list_mail_log(db)
    assert (mail.recipient, mail.subject, mail.detail) == (text, text, text)
    assert repo_ops.last_mail_error(db) == text
    assert repo_ops.llm_call_stats(db, since_ts=T0)["last_error"] == text
    assert repo_ops.latest_job_runs(db)[text]["message"] == text
    assert repo_ops.get_setting(db, text) == {"text": text}


def test_settings_keep_numpy_values_as_plain_json(db: Database) -> None:
    np = pytest.importorskip("numpy")

    repo_ops.set_setting(db, "k", {"n": np.int64(3), "x": np.float64(0.5), "list": np.array([1, 2])}, T0)

    assert repo_ops.get_setting(db, "k") == {"n": 3, "x": 0.5, "list": [1, 2]}
    with pytest.raises(ValueError, match="JSON"):
        repo_ops.set_setting(db, "k", {"x": float("nan")}, T0 + 1)  # not valid JSON: rejected, nothing changes
    assert repo_ops.get_setting(db, "k") == {"n": 3, "x": 0.5, "list": [1, 2]}


def test_store_reads_what_another_thread_updated(db: Database) -> None:
    store = SettingsStore(db)
    assert store.get() == RuntimeSettings()  # fills the cache in this thread

    updater = threading.Thread(target=store.update, args=({"signals_enabled": False}, T0))
    updater.start()
    updater.join(timeout=30)

    assert store.get() == RuntimeSettings(signals_enabled=False)


def test_update_with_an_empty_patch_keeps_the_values(db: Database) -> None:
    store = SettingsStore(db)
    before = store.update({"max_open_positions": 3}, T0)

    assert store.update({}, T0 + 60) == before
    assert store.get() == before


def test_a_failed_update_leaves_cache_and_row_as_they_were_for_other_stores(db: Database) -> None:
    store = SettingsStore(db)
    store.update({"mail_max_per_hour": 20}, T0)
    assert store.get().mail_max_per_hour == 20  # cached

    with pytest.raises(ValidationError):
        store.update({"mail_max_per_hour": 20, "max_gross_leverage": 99}, T0 + 60)
    with pytest.raises(ValueError, match="typo"):
        store.update({"mail_max_per_hour": 30, "typo": 1}, T0 + 60)

    assert store.get() == SettingsStore(db).get() == RuntimeSettings(mail_max_per_hour=20)


def test_finish_validation_run_rejects_a_result_that_is_not_json(db: Database) -> None:
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={"days": 5})

    with pytest.raises(ValueError, match="JSON"):
        repo_ops.finish_validation_run(db, run_id, status="done", result={"overall": {"profit_factor": float("inf")}})

    run = repo_ops.get_validation_run(db, run_id)
    assert run is not None and (run.status, run.result, run.message) == ("running", None, "")


def test_finish_validation_run_rejects_an_unknown_status_and_ignores_an_unknown_run(db: Database) -> None:
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={})

    with pytest.raises(sqlite3.IntegrityError):
        repo_ops.finish_validation_run(db, run_id, status="aborted")
    repo_ops.finish_validation_run(db, 99, status="done")

    run = repo_ops.get_validation_run(db, run_id)
    assert run is not None and run.status == "running"
    assert len(repo_ops.list_validation_runs(db)) == 1


def test_validation_result_round_trip_keeps_optional_numbers_that_are_missing(db: Database) -> None:
    result = ValidationResultOut(
        period_start_ts=T0 - DAY,
        period_end_ts=T0,
        trading_days=1,
        symbols=1,
        decisions=0,
        overall=StatBlock(),  # no trades: every rate is None, not NaN
    )
    run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={})

    repo_ops.finish_validation_run(db, run_id, status="done", result=result.model_dump())

    run = repo_ops.get_validation_run(db, run_id)
    assert run is not None and run.result == result
    assert run.result is not None and run.result.overall.win_rate is None and run.result.baseline is None


def test_job_runs_are_trimmed_by_start_time_and_not_by_insertion_order(db: Database) -> None:
    with db.write():
        for n in range(500):
            repo_ops.log_job_run(db, job="minute", started_ts=T0 + 60 * n, duration_ms=n, ok=True)

    repo_ops.log_job_run(db, job="minute", started_ts=T0 - 60, duration_ms=1, ok=False)  # a late report of an old run

    [minute] = db.query("SELECT COUNT(*), MIN(started_ts), MAX(started_ts) FROM job_runs WHERE job = 'minute'")
    assert tuple(minute) == (500, T0, T0 + 60 * 499)
    assert repo_ops.latest_job_runs(db)["minute"]["ok"] is True


def test_log_writes_join_the_callers_transaction(db: Database) -> None:
    with pytest.raises(RuntimeError), db.write():
        repo_ops.log_activity(db, ts=T0, level="info", category="system", message="x")
        log_mail(db, T0, "sent")
        repo_ops.log_llm_call(db, ts=T0, kind="news", ok=True, latency_ms=1)
        repo_ops.log_job_run(db, job="minute", started_ts=T0, duration_ms=1, ok=True)
        repo_ops.set_setting(db, "k", 1, T0)
        run_id = repo_ops.create_validation_run(db, ts=T0, engine_version="v1", params={})
        repo_ops.finish_validation_run(db, run_id, status="failed")
        repo_ops.prune_ops(db, now_ts=T0)
        raise RuntimeError("the job failed after logging")

    for table in ("activity_log", "mail_log", "llm_calls", "job_runs", "settings", "validation_runs"):
        assert db.scalar(f"SELECT COUNT(*) FROM {table}") == 0, table


def test_last_mail_ts_for_symbol_ignores_mails_of_deleted_signals_and_other_symbols(db: Database) -> None:
    nvda = add_signal(db, "NVDA")
    log_mail(db, T0 + 10, "sent", signal_id=nvda)
    log_mail(db, T0 + 99, "sent", signal_id=nvda + 1000)  # no such signal

    assert repo_ops.last_mail_ts_for_symbol(db, "NVDA") == T0 + 10
    assert repo_ops.last_mail_ts_for_symbol(db, "nvda") is None  # symbols are case-sensitive


def test_last_mail_error_within_one_second_follows_the_order_of_the_attempts(db: Database) -> None:
    log_mail(db, T0, "sent")
    log_mail(db, T0 + 5, "failed", detail="SMTP 421")
    log_mail(db, T0 + 5, "skipped", detail="Limit erreicht")
    assert repo_ops.last_mail_error(db) == "SMTP 421"

    log_mail(db, T0 + 5, "sent")  # same second, logged later: the retry went through
    assert repo_ops.last_mail_error(db) == ""

    log_mail(db, T0 + 5, "failed", detail="SMTP 554")
    log_mail(db, T0 + 4, "sent")  # an older delivery that is logged late does not hide the newer failure
    assert repo_ops.last_mail_error(db) == "SMTP 554"


def test_last_mail_error_with_skipped_mails_only(db: Database) -> None:
    for second in range(3):
        log_mail(db, T0 + second, "skipped", detail="Mail-Versand ist ausgeschaltet")

    assert repo_ops.last_mail_error(db) == ""


def test_status_page_readers_use_the_indexes(db: Database) -> None:
    def plan_of(run: Any) -> str:
        return " | ".join(query_plan(db, run))

    stats_plan = query_plan(db, lambda: repo_ops.db_stats(db))
    # exactly one full pass, the unavoidable bar count; the bar range is found by primary-key seeks per symbol
    assert [step for step in stats_plan if step.startswith("SCAN") and "INDEX" not in step] == [
        "SCAN CONSTANT ROW",
        "SCAN bars_1m",
        "SCAN symbols",
    ]
    assert sum("SEARCH bars_1m USING PRIMARY KEY (symbol=?)" in step for step in stats_plan) == 2
    assert any(step.startswith("SCAN news USING COVERING INDEX") for step in stats_plan)
    assert any(step.startswith("SCAN signals USING COVERING INDEX") for step in stats_plan)

    assert "mail_log_ts_idx" in plan_of(lambda: repo_ops.count_mails(db, since_ts=T0))
    assert "mail_log_signal_idx" in plan_of(lambda: repo_ops.last_mail_ts_for_symbol(db, "NVDA"))
    assert "llm_calls_ts_idx" in plan_of(lambda: repo_ops.llm_call_stats(db, since_ts=T0))
    assert "mail_log_ts_idx" in plan_of(lambda: repo_ops.last_mail_error(db))
