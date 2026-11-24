import asyncio
from datetime import datetime

import pytest

from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.db.database import Database
from trading_local.events import EventBus
from trading_local.runtime.scheduler import JOB_LABELS_DE, Runtime


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []


class FakeData:
    def __init__(self, rec: Recorder, fail: bool = False) -> None:
        self.rec, self.fail = rec, fail

    async def start(self) -> None:
        self.rec.calls.append("data.start")

    async def stop(self) -> None:
        self.rec.calls.append("data.stop")

    async def refresh(self, now: datetime) -> list[str]:
        self.rec.calls.append("data.refresh")
        if self.fail:
            raise RuntimeError("provider down")
        return ["NVDA"]


class FakeBroker:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    def process(self, now: datetime) -> list:
        self.rec.calls.append("broker.process")
        return []


class FakeEngine:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec
        self.updated: list[list[str]] = []

    async def on_minute(self, now: datetime, updated) -> None:
        self.rec.calls.append("engine.on_minute")
        self.updated.append(list(updated))

    async def stop(self) -> None:
        self.rec.calls.append("engine.stop")


class FakeMovers:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec
        self.updated: list[list[str]] = []

    async def on_minute(self, now: datetime, updated) -> list:
        self.rec.calls.append("movers.on_minute")
        self.updated.append(list(updated))
        return ["move"]

    async def stop(self) -> None:
        self.rec.calls.append("movers.stop")


class FakeNews:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    async def start(self) -> None:
        self.rec.calls.append("news.start")

    async def stop(self) -> None:
        self.rec.calls.append("news.stop")

    async def poll(self, now: datetime) -> int:
        self.rec.calls.append("news.poll")
        return 3


class FakeLlm:
    def __init__(self, rec: Recorder) -> None:
        self.rec = rec

    async def start(self) -> None:
        self.rec.calls.append("llm.start")

    async def stop(self) -> None:
        self.rec.calls.append("llm.stop")


def make_runtime(settings: Settings, db: Database, clock: ManualClock, *, fail_data: bool = False):
    rec = Recorder()
    engine = FakeEngine(rec)
    runtime = Runtime(
        settings=settings,
        db=db,
        clock=clock,
        bus=EventBus(),
        data=FakeData(rec, fail=fail_data),
        broker=FakeBroker(rec),
        engine=engine,
        news=FakeNews(rec),
        llm=FakeLlm(rec),
    )
    return runtime, rec, engine


async def test_cycle_runs_steps_in_fixed_order(settings: Settings, db: Database, clock: ManualClock) -> None:
    runtime, rec, engine = make_runtime(settings, db, clock)
    assert await runtime.run_cycle() is True
    assert rec.calls == ["data.refresh", "broker.process", "engine.on_minute"]
    assert engine.updated == [["NVDA"]]
    jobs = {j.job: j for j in runtime.jobs()}
    assert set(jobs) == set(JOB_LABELS_DE)
    assert jobs["market_data"].ok and jobs["market_data"].message.startswith("1 Symbol")
    rows = db.query("SELECT job, ok FROM job_runs ORDER BY id")
    assert [r["job"] for r in rows] == ["market_data", "broker", "engine"]


async def test_price_trigger_runs_last_and_stops_before_the_news(tmp_path, db: Database, clock: ManualClock) -> None:
    from trading_local.testing import make_settings

    rec = Recorder()
    movers = FakeMovers(rec)
    runtime = Runtime(
        settings=make_settings(tmp_path, offline=False),
        db=db,
        clock=clock,
        bus=EventBus(),
        data=FakeData(rec),
        broker=FakeBroker(rec),
        engine=FakeEngine(rec),
        news=FakeNews(rec),
        llm=FakeLlm(rec),
        movers=movers,
    )
    assert await runtime.run_cycle() is True
    assert rec.calls == ["data.refresh", "broker.process", "engine.on_minute", "movers.on_minute"]
    assert movers.updated == [["NVDA"]]
    assert {j.job: j for j in runtime.jobs()}["movers"].message == "1 ungewöhnliche Bewegung"
    rec.calls.clear()
    await runtime.start()
    await asyncio.sleep(0)
    await runtime.stop()
    assert rec.calls[-5:] == ["engine.stop", "movers.stop", "news.stop", "data.stop", "llm.stop"]


async def test_same_minute_is_not_processed_twice(settings: Settings, db: Database, clock: ManualClock) -> None:
    runtime, rec, _ = make_runtime(settings, db, clock)
    assert await runtime.run_cycle() is True
    clock.advance(30)
    assert await runtime.run_cycle() is False
    clock.advance(30)
    assert await runtime.run_cycle() is True
    assert rec.calls.count("data.refresh") == 2


async def test_failing_step_does_not_stop_the_cycle(settings: Settings, db: Database, clock: ManualClock) -> None:
    runtime, rec, engine = make_runtime(settings, db, clock, fail_data=True)
    await runtime.run_cycle()
    assert rec.calls == ["data.refresh", "broker.process", "engine.on_minute"]
    assert engine.updated == [[]]
    jobs = {j.job: j for j in runtime.jobs()}
    assert jobs["market_data"].ok is False and "provider down" in jobs["market_data"].message
    assert jobs["engine"].ok is True


async def test_offline_start_does_not_start_services(tmp_path, db: Database, clock: ManualClock) -> None:
    from trading_local.testing import make_settings

    runtime, rec, _ = make_runtime(make_settings(tmp_path, offline=True), db, clock)
    await runtime.start()
    await runtime.stop()
    assert rec.calls == []
    assert db.scalar("SELECT COUNT(*) FROM activity_log WHERE category = 'system'") == 1


async def test_online_start_and_stop_order(tmp_path, db: Database, clock: ManualClock) -> None:
    from trading_local.testing import make_settings

    runtime, rec, _ = make_runtime(make_settings(tmp_path, offline=False), db, clock)
    await runtime.start()
    await asyncio.sleep(0)
    await runtime.stop()
    assert rec.calls[:3] == ["llm.start", "data.start", "news.start"]
    assert rec.calls[-4:] == ["engine.stop", "news.stop", "data.stop", "llm.stop"]


async def test_news_service_is_not_started_when_news_are_disabled(tmp_path, db: Database, clock: ManualClock) -> None:
    from trading_local.testing import make_settings

    runtime, rec, _ = make_runtime(make_settings(tmp_path, offline=False, enable_news=False), db, clock)
    await runtime.start()
    await runtime.stop()
    assert "news.start" not in rec.calls
    assert rec.calls[:2] == ["llm.start", "data.start"]


async def test_calendar_is_warmed_up_off_the_event_loop(settings: Settings, db: Database, clock: ManualClock) -> None:
    import threading

    seen: list[tuple[str, bool]] = []
    loop_thread = threading.get_ident()

    class RecordingCalendar:
        def exchanges(self) -> list[str]:
            return ["XNYS", "XETR"]

        def state(self, exchange: str, at: datetime) -> None:
            seen.append((exchange, threading.get_ident() != loop_thread))

    rec = Recorder()
    runtime = Runtime(
        settings=settings,  # offline: the warm-up runs even without background jobs
        db=db,
        clock=clock,
        bus=EventBus(),
        data=FakeData(rec),
        broker=FakeBroker(rec),
        engine=FakeEngine(rec),
        news=FakeNews(rec),
        llm=FakeLlm(rec),
        calendar=RecordingCalendar(),  # type: ignore[arg-type]
    )
    await runtime.start()  # returns only after the calendars are built
    assert seen == [("XNYS", True), ("XETR", True)]
    await runtime.stop()


async def test_housekeeping_purges_old_news_but_keeps_recent(
    settings: Settings, db: Database, clock: ManualClock
) -> None:
    from trading_local.db import repo_news
    from trading_local.models import NewsItem
    from trading_local.runtime.scheduler import NEWS_RETENTION_DAYS

    now = clock.ts()
    old = NewsItem("old", "k-old", "feed", "https://example.invalid/a", "Alt", now - NEWS_RETENTION_DAYS * 86400 - 60)
    new = NewsItem("new", "k-new", "feed", "https://example.invalid/b", "Neu", now - 3600)
    repo_news.insert_news(db, [old, new])
    runtime, _, _ = make_runtime(settings, db, clock)

    message = await runtime._housekeeping()

    assert [r["uid"] for r in db.query("SELECT uid FROM news")] == ["new"]
    assert "1 Nachrichten" in message


def test_loop_exception_filter_silences_only_connection_reset_during_cleanup(caplog) -> None:
    import logging

    from trading_local.runtime.scheduler import install_loop_exception_filter

    loop = asyncio.new_event_loop()
    try:
        install_loop_exception_filter(loop)
        with caplog.at_level(logging.DEBUG, logger="asyncio"):
            loop.call_exception_handler(
                {
                    "message": "Exception in callback _ProactorBasePipeTransport._call_connection_lost()",
                    "exception": ConnectionResetError(10054, "reset by peer"),
                }
            )
            loop.call_exception_handler({"message": "Task exception was never retrieved", "exception": ValueError("x")})
    finally:
        loop.close()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "never retrieved" in errors[0].getMessage()


@pytest.mark.parametrize("offset", [0.0, 3.0, 30.0])
async def test_sleep_until_targets_next_offset(
    monkeypatch, settings: Settings, db: Database, clock: ManualClock, offset: float
) -> None:
    runtime, _, _ = make_runtime(settings, db, clock)
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("trading_local.runtime.scheduler.time.time", lambda: 1_000_020.5)  # second 0.5 of a minute
    monkeypatch.setattr("trading_local.runtime.scheduler.asyncio.sleep", fake_sleep)
    await runtime._sleep_until(offset)
    assert 0.0 <= slept[0] <= 60.0
    assert (1_000_020.5 + slept[0]) % 60 == pytest.approx(offset % 60)
