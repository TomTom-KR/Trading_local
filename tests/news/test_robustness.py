"""Hostile input, boundaries, failure injection and concurrency of the news pipeline (review tests)."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta

import httpx
import pytest

from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.db import repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.events import TOPIC_NEWS, EventBus
from trading_local.news import service as service_module
from trading_local.news.fetcher import SUMMARY_MAX_CHARS, clean_text, parse_published, title_key
from trading_local.testing import make_settings

from tests.news.conftest import ServiceFactory
from tests.news.helpers import (
    NOW,
    NOW_TS,
    Entry,
    FakeLlm,
    FeedServer,
    drain,
    fresh,
    make_feed,
    rss,
    wait_for_analyses,
)
from tests.news.test_service import order_titles, statuses, stored_item


def elapsed_s(func: Callable[[str], object], arg: str) -> float:
    started = time.perf_counter()
    func(arg)
    return time.perf_counter() - started


# ------------------------------------------------------------------------------------------------
# Hostile text: every regex must stay linear on untrusted feed content
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "title",
    ["x - " + "A" * 396 + "!", "x - " + "Ab" * 198 + "!", "Nvidia - " + "NVDA." * 78 + "!", "a | " + "ÄÖÜ" * 130 + "?"],
    ids=["upper-run", "camel-run", "ticker-run", "umlaut-run"],
)
def test_title_key_is_fast_on_pathological_titles(title: str) -> None:
    assert elapsed_s(title_key, title) < 0.2


def test_title_key_still_strips_publisher_suffixes() -> None:
    assert title_key("Fed holds rates - The Wall Street Journal") == title_key("Fed holds rates")
    assert title_key("Fed holds rates | Reuters") == title_key("Fed holds rates")
    assert title_key("Dax schließt fester - Anleger optimistisch") != title_key("Dax schließt fester")


@pytest.mark.parametrize(
    "text",
    [
        "<table>" * 300_000,  # unclosed blocks: each one used to scan to the end of the text
        "<figure><p>x" * 100_000,
        "the post " * 200_000,
        "word " * 400_000,
        "&amp;lt;" * 200_000,
    ],
    ids=["unclosed-table", "unclosed-figure", "teaser", "words", "entities"],
)
def test_clean_text_is_fast_and_bounded_on_huge_html(text: str) -> None:
    started = time.perf_counter()
    value = clean_text(text)
    assert time.perf_counter() - started < 0.5
    assert len(value) <= SUMMARY_MAX_CHARS and "<table" not in value and "<figure" not in value


def test_clean_text_still_removes_blocks() -> None:
    html = "<p>Vorher</p><TABLE class='x'><tr><td>Zelle</td></tr></table ><script>alert(1)</script> nachher"
    assert clean_text(html) == "Vorher nachher"
    assert clean_text("<figure><img src='a.png'><figcaption>Bild</figcaption></figure>Text") == "Text"
    assert clean_text("<table><tr><td>offen</td></tr> Rest") == "offen Rest"  # unclosed: tags removed, text kept
    long_markup = "<figure>" + "<img srcset='x'>" * 2_000 + "</figure>" + "Inhalt nach langem Bildblock"
    assert clean_text(long_markup) == "Inhalt nach langem Bildblock"


def test_parse_published_ignores_absurd_input_quickly() -> None:
    assert elapsed_s(parse_published, "Mon, " * 50_000) < 0.2
    assert parse_published("Fri, 20 Nov 2026 19:45:00 +0100") is not None


# ------------------------------------------------------------------------------------------------
# Selection policy boundaries
# ------------------------------------------------------------------------------------------------
async def test_age_and_quality_boundaries(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    good, plain = make_feed("t_quality4", quality=4), make_feed("t_quality3", quality=3)
    server.serve(good, rss([
        Entry("Nvidia opens research lab in Munich", NOW_TS - 12 * 3600, "https://news.test/a"),
        Entry("SAP names new finance chief", NOW_TS - 12 * 3600 - 1, "https://news.test/b"),
        fresh("Harbour authority publishes annual traffic statistics"),
    ]))
    server.serve(plain, rss([fresh("Garden show attracts record crowd of visitors")]))
    service = make_service([good, plain], llm)
    await service.start()
    assert await service.poll(NOW) == 4
    await wait_for_analyses(service)
    found = statuses(db)
    assert found["Nvidia opens research lab in Munich"] == ("analyzed", "")  # exactly 12 h: not 'more than'
    assert found["SAP names new finance chief"] == ("skipped", "alt")
    assert found["Harbour authority publishes annual traffic statistics"] == ("analyzed", "")  # quality 4
    assert found["Garden show attracts record crowd of visitors"] == ("skipped", "kein_bezug")  # quality 3


async def test_disabled_feeds_are_neither_polled_nor_counted(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm
) -> None:
    off, on = make_feed("t_off_feed", enabled=False), make_feed("t_on_feed")
    server.serve(off, rss([fresh("Nvidia unveils new AI chip")]))
    server.serve(on, rss([]))
    service = make_service([off, on], llm)
    await service.poll(NOW)  # poll before start initialises the service itself
    assert server.count(off) == 0 and server.count(on) == 1
    assert service.status().feeds_total == 1


# ------------------------------------------------------------------------------------------------
# Timing: slow hosts, receive time, concurrency, cancellation
# ------------------------------------------------------------------------------------------------
def hanging(cancelled: list[bool]) -> Callable[[httpx.Request], Awaitable[httpx.Response]]:
    """A handler whose response never arrives; appends to ``cancelled`` when its request is cancelled."""

    async def hang(request: httpx.Request) -> httpx.Response:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200)

    return hang


async def test_a_slow_host_cannot_hold_the_round(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(service_module, "ROUND_DEADLINE_S", 0.1)
    slow, fast = make_feed("t_slowpoke", poll_seconds=60), make_feed("t_quick", poll_seconds=60)
    cancelled: list[bool] = []
    server.on(slow, hanging(cancelled))
    server.serve(fast, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([slow, fast], llm)
    await service.start()
    started = time.perf_counter()
    assert await service.poll(NOW) == 1
    assert time.perf_counter() - started < 1.0
    assert cancelled == [True]
    states = repo_news.list_feed_states(db)
    assert "t_slowpoke" not in states  # untouched: still due, no failure counted
    assert states["t_quick"].last_ok_ts == NOW_TS
    now_clock.advance(60)
    server.serve(slow, rss([fresh("Late story")]))
    server.handlers.clear()
    assert await service.poll(now_clock.now()) == 1  # retried in the next round
    assert server.count(slow) == 2


async def test_fetched_ts_is_the_time_the_body_arrived(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock
) -> None:
    feed = make_feed("t_arrival")

    def delayed(request: httpx.Request) -> httpx.Response:
        now_clock.advance(7)  # the response arrives 7 s after the poll started
        return httpx.Response(200, content=rss([fresh("Nvidia unveils new AI chip")]))

    server.on(feed, delayed)
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    (item, _), = repo_news.list_news(db, include_duplicates=True)
    assert item.fetched_ts == NOW_TS + 7
    assert repo_news.list_feed_states(db)["t_arrival"].next_poll_ts == NOW_TS + 60  # schedule: poll time


async def test_overlapping_polls_and_cancellation(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    feed = make_feed("t_overlap")
    gate = asyncio.Event()
    cancelled: list[bool] = []

    async def held(request: httpx.Request) -> httpx.Response:
        try:
            await gate.wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(200, content=rss([fresh("Nvidia unveils new AI chip")]))

    server.on(feed, held)
    service = make_service([feed], llm)
    await service.start()
    first = asyncio.create_task(service.poll(NOW))
    await asyncio.sleep(0.02)
    assert await asyncio.wait_for(service.poll(NOW), timeout=0.5) == 0  # skipped, never waits
    gate.set()
    assert await first == 1

    gate.clear()
    server.serve(feed, b"")
    server.on(feed, held)
    await service.stop()
    restarted = make_service([feed], llm)
    poll = asyncio.create_task(restarted.poll(NOW + timedelta(minutes=1)))  # the feed is due again
    await asyncio.sleep(0.02)
    poll.cancel()  # the runtime cancels its news loop on shutdown
    with pytest.raises(asyncio.CancelledError):
        await poll
    assert cancelled == [True]  # the in-flight request was cancelled as well
    others = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
    assert others == []


async def test_stop_during_a_poll_cancels_its_requests(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, caplog: pytest.LogCaptureFixture
) -> None:
    feed = make_feed("t_stop_poll")
    cancelled: list[bool] = []
    server.on(feed, hanging(cancelled))
    service = make_service([feed], llm)
    await service.start()
    poll = asyncio.create_task(service.poll(NOW))
    await asyncio.sleep(0.02)
    await asyncio.wait_for(service.stop(), timeout=1.0)
    assert await asyncio.wait_for(poll, timeout=1.0) == 0
    assert cancelled == [True]
    assert not [record for record in caplog.records if record.levelname == "ERROR"]


# ------------------------------------------------------------------------------------------------
# Failure injection and final status announcements
# ------------------------------------------------------------------------------------------------
async def test_announced_items_are_announced_again_when_they_are_given_up(
    make_service: ServiceFactory, server: FeedServer, now_clock: ManualClock, bus: EventBus
) -> None:
    bus.bind_loop()
    subscription = bus.subscribe()
    feed = make_feed("t_gone")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([feed], FakeLlm(now_clock, unavailable=True))
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    events = [e.data for e in await drain(subscription) if e.topic == TOPIC_NEWS]
    assert [data["status"] for data in events] == ["queued", "skipped"]
    assert events[-1]["analysis"] is None


async def test_back_pressure_announces_dropped_waiting_items(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, bus: EventBus, now_clock: ManualClock
) -> None:
    bus.bind_loop()
    feed = make_feed("t_flood2")
    server.serve(feed, rss(order_titles(range(1, 51))))
    llm.gate = asyncio.Event()
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    subscription = bus.subscribe()
    now_clock.advance(60)
    server.serve(feed, rss(order_titles(range(1, 56))))
    assert await service.poll(now_clock.now()) == 5
    events = [e.data for e in await drain(subscription) if e.topic == TOPIC_NEWS]
    dropped = sorted(int(data["title"].split()[4]) for data in events if data["status"] == "skipped")
    assert dropped == [11, 12, 13, 14, 15]  # waiting before, now given up for fresher news
    assert sorted(int(data["title"].split()[4]) for data in events if data["status"] == "queued") == [
        51, 52, 53, 54, 55
    ]
    llm.gate.set()
    await wait_for_analyses(service)


async def test_activity_log_failure_does_not_undo_the_analysis(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> int:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(repo_ops, "log_activity", broken)
    feed = make_feed("t_activity")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    llm.relevance = {"Nvidia": 5}
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("analyzed", "")


async def test_invalid_analysis_marks_only_that_item_as_error(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    feed = make_feed("t_invalid")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip"), fresh("SAP raises cloud outlook")]))
    llm.relevance = {"Nvidia": 9}  # violates the 1..5 contract (CHECK constraint)
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    found = statuses(db)
    assert found["Nvidia unveils new AI chip"] == ("error", "analysefehler")
    assert found["SAP raises cloud outlook"] == ("analyzed", "")


async def test_recovery_runs_once_the_service_becomes_ready_after_a_failed_start(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_news.insert_news(db, [stored_item("Nvidia pending story", minutes_ago=10, status="queued", hinted=["NVDA"])])
    feed = make_feed("t_alpha")
    server.serve(feed, rss([]))
    original = repo_news.list_feed_states

    def broken(*args: object, **kwargs: object) -> dict[str, repo_news.FeedState]:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(repo_news, "list_feed_states", broken)
    service = make_service([feed], llm)
    await service.start()  # never raises
    assert llm.calls == []
    monkeypatch.setattr(repo_news, "list_feed_states", original)
    await service.poll(NOW)
    await wait_for_analyses(service)
    assert [item.title for item, _ in llm.calls] == ["Nvidia pending story"]
    assert statuses(db)["Nvidia pending story"] == ("analyzed", "")
    await service.poll(NOW)
    await wait_for_analyses(service)
    assert len(llm.calls) == 1  # recovery runs once


async def test_start_does_nothing_when_news_are_switched_off(
    make_service: ServiceFactory, llm: FakeLlm, db: Database, settings: Settings
) -> None:
    repo_news.insert_news(db, [stored_item("Nvidia pending story", minutes_ago=10, status="queued", hinted=["NVDA"])])
    service = make_service([make_feed("t_alpha")], llm, service_settings=make_settings(settings.data_dir,
                                                                                         enable_news=False))
    await service.start()
    await asyncio.sleep(0.02)
    assert llm.calls == []
    assert statuses(db)["Nvidia pending story"] == ("queued", "")
