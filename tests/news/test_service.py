"""NewsService: the poll round, selection policy, back-pressure, recovery, events and status."""

import asyncio
from datetime import timedelta

import httpx
import pytest

from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.contracts import News
from trading_local.db import repo_market, repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_news import FeedState
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import TOPIC_ACTIVITY, TOPIC_NEWS, EventBus
from trading_local.models import Instrument, NewsItem
from trading_local.news import build_news, service as service_module
from trading_local.news.service import MAX_OUTSTANDING, NewsService
from trading_local.testing import make_settings

from tests.news.conftest import ServiceFactory
from tests.news.helpers import NOW, NOW_TS, Entry, FakeLlm, FeedServer, drain, fresh, make_feed, rss, wait_for_analyses


def by_title(db: Database) -> dict[str, NewsItem]:
    return {item.title: item for item, _ in repo_news.list_news(db, include_duplicates=True, limit=500)}


def statuses(db: Database) -> dict[str, tuple[str, str]]:
    return {title: (item.status, item.status_reason) for title, item in by_title(db).items()}


# ------------------------------------------------------------------------------------------------
# A full round
# ------------------------------------------------------------------------------------------------
async def test_full_poll_round(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, bus: EventBus, db: Database
) -> None:
    bus.bind_loop()
    subscription = bus.subscribe()
    alpha = make_feed("t_alpha", quality=3)
    beta = make_feed("t_beta", quality=5)
    server.serve(alpha, rss([
        fresh("Nvidia beats estimates as data center sales soar", minutes_ago=6),
        fresh("Local bakery wins regional award"),
        Entry("SAP sells its old printing unit", NOW_TS - 13 * 3600, "https://news.test/sap-old"),
    ]))
    server.serve(beta, rss([
        fresh("Weather service expects a mild winter season", minutes_ago=3),
        Entry("Nvidia beats estimates as data center sales soar", NOW_TS - 120, "https://other.test/nvda"),
    ]))
    llm.relevance = {"Nvidia": 5}
    service = make_service([alpha, beta], llm)
    await service.start()

    assert await service.poll(NOW) == 5
    await wait_for_analyses(service)

    items = list(repo_news.list_news(db, include_duplicates=True))
    nvidia = [(item, analysis) for item, analysis in items if item.title.startswith("Nvidia")]
    original, duplicate = sorted(nvidia, key=lambda pair: pair[0].status == "duplicate")
    assert original[0].status == "analyzed" and original[0].hinted_symbols == ["NVDA"]
    assert original[1] is not None and original[1].relevance == 5 and original[1].symbols == {"NVDA": 0.5}
    assert duplicate[0].status == "duplicate" and duplicate[0].duplicate_of == original[0].id
    found = statuses(db)
    assert found["Local bakery wins regional award"] == ("skipped", "kein_bezug")
    assert found["SAP sells its old printing unit"] == ("skipped", "alt")
    assert found["Weather service expects a mild winter season"] == ("analyzed", "")  # quality 5: no match needed
    assert all(item.fetched_ts == NOW_TS for item, _ in items)

    assert sorted((item.title[:7], candidates) for item, candidates in llm.calls) == [
        ("Nvidia ", ["NVDA"]),
        ("Weather", []),
    ]
    assert [row.news_id for row in repo_news.symbol_news(db, "NVDA", asof_ts=NOW_TS, window_s=3600)] == [
        original[0].id
    ]

    events = await drain(subscription)
    news_events = [event.data for event in events if event.topic == TOPIC_NEWS]
    assert sorted((data["status"], data["title"][:7]) for data in news_events) == [
        ("analyzed", "Nvidia "), ("analyzed", "Weather"), ("queued", "Nvidia "), ("queued", "Weather"),
    ]
    queued = next(data for data in news_events if data["status"] == "queued" and data["title"].startswith("Nvidia"))
    assert queued["analysis"] is None and queued["source_name"] == f"Feed {queued['source']}"
    analysed = next(data for data in news_events if data["status"] == "analyzed" and data["title"].startswith("Nvidia"))
    assert analysed["analysis"]["relevance"] == 5 and analysed["analysis"]["summary_de"] == "Kurzfassung"
    activity = [event.data for event in events if event.topic == TOPIC_ACTIVITY]
    assert [data["message"] for data in activity] == [
        "Wichtige Nachricht (Relevanz 5): Nvidia beats estimates as data center sales soar"
    ]
    assert [entry.ref_id for entry in repo_ops.list_activity(db, category="news")] == [original[0].id]

    status = service.status()
    assert (status.feeds_total, status.feeds_failing) == (2, 0)
    assert (status.items_last_hour, status.analyzed_last_hour, status.backlog) == (5, 2, 0)
    assert status.last_poll_ts == NOW_TS
    feeds = {feed.feed_id: feed for feed in service.feeds()}
    assert (feeds["t_alpha"].items_total, feeds["t_beta"].items_total) == (3, 2)
    assert feeds["t_alpha"].last_ok_ts == NOW_TS and feeds["t_alpha"].name == "Feed t_alpha"
    stored = repo_news.list_feed_states(db)
    assert stored["t_alpha"].next_poll_ts == NOW_TS + 60 and stored["t_beta"].items_total == 2


async def test_poll_returns_before_the_llm_finishes(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    feed = make_feed("t_gate")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    llm.gate = asyncio.Event()
    service = make_service([feed], llm)
    await service.start()
    assert await asyncio.wait_for(service.poll(NOW), timeout=2.0) == 1
    await asyncio.sleep(0.02)
    assert len(llm.calls) == 1  # handed to the LLM ...
    assert statuses(db)["Nvidia unveils new AI chip"] == ("queued", "")  # ... but not finished
    assert service.status().backlog == 1
    llm.gate.set()
    await wait_for_analyses(service)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("analyzed", "")
    assert service.status().backlog == 0


async def test_second_poll_only_stores_new_items(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock
) -> None:
    feed = make_feed("t_repeat")
    first = [fresh("Nvidia unveils new AI chip"), fresh("Bitcoin rebounds above 60,000")]
    server.serve(feed, rss(first))
    service = make_service([feed], llm)
    await service.start()
    assert await service.poll(NOW) == 2
    now_clock.advance(60)
    server.serve(feed, rss([*first, fresh("SAP raises cloud outlook", minutes_ago=1)]))
    assert await service.poll(now_clock.now()) == 1
    await wait_for_analyses(service)
    assert len(llm.calls) == 3
    assert by_title(db)["SAP raises cloud outlook"].fetched_ts == NOW_TS + 60


# ------------------------------------------------------------------------------------------------
# Selection policy
# ------------------------------------------------------------------------------------------------
async def test_llm_switched_off_in_the_runtime_settings(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, runtime_settings: SettingsStore
) -> None:
    runtime_settings.update({"news_llm_enabled": False}, NOW_TS)
    feed = make_feed("t_off")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("skipped", "llm_aus")
    assert llm.calls == []


async def test_llm_disabled_in_the_settings(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, settings: Settings
) -> None:
    feed = make_feed("t_disabled")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([feed], llm, service_settings=make_settings(settings.data_dir, enable_llm=False))
    await service.start()
    await service.poll(NOW)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("skipped", "llm_aus")


async def test_llm_unavailable_and_failing(
    make_service: ServiceFactory, server: FeedServer, now_clock: ManualClock, db: Database
) -> None:
    unavailable_feed, failing_feed = make_feed("t_none"), make_feed("t_fail")
    server.serve(unavailable_feed, rss([fresh("Nvidia unveils new AI chip")]))
    server.serve(failing_feed, rss([fresh("SAP raises cloud outlook")]))
    unavailable = make_service([unavailable_feed], FakeLlm(now_clock, unavailable=True))
    failing = make_service([failing_feed], FakeLlm(now_clock, fail=True))
    for service in (unavailable, failing):
        await service.start()
        await service.poll(NOW)
        await wait_for_analyses(service)
    found = statuses(db)
    assert found["Nvidia unveils new AI chip"] == ("skipped", "llm_nicht_verfuegbar")
    assert found["SAP raises cloud outlook"] == ("error", "analysefehler")


async def test_llm_may_only_name_candidates(
    make_service: ServiceFactory, server: FeedServer, now_clock: ManualClock, db: Database
) -> None:
    feed = make_feed("t_extra")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([feed], FakeLlm(now_clock, extra_symbol="AAPL"))
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    item = by_title(db)["Nvidia unveils new AI chip"]
    _, analysis = repo_news.get_news(db, item.id or 0) or (None, None)
    assert analysis is not None and analysis.symbols == {"NVDA": 0.5}


async def test_macro_headline_gets_the_mapped_index_as_candidate(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    feed = make_feed("t_macro", quality=3)
    server.serve(feed, rss([fresh("EZB senkt überraschend den Leitzins"), fresh("Fed holds rates, Nvidia slips")]))
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    assert by_title(db)["EZB senkt überraschend den Leitzins"].hinted_symbols == ["^GDAXI"]
    assert sorted(candidates for _, candidates in llm.calls) == [["NVDA", "^GSPC"], ["^GDAXI"]]


async def test_llm_switched_off_while_items_wait(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, runtime_settings: SettingsStore
) -> None:
    feed = make_feed("t_switch")
    server.serve(feed, rss(order_titles(range(10))))
    llm.gate = asyncio.Event()
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await asyncio.sleep(0.02)
    assert len(llm.calls) == service_module.LLM_DISPATCH_LIMIT
    runtime_settings.update({"news_llm_enabled": False}, NOW_TS)
    llm.gate.set()
    await wait_for_analyses(service)
    reasons = sorted(status for status in statuses(db).values())
    assert reasons.count(("analyzed", "")) == 8 and reasons.count(("skipped", "llm_aus")) == 2


# ------------------------------------------------------------------------------------------------
# Back-pressure
# ------------------------------------------------------------------------------------------------
def order_titles(numbers: range) -> list[Entry]:
    """Distinct NVDA headlines; a higher number was published later (fresher)."""
    return [fresh(f"Nvidia wins order number {i} from customer {i}", minutes_ago=200 - i) for i in numbers]


async def test_back_pressure_drops_the_oldest_waiting_items(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, bus: EventBus, db: Database,
    now_clock: ManualClock,
) -> None:
    bus.bind_loop()
    subscription = bus.subscribe()
    feed = make_feed("t_flood")
    server.serve(feed, rss(order_titles(range(1, 51))))
    llm.gate = asyncio.Event()
    service = make_service([feed], llm)
    await service.start()

    assert await service.poll(NOW) == 50
    await asyncio.sleep(0.02)
    number = {f"Nvidia wins order number {i} from customer {i}": i for i in range(1, 56)}
    found = {number[title]: status for title, status in statuses(db).items()}
    assert {i for i, status in found.items() if status == ("skipped", "rueckstau")} == set(range(1, 11))
    assert {i for i, status in found.items() if status == ("queued", "")} == set(range(11, 51))
    assert [number[item.title] for item, _ in llm.calls] == list(range(50, 42, -1))  # newest first, 8 at a time
    queued_events = [e for e in await drain(subscription) if e.topic == TOPIC_NEWS and e.data["status"] == "queued"]
    assert len(queued_events) == MAX_OUTSTANDING  # only admitted items are announced

    now_clock.advance(60)
    server.serve(feed, rss(order_titles(range(1, 56))))
    assert await service.poll(now_clock.now()) == 5
    found = {number[title]: status for title, status in statuses(db).items()}
    assert {i for i, status in found.items() if status == ("skipped", "rueckstau")} == set(range(1, 16))
    assert service.status().backlog == MAX_OUTSTANDING

    llm.gate.set()
    await wait_for_analyses(service)
    found = {number[title]: status for title, status in statuses(db).items()}
    assert {i for i, status in found.items() if status == ("analyzed", "")} == set(range(16, 56))
    assert len(llm.calls) == MAX_OUTSTANDING + 0  # 40 analyses, no dropped item reached the LLM


# ------------------------------------------------------------------------------------------------
# Restart recovery
# ------------------------------------------------------------------------------------------------
def stored_item(title: str, *, minutes_ago: int, status: str, hinted: list[str], source: str = "t_alpha") -> NewsItem:
    return NewsItem(
        uid=f"uid-{title}",
        title_key=f"key-{title}",
        source=source,
        url="https://news.test/" + title.replace(" ", "-"),
        title=title,
        fetched_ts=NOW_TS - minutes_ago * 60,
        hinted_symbols=hinted,
        status=status,  # type: ignore[arg-type]
    )


async def test_restart_requeues_pending_items_newest_first(
    make_service: ServiceFactory, llm: FakeLlm, db: Database
) -> None:
    repo_news.insert_news(db, [
        stored_item("Nvidia older pending story", minutes_ago=30, status="queued", hinted=["NVDA"]),
        stored_item("Nvidia newer pending story", minutes_ago=10, status="queued", hinted=["NVDA"]),
        stored_item("Nvidia story from before the gap", minutes_ago=180, status="queued", hinted=["NVDA"]),
        stored_item("Unmatched new story", minutes_ago=20, status="new", hinted=[]),
        stored_item("Already analysed story", minutes_ago=5, status="analyzed", hinted=["NVDA"]),
    ])
    llm.gate = asyncio.Event()
    service = make_service([make_feed("t_alpha", quality=3)], llm)
    await service.start()
    await asyncio.sleep(0.02)
    assert [item.title for item, _ in llm.calls] == ["Nvidia newer pending story", "Nvidia older pending story"]
    assert llm.calls[0][1] == ["NVDA"]
    llm.gate.set()
    await wait_for_analyses(service)
    found = statuses(db)
    assert found["Nvidia newer pending story"] == ("analyzed", "")
    assert found["Nvidia older pending story"] == ("analyzed", "")
    assert found["Nvidia story from before the gap"] == ("skipped", "alt")
    assert found["Unmatched new story"] == ("skipped", "kein_bezug")
    assert service.status().backlog == 0


async def test_stop_cancels_analyses_and_keeps_items_queued(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    feed = make_feed("t_stop")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    llm.gate = asyncio.Event()
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    await asyncio.sleep(0.02)
    await asyncio.wait_for(service.stop(), timeout=2.0)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("queued", "")
    assert await service.poll(NOW + timedelta(minutes=1)) == 0
    # the next start picks the item up again
    restarted = make_service([feed], FakeLlm(llm.clock))
    await restarted.start()
    await wait_for_analyses(restarted)
    assert statuses(db)["Nvidia unveils new AI chip"] == ("analyzed", "")


# ------------------------------------------------------------------------------------------------
# Duplicates
# ------------------------------------------------------------------------------------------------
async def test_near_duplicate_from_another_source(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database
) -> None:
    alpha, beta = make_feed("t_alpha", quality=5), make_feed("t_beta", quality=5)
    server.serve(alpha, rss([fresh("UN climate talks collapse as EU and others reject deal")]))
    server.serve(beta, rss([fresh("UN climate talks in Turkey collapse as nations reject deal")]))
    service = make_service([alpha, beta], llm)
    await service.start()
    await service.poll(NOW)
    await wait_for_analyses(service)
    items = by_title(db)
    duplicates = [item for item in items.values() if item.status == "duplicate"]
    assert len(duplicates) == 1 and len(llm.calls) == 1
    assert duplicates[0].duplicate_of == next(item.id for item in items.values() if item.status == "analyzed")


async def test_near_duplicate_index_survives_a_restart(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock
) -> None:
    alpha, beta = make_feed("t_alpha", quality=5), make_feed("t_beta", quality=5)
    server.serve(alpha, rss([fresh("Intuit to sell Mailchimp to Bending Spoons for $3.68 billion")]))
    first = make_service([alpha], llm)
    await first.start()
    await first.poll(NOW)
    await wait_for_analyses(first)
    await first.stop()

    now_clock.advance(3600)
    server.serve(beta, rss([Entry("Bending Spoons buys Mailchimp from Intuit for $3.68 billion", NOW_TS + 3000,
                                  "https://other.test/mailchimp")]))
    second = make_service([beta], FakeLlm(now_clock))
    await second.start()
    assert await second.poll(now_clock.now()) == 1
    item = by_title(db)["Bending Spoons buys Mailchimp from Intuit for $3.68 billion"]
    assert item.status == "duplicate"
    assert item.duplicate_of == by_title(db)["Intuit to sell Mailchimp to Bending Spoons for $3.68 billion"].id


# ------------------------------------------------------------------------------------------------
# Feeds, failures and politeness
# ------------------------------------------------------------------------------------------------
async def test_one_broken_feed_never_affects_the_others(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock
) -> None:
    broken, good = make_feed("t_broken", poll_seconds=300), make_feed("t_good", poll_seconds=60)
    server.on(broken, lambda request: httpx.Response(500))
    server.serve(good, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([broken, good], llm)
    await service.start()
    assert await service.poll(NOW) == 1
    status = service.status()
    assert (status.feeds_total, status.feeds_failing) == (2, 1)
    feeds = {feed.feed_id: feed for feed in service.feeds()}
    assert (feeds["t_broken"].fail_count, feeds["t_broken"].last_error) == (1, "HTTP 500")
    assert repo_news.list_feed_states(db)["t_broken"].next_poll_ts == NOW_TS + 300
    now_clock.advance(60)
    await service.poll(now_clock.now())
    assert (server.count(broken), server.count(good)) == (1, 2)  # the broken feed backs off


async def test_unreachable_host_is_skipped_for_the_rest_of_the_round(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock
) -> None:
    first, second = (make_feed(name, host="slow.test", poll_seconds=300) for name in ("t_slow_a", "t_slow_b"))

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    server.on(first, refuse)
    server.on(second, refuse)
    service = make_service([first, second], llm)
    await service.start()
    await service.poll(NOW)
    assert server.count(first) + server.count(second) == 1
    (failed,) = repo_news.list_feed_states(db).values()  # the skipped feed was not touched at all
    assert (failed.fail_count, failed.last_error) == (1, "Verbindungsfehler (ConnectError)")
    now_clock.advance(60)
    await service.poll(now_clock.now())
    assert server.count(first) + server.count(second) == 2  # the skipped feed was due and is fetched now


async def test_host_budget_defers_feeds_to_the_next_round(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, now_clock: ManualClock
) -> None:
    feeds = [make_feed(f"t_host_{i}", host="busy.test", poll_seconds=300) for i in range(6)]
    for feed in feeds:
        server.serve(feed, rss([fresh(f"Headline {feed.id} about something")]))
    service = make_service(feeds, llm)
    await service.start()
    await service.poll(NOW)
    assert len(server.requests) == service_module.HOST_BUDGET_PER_MINUTE
    now_clock.advance(60)
    await service.poll(now_clock.now())
    assert len(server.requests) == 6
    assert all(server.count(feed) == 1 for feed in feeds)


async def test_conditional_get_through_the_service(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, now_clock: ManualClock
) -> None:
    feed = make_feed("t_etag")
    body = rss([fresh("Nvidia unveils new AI chip")])
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("If-None-Match"))
        if request.headers.get("If-None-Match") == '"v7"':
            return httpx.Response(304)
        return httpx.Response(200, content=body, headers={"ETag": '"v7"'})

    server.on(feed, handler)
    service = make_service([feed], llm)
    await service.start()
    assert await service.poll(NOW) == 1
    now_clock.advance(60)
    assert await service.poll(now_clock.now()) == 0
    assert seen == [None, '"v7"']
    assert service.feeds()[0].last_ok_ts == NOW_TS + 60


async def test_matcher_follows_changes_of_the_instrument_book(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, book: InstrumentBook,
    now_clock: ManualClock,
) -> None:
    feed = make_feed("t_book", quality=3)
    server.serve(feed, rss([fresh("Rheinmetall wins tank order from Italy")]))
    service = make_service([feed], llm)
    await service.start()
    await service.poll(NOW)
    assert statuses(db)["Rheinmetall wins tank order from Italy"] == ("skipped", "kein_bezug")
    repo_market.upsert_instruments(db, [Instrument("RHM.DE", "Rheinmetall AG", "equity", "EU", "XETR", "EUR",
                                                   timezone="Europe/Berlin")])
    book.reload()
    now_clock.advance(60)
    server.serve(feed, rss([fresh("Rheinmetall wins tank order from Italy"), fresh("Rheinmetall raises outlook")]))
    await service.poll(now_clock.now())
    await wait_for_analyses(service)
    assert by_title(db)["Rheinmetall raises outlook"].hinted_symbols == ["RHM.DE"]
    assert statuses(db)["Rheinmetall raises outlook"] == ("analyzed", "")


async def test_poll_never_raises_and_retries_after_a_storage_failure(
    make_service: ServiceFactory, server: FeedServer, llm: FakeLlm, db: Database, now_clock: ManualClock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feed = make_feed("t_db")
    server.serve(feed, rss([fresh("Nvidia unveils new AI chip")]))
    service = make_service([feed], llm)
    await service.start()
    original = repo_news.insert_news

    def broken(*args: object, **kwargs: object) -> list[NewsItem]:
        raise RuntimeError("disk full")

    monkeypatch.setattr(repo_news, "insert_news", broken)
    assert await service.poll(NOW) == 0
    assert service.status().last_poll_ts is None
    monkeypatch.setattr(repo_news, "insert_news", original)
    now_clock.advance(60)
    assert await service.poll(now_clock.now()) == 1  # same content again: nothing was lost
    assert server.count(feed) == 2


async def test_status_and_feeds_before_start_read_the_database(
    make_service: ServiceFactory, llm: FakeLlm, db: Database
) -> None:
    repo_news.save_feed_state(db, FeedState("t_alpha", last_ok_ts=NOW_TS - 60, fail_count=2, last_error="HTTP 503",
                                            items_total=42))
    service = make_service([make_feed("t_alpha"), make_feed("t_beta")], llm)
    feeds = {feed.feed_id: feed for feed in service.feeds()}
    alpha = feeds["t_alpha"]
    assert (alpha.fail_count, alpha.items_total, alpha.last_error) == (2, 42, "HTTP 503")
    assert feeds["t_beta"].last_ok_ts is None
    status = service.status()
    assert (status.feeds_total, status.feeds_failing, status.last_poll_ts) == (2, 1, None)


def test_factory_builds_a_news_service(
    settings: Settings, db: Database, now_clock: ManualClock, bus: EventBus, book: InstrumentBook, llm: FakeLlm
) -> None:
    news = build_news(settings=settings, db=db, clock=now_clock, bus=bus, instruments=book,
                      runtime_settings=SettingsStore(db), llm=llm)
    assert isinstance(news, NewsService) and isinstance(news, News)
    assert len(news.feeds()) == 147
