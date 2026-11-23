"""Helpers of the news tests (offline: recorded feed bodies and ``httpx.MockTransport``)."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import httpx

from trading_local.clock import ManualClock, from_ts
from trading_local.contracts import TradeReview, TradeReviewContext
from trading_local.dto import LlmStatusOut
from trading_local.events import Event, Subscription
from trading_local.models import ChartAnalysis, Instrument, NewsAnalysis, NewsItem
from trading_local.news.feeds import Feed
from trading_local.news.fetcher import make_client
from trading_local.news.service import NewsService

FIXTURES = Path(__file__).parent / "fixtures"
# Sunday 2026-11-22 19:00 UTC: shortly after the feed fixtures were recorded.
NOW = datetime(2026, 11, 22, 19, 0, tzinfo=UTC)
NOW_TS = int(NOW.timestamp())


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def universe() -> list[Instrument]:
    """The 109 instruments of the default universe (symbol, name, asset class; venue data is irrelevant here)."""
    rows = json.loads((FIXTURES / "universe.json").read_text(encoding="utf-8"))
    return [Instrument(symbol, name, asset_class, "US", "XNYS", "USD") for symbol, name, asset_class in rows]


def make_feed(feed_id: str, *, host: str | None = None, quality: int = 3, poll_seconds: int = 60, **extra: Any) -> Feed:
    """A feed on a fake host: ``https://<host>/<feed_id>.xml`` (default host derived from the id)."""
    url = f"https://{host or feed_id.replace('_', '-') + '.test'}/{feed_id}.xml"
    return Feed(feed_id, f"Feed {feed_id}", url, "GLOBAL", "en", "markets", quality, poll_seconds, **extra)


@dataclass(slots=True)
class Entry:
    """One RSS item for ``rss``."""

    title: str
    published_ts: int | None = None
    link: str | None = None
    summary: str = ""
    guid: str | None = None


def rss(entries: Iterable[Entry], *, channel: str = "Test feed") -> bytes:
    """A minimal RSS 2.0 document."""
    parts = [f'<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel><title>{escape(channel)}</title>']
    for entry in entries:
        parts.append(f"<item><title>{escape(entry.title)}</title>")
        if entry.link is not None:
            parts.append(f"<link>{escape(entry.link)}</link>")
        if entry.guid is not None:
            parts.append(f"<guid>{escape(entry.guid)}</guid>")
        if entry.summary:
            parts.append(f"<description>{escape(entry.summary)}</description>")
        if entry.published_ts is not None:
            parts.append(f"<pubDate>{format_datetime(from_ts(entry.published_ts), usegmt=True)}</pubDate>")
        parts.append("</item>")
    parts.append("</channel></rss>")
    return "".join(parts).encode("utf-8")


def fresh(title: str, *, minutes_ago: int = 5, summary: str = "") -> Entry:
    """An item published shortly before ``NOW`` with a link derived from the title."""
    slug = "".join(ch if ch.isalnum() else "-" for ch in title.lower())[:90]
    return Entry(title, NOW_TS - minutes_ago * 60, f"https://news.test/{slug}", summary)


# ------------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------------
type Handler = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


class FeedServer:
    """Serves feed documents by URL through ``httpx.MockTransport`` and records every request."""

    def __init__(self) -> None:
        self.bodies: dict[str, bytes] = {}
        self.handlers: dict[str, Handler] = {}
        self.requests: list[httpx.Request] = []

    def serve(self, feed: Feed, body: bytes) -> None:
        self.bodies[feed.url] = body

    def on(self, feed: Feed, handler: Handler) -> None:
        self.handlers[feed.url] = handler

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url in self.handlers:
            response = self.handlers[url](request)
            return await response if isinstance(response, Awaitable) else response
        if url in self.bodies:
            return httpx.Response(200, content=self.bodies[url], headers={"content-type": "application/rss+xml"})
        return httpx.Response(404)

    def count(self, feed: Feed) -> int:
        return sum(1 for request in self.requests if str(request.url) == feed.url)

    def client(self) -> httpx.AsyncClient:
        return make_client(httpx.MockTransport(self))


# ------------------------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------------------------
@dataclass
class FakeLlm:
    """``contracts.Llm`` with canned news analyses.

    ``gate``: analyses wait until the event is set. ``relevance``: by title substring (first match wins,
    default 3). ``unavailable`` returns None, ``fail`` raises, ``extra_symbol`` names a non-candidate.
    """

    clock: ManualClock
    gate: asyncio.Event | None = None
    relevance: Mapping[str, int] = field(default_factory=dict)
    unavailable: bool = False
    fail: bool = False
    extra_symbol: str | None = None
    calls: list[tuple[NewsItem, list[str]]] = field(default_factory=list)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> LlmStatusOut:
        return LlmStatusOut(model="fake")

    async def analyze_news(self, item: NewsItem, candidates: Sequence[Instrument]) -> NewsAnalysis | None:
        self.calls.append((item, [instrument.symbol for instrument in candidates]))
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("model crashed")
        if self.unavailable:
            return None
        assert item.id is not None
        symbols = {instrument.symbol: 0.5 for instrument in candidates}
        if self.extra_symbol:
            symbols[self.extra_symbol] = -0.5
        relevance = next((value for key, value in self.relevance.items() if key in item.title), 3)
        return NewsAnalysis(
            news_id=item.id,
            analyzed_ts=self.clock.ts(),
            model="fake",
            relevance=relevance,
            sentiment=0.4,
            category="other",
            horizon="days",
            summary_de="Kurzfassung",
            confidence=0.7,
            symbols=symbols,
            latency_ms=5,
        )

    async def review_trade(self, ctx: TradeReviewContext, *, timeout_s: float) -> TradeReview:
        return TradeReview(verdict="unavailable")

    async def analyze_chart(
        self, instrument: Instrument, timeframe: str, chart_png: bytes, facts: Mapping[str, Any]
    ) -> ChartAnalysis | None:
        return None


# ------------------------------------------------------------------------------------------------
# Waiting
# ------------------------------------------------------------------------------------------------
async def wait_for_analyses(service: NewsService, timeout_s: float = 2.0) -> None:
    """Wait until the service has no queued or running analysis left."""
    async with asyncio.timeout(timeout_s):
        while service._waiting or service._running:  # noqa: SLF001 - test helper
            await asyncio.sleep(0.005)
    await asyncio.sleep(0.02)  # let publications from the worker thread reach the loop


async def drain(subscription: Subscription) -> list[Event]:
    """All events delivered so far (waits briefly for events published from the worker thread)."""
    events: list[Event] = []
    while True:
        try:
            events.append(await asyncio.wait_for(subscription.get(), timeout=0.05))
        except TimeoutError:
            return events
