"""The agenda service: macro data, central-bank decisions and earnings dates with expectation and actual value
(implements ``contracts.Agenda``).

``refresh`` runs once a minute (own loop of the runtime) and does three things, each failing on its own:

1. ForexFactory weekly calendar, hourly (after a failure again in 10 minutes): the week's macro releases and
   central-bank decisions with importance, forecast and previous value. Entries that disappeared from the week
   (moved or cancelled) are deleted unless an actual value was recorded.
2. Actual values from the FinancialJuice headlines stored by the news service since the last scan: each is
   matched to the calendar entry of the same currency released within ``MATCH_WINDOW_S`` of the headline's
   publication time (``release_ts``: items fetched in a burst after a start carry the burst's time, not the
   release's) by title similarity and equal forecast / previous (``sources.best_match``). The surprise is actual minus forecast; with at least six
   earlier surprises of the same event it is also expressed in their root mean square (``surprise_z``). An
   actual value without a scheduled entry becomes an entry of importance 1. High-importance releases are
   written to the activity log.
3. Earnings dates from Yahoo (``MarketData.earnings``), ``EARNINGS_PER_REFRESH`` companies per minute: every
   active share every 12 hours, more often around a release (3 hours within two days before it, 30 minutes
   for three days after it until the reported EPS is known). Upcoming dates that disappeared are deleted.

What the agenda is used for: the signal engine blocks new entries shortly before important events of the
instrument (``rules.blocking_event``; preference ``event_block_enabled``) and hands upcoming events and recent
surprises to the trade review; the GUI shows the calendar.

Threads: database work runs in one dedicated worker thread; the event loop does the HTTP requests.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Final

import httpx

from trading_local.agenda import rules, sources
from trading_local.agenda.sources import PublishedValue
from trading_local.analysis.readings import de_number
from trading_local.clock import Clock, to_ts
from trading_local.contracts import MarketData
from trading_local.db import repo_agenda, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_agenda import AgendaEvent
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus
from trading_local.models import Instrument

log = logging.getLogger(__name__)

DAY_S: Final = 86_400
FOREXFACTORY_INTERVAL_S: Final = 3600
FOREXFACTORY_RETRY_S: Final = 600
MATCH_WINDOW_S: Final = 45 * 60
MAX_PUBLISH_LAG_S: Final = 12 * 3600  # a publication time further back than this is not believed
HEADLINE_LOOKBACK_S: Final = DAY_S  # the first scan after a start reads the headlines of the last day
HEADLINE_BATCH: Final = 500
EARNINGS_PER_REFRESH: Final = 2
EARNINGS_INTERVAL_S: Final = 12 * 3600
EARNINGS_SOON_S: Final = 2 * DAY_S
EARNINGS_SOON_INTERVAL_S: Final = 3 * 3600
EARNINGS_REPORT_WINDOW_S: Final = 3 * DAY_S
EARNINGS_REPORT_INTERVAL_S: Final = 1800
EARNINGS_HORIZON_S: Final = 400 * DAY_S  # upcoming earnings rows of Yahoo are replaced within this horizon
ACTIVITY_MIN_IMPORTANCE: Final = 3
USER_AGENT: Final = "trading-local/0.1 (economic calendar)"


def release_ts(published_ts: int | None, fetched_ts: int) -> int:
    """When a headline's value was released: the feed's publication time if it is plausible (at most
    ``MAX_PUBLISH_LAG_S`` before and five minutes after the time it was seen), otherwise the time it was seen.
    Items fetched in a burst after a start or an outage carry the time of the burst, not of the release."""
    if published_ts is not None and fetched_ts - MAX_PUBLISH_LAG_S <= published_ts <= fetched_ts + 300:
        return int(published_ts)
    return fetched_ts


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=20.0, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})


def release_message_de(event: AgendaEvent, value: PublishedValue, surprise_z: float | None) -> str:
    """Activity-log text of a released value, e.g. 'Konjunkturdaten USD: CPI m/m 0.4% (Prognose 0.3%) – über
    Erwartung (+1,8σ)'."""
    forecast = rules.parse_value(event.forecast_text) or value.forecast
    label = rules.SURPRISE_LABELS_DE[rules.surprise_label(value.actual, forecast)]
    expected = f" (Prognose {event.forecast_text or value.forecast_text})" if forecast is not None else ""
    text = f"Konjunkturdaten {event.currency}: {event.title} {value.actual_text}{expected}"
    if label:
        text += f" – {label}"
    if surprise_z is not None:
        text += f" ({de_number(surprise_z, 1, signed=True)}σ)"
    return text


class AgendaService:
    """Implements ``contracts.Agenda``. ``start`` / ``refresh`` / ``stop`` run on the event loop."""

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        bus: EventBus,
        instruments: InstrumentBook,
        data: MarketData,
        client_factory: Callable[[], httpx.AsyncClient] = make_client,
    ) -> None:
        self._db = db
        self._clock = clock
        self._bus = bus
        self._instruments = instruments
        self._data = data
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agenda")
        self._refresh_lock = asyncio.Lock()
        self._closed = False
        self._forexfactory_next_ts = 0
        self._last_news_id: int | None = None  # worker thread
        self._earnings_checked: dict[str, int] = {}

    async def start(self) -> None:
        if self._closed or self._client is not None:
            return
        try:
            self._client = self._client_factory()
        except Exception:
            log.exception("agenda: start failed")

    async def stop(self) -> None:
        self._closed = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await asyncio.to_thread(self._worker.shutdown, True)

    async def refresh(self, now: datetime) -> int:
        """Calendar, actual values and earnings dates (see the module docstring). Never raises; returns the
        number of rows written."""
        if self._closed or self._client is None or self._refresh_lock.locked():
            return 0
        async with self._refresh_lock:
            now_ts = to_ts(now)
            written = 0
            for step in (self._refresh_forexfactory, self._scan_headlines, self._refresh_earnings):
                try:
                    written += await step(now_ts)
                except Exception:
                    if self._closed:
                        return written
                    log.exception("agenda: %s failed", step.__name__)
            return written

    # -- 1. ForexFactory ------------------------------------------------------------------------
    async def _refresh_forexfactory(self, now_ts: int) -> int:
        client = self._client
        if client is None or now_ts < self._forexfactory_next_ts:
            return 0
        try:
            response = await client.get(sources.FOREXFACTORY_URL)
            response.raise_for_status()
            events = sources.parse_forexfactory(response.json(), now_ts=now_ts)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("agenda: ForexFactory calendar unavailable: %s", exc)
            self._forexfactory_next_ts = now_ts + FOREXFACTORY_RETRY_S
            return 0
        self._forexfactory_next_ts = now_ts + FOREXFACTORY_INTERVAL_S
        return await self._in_worker(self._store_week, events)

    def _store_week(self, events: list[AgendaEvent]) -> int:
        if not events:
            return 0
        written = repo_agenda.upsert_events(self._db, events)
        removed = repo_agenda.delete_unlisted(
            self._db,
            source=sources.SOURCE_FOREXFACTORY,
            start_ts=min(event.event_ts for event in events),
            end_ts=max(event.event_ts for event in events) + 1,
            keep_uids={event.uid for event in events},
        )
        if removed:
            log.info("agenda: %d calendar entries moved or cancelled", removed)
        return written

    # -- 2. actual values -----------------------------------------------------------------------
    async def _scan_headlines(self, now_ts: int) -> int:
        return await self._in_worker(self._scan_headlines_blocking, now_ts)

    def _scan_headlines_blocking(self, now_ts: int) -> int:
        newest = self._db.scalar("SELECT MAX(id) FROM news", default=0)
        since = self._last_news_id
        if since is None:
            since = self._db.scalar(
                "SELECT MAX(id) FROM news WHERE fetched_ts < ?", (now_ts - HEADLINE_LOOKBACK_S,), default=0
            )
        rows = self._db.query(
            "SELECT id, title, fetched_ts, published_ts FROM news WHERE id > ? AND id <= ? AND source = ? "
            "AND instr(title, 'Actual') > 0 ORDER BY id LIMIT ?",
            (since, newest, sources.FINANCIALJUICE_FEED, HEADLINE_BATCH),
        )
        written = 0
        for row in rows:
            fetched = int(row["fetched_ts"])
            try:
                written += self._apply_headline(
                    int(row["id"]), str(row["title"]), fetched, release_ts(row["published_ts"], fetched)
                )
            except Exception:
                log.exception("agenda: headline %s could not be applied", row["id"])
        self._last_news_id = int(rows[-1]["id"]) if len(rows) == HEADLINE_BATCH else int(newest)
        return written

    def _apply_headline(self, news_id: int, headline: str, ts: int, released_ts: int) -> int:
        """Apply one headline seen at ``ts`` (the as-of time of its value) that was published at ``released_ts``."""
        value = sources.parse_actual_headline(headline)
        if value is None:
            return 0
        candidates = repo_agenda.events_between(
            self._db,
            start_ts=released_ts - MATCH_WINDOW_S,
            end_ts=released_ts + MATCH_WINDOW_S,
            kinds=("macro", "central_bank"),
        )
        event = sources.best_match(value, candidates)
        if event is None:
            if any(
                other.actual is not None
                and (not value.currency or other.currency == value.currency)
                and rules.title_similarity(value.title, other.title) >= 0.5
                for other in candidates
            ):
                return 0  # the same release posted again (revision, other wording)
            new = sources.event_from_headline(value, news_id=news_id, ts=ts, released_ts=released_ts)
            new.surprise_z = rules.standardised(
                new.surprise,
                repo_agenda.earlier_surprises(self._db, currency=new.currency, title=new.title, before_ts=new.event_ts),
            )
            return repo_agenda.upsert_events(self._db, [new])
        if event.actual is not None or event.id is None:
            return 0
        forecast = rules.parse_value(event.forecast_text) or value.forecast
        diff = rules.surprise(value.actual, forecast)
        z = rules.standardised(
            diff,
            repo_agenda.earlier_surprises(self._db, currency=event.currency, title=event.title, before_ts=event.event_ts),
        )
        stored = repo_agenda.set_actual(
            self._db,
            event.id,
            actual=value.actual.number,
            actual_text=value.actual_text,
            actual_ts=ts,
            surprise=diff,
            surprise_z=z,
            unit=value.actual.unit,
        )
        if stored and event.importance >= ACTIVITY_MIN_IMPORTANCE:
            try:
                repo_ops.log_activity(
                    self._db, ts=ts, level="info", category="news", message=release_message_de(event, value, z),
                    bus=self._bus,
                )  # fmt: skip
            except Exception:
                log.exception("agenda: activity entry of %s failed", event.uid)
        return int(stored)

    # -- 3. earnings ----------------------------------------------------------------------------
    async def _refresh_earnings(self, now_ts: int) -> int:
        written = 0
        for instrument in await self._in_worker(self._due_companies, now_ts):
            if self._closed:
                break
            self._earnings_checked[instrument.symbol] = now_ts
            releases = await self._data.earnings(instrument.symbol)
            if releases is None:
                continue  # failed: tried again at the next regular check
            events = sources.earnings_events(instrument, releases, now_ts=now_ts)
            written += await self._in_worker(self._store_earnings, instrument.symbol, events, now_ts)
        return written

    def _due_companies(self, now_ts: int) -> list[Instrument]:
        """The shares whose earnings dates are due for a check, most overdue first."""
        nearby = repo_agenda.events_between(
            self._db, start_ts=now_ts - EARNINGS_REPORT_WINDOW_S, end_ts=now_ts + EARNINGS_SOON_S, kinds=("earnings",)
        )
        unreported = {event.symbol for event in nearby if event.event_ts <= now_ts and event.actual is None}
        upcoming = {event.symbol for event in nearby if event.event_ts > now_ts}
        due: list[tuple[int, int, str, Instrument]] = []
        for instrument in self._instruments.all(active_only=True):
            if instrument.asset_class != "equity":
                continue
            checked = self._earnings_checked.get(instrument.symbol)
            if checked is None:
                due.append((0, instrument.sort_order, instrument.symbol, instrument))
                continue
            interval = EARNINGS_INTERVAL_S
            if instrument.symbol in upcoming:
                interval = EARNINGS_SOON_INTERVAL_S
            if instrument.symbol in unreported:
                interval = EARNINGS_REPORT_INTERVAL_S
            if checked + interval <= now_ts:
                due.append((checked + interval, instrument.sort_order, instrument.symbol, instrument))
        return [item[3] for item in sorted(due, key=lambda item: item[:3])[:EARNINGS_PER_REFRESH]]

    def _store_earnings(self, symbol: str, events: list[AgendaEvent], now_ts: int) -> int:
        written = repo_agenda.upsert_events(self._db, events)
        repo_agenda.delete_unlisted(
            self._db,
            source=sources.SOURCE_YAHOO,
            start_ts=now_ts - DAY_S,
            end_ts=now_ts + EARNINGS_HORIZON_S,
            keep_uids={event.uid for event in events},
            symbol=symbol,
        )
        return written

    async def _in_worker[T](self, func: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._worker, functools.partial(func, *args))
