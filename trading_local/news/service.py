"""The news service: polls the feeds, stores new items, matches them to instruments and hands the relevant ones
to the LLM in the background (implements ``contracts.News``).

One poll round (``poll``, once per minute)
1. Select the feeds that are due (``FeedState.next_poll_ts``), most overdue first, at most
   ``HOST_BUDGET_PER_MINUTE`` requests per host and minute; fetch them concurrently with one request
   at a time per host. A host that times out is skipped for the rest of the round. Fetches still running
   after ``ROUND_DEADLINE_S`` are cancelled and stay due, so one slow host never holds the round.
2. In the news worker thread: match every unseen item to instruments and macro topics, mark near-duplicates
   (``NearDupIndex``, grouped by the matched symbols), store the new items (``repo_news.insert_news`` also
   flags same-title duplicates) and persist the feed states - all in one transaction.
3. Decide per new non-duplicate item: published more than 12 h ago -> skipped 'alt' (a first start must not
   flood the LLM); no instrument and no macro match from a feed with quality < 4 -> skipped 'kein_bezug';
   LLM switched off -> skipped 'llm_aus'; otherwise 'queued' (announced on TOPIC_NEWS right away).
4. Queued items are analysed in background tasks; ``poll`` never waits for them.

Priority: every waiting item has an urgency class (``_urgency``), most urgent first:
  0 the item names an instrument whose price has just moved unusually (``prioritize``, the price trigger),
  1 an ad-hoc disclosure (press-release feed with quality >= 4) naming a tradable instrument,
  2 the item names a tradable instrument and comes from a feed with quality >= 3,
  3 everything else that is analysed (weak feeds, indices or FX only, unmatched items of good feeds).
Within a class fresher items come first. The class is re-evaluated whenever an item is picked, so a price
trigger moves items that are already waiting to the front.

Back-pressure: at most ``MAX_OUTSTANDING`` analyses are outstanding (waiting + handed to the LLM). When full,
the LEAST important waiting item is dropped (skipped 'rueckstau'): the highest class, and within it the oldest
item; a newcomer that would itself be the least important one is dropped instead. Only items that have not
been handed to the LLM yet can be dropped, so no model work is wasted. Waiting items are handed over in
priority order, at most ``LLM_DISPATCH_LIMIT`` at a time (see there). An item that was announced as 'queued'
and is given up later (dropped, LLM unavailable, failed) is announced again with its final status, so the GUI
never shows a pending analysis that will not come.

Price trigger (``prioritize``): the instruments of an unusual price move get urgency 0 for ``BOOST_S``;
items naming them that were dropped from the backlog or that the LLM could not take within the last
``RETRY_WINDOW_S`` are queued again.

Time semantics: the as-of time of an item is ``fetched_ts``: the clock time at which its feed's response had
arrived (first seen; never earlier than the poll time). ``published_ts`` is display data.

Threads: feed parsing, matching, the near-duplicate index and all database work run in ONE worker thread (one
database connection, no locks around the in-memory indexes, never more than one CPU-bound news thread competing
for the GIL); the event loop only does network I/O and bookkeeping. ``status`` and ``feeds`` may be called from
any thread.
"""

import asyncio
import functools
import logging
import threading
from collections import Counter, deque
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Final

import httpx

from trading_local import presenters
from trading_local.clock import Clock, to_ts
from trading_local.config import Settings
from trading_local.contracts import Llm
from trading_local.db import repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_news import FeedState
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import FeedStateOut, NewsStatusOut
from trading_local.events import TOPIC_NEWS, EventBus
from trading_local.models import Instrument, NewsAnalysis, NewsItem, NewsStatus
from trading_local.news.feeds import FEEDS, Feed
from trading_local.news.fetcher import FeedHints, FetchResult, HostLimiter, fetch_feed, make_client
from trading_local.news.matcher import AliasMatcher, NearDupIndex

log = logging.getLogger(__name__)

MAX_OUTSTANDING: Final = 40  # analyses waiting or handed to the LLM
# Analyses handed to the LLM at the same time. The LLM queue serves news newest-first (``llm.jobs``), so every
# item handed over later overtakes the ones handed over before it: with 8 in flight, the 7 best items of a
# flood waited behind every later dispatch until their 20 minutes ran out. With 2, one item runs while the
# next waits, the GPU never idles between news, and the order of this queue is the order of the analyses.
LLM_DISPATCH_LIMIT: Final = 2
# urgency classes (smaller = more urgent), see the module docstring
URGENCY_TRIGGERED: Final = 0
URGENCY_DISCLOSURE: Final = 1
URGENCY_TRADABLE: Final = 2
URGENCY_OTHER: Final = 3
DISCLOSURE_MIN_QUALITY: Final = 4  # press-release feeds this good are the ad-hoc disclosure feeds
TRADABLE_MIN_QUALITY: Final = 3
BOOST_S: Final = 3600  # a price trigger keeps the news of its instrument at urgency 0 this long
RETRY_WINDOW_S: Final = 6 * 3600  # dropped items fetched this recently get another chance on a price trigger
ANALYSIS_TIMEOUT_S: Final = 900.0  # safety net: an analysis that never returns frees its slot
MAX_ITEM_AGE_S: Final = 12 * 3600  # older publications are stored but not analysed
MIN_QUALITY_WITHOUT_MATCH: Final = 4  # items without any match are analysed only from feeds this good
IMPORTANT_RELEVANCE: Final = 4  # analyses at least this relevant are written to the activity log
RECOVERY_WINDOW_S: Final = 2 * 3600  # pending items younger than this are re-queued after a restart
RECOVERY_SCAN_S: Final = 30 * 86_400  # older pending items within this horizon are closed as 'alt'
NEAR_DUP_WINDOW_S: Final = 48 * 3600
HOST_BUDGET_PER_MINUTE: Final = 4  # requests per host and minute: every host stays under 5
# Requests younger than this count against the host budget. Shorter than the poll interval, so that a previous
# round that started a little late never blocks the current one; rounds in quick succession share the budget.
HOST_WINDOW_S: Final = 50
FETCH_CONCURRENCY: Final = 8
# Fetches still running after this many seconds are cancelled (their feeds stay due). The runtime polls once a
# minute and awaits the poll: without a bound, four slow feeds on one host (25 s request deadline each, one at
# a time) would hold every other feed's items for 100 s and make the next poll skip.
ROUND_DEADLINE_S: Final = 40.0
DUE_TOLERANCE_S: Final = 10  # the poll runs once a minute: a feed due within 10 s is polled now
PRUNE_INTERVAL_S: Final = 600
SEEN_UID_TTL_S: Final = 3 * 86_400  # uids not seen in any parsed feed body for this long are forgotten
REBUILD_LIMIT: Final = 50_000  # stored headlines read at start-up to rebuild the near-duplicate index
ACTIVITY_TITLE_CHARS: Final = 200

# status_reason codes of skipped items (stored, shown in the GUI)
REASON_OLD: Final = "alt"
REASON_NO_MATCH: Final = "kein_bezug"
REASON_LLM_OFF: Final = "llm_aus"
REASON_LLM_UNAVAILABLE: Final = "llm_nicht_verfuegbar"
REASON_BACKLOG: Final = "rueckstau"
REASON_ANALYSIS_FAILED: Final = "analysefehler"

type _StatusUpdate = tuple[NewsItem, NewsStatus, str]


def _freshness(item: NewsItem) -> tuple[int, int, int]:
    """Sort key: larger = fresher (first seen later, then published later, then stored later)."""
    return item.fetched_ts, item.published_ts or 0, item.id or 0


def _stored_id(item: NewsItem) -> int:
    """The database id of an item; every item the service queues or updates has been stored."""
    if item.id is None:
        raise ValueError("news item has not been stored")
    return item.id


class NewsService:
    """Implements ``contracts.News``. Create it with ``build_news``; ``start`` / ``stop`` bracket its life."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        instruments: InstrumentBook,
        runtime_settings: SettingsStore,
        llm: Llm,
        feeds: Sequence[Feed] = FEEDS,
        client_factory: Callable[[], httpx.AsyncClient] = make_client,
        host_gap_s: float = 1.0,
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._bus = bus
        self._book = instruments
        self._runtime_settings = runtime_settings
        self._llm = llm
        self._feeds = {feed.id: feed for feed in feeds}
        self._source_names = {feed.id: feed.name for feed in feeds}
        self._client_factory = client_factory
        self._client: httpx.AsyncClient | None = None
        self._limiter = HostLimiter(host_gap_s)
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="news-worker")
        self._ready = False
        self._recovered = False
        self._ready_lock = asyncio.Lock()
        self._poll_lock = asyncio.Lock()
        self._closed = False
        self._last_poll_ts: int | None = None
        # polling state (event loop writes, any thread reads under the lock)
        self._lock = threading.Lock()
        self._states: dict[str, FeedState] = {}
        self._states_loaded = False
        self._body_hashes: dict[str, str] = {}
        self._host_requests: dict[str, deque[int]] = {}
        self._fetches: set[asyncio.Task[tuple[Feed, FetchResult] | None]] = set()  # the running round (loop only)
        # analysis queue (event loop only)
        self._waiting: dict[int, NewsItem] = {}
        self._running: dict[int, asyncio.Task[None]] = {}
        self._boosts: dict[str, int] = {}  # symbol -> end of its price-trigger priority (epoch seconds)
        # worker-thread state
        self._matcher: AliasMatcher | None = None
        self._matcher_key: tuple[Instrument, ...] = ()
        self._neardup = NearDupIndex(window_s=NEAR_DUP_WINDOW_S)
        self._seen: dict[str, int] = {}  # uid -> last time it was seen in a parsed feed body
        self._last_prune_ts = 0

    # -- life cycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        """Load the feed states, rebuild the in-memory indexes from the last 48 h and re-queue the analyses
        that were pending at shutdown (fetched within the last 2 h, newest first). Never raises: whatever
        failed is retried by the next ``poll``. Does nothing when news are switched off (``enable_news``): the
        runtime then never polls, and no analysis may be started either."""
        if not self._settings.enable_news:
            return
        try:
            await self._ensure_ready()
        except Exception:
            log.exception("news: start-up failed; polling will retry the initialisation")

    async def stop(self) -> None:
        """Cancel the fetches of a running poll and the background analyses (their items stay 'queued' for the
        next start), close the HTTP client and wait for pending database writes."""
        self._closed = True
        tasks = [*self._fetches, *self._running.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._running.clear()
        self._waiting.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await asyncio.to_thread(self._worker.shutdown, True)

    async def _ensure_ready(self) -> None:
        """Initialise once (feed states, indexes, HTTP client), then recover the pending analyses once. A step
        that fails raises and is repeated by the next call."""
        async with self._ready_lock:
            if not self._ready:
                now_ts = self._clock.ts()
                stored = await self._in_worker(repo_news.list_feed_states, self._db)
                with self._lock:
                    self._states = {feed_id: stored.get(feed_id) or FeedState(feed_id) for feed_id in self._feeds}
                    self._states_loaded = True
                await self._in_worker(self._rebuild_indexes, now_ts)
                if self._client is None:
                    self._client = self._client_factory()
                self._ready = True
            if not self._recovered:
                await self._recover()
                self._recovered = True

    async def _recover(self) -> None:
        now_ts = self._clock.ts()
        pending = await self._in_worker(
            repo_news.pending_news, self._db, now_ts=now_ts, max_age_s=RECOVERY_SCAN_S, limit=5000
        )
        recent = [item for item in pending if item.fetched_ts >= now_ts - RECOVERY_WINDOW_S]
        stale: list[_StatusUpdate] = [
            (item, "skipped", REASON_OLD) for item in pending if item.fetched_ts < now_ts - RECOVERY_WINDOW_S
        ]
        await self._decide_and_queue(recent, now_ts, announce=False, extra=stale)
        if pending:
            log.info("news: %d pending analyses re-queued, %d closed as too old", len(recent), len(stale))

    # -- price trigger --------------------------------------------------------------------------
    async def prioritize(self, symbols: Collection[str], now: datetime) -> int:
        """Pull the news of ``symbols`` forward after an unusual price move (see the module docstring).

        Waiting and newly arriving items that name one of the symbols get urgency 0 for ``BOOST_S``; items
        naming them that were dropped from the backlog ('rueckstau') or that the LLM could not take
        ('llm_nicht_verfuegbar') within ``RETRY_WINDOW_S`` are queued again. Unknown symbols are ignored.
        Never raises; returns the number of items queued again.
        """
        if self._closed or not self._settings.enable_news:
            return 0
        try:
            now_ts = to_ts(now)
            wanted = sorted({symbol for symbol in symbols if self._book.get(symbol) is not None})
            if not wanted:
                return 0
            for symbol in wanted:
                self._boosts[symbol] = max(self._boosts.get(symbol, 0), now_ts + BOOST_S)
            self._boosts = {symbol: until for symbol, until in self._boosts.items() if until > now_ts}
            if not self._ready:  # not initialised yet: the boost applies, there is nothing to re-queue
                self._dispatch()
                return 0
            retry = await self._in_worker(
                repo_news.skipped_news_for,
                self._db,
                wanted,
                since_ts=now_ts - RETRY_WINDOW_S,
                reasons=(REASON_BACKLOG, REASON_LLM_UNAVAILABLE),
            )
            admitted = await self._decide_and_queue(retry, now_ts, announce=True)
            if admitted:
                log.info("news: price trigger %s: %d item(s) queued again", ", ".join(wanted), admitted)
            return admitted
        except Exception:
            log.exception("news: prioritising %s failed", ", ".join(sorted(symbols)))
            return 0

    # -- polling --------------------------------------------------------------------------------
    async def poll(self, now: datetime) -> int:
        """Fetch the due feeds, store new items and queue their analyses. Never raises; returns the number of
        new items stored (duplicates included). Does not wait for the analyses."""
        if self._closed:
            return 0
        if self._poll_lock.locked():
            log.warning("news: the previous poll is still running; this one is skipped")
            return 0
        async with self._poll_lock:
            try:
                return await self._poll(to_ts(now))
            except Exception:
                if self._closed:  # stop() ran meanwhile: client and worker are gone, unsaved items come again
                    log.info("news: poll interrupted by stop")
                else:
                    log.exception("news: poll failed")
                return 0

    async def _poll(self, now_ts: int) -> int:
        await self._ensure_ready()
        due = self._due_feeds(now_ts)
        results = await self._fetch_round(due, now_ts)
        stored, states = await self._in_worker(self._ingest, results, now_ts)
        # Polling state is taken over only now: if storing failed, the next poll fetches the same content again.
        with self._lock:
            for state in states:
                self._states[state.feed_id] = state
        for feed, result in results:
            if result.body_hash is not None:
                self._body_hashes[feed.id] = result.body_hash
        await self._decide_and_queue([item for item in stored if item.status != "duplicate"], now_ts, announce=True)
        self._last_poll_ts = now_ts
        return len(stored)

    def _due_feeds(self, now_ts: int) -> list[Feed]:
        """Enabled feeds whose next poll is due, most overdue first (tier 1 and better feeds first on ties),
        within the per-host request budget; feeds over the budget stay due for the next round."""
        with self._lock:
            states = dict(self._states)
        next_poll = {feed_id: (states.get(feed_id) or FeedState(feed_id)).next_poll_ts for feed_id in self._feeds}
        candidates = sorted(
            (feed for feed in self._feeds.values() if feed.enabled and next_poll[feed.id] <= now_ts + DUE_TOLERANCE_S),
            key=lambda feed: (next_poll[feed.id], feed.tier, -feed.quality, feed.id),
        )
        due: list[Feed] = []
        for feed in candidates:
            window = self._host_requests.setdefault(feed.host, deque())
            while window and window[0] <= now_ts - HOST_WINDOW_S:
                window.popleft()
            if len(window) < HOST_BUDGET_PER_MINUTE:
                window.append(now_ts)
                due.append(feed)
        return due

    async def _fetch_round(self, due: Sequence[Feed], now_ts: int) -> list[tuple[Feed, FetchResult]]:
        """Fetch the due feeds concurrently (one request at a time per host) and return the results.

        Fetches still running after ``ROUND_DEADLINE_S`` are cancelled; their feeds keep their state and stay
        due, so they come first in the next round. Every item is stamped with the clock time at which its
        feed's response had arrived (at least ``now_ts``). Cancelling the poll cancels all fetches.
        """
        client = self._client
        if client is None:
            raise RuntimeError("news service is not ready")
        if not due:
            return []
        with self._lock:
            states = {feed.id: self._states.get(feed.id) or FeedState(feed.id) for feed in due}
        semaphore = asyncio.Semaphore(FETCH_CONCURRENCY)
        unreachable: set[str] = set()

        async def fetch(feed: Feed) -> tuple[Feed, FetchResult] | None:
            async with self._limiter.slot(feed.host):
                if feed.host in unreachable:  # stays due: retried next round
                    return None
                async with semaphore:
                    result = await fetch_feed(
                        client,
                        feed,
                        states[feed.id],
                        now_ts,
                        body_hash=self._body_hashes.get(feed.id),
                        parse_executor=self._worker,
                    )
                if result.unreachable:
                    unreachable.add(feed.host)
                received_ts = max(now_ts, self._clock.ts())
                for item in result.items:
                    item.fetched_ts = received_ts
                return feed, result

        tasks = {asyncio.create_task(fetch(feed), name=f"news-fetch-{feed.id}"): feed for feed in due}
        self._fetches.update(tasks)
        try:
            await asyncio.wait(tasks, timeout=ROUND_DEADLINE_S)
        finally:
            late = [task for task in tasks if not task.done()]
            for task in late:
                task.cancel()
            if late:
                await asyncio.gather(*late, return_exceptions=True)
            self._fetches.difference_update(tasks)
        results: list[tuple[Feed, FetchResult]] = []
        for task, feed in tasks.items():
            if task.cancelled():
                log.info("news: feed %s did not finish within the poll round; it stays due", feed.id)
            elif (error := task.exception()) is not None:
                log.error("news: fetching feed %s failed unexpectedly: %r", feed.id, error)
            elif (outcome := task.result()) is not None:
                results.append(outcome)
        return results

    # -- worker thread: matching and storing ----------------------------------------------------
    def _ingest(
        self, results: Sequence[tuple[Feed, FetchResult]], now_ts: int
    ) -> tuple[list[NewsItem], list[FeedState]]:
        """Match, de-duplicate and store the fetched items; persist the feed states. Returns the stored items
        (in input order, duplicates included) and the new feed states. Runs in the worker thread."""
        if not results:
            return [], []
        matcher = self._current_matcher()
        unseen: list[NewsItem] = []
        uids_in_batch: set[str] = set()
        for _, result in results:
            for item in result.items:
                if item.uid in uids_in_batch:
                    continue
                uids_in_batch.add(item.uid)
                if item.uid in self._seen:
                    self._seen[item.uid] = now_ts
                    continue
                hints = result.hints.get(item.uid, FeedHints())
                item.hinted_symbols = matcher.analyse(
                    item.title, item.summary, hint_tickers=hints.tickers, hint_isins=hints.isins
                ).hinted
                unseen.append(item)
        stored: list[NewsItem] = []
        added: list[int] = []
        per_feed: Counter[str] = Counter()
        try:
            with self._db.write():
                for item in unseen:
                    group = frozenset(item.hinted_symbols)
                    original = self._neardup.find(item.title, item.fetched_ts, group=group, source=item.source)
                    if original is not None:
                        item.status, item.duplicate_of = "duplicate", original
                    if not repo_news.insert_news(self._db, [item]):
                        continue  # known uid (stored before the seen-cache horizon)
                    stored.append(item)
                    per_feed[item.source] += 1
                    ref = item.duplicate_of or _stored_id(item)
                    added.append(self._neardup.add(item.title, item.fetched_ts, ref, group=group, source=item.source))
                states = [
                    replace(result.state, items_total=result.state.items_total + per_feed[feed.id])
                    for feed, result in results
                ]
                for state in states:
                    repo_news.save_feed_state(self._db, state)
        except BaseException:
            for key in added:  # the transaction rolled back: forget its headlines
                self._neardup.discard(key)
            raise
        for item in unseen:
            self._seen[item.uid] = now_ts
        self._prune(now_ts)
        return stored, states

    def _current_matcher(self) -> AliasMatcher:
        """The matcher for the current instrument book (rebuilt whenever the book changed)."""
        instruments = tuple(self._book.all(active_only=False))
        if self._matcher is None or instruments != self._matcher_key:
            self._matcher = AliasMatcher(instruments)
            self._matcher_key = instruments
            log.info("news: alias matcher built for %d active instruments", sum(i.active for i in instruments))
        return self._matcher

    def _rebuild_indexes(self, now_ts: int) -> None:
        """Seed the seen-uid cache and the near-duplicate index from the headlines of the last 48 hours."""
        rows = repo_news.list_news(
            self._db, since_ts=now_ts - NEAR_DUP_WINDOW_S, include_duplicates=True, limit=REBUILD_LIMIT
        )
        for item, _ in reversed(rows):  # oldest first
            self._seen[item.uid] = now_ts
            self._neardup.add(
                item.title,
                item.fetched_ts,
                item.duplicate_of or _stored_id(item),
                group=frozenset(item.hinted_symbols),
                source=item.source,
            )
        self._last_prune_ts = now_ts
        log.info("news: near-duplicate index rebuilt from %d stored headlines", len(rows))

    def _prune(self, now_ts: int) -> None:
        if now_ts - self._last_prune_ts < PRUNE_INTERVAL_S:
            return
        self._last_prune_ts = now_ts
        self._neardup.prune(now_ts)
        cutoff = now_ts - SEEN_UID_TTL_S
        self._seen = {uid: seen_ts for uid, seen_ts in self._seen.items() if seen_ts >= cutoff}

    def _write_statuses(self, updates: Sequence[_StatusUpdate]) -> None:
        with self._db.write():
            for item, status, reason in updates:
                repo_news.set_news_status(self._db, _stored_id(item), status, reason)

    def _store_analysis(self, item: NewsItem, analysis: NewsAnalysis) -> None:
        """Persist the analysis, then announce it (and log important news). Runs in the worker thread.

        The announcements come after the commit and cannot undo it: a failing activity-log write is logged,
        the item stays 'analyzed'.
        """
        repo_news.save_analysis(self._db, analysis)
        self._announce(replace(item, status="analyzed", status_reason=""), analysis)
        if analysis.relevance < IMPORTANT_RELEVANCE:
            return
        try:
            repo_ops.log_activity(
                self._db,
                ts=self._clock.ts(),
                level="info",
                category="news",
                message=f"Wichtige Nachricht (Relevanz {analysis.relevance}): {item.title[:ACTIVITY_TITLE_CHARS]}",
                ref_type="news",
                ref_id=item.id,
                bus=self._bus,
            )
        except Exception:
            log.exception("news: item %d is analysed, but its activity-log entry failed", item.id)

    # -- selection and back-pressure (event loop) -----------------------------------------------
    async def _decide_and_queue(
        self, items: Sequence[NewsItem], now_ts: int, *, announce: bool, extra: Sequence[_StatusUpdate] = ()
    ) -> int:
        """Apply the selection policy, admit the selected items to the analysis queue (dropping the least
        important waiting ones when full), persist all status changes and start analyses. Returns the
        number of items admitted."""
        llm_on = self._llm_enabled()
        updates = list(extra)
        selected: list[NewsItem] = []
        for item in items:
            reason = self._skip_reason(item, now_ts, llm_on)
            if reason is None:
                selected.append(item)
            else:
                updates.append((item, "skipped", reason))
        admitted, dropped = self._admit(selected)
        given_up = [item for item in dropped if item.status == "queued"]  # waiting before: announced or listed
        updates += [(item, "queued", "") for item in admitted]
        updates += [(item, "skipped", REASON_BACKLOG) for item in dropped]
        if updates:
            await self._in_worker(self._write_statuses, updates)
        for item, status, reason in updates:
            item.status, item.status_reason = status, reason
        for item in [*(admitted if announce else ()), *given_up]:
            self._announce(item)
        if dropped:
            log.info("news: analysis queue full, %d item(s) dropped", len(dropped))
        self._dispatch()
        return len(admitted)

    def _skip_reason(self, item: NewsItem, now_ts: int, llm_on: bool) -> str | None:
        """Why an item is not analysed, or None if it is."""
        if item.published_ts is not None and item.published_ts < now_ts - MAX_ITEM_AGE_S:
            return REASON_OLD
        feed = self._feeds.get(item.source)
        if not item.hinted_symbols and (feed is None or feed.quality < MIN_QUALITY_WITHOUT_MATCH):
            return REASON_NO_MATCH
        if not llm_on:
            return REASON_LLM_OFF
        return None

    def _urgency(self, item: NewsItem, now_ts: int) -> int:
        """The item's urgency class (see the module docstring; smaller = more urgent)."""
        if any(self._boosts.get(symbol, 0) > now_ts for symbol in item.hinted_symbols):
            return URGENCY_TRIGGERED
        feed = self._feeds.get(item.source)
        quality = 0 if feed is None else feed.quality
        tradable = any(
            instrument is not None and instrument.tradable and instrument.active
            for instrument in map(self._book.get, item.hinted_symbols)
        )
        if not tradable:
            return URGENCY_OTHER
        if feed is not None and feed.category == "press_release" and quality >= DISCLOSURE_MIN_QUALITY:
            return URGENCY_DISCLOSURE
        return URGENCY_TRADABLE if quality >= TRADABLE_MIN_QUALITY else URGENCY_OTHER

    def _rank(self, item: NewsItem, now_ts: int) -> tuple[int, int, int, int]:
        """Sort key: larger = more important (more urgent class first, then fresher)."""
        return (-self._urgency(item, now_ts), *_freshness(item))

    def _admit(self, items: Sequence[NewsItem]) -> tuple[list[NewsItem], list[NewsItem]]:
        """Add items to the waiting list within ``MAX_OUTSTANDING``, most important first. Returns (admitted,
        dropped); dropped can contain items that were waiting before (their analysis was announced but is
        now abandoned)."""
        now_ts = self._clock.ts()
        admitted: dict[int, NewsItem] = {}
        dropped: list[NewsItem] = []
        for item in sorted(items, key=lambda candidate: self._rank(candidate, now_ts), reverse=True):
            news_id = _stored_id(item)
            if news_id in self._waiting or news_id in self._running:
                continue
            if len(self._waiting) + len(self._running) >= MAX_OUTSTANDING:
                least = min(self._waiting.values(), key=lambda waiting: self._rank(waiting, now_ts), default=None)
                if least is None or self._rank(least, now_ts) >= self._rank(item, now_ts):
                    dropped.append(item)
                    continue
                del self._waiting[_stored_id(least)]
                admitted.pop(_stored_id(least), None)
                dropped.append(least)
            self._waiting[news_id] = item
            admitted[news_id] = item
        return list(admitted.values()), dropped

    def _dispatch(self) -> None:
        """Hand waiting items to the LLM, most important first, while fewer than ``LLM_DISPATCH_LIMIT`` are
        running."""
        now_ts = self._clock.ts()
        while not self._closed and self._waiting and len(self._running) < LLM_DISPATCH_LIMIT:
            item = max(self._waiting.values(), key=lambda waiting: self._rank(waiting, now_ts))
            news_id = _stored_id(item)
            del self._waiting[news_id]
            task = asyncio.create_task(self._analyse(item), name=f"news-analysis-{news_id}")
            self._running[news_id] = task
            task.add_done_callback(functools.partial(self._analysis_done, news_id))

    def _analysis_done(self, news_id: int, task: asyncio.Task[None]) -> None:
        self._running.pop(news_id, None)
        if not task.cancelled() and task.exception() is not None:
            log.error("news: analysis task of item %d failed", news_id, exc_info=task.exception())
        self._dispatch()

    async def _analyse(self, item: NewsItem) -> None:
        """Run one analysis and persist its outcome. Never raises (except on cancellation)."""
        try:
            if not self._llm_enabled():
                await self._give_up(item, "skipped", REASON_LLM_OFF)
                return
            candidates = self._candidates(item)
            try:
                async with asyncio.timeout(ANALYSIS_TIMEOUT_S):
                    analysis = await self._llm.analyze_news(item, candidates)
            except TimeoutError:
                log.warning("news: analysis of item %d timed out", item.id)
                analysis = None
            if analysis is None:
                await self._give_up(item, "skipped", REASON_LLM_UNAVAILABLE)
                return
            await self._in_worker(self._store_analysis, item, self._checked(item, analysis, candidates))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("news: analysis of item %d failed", item.id)
            try:
                await self._give_up(item, "error", REASON_ANALYSIS_FAILED)
            except Exception:
                log.exception("news: could not mark item %d as failed", item.id)

    async def _give_up(self, item: NewsItem, status: NewsStatus, reason: str) -> None:
        """Persist that the analysis of a queued item will not come and announce the item's final status."""
        await self._in_worker(self._write_statuses, [(item, status, reason)])
        self._announce(replace(item, status=status, status_reason=reason))

    def _candidates(self, item: NewsItem) -> list[Instrument]:
        """The instruments the LLM may name: alias matches (strongest first, at most 8), then macro indices."""
        found = [self._book.get(symbol) for symbol in item.hinted_symbols]
        return [instrument for instrument in found if instrument is not None and instrument.active]

    @staticmethod
    def _checked(item: NewsItem, analysis: NewsAnalysis, candidates: Sequence[Instrument]) -> NewsAnalysis:
        """Enforce the contract at the boundary: the analysis belongs to this item and names candidates only."""
        allowed = {instrument.symbol for instrument in candidates}
        symbols = {symbol: value for symbol, value in analysis.symbols.items() if symbol in allowed}
        if len(symbols) < len(analysis.symbols):
            log.warning("news: item %d: dropped %d symbol(s) outside the candidates",
                        item.id, len(analysis.symbols) - len(symbols))
        if analysis.news_id != item.id or symbols != analysis.symbols:
            analysis = replace(analysis, news_id=_stored_id(item), symbols=symbols)
        return analysis

    def _llm_enabled(self) -> bool:
        return self._settings.enable_llm and self._runtime_settings.get().news_llm_enabled

    def _announce(self, item: NewsItem, analysis: NewsAnalysis | None = None) -> None:
        """Publish the item on TOPIC_NEWS (a GUI notification from any thread: failures are logged, not raised)."""
        try:
            payload = presenters.news_out(item, analysis, source_names=self._source_names).model_dump()
        except Exception:
            log.exception("news: could not announce item %s", item.id)
            return
        self._bus.publish(TOPIC_NEWS, payload)

    async def _in_worker[**P, T](self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._worker, functools.partial(func, *args, **kwargs))

    # -- status ---------------------------------------------------------------------------------
    def status(self) -> NewsStatusOut:
        states = self._state_snapshot()
        enabled = [feed for feed in self._feeds.values() if feed.enabled]
        failing = sum(1 for feed in enabled if feed.id in states and states[feed.id].fail_count > 0)
        counts = repo_news.news_counts(self._db, since_ts=self._clock.ts() - 3600)
        return NewsStatusOut(
            feeds_total=len(enabled),
            feeds_failing=failing,
            items_last_hour=counts["items"],
            analyzed_last_hour=counts["analyzed"],
            backlog=counts["backlog"],
            last_poll_ts=self._last_poll_ts,
        )

    def feeds(self) -> list[FeedStateOut]:
        """Every registered feed with its polling state (registry order)."""
        states = self._state_snapshot()
        out: list[FeedStateOut] = []
        for feed in self._feeds.values():
            state = states.get(feed.id) or FeedState(feed.id)
            out.append(
                FeedStateOut(
                    feed_id=feed.id,
                    name=feed.name,
                    region=feed.region,
                    language=feed.language,
                    category=feed.category,
                    last_ok_ts=state.last_ok_ts,
                    last_error=state.last_error,
                    fail_count=state.fail_count,
                    items_total=state.items_total,
                )
            )
        return out

    def _state_snapshot(self) -> dict[str, FeedState]:
        """The in-memory feed states, or the stored ones before the service has been started."""
        with self._lock:
            if self._states_loaded:
                return dict(self._states)
        return repo_news.list_feed_states(self._db)
