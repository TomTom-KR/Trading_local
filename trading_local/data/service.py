"""Market-data service: implements ``contracts.MarketData`` on top of a ``Provider``.

Data paths (recommended ingestion strategy of the market-data probe)
* Stream: ticks update the quote book and, while the instrument's venue is open at the tick's time, are
  aggregated into 1-minute bars (source 'stream', provisional).
* Snapshot quotes: ONE batched request for all active symbols every 15 s while any venue is open (every
  5 minutes otherwise). They update the quote book; for symbols of open venues that get no stream ticks
  they are sampled into 1-minute bars (source 'quote').
* REST reconciliation (source 'rest', authoritative - ``repo_market`` enforces the precedence
  rest > stream > quote): round-robin over the symbols whose venue is open or closed less than
  30 minutes + feed delay ago, so that each one is refetched about every 5 minutes with a window covering
  the last 20 minutes of the source's (possibly delayed) data, i.e. from now - 20 minutes - feed delay (or
  from the end of its last successful window after an outage, at most 7 days back).
  The requests run ``rest_offset_s`` after ``refresh`` (away from the minute boundary, when the previous
  minute is final at Yahoo) and stay inside the per-minute request budget.
* Hole repair: when a stream/quote bar that extends a symbol's series does not directly follow the previous
  stored bar although the venue traded in between (stream gap, restart, sleep), the symbol is due for the
  reconciliation at once (at most every ``hole_repair_min_s``), ahead of the round-robin and with a window
  reaching back to the hole (the symbols whose holes are older than the plain window share one request
  with the oldest of their starts). Consumers that act on settled bars only (``BarReader.settled_bars_after``)
  wait at such a hole, so it must be filled quickly.
* Session-final fetch per venue 20-30 minutes (+ feed delay) after its close with pre/post data, which
  carries the closing-auction print stamped at the session end.
* Daily bars once after each venue's session completed (+ 30 minutes + feed delay); only COMPLETED
  sessions are stored (``calendar.last_completed_session``). The UTC-day venues (CRYPTO, FX, CME) trade
  straight through midnight, so their daily bar is fetched one minute (+ feed delay) after 00:00 UTC and
  retried with growing pauses until the completed day is stored: decisions after midnight need it. A
  second request at the regular offset (30 minutes + feed delay) stores the source's final values.
* While the provider's circuit breaker pauses REST access nothing is planned; session-final and daily jobs
  stay queued (also when a new rate limit refuses them) and run after the pause.
* Background backfill at start (restart-safe, ~1 request per second): 5 years of daily bars, then the most
  recent 8-day window of 1-minute bars for all symbols, then older windows back to 29 days, including the
  gap a long downtime left between the stored bars and the recent window.

Invariants: only CLOSED minute bars are written (open time + 60 <= the time the data was requested, and
never the newest row of a live request while it can still be the source's forming minute - Yahoo appends
a flat snapshot of the minute that holds its newest, for delayed feeds delayed, price);
FX and crypto bars carry volume 0 (their 1-minute volume is not meaningful at Yahoo); stream and quote bars
take volume from the session volume only for equities and futures. ``DataOptions`` relaxes the Yahoo
specifics for other sources (``zero_volume_classes``, ``drop_forming_live_row``).

Candle sources (``provider.CandleProvider``, e.g. Kraken): the stream delivers the exchange's own CLOSED
1-minute candles (``on_bar``, source 'stream'); ticks then only update the quote book, and snapshot quotes are
never sampled into bars (a minute without trades has no bar). The REST reconciliation confirms the candles.

Several sources: ``RoutedMarketData`` combines one ``MarketDataService`` per provider ("lane"). Each lane
sees only its instruments (``InstrumentView``) and has its own provider, request budget, circuit breaker,
stream, schedules and database thread, so an outage or rate limit of one source never touches the symbols
of another. ``build_market_data`` routes the instruments of the pseudo venue KRAKEN to the Kraken lane and
everything else to Yahoo.

Threads: the provider owns its worker thread; all database work of this service runs in ONE dedicated
thread (keeps the number of database threads small, see the GIL note of the architecture). The event loop
only touches in-memory state and never waits for a network answer inside ``refresh`` once ``start`` runs
the background loops. ``prepare`` initialises the schedules in the database thread as well (calendar
look-ups and database reads stay off the loop; nothing else touches that state before ``prepare``
returns). ``quote``, ``quotes`` and ``status`` are safe from any thread.
"""

import asyncio
import contextlib
import functools
import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from trading_local.clock import Clock, floor_ts, from_ts, to_ts
from trading_local.config import Settings
from trading_local.contracts import MarketCalendar
from trading_local.data.aggregator import MinuteAggregator
from trading_local.data.kraken import BACKFILL_DAYS, OHLC_LIMIT
from trading_local.data.provider import CandleProvider, Provider, ProviderError, RateLimitError, Tick, TickStream
from trading_local.data.quotebook import QuoteBook
from trading_local.db import repo_market, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS, InstrumentBook
from trading_local.dto import DataStatusOut
from trading_local.events import TOPIC_BAR, TOPIC_QUOTES, EventBus
from trading_local.models import Bar, EarningsDate, Instrument, Quote
from trading_local.presenters import quote_out
from trading_local.universe import REGION_SORT_BASE, USER_SORT_OFFSET, seed_default_universe

log = logging.getLogger(__name__)

DAY_S = 86_400
VOLUME_FROM_SESSION = frozenset({"equity", "future"})  # stream/quote bars: volume from the session volume
NO_REST_VOLUME = frozenset({"fx", "crypto"})  # Yahoo: REST 1-minute volume is not meaningful, stored as 0
NO_CLOSING_AUCTION = frozenset({"CRYPTO", "KRAKEN", "FX", "CME"})  # venues without a session-final fetch
# Pseudo venues whose session is the UTC day and whose trading continues across midnight: the completed
# day is needed right away, so its daily bar is fetched promptly (``daily_prompt_after_s``) and retried.
UTC_DAY_VENUES = frozenset({"CRYPTO", "KRAKEN", "FX", "CME"})


@dataclass(frozen=True, slots=True)
class DataOptions:
    """Cadence and budget of the service (defaults follow the market-data probe's recommendations)."""

    rest_offset_s: float = 20.0  # REST work starts this long after refresh (0: inline, used by tests)
    quote_interval_open_s: int = 15  # snapshot cadence while any venue is open
    quote_interval_closed_s: int = 300
    request_budget_per_min: int = 26  # all REST requests of one minute (quotes + bars)
    recon_interval_s: int = 300  # each eligible symbol is refetched about this often
    recon_lookback_s: int = 1_200  # reconciliation window: now - 20 minutes
    recon_overlap_s: int = 600  # overlap with the previous window after an outage
    max_gap_repair_s: int = 7 * DAY_S  # how far back an outage is repaired by the reconciliation
    hole_repair_min_s: int = 120  # a symbol with a reported hole is refetched once its last request is this old
    after_close_s: int = 1_800  # venues stay eligible this long (+ feed delay) after their close
    final_after_s: int = 1_200  # session-final fetch this long (+ feed delay) after the close ...
    final_until_s: int = 3_600  # ... or later if the service was not running then, up to this long
    final_lookback_s: int = 1_800  # window of the session-final fetch before the close
    daily_after_s: int = 1_800  # daily bars of a session are fetched this long (+ delay) after it completed
    daily_prompt_after_s: int = 60  # ... for UTC_DAY_VENUES this long (+ delay) after 00:00 UTC
    daily_retry_s: int = 60  # UTC-day venues: first pause before a missing daily bar is fetched again ...
    daily_retry_max_s: int = 3_600  # ... doubling per attempt up to this pause ...
    daily_retry_limit: int = 10  # ... for at most this many attempts per session
    daily_update_days: int = 10
    daily_years: int = 5
    minute_window_s: int = 8 * DAY_S - 60  # Yahoo: at most 8 days of 1-minute bars per request
    minute_lookback_s: int = 29 * DAY_S  # Yahoo: 1-minute bars of the last 30 days (one day of margin)
    backfill_pause_s: float = 1.0  # pause per backfill request (~1 request per second)
    backfill_batch: int = 4  # symbols per backfill request batch
    stream_silence_s: int = 180  # a symbol counts as streaming if a usable tick arrived within this time
    stale_margin_min: int = 5  # status: data older than feed delay + this margin is stale
    mass_failure_min_symbols: int = 4  # "empty for most open-market symbols" needs at least this many
    publish_interval_s: float = 1.0  # TOPIC_QUOTES at most once per this interval
    # --- source specifics (defaults: Yahoo) ------------------------------------------------------
    source_label: str = "Yahoo"  # name of the data source in activity entries
    zero_volume_classes: frozenset[str] = NO_REST_VOLUME  # asset classes whose REST volume is stored as 0
    drop_forming_live_row: bool = True  # the newest row of a live request may be the source's forming minute
    max_stream_bar_age_s: int = 3_600  # candle sources: older stream candles (reconnect snapshots) are ignored


# Kraken (pseudo venue KRAKEN): real-time public data, 1 request per second at most (enforced by the provider),
# OHLC covers the newest 720 minutes, older 1-minute history comes from trades (BACKFILL_DAYS back).
KRAKEN_OPTIONS = DataOptions(
    rest_offset_s=10.0,  # Kraken commits a candle right after its minute; no delay to wait for
    quote_interval_open_s=30,  # REST ticker as fallback/day statistics: the WS ticker delivers live quotes
    quote_interval_closed_s=30,
    request_budget_per_min=20,
    recon_interval_s=180,  # every pair is confirmed by REST about every 3 minutes (settles stream candles)
    max_gap_repair_s=(OHLC_LIMIT - 15) * 60,  # the reconciliation stays inside the OHLC range
    minute_window_s=(OHLC_LIMIT - 10) * 60,  # recent window = one OHLC request; older windows = trades
    minute_lookback_s=BACKFILL_DAYS * DAY_S,
    daily_years=2,  # OHLC interval 1440 serves 720 days
    backfill_pause_s=0.0,  # the provider's rate limiter paces the history requests
    backfill_batch=1,  # one pair per history job: fine-grained progress, resumable per window
    mass_failure_min_symbols=1_000_000,  # Kraken reports failures explicitly, empty answers are genuine
    source_label="Kraken",
    zero_volume_classes=frozenset(),  # exchange volume is real
    drop_forming_live_row=False,  # the provider drops Kraken's uncommitted candle itself
)


@dataclass(frozen=True, slots=True)
class _Venue:
    """Session facts of one venue at the planning time."""

    key: str
    is_open: bool
    in_break: bool
    last_close_ts: int | None  # end of the last regular segment that ended before now (None while open)
    delay_s: int  # largest feed delay of the venue's active instruments


@dataclass(frozen=True, slots=True)
class _Hole:
    """A traded hole that live bars left behind a symbol's previous stored bar, awaiting a REST repair."""

    start_ts: int  # open time of the first missing minute (of the oldest unrepaired hole of the symbol)
    seen_ts: int  # when it was (last) reported: a later report replaces the mark and needs another request


@dataclass(slots=True)
class _DailyRetry:
    """UTC-day venues: a symbol whose daily bar of ``session`` is not stored yet."""

    session: date
    attempts: int = 0
    next_ts: int = 0  # the symbol is queued again once this time has passed


@dataclass(frozen=True, slots=True)
class _RestJob:
    kind: Literal["minutes", "final", "daily"]
    symbols: tuple[str, ...]
    start_ts: int
    close_ts: int | None = None  # session-final: the close whose last bars and closing print are fetched
    repairs: tuple[tuple[str, _Hole], ...] = ()  # reconciliation: holes the window was planned to repair


@dataclass(frozen=True, slots=True)
class _HistoryJob:
    kind: Literal["daily", "recent", "older"]
    symbols: tuple[str, ...]
    start_ts: int
    end_ts: int | None = None


def _batches(symbols: Sequence[str], size: int) -> list[tuple[str, ...]]:
    return [tuple(symbols[index : index + size]) for index in range(0, len(symbols), size)]


def _day_start_ts(day: str) -> int:
    return to_ts(datetime.fromisoformat(day).replace(tzinfo=UTC))


def _log_future_failure(future: Future[Any]) -> None:
    if not future.cancelled() and (exc := future.exception()) is not None:
        log.error("background database write failed: %s", exc)


class InstrumentView:
    """The instruments of one provider lane: a filtered, read-only window onto the shared ``InstrumentBook``.

    Offers the part of the book's interface the service uses; instruments outside the lane do not exist for
    it (``get`` returns None). It always reflects the book's current snapshot (``reload`` reloads the book).
    """

    def __init__(self, book: InstrumentBook, owns: Callable[[Instrument], bool]) -> None:
        self._book = book
        self._owns = owns

    def reload(self) -> None:
        self._book.reload()

    def get(self, symbol: str) -> Instrument | None:
        instrument = self._book.get(symbol)
        return instrument if instrument is not None and self._owns(instrument) else None

    def require(self, symbol: str) -> Instrument:
        instrument = self.get(symbol)
        if instrument is None:
            raise KeyError(symbol)
        return instrument

    def all(self, *, active_only: bool = True) -> list[Instrument]:
        return [instrument for instrument in self._book.all(active_only=active_only) if self._owns(instrument)]

    def tradable(self) -> list[Instrument]:
        return [instrument for instrument in self._book.tradable() if self._owns(instrument)]

    def by_exchange(self) -> dict[str, list[Instrument]]:
        groups: dict[str, list[Instrument]] = {}
        for instrument in self.all():
            groups.setdefault(instrument.exchange, []).append(instrument)
        return groups

    def symbols(self, *, active_only: bool = True) -> list[str]:
        return [instrument.symbol for instrument in self.all(active_only=active_only)]

    def __contains__(self, symbol: object) -> bool:
        return isinstance(symbol, str) and self.get(symbol) is not None


class MarketDataService:
    """``contracts.MarketData`` implementation. Not restartable: ``stop()`` releases its threads for good."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        calendar: MarketCalendar,
        instruments: InstrumentBook | InstrumentView,
        provider: Provider,
        options: DataOptions | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._clock = clock
        self._bus = bus
        self._calendar = calendar
        self._instruments = instruments
        self._provider = provider
        self._opts = options or DataOptions()
        self._network = not settings.offline
        self._user_tz = ZoneInfo(settings.timezone)
        # A candle source delivers closed exchange candles: ticks and snapshot quotes then build no bars.
        self._candle_source = isinstance(provider, CandleProvider)

        self._book = QuoteBook()
        self._stream_bars = MinuteAggregator("stream")
        self._quote_bars = MinuteAggregator("quote")
        self._pending: list[Bar] = []  # bars completed by ticks, written by the next refresh
        self._stream: TickStream | None = None
        self._subscribed: list[str] = []
        self._last_stream_tick: dict[str, float] = {}  # arrival time of the newest usable stream tick

        self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="market-data-db")
        self._tasks: list[asyncio.Task[None]] = []
        self._quote_task: asyncio.Task[None] | None = None  # background snapshot loop (owns the cadence)
        self._rest_task: asyncio.Task[None] | None = None
        self._refresh_lock = asyncio.Lock()
        self._quote_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()
        self._subscription_lock = asyncio.Lock()
        self._stopped = False

        # state shared with the database thread and with status() callers
        self._lock = threading.Lock()
        self._last_bar_ts: dict[str, int] = {}
        self._updated: set[str] = set()
        self._last_tick_ts: int | None = None
        self._last_rest_ok_ts: int | None = None
        self._backfill_progress: float | None = None

        # scheduling state (event loop only)
        self._last_quote_poll: int | None = None
        self._rest_fetched: dict[str, int] = {}  # last REST reconciliation request per symbol
        self._rest_until: dict[str, int] = {}  # end of the last successful REST window per symbol
        self._repair_from: dict[str, _Hole] = {}  # symbol -> traded hole the reconciliation must repair
        self._final_done: dict[str, int] = {}  # venue -> close whose session-final fetch was queued
        self._final_queue: deque[tuple[int, str]] = deque()  # (close, symbol) awaiting the session-final fetch
        self._daily_done: dict[str, date] = {}  # venue -> newest session whose daily bars were queued
        self._daily_queue: deque[str] = deque()
        self._daily_missing: dict[str, _DailyRetry] = {}  # UTC-day venues: daily bars not stored yet
        self._daily_confirmed: dict[str, date] = {}  # UTC-day venue -> newest session whose final fetch was queued
        self._quote_misses: dict[str, int] = {}
        self._quarantine: set[str] = set()
        self._lookup_failures: dict[str, int] = {}
        self._logged_pause: int | None = None

    # ------------------------------------------------------------------------------------------
    # MarketData protocol
    # ------------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.prepare()
        if not self._network:
            log.info("offline mode: market data serves the stored data only")
            return
        if self._settings.enable_stream:
            if isinstance(self._provider, CandleProvider):
                self._stream = self._provider.open_candle_stream(self.on_tick, self.on_bar, self._on_stream_state)
            else:
                self._stream = self._provider.open_stream(self.on_tick, self._on_stream_state)
            await self._sync_subscription()  # before connecting: the shards start with the full symbol set
            await self._stream.connect()
        self._quote_task = asyncio.create_task(self._quote_loop(), name="market-data-quotes")
        self._tasks = [
            self._quote_task,
            asyncio.create_task(self._publish_loop(), name="market-data-publish"),
            asyncio.create_task(self.run_backfill(), name="market-data-backfill"),
        ]

    async def stop(self) -> None:
        self._stopped = True  # a refresh still running must not start new REST work
        tasks = [*self._tasks, *([self._rest_task] if self._rest_task is not None else [])]
        self._tasks, self._rest_task, self._quote_task = [], None, None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.close()
            self._stream = None
        with contextlib.suppress(Exception):
            await self._persist_quotes()
        with contextlib.suppress(Exception):
            await self._provider.close()
        await asyncio.to_thread(self._db_executor.shutdown, True)

    async def refresh(self, now: datetime) -> list[str]:
        if self._stopped:
            return []
        try:
            async with self._refresh_lock:
                await self._refresh(now)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("market-data refresh failed")
        return self._take_updated()

    def quote(self, symbol: str) -> Quote | None:
        return self._book.get(symbol)

    def quotes(self) -> dict[str, Quote]:
        return self._book.snapshot()

    def status(self) -> DataStatusOut:
        now = self._clock.now()
        now_ts = to_ts(now)
        active = self._instruments.all()
        quotes = self._book.snapshot()
        with self._lock:
            last_bars = dict(self._last_bar_ts)
            last_tick, last_rest, progress = self._last_tick_ts, self._last_rest_ok_ts, self._backfill_progress
        stale = sum(self._is_stale(instrument, now, now_ts, quotes, last_bars) for instrument in active)
        return DataStatusOut(
            stream_connected=self._stream is not None and self._stream.connected,
            last_tick_ts=last_tick,
            last_rest_ok_ts=last_rest,
            rate_limited_until=self._provider.paused_until(),
            symbols_active=len(active),
            symbols_stale=stale,
            backfill_progress=progress,
        )

    async def lookup(self, symbol: str) -> Instrument | None:
        key = symbol.strip().upper()
        if not self._network or not key or self._lookup_failures.get(key, 0) >= 2:
            return None
        try:
            instrument = await self._provider.lookup(key)
        except (ProviderError, ValueError) as exc:
            log.warning("lookup of %s failed: %s", key, exc)
            return None
        except Exception:
            log.exception("lookup of %s failed", key)
            return None
        if instrument is None or instrument.exchange not in self._calendar.exchanges():
            self._lookup_failures[key] = self._lookup_failures.get(key, 0) + 1
            return None
        self._lookup_failures.pop(key, None)
        return replace(instrument, sort_order=REGION_SORT_BASE[instrument.region] + USER_SORT_OFFSET)

    async def earnings(self, symbol: str) -> list[EarningsDate] | None:
        """Earnings releases from the provider (``fetch_earnings``); [] for providers without them."""
        fetch = getattr(self._provider, "fetch_earnings", None)
        if fetch is None:
            return []
        if not self._network:
            return None
        try:
            return list(await fetch(symbol))
        except (ProviderError, ValueError) as exc:
            log.warning("earnings dates of %s unavailable: %s", symbol, exc)
        except Exception:
            log.exception("earnings dates of %s failed", symbol)
        return None

    async def ensure_history(self, symbol: str) -> None:
        instrument = self._instruments.get(symbol)
        if instrument is None:
            log.warning("ensure_history: %s is not in the instrument book", symbol)
            return
        if not self._network:
            return
        await self._sync_subscription()
        try:
            jobs = await self._db_call(self._plan_history, self._clock.ts(), [instrument])
            await self._run_history(jobs, track_progress=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("history download for %s failed", symbol)

    # ------------------------------------------------------------------------------------------
    # Public helpers (start-up steps, background loops, other tick sources, tests)
    # ------------------------------------------------------------------------------------------
    async def prepare(self) -> None:
        """First step of ``start()`` without any network activity: seed the universe if the instruments
        table is empty, load the persisted quotes and newest bar times, initialise the schedules.

        Everything runs in the database thread: the schedules need calendar look-ups (which build a venue's
        calendar if it is not cached yet) and database reads, neither of which may stall the event loop."""
        await self._db_call(self._load_state)
        await self._db_call(self._init_schedule, self._clock.now())

    def on_tick(self, tick: Tick) -> None:
        """Stream callback (event loop): update the quote book and, while the instrument's venue is open at
        the tick's time and the tick is not an old replay, aggregate it into a 'stream' bar."""
        instrument = self._instruments.get(tick.symbol)
        if instrument is None or not instrument.active:
            return
        now = self._clock.now().timestamp()
        if tick.ts > now + 60:
            return
        self._book.update_from_tick(tick)
        with self._lock:
            self._last_tick_ts = int(tick.received_ts)
        if self._candle_source:
            return  # the exchange's own candles arrive through on_bar
        if self._usable_for_bars(instrument, tick.ts, now):
            self._last_stream_tick[tick.symbol] = tick.received_ts
            self._pending.extend(self._stream_bars.add(self._bar_tick(instrument, tick)))

    def on_bar(self, bar: Bar) -> None:
        """Candle-stream callback (event loop): a CLOSED exchange candle becomes a pending 'stream' bar.

        Ignored: symbols outside this service or inactive, candles of a future minute, candles older than
        ``max_stream_bar_age_s`` (reconnect snapshots - the REST reconciliation covers them) and invalid prices.
        The bar is written by the next ``refresh`` once its minute is over by the LOCAL clock (repo precedence:
        it never replaces a REST bar). A candle that the exchange closed while the local clock still shows the
        last second of its minute (the local clock may run a little behind the exchange's) waits in the queue."""
        instrument = self._instruments.get(bar.symbol)
        if instrument is None or not instrument.active:
            return
        now = self._clock.now().timestamp()
        if bar.ts > now + BAR_SECONDS or now - bar.ts > self._opts.max_stream_bar_age_s:
            return
        prices = (bar.open, bar.high, bar.low, bar.close)
        if not all(math.isfinite(p) and p > 0 for p in prices) or not (bar.low <= min(prices) <= max(prices) <= bar.high):
            return
        with self._lock:
            self._last_tick_ts = int(now)
        self._last_stream_tick[bar.symbol] = now
        volume = bar.volume if math.isfinite(bar.volume) and bar.volume > 0 else 0.0
        self._pending.append(replace(bar, volume=volume, source="stream"))

    def publish_quotes(self) -> int:
        """Announce the quotes that changed since the previous call on ``TOPIC_QUOTES``; returns their number."""
        changed = self._book.take_changed("publish")
        if changed:
            self._bus.publish(TOPIC_QUOTES, {"items": [quote_out(quote).model_dump() for quote in changed]})
        return len(changed)

    async def poll_quotes(self, now_ts: int) -> None:
        """Fetch one snapshot of all active symbols if the cadence says it is due. Never raises."""
        async with self._quote_lock:
            interval = (
                self._opts.quote_interval_open_s
                if self._any_venue_open(from_ts(now_ts))
                else self._opts.quote_interval_closed_s
            )
            if self._last_quote_poll is not None and now_ts - self._last_quote_poll < interval:
                return
            self._last_quote_poll = now_ts
            symbols = self._instruments.symbols()
            if not symbols:
                return
            try:
                quotes = await self._provider.fetch_quotes(symbols)
            except RateLimitError:
                return
            except ProviderError as exc:
                log.warning("snapshot quotes failed: %s", exc)
                return
            except Exception:
                log.exception("snapshot quotes failed")
                return
            self._mark_rest_ok()
            self._apply_snapshot(symbols, quotes)

    async def run_backfill(self) -> None:
        """Download the missing history of all active symbols (restart-safe, polite). Never raises."""
        try:
            jobs = await self._db_call(self._plan_history, self._clock.ts(), self._active_instruments())
            if not jobs:
                return
            await self._run_history(jobs, track_progress=True)
            symbols = len({symbol for job in jobs for symbol in job.symbols})
            requests = sum(len(job.symbols) for job in jobs)
            label = self._opts.source_label
            await self._activity(
                "info", f"Historische Kursdaten geladen ({label}: {symbols} Symbole, {requests} Abrufe)."
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("history backfill failed")
        finally:
            with self._lock:
                self._backfill_progress = None

    # ------------------------------------------------------------------------------------------
    # Start-up
    # ------------------------------------------------------------------------------------------
    def _load_state(self) -> None:
        """(db thread) Seed the universe if the table is empty, load persisted quotes and newest bar times."""
        if not repo_market.list_instruments(self._db, active_only=False):
            written = seed_default_universe(self._db)
            self._instruments.reload()
            log.info("seeded the default universe (%d instruments)", written)
        self._book.load(repo_market.load_quotes(self._db).values())
        last_bars = repo_market.symbols_with_bars(self._db)
        with self._lock:
            self._last_bar_ts.update(last_bars)

    def _init_schedule(self, now: datetime) -> None:
        """(db thread, inside ``prepare``) Initial reconciliation times and daily-bar schedule.

        Sessions that completed before the start count as queued (for the UTC-day venues: also as
        confirmed, if the confirming offset had passed): the background backfill fetches what the database
        lacks. A UTC-day venue's symbol whose newest stored daily bar is older than the venue's
        last completed session is registered as missing, with its first retry ``daily_after_s`` later (the
        backfill gets the first chance)."""
        now_ts = to_ts(now)
        for position, instrument in enumerate(self._active_instruments()):
            # spread the first reconciliation round over five minutes (the backfill covers the start)
            self._rest_fetched.setdefault(instrument.symbol, now_ts - (position % 5) * 60)
        by_venue = self._symbols_by_venue()
        for venue, delay_s in self._venue_delays().items():
            try:
                session = self._calendar.last_completed_session(
                    venue, from_ts(now_ts - self._daily_after_s(venue) - delay_s)
                )
                confirmed = self._calendar.last_completed_session(
                    venue, from_ts(now_ts - self._opts.daily_after_s - delay_s)
                )
            except KeyError:
                continue
            if session is None:
                continue
            self._daily_done[venue] = session
            if venue not in UTC_DAY_VENUES:
                continue
            if confirmed is not None:
                self._daily_confirmed[venue] = confirmed
            for symbol in by_venue.get(venue, []):
                last_day = repo_market.last_day(self._db, symbol)
                # without any daily bar the backfill downloads the full history; the retry is for the update
                if last_day is not None and last_day < session.isoformat():
                    retry = _DailyRetry(session, next_ts=now_ts + self._opts.daily_after_s)
                    self._daily_missing.setdefault(symbol, retry)

    # ------------------------------------------------------------------------------------------
    # Minute cycle
    # ------------------------------------------------------------------------------------------
    async def _refresh(self, now: datetime) -> None:
        now_ts = to_ts(now)
        if self._network:
            await self._sync_subscription()
            if self._quote_task is None:  # without the background loop (not started) refresh keeps the cadence
                await self.poll_quotes(now_ts)
        await self._persist_live_bars(now_ts)
        if self._network and not self._stopped and (self._rest_task is None or self._rest_task.done()):
            jobs = self._plan_rest(now_ts)
            if jobs and self._opts.rest_offset_s <= 0:
                await self._run_rest(jobs)
            elif jobs:
                self._rest_task = asyncio.create_task(self._delayed_rest(jobs), name="market-data-rest")
        await self._persist_quotes()
        await self._check_pause()

    def _take_updated(self) -> list[str]:
        with self._lock:
            updated, self._updated = sorted(self._updated), set()
        return updated

    async def _persist_live_bars(self, now_ts: int) -> None:
        bars = [*self._pending, *self._stream_bars.flush(now_ts), *self._quote_bars.flush(now_ts)]
        closed = [bar for bar in bars if bar.ts + BAR_SECONDS <= now_ts]
        self._pending = [bar for bar in bars if bar.ts + BAR_SECONDS > now_ts]
        if closed:
            holes = await self._db_call(self._write_live_bars, closed)
            for symbol, start_ts in holes.items():
                known = self._repair_from.get(symbol)
                oldest = start_ts if known is None else min(known.start_ts, start_ts)
                self._repair_from[symbol] = _Hole(oldest, now_ts)

    async def _persist_quotes(self) -> None:
        changed = self._book.take_changed("persist")
        if changed:
            await self._db_call(repo_market.upsert_quotes, self._db, changed)

    async def _sync_subscription(self) -> None:
        """Subscribe the stream to the active, not quarantined symbols (serialised: refresh and
        ensure_history may call it at the same time)."""
        async with self._subscription_lock:
            if self._stream is None:
                return
            wanted = [symbol for symbol in self._instruments.symbols() if symbol not in self._quarantine]
            if wanted != self._subscribed:
                await self._stream.subscribe(wanted)
                self._subscribed = wanted

    async def _check_pause(self) -> None:
        """Activity entries when the REST pause of the circuit breaker starts and ends."""
        paused = self._provider.paused_until()
        if paused is not None and paused != self._logged_pause:
            self._logged_pause = paused
            until = from_ts(paused).astimezone(self._user_tz)
            label = self._opts.source_label
            await self._activity(
                "warning", f"{label} begrenzt die Abrufe: Kursdaten-Abrufe pausiert bis {until:%H:%M} Uhr."
            )
        elif paused is None and self._logged_pause is not None:
            self._logged_pause = None
            await self._activity("info", f"Kursdaten-Abrufe bei {self._opts.source_label} laufen wieder.")

    # ------------------------------------------------------------------------------------------
    # Stream and snapshot quotes
    # ------------------------------------------------------------------------------------------
    def _on_stream_state(self, connected: bool) -> None:
        label = self._opts.source_label
        if connected:
            self._activity_soon("info", f"Kursstream {label} verbunden.")
        else:
            self._activity_soon(
                "warning", f"Kursstream {label} getrennt – die Verbindung wird automatisch wiederhergestellt."
            )

    def _apply_snapshot(self, symbols: Sequence[str], quotes: Sequence[Quote]) -> None:
        returned = {quote.symbol for quote in quotes}
        if quotes:  # an empty answer says nothing about single symbols
            for symbol in symbols:
                if symbol in returned:
                    self._quote_misses.pop(symbol, None)
                    self._quarantine.discard(symbol)
                    continue
                self._quote_misses[symbol] = misses = self._quote_misses.get(symbol, 0) + 1
                if misses >= 2 and symbol not in self._quarantine:
                    self._quarantine.add(symbol)
                    log.warning("%s is unknown at the data source: no REST requests until it reappears", symbol)
        now = self._clock.now().timestamp()
        for quote in quotes:
            instrument = self._instruments.get(quote.symbol)
            if instrument is None:
                continue
            self._book.update(quote)
            if self._candle_source:
                continue  # no trade, no bar: snapshot prices are never sampled into bars of a candle source
            streaming = now - self._last_stream_tick.get(quote.symbol, -math.inf) <= self._opts.stream_silence_s
            if not streaming and self._usable_for_bars(instrument, quote.ts, now):
                tick = Tick(quote.symbol, float(quote.ts), quote.price, float(quote.received_ts), quote.day_volume)
                self._pending.extend(self._quote_bars.add(self._bar_tick(instrument, tick)))

    def _usable_for_bars(self, instrument: Instrument, ts: float, now: float) -> bool:
        """A tick builds bars only if its venue was open at its time and it is not an old replay."""
        if now - ts > instrument.delay_min * 60 + 600:
            return False
        try:
            return self._calendar.is_open(instrument.exchange, from_ts(ts))
        except KeyError:
            return False

    @staticmethod
    def _bar_tick(instrument: Instrument, tick: Tick) -> Tick:
        if instrument.asset_class in VOLUME_FROM_SESSION or tick.day_volume is None:
            return tick
        return replace(tick, day_volume=None)

    # ------------------------------------------------------------------------------------------
    # REST planning
    # ------------------------------------------------------------------------------------------
    def _plan_rest(self, now_ts: int) -> list[_RestJob]:
        """This minute's REST jobs within the request budget: session finals first (they have a deadline),
        then the reconciliation round (hole repairs first, on top of the round-robin quota), then daily
        updates. Nothing is planned while REST access is paused; due session finals and daily updates are
        queued anyway and wait for the end of the pause (reported holes stay marked as well)."""
        venues = self._venues(from_ts(now_ts))
        by_venue = self._symbols_by_venue()
        self._queue_finals(now_ts, venues, by_venue)
        self._queue_daily_updates(now_ts, venues, by_venue)
        if self._provider.paused_until() is not None:
            return []
        any_open = any(venue.is_open for venue in venues.values())
        polls_per_minute = math.ceil(60 / self._opts.quote_interval_open_s) if any_open else 1
        budget = self._opts.request_budget_per_min - polls_per_minute
        jobs: list[_RestJob] = []

        finals: dict[int, list[str]] = {}
        while budget > 0 and self._final_queue:
            close, symbol = self._final_queue.popleft()
            finals.setdefault(close, []).append(symbol)
            budget -= 1
        for close, symbols in finals.items():
            jobs.append(_RestJob("final", tuple(symbols), close - self._opts.final_lookback_s, close))

        eligible = [
            symbol
            for venue in venues.values()
            if venue.is_open or self._closed_recently(venue, now_ts)
            for symbol in by_venue.get(venue.key, [])
        ]
        if eligible and budget > 0:
            quota = math.ceil(len(eligible) * 60 / self._opts.recon_interval_s)
            candidates = [s for s in eligible if self._recon_due(s, now_ts)]
            repairs = sum(1 for s in candidates if s in self._repair_from)
            # hole repairs first and on top of the round-robin quota (still within the request budget)
            due = sorted(candidates, key=lambda s: (s not in self._repair_from, self._rest_fetched.get(s, 0)))[
                : min(quota + repairs, budget)
            ]
            budget -= len(due)
            groups = self._recon_groups(due, now_ts)
            for symbol in due:
                self._rest_fetched[symbol] = now_ts
            for start, symbols in sorted(groups.items()):
                planned = tuple((s, self._repair_from[s]) for s in symbols if s in self._repair_from)
                jobs.append(_RestJob("minutes", tuple(symbols), start, repairs=planned))

        if budget > 0 and self._daily_queue:
            symbols = [self._daily_queue.popleft() for _ in range(min(budget, len(self._daily_queue)))]
            for symbol in symbols:
                self._count_daily_attempt(symbol, now_ts)
            jobs.append(_RestJob("daily", tuple(symbols), now_ts - self._opts.daily_update_days * DAY_S))
        return jobs

    def _recon_due(self, symbol: str, now_ts: int) -> bool:
        """Due for the reconciliation: the round-robin interval passed, or a reported hole waits for repair
        and the symbol's last request is at least ``hole_repair_min_s`` old."""
        age = now_ts - self._rest_fetched.get(symbol, 0)
        if age >= self._opts.recon_interval_s - 60:
            return True
        return symbol in self._repair_from and age >= self._opts.hole_repair_min_s

    def _closed_recently(self, venue: _Venue, now_ts: int) -> bool:
        """True during the 30 minutes (+ feed delay) after the venue's last regular segment ended."""
        close = venue.last_close_ts
        return close is not None and now_ts - close < self._opts.after_close_s + venue.delay_s

    def _recon_start(self, symbol: str, now_ts: int, *, with_hole: bool = True) -> int:
        """Window start: 20 minutes before the newest data the source can have (now - feed delay), earlier
        after an outage (end of the last good window - delay - overlap) and for a reported hole (the bar
        before it; ignored with ``with_hole=False``), at most ``max_gap_repair_s`` back."""
        instrument = self._instruments.get(symbol)
        delay_s = instrument.delay_min * 60 if instrument is not None else 0
        minute = floor_ts(now_ts)
        start = minute - delay_s - self._opts.recon_lookback_s
        until = self._rest_until.get(symbol)
        if until is not None:
            start = min(start, floor_ts(until) - delay_s - self._opts.recon_overlap_s)
        hole = self._repair_from.get(symbol) if with_hole else None
        if hole is not None:
            start = min(start, floor_ts(hole.start_ts) - BAR_SECONDS)
        return max(start, minute - self._opts.max_gap_repair_s)

    def _recon_groups(self, symbols: Sequence[str], now_ts: int) -> dict[int, list[str]]:
        """The symbols of this minute's reconciliation grouped by window start (one request per group).

        Symbols whose window reaches back to a hole older than the plain window share the oldest of those
        starts: after a restart every symbol's hole begins at a slightly different minute, and ONE request
        for all of them keeps the mass-failure check of ``_minutes_job`` (it needs several symbols per
        request) effective against a source that hides a refusal behind empty answers."""
        starts = {symbol: self._recon_start(symbol, now_ts) for symbol in symbols}
        reaching = [s for s in symbols if starts[s] < self._recon_start(s, now_ts, with_hole=False)]
        if reaching:
            common = min(starts[symbol] for symbol in reaching)
            starts.update(dict.fromkeys(reaching, common))
        groups: dict[int, list[str]] = {}
        for symbol in symbols:
            groups.setdefault(starts[symbol], []).append(symbol)
        return groups

    def _queue_finals(self, now_ts: int, venues: Mapping[str, _Venue], by_venue: Mapping[str, list[str]]) -> None:
        """Queue the session-final fetch of every venue whose close was 20-60 minutes (+ delay) ago."""
        for venue in venues.values():
            close = venue.last_close_ts
            if venue.key in NO_CLOSING_AUCTION or close is None or venue.in_break:
                continue
            age = now_ts - close - venue.delay_s
            if self._opts.final_after_s <= age <= self._opts.final_until_s and self._final_done.get(venue.key) != close:
                self._final_done[venue.key] = close
                self._final_queue.extend((close, symbol) for symbol in by_venue.get(venue.key, []))

    def _queue_daily_updates(self, now_ts: int, venues: Mapping[str, _Venue], by_venue: Mapping[str, list[str]]) -> None:
        """Queue the daily bars of every venue whose next session completed ``daily_after_s`` + delay ago
        (``daily_prompt_after_s`` + delay for the UTC-day venues, confirmed at ``daily_after_s``), then the
        retries that are due."""
        for venue in venues.values():
            try:
                session = self._calendar.last_completed_session(
                    venue.key, from_ts(now_ts - self._daily_after_s(venue.key) - venue.delay_s)
                )
            except KeyError:
                continue
            done = self._daily_done.get(venue.key)
            if session is None or (done is not None and session <= done):
                continue
            self._daily_done[venue.key] = session
            symbols = by_venue.get(venue.key, [])
            self._daily_queue.extend(s for s in symbols if s not in self._daily_queue)
            if venue.key in UTC_DAY_VENUES:
                self._daily_missing.update((symbol, _DailyRetry(session)) for symbol in symbols)
        self._queue_daily_confirmations(now_ts, venues, by_venue)
        self._queue_daily_retries(now_ts)

    def _queue_daily_confirmations(
        self, now_ts: int, venues: Mapping[str, _Venue], by_venue: Mapping[str, list[str]]
    ) -> None:
        """UTC-day venues: queue the completed day's daily bars once more at the regular ``daily_after_s``
        (+ delay) offset. The prompt request right after midnight may still get the source's provisional
        values of the day's last minutes; this one stores the final values, which every later decision of
        the day and the validation replay read."""
        for venue in venues.values():
            if venue.key not in UTC_DAY_VENUES:
                continue
            try:
                session = self._calendar.last_completed_session(
                    venue.key, from_ts(now_ts - self._opts.daily_after_s - venue.delay_s)
                )
            except KeyError:
                continue
            done = self._daily_confirmed.get(venue.key)
            if session is None or (done is not None and session <= done):
                continue
            self._daily_confirmed[venue.key] = session
            self._daily_queue.extend(s for s in by_venue.get(venue.key, []) if s not in self._daily_queue)

    def _queue_daily_retries(self, now_ts: int) -> None:
        """Queue again every UTC-day symbol whose daily bar is still missing once its pause is over; give up
        after ``daily_retry_limit`` attempts (the next session's update covers the day again)."""
        for symbol, retry in list(self._daily_missing.items()):
            instrument = self._instruments.get(symbol)
            if instrument is None or not instrument.active or retry.attempts >= self._opts.daily_retry_limit:
                del self._daily_missing[symbol]
                if instrument is not None and instrument.active:
                    log.warning(
                        "%s: daily bar of %s still missing after %d attempts", symbol, retry.session, retry.attempts
                    )
                continue
            if now_ts >= retry.next_ts and symbol not in self._quarantine and symbol not in self._daily_queue:
                self._daily_queue.append(symbol)

    def _count_daily_attempt(self, symbol: str, now_ts: int) -> None:
        """A daily request for ``symbol`` is planned now: schedule the next retry with a doubled pause."""
        retry = self._daily_missing.get(symbol)
        if retry is None:
            return
        retry.attempts += 1
        pause = self._opts.daily_retry_s * 2 ** (retry.attempts - 1)
        retry.next_ts = now_ts + min(pause, self._opts.daily_retry_max_s)

    def _daily_after_s(self, venue: str) -> int:
        """How long (without the feed delay) after a session completed its daily bars are fetched."""
        return self._opts.daily_prompt_after_s if venue in UTC_DAY_VENUES else self._opts.daily_after_s

    def _venues(self, now: datetime) -> dict[str, _Venue]:
        venues: dict[str, _Venue] = {}
        for key, delay_s in self._venue_delays().items():
            try:
                state = self._calendar.state(key, now)
                last_close: int | None = None
                if not state.is_open:
                    segments = self._calendar.segments(key, now - timedelta(days=5), now)
                    last_close = to_ts(segments[-1].end) if segments else None
                venues[key] = _Venue(key, state.is_open, state.phase == "break", last_close, delay_s)
            except KeyError:
                log.debug("venue %s is unknown to the calendar", key)
        return venues

    def _venue_delays(self) -> dict[str, int]:
        delays: dict[str, int] = {}
        for instrument in self._active_instruments():
            delays[instrument.exchange] = max(delays.get(instrument.exchange, 0), instrument.delay_min * 60)
        return delays

    def _symbols_by_venue(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for instrument in self._active_instruments():
            if instrument.symbol not in self._quarantine:
                groups.setdefault(instrument.exchange, []).append(instrument.symbol)
        return groups

    def _active_instruments(self) -> list[Instrument]:
        return self._instruments.all()

    def _any_venue_open(self, now: datetime) -> bool:
        for venue in self._venue_delays():
            with contextlib.suppress(KeyError):
                if self._calendar.is_open(venue, now):
                    return True
        return False

    # ------------------------------------------------------------------------------------------
    # REST execution
    # ------------------------------------------------------------------------------------------
    async def _delayed_rest(self, jobs: list[_RestJob]) -> None:
        await asyncio.sleep(self._opts.rest_offset_s)
        await self._run_rest(jobs)

    async def _run_rest(self, jobs: Sequence[_RestJob]) -> None:
        """Execute planned REST jobs in order; stops at a rate limit and puts the session-final and daily
        jobs it could not run back at the front of their queues. Never raises."""
        for index, job in enumerate(jobs):
            try:
                if job.kind == "daily":
                    await self._daily_job(job.symbols, job.start_ts)
                else:
                    await self._minutes_job(job)
            except RateLimitError:
                self._requeue(jobs[index:])
                break
            except ProviderError as exc:
                log.warning("REST %s request for %d symbols failed: %s", job.kind, len(job.symbols), exc)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("REST %s job failed", job.kind)
        await self._check_pause()

    def _requeue(self, jobs: Sequence[_RestJob]) -> None:
        """Return refused session-final and daily jobs to the front of their queues (order kept). Refused
        reconciliation symbols need nothing: their next window starts at the end of the last good one."""
        for job in reversed(jobs):
            if job.kind == "final" and job.close_ts is not None:
                self._final_queue.extendleft(reversed([(job.close_ts, symbol) for symbol in job.symbols]))
            elif job.kind == "daily":
                self._daily_queue.extendleft(reversed([s for s in job.symbols if s not in self._daily_queue]))

    async def _minutes_job(self, job: _RestJob) -> None:
        request_ts = self._clock.ts()
        frames = await self._provider.fetch_minutes(job.symbols, start_ts=job.start_ts, prepost=job.kind == "final")
        self._mark_rest_ok()
        frames = {
            symbol: self._storable_minutes(symbol, frame, request_ts, live=True, last_ts=job.close_ts)
            for symbol, frame in frames.items()
        }
        if job.kind == "minutes" and self._mass_failure(job.symbols, frames, request_ts):
            self._provider.trip("no minute bars for most open-market symbols")
            raise RateLimitError("empty answers for most open-market symbols")
        await self._db_call(self._write_rest_frames, frames, live=True)
        if job.kind == "minutes":
            for symbol in job.symbols:
                self._rest_until[symbol] = request_ts
            for symbol, hole in job.repairs:  # repaired, unless a new hole was reported meanwhile
                frame = frames.get(symbol)
                answered = frame is not None and not frame.empty
                # An empty answer for a symbol that must have bars may be a silent refusal: keep the hole
                # marked so that the next round asks again (FX/futures holes are accepted, see _expects_bars).
                if self._repair_from.get(symbol) == hole and (answered or not self._expects_bars(symbol, request_ts)):
                    del self._repair_from[symbol]

    async def _daily_job(self, symbols: Sequence[str], start_ts: int) -> None:
        frames = await self._provider.fetch_daily(symbols, start_ts=start_ts)
        self._mark_rest_ok()
        now = self._clock.now()
        completed = {symbol: self._completed_days(symbol, frame, now) for symbol, frame in frames.items()}
        await self._db_call(self._write_daily_frames, completed)
        waiting = [symbol for symbol in symbols if symbol in self._daily_missing]
        if waiting:
            stored = await self._db_call(self._last_days, waiting)
            for symbol in waiting:
                retry = self._daily_missing.get(symbol)
                last_day = stored.get(symbol)
                if retry is not None and last_day is not None and last_day >= retry.session.isoformat():
                    del self._daily_missing[symbol]

    def _storable_minutes(
        self, symbol: str, frame: pd.DataFrame, request_ts: int, *, live: bool, last_ts: int | None = None
    ) -> pd.DataFrame:
        """Closed bars only (open + 60 <= request time), up to ``last_ts`` if given, volume rule applied.

        For a ``live`` request (no window end) the newest row is dropped as well while it can still be the
        source's forming minute: Yahoo merges its latest price into the minute that holds it, which is
        ``now - feed delay`` for a delayed venue. A complete bar dropped this way is stored by the next round.
        """
        instrument = self._instruments.get(symbol)
        delay_s = instrument.delay_min * 60 if instrument is not None else 0
        index = frame.index.to_numpy()
        keep = index + BAR_SECONDS <= request_ts
        if live and self._opts.drop_forming_live_row and len(index) and index[-1] >= request_ts - delay_s - 2 * BAR_SECONDS:
            keep[-1] = False
        if last_ts is not None:
            keep &= index <= last_ts
        frame = frame[keep]
        if instrument is not None and instrument.asset_class in self._opts.zero_volume_classes and not frame.empty:
            frame = frame.assign(volume=0.0)
        return frame

    def _completed_days(self, symbol: str, frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
        """Daily rows of sessions that had completely ended at ``now`` (a still-forming row is dropped)."""
        instrument = self._instruments.get(symbol)
        try:
            last = None if instrument is None else self._calendar.last_completed_session(instrument.exchange, now)
        except KeyError:
            last = None
        if last is None:
            return frame.iloc[0:0]
        return frame[frame.index <= last.isoformat()]

    def _mass_failure(self, symbols: Sequence[str], frames: Mapping[str, pd.DataFrame], request_ts: int) -> bool:
        """yf.download hides rate limits and network errors behind empty frames: treat 'no data for most
        symbols whose venue traded during the window' as a failure."""
        expected = [symbol for symbol in symbols if self._expects_bars(symbol, request_ts)]
        if len(expected) < self._opts.mass_failure_min_symbols:
            return False
        empty = sum(1 for symbol in expected if symbol not in frames or frames[symbol].empty)
        return empty * 2 > len(expected)

    def _expects_bars(self, symbol: str, request_ts: int) -> bool:
        """True if minute bars must exist for the window: the venue traded during the last minutes (feed delay
        considered). FX pairs and futures never count: their nominal 24h sessions contain real phases without
        trades for many symbols (exotic pairs on Sunday evening, illiquid contract hours), so empty answers for
        them say nothing about the health of the data source."""
        instrument = self._instruments.get(symbol)
        if instrument is None or instrument.asset_class in ("fx", "future"):
            return False
        lagged = request_ts - instrument.delay_min * 60
        try:
            return self._calendar.is_open(instrument.exchange, from_ts(lagged - 600)) and self._calendar.is_open(
                instrument.exchange, from_ts(lagged - 120)
            )
        except KeyError:
            return False

    def _mark_rest_ok(self) -> None:
        with self._lock:
            self._last_rest_ok_ts = self._clock.ts()

    # ------------------------------------------------------------------------------------------
    # History (backfill and ensure_history)
    # ------------------------------------------------------------------------------------------
    def _plan_history(self, now_ts: int, instruments: Sequence[Instrument]) -> list[_HistoryJob]:
        """(db thread) Jobs for the history the database lacks: daily, recent 1-minute window, older windows."""
        ordered = sorted(
            (i for i in instruments if i.symbol not in self._quarantine),
            key=lambda i: (not i.tradable, i.sort_order, i.symbol),
        )
        now = from_ts(now_ts)
        minute = floor_ts(now_ts)
        batch = self._opts.backfill_batch
        jobs: list[_HistoryJob] = []

        full: list[str] = []
        updates: list[tuple[str, int]] = []
        for instrument in ordered:
            last_day = repo_market.last_day(self._db, instrument.symbol)
            if last_day is None:
                full.append(instrument.symbol)
                continue
            with contextlib.suppress(KeyError):
                completed = self._calendar.last_completed_session(instrument.exchange, now)
                if completed is not None and last_day < completed.isoformat():
                    updates.append((instrument.symbol, _day_start_ts(last_day) - 5 * DAY_S))
        full_start = now_ts - int(self._opts.daily_years * 365.25 * DAY_S)
        jobs.extend(_HistoryJob("daily", symbols, full_start) for symbols in _batches(full, batch))
        for index in range(0, len(updates), batch):
            chunk = updates[index : index + batch]
            jobs.append(_HistoryJob("daily", tuple(s for s, _ in chunk), min(start for _, start in chunk)))

        recent_start = minute - self._opts.minute_window_s
        recent: list[tuple[str, int]] = []
        older: dict[tuple[int, int], list[str]] = {}
        oldest = minute - self._opts.minute_lookback_s
        for instrument in ordered:
            span = repo_market.bar_range(self._db, instrument.symbol)
            if span is None:
                recent.append((instrument.symbol, recent_start))
                windows = self._older_windows(instrument.exchange, recent_start, oldest)
            else:
                first, last = span
                if self._traded_between(instrument.exchange, last + BAR_SECONDS, now_ts):
                    recent.append((instrument.symbol, max(recent_start, last - self._opts.recon_overlap_s)))
                windows = self._older_windows(instrument.exchange, first, oldest)
                if last < recent_start:  # a downtime longer than the recent window left a gap
                    gap_start = max(oldest, last - self._opts.recon_overlap_s)
                    windows += self._older_windows(instrument.exchange, recent_start, gap_start)
            for window in windows:
                older.setdefault(window, []).append(instrument.symbol)
        recent.sort(key=lambda item: item[1])
        for index in range(0, len(recent), batch):
            chunk = recent[index : index + batch]
            jobs.append(_HistoryJob("recent", tuple(s for s, _ in chunk), min(start for _, start in chunk)))
        for (start, end), symbols in sorted(older.items(), key=lambda item: -item[0][1]):
            jobs.extend(_HistoryJob("older", chunk, start, end) for chunk in _batches(symbols, batch))
        return jobs

    def _older_windows(self, venue: str, until_ts: int, oldest_ts: int) -> list[tuple[int, int]]:
        """Minute-aligned, overlapping windows of at most 8 days from ``until_ts`` (overlapping it) back to
        ``oldest_ts``, newest first; windows without any trading time of the venue are left out."""
        windows = []
        end = floor_ts(until_ts) + self._opts.recon_overlap_s
        while end - self._opts.recon_overlap_s > oldest_ts:
            start = max(oldest_ts, end - self._opts.minute_window_s)
            if self._traded_between(venue, start, end - self._opts.recon_overlap_s):
                windows.append((start, end))
            end = start + self._opts.recon_overlap_s
            if start == oldest_ts:
                break
        return windows

    def _traded_between(self, venue: str, start_ts: int, end_ts: int) -> bool:
        if end_ts <= start_ts:
            return False
        try:
            return bool(self._calendar.segments(venue, from_ts(start_ts), from_ts(end_ts)))
        except KeyError:
            return False

    async def _run_history(self, jobs: Sequence[_HistoryJob], *, track_progress: bool) -> None:
        for done, job in enumerate(jobs):
            if track_progress:
                with self._lock:
                    self._backfill_progress = done / len(jobs)
            async with self._history_lock:
                await self._history_job(job)
                await asyncio.sleep(self._opts.backfill_pause_s * len(job.symbols))

    async def _history_job(self, job: _HistoryJob) -> None:
        """Run one history job; waits out REST pauses and retries, skips the job on other failures."""
        while True:
            await self._wait_for_pause_end()
            try:
                if job.kind == "daily":
                    await self._daily_job(job.symbols, job.start_ts)
                else:
                    await self._history_minutes(job)
                return
            except RateLimitError:
                await self._check_pause()
                if self._provider.paused_until() is None:  # refused without a pause: do not spin
                    log.warning("history %s job for %s refused by the source", job.kind, ", ".join(job.symbols))
                    return
            except (ProviderError, ValueError) as exc:
                log.warning("history %s job for %s failed: %s", job.kind, ", ".join(job.symbols), exc)
                return
            except Exception:  # one bad answer must not end the whole backfill
                log.exception("history %s job for %s failed", job.kind, ", ".join(job.symbols))
                return

    async def _history_minutes(self, job: _HistoryJob) -> None:
        """Fetch and store one 1-minute history job. A job up to now ("recent") that ran so late that its
        start is more than one window back (pauses, a long queue) first fetches the overdue part as closed,
        overlapping windows: a hole in the middle of the history would never be noticed later."""
        start = job.start_ts
        if job.end_ts is None:
            start = max(start, self._clock.ts() - self._opts.minute_lookback_s)
            while start < self._clock.ts() - self._opts.minute_window_s:
                end = start + self._opts.minute_window_s
                await self._fetch_history_window(job.symbols, start, end)
                start = end - self._opts.recon_overlap_s
        request_ts = await self._fetch_history_window(job.symbols, start, job.end_ts)
        if job.kind == "recent":
            for symbol in job.symbols:
                self._rest_until[symbol] = request_ts
                self._rest_fetched[symbol] = request_ts

    async def _fetch_history_window(self, symbols: Sequence[str], start_ts: int, end_ts: int | None) -> int:
        """Fetch ``[start_ts, end_ts)`` (``None``: up to now) and store the closed bars; returns the request time."""
        request_ts = self._clock.ts()
        live = end_ts is None
        if live:
            start_ts = max(start_ts, request_ts - self._opts.minute_window_s)
        frames = await self._provider.fetch_minutes(symbols, start_ts=start_ts, end_ts=end_ts)
        self._mark_rest_ok()
        frames = {
            symbol: self._storable_minutes(symbol, frame, request_ts, live=live) for symbol, frame in frames.items()
        }
        await self._db_call(self._write_rest_frames, frames, live=False)
        return request_ts

    async def _wait_for_pause_end(self) -> None:
        while (paused := self._provider.paused_until()) is not None:
            await self._check_pause()
            await asyncio.sleep(min(60.0, max(1.0, paused - self._clock.ts())))
        await self._check_pause()

    # ------------------------------------------------------------------------------------------
    # Database thread
    # ------------------------------------------------------------------------------------------
    async def _db_call[T](self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._db_executor, functools.partial(func, *args, **kwargs))

    def _write_live_bars(self, bars: Iterable[Bar]) -> dict[str, int]:
        """(db thread) Store stream/quote bars one by one in one transaction; announce the written ones.

        Returns the traded holes the written bars revealed (see ``_live_holes``)."""
        with self._lock:
            newest_before = dict(self._last_bar_ts)
        written: list[Bar] = []
        with self._db.write():
            for bar in bars:
                if repo_market.upsert_bar_rows(self._db, [bar]):
                    written.append(bar)
        holes = self._live_holes(written, newest_before)
        for bar in written:
            self._register_bar(bar.symbol, bar.ts)
            self._publish_bar(bar.symbol, bar.ts, (bar.open, bar.high, bar.low, bar.close, bar.volume), bar.source)
        return holes

    def _live_holes(self, written: Sequence[Bar], newest_before: Mapping[str, int]) -> dict[str, int]:
        """(db thread) Symbol -> open time of the first missing minute of the oldest hole the written bars left.

        Only bars that extend a symbol's series (newer than its newest bar before this write) are checked: a
        hole is left when such a bar does not directly follow the previous stored bar and the venue traded
        in between (regular segments of the calendar). Bars that fill in or correct older minutes never
        report a hole, so every hole is reported once, by the bar right after it."""
        holes: dict[str, int] = {}
        for bar in sorted(written, key=lambda b: (b.symbol, b.ts)):
            newest = newest_before.get(bar.symbol)
            if bar.symbol in holes or (newest is not None and bar.ts <= newest):
                continue
            previous = repo_market.last_bar(self._db, bar.symbol, before_ts=bar.ts)
            if previous is None or bar.ts - previous.ts <= BAR_SECONDS:
                continue
            instrument = self._instruments.get(bar.symbol)
            if instrument is not None and self._traded_between(
                instrument.exchange, previous.ts + BAR_SECONDS, bar.ts
            ):
                holes[bar.symbol] = previous.ts + BAR_SECONDS
        return holes

    def _write_rest_frames(self, frames: Mapping[str, pd.DataFrame], *, live: bool) -> None:
        """(db thread) Store REST frames. Live rounds announce the bars at or after the previous newest bar."""
        with self._db.write():
            for symbol, frame in frames.items():
                if frame.empty:
                    continue
                repo_market.upsert_bars_1m(self._db, symbol, frame, "rest")
        for symbol, frame in frames.items():
            if frame.empty:
                continue
            with self._lock:
                previous = self._last_bar_ts.get(symbol)
            if live:
                tail = frame if previous is None else frame[frame.index >= previous]
                for ts, row in zip(tail.index.tolist(), tail.itertuples(index=False, name=None), strict=True):
                    self._publish_bar(symbol, ts, row, "rest")
            self._register_bar(symbol, int(frame.index[-1]))

    def _write_daily_frames(self, frames: Mapping[str, pd.DataFrame]) -> None:
        with self._db.write():
            for symbol, frame in frames.items():
                if not frame.empty:
                    repo_market.upsert_bars_1d(self._db, symbol, frame)

    def _last_days(self, symbols: Iterable[str]) -> dict[str, str | None]:
        """(db thread) Session date of the newest stored daily bar per symbol."""
        return {symbol: repo_market.last_day(self._db, symbol) for symbol in symbols}

    def _register_bar(self, symbol: str, ts: int) -> None:
        with self._lock:
            if ts > self._last_bar_ts.get(symbol, -1):
                self._last_bar_ts[symbol] = ts
                self._updated.add(symbol)

    def _publish_bar(self, symbol: str, ts: int, values: Sequence[float], source: str) -> None:
        self._bus.publish(TOPIC_BAR, {"symbol": symbol, "bar": [int(ts), *map(float, values)], "source": source})

    async def _activity(self, level: str, message: str) -> None:
        try:
            await self._db_call(self._log_activity, level, message)
        except Exception:
            log.exception("could not write activity entry")

    def _activity_soon(self, level: str, message: str) -> None:
        """Activity entry from a synchronous callback (written in the database thread, errors logged)."""
        try:
            future = self._db_executor.submit(self._log_activity, level, message)
        except RuntimeError:  # executor already shut down
            return
        future.add_done_callback(_log_future_failure)

    def _log_activity(self, level: str, message: str) -> None:
        repo_ops.log_activity(
            self._db, ts=self._clock.ts(), level=level, category="data", message=message, bus=self._bus
        )

    # ------------------------------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------------------------------
    def _is_stale(
        self, instrument: Instrument, now: datetime, now_ts: int, quotes: Mapping[str, Quote], last_bars: Mapping[str, int]
    ) -> bool:
        allowed_min = instrument.delay_min + self._opts.stale_margin_min
        try:
            since_open = self._calendar.minutes_since_open(instrument.exchange, now)
        except KeyError:
            return False
        if since_open is None or since_open < allowed_min:
            return False
        quote = quotes.get(instrument.symbol)
        last_bar = last_bars.get(instrument.symbol)
        newest = max((quote.ts if quote else -1), (last_bar + BAR_SECONDS if last_bar is not None else -1))
        return now_ts - newest > allowed_min * 60

    # ------------------------------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------------------------------
    async def _quote_loop(self) -> None:
        while True:
            try:
                await self.poll_quotes(self._clock.ts())
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("quote poll failed")
            last = self._last_quote_poll if self._last_quote_poll is not None else self._clock.ts()
            wait = last + self._opts.quote_interval_open_s - self._clock.ts()
            await asyncio.sleep(min(float(self._opts.quote_interval_open_s), max(1.0, wait)))

    async def _publish_loop(self) -> None:
        while True:
            await asyncio.sleep(self._opts.publish_interval_s)
            try:
                self.publish_quotes()
            except Exception:
                log.exception("quote publish failed")


# ------------------------------------------------------------------------------------------------
# Several data sources
# ------------------------------------------------------------------------------------------------
def _claims_nothing(symbol: str) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class Lane:
    """One data source of ``RoutedMarketData``: its service and the instruments it serves.

    ``owns`` decides for instruments of the book; ``claims`` for symbols that are not in the book yet
    (``lookup`` of a symbol the user wants to add)."""

    name: str
    service: MarketDataService
    owns: Callable[[Instrument], bool]
    claims: Callable[[str], bool] = _claims_nothing


class RoutedMarketData:
    """``contracts.MarketData`` over one ``MarketDataService`` per data source (see the module docstring).

    Every instrument is served by the first lane that owns it; the FIRST lane is also the default for symbols
    no lane owns (retired instruments, unknown symbols). The lanes run independently: ``refresh`` runs them
    side by side, a failing lane is logged and never stops the others. ``status`` combines them: the stream
    counts as connected only if the stream of EVERY lane that serves active symbols is connected, the REST
    time is the oldest of those lanes (None until each of them succeeded once), a rate limit of any lane and
    the slowest running backfill are reported.
    """

    def __init__(self, lanes: Sequence[Lane], instruments: InstrumentBook) -> None:
        if not lanes:
            raise ValueError("at least one lane is needed")
        self._lanes = tuple(lanes)
        self._instruments = instruments

    @property
    def lanes(self) -> tuple[Lane, ...]:
        return self._lanes

    def lane_for(self, symbol: str) -> Lane:
        instrument = self._instruments.get(symbol)
        if instrument is not None:
            for lane in self._lanes:
                if lane.owns(instrument):
                    return lane
        return self._lanes[0]

    # -- MarketData -----------------------------------------------------------------------------
    async def start(self) -> None:
        for lane in self._lanes:  # one after the other: the first lane seeds an empty instruments table
            try:
                await lane.service.start()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("market data source %s failed to start", lane.name)

    async def stop(self) -> None:
        results = await asyncio.gather(*(lane.service.stop() for lane in self._lanes), return_exceptions=True)
        for lane, result in zip(self._lanes, results, strict=True):
            if isinstance(result, Exception):
                log.error("market data source %s did not stop cleanly: %s", lane.name, result)

    async def refresh(self, now: datetime) -> list[str]:
        results = await asyncio.gather(*(lane.service.refresh(now) for lane in self._lanes), return_exceptions=True)
        updated: set[str] = set()
        for lane, result in zip(self._lanes, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                log.error("market data source %s: refresh failed: %s", lane.name, result)
                continue
            updated.update(result)
        return sorted(updated)

    def quote(self, symbol: str) -> Quote | None:
        return self.lane_for(symbol).service.quote(symbol)

    def quotes(self) -> dict[str, Quote]:
        merged: dict[str, Quote] = {}
        for lane in self._lanes:
            for symbol, quote in lane.service.quotes().items():
                if self.lane_for(symbol) is lane:
                    merged[symbol] = quote
        return merged

    def status(self) -> DataStatusOut:
        statuses = [lane.service.status() for lane in self._lanes]
        serving = [status for status in statuses if status.symbols_active > 0]
        rest = [status.last_rest_ok_ts for status in serving]
        return DataStatusOut(
            stream_connected=bool(serving) and all(status.stream_connected for status in serving),
            last_tick_ts=max((s.last_tick_ts for s in statuses if s.last_tick_ts is not None), default=None),
            last_rest_ok_ts=None if not rest or any(ts is None for ts in rest) else min(ts for ts in rest if ts),
            rate_limited_until=max(
                (s.rate_limited_until for s in statuses if s.rate_limited_until is not None), default=None
            ),
            symbols_active=sum(status.symbols_active for status in statuses),
            symbols_stale=sum(status.symbols_stale for status in statuses),
            backfill_progress=min(
                (s.backfill_progress for s in statuses if s.backfill_progress is not None), default=None
            ),
        )

    async def lookup(self, symbol: str) -> Instrument | None:
        key = symbol.strip().upper()
        if self._instruments.get(key) is not None:
            lane = self.lane_for(key)
        else:
            lane = next((lane for lane in self._lanes if lane.claims(key)), self._lanes[0])
        return await lane.service.lookup(key)

    async def ensure_history(self, symbol: str) -> None:
        await self.lane_for(symbol).service.ensure_history(symbol)

    async def earnings(self, symbol: str) -> list[EarningsDate] | None:
        return await self.lane_for(symbol).service.earnings(symbol)


__all__ = ["KRAKEN_OPTIONS", "DataOptions", "InstrumentView", "Lane", "MarketDataService", "RoutedMarketData"]
