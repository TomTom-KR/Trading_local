"""Crypto derivatives: funding, open interest and liquidations of the Kraken Futures perpetual of every active
crypto instrument (implements ``contracts.Derivatives``).

For crypto, positioning moves prices faster than headlines: crowded longs pay a high funding rate, a quickly
rising open interest means fresh leverage, and liquidations cascade within minutes. The news coverage of the
coins is thin (in the first days 78 analysed items for BTC, 9 for ETH, 4 for SOL), so these data are the
time-critical input for crypto.

Once a minute (own loop of the runtime, ``poll``):
1. ONE tickers request: a snapshot per instrument (funding, predicted funding, open interest, mark and index
   price) at the current minute.
2. Per instrument the 1-minute liquidation volume since the newest stored minute; the last
   ``LIQ_REFETCH_MIN`` minutes are fetched again (the newest values may still change), only complete minutes
   are stored.
3. Liquidation burst -> price trigger: when the last 5 complete minutes liquidated at least
   ``LIQ_BURST_RATIO`` times the mean 5-minute volume of the last 7 days, more than 99.9 % of those 5-minute
   buckets and at least ``LIQ_MIN_USD``, the coin's news are pulled forward (``News.prioritize``) and an
   activity entry is written. On the real history of 2026-11-18 to 2026-11-24 (a sell-off week) this fired
   about 1.7 times a day over all ten coins; with the 99 % quantile and 50,000 USD it fired 7.8 times a day. At least one day of liquidation history is required; one trigger per
   instrument per ``TRIGGER_COOLDOWN_S``; switched with ``RuntimeSettings.price_trigger_enabled``.

Start: a background backfill per instrument - realised hourly funding of the last ``FUNDING_BACKFILL_DAYS``
days (skipped when the database already holds it), hourly open interest of the last ``OI_BACKFILL_DAYS``
days (5-minute values for the last 26 hours) and 1-minute liquidations of the last ``LIQ_BACKFILL_DAYS`` days (from the newest stored minute on) - so
that rankings and the burst rule work from the first minutes. Backfilled rows never overwrite live rows.

The metrics (``repo_derivs.metrics``) go into the trade review of crypto candidates, into the feature snapshot
of every crypto signal and into the GUI. They do not change the engine's rules (the shadow journal shows
later whether they would have helped).

Threads: database work runs in one dedicated worker thread; the event loop does the HTTP requests.
"""

import asyncio
import functools
import logging
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Final

from trading_local.analysis.readings import de_number
from trading_local.clock import Clock, to_ts
from trading_local.contracts import News
from trading_local.data.kraken_futures import KrakenFuturesClient, perpetual_symbol
from trading_local.data.provider import ProviderError, RateLimitError
from trading_local.db import repo_derivs, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_derivs import DerivMetrics, DerivSnapshot
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus

log = logging.getLogger(__name__)

LIQ_REFETCH_MIN: Final = 3
LIQ_LIVE_LOOKBACK_S: Final = 3600  # without stored liquidations a live poll starts this far back
FUNDING_BACKFILL_DAYS: Final = 60
OI_BACKFILL_DAYS: Final = 30
OI_FINE_BACKFILL_S: Final = 26 * 3600  # 5-minute open interest of the last 26 hours
LIQ_BACKFILL_DAYS: Final = 7
LIQ_BURST_RATIO: Final = 10.0
LIQ_BURST_QUANTILE: Final = 0.999
LIQ_MIN_USD: Final = 100_000.0
LIQ_MIN_HISTORY_S: Final = 86_400  # the burst rule needs at least one day of liquidation minutes
TRIGGER_COOLDOWN_S: Final = 1800
_DAY_S: Final = 86_400


def burst_message_de(symbol: str, volume: float, usd: float, ratio: float | None, requeued: int) -> str:
    """Activity-log text, e.g. 'Liquidationswelle: BTC-EUR 2,40 BTC (≈ 143.600 USD) in 5 Min, 18-fach üblich –
    Nachrichten vorgezogen'."""
    base = symbol.split("-", 1)[0]
    digits = 2 if volume < 100 else 0
    usual = "" if ratio is None else f", {de_number(ratio, 0)}-fach üblich"
    text = (
        f"Liquidationswelle: {symbol} {de_number(volume, digits)} {base} (≈ {de_number(usd, 0)} USD) in 5 Min"
        f"{usual} – Nachrichten vorgezogen"
    )
    if requeued == 1:
        return text + ", 1 Meldung erneut eingereiht"
    if requeued > 1:
        return text + f", {requeued} Meldungen erneut eingereiht"
    return text


def is_burst(window: float, reference: list[float], usd: float) -> bool:
    """The burst rule on the last 5 minutes (``window``) and the 5-minute sums of the reference period."""
    if window <= 0 or usd < LIQ_MIN_USD or not reference:
        return False
    usual = statistics.fmean(reference)
    ordered = sorted(reference)
    top = ordered[min(len(ordered) - 1, int(LIQ_BURST_QUANTILE * len(ordered)))]
    return window >= LIQ_BURST_RATIO * usual and window > top


class KrakenDerivatives:
    """Implements ``contracts.Derivatives``. ``start`` / ``poll`` / ``stop`` run on the event loop; ``metrics``
    may be called from any thread."""

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        bus: EventBus,
        instruments: InstrumentBook,
        runtime_settings: SettingsStore,
        news: News,
        client_factory: Callable[[], KrakenFuturesClient] | None = None,
    ) -> None:
        self._db = db
        self._clock = clock
        self._bus = bus
        self._instruments = instruments
        self._runtime_settings = runtime_settings
        self._news = news
        self._client_factory = client_factory or (lambda: KrakenFuturesClient(clock))
        self._client: KrakenFuturesClient | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="derivatives")
        self._backfill_task: asyncio.Task[None] | None = None
        self._poll_lock = asyncio.Lock()
        self._closed = False
        self._last_trigger: dict[str, int] = {}

    # -- life cycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        """Create the client and start the background backfill. Returns at once; never raises."""
        if self._closed or self._client is not None:
            return
        try:
            self._client = self._client_factory()
            self._backfill_task = asyncio.create_task(self._backfill(), name="derivatives-backfill")
        except Exception:
            log.exception("derivatives: start failed")

    async def stop(self) -> None:
        self._closed = True
        task, self._backfill_task = self._backfill_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._client is not None:
            await self._client.close()
            self._client = None
        await asyncio.to_thread(self._worker.shutdown, True)

    async def wait_backfill(self) -> None:
        """Wait until the start-up backfill has finished (tests and tools)."""
        task = self._backfill_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    # -- contracts.Derivatives ------------------------------------------------------------------
    def targets(self) -> dict[str, str]:
        """Active crypto instruments -> their perpetual."""
        found: dict[str, str] = {}
        for instrument in self._instruments.all(active_only=True):
            perpetual = perpetual_symbol(instrument)
            if perpetual is not None:
                found[instrument.symbol] = perpetual
        return found

    def metrics(self, symbol: str, asof_ts: int | None = None) -> DerivMetrics | None:
        """The derivatives picture of ``symbol`` as known at ``asof_ts`` (default: now); None for instruments
        without a perpetual or without data. Thread-safe; reads the database."""
        if symbol not in self.targets():
            return None
        return repo_derivs.metrics(self._db, symbol, asof_ts=self._clock.ts() if asof_ts is None else asof_ts)

    async def poll(self, now: datetime) -> int:
        """Snapshot, liquidations and the burst rule (see the module docstring). Never raises; returns the number
        of rows written."""
        if self._closed or self._client is None or self._poll_lock.locked():
            return 0
        async with self._poll_lock:
            try:
                return await self._poll(to_ts(now), now)
            except Exception:
                if not self._closed:
                    log.exception("derivatives: poll failed")
                return 0

    # -- polling --------------------------------------------------------------------------------
    async def _poll(self, now_ts: int, now: datetime) -> int:
        client = self._client
        targets = self.targets()
        if client is None or not targets:
            return 0
        minute = now_ts // 60 * 60
        written = 0
        try:
            tickers = await client.tickers()
        except (ProviderError, RateLimitError) as exc:
            log.warning("derivatives: tickers unavailable: %s", exc)
            tickers = {}
        rows = []
        for symbol, perpetual in targets.items():
            ticker = tickers.get(perpetual)
            if ticker is None or ticker.suspended or ticker.mark_price is None:
                continue
            rows.append(
                DerivSnapshot(
                    symbol=symbol,
                    ts=minute,
                    received_ts=now_ts,
                    funding_rate=ticker.funding_rate,
                    funding_pred=ticker.funding_pred,
                    open_interest=ticker.open_interest,
                    mark_price=ticker.mark_price,
                    index_price=ticker.index_price,
                )
            )
        written += await self._in_worker(repo_derivs.upsert_snapshots, self._db, rows)
        for symbol, perpetual in targets.items():
            newest = await self._in_worker(repo_derivs.newest_ts, self._db, "deriv_liquidations", symbol)
            since = minute - LIQ_LIVE_LOOKBACK_S if newest is None else max(minute - LIQ_LIVE_LOOKBACK_S,
                                                                            newest - LIQ_REFETCH_MIN * 60)  # fmt: skip
            try:
                points = await client.series(perpetual, "liquidation-volume", interval_s=60, since_ts=since)
            except (ProviderError, RateLimitError) as exc:
                log.warning("derivatives: liquidations of %s unavailable: %s", perpetual, exc)
                continue
            complete = [(ts, volume) for ts, volume in points if ts + 60 <= now_ts]
            written += await self._in_worker(
                repo_derivs.upsert_liquidations, self._db, symbol, complete, received_ts=now_ts
            )
        if self._runtime_settings.get().price_trigger_enabled:
            await self._check_bursts(list(targets), now_ts, now)
        return written

    async def _check_bursts(self, symbols: list[str], now_ts: int, now: datetime) -> None:
        for symbol in symbols:
            if now_ts - self._last_trigger.get(symbol, -TRIGGER_COOLDOWN_S) < TRIGGER_COOLDOWN_S:
                continue
            try:
                burst = await self._in_worker(self._burst, symbol, now_ts)
                if burst is None:
                    continue
                self._last_trigger[symbol] = now_ts
                volume, usd, ratio = burst
                log.info("derivatives: liquidation burst %s: %.4g contracts (%.0f USD)", symbol, volume, usd)
                requeued = await self._news.prioritize([symbol], now)
                await self._in_worker(
                    repo_ops.log_activity,
                    self._db,
                    ts=now_ts,
                    level="info",
                    category="data",
                    message=burst_message_de(symbol, volume, usd, ratio, requeued),
                    bus=self._bus,
                )
            except Exception:
                log.exception("derivatives: burst check of %s failed", symbol)

    def _burst(self, symbol: str, now_ts: int) -> tuple[float, float, float | None] | None:
        """(contracts, USD, ratio to the usual 5 minutes) of a liquidation burst in the last 5 minutes, or None."""
        minute_end = now_ts // 60 * 60
        window_start = minute_end - 300
        oldest = repo_derivs.first_liquidation_ts(self._db, symbol, asof_ts=now_ts)
        if oldest is None or oldest > window_start - LIQ_MIN_HISTORY_S:
            return None
        window = repo_derivs.liquidations_between(
            self._db, symbol, start_ts=window_start, end_ts=minute_end, asof_ts=now_ts
        )
        if window <= 0:
            return None
        reference = repo_derivs.liquidation_buckets(
            self._db, symbol, start_ts=max(oldest, window_start - repo_derivs.LIQ_REFERENCE_S),
            end_ts=window_start, asof_ts=now_ts,
        )  # fmt: skip
        mark = self._db.scalar(
            "SELECT mark_price FROM deriv_snapshots WHERE symbol = ? AND ts <= ? AND mark_price IS NOT NULL "
            "ORDER BY ts DESC LIMIT 1",
            (symbol, now_ts),
        )
        usd = window * float(mark) if mark else 0.0
        if not is_burst(window, reference, usd):
            return None
        usual = statistics.fmean(reference)
        return window, usd, (window / usual if usual > 0 else None)

    # -- backfill -------------------------------------------------------------------------------
    async def _backfill(self) -> None:
        for symbol, perpetual in self.targets().items():
            if self._closed:
                return
            try:
                await self._backfill_symbol(symbol, perpetual)
            except asyncio.CancelledError:
                raise
            except (ProviderError, RateLimitError) as exc:
                log.warning("derivatives: backfill of %s incomplete: %s", perpetual, exc)
            except Exception:
                log.exception("derivatives: backfill of %s failed", perpetual)
        log.info("derivatives: backfill finished")

    async def _backfill_symbol(self, symbol: str, perpetual: str) -> None:
        client = self._client
        if client is None:
            return
        now_ts = self._clock.ts()
        if not await self._in_worker(self._has_funding_history, symbol, now_ts):
            history = await client.funding_history(perpetual)
            start = now_ts - FUNDING_BACKFILL_DAYS * _DAY_S
            rows = [
                DerivSnapshot(symbol=symbol, ts=ts, received_ts=now_ts, funding_rate=rate)
                for ts, rate in history
                if start <= ts <= now_ts and ts % 3600 == 0
            ]
            await self._in_worker(repo_derivs.upsert_snapshots, self._db, rows, replace=False)
        for interval_s, since in (
            (3600, now_ts - OI_BACKFILL_DAYS * _DAY_S),
            (300, now_ts - OI_FINE_BACKFILL_S),  # fine enough for the 1-hour change right after a start
        ):
            interest = await client.series(
                perpetual, "open-interest", interval_s=interval_s, since_ts=since, background=True
            )
            rows = [  # a candle's close belongs to the end of its interval
                DerivSnapshot(symbol=symbol, ts=ts + interval_s, received_ts=now_ts, open_interest=value)
                for ts, value in interest
                if ts + interval_s <= now_ts and value >= 0
            ]
            await self._in_worker(repo_derivs.upsert_snapshots, self._db, rows, replace=False)
        newest = await self._in_worker(repo_derivs.newest_ts, self._db, "deriv_liquidations", symbol)
        floor = now_ts - LIQ_BACKFILL_DAYS * _DAY_S
        since = floor if newest is None or newest < floor else newest + 60
        if since < now_ts - 60:
            points = await client.series(perpetual, "liquidation-volume", interval_s=60, since_ts=since, background=True)
            complete = [(ts, volume) for ts, volume in points if ts + 60 <= now_ts]
            await self._in_worker(
                repo_derivs.upsert_liquidations, self._db, symbol, complete, received_ts=now_ts, replace=False
            )

    def _has_funding_history(self, symbol: str, now_ts: int) -> bool:
        """At least 90 % of the hourly funding rows of the last 30 days are stored."""
        count = self._db.scalar(
            "SELECT COUNT(*) FROM deriv_snapshots WHERE symbol = ? AND ts % 3600 = 0 AND ts >= ? "
            "AND funding_rate IS NOT NULL",
            (symbol, now_ts - repo_derivs.FUNDING_WINDOW_S),
            default=0,
        )
        return int(count) >= 0.9 * repo_derivs.FUNDING_WINDOW_S / 3600

    async def _in_worker[T](self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._worker, functools.partial(func, *args, **kwargs))
