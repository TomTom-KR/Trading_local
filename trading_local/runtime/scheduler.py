"""Background scheduler: the minute cycle, the news poll and housekeeping.

The minute cycle runs a few seconds after every full minute in a FIXED order, so that each step sees the
results of the previous one:

    1. data.refresh(now)        closed 1-minute bars and quotes are in the database
    2. broker.process(now)      pending orders fill on the new bars, stops/targets trigger
    3. engine.on_minute(now)    open signals are tracked, new candidates are detected
    4. movers.on_minute(now)    unusual price moves pull the news of their instruments forward (price trigger;
                                last, because it serves the news analysis and must never delay trading)

The news poll, the crypto-derivatives poll and the agenda refresh run on their own once-per-minute loops (slow,
independent HTTP sources) and housekeeping runs hourly. A step that fails is logged and the cycle continues with the next step; a cycle
that overruns the next minute boundary causes the missed minutes to be skipped, never queued.
"""

import asyncio
import contextlib
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from trading_local.barstore import set_live_since
from trading_local.clock import Clock, floor_ts, from_ts
from trading_local.config import Settings
from trading_local.contracts import (
    Agenda,
    Broker,
    Derivatives,
    Engine,
    Llm,
    MarketCalendar,
    MarketData,
    News,
    PriceTrigger,
)
from trading_local.db import repo_derivs, repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.dto import JobStatusOut
from trading_local.events import EventBus

log = logging.getLogger(__name__)

CYCLE_OFFSET_S = 3.0  # run the minute cycle this many seconds after the minute boundary
NEWS_OFFSET_S = 30.0  # poll news in the middle of the minute, away from the market-data burst
DERIVATIVES_OFFSET_S = 20.0  # Kraken Futures: after the minute cycle, before the news poll
AGENDA_OFFSET_S = 45.0  # after the news poll: actual values posted in this minute's headlines are stored
HOUSEKEEPING_INTERVAL_S = 3600.0
OPS_RETENTION_DAYS = 30
# News (and their analyses) are kept long enough for signal reviews and validation replays; 1-minute bars
# are never deleted - the minute history is the platform's most valuable asset.
NEWS_RETENTION_DAYS = 180
# Minute snapshots and liquidations of the perpetuals; the hourly funding rows are kept for good.
DERIVATIVES_RETENTION_DAYS = 90

JOB_LABELS_DE: dict[str, str] = {
    "market_data": "Kurse und Minutenbalken",
    "broker": "Simulation (Orders, Stopps, Ziele)",
    "engine": "Signal-Engine",
    "movers": "Kursauslöser (Nachrichten vorziehen)",
    "derivatives": "Krypto-Derivate (Kraken Futures)",
    "agenda": "Terminkalender",
    "news": "Nachrichten abrufen",
    "housekeeping": "Aufräumen",
}


def install_loop_exception_filter(loop: asyncio.AbstractEventLoop) -> None:
    """Silence one known-harmless event-loop error, pass everything else to the default handler.

    On Windows the proactor loop calls ``socket.shutdown`` while cleaning up a connection that the peer has
    already reset (a closed browser tab, a dropped WebSocket). That raises ``ConnectionResetError`` inside
    ``_call_connection_lost``, which asyncio would log as an ERROR with a traceback although nothing is lost.
    """

    def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, ConnectionResetError) and "_call_connection_lost" in str(context.get("message", "")):
            log.debug("peer reset a connection during cleanup: %s", exc)
            return
        loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


class Runtime:
    """Owns the background tasks. Implements ``services.RuntimeInfo``."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        data: MarketData,
        broker: Broker,
        engine: Engine,
        news: News,
        llm: Llm,
        calendar: MarketCalendar | None = None,
        movers: PriceTrigger | None = None,
        derivatives: Derivatives | None = None,
        agenda: Agenda | None = None,
    ) -> None:
        self.settings = settings
        self.calendar = calendar
        self.db = db
        self.clock = clock
        self.bus = bus
        self.data = data
        self.broker = broker
        self.engine = engine
        self.news = news
        self.llm = llm
        self.movers = movers
        self.derivatives = derivatives
        self.agenda = agenda
        self.started_ts = clock.ts()
        self.offline = settings.offline
        self._tasks: list[asyncio.Task[None]] = []
        self._jobs: dict[str, JobStatusOut] = {}
        self._jobs_lock = threading.Lock()
        self._running = False
        self._last_cycle_minute: int | None = None

    # -- RuntimeInfo ----------------------------------------------------------------------------
    def jobs(self) -> list[JobStatusOut]:
        with self._jobs_lock:
            known = dict(self._jobs)
        return [known.get(job) or JobStatusOut(job=job, label_de=label) for job, label in JOB_LABELS_DE.items()]

    # -- life cycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        """Start the services and the background loops (nothing but the event bus in offline mode)."""
        self.bus.bind_loop()
        install_loop_exception_filter(asyncio.get_running_loop())
        self.started_ts = self.clock.ts()
        if self.calendar is not None:
            # Building the exchange calendars takes a few seconds of CPU. Do it in a worker thread and BEFORE
            # the services start: they query the calendar from the event loop, which would otherwise stall
            # (and starve e.g. the LLM warm-up's connect timeout).
            await self._warm_calendar()
        if self.offline:
            log.info("offline mode: background jobs are disabled")
            await self._activity("info", "Offline-Modus: keine Datenabrufe, keine Analysen.")
            return
        self._running = True
        # Bars stored before this start (or missing because of the downtime) settle only through REST or after
        # a full settle period of live operation - never immediately by age (barstore.settle_horizon).
        set_live_since(self.clock.ts())
        starters = [("llm", self.llm.start), ("market data", self.data.start)]
        if self.derivatives is not None:
            starters.append(("derivatives", self.derivatives.start))
        if self.agenda is not None:
            starters.append(("agenda", self.agenda.start))
        if self.settings.enable_news:
            starters.append(("news", self.news.start))
        for name, starter in starters:
            try:
                await starter()
            except Exception:
                log.exception("failed to start %s", name)
        self._tasks += [
            asyncio.create_task(self._minute_loop(), name="minute-cycle"),
            asyncio.create_task(self._news_loop(), name="news-poll"),
            asyncio.create_task(self._derivatives_loop(), name="derivatives-poll"),
            asyncio.create_task(self._agenda_loop(), name="agenda-refresh"),
            asyncio.create_task(self._housekeeping_loop(), name="housekeeping"),
        ]
        await self._activity("info", "Plattform gestartet.")

    async def stop(self) -> None:
        """Stop the loops first, then the services in reverse start order."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []
        if self.offline:
            return
        set_live_since(None)
        stoppers: list[tuple[str, Callable[[], Awaitable[None]]]] = [("engine", self.engine.stop)]
        if self.movers is not None:
            stoppers.append(("price trigger", self.movers.stop))
        if self.derivatives is not None:
            stoppers.append(("derivatives", self.derivatives.stop))
        if self.agenda is not None:
            stoppers.append(("agenda", self.agenda.stop))
        for name, stopper in (
            *stoppers,
            ("news", self.news.stop),
            ("market data", self.data.stop),
            ("llm", self.llm.stop),
        ):
            try:
                await asyncio.wait_for(stopper(), timeout=15)
            except Exception:
                log.exception("failed to stop %s", name)

    # -- loops ----------------------------------------------------------------------------------
    async def _sleep_until(self, offset_s: float, period_s: float = 60.0) -> None:
        now = time.time()
        target = (now // period_s + 1) * period_s + offset_s
        if target - now > period_s:  # offset already passed in the current period
            target -= period_s
        await asyncio.sleep(max(0.0, target - now))

    async def _minute_loop(self) -> None:
        while self._running:
            await self._sleep_until(CYCLE_OFFSET_S)
            await self.run_cycle()

    async def _news_loop(self) -> None:
        if not self.settings.enable_news:
            return
        while self._running:
            await self._sleep_until(NEWS_OFFSET_S)
            await self._run_job("news", lambda: self.news.poll(self.clock.now()))

    async def _derivatives_loop(self) -> None:
        derivatives = self.derivatives
        if derivatives is None:
            return
        while self._running:
            await self._sleep_until(DERIVATIVES_OFFSET_S)
            await self._run_job("derivatives", lambda: derivatives.poll(self.clock.now()))

    async def _agenda_loop(self) -> None:
        agenda = self.agenda
        if agenda is None:
            return
        while self._running:
            await self._sleep_until(AGENDA_OFFSET_S)
            await self._run_job("agenda", lambda: agenda.refresh(self.clock.now()))

    async def _housekeeping_loop(self) -> None:
        while self._running:
            await asyncio.sleep(HOUSEKEEPING_INTERVAL_S)
            await self._run_job("housekeeping", self._housekeeping)

    # -- the minute cycle -----------------------------------------------------------------------
    async def run_cycle(self) -> bool:
        """Run one minute cycle for the current minute. Returns False if this minute was already processed."""
        now = self.clock.now()
        minute = floor_ts(now.timestamp())
        if self._last_cycle_minute is not None and minute <= self._last_cycle_minute:
            return False
        if self._last_cycle_minute is not None and minute - self._last_cycle_minute > 60:
            skipped = (minute - self._last_cycle_minute) // 60 - 1
            log.warning("minute cycle skipped %d minute(s) (previous cycle overran or the machine slept)", skipped)
        self._last_cycle_minute = minute

        updated: list[str] = []

        async def refresh() -> str:
            nonlocal updated
            updated = await self.data.refresh(now)
            return f"{len(updated)} Symbole mit neuen Balken"

        async def process() -> str:
            closed = await asyncio.to_thread(self.broker.process, now)
            return f"{len(closed)} Positionen geschlossen" if closed else ""

        async def evaluate() -> str:
            await self.engine.on_minute(now, updated)
            return ""

        await self._run_job("market_data", refresh)
        await self._run_job("broker", process)
        await self._run_job("engine", evaluate)
        movers = self.movers
        if movers is not None:

            async def watch() -> str:
                moves = await movers.on_minute(now, updated)
                if not moves:
                    return ""
                return f"{len(moves)} ungewöhnliche Bewegung{'en' if len(moves) > 1 else ''}"

            await self._run_job("movers", watch)
        return True

    async def _run_job(self, job: str, func: Callable[[], Awaitable[Any]]) -> None:
        started = self.clock.ts()
        t0 = time.perf_counter()
        ok, message = True, ""
        try:
            result = await func()
            if isinstance(result, str):
                message = result
            elif isinstance(result, int):
                message = f"{result} neue Einträge" if result else ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ok, message = False, f"{type(exc).__name__}: {exc}"[:300]
            log.exception("job %s failed", job)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        status = JobStatusOut(
            job=job,
            label_de=JOB_LABELS_DE.get(job, job),
            last_started_ts=started,
            last_duration_ms=duration_ms,
            ok=ok,
            message=message,
        )
        with self._jobs_lock:
            self._jobs[job] = status
        try:
            await asyncio.to_thread(
                repo_ops.log_job_run,
                self.db,
                job=job,
                started_ts=started,
                duration_ms=duration_ms,
                ok=ok,
                message=message,
            )
        except Exception:
            log.exception("could not record job run of %s", job)

    async def _warm_calendar(self) -> None:
        calendar = self.calendar
        if calendar is None:
            return
        now = self.clock.now()

        def warm() -> None:
            for exchange in calendar.exchanges():
                calendar.state(exchange, now)

        t0 = time.perf_counter()
        try:
            await asyncio.to_thread(warm)
            log.info("trading calendars ready in %.1f s", time.perf_counter() - t0)
        except Exception:
            log.exception("calendar warm-up failed")

    async def _housekeeping(self) -> str:
        now_ts = self.clock.ts()
        await asyncio.to_thread(repo_ops.prune_ops, self.db, now_ts=now_ts, keep_days=OPS_RETENTION_DAYS)
        deleted = await asyncio.to_thread(
            repo_news.delete_news_before, self.db, now_ts - NEWS_RETENTION_DAYS * 86400
        )
        await asyncio.to_thread(repo_derivs.prune, self.db, before_ts=now_ts - DERIVATIVES_RETENTION_DAYS * 86400)
        return (
            f"Betriebsdaten älter als {OPS_RETENTION_DAYS} Tage und {deleted} Nachrichten älter als "
            f"{NEWS_RETENTION_DAYS} Tage gelöscht ({from_ts(now_ts):%d.%m.%Y})"
        )

    async def _activity(self, level: str, message: str) -> None:
        try:
            await asyncio.to_thread(
                repo_ops.log_activity,
                self.db,
                ts=self.clock.ts(),
                level=level,
                category="system",
                message=message,
                bus=self.bus,
            )
        except Exception:
            log.exception("could not write activity entry")
