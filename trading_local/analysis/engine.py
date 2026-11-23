"""The live signal engine (implements ``contracts.Engine``).

Minute cycle (``on_minute``), always in this order:

1. TRACK      open signals advance on new closed 1-minute bars (entry, stop, target, time stop, expiry);
              signals left in 'review' for more than 10 minutes (crash recovery) become shadow signals.
2. DETECT     every eligible instrument whose newest COMPLETE fast bucket has not been evaluated yet gets a
              fresh feature row (``features.build_features`` on ~12 sessions of closed bars); the deterministic
              setups are checked on its last row.
3. FILTER     disabled setups and directions, one open signal per symbol, cooldown per direction.
4. SCORE      composite score, calibrated win probability and expected R after costs; the signal is stored.
5. REVIEW     candidates above ``min_score_review`` go to the LLM in the background (at most 3 waiting).
6. FINALISE   the decision is taken at issue time: actionable recommendations go to the paper broker (AI
              portfolio) and the mailer; all other candidates are tracked as shadow signals exactly like the
              actionable ones, so that the thresholds and the LLM can be evaluated out-of-sample.
7. CHART SCAN one routine LLM chart analysis at a time, for the open instrument analysed longest ago.

Threading: all database and pandas work of the engine runs in ONE dedicated worker thread; the event loop
only orchestrates. Measured on this machine (100 symbols, ~5,500 one-minute bars each): the feature step
takes 2.8 s in one thread but 5.4 s in two and 10 s in four, because sqlite3 (which releases the GIL once
per fetched row) and pandas contend for the GIL. LLM reviews, chart rendering and mails run as background
tasks; ``stop()`` gives mails in flight a short grace period, then cancels and awaits everything and notes
the mails that could not be sent in the activity log.

No look-ahead: bars are read only through ``BarReader`` as of the decision time, news through
``repo_news.symbol_news`` (filtered on the time the analysis became known) and the calibration only sees
signals that had closed before the decision time. The tracker acts on SETTLED bars only
(``BarReader.settled_bars_after``): stream/quote bars that REST may still correct and bars behind a hole that
REST may still fill are processed once they are final, at their own times and prices. Its wall-clock fallback
(``tracking.expire_stale``) also respects downtimes: no data arrives while the platform is down or asleep, so
during the first ``SETTLE_S`` seconds after the engine's first cycle, and after a resume (two cycles more than
``TRACKING_RESUME_GAP_S`` apart), no minute counts as settled by age - the backfill of the downtime gets the
chance to deliver an entry or exit bar before a signal is cancelled or closed at market.
"""

import asyncio
import functools
import logging
import math
import threading
import time
from collections.abc import Callable, Collection, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from trading_local import presenters
from trading_local.analysis import readings, scoring, setups, tracking
from trading_local.analysis.calibration import Calibrator
from trading_local.analysis.features import NewsContext, build_features, feature_float, news_context
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.analysis.setups import SetupHit
from trading_local.analysis.stats import group_stats, stat_block
from trading_local.barstore import SETTLE_S, BarReader, resample
from trading_local.clock import Clock, from_ts, to_ts
from trading_local.config import CostParams, RuntimeSettings, Settings
from trading_local.contracts import (
    Broker,
    ChartRenderer,
    Llm,
    Mailer,
    MarketCalendar,
    TradeReview,
    TradeReviewContext,
)
from trading_local.db import repo_news, repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS, InstrumentBook
from trading_local.db.repo_news import SymbolNews
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import AnalysisOut, SignalStatsOut, StatBlock
from trading_local.events import TOPIC_CHART_ANALYSIS, TOPIC_SIGNAL, EventBus
from trading_local.models import SIGNAL_CLOSED_STATUSES, Candidate, Instrument, Signal

log = logging.getLogger(__name__)

P = ENGINE_PARAMS

WARMUP_SESSIONS = 12  # trading dates of 1-minute history behind every feature row (warm-up of the indicators)
# Completed daily bars behind every feature row. The EMA200 is seeded with an SMA, so its value depends on
# where the window starts; with 1000 bars the seed weight is below 0.001 and live and replay agree closely.
DAILY_BARS = 1000
STALE_GRACE_MIN = 10  # data counts as stale when the newest bar closed more than feed delay + this ago
REVIEW_STUCK_S = 600  # a signal still in 'review' after this long is finalised as shadow (crash recovery)
MAX_WAITING_REVIEWS = 3  # LLM reviews that may wait at the same time
NO_REVIEW_MARGIN = 0.08  # extra score a candidate needs to be actionable without an LLM verdict
CALIBRATION_REFRESH_S = 600  # the calibrator is refitted at most this often
ANALYSIS_MAX_AGE_MIN = 15  # trading minutes after which a cached feature row is recomputed for the GUI
REVIEW_GRACE_S = 15.0  # the engine gives up on a review this long after the LLM's own timeout
CHART_TIMEOUT_S = 60.0  # longest wait for one rendered chart
CHART_ANALYSIS_TIMEOUT_S = 600.0  # longest wait for one (lowest-priority) LLM chart analysis
# stop() lets entry and exit mails in flight finish for at most this long before it cancels them. It stays
# below the 15 s that Runtime.stop allows the whole engine shutdown (worker job and cancellation included).
MAIL_STOP_GRACE_S = 10.0
COOLDOWN_LOOKBACK = 10  # newest closed signals of a symbol checked for the cooldown
DETECTION_BUDGET_S = 3.0  # a detection pass slower than this is logged as a warning
ROUTINE_TIMEFRAME = "15m"
CHART_TIMEFRAMES: dict[str, int | None] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": None}  # None: daily
MIN_CHART_BARS = 30  # the chart renderer needs at least this many bars
# Two tracking cycles further apart than this mean the platform was interrupted (laptop sleep, suspended
# process): data collection resumes with the later cycle (see SignalEngine._settle_horizon).
TRACKING_RESUME_GAP_S = 180
# Settle horizon before every bar: no minute counts as settled by age (see SignalEngine._settle_horizon).
_NOTHING_SETTLED = -(2**62)

# Why a candidate was not issued (Signal.shadow_reason).
SHADOW_BELOW_THRESHOLD = "unter_schwelle"
SHADOW_EXPECTANCY = "erwartungswert"
SHADOW_LLM_VETO = "llm_veto"
SHADOW_LLM_UNAVAILABLE = "llm_nicht_verfuegbar"
SHADOW_MARKET_CLOSED = "markt_geschlossen"
SHADOW_STALE_DATA = "daten_veraltet"

NOTE_UNKNOWN_SYMBOL = "Unbekanntes Symbol."
NOTE_TOO_LITTLE_DATA = (
    f"Zu wenig Kursdaten für eine Analyse (mindestens {P.min_bars_slow} abgeschlossene "
    f"{P.slow_minutes}-Minuten-Balken nötig)."
)
NOTE_MARKET_CLOSED = "Börse geschlossen – Stand des letzten Handels."
NOTE_ANALYSIS_FAILED = "Die Analyse ist vorübergehend nicht verfügbar."

# Why a chart analysis requested from the GUI produced no result (activity log, category 'llm').
CHART_NO_CHART_DE = "es konnte kein Chart erstellt werden"
CHART_TIMEOUT_DE = "die KI hat nicht rechtzeitig geantwortet"
CHART_NO_ANSWER_DE = "die KI hat keine verwertbare Antwort geliefert"
CHART_ERROR_DE = "interner Fehler (Details im Log)"

# Kinds of background mails (stop() reports the ones it had to cancel).
MAIL_ENTRY = "entry"
MAIL_EXIT = "exit"
_MAIL_LABELS_DE: dict[str, str] = {MAIL_ENTRY: "Einstiegs-Mail", MAIL_EXIT: "Ausstiegs-Mail"}

REGION_LABELS_DE: dict[str, str] = {"US": "USA", "EU": "Europa", "ASIA": "Asien", "GLOBAL": "Global"}
DIRECTION_LABELS_DE: dict[str, str] = {"LONG": "Long", "SHORT": "Short"}
VERDICT_LABELS_DE: dict[str, str] = {
    "confirm": "bestätigt",
    "reduce": "reduziert",
    "veto": "Veto",
    "none": "ohne Prüfung",
}
_VERDICTS: frozenset[str] = frozenset(("confirm", "reduce", "veto", "unavailable"))
_EXIT_REASONS_DE: dict[str, str] = {"time_stop": "Zeitstopp", "session_close": "vor Handelsschluss"}
_CANCEL_REASONS_DE: dict[str, str] = {
    tracking.REASON_DRIFT: "Kurs vor dem Einstieg davongelaufen",
    tracking.REASON_NO_ENTRY: "kein Einstiegskurs rechtzeitig",
}
_DAY_SECONDS = 86_400


# ------------------------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------------------------
def warmup_start_ts(calendar: MarketCalendar, exchange: str, asof_ts: int, sessions: int = WARMUP_SESSIONS) -> int:
    """Open time of the oldest of the newest ``sessions`` trading dates that started before ``asof_ts``
    (the current trading date counts as one of them).

    Regular segments are cut at every UTC midnight and each piece gets ``calendar.session_date`` of its
    start, so a 24/7 venue counts one trading date per UTC day (same convention as ``features``). Falls back
    to the start of the search window (``sessions * 7 / 5 + 10`` calendar days) if it holds fewer dates.
    """
    end = from_ts(asof_ts)
    start = end - timedelta(days=sessions * 7 // 5 + 10)
    first_piece: dict[date, int] = {}
    for segment in calendar.segments(exchange, start, end):
        begin, finish = to_ts(segment.start), to_ts(segment.end)
        while begin < finish:
            first_piece.setdefault(calendar.session_date(exchange, from_ts(begin)), begin)
            begin = min(finish, (begin // _DAY_SECONDS + 1) * _DAY_SECONDS)
    if not first_piece:
        return to_ts(start)
    return first_piece[sorted(first_piece)[-sessions:][0]]


def closed_ts(signal: Signal) -> int:
    """Best known time at which a closed signal stopped being open (exit, last processed bar, entry deadline)."""
    for value in (signal.exit_ts, signal.checked_ts, signal.entry_deadline_ts, signal.issued_ts):
        if value is not None:
            return int(value)
    return int(signal.created_ts)


def unavailable_review() -> TradeReview:
    """The verdict used when no LLM review could be obtained."""
    return TradeReview(verdict="unavailable")


def checked_review(review: object) -> TradeReview:
    """The LLM's answer in a form that can be stored, or 'unavailable' if it is not a usable review.

    A review whose verdict is not one of 'confirm' / 'reduce' / 'veto' / 'unavailable' carries no decision
    and counts as unavailable (its texts are dropped). Texts become stripped strings, the conviction a finite
    number in 0..1 or None, the latency a non-negative int or None.
    """
    if not isinstance(review, TradeReview) or review.verdict not in _VERDICTS:
        log.warning("LLM review not usable (%r); counted as unavailable", getattr(review, "verdict", review))
        return unavailable_review()
    conviction = review.conviction
    if isinstance(conviction, (int, float)) and not isinstance(conviction, bool) and math.isfinite(conviction):
        conviction = min(max(float(conviction), 0.0), 1.0)
    else:
        conviction = None
    latency = review.latency_ms
    if not isinstance(latency, int) or isinstance(latency, bool) or latency < 0:
        latency = None
    return TradeReview(
        verdict=review.verdict,
        conviction=conviction,
        rationale_de=review.rationale_de.strip() if isinstance(review.rationale_de, str) else "",
        risks_de=review.risks_de.strip() if isinstance(review.risks_de, str) else "",
        latency_ms=latency,
    )


def _session_minutes(row: Mapping[str, Any]) -> int | None:
    """Regular trading minutes of the row's trading date (for the daily-trend horizon)."""
    total = feature_float(row, "minutes_since_open") + feature_float(row, "minutes_to_close")
    return round(total) if math.isfinite(total) and total > 0 else None


def _signal_label(signal: Signal) -> str:
    return f"{signal.direction} {signal.symbol} ({presenters.setup_label_de(signal.setup)})"


def _shadow_text_de(signal: Signal, prefs: RuntimeSettings) -> str:
    """German explanation of ``signal.shadow_reason``."""
    score = readings.de_number(signal.score, 2)
    reason = signal.shadow_reason
    if reason == SHADOW_BELOW_THRESHOLD:
        if signal.score < prefs.min_score_actionable:
            return f"Score {score} unter der Schwelle {readings.de_number(prefs.min_score_actionable, 2)}"
        margin = readings.de_number(prefs.min_score_actionable + NO_REVIEW_MARGIN, 2)
        return f"Score {score} ohne KI-Prüfung unter {margin}"
    if reason == SHADOW_EXPECTANCY:
        return f"Erwartungswert nach Slippage nicht positiv ({readings.de_number(signal.expected_r, 2, signed=True)} R)"
    if reason == SHADOW_LLM_VETO:
        return "Veto der KI-Prüfung"
    if reason == SHADOW_LLM_UNAVAILABLE:
        margin = prefs.min_score_actionable + NO_REVIEW_MARGIN
        if signal.score < margin:
            return f"KI-Prüfung nicht verfügbar und Score {score} unter {readings.de_number(margin, 2)}"
        return "KI-Prüfung nicht abgeschlossen"
    if reason == SHADOW_MARKET_CLOSED:
        return "Börse bei Freigabe geschlossen"
    if reason == SHADOW_STALE_DATA:
        return "Kursdaten bei Freigabe veraltet"
    return reason


def finalise_message_de(signal: Signal, prefs: RuntimeSettings) -> str:
    """Activity-log text of a finalised candidate, e.g. 'Empfehlung: LONG NVDA (Trend-Rücksetzer), Konfidenz 58 %'."""
    if signal.actionable:
        return f"Empfehlung: {_signal_label(signal)}, Konfidenz {round(signal.confidence * 100)} %"
    return f"Kandidat verworfen: {_signal_label(signal)} – {_shadow_text_de(signal, prefs)}"


def tracking_messages_de(signal: Signal, *, entered: bool) -> list[str]:
    """Activity-log texts for an entry and / or the close of a tracked signal in one cycle."""
    label = f"{_signal_label(signal)}, Signal #{signal.id}"
    messages: list[str] = []
    if entered and signal.entry_price is not None:
        messages.append(f"Einstieg: {label} zu {readings.de_price(signal.entry_price)}")
    result = "" if signal.outcome_r is None else f", Ergebnis {readings.de_number(signal.outcome_r, 2, signed=True)} R"
    if signal.status == "target":
        messages.append(f"Ziel erreicht: {label}{result}")
    elif signal.status == "stopped":
        messages.append(f"Stopp ausgelöst: {label}{result}")
    elif signal.status == "expired":
        why = _EXIT_REASONS_DE.get(signal.status_reason, signal.status_reason)
        messages.append(f"Zeitablauf ({why}): {label}{result}")
    elif signal.status == "cancelled":
        why = _CANCEL_REASONS_DE.get(signal.status_reason, signal.status_reason)
        messages.append(f"Kein Einstieg: {label} – {why}")
    return messages


def sample_note_de(n: int) -> str:
    """Honest German remark on the size of the outcome sample."""
    if n == 0:
        return "Noch keine abgeschlossenen Signale dieser Engine-Version – noch keine Aussage möglich."
    if n == 1:
        return "Erst 1 abgeschlossenes Signal – statistisch noch nicht belastbar."
    if n < 100:
        return f"Erst {n} abgeschlossene Signale – statistisch noch nicht belastbar."
    return f"{n} abgeschlossene Signale. Die 95-%-Intervalle zeigen die verbleibende statistische Unsicherheit."


def _verdict_group(signal: Signal) -> str:
    return signal.llm_verdict if signal.llm_verdict in ("confirm", "reduce", "veto") else "none"


def _apply_review(signal: Signal, review: TradeReview | None) -> None:
    """Copy the LLM verdict onto the signal (None: the review was skipped; see ``checked_review``)."""
    if review is None:
        signal.llm_verdict = None
        return
    review = checked_review(review)
    signal.llm_verdict = review.verdict
    signal.llm_conviction = review.conviction
    signal.llm_latency_ms = review.latency_ms
    signal.rationale_de = review.rationale_de
    signal.risks_de = review.risks_de


# ------------------------------------------------------------------------------------------------
# Value objects
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Snapshot:
    """The newest feature row of one symbol and its news context, computed at ``as_of``."""

    symbol: str
    as_of: int
    row: dict[str, Any]  # the last row of features.build_features
    news: NewsContext
    news_rows: tuple[SymbolNews, ...]  # newest first

    @property
    def data_ts(self) -> int:
        return int(self.row["data_ts"])


@dataclass(frozen=True, slots=True)
class _Detection:
    instrument: Instrument
    snapshot: Snapshot
    hits: list[SetupHit]


@dataclass(slots=True)
class _Pending:
    """A stored candidate on its way from 'review' to issued."""

    signal: Signal
    instrument: Instrument
    candidate: Candidate | None = None  # None for signals recovered from the database
    snapshot: Snapshot | None = None
    chart_png: bytes | None = None
    waiting: bool = True  # still waiting for the LLM verdict (may be evicted from the backlog)
    task: "asyncio.Task[Any] | None" = None

    @property
    def signal_id(self) -> int:
        if self.signal.id is None:
            raise ValueError("pending signal has not been stored")
        return self.signal.id


class _Pass:
    """Bar reader and per-pass caches for one decision time (warm-up start per venue, benchmark frames)."""

    def __init__(self, db: Database, instruments: InstrumentBook, calendar: MarketCalendar, now_ts: int) -> None:
        self.now_ts = now_ts
        self.reader = BarReader(db, instruments, calendar, now_ts, apply_feed_delay=False)
        self.evaluated = 0  # feature rows computed in this pass
        self._calendar = calendar
        self._starts: dict[str, int] = {}
        self._benchmarks: dict[tuple[str, int], pd.DataFrame | None] = {}

    def warmup_start(self, exchange: str) -> int:
        if exchange not in self._starts:
            self._starts[exchange] = warmup_start_ts(self._calendar, exchange, self.now_ts)
        return self._starts[exchange]

    def benchmark(self, symbol: str | None, start_ts: int) -> pd.DataFrame | None:
        """Closed 1-minute bars of a benchmark from ``start_ts`` (read once per pass); None without bars."""
        if symbol is None:
            return None
        key = (symbol, start_ts)
        if key not in self._benchmarks:
            frame = self.reader.minutes(symbol, start_ts=start_ts)
            self._benchmarks[key] = None if frame.empty else frame
        return self._benchmarks[key]


# ------------------------------------------------------------------------------------------------
# The engine
# ------------------------------------------------------------------------------------------------
class SignalEngine:
    """Live signal engine. ``on_minute``, ``request_chart_analysis``, ``drain`` and ``stop`` run on the event
    loop; ``analysis`` and ``stats`` are synchronous and may be called from any thread."""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        calendar: MarketCalendar,
        instruments: InstrumentBook,
        runtime_settings: SettingsStore,
        costs: CostParams,
        llm: Llm,
        charts: ChartRenderer,
        mailer: Mailer,
        broker: Broker,
    ) -> None:
        self.version: str = ENGINE_PARAMS.version
        self._settings = settings
        self._db = db
        self._clock = clock
        self._bus = bus
        self._calendar = calendar
        self._instruments = instruments
        self._runtime_settings = runtime_settings
        self._costs = costs
        self._llm = llm
        self._charts = charts
        self._mailer = mailer
        self._broker = broker
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signal-engine")
        # event-loop state
        self._tasks: set[asyncio.Task[Any]] = set()
        self._mails: dict[asyncio.Task[Any], tuple[str, Signal]] = {}  # running mail task -> (kind, signal)
        self._reviews: dict[int, _Pending] = {}
        self._chart_inflight: dict[str, tuple[asyncio.Task[Any], str]] = {}  # symbol -> (task, timeframe)
        self._chart_requested: set[str] = set()  # symbols whose running chart analysis the GUI asked for
        self._chart_attempts: dict[str, int] = {}
        # worker-thread state
        self._last_bucket: dict[str, int] = {}
        self._tracking_since: int | None = None  # first cycle of the current uninterrupted run of tracking cycles
        self._last_tracking_ts: int | None = None  # time of the latest tracking cycle
        # shared state
        self._snapshots: dict[str, Snapshot] = {}
        self._snapshot_lock = threading.Lock()
        self._calibration: tuple[int, Calibrator] | None = None
        self._calibration_lock = threading.Lock()

    # -- contracts.Engine -----------------------------------------------------------------------
    async def on_minute(self, now: datetime, updated: Collection[str]) -> None:
        """Track, detect, review and schedule (see the module docstring). Never raises.

        ``updated`` is not needed: the newest complete bucket of every instrument is read from the database,
        and buckets that were evaluated already are remembered per symbol.
        """
        try:
            now_ts = to_ts(now)
            prefs = self._prefs()
        except Exception:
            log.exception("signal engine: the minute cycle could not start")
            return
        await self._guarded("tracking", self._track_step(now_ts))
        if prefs.signals_enabled:
            await self._guarded("detection", self._detect_step(now_ts, prefs))
        await self._guarded("chart scan", self._chart_scan_step(now_ts, prefs))

    def analysis(self, symbol: str) -> AnalysisOut:
        """Technical summary of one symbol from the cached feature row (recomputed when it is older than
        15 trading minutes). Never raises: problems are reported through ``available`` / ``note_de``."""
        try:
            return self._analysis(symbol)
        except Exception:
            log.exception("analysis of %s failed", symbol)
            return AnalysisOut(symbol=symbol, available=False, note_de=NOTE_ANALYSIS_FAILED)

    def stats(self, since_ts: int | None = None) -> SignalStatsOut:
        """Outcome statistics of this engine version's closed signals that exited at or after ``since_ts``.

        Groups (setup, region, direction, LLM verdict) contain actionable and shadow signals alike; the
        calibration table is the one of the current calibrator (all outcomes before now).
        """
        try:
            return self._stats(since_ts)
        except Exception:
            log.exception("signal statistics failed")
            return SignalStatsOut(
                since_ts=since_ts,
                engine_version=self.version,
                actionable=StatBlock(),
                shadow=StatBlock(),
                note_de="Die Statistik ist vorübergehend nicht verfügbar.",
            )

    async def request_chart_analysis(self, symbol: str, timeframe: str = ROUTINE_TIMEFRAME) -> bool:
        """Queue one LLM chart analysis now (ignores the scan interval; one at a time per symbol).

        False when nothing would be produced: the symbol or timeframe is unknown, the LLM is disabled, there
        are fewer than 30 bars of that timeframe (the renderer's minimum), an analysis of ANOTHER timeframe
        is running for this symbol, or the engine cannot take the job (stopped, error). True if an analysis
        was queued or one of this symbol and timeframe is already running. A requested analysis always ends
        with a result (stored, announced on 'chart_analysis') or, when the chart or the model's answer is
        missing, with a German note in the activity log (category 'llm', level 'warning').
        """
        try:
            instrument = self._instruments.get(symbol)
            if instrument is None or not self._settings.enable_llm or timeframe not in CHART_TIMEFRAMES:
                return False
            if symbol in self._chart_inflight:
                return self._join_chart(symbol, timeframe)
            now_ts = self._clock.ts()
            if not await self._run(self._enough_chart_data, symbol, timeframe, now_ts):
                return False
            if symbol in self._chart_inflight:  # started while the data was checked
                return self._join_chart(symbol, timeframe)
            self._start_chart(instrument, timeframe, now_ts)
            self._chart_requested.add(symbol)
            return True
        except Exception:
            log.exception("chart analysis request for %s failed", symbol)
            return False

    async def stop(self) -> None:
        """Stop all background work, then let the worker thread finish.

        Entry and exit mails in flight get up to ``MAIL_STOP_GRACE_S`` to finish (mails started meanwhile
        by finishing reviews included); then every remaining task is cancelled and awaited. Each mail that
        had to be cancelled - and so was possibly not sent; an entry mail counts as sent once the mailer has
        marked the signal - is noted in the activity log (category 'mail', level 'warning'). Reviews that
        were cancelled leave their signals in 'review'; the next engine finalises them as shadow signals
        after 10 minutes (crash recovery).
        """
        try:
            await self._wait_for_mails(MAIL_STOP_GRACE_S)
        finally:
            unsent = await self._cancel_tasks()
            self._reviews.clear()
            if unsent:
                try:
                    await self._run(self._record_unsent_mails, unsent, self._clock.ts())
                except Exception:
                    log.exception("could not record %d unsent mails", len(unsent))
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def drain(self) -> None:
        """Wait until every background task (reviews, mails, chart analyses) has finished.

        For tests and orderly shutdowns; a task that never ends (a hanging dependency) blocks this call.
        """
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- plumbing -------------------------------------------------------------------------------
    def _prefs(self) -> RuntimeSettings:
        return self._runtime_settings.get()

    async def _run[T](self, func: Callable[..., T], *args: Any) -> T:
        """Run blocking work in the engine's worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args))

    @staticmethod
    async def _guarded(name: str, step: Coroutine[Any, Any, None]) -> None:
        try:
            await step
        except Exception:
            log.exception("signal engine: %s step failed", name)

    def _spawn(self, coro: Coroutine[Any, Any, Any], name: str) -> "asyncio.Task[Any]":
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: "asyncio.Task[Any]") -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            log.error("signal engine: background task %s failed", task.get_name(), exc_info=error)

    def _spawn_mail(self, coro: Coroutine[Any, Any, Any], kind: str, signal: Signal) -> None:
        """Send an entry (``MAIL_ENTRY``) or exit (``MAIL_EXIT``) mail of ``signal`` in the background."""
        prefix = "mail-signal" if kind == MAIL_ENTRY else "mail-exit"
        task = self._spawn(coro, f"{prefix}-{signal.id}")
        self._mails[task] = (kind, signal)
        task.add_done_callback(self._mail_done)

    def _mail_done(self, task: "asyncio.Task[Any]") -> None:
        self._mails.pop(task, None)

    async def _wait_for_mails(self, grace_s: float) -> None:
        """Wait until no mail task is running, at most ``grace_s`` seconds."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + grace_s
        while True:
            mails = [task for task in self._mails if not task.done()]
            remaining = deadline - loop.time()
            if not mails or remaining <= 0.0:
                return
            await asyncio.wait(mails, timeout=remaining)

    async def _cancel_tasks(self) -> list[tuple[str, Signal]]:
        """Cancel and await every background task; returns (kind, signal) of the mails that were cancelled
        and are not known to be delivered."""
        unsent: list[tuple[str, Signal]] = []
        while self._tasks:
            tasks = list(self._tasks)
            mails = {task: self._mails[task] for task in tasks if task in self._mails}
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for task, (kind, signal) in mails.items():
                delivered = kind == MAIL_ENTRY and signal.mailed_ts is not None
                if task.cancelled() and not delivered:
                    unsent.append((kind, signal))
        return unsent

    def _record_unsent_mails(self, unsent: list[tuple[str, Signal]], ts: int) -> None:
        """Activity-log entry for every mail that was cancelled at shutdown."""
        for kind, signal in unsent:
            label = _MAIL_LABELS_DE.get(kind, "Mail")
            message = (
                f"{label} beim Beenden der Plattform abgebrochen – möglicherweise nicht verschickt: "
                f"{_signal_label(signal)}, Signal #{signal.id}"
            )
            log.warning("%s mail of signal %s cancelled at shutdown", kind, signal.id)
            try:
                repo_ops.log_activity(
                    self._db,
                    ts=ts,
                    level="warning",
                    category="mail",
                    message=message,
                    ref_type="signal",
                    ref_id=signal.id,
                    bus=self._bus,
                )
            except Exception:
                log.exception("could not record the unsent %s mail of signal %s", kind, signal.id)

    def _publish(self, signal: Signal, instrument: Instrument | None) -> None:
        try:
            self._bus.publish(TOPIC_SIGNAL, presenters.signal_out(signal, instrument).model_dump())
        except Exception:
            log.exception("could not announce signal %s", signal.id)

    def _activity(self, ts: int, message: str, signal: Signal) -> None:
        try:
            repo_ops.log_activity(
                self._db,
                ts=ts,
                level="info",
                category="signal",
                message=message,
                ref_type="signal",
                ref_id=signal.id,
                bus=self._bus,
            )
        except Exception:
            log.exception("could not write the activity entry for signal %s", signal.id)

    def _persist(self, signal: Signal) -> None:
        """Write the full signal row, keeping a ``mailed_ts`` the mailer stored in the meantime (the mailer
        writes only that column; without the merge a full-row update would erase it)."""
        with self._db.write():
            if signal.mailed_ts is None and signal.id is not None:
                stored = repo_signals.get_signal(self._db, signal.id)
                if stored is not None:
                    signal.mailed_ts = stored.mailed_ts
            repo_signals.update_signal(self._db, signal)

    def _venue_open(self, instrument: Instrument, at: datetime) -> bool:
        try:
            return self._calendar.is_open(instrument.exchange, at)
        except KeyError:
            log.warning("%s: unknown trading calendar %r", instrument.symbol, instrument.exchange)
            return False

    def _benchmark_name(self, instrument: Instrument) -> str | None:
        if instrument.benchmark is None:
            return None
        benchmark = self._instruments.get(instrument.benchmark)
        return instrument.benchmark if benchmark is None else benchmark.name

    async def _render(
        self, symbol: str, timeframe: str = ROUTINE_TIMEFRAME, *, asof_ts: int, signal: Signal | None = None
    ) -> bytes | None:
        """A chart from the renderer, or None (too little data, error or no answer within 60 s)."""
        try:
            return await asyncio.wait_for(
                self._charts.render(symbol, timeframe, asof_ts=asof_ts, signal=signal), CHART_TIMEOUT_S
            )
        except Exception:
            log.exception("chart %s %s could not be rendered", symbol, timeframe)
            return None

    # -- 1. tracking ----------------------------------------------------------------------------
    async def _track_step(self, now_ts: int) -> None:
        exit_mails, stuck = await self._run(self._track_blocking, now_ts)
        for signal in exit_mails:
            instrument = self._instruments.get(signal.symbol)
            if instrument is not None:
                self._spawn_mail(self._mailer.send_exit(signal, instrument), MAIL_EXIT, signal)
        for signal in stuck:
            await self._recover_review(signal)

    def _track_blocking(self, now_ts: int) -> tuple[list[Signal], list[Signal]]:
        """Advance every open signal. Returns (closed signals that need an exit mail, signals stuck in review)."""
        reader = BarReader(self._db, self._instruments, self._calendar, now_ts, apply_feed_delay=False)
        last = self._last_tracking_ts
        if self._tracking_since is None or last is None or not 0 <= now_ts - last <= TRACKING_RESUME_GAP_S:
            self._tracking_since = now_ts  # (re)start, resume after an interruption, or the clock went back
        self._last_tracking_ts = now_ts
        exit_mails: list[Signal] = []
        stuck: list[Signal] = []
        for signal in repo_signals.open_signals(self._db):
            if signal.status == "review":
                if signal.created_ts < now_ts - REVIEW_STUCK_S:
                    stuck.append(signal)
                continue
            try:
                if self._track_signal(signal, reader, now_ts) and self._needs_exit_mail(signal):
                    exit_mails.append(signal)
            except Exception:
                log.exception("tracking of signal %s (%s) failed", signal.id, signal.symbol)
        return exit_mails, stuck

    def _track_signal(self, signal: Signal, reader: BarReader, now_ts: int) -> bool:
        """Advance one signal on its new bars; persist and announce a change. Returns True if it closed."""
        instrument = self._instruments.get(signal.symbol)
        if instrument is None:
            log.warning("signal %s: instrument %s is unknown; not tracked", signal.id, signal.symbol)
            return False
        had_entry = signal.entry_ts is not None
        # Settled bars only: provisional stream/quote bars and bars behind a hole that REST may still fill are
        # held back, so checked_ts never moves past a minute whose final data has not been seen.
        bars = reader.settled_bars_after(signal.symbol, signal.checked_ts, start_ts=signal.issued_ts)
        changed = tracking.advance(signal, bars, instrument=instrument, calendar=self._calendar, costs=self._costs)
        last_close = reader.last_price(signal.symbol) if signal.status == "active" else None
        expired = tracking.expire_stale(
            signal,
            now_ts=now_ts,
            instrument=instrument,
            calendar=self._calendar,
            last_close=last_close,
            costs=self._costs,
            settle_horizon_ts=self._settle_horizon(reader, signal.symbol, now_ts),
        )
        if not (changed or expired):
            return False
        self._persist(signal)
        self._publish(signal, instrument)
        if signal.actionable:
            for message in tracking_messages_de(signal, entered=not had_entry and signal.entry_ts is not None):
                self._activity(now_ts, message, signal)
        return signal.status in SIGNAL_CLOSED_STATUSES

    def _settle_horizon(self, reader: BarReader, symbol: str, now_ts: int) -> int:
        """``BarReader.settle_horizon`` for the tracker's wall-clock fallback, aware of downtimes.

        A minute is settled by age once its data had ``SETTLE_S`` seconds to arrive. While the platform was
        down nothing could arrive: after a restart, a crash or a laptop sleep the backfill and the hole repair
        store the downtime's bars seconds to minutes after the resume, and until then every minute of the
        downtime would look settled by age (``settle_horizon`` only looks at the clock). So during the first
        ``SETTLE_S`` seconds of an uninterrupted run of tracking cycles - after the first cycle, or after two
        cycles more than ``TRACKING_RESUME_GAP_S`` apart - no minute counts as settled by age
        (``_NOTHING_SETTLED``); minutes in which the venue did not trade never hold anything up.
        """
        since = self._tracking_since
        if since is not None and now_ts < since + SETTLE_S:
            return _NOTHING_SETTLED
        return reader.settle_horizon(symbol)

    @staticmethod
    def _needs_exit_mail(signal: Signal) -> bool:
        """A mailed recommendation that closed: target, stop, time stop - or cancelled before the entry, which
        gets the mailer's 'ENTFÄLLT' notice (whoever entered anyway is told to close the position)."""
        return signal.actionable and signal.mailed_ts is not None and signal.status in SIGNAL_CLOSED_STATUSES

    async def _recover_review(self, signal: Signal) -> None:
        """Finalise a signal that has been in 'review' for too long as a shadow signal."""
        pending = self._reviews.get(signal.id) if signal.id is not None else None
        if pending is not None:
            if not pending.waiting:
                return  # its verdict is in and it is being finalised right now
            self._drop_review(pending)
        if signal.id is not None and await self._run(self._recover_blocking, signal.id) is not None:
            log.info("signal %s (%s) was stuck in review and became a shadow signal", signal.id, signal.symbol)

    def _recover_blocking(self, signal_id: int) -> Signal | None:
        """Re-read the signal inside one transaction and finalise it only if it is still in 'review'."""
        with self._db.write():
            current = repo_signals.get_signal(self._db, signal_id)
            if current is None or current.status != "review":
                return None
            instrument = self._instruments.get(current.symbol)
            if instrument is None:
                log.warning("signal %s: instrument %s is unknown; left in review", signal_id, current.symbol)
                return None
            return self._finalise_blocking(_Pending(current, instrument), unavailable_review(), SHADOW_LLM_UNAVAILABLE)

    # -- 2.-4. detection, filter, scoring ---------------------------------------------------------
    async def _detect_step(self, now_ts: int, prefs: RuntimeSettings) -> None:
        created = await self._run(self._detect_blocking, now_ts, prefs)
        for pending in created:
            try:
                await self._route(pending, prefs)
            except Exception:
                log.exception("signal %s (%s) could not be routed", pending.signal.id, pending.signal.symbol)

    def _detect_blocking(self, now_ts: int, prefs: RuntimeSettings) -> list[_Pending]:
        """Evaluate the newest complete bucket of every eligible instrument; store and return new candidates."""
        started = time.perf_counter()
        cycle = _Pass(self._db, self._instruments, self._calendar, now_ts)
        open_symbols = {signal.symbol for signal in repo_signals.open_signals(self._db)}
        created: list[_Pending] = []
        for instrument in self._eligible(now_ts, prefs):
            try:
                detection = self._detect_symbol(instrument, cycle)
                pending = None if detection is None else self._create_candidate(detection, prefs, now_ts, open_symbols)
            except Exception:
                log.exception("signal detection for %s failed", instrument.symbol)
                continue
            if pending is not None:
                created.append(pending)
                open_symbols.add(instrument.symbol)
        elapsed = time.perf_counter() - started
        level = logging.WARNING if elapsed > DETECTION_BUDGET_S else logging.DEBUG
        log.log(
            level, "detection pass: %d feature rows, %d candidates in %.2f s", cycle.evaluated, len(created), elapsed
        )
        return created

    def _eligible(self, now_ts: int, prefs: RuntimeSettings) -> list[Instrument]:
        """Active, tradable instruments of an enabled region whose venue is in its regular session."""
        now = from_ts(now_ts)
        regions = set(prefs.signal_regions)
        return [
            instrument
            for instrument in self._instruments.tradable()
            if instrument.region in regions and self._venue_open(instrument, now)
        ]

    def _detect_symbol(self, instrument: Instrument, cycle: _Pass) -> _Detection | None:
        """Setup hits on the newest complete fast bucket, if that bucket is new and fresh.

        A bucket is complete when its end is <= the close time of the newest stored bar (the rule of
        ``build_features``). The first bucket seen for a symbol only initialises the memory, so that a
        restarted engine never re-evaluates old buckets.
        """
        step = P.decision_minutes * BAR_SECONDS
        last = cycle.reader.last_bar(instrument.symbol)
        if last is None:
            return None
        bucket_end = (last.ts + BAR_SECONDS) // step * step
        known = self._last_bucket.get(instrument.symbol)
        if known is not None and bucket_end <= known:
            return None
        self._last_bucket[instrument.symbol] = bucket_end
        if known is None:
            return None
        if bucket_end < cycle.now_ts - (instrument.delay_min + STALE_GRACE_MIN) * 60:
            log.debug("%s: newest bucket %d is stale", instrument.symbol, bucket_end)
            return None
        snapshot = self._snapshot(instrument, cycle)
        if snapshot is None or snapshot.data_ts != bucket_end:
            return None
        self._remember(snapshot)
        hits = setups.detect(
            snapshot.row, instrument=instrument, news=snapshot.news, session_minutes=_session_minutes(snapshot.row)
        )
        return _Detection(instrument, snapshot, hits) if hits else None

    def _snapshot(self, instrument: Instrument, cycle: _Pass) -> Snapshot | None:
        """Newest feature row of ``instrument`` as known at the pass time; None with too little data
        (fewer than ``min_bars_slow`` completed slow bars)."""
        start_ts = cycle.warmup_start(instrument.exchange)
        minutes = cycle.reader.minutes(instrument.symbol, start_ts=start_ts)
        if minutes.empty:
            return None
        cycle.evaluated += 1
        features = build_features(
            minutes,
            cycle.reader.daily(instrument.symbol, limit=DAILY_BARS),
            instrument=instrument,
            calendar=self._calendar,
            benchmark_minutes=cycle.benchmark(instrument.benchmark, start_ts),
        )
        if features.empty:
            return None
        data_ts = int(features.index[-1])
        if len(resample(minutes, P.slow_minutes, complete_before_ts=data_ts)) < P.min_bars_slow:
            return None
        news_rows = repo_news.symbol_news(
            self._db, instrument.symbol, asof_ts=cycle.now_ts, window_s=P.news_window_min * 60
        )
        row = features.iloc[-1].to_dict()  # plain Python values (float, int, bool, str)
        return Snapshot(
            instrument.symbol, cycle.now_ts, row, news_context(news_rows, asof_ts=cycle.now_ts), tuple(news_rows)
        )

    def _remember(self, snapshot: Snapshot) -> None:
        with self._snapshot_lock:
            cached = self._snapshots.get(snapshot.symbol)
            if cached is None or cached.as_of <= snapshot.as_of:
                self._snapshots[snapshot.symbol] = snapshot

    def _create_candidate(
        self, detection: _Detection, prefs: RuntimeSettings, now_ts: int, open_symbols: set[str]
    ) -> _Pending | None:
        """Filter the hits, score the best one and store it as a signal in 'review'. Dropped hits are not stored."""
        instrument, snapshot = detection.instrument, detection.snapshot
        if instrument.symbol in open_symbols:
            return None
        hits = [hit for hit in detection.hits if self._allowed(hit, prefs)]
        hits = self._without_cooldown(instrument.symbol, hits, now_ts, prefs.signal_cooldown_min)
        if not hits:
            return None
        scored = [
            (scoring.score_hit(hit, snapshot.row, instrument=instrument, news=snapshot.news), hit) for hit in hits
        ]
        (score, components), hit = max(scored, key=lambda item: item[0][0])
        rr = hit.target_dist / hit.stop_dist
        calibrator = self._calibrator(now_ts)
        cost_r = self._costs.round_trip_cost_r(instrument, feature_float(snapshot.row, "price"), hit.stop_dist)
        candidate = tracking.candidate_from_hit(
            hit,
            snapshot.row,
            instrument=instrument,
            decision_ts=now_ts,
            score=score,
            components=components,
            news=snapshot.news,
        )
        signal = tracking.make_signal(
            candidate,
            confidence=round(calibrator.probability(score, rr), 4),
            expected_r=round(calibrator.expected_r(score, rr, cost_r), 4),
            engine_version=self.version,
        )
        repo_signals.insert_signal(self._db, signal)
        self._publish(signal, instrument)
        return _Pending(signal, instrument, candidate, snapshot)

    @staticmethod
    def _allowed(hit: SetupHit, prefs: RuntimeSettings) -> bool:
        if hit.setup in prefs.disabled_setups:
            return False
        return prefs.allow_long if hit.direction == "LONG" else prefs.allow_short

    def _without_cooldown(self, symbol: str, hits: list[SetupHit], now_ts: int, cooldown_min: int) -> list[SetupHit]:
        """Drop hits whose direction had a signal of this symbol closed within the cooldown.

        At most one signal per symbol is open at a time, so the newest closed signals are also the most
        recently closed ones; ``COOLDOWN_LOOKBACK`` of them are checked.
        """
        if not hits or cooldown_min <= 0:
            return hits
        recent = repo_signals.list_signals(
            self._db, symbol=symbol, statuses=SIGNAL_CLOSED_STATUSES, limit=COOLDOWN_LOOKBACK
        )
        cutoff = now_ts - cooldown_min * 60
        cooling = {signal.direction for signal in recent if closed_ts(signal) >= cutoff}
        return [hit for hit in hits if hit.direction not in cooling]

    def _calibrator(self, now_ts: int) -> Calibrator:
        """Calibrator fitted on this version's outcomes that closed before ``now_ts`` (refitted every 10 min,
        and at once when the clock went backwards)."""
        with self._calibration_lock:
            fitted = self._calibration
            if fitted is None or now_ts < fitted[0] or now_ts - fitted[0] >= CALIBRATION_REFRESH_S:
                outcomes = repo_signals.closed_outcomes(self._db, engine_version=self.version, before_ts=now_ts)
                fitted = (now_ts, Calibrator.from_signals(outcomes, before_ts=now_ts))
                self._calibration = fitted
            return fitted[1]

    # -- 5. review ------------------------------------------------------------------------------
    async def _route(self, pending: _Pending, prefs: RuntimeSettings) -> None:
        """Finalise at once (review skipped or impossible) or queue the LLM review."""
        if pending.signal.score < prefs.min_score_review:
            await self._finalise(pending, None)
        elif not self._settings.enable_llm:
            await self._finalise(pending, unavailable_review())
        elif not prefs.llm_review:
            await self._finalise(pending, None)
        else:
            await self._admit_review(pending)

    async def _admit_review(self, pending: _Pending) -> None:
        """Start the review; with a full backlog the lowest-score candidate is finalised without review."""
        waiting = [other for other in self._reviews.values() if other.waiting]
        if len(waiting) < MAX_WAITING_REVIEWS:
            self._start_review(pending)
            return
        lowest = min([pending, *waiting], key=lambda candidate: candidate.signal.score)  # ties: the newcomer
        if lowest is pending:
            await self._finalise(pending, unavailable_review())
            return
        self._drop_review(lowest)
        self._start_review(pending)
        await self._finalise(lowest, unavailable_review())

    def _start_review(self, pending: _Pending) -> None:
        self._reviews[pending.signal_id] = pending
        pending.task = self._spawn(self._review(pending), f"review-{pending.signal_id}")

    def _drop_review(self, pending: _Pending) -> None:
        """Take a waiting review out of the backlog and cancel its task."""
        self._reviews.pop(pending.signal_id, None)
        pending.waiting = False
        if pending.task is not None:
            pending.task.cancel()

    async def _review(self, pending: _Pending) -> None:
        """Background task: chart, LLM verdict, then finalisation."""
        try:
            review = checked_review(await self._ask_llm(pending))
        except Exception:
            log.exception("LLM review of signal %s failed", pending.signal_id)
            review = unavailable_review()
        if self._reviews.get(pending.signal_id) is not pending:
            return  # evicted or recovered in the meantime
        pending.waiting = False
        try:
            await self._finalise(pending, review)
        finally:
            self._reviews.pop(pending.signal_id, None)

    async def _ask_llm(self, pending: _Pending) -> TradeReview:
        signal = pending.signal
        timeout_s = float(self._prefs().llm_review_timeout_s)
        pending.chart_png = await self._render(signal.symbol, asof_ts=signal.created_ts, signal=signal)
        context = self._review_context(pending)
        return await asyncio.wait_for(self._llm.review_trade(context, timeout_s=timeout_s), timeout_s + REVIEW_GRACE_S)

    def _review_context(self, pending: _Pending) -> TradeReviewContext:
        candidate, snapshot, instrument = pending.candidate, pending.snapshot, pending.instrument
        if candidate is None:
            raise ValueError(f"signal {pending.signal_id} has no candidate to review")
        benchmark = self._benchmark_name(instrument)
        try:
            to_close = self._calendar.minutes_to_close(instrument.exchange, from_ts(candidate.decision_ts))
        except KeyError:
            to_close = None
        facts = readings.review_facts(
            candidate,
            instrument=instrument,
            minutes_to_close=to_close,
            benchmark_name=benchmark,
            news_score=None if snapshot is None else snapshot.news.score,
        )
        return TradeReviewContext(
            instrument=instrument,
            candidate=candidate,
            facts=facts,
            headlines=self._headlines(snapshot),
            market_note=readings.market_note_de(candidate.features, benchmark),
            chart_png=pending.chart_png,
        )

    @staticmethod
    def _headlines(snapshot: Snapshot | None) -> list[tuple[str, float]]:
        """(German summary or title, sentiment) of the news behind the candidate, newest first, at most 5."""
        if snapshot is None:
            return []
        ids = set(snapshot.news.news_ids)
        rows = [row for row in snapshot.news_rows if row.news_id in ids]
        return [((row.summary_de or row.title).strip(), float(row.sentiment)) for row in rows][: P.news_max_ids]

    # -- 6. finalisation ------------------------------------------------------------------------
    async def _finalise(self, pending: _Pending, review: TradeReview | None) -> None:
        """Decide and issue; actionable signals go to the broker and (in the background) to the mailer."""
        signal = await self._run(self._finalise_and_execute, pending, review)
        if signal.actionable:
            self._spawn_mail(self._mail_entry(pending, signal), MAIL_ENTRY, signal)

    def _finalise_and_execute(self, pending: _Pending, review: TradeReview | None) -> Signal:
        """Finalise and, for an actionable signal, place the AI portfolio's order in the SAME worker job, so
        that a cancelled cycle or review (shutdown) can never leave an issued recommendation without its order."""
        signal = self._finalise_blocking(pending, review, "")
        if signal.actionable:
            self._execute_order(signal)
        return signal

    def _finalise_blocking(self, pending: _Pending, review: TradeReview | None, forced_reason: str) -> Signal:
        """Store the verdict, decide actionable / shadow at issue time t = now and start tracking."""
        signal, instrument = pending.signal, pending.instrument
        t_issue = self._clock.ts()
        prefs = self._prefs()
        _apply_review(signal, review)
        if not signal.rationale_de:
            signal.rationale_de = readings.rationale_de(signal, signal.features)
        reason = forced_reason or self._shadow_reason(signal, instrument, prefs, t_issue)
        signal.actionable = not reason
        signal.shadow_reason = reason
        tracking.issue(signal, issued_ts=t_issue, instrument=instrument)
        self._persist(signal)
        self._publish(signal, instrument)
        self._activity(t_issue, finalise_message_de(signal, prefs), signal)
        return signal

    def _shadow_reason(self, signal: Signal, instrument: Instrument, prefs: RuntimeSettings, t_issue: int) -> str:
        """'' if the signal is actionable at ``t_issue``, otherwise the first rule it fails."""
        threshold = prefs.min_score_actionable
        if signal.score < threshold:
            return SHADOW_BELOW_THRESHOLD
        if signal.expected_r <= 0.0:
            return SHADOW_EXPECTANCY
        if signal.llm_verdict == "veto":
            return SHADOW_LLM_VETO
        if signal.llm_verdict in (None, "unavailable") and signal.score < threshold + NO_REVIEW_MARGIN:
            return SHADOW_LLM_UNAVAILABLE if signal.llm_verdict == "unavailable" else SHADOW_BELOW_THRESHOLD
        if not self._venue_open(instrument, from_ts(t_issue)):
            return SHADOW_MARKET_CLOSED
        if self._data_stale(instrument, t_issue):
            return SHADOW_STALE_DATA
        return ""

    def _data_stale(self, instrument: Instrument, now_ts: int) -> bool:
        last = BarReader(self._db, self._instruments, self._calendar, now_ts).last_bar(instrument.symbol)
        return last is None or last.ts + BAR_SECONDS < now_ts - (instrument.delay_min + STALE_GRACE_MIN) * 60

    def _execute_order(self, signal: Signal) -> None:
        try:
            self._broker.execute_signal(signal)
        except Exception:
            log.exception("broker failed on signal %s", signal.id)

    async def _mail_entry(self, pending: _Pending, signal: Signal) -> None:
        """Entry mail with the review chart (rendered now if there was no review and charts are attached)."""
        chart = pending.chart_png
        if chart is None and self._prefs().mail_attach_chart:
            chart = await self._render(signal.symbol, asof_ts=signal.issued_ts or signal.created_ts, signal=signal)
        await self._mailer.send_signal(signal, pending.instrument, chart)

    # -- 7. chart analyses ----------------------------------------------------------------------
    async def _chart_scan_step(self, now_ts: int, prefs: RuntimeSettings) -> None:
        if not (prefs.chart_scan_enabled and self._settings.enable_llm) or self._chart_inflight:
            return
        instrument = await self._run(
            self._next_chart_instrument, now_ts, prefs.chart_scan_interval_min, dict(self._chart_attempts)
        )
        if instrument is not None and instrument.symbol not in self._chart_inflight:
            self._start_chart(instrument, ROUTINE_TIMEFRAME, now_ts)

    def _next_chart_instrument(self, now_ts: int, interval_min: int, attempts: dict[str, int]) -> Instrument | None:
        """The open, tradable instrument whose last chart analysis (or attempt) is oldest and at least
        ``interval_min`` old; never analysed comes first, ties by sort order."""
        stored = repo_signals.last_chart_analysis_ts(self._db)
        now = from_ts(now_ts)
        due: list[tuple[int, int, str, Instrument]] = []
        for instrument in self._instruments.tradable():
            seen = max(stored.get(instrument.symbol, -1), attempts.get(instrument.symbol, -1))
            if seen >= 0 and now_ts - seen < interval_min * 60:
                continue
            if self._venue_open(instrument, now):
                due.append((seen, instrument.sort_order, instrument.symbol, instrument))
        return min(due, key=lambda item: item[:3])[3] if due else None

    def _start_chart(self, instrument: Instrument, timeframe: str, asof_ts: int) -> None:
        symbol = instrument.symbol
        self._chart_attempts[symbol] = asof_ts
        task = self._spawn(self._chart_analysis(instrument, timeframe, asof_ts), f"chart-{symbol}")
        self._chart_inflight[symbol] = (task, timeframe)
        task.add_done_callback(functools.partial(self._chart_done, symbol))

    def _chart_done(self, symbol: str, task: "asyncio.Task[Any]") -> None:
        inflight = self._chart_inflight.get(symbol)
        if inflight is not None and inflight[0] is task:
            del self._chart_inflight[symbol]
            self._chart_requested.discard(symbol)

    def _join_chart(self, symbol: str, timeframe: str) -> bool:
        """A request for a symbol whose chart analysis is running: True (and its outcome is reported like a
        requested one) if it has the requested timeframe, False if it analyses another one."""
        if self._chart_inflight[symbol][1] != timeframe:
            return False
        self._chart_requested.add(symbol)
        return True

    async def _chart_analysis(self, instrument: Instrument, timeframe: str, asof_ts: int) -> None:
        """Render, ask the LLM (at most ``CHART_ANALYSIS_TIMEOUT_S``), store and announce one chart analysis.

        An analysis the GUI asked for that produces nothing (no chart, no or no usable answer, an error) ends
        with a German note in the activity log, so a request never stays without an answer. Routine scans
        that fail are only logged (the scan retries after its interval).
        """
        symbol = instrument.symbol
        try:
            failure = await self._produce_chart_analysis(instrument, timeframe, asof_ts)
        except Exception:
            log.exception("chart analysis of %s %s failed", symbol, timeframe)
            failure = CHART_ERROR_DE
        if failure and symbol in self._chart_requested:
            try:
                await self._run(self._chart_failure_note, symbol, timeframe, failure)
            except Exception:
                log.exception("could not report the failed chart analysis of %s", symbol)

    async def _produce_chart_analysis(self, instrument: Instrument, timeframe: str, asof_ts: int) -> str:
        """One chart analysis; '' when it was stored and announced, otherwise the German reason why not."""
        symbol = instrument.symbol
        png = await self._render(symbol, timeframe, asof_ts=asof_ts)
        if png is None:
            log.info("chart analysis of %s %s skipped: no chart", symbol, timeframe)
            return CHART_NO_CHART_DE
        facts = await self._run(self._chart_facts, instrument, asof_ts)
        try:
            result = await asyncio.wait_for(
                self._llm.analyze_chart(instrument, timeframe, png, facts), CHART_ANALYSIS_TIMEOUT_S
            )
        except TimeoutError:
            log.warning("chart analysis of %s %s: no answer within %.0f s", symbol, timeframe, CHART_ANALYSIS_TIMEOUT_S)
            return CHART_TIMEOUT_DE
        if result is None:
            log.info("chart analysis of %s %s: no usable answer from the model", symbol, timeframe)
            return CHART_NO_ANSWER_DE
        stored = await self._run(repo_signals.insert_chart_analysis, self._db, result)
        self._bus.publish(TOPIC_CHART_ANALYSIS, presenters.chart_analysis_out(stored).model_dump())
        return ""

    def _chart_failure_note(self, symbol: str, timeframe: str, reason: str) -> None:
        """Activity-log note that a requested chart analysis produced no result."""
        repo_ops.log_activity(
            self._db,
            ts=self._clock.ts(),
            level="warning",
            category="llm",
            message=f"Chartanalyse {symbol} ({timeframe}) ohne Ergebnis: {reason}.",
            bus=self._bus,
        )

    def _chart_facts(self, instrument: Instrument, asof_ts: int) -> dict[str, float]:
        snapshot = self._current_snapshot(instrument, asof_ts)
        if snapshot is not None:
            return readings.chart_facts(snapshot.row)
        reader = BarReader(self._db, self._instruments, self._calendar, asof_ts)
        return readings.chart_facts(None, reader.last_price(instrument.symbol))

    def _enough_chart_data(self, symbol: str, timeframe: str, now_ts: int) -> bool:
        reader = BarReader(self._db, self._instruments, self._calendar, now_ts)
        minutes = CHART_TIMEFRAMES[timeframe]
        if minutes is None:
            frame = reader.daily(symbol, limit=MIN_CHART_BARS)
        else:
            frame = reader.resampled(symbol, minutes, lookback_bars=MIN_CHART_BARS)
        return len(frame) >= MIN_CHART_BARS

    # -- analysis and statistics ----------------------------------------------------------------
    def _analysis(self, symbol: str) -> AnalysisOut:
        instrument = self._instruments.get(symbol)
        if instrument is None:
            return AnalysisOut(symbol=symbol, available=False, note_de=NOTE_UNKNOWN_SYMBOL)
        now_ts = self._clock.ts()
        stored_chart = repo_signals.latest_chart_analysis(self._db, symbol)
        chart = None if stored_chart is None else presenters.chart_analysis_out(stored_chart)
        snapshot = self._current_snapshot(instrument, now_ts)
        if snapshot is None:
            return AnalysisOut(symbol=symbol, available=False, note_de=NOTE_TOO_LITTLE_DATA, chart_analysis=chart)
        return readings.analysis_out(
            symbol,
            snapshot.row,
            as_of=snapshot.as_of,
            news_score=snapshot.news.score,
            news_count=snapshot.news.count,
            benchmark_name=self._benchmark_name(instrument),
            chart_analysis=chart,
            note_de="" if self._venue_open(instrument, from_ts(now_ts)) else NOTE_MARKET_CLOSED,
        )

    def _current_snapshot(self, instrument: Instrument, now_ts: int) -> Snapshot | None:
        """The cached snapshot unless it is older than 15 trading minutes; otherwise a fresh one (same code
        path as the detection) that replaces it."""
        with self._snapshot_lock:
            cached = self._snapshots.get(instrument.symbol)
        if cached is not None and cached.as_of <= now_ts:
            if self._trading_minutes(instrument.exchange, cached.as_of, now_ts) <= ANALYSIS_MAX_AGE_MIN:
                return cached
        snapshot = self._snapshot(instrument, _Pass(self._db, self._instruments, self._calendar, now_ts))
        if snapshot is not None:
            self._remember(snapshot)
        return snapshot

    def _trading_minutes(self, exchange: str, start_ts: int, end_ts: int) -> float:
        """Regular trading minutes between two instants."""
        if end_ts <= start_ts:
            return 0.0
        segments = self._calendar.segments(exchange, from_ts(start_ts), from_ts(end_ts))
        return sum((segment.end - segment.start).total_seconds() for segment in segments) / 60.0

    def _stats(self, since_ts: int | None) -> SignalStatsOut:
        signals = repo_signals.closed_outcomes(self._db, engine_version=self.version, since_ts=since_ts)
        actionable = [signal.outcome_r for signal in signals if signal.actionable and signal.outcome_r is not None]
        shadow = [signal.outcome_r for signal in signals if not signal.actionable and signal.outcome_r is not None]
        return SignalStatsOut(
            since_ts=since_ts,
            engine_version=self.version,
            actionable=stat_block(actionable),
            shadow=stat_block(shadow),
            by_setup=group_stats(signals, lambda signal: signal.setup, presenters.SETUP_LABELS_DE),
            by_region=group_stats(signals, self._region_of, REGION_LABELS_DE),
            by_direction=group_stats(signals, lambda signal: signal.direction, DIRECTION_LABELS_DE),
            by_llm_verdict=group_stats(signals, _verdict_group, VERDICT_LABELS_DE),
            calibration=self._calibrator(self._clock.ts()).bins(),
            note_de=sample_note_de(len(signals)),
        )

    def _region_of(self, signal: Signal) -> str:
        instrument = self._instruments.get(signal.symbol)
        return "GLOBAL" if instrument is None else instrument.region
