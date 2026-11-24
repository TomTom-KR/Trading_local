"""Service interfaces of the platform (structural typing via ``typing.Protocol``).

Every module implements one of these protocols; consumers depend ONLY on the protocol, never on the
concrete class of another module. The runtime (``trading_local/runtime``) creates the concrete objects
and wires them together.

General rules for all implementations
* Never read the wall clock directly: take a ``Clock`` (``trading_local.clock``) or an explicit ``now``.
* All ``datetime`` values are timezone-aware UTC. All ``*_ts`` values are epoch seconds UTC.
* Blocking work (HTTP via requests/yfinance, pandas number crunching, matplotlib, SMTP) must not run on
  the event loop: use ``asyncio.to_thread`` or the dedicated executors provided by the runtime.
* Methods documented as "never raises" must catch and log their own errors: one failing symbol, feed,
  mail or LLM call must never stop the minute cycle.
* German user-facing texts (fields ending in ``_de``, activity-log messages, error messages for the GUI).
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from trading_local.dto import (
    AnalysisOut,
    DataStatusOut,
    EquityCurvesOut,
    FeedStateOut,
    LlmStatusOut,
    MailStatusOut,
    MailTestOut,
    NewsStatusOut,
    OrderOut,
    OrderRequest,
    PortfolioDetailOut,
    PortfolioSummaryOut,
    PositionOut,
    SignalStatsOut,
)
from trading_local.models import (
    Candidate,
    ChartAnalysis,
    EarningsDate,
    Instrument,
    LlmVerdict,
    NewsAnalysis,
    NewsItem,
    Position,
    Quote,
    SessionSegment,
    SessionState,
    Signal,
)


# ------------------------------------------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------------------------------------------
@runtime_checkable
class MarketCalendar(Protocol):
    """Trading sessions of all venues (regular hours, lunch breaks, holidays, early closes).

    Pure and synchronous; every call must be fast (cached) because it is used inside tight loops.
    ``exchange`` is the calendar key stored in ``Instrument.exchange`` (MIC such as 'XNYS', 'XETR',
    or the pseudo venues 'CRYPTO', 'FX', 'CME'). Unknown keys raise ``KeyError``.

    Pseudo venues: their trading date is the UTC date (this matches how the data source dates their daily
    bars), so ``session_date`` = UTC date, ``last_completed_session`` = the previous UTC date, and
    ``minutes_since_open`` / ``minutes_to_close`` count from / to UTC midnight while the venue is open.
    """

    def exchanges(self) -> list[str]:
        """All supported calendar keys."""
        ...

    def display_name(self, exchange: str) -> str:
        """Human-readable venue name, e.g. 'Xetra', 'New York', 'Tokio'."""
        ...

    def timezone(self, exchange: str) -> str:
        """IANA time zone of the venue."""
        ...

    def state(self, exchange: str, at: datetime) -> SessionState:
        """Session phase at ``at`` plus the surrounding phase boundaries."""
        ...

    def is_open(self, exchange: str, at: datetime) -> bool:
        """True only inside a regular trading segment (False in lunch breaks, pre/post market, holidays)."""
        ...

    def segments(self, exchange: str, start: datetime, end: datetime, *, extended: bool = False) -> list[SessionSegment]:
        """Trading windows intersecting ``[start, end)``, ascending and clipped to that range.

        Regular segments only by default (a day with a lunch break yields two segments);
        ``extended=True`` adds pre-market / after-hours windows where they exist. 24/7 venues are returned
        as one segment per UTC day (split at UTC midnight); consumers should still tolerate adjacent
        segments that touch.
        """
        ...

    def session_date(self, exchange: str, at: datetime) -> date:
        """The venue's trading date ``at`` belongs to (local calendar date; for 24h venues the venue's convention)."""
        ...

    def last_completed_session(self, exchange: str, at: datetime) -> date | None:
        """Most recent trading date whose regular session has completely ended at ``at``.

        Daily bars may only be used up to and including this date - anything later would be look-ahead.
        """
        ...

    def minutes_since_open(self, exchange: str, at: datetime) -> int | None:
        """Trading minutes elapsed since today's first regular segment opened (breaks excluded); None if not open."""
        ...

    def minutes_to_close(self, exchange: str, at: datetime) -> int | None:
        """Trading minutes left until today's LAST regular segment ends (breaks excluded); None if not open."""
        ...

    def add_trading_minutes(self, exchange: str, start: datetime, minutes: int) -> datetime:
        """The instant at which ``minutes`` trading minutes have passed after ``start`` (skips closed periods)."""
        ...


# ------------------------------------------------------------------------------------------------
# Market data
# ------------------------------------------------------------------------------------------------
@runtime_checkable
class MarketData(Protocol):
    """Collects quotes and 1-minute bars and writes them to the database (tables bars_1m, bars_1d, quotes)."""

    async def start(self) -> None:
        """Connect the stream and start the background history backfill. Returns quickly."""
        ...

    async def stop(self) -> None: ...

    async def refresh(self, now: datetime) -> list[str]:
        """Per-minute duty, never raises: persist stream-built bars, reconcile with the REST source for
        venues that are open (within the safe request budget) and persist the quote book.

        Returns the symbols that received at least one NEW CLOSED 1-minute bar since the previous call.
        """
        ...

    def quote(self, symbol: str) -> Quote | None:
        """Latest known quote from memory (thread-safe, no I/O)."""
        ...

    def quotes(self) -> dict[str, Quote]:
        """Snapshot copy of all latest quotes (thread-safe, no I/O)."""
        ...

    def status(self) -> DataStatusOut: ...

    async def lookup(self, symbol: str) -> Instrument | None:
        """Validate a symbol at the data source and derive its instrument record (None if unknown)."""
        ...

    async def ensure_history(self, symbol: str) -> None:
        """Download daily and 1-minute history for a newly added symbol."""
        ...

    async def earnings(self, symbol: str) -> list[EarningsDate] | None:
        """Past and upcoming earnings releases of a company from the data source (newest first). Never raises;
        None when the request failed, an empty list when the source knows no dates (e.g. for crypto)."""
        ...


@runtime_checkable
class Derivatives(Protocol):
    """Funding rates, open interest and liquidations of the crypto instruments' perpetual futures (Kraken
    Futures, public data). Writes the tables deriv_snapshots and deriv_liquidations."""

    async def start(self) -> None:
        """Start the background backfill of the history. Returns quickly."""
        ...

    async def stop(self) -> None: ...

    async def poll(self, now: datetime) -> int:
        """Per-minute duty, never raises: store the current snapshot and the new liquidation minutes; a
        liquidation burst pulls the coin's news forward (``News.prioritize``). Returns rows written."""
        ...

    def metrics(self, symbol: str, asof_ts: int | None = None) -> Any:
        """The derivatives picture of a crypto instrument as known at ``asof_ts`` (``repo_derivs.DerivMetrics``),
        or None without perpetual or data. Thread-safe."""
        ...


@runtime_checkable
class FxRates(Protocol):
    def rate(self, currency: str, base: str, at_ts: int | None = None) -> float | None:
        """Units of ``base`` currency per ONE unit of ``currency`` (e.g. rate('USD', 'EUR') ~ 0.92).

        ``at_ts=None`` uses the latest quote; otherwise the last 1-minute bar at or before ``at_ts``
        (falling back to daily closes). Returns 1.0 when both currencies are equal and None when no
        rate is known.
        """
        ...


# ------------------------------------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class TradeReviewContext:
    """Everything the LLM gets to review one trade candidate. Prepared by the signal engine."""

    instrument: Instrument
    candidate: Candidate
    facts: dict[str, Any]  # compact, rounded fact sheet with readable keys (trend, RSI, ATR, volume ...)
    headlines: list[tuple[str, float]] = field(default_factory=list)  # (text, sentiment -1..1), newest first, max 5
    market_note: str = ""  # e.g. 'S&P 500 heute +0,4 %, 15-Min-Trend aufwaerts'
    chart_png: bytes | None = None


@dataclass(slots=True)
class TradeReview:
    """Implementations must return only the listed verdicts and a conviction within 0..1 (or None)."""

    verdict: LlmVerdict  # 'confirm' | 'reduce' | 'veto' | 'unavailable' (timeout, error, LLM disabled)
    conviction: float | None = None  # 0..1
    rationale_de: str = ""
    risks_de: str = ""
    latency_ms: int | None = None


@runtime_checkable
class Llm(Protocol):
    """Gateway to the local Ollama model. Owns ONE priority queue and runs one request at a time
    (the GPU is the bottleneck). Priority: trade reviews > fresh news > routine chart analyses.
    Every call is logged to the table ``llm_calls``."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def status(self) -> LlmStatusOut: ...

    async def analyze_news(self, item: NewsItem, candidates: Sequence[Instrument]) -> NewsAnalysis | None:
        """Classify one news item (relevance, sentiment, category, horizon, affected symbols, German summary).

        ``candidates`` are the only instruments the model may name (constrained output, no invented tickers).
        The implementation may batch several queued items into one model call. Never raises; returns None
        when the model is unavailable or the job was dropped as stale. The caller persists the result.
        """
        ...

    async def review_trade(self, ctx: TradeReviewContext, *, timeout_s: float) -> TradeReview:
        """Time-critical review of a trade candidate. Never raises; verdict 'unavailable' on timeout/error."""
        ...

    async def analyze_chart(
        self, instrument: Instrument, timeframe: str, chart_png: bytes, facts: Mapping[str, Any]
    ) -> ChartAnalysis | None:
        """Vision analysis of a rendered chart (lowest priority). Never raises. The caller persists the result.

        Returns None when the model is unavailable or when the job waited longer than its deadline in the
        queue (dropped as stale, at most about 30 minutes)."""
        ...


# ------------------------------------------------------------------------------------------------
# News
# ------------------------------------------------------------------------------------------------
@runtime_checkable
class News(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def poll(self, now: datetime) -> int:
        """Fetch all feeds that are due, store new items (table news), match them to instruments and hand
        the relevant ones to the LLM in the background (results go to news_analysis / news_symbols).
        Never raises. Returns the number of new items stored. Does NOT wait for the analyses."""
        ...

    async def prioritize(self, symbols: Collection[str], now: datetime) -> int:
        """Pull the news of ``symbols`` forward after an unusual price move (price trigger): their waiting
        and newly arriving items are analysed first for a while, and recently dropped items naming them are
        queued again. Never raises; returns the number of items queued again."""
        ...

    def status(self) -> NewsStatusOut: ...

    def feeds(self) -> list[FeedStateOut]: ...


@runtime_checkable
class Agenda(Protocol):
    """Macro data, central-bank decisions and earnings dates with forecast, actual value and surprise (table
    agenda_events). Readers use ``repo_agenda`` with an as-of time; this service only keeps the table current."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def refresh(self, now: datetime) -> int:
        """Per-minute duty, never raises: calendar of the week (hourly), actual values from new headlines,
        earnings dates of a few companies. Returns rows written."""
        ...


@runtime_checkable
class PriceTrigger(Protocol):
    """Watches the newest bars for unusual moves with unusual volume and pulls the news of the instrument
    forward (``News.prioritize``). Runs after the market-data step of the minute cycle."""

    async def on_minute(self, now: datetime, updated: Collection[str]) -> list[Any]:
        """Check the instruments that received new closed bars; act on unusual moves. Never raises.
        Returns the moves that triggered."""
        ...

    async def stop(self) -> None: ...


# ------------------------------------------------------------------------------------------------
# Charts and mail
# ------------------------------------------------------------------------------------------------
@runtime_checkable
class ChartRenderer(Protocol):
    async def render(
        self,
        symbol: str,
        timeframe: str = "15m",
        *,
        asof_ts: int | None = None,
        bars: int = 120,
        signal: Signal | None = None,
        width: int = 1024,
        height: int = 576,
    ) -> bytes | None:
        """PNG candlestick chart (EMA lines, volume; entry/stop/target lines when ``signal`` is given) built
        from database bars that were closed at ``asof_ts`` (default: now). None if there is too little data.
        Never raises. Rendering runs off the event loop and is serialised (matplotlib is not thread-safe)."""
        ...


@runtime_checkable
class Mailer(Protocol):
    """Sends alert mails according to MAIL_MODE (smtp | file | off) and logs every attempt to mail_log.
    Applies the rate limits from the runtime settings. Never raises."""

    async def send_signal(self, signal: Signal, instrument: Instrument, chart_png: bytes | None = None) -> bool:
        """Entry recommendation (LONG = buy, SHORT = sell short). True if a mail was sent or written to file."""
        ...

    async def send_exit(self, signal: Signal, instrument: Instrument) -> bool:
        """Exit recommendation for a signal that was mailed before (target reached, stopped, time stop), or the
        notice that a mailed recommendation was cancelled before its entry ('ENTFÄLLT')."""
        ...

    async def send_test(self) -> MailTestOut: ...

    def status(self) -> MailStatusOut: ...


# ------------------------------------------------------------------------------------------------
# Paper trading
# ------------------------------------------------------------------------------------------------
class OrderRejected(Exception):
    """An order violates a rule of the simulation. ``str(exc)`` is a German message for the user."""


@runtime_checkable
class Broker(Protocol):
    """Paper broker for the two portfolios 'ai' and 'me' (identical rules and costs).

    Fill rule (no look-ahead, identical for both portfolios and for the validation replay):
    a market order placed at time T fills at the OPEN of the first 1-minute bar whose open time is >= T,
    plus slippage. Until that bar has arrived AND is settled (``BarReader.first_settled_bar_at_or_after``:
    confirmed by the REST reconciliation or old enough) the order stays pending - e.g. while the market is
    closed or the data feed is delayed.
    Thread-safe: methods are called from API worker threads and from the runtime."""

    def place_order(self, portfolio_key: str, req: OrderRequest) -> OrderOut:
        """Validate and store a manual order. Raises ``OrderRejected`` with a German explanation."""
        ...

    def cancel_order(self, portfolio_key: str, order_id: int) -> OrderOut:
        """Cancel a pending order. Raises ``OrderRejected`` if it is not pending / not found."""
        ...

    def close_position(self, portfolio_key: str, position_id: int) -> OrderOut:
        """Place a closing order for an open position. Raises ``OrderRejected``."""
        ...

    def update_position(
        self, portfolio_key: str, position_id: int, *, stop_price: float | None, target_price: float | None
    ) -> PositionOut:
        """Replace both protective levels of an open position (None removes a level). Raises ``OrderRejected``."""
        ...

    def execute_signal(self, signal: Signal) -> OrderOut | None:
        """AI portfolio: size the trade by risk and place the order for an actionable signal.
        Returns None (and writes an activity-log entry with the reason) when the trade is skipped. Never raises."""
        ...

    def process(self, now: datetime) -> list[Position]:
        """Per-minute duty, never raises: fill pending orders on newly arrived bars, trigger stops, targets,
        time stops and session-close flattening, expire orders, write equity snapshots.
        Returns the positions that were CLOSED in this call."""
        ...

    def summary(self, portfolio_key: str) -> PortfolioSummaryOut: ...

    def detail(self, portfolio_key: str) -> PortfolioDetailOut: ...

    def trades(self, portfolio_key: str, limit: int = 200) -> list[PositionOut]:
        """Closed positions, newest first."""
        ...

    def orders(self, portfolio_key: str, limit: int = 200) -> list[OrderOut]:
        """All orders, newest first."""
        ...

    def equity_curves(self, since_ts: int | None = None, max_points: int = 600) -> EquityCurvesOut: ...

    def reset(self, start_capital: float | None = None) -> None:
        """Close everything and restart BOTH portfolios with the same capital (new comparison period).

        Raises ``OrderRejected`` (German text) for an invalid capital."""
        ...


# ------------------------------------------------------------------------------------------------
# Signal engine
# ------------------------------------------------------------------------------------------------
@runtime_checkable
class Engine(Protocol):
    version: str  # engine version tag stored with every signal (changes whenever rules/parameters change)

    async def on_minute(self, now: datetime, updated: Collection[str]) -> None:
        """Per-minute duty, never raises: update open signals with new bars (entries, stops, targets, expiry),
        detect new candidates for venues that are open, start LLM reviews, finalise reviewed candidates
        (-> Broker.execute_signal and Mailer.send_signal for actionable ones) and schedule routine chart
        analyses. Must return within a few seconds: LLM work continues in background tasks."""
        ...

    def analysis(self, symbol: str) -> AnalysisOut:
        """Current technical summary for the GUI (from cache or computed on demand; no LLM call)."""
        ...

    def stats(self, since_ts: int | None = None) -> SignalStatsOut: ...

    async def request_chart_analysis(self, symbol: str, timeframe: str = "15m") -> bool:
        """Queue an LLM chart analysis on demand (GUI button). False if it cannot be queued."""
        ...

    async def stop(self) -> None:
        """Let mails in flight finish (bounded grace period), then cancel the background tasks; mails that
        could not be sent are noted in the activity log."""
        ...


__all__ = [
    "Agenda",
    "Broker",
    "ChartRenderer",
    "Derivatives",
    "Engine",
    "FxRates",
    "Llm",
    "Mailer",
    "MarketCalendar",
    "MarketData",
    "News",
    "OrderRejected",
    "PriceTrigger",
    "TradeReview",
    "TradeReviewContext",
]
