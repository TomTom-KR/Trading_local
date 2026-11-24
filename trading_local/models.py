"""Domain types shared by all modules.

Pure data: no I/O and no business logic. Every dataclass that mirrors a database table uses the
same field names as the columns in ``trading_local/db/schema.sql`` (JSON columns are exposed as
Python lists/dicts, 0/1 flags as ``bool``).

Conventions (see schema.sql):
* timestamps are ``int`` epoch seconds in UTC; bar timestamps are the bar OPEN time
* prices are in the units the data source quotes (London shares in pence);
  ``Instrument.price_scale`` converts a quoted price into units of ``Instrument.currency``
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Direction = Literal["LONG", "SHORT"]
AssetClass = Literal["equity", "index", "crypto", "fx", "future"]
Region = Literal["US", "EU", "ASIA", "GLOBAL"]
BarSource = Literal["rest", "stream", "quote"]

# Trading-session phase of a venue.
#   open   regular trading session (the only phase in which orders fill and signals are issued)
#   break  intraday pause inside a trading day (lunch break in Tokyo / Hong Kong / Shanghai)
#   pre    pre-market window before the regular session (informational)
#   post   after-hours window after the regular session (informational)
#   closed everything else: night, weekend, holiday
SessionPhase = Literal["open", "break", "pre", "post", "closed"]

NewsStatus = Literal["new", "queued", "analyzed", "skipped", "duplicate", "error"]
NewsCategory = Literal[
    "earnings", "guidance", "m_and_a", "analyst", "macro", "regulation_legal", "product", "management", "other"
]
NewsHorizon = Literal["intraday", "days", "weeks"]

SignalStatus = Literal["review", "pending_entry", "active", "target", "stopped", "expired", "cancelled"]
LlmVerdict = Literal["confirm", "reduce", "veto", "unavailable"]
ChartBias = Literal["bullish", "bearish", "neutral"]

OrderIntent = Literal["open", "close"]
OrderOrigin = Literal["manual", "signal", "stop", "target", "time_stop", "session_close", "reset"]
OrderStatus = Literal["pending", "filled", "cancelled", "rejected"]
PositionStatus = Literal["open", "closed"]
PortfolioKey = Literal["ai", "me"]

SIGNAL_OPEN_STATUSES: tuple[str, ...] = ("review", "pending_entry", "active")
SIGNAL_CLOSED_STATUSES: tuple[str, ...] = ("target", "stopped", "expired", "cancelled")
# Closed states that carry a realised outcome (cancelled signals never had an entry).
SIGNAL_OUTCOME_STATUSES: tuple[str, ...] = ("target", "stopped", "expired")


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    name: str
    asset_class: AssetClass
    region: Region
    exchange: str  # trading-calendar key: MIC ('XNYS', 'XETR', ...) or 'CRYPTO' / 'FX' / 'CME'
    currency: str  # ISO 4217
    price_scale: float = 1.0
    timezone: str = "UTC"
    delay_min: int = 0
    tradable: bool = True
    active: bool = True
    benchmark: str | None = None
    aliases: tuple[str, ...] = ()
    sort_order: int = 0


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: int  # bar open time (epoch seconds UTC)
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    source: BarSource = "rest"


@dataclass(slots=True)
class Quote:
    symbol: str
    ts: int  # time of the last trade/quote as reported by the source
    received_ts: int
    price: float
    bid: float | None = None
    ask: float | None = None
    prev_close: float | None = None
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    day_volume: float | None = None
    change_pct: float | None = None
    market_state: str | None = None
    source: str = "quote"


@dataclass(frozen=True, slots=True)
class SessionState:
    """Trading-session state of one venue at one instant (all datetimes timezone-aware UTC)."""

    exchange: str
    phase: SessionPhase
    is_open: bool  # True only while phase == 'open'
    since: datetime | None  # start of the current phase
    until: datetime | None  # end of the current phase (= next phase change)
    next_open: datetime | None  # next start of a regular trading segment strictly after `at` (None: always open)
    next_close: datetime | None  # end of the current (if open) or of the next regular trading segment
    note: str = ""  # short German explanation, e.g. 'Wochenende', 'Feiertag', 'Mittagspause'


@dataclass(frozen=True, slots=True)
class SessionSegment:
    """One contiguous window of a venue's trading day, used for the 24h session ribbon."""

    exchange: str
    start: datetime  # UTC
    end: datetime  # UTC
    kind: Literal["regular", "pre", "post"] = "regular"


@dataclass(frozen=True, slots=True)
class EarningsDate:
    """One earnings release of a company as published by the data source (past and upcoming)."""

    symbol: str
    ts: int  # release time (UTC); midnight of the venue's local day when the source gives no time
    time_known: bool
    eps_estimate: float | None = None  # consensus
    eps_actual: float | None = None  # reported (None for upcoming releases)
    surprise_pct: float | None = None  # (reported - estimate) / |estimate| * 100


@dataclass(slots=True)
class NewsItem:
    uid: str
    title_key: str
    source: str
    url: str
    title: str
    fetched_ts: int
    summary: str = ""
    lang: str = "en"
    published_ts: int | None = None
    hinted_symbols: list[str] = field(default_factory=list)
    status: NewsStatus = "new"
    status_reason: str = ""
    duplicate_of: int | None = None
    id: int | None = None


@dataclass(slots=True)
class NewsAnalysis:
    news_id: int
    analyzed_ts: int
    model: str
    relevance: int  # 1..5
    sentiment: float  # -1..1 overall
    category: NewsCategory
    horizon: NewsHorizon
    summary_de: str
    confidence: float = 0.5
    # per-symbol sentiment (-1..1) for the affected watchlist symbols
    symbols: dict[str, float] = field(default_factory=dict)
    latency_ms: int | None = None


@dataclass(slots=True)
class ChartAnalysis:
    symbol: str
    ts: int
    timeframe: str
    model: str
    bias: ChartBias
    strength: int  # 1..5
    comment_de: str
    pattern: str = ""
    support: float | None = None
    resistance: float | None = None
    latency_ms: int | None = None
    id: int | None = None


@dataclass(slots=True)
class Candidate:
    """A trade idea detected by a deterministic setup, before review and persistence."""

    symbol: str
    direction: Direction
    setup: str
    timeframe: str
    intraday: bool
    horizon_min: int
    decision_ts: int  # time the decision is made (close time of the decision bar)
    data_ts: int  # close time of the newest bar used
    ref_price: float
    atr: float
    stop_dist: float
    target_dist: float
    components: dict[str, float] = field(default_factory=dict)  # score components 0..1 by name
    score: float = 0.0
    features: dict[str, Any] = field(default_factory=dict)
    news_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class Signal:
    created_ts: int
    data_ts: int
    symbol: str
    direction: Direction
    setup: str
    timeframe: str
    horizon_min: int
    ref_price: float
    atr: float
    stop_dist: float
    target_dist: float
    score: float
    confidence: float
    engine_version: str
    status: SignalStatus = "review"
    intraday: bool = True
    components: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    expected_r: float = 0.0
    issued_ts: int | None = None
    llm_verdict: LlmVerdict | None = None
    llm_conviction: float | None = None
    llm_latency_ms: int | None = None
    rationale_de: str = ""
    risks_de: str = ""
    news_ids: list[int] = field(default_factory=list)
    actionable: bool = False
    shadow_reason: str = ""
    status_reason: str = ""
    entry_deadline_ts: int | None = None
    checked_ts: int | None = None  # open time of the last 1-minute bar processed by the tracker
    entry_ts: int | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    expires_ts: int | None = None
    exit_ts: int | None = None
    exit_price: float | None = None
    outcome_r: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    mailed_ts: int | None = None
    id: int | None = None

    @property
    def rr(self) -> float:
        """Reward-to-risk ratio of the planned trade."""
        return self.target_dist / self.stop_dist if self.stop_dist > 0 else 0.0

    @property
    def sign(self) -> int:
        return 1 if self.direction == "LONG" else -1


@dataclass(slots=True)
class Portfolio:
    key: str  # 'ai' | 'me'
    name: str
    base_currency: str
    start_cash: float
    cash: float
    created_ts: int
    reset_ts: int
    fees_paid: float = 0.0
    id: int | None = None


@dataclass(slots=True)
class Order:
    portfolio_id: int
    created_ts: int
    symbol: str
    intent: OrderIntent
    side: Direction  # side of the position that is opened / closed
    origin: OrderOrigin
    status: OrderStatus = "pending"
    qty: float | None = None
    notional_base: float | None = None
    ref_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    stop_dist: float | None = None
    target_dist: float | None = None
    max_drift: float | None = None
    time_stop_ts: int | None = None
    flatten_at_close: bool = False
    position_id: int | None = None
    signal_id: int | None = None
    status_reason: str = ""
    expire_ts: int | None = None
    filled_ts: int | None = None
    filled_qty: float | None = None
    fill_price: float | None = None
    fill_fx: float | None = None
    fee_base: float | None = None
    id: int | None = None


@dataclass(slots=True)
class Position:
    portfolio_id: int
    symbol: str
    side: Direction
    qty: float
    entry_price: float
    entry_fx: float
    opened_ts: int
    status: PositionStatus = "open"
    stop_price: float | None = None
    target_price: float | None = None
    time_stop_ts: int | None = None
    flatten_at_close: bool = False
    signal_id: int | None = None
    checked_ts: int | None = None  # open time of the last 1-minute bar checked for stop/target
    closed_ts: int | None = None
    exit_price: float | None = None
    exit_fx: float | None = None
    exit_reason: str | None = None
    fees_base: float = 0.0
    realized_pnl: float | None = None
    id: int | None = None

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1


@dataclass(frozen=True, slots=True)
class EquityPoint:
    portfolio_id: int
    ts: int
    equity: float
    cash: float
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    open_positions: int = 0
