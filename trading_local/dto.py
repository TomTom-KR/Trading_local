"""API payload models (pydantic). This file is the contract between backend services, the REST/WebSocket
API and the browser GUI.

Rules
* All timestamps are epoch seconds (UTC) as ``int``. The GUI converts them to local time.
* Money amounts are in the portfolio base currency unless a field name says otherwise.
* Prices are in the units the data source quotes (London shares in pence); ``price_scale``/``currency``
  on the instrument explain the unit.
* Texts ending in ``_de`` are German and may be shown to the user verbatim.
* Series are compact arrays: bars ``[ts, open, high, low, close, volume]``, lines ``[ts, value]``.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from trading_local.models import (
    AssetClass,
    Direction,
    LlmVerdict,
    OrderIntent,
    OrderOrigin,
    OrderStatus,
    Region,
    SessionPhase,
    SignalStatus,
)


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------------------------
# Market data
# ------------------------------------------------------------------------------------------------
class QuoteOut(ApiModel):
    symbol: str
    price: float
    ts: int  # time of the last trade/quote
    change_pct: float | None = None  # percent versus previous close
    prev_close: float | None = None
    day_open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    day_volume: float | None = None
    bid: float | None = None
    ask: float | None = None
    market_state: str | None = None
    source: str = "quote"


class SignalBrief(ApiModel):
    id: int
    direction: Direction
    status: SignalStatus
    confidence: float
    created_ts: int


class InstrumentOut(ApiModel):
    symbol: str
    name: str
    asset_class: AssetClass
    region: Region
    exchange: str
    exchange_name: str = ""  # human readable, e.g. 'Xetra'
    currency: str
    price_scale: float = 1.0
    timezone: str = "UTC"
    delay_min: int = 0
    tradable: bool = True
    active: bool = True
    benchmark: str | None = None
    sort_order: int = 0
    phase: SessionPhase = "closed"
    is_open: bool = False
    quote: QuoteOut | None = None
    open_signal: SignalBrief | None = None  # newest actionable signal that is still open


class InstrumentCreate(RequestModel):
    symbol: str = Field(min_length=1, max_length=24)


class InstrumentUpdate(RequestModel):
    active: bool | None = None
    tradable: bool | None = None


class BarsOut(ApiModel):
    symbol: str
    timeframe: Literal["1m", "5m", "15m", "1h", "1d"]
    currency: str
    price_scale: float = 1.0
    delay_min: int = 0
    bars: list[tuple[int, float, float, float, float, float]]  # [ts, open, high, low, close, volume], ascending
    overlays: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)  # 'ema20', 'ema50', 'ema200', 'vwap'


class SparklinesOut(ApiModel):
    # symbol -> [[ts, close], ...] of the current (or last) trading session, at most ~80 points
    series: dict[str, list[tuple[int, float]]]


# ------------------------------------------------------------------------------------------------
# Trading sessions
# ------------------------------------------------------------------------------------------------
class ExchangeSessionOut(ApiModel):
    exchange: str
    name: str
    region: Region
    timezone: str
    phase: SessionPhase
    is_open: bool
    since: int | None = None
    until: int | None = None
    next_open: int | None = None
    next_close: int | None = None
    note: str = ""


class RegionSessionOut(ApiModel):
    region: Region
    label: str  # 'Asien' | 'Europa' | 'USA'
    is_open: bool  # at least one of the region's venues is in its regular session
    exchanges: list[ExchangeSessionOut]


class TimelineSegmentOut(ApiModel):
    exchange: str
    region: Region
    start: int
    end: int
    kind: Literal["regular", "pre", "post"] = "regular"


class SessionsOut(ApiModel):
    now: int
    regions: list[RegionSessionOut]  # ordered ASIA, EU, US (the way the trading day moves round the globe)
    timeline_start: int
    timeline_end: int
    timeline: list[TimelineSegmentOut]


# ------------------------------------------------------------------------------------------------
# News
# ------------------------------------------------------------------------------------------------
class NewsImpactOut(ApiModel):
    symbol: str
    sentiment: float  # -1..1


class NewsAnalysisOut(ApiModel):
    analyzed_ts: int
    relevance: int  # 1..5
    sentiment: float  # -1..1
    confidence: float
    category: str
    horizon: str
    summary_de: str
    symbols: list[NewsImpactOut] = Field(default_factory=list)


class NewsOut(ApiModel):
    id: int
    source: str
    source_name: str = ""
    url: str
    title: str
    summary: str = ""
    lang: str = "en"
    published_ts: int | None = None
    fetched_ts: int
    status: str
    hinted_symbols: list[str] = Field(default_factory=list)
    analysis: NewsAnalysisOut | None = None


# ------------------------------------------------------------------------------------------------
# Agenda (macro data, central banks, earnings)
# ------------------------------------------------------------------------------------------------
class AgendaEventOut(ApiModel):
    """One calendar entry with expectation, actual value and surprise."""

    id: int
    kind: Literal["macro", "central_bank", "earnings"]
    kind_de: str  # 'Konjunkturdaten' | 'Notenbank' | 'Quartalszahlen'
    source: str
    currency: str = ""  # macro and central-bank events
    symbol: str | None = None  # earnings
    name: str = ""  # instrument name of an earnings release
    title: str  # as published (English for macro events)
    event_ts: int
    time_known: bool = True
    importance: int  # 1 low, 2 medium, 3 high
    forecast_text: str = ""
    previous_text: str = ""
    actual_text: str = ""
    surprise: float | None = None  # actual - forecast (earnings: percent of the estimate)
    surprise_z: float | None = None  # in units of the event's earlier surprises
    surprise_label: Literal["above", "below", "inline", ""] = ""
    result_de: str = ""  # e.g. '0.4% statt 0.3% – über Erwartung (+1,8σ)'
    affected: int = 0  # tradable instruments the event concerns
    block_from_ts: int | None = None  # start of the entry block before the release (macro and central banks)
    blocks_entries: bool = False  # the entry block is active right now


# ------------------------------------------------------------------------------------------------
# Technical analysis / chart analysis
# ------------------------------------------------------------------------------------------------
class ReadingOut(ApiModel):
    """One human-readable technical reading, e.g. 'Trend 15 min: aufwaerts (EMA20 ueber EMA50)'."""

    key: str
    label_de: str
    value_de: str
    stance: Literal["bullish", "bearish", "neutral"] = "neutral"


class LevelOut(ApiModel):
    price: float
    kind: Literal["support", "resistance", "vwap", "prev_close", "day_high", "day_low", "prev_high", "prev_low"]
    label_de: str


class ChartAnalysisOut(ApiModel):
    id: int
    symbol: str
    ts: int
    timeframe: str
    model: str
    bias: Literal["bullish", "bearish", "neutral"]
    strength: int
    pattern: str = ""
    support: float | None = None
    resistance: float | None = None
    comment_de: str
    latency_ms: int | None = None


class AnalysisOut(ApiModel):
    symbol: str
    as_of: int | None = None  # decision time of the feature snapshot
    data_ts: int | None = None  # close time of the newest bar used
    available: bool = True  # False when there is not enough data yet
    note_de: str = ""
    features: dict[str, float | None] = Field(default_factory=dict)
    readings: list[ReadingOut] = Field(default_factory=list)
    levels: list[LevelOut] = Field(default_factory=list)
    news_score: float | None = None  # -1..1 time-decayed news sentiment
    chart_analysis: ChartAnalysisOut | None = None


# ------------------------------------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------------------------------------
class SignalOut(ApiModel):
    id: int
    created_ts: int
    issued_ts: int | None = None
    data_ts: int
    symbol: str
    name: str = ""
    region: Region = "GLOBAL"
    currency: str = ""
    delay_min: int = 0
    direction: Direction
    setup: str
    setup_label_de: str = ""
    timeframe: str
    intraday: bool = True
    horizon_min: int
    ref_price: float
    atr: float
    stop_dist: float
    target_dist: float
    rr: float
    score: float
    confidence: float
    expected_r: float = 0.0
    llm_verdict: LlmVerdict | None = None
    llm_conviction: float | None = None
    rationale_de: str = ""
    risks_de: str = ""
    actionable: bool = False
    shadow_reason: str = ""
    status: SignalStatus
    status_reason: str = ""
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
    engine_version: str = ""
    last_price: float | None = None  # latest quote (for open signals)
    open_r: float | None = None  # unrealised result in R for active signals


class SignalDetailOut(SignalOut):
    components: dict[str, float] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    news: list[NewsOut] = Field(default_factory=list)


class StatBlock(ApiModel):
    """Outcome statistics of a set of closed signals (results in R multiples after costs)."""

    n: int = 0
    wins: int = 0
    win_rate: float | None = None
    win_rate_lo: float | None = None  # 95 % Wilson interval
    win_rate_hi: float | None = None
    avg_r: float | None = None
    avg_r_lo: float | None = None  # 95 % bootstrap interval
    avg_r_hi: float | None = None
    total_r: float = 0.0
    profit_factor: float | None = None


class GroupStat(ApiModel):
    key: str
    label_de: str
    stats: StatBlock


class CalibrationBin(ApiModel):
    lo: float
    hi: float
    n: int
    predicted: float  # mean predicted win probability in the bin
    observed: float | None = None  # observed win rate (None when n == 0)


class SignalStatsOut(ApiModel):
    since_ts: int | None = None
    engine_version: str = ""
    actionable: StatBlock  # signals that were issued to the user
    shadow: StatBlock  # tracked but not issued
    by_setup: list[GroupStat] = Field(default_factory=list)
    by_region: list[GroupStat] = Field(default_factory=list)
    by_direction: list[GroupStat] = Field(default_factory=list)
    by_llm_verdict: list[GroupStat] = Field(default_factory=list)
    calibration: list[CalibrationBin] = Field(default_factory=list)
    note_de: str = ""


# ------------------------------------------------------------------------------------------------
# Paper trading
# ------------------------------------------------------------------------------------------------
class PositionOut(ApiModel):
    id: int
    portfolio_key: str
    symbol: str
    name: str = ""
    currency: str = ""
    side: Direction
    qty: float
    entry_price: float
    entry_fx: float
    opened_ts: int
    stop_price: float | None = None
    target_price: float | None = None
    time_stop_ts: int | None = None
    flatten_at_close: bool = False
    signal_id: int | None = None
    status: Literal["open", "closed"]
    # open positions, marked to market
    last_price: float | None = None
    last_ts: int | None = None
    exposure_base: float | None = None  # signed market value in base currency (negative for shorts)
    unrealized_pnl: float | None = None
    unrealized_pct: float | None = None
    # closed positions
    closed_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    fees_base: float = 0.0
    realized_pnl: float | None = None
    realized_pct: float | None = None


class OrderOut(ApiModel):
    id: int
    portfolio_key: str
    created_ts: int
    symbol: str
    name: str = ""
    intent: OrderIntent
    side: Direction
    qty: float | None = None
    notional_base: float | None = None
    ref_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    origin: OrderOrigin
    status: OrderStatus
    status_reason: str = ""
    expire_ts: int | None = None
    filled_ts: int | None = None
    filled_qty: float | None = None
    fill_price: float | None = None
    fill_fx: float | None = None
    fee_base: float | None = None
    signal_id: int | None = None
    position_id: int | None = None


class PortfolioSummaryOut(ApiModel):
    key: str
    name: str
    base_currency: str
    start_cash: float
    cash: float
    equity: float
    buying_power: float  # equity * max leverage - gross exposure
    long_exposure: float = 0.0
    short_exposure: float = 0.0  # positive number
    gross_leverage: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    total_return_pct: float = 0.0
    day_pnl: float = 0.0
    day_return_pct: float = 0.0
    open_positions: int = 0
    pending_orders: int = 0
    closed_trades: int = 0
    wins: int = 0
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    max_drawdown_pct: float | None = None
    sharpe: float | None = None  # annualised, from daily returns; None until enough history exists
    reset_ts: int
    as_of_ts: int


class PortfolioDetailOut(ApiModel):
    summary: PortfolioSummaryOut
    positions: list[PositionOut] = Field(default_factory=list)  # open positions
    pending_orders: list[OrderOut] = Field(default_factory=list)


class EquityCurvesOut(ApiModel):
    since_ts: int
    base_currency: str
    start_cash: float
    ai: list[tuple[int, float]] = Field(default_factory=list)
    me: list[tuple[int, float]] = Field(default_factory=list)
    # buy-and-hold of a benchmark index, scaled to start_cash at since_ts, in base currency
    benchmarks: dict[str, list[tuple[int, float]]] = Field(default_factory=dict)


class OrderRequest(RequestModel):
    """Manual order for the user's portfolio. Exactly one of ``amount_base`` / ``qty`` must be given."""

    symbol: str
    side: Direction
    amount_base: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    qty: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    stop_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    target_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    signal_id: int | None = Field(default=None, ge=1, le=2**63 - 1)  # set when the user follows an AI signal


class PositionUpdateRequest(RequestModel):
    """Set or remove (``None``) the protective levels of an open position.

    Replace semantics: BOTH levels are replaced; an omitted field removes that level. The GUI always sends
    both values."""

    stop_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    target_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class ResetRequest(RequestModel):
    confirm: bool
    start_capital: float | None = Field(default=None, gt=0, allow_inf_nan=False)


# ------------------------------------------------------------------------------------------------
# Validation (walk-forward / replay report)
# ------------------------------------------------------------------------------------------------
class FoldOut(ApiModel):
    index: int
    start_ts: int
    end_ts: int
    stats: StatBlock


class BaselineOut(ApiModel):
    """Random-entry Monte Carlo with identical stops, targets, holding times and costs."""

    n_sims: int
    mean_avg_r: float
    p95_avg_r: float
    p_value: float | None = None  # share of random runs that did at least as well as the engine


class ValidationResultOut(ApiModel):
    period_start_ts: int
    period_end_ts: int
    trading_days: int
    symbols: int
    decisions: int  # number of evaluated decision points
    overall: StatBlock
    by_setup: list[GroupStat] = Field(default_factory=list)
    by_region: list[GroupStat] = Field(default_factory=list)
    by_direction: list[GroupStat] = Field(default_factory=list)
    folds: list[FoldOut] = Field(default_factory=list)  # consecutive time blocks: stability over time
    calibration: list[CalibrationBin] = Field(default_factory=list)  # fitted on earlier folds only
    baseline: BaselineOut | None = None
    equity_r: list[tuple[int, float]] = Field(default_factory=list)  # cumulative R over time
    warnings_de: list[str] = Field(default_factory=list)
    verdict_de: str = ""


class ValidationRunOut(ApiModel):
    id: int
    ts: int
    status: Literal["running", "done", "failed"]
    engine_version: str
    message: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    result: ValidationResultOut | None = None


class ValidationRequest(RequestModel):
    days: int = Field(default=30, ge=2, le=365)


# ------------------------------------------------------------------------------------------------
# Operations
# ------------------------------------------------------------------------------------------------
class ActivityOut(ApiModel):
    id: int
    ts: int
    level: Literal["info", "warning", "error"]
    category: str
    message: str
    ref_type: str | None = None
    ref_id: int | None = None


class MailLogOut(ApiModel):
    id: int
    ts: int
    kind: str
    signal_id: int | None = None
    recipient: str
    subject: str
    status: str
    detail: str = ""


class MailTestOut(ApiModel):
    ok: bool
    status: str  # 'sent' | 'file' | 'skipped' | 'failed'
    detail_de: str = ""


class FeedStateOut(ApiModel):
    feed_id: str
    name: str
    region: str
    language: str
    category: str
    last_ok_ts: int | None = None
    last_error: str = ""
    fail_count: int = 0
    items_total: int = 0


class LlmCallOut(ApiModel):
    id: int
    ts: int
    kind: str
    ok: bool
    latency_ms: int
    queue_wait_ms: int = 0
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""


class JobStatusOut(ApiModel):
    job: str
    label_de: str = ""
    last_started_ts: int | None = None
    last_duration_ms: int | None = None
    ok: bool = True
    message: str = ""


class LlmStatusOut(ApiModel):
    model: str
    enabled: bool = True
    available: bool = False
    busy: bool = False
    queue_depth: int = 0
    last_latency_ms: int | None = None
    calls_last_hour: int = 0
    errors_last_hour: int = 0
    last_error: str = ""


class DataStatusOut(ApiModel):
    stream_connected: bool = False
    last_tick_ts: int | None = None
    last_rest_ok_ts: int | None = None
    rate_limited_until: int | None = None
    symbols_active: int = 0
    symbols_stale: int = 0  # open market but no fresh data
    backfill_progress: float | None = None  # 0..1 while the initial history download runs


class NewsStatusOut(ApiModel):
    feeds_total: int = 0
    feeds_failing: int = 0
    items_last_hour: int = 0
    analyzed_last_hour: int = 0
    backlog: int = 0
    last_poll_ts: int | None = None


class MailStatusOut(ApiModel):
    mode: str = "off"  # 'smtp' | 'file' | 'off'
    configured: bool = False
    recipient_masked: str = ""
    sent_last_hour: int = 0
    last_error: str = ""


class DbStatusOut(ApiModel):
    size_mb: float = 0.0
    bars_1m: int = 0
    first_bar_ts: int | None = None
    last_bar_ts: int | None = None
    news: int = 0
    signals: int = 0


class SystemStatusOut(ApiModel):
    server_time: int
    started_ts: int
    version: str
    engine_version: str
    base_currency: str
    offline: bool = False  # True: background jobs are disabled (read-only demo / test mode)
    llm: LlmStatusOut
    data: DataStatusOut
    news: NewsStatusOut
    mail: MailStatusOut
    db: DbStatusOut
    jobs: list[JobStatusOut] = Field(default_factory=list)


class ErrorOut(ApiModel):
    detail: str  # German message that can be shown to the user
