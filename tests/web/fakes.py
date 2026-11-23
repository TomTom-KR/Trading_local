"""Small stand-ins for the services the web layer talks to (everything except the repositories)."""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from trading_local.contracts import OrderRejected, TradeReview, TradeReviewContext
from trading_local.dto import (
    AnalysisOut,
    DataStatusOut,
    EquityCurvesOut,
    FeedStateOut,
    JobStatusOut,
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
    StatBlock,
)
from trading_local.models import ChartAnalysis, Instrument, NewsAnalysis, NewsItem, Position, Quote, Signal

REJECTED_SYMBOL = "REJECT"
REJECTION_TEXT = "Nicht genügend Kaufkraft für diese Order."


class FakeMarketData:
    def __init__(self) -> None:
        self.quote_book: dict[str, Quote] = {}
        self.lookups: dict[str, Instrument] = {}
        self.lookup_error: Exception | None = None
        self.history_requests: list[str] = []

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def refresh(self, now: datetime) -> list[str]:
        return []

    def quote(self, symbol: str) -> Quote | None:
        return self.quote_book.get(symbol)

    def quotes(self) -> dict[str, Quote]:
        return dict(self.quote_book)

    def status(self) -> DataStatusOut:
        return DataStatusOut(stream_connected=True, symbols_active=len(self.quote_book))

    async def lookup(self, symbol: str) -> Instrument | None:
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.lookups.get(symbol)

    async def ensure_history(self, symbol: str) -> None:
        self.history_requests.append(symbol)


class FakeFx:
    def rate(self, currency: str, base: str, at_ts: int | None = None) -> float | None:
        return 1.0 if currency == base else 0.9


class FakeLlm:
    def __init__(self) -> None:
        self.status_error: Exception | None = None

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def status(self) -> LlmStatusOut:
        if self.status_error is not None:
            raise self.status_error
        return LlmStatusOut(model="qwen3.5:9b", available=True, queue_depth=2)

    async def analyze_news(self, item: NewsItem, candidates: Sequence[Instrument]) -> NewsAnalysis | None:
        return None

    async def review_trade(self, ctx: TradeReviewContext, *, timeout_s: float) -> TradeReview:
        return TradeReview(verdict="unavailable")

    async def analyze_chart(
        self, instrument: Instrument, timeframe: str, chart_png: bytes, facts: Mapping[str, Any]
    ) -> ChartAnalysis | None:
        return None


class FakeNews:
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def poll(self, now: datetime) -> int:
        return 0

    def status(self) -> NewsStatusOut:
        return NewsStatusOut(feeds_total=3, items_last_hour=7)

    def feeds(self) -> list[FeedStateOut]:
        return [
            FeedStateOut(feed_id="reuters_business", name="Reuters Wirtschaft", region="GLOBAL", language="en",
                         category="business", items_total=12),
        ]


class FakeCharts:
    async def render(self, symbol: str, timeframe: str = "15m", **kwargs: Any) -> bytes | None:
        return None


class FakeMailer:
    def __init__(self, recipient: str) -> None:
        self.recipient = recipient  # deliberately NOT masked: the web layer must mask it itself

    async def send_signal(self, signal: Signal, instrument: Instrument, chart_png: bytes | None = None) -> bool:
        return False

    async def send_exit(self, signal: Signal, instrument: Instrument) -> bool:
        return False

    async def send_test(self) -> MailTestOut:
        return MailTestOut(ok=True, status="file", detail_de="Testmail in den Postausgang geschrieben.")

    def status(self) -> MailStatusOut:
        return MailStatusOut(mode="file", configured=True, recipient_masked=self.recipient)


@dataclass
class FakeBroker:
    now_ts: int
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def summary(self, portfolio_key: str) -> PortfolioSummaryOut:
        return PortfolioSummaryOut(
            key=portfolio_key, name={"ai": "KI", "me": "Ich"}[portfolio_key], base_currency="EUR",
            start_cash=100_000.0, cash=100_000.0, equity=100_000.0, buying_power=100_000.0,
            reset_ts=self.now_ts - 86_400, as_of_ts=self.now_ts,
        )

    def detail(self, portfolio_key: str) -> PortfolioDetailOut:
        return PortfolioDetailOut(summary=self.summary(portfolio_key), positions=[self._position(portfolio_key, 7)])

    def trades(self, portfolio_key: str, limit: int = 200) -> list[PositionOut]:
        self.calls.append(("trades", portfolio_key, limit))
        return [self._position(portfolio_key, 3, status="closed")]

    def orders(self, portfolio_key: str, limit: int = 200) -> list[OrderOut]:
        self.calls.append(("orders", portfolio_key, limit))
        return [self._order(portfolio_key, 5)]

    def equity_curves(self, since_ts: int | None = None, max_points: int = 600) -> EquityCurvesOut:
        self.calls.append(("equity", since_ts, max_points))
        return EquityCurvesOut(since_ts=since_ts or 0, base_currency="EUR", start_cash=100_000.0,
                               ai=[(self.now_ts, 100_500.0)], me=[(self.now_ts, 99_800.0)])

    def place_order(self, portfolio_key: str, req: OrderRequest) -> OrderOut:
        self.calls.append(("place", portfolio_key, req))
        if req.symbol == REJECTED_SYMBOL:
            raise OrderRejected(REJECTION_TEXT)
        return self._order(portfolio_key, 11, symbol=req.symbol)

    def cancel_order(self, portfolio_key: str, order_id: int) -> OrderOut:
        self.calls.append(("cancel", portfolio_key, order_id))
        if order_id == 404:
            raise OrderRejected("Die Order 404 ist nicht mehr offen.")
        return self._order(portfolio_key, order_id, status="cancelled")

    def close_position(self, portfolio_key: str, position_id: int) -> OrderOut:
        self.calls.append(("close", portfolio_key, position_id))
        return self._order(portfolio_key, 12, intent="close")

    def update_position(
        self, portfolio_key: str, position_id: int, *, stop_price: float | None, target_price: float | None
    ) -> PositionOut:
        self.calls.append(("update", portfolio_key, position_id, stop_price, target_price))
        if stop_price is not None and target_price is not None and stop_price >= target_price:
            raise OrderRejected("Der Stopp muss bei einer Long-Position unter dem Ziel liegen.")
        return self._position(portfolio_key, position_id).model_copy(
            update={"stop_price": stop_price, "target_price": target_price}
        )

    def execute_signal(self, signal: Signal) -> OrderOut | None:
        return None

    def process(self, now: datetime) -> list[Position]:
        return []

    def reset(self, start_capital: float | None = None) -> None:
        self.calls.append(("reset", start_capital))

    def _order(self, key: str, order_id: int, *, symbol: str = "NVDA", status: str = "pending",
               intent: str = "open") -> OrderOut:
        return OrderOut(id=order_id, portfolio_key=key, created_ts=self.now_ts, symbol=symbol, intent=intent,
                        side="LONG", qty=10.0, origin="manual", status=status)

    def _position(self, key: str, position_id: int, *, status: str = "open") -> PositionOut:
        return PositionOut(id=position_id, portfolio_key=key, symbol="NVDA", side="LONG", qty=10.0,
                           entry_price=180.0, entry_fx=0.92, opened_ts=self.now_ts - 3600, status=status)


class FakeEngine:
    version = "engine-test-1"

    def __init__(self) -> None:
        self.analysis_error: Exception | None = None
        self.chart_requests: list[tuple[str, str]] = []
        self.chart_queue_accepts = True

    async def on_minute(self, now: datetime, updated: Collection[str]) -> None: ...

    def analysis(self, symbol: str) -> AnalysisOut:
        if self.analysis_error is not None:
            raise self.analysis_error
        return AnalysisOut(symbol=symbol, note_de="Aufwärtstrend", features={"rsi14": 55.0})

    def stats(self, since_ts: int | None = None) -> SignalStatsOut:
        return SignalStatsOut(since_ts=since_ts, engine_version=self.version, actionable=StatBlock(n=3),
                              shadow=StatBlock(n=5))

    async def request_chart_analysis(self, symbol: str, timeframe: str = "15m") -> bool:
        self.chart_requests.append((symbol, timeframe))
        return self.chart_queue_accepts

    async def stop(self) -> None: ...


class FakeRuntime:
    def __init__(self, started_ts: int) -> None:
        self.started_ts = started_ts
        self.offline = True

    def jobs(self) -> list[JobStatusOut]:
        return [JobStatusOut(job="market_data", label_de="Kurse und Minutenbalken", last_started_ts=self.started_ts)]
