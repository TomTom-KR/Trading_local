"""Scripted stand-in for ``OllamaClient`` and sample inputs for the LLM tests."""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from trading_local.contracts import TradeReviewContext
from trading_local.llm.client import ChatResult, LlmError
from trading_local.models import Candidate, Instrument, NewsItem
from trading_local.testing import SAMPLE_INSTRUMENTS

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
NVDA, SAP, BTC = INSTRUMENTS["NVDA"], INSTRUMENTS["SAP.DE"], INSTRUMENTS["BTC-USD"]
T0 = 1_795_186_800  # 2026-11-20 15:00 UTC, the time of the shared clock fixture
DEFAULT_FACTS = {"Setup": "Ausbruch aus der Eröffnungsspanne", "Trend 15 Min": "aufwärts", "RSI 15 Min": 61.2}


@dataclass(slots=True)
class FakeCall:
    system: str
    user: str
    schema: dict[str, Any]
    images: tuple[bytes, ...]
    think: bool
    timeout_s: float
    options: dict[str, Any]

    @property
    def kind(self) -> str:
        properties = self.schema["properties"]
        if "summary_de" in properties:
            return "news"
        return "trade_review" if "verdict" in properties else "chart"


type Reply = Mapping[str, Any] | LlmError | Callable[[FakeCall], Mapping[str, Any]]


def news_answer(call: FakeCall | None = None, **overrides: Any) -> dict[str, Any]:
    """A valid news answer naming the first candidate of the call's schema (if any)."""
    impacts = []
    if call is not None:
        enum = call.schema["properties"]["impacts"]["items"]["properties"]["symbol"].get("enum", [])
        impacts = [{"symbol": enum[0], "sentiment": 0.6}] if enum else []
    answer = {
        "summary_de": "Nvidia übertrifft die Erwartungen.",
        "impacts": impacts,
        "category": "earnings",
        "horizon": "days",
        "sentiment": 0.6,
        "relevance": 4,
        "confidence": 0.8,
    }
    return {**answer, **overrides}


def review_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "reasons_de": ["Trend 15 Min aufwärts", "Chance/Risiko-Verhältnis 2.0"],
        "risks_de": ["Keine wesentlichen Gegenargumente"],
        "verdict": "confirm",
        "conviction": 0.85,
    }
    return {**answer, **overrides}


def chart_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "structure": "uptrend",
        "bias": "bullish",
        "strength": 4,
        "support": [98.5, 95.0],
        "resistance": [103.0, 110.0],
        "comment_de": "Der Kurs steigt über EMA 20. Wichtig ist der Widerstand bei 103.",
    }
    return {**answer, **overrides}


def auto_reply(call: FakeCall) -> dict[str, Any]:
    """A valid answer for whatever job type the call is."""
    match call.kind:
        case "news":
            return news_answer(call)
        case "trade_review":
            return review_answer()
        case _:
            return chart_answer()


class FakeOllamaClient:
    """``ChatClient`` double: scripted replies, optional delay or gate, records every call.

    Replies are consumed in order (a dict is returned as data, an ``LlmError`` is raised, a callable gets
    the call); without scripted replies ``default`` answers. ``gate`` (if set) blocks every call until it is
    opened. ``timeout_s`` is honoured like the real client (``LlmError`` 'timeout').
    """

    def __init__(
        self, replies: Iterable[Reply] = (), *, default: Callable[[FakeCall], Mapping[str, Any]] = auto_reply
    ) -> None:
        self.replies: deque[Reply] = deque(replies)
        self.default = default
        self.delay_s = 0.0
        self.gate: asyncio.Event | None = None
        self.calls: list[FakeCall] = []
        self.cancelled: list[FakeCall] = []
        self.call_started = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.healthy = True
        self.health_calls = 0
        self.warmups = 0
        self.warmup_error: LlmError | None = None
        self.closed = False

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: Sequence[bytes] = (),
        think: bool,
        timeout_s: float,
        options: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        call = FakeCall(system, user, schema, tuple(images), think, timeout_s, dict(options or {}))
        self.calls.append(call)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.call_started.set()
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                if self.gate is not None:
                    await self.gate.wait()
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
        except TimeoutError:
            raise timeout_error(timeout_s, latency_ms=_ms(started)) from None
        except asyncio.CancelledError:
            self.cancelled.append(call)
            raise
        finally:
            self.active -= 1
        reply = self.replies.popleft() if self.replies else self.default
        if isinstance(reply, LlmError):
            raise reply
        data = reply(call) if callable(reply) else reply
        return ChatResult(data=dict(data), prompt_tokens=800, output_tokens=90, latency_ms=max(_ms(started), 7))

    async def health(self) -> bool:
        self.health_calls += 1
        return self.healthy

    async def warm_up(self) -> int:
        self.warmups += 1
        if self.warmup_error is not None:
            raise self.warmup_error
        return 5

    async def close(self) -> None:
        self.closed = True


def _ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def unreachable_error() -> LlmError:
    return LlmError("Ollama not reachable", reason_de="Ollama nicht erreichbar", unreachable=True)


def timeout_error(seconds: float, *, latency_ms: int = 0) -> LlmError:
    return LlmError(
        f"timeout after {seconds:g} s", reason_de=f"Zeitüberschreitung nach {seconds:g} s", latency_ms=latency_ms
    )


def make_news(
    news_id: int = 1, title: str = "Nvidia beats estimates", summary: str = "Shares rise 4 %.", **kw: Any
) -> NewsItem:
    return NewsItem(
        uid=f"uid-{news_id}",
        title_key=title.lower(),
        source="reuters_business",
        url=f"https://example.invalid/{news_id}",
        title=title,
        fetched_ts=T0,
        summary=summary,
        published_ts=T0 - 120,
        id=news_id,
        **kw,
    )


def make_candidate(direction: str = "LONG", **overrides: Any) -> Candidate:
    values: dict[str, Any] = {
        "symbol": "NVDA",
        "direction": direction,
        "setup": "orb_breakout",
        "timeframe": "15m",
        "intraday": True,
        "horizon_min": 120,
        "decision_ts": T0,
        "data_ts": T0 - 60,
        "ref_price": 100.0,
        "atr": 1.0,
        "stop_dist": 1.5,
        "target_dist": 3.0,
        "score": 0.64,
    }
    values.update(overrides)
    return Candidate(**values)


def make_review_context(
    direction: str = "LONG",
    *,
    instrument: Instrument = NVDA,
    facts: dict[str, Any] | None = None,
    headlines: list[tuple[str, float]] | None = None,
    chart_png: bytes | None = None,
    **candidate: Any,
) -> TradeReviewContext:
    return TradeReviewContext(
        instrument=instrument,
        candidate=make_candidate(direction, symbol=instrument.symbol, **candidate),
        facts=facts if facts is not None else dict(DEFAULT_FACTS),
        headlines=headlines if headlines is not None else [("Nvidia erhält Großauftrag", 0.7)],
        market_note="S&P 500: 15-Min-Trend aufwärts, 60 Min +0,40 %",
        chart_png=chart_png,
    )


@dataclass(slots=True)
class Gate:
    """Helper: block the fake client, wait until the first call is in flight, then open."""

    client: FakeOllamaClient
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self.client.gate = self.event

    async def wait_busy(self, calls: int = 1) -> None:
        async with asyncio.timeout(2):
            while len(self.client.calls) < calls:
                await asyncio.sleep(0.001)

    def open(self) -> None:
        self.event.set()
