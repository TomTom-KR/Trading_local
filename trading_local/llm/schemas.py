"""Output models of the three LLM job types, the JSON schemas that constrain the model's answer, and the
conversion of a validated answer into the domain objects.

Two layers on purpose:

* The request schema is turned into a grammar by Ollama. Measured on this model (LLM probe): enums, required
  keys, integer bounds and ``maxItems`` are enforced; number bounds, string lengths and ``uniqueItems`` are
  NOT (string length limits even corrupted the output). The schemas therefore carry no such keywords, and
  the key order is deliberate: the model writes in schema order, so the text fields come first and act as
  cheap reasoning before the scores and the verdict.
* The pydantic models re-check every answer. Numbers are clamped into range, lists de-duplicated and
  trimmed, text normalised and stripped of links and e-mail addresses (prompt injection through a news
  headline must not put an attacker's link into a recommendation mail). An answer that cannot be repaired
  raises ``OutputError``.
"""

import math
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

from trading_local.contracts import TradeReview
from trading_local.models import ChartAnalysis, ChartBias, NewsAnalysis, NewsCategory, NewsHorizon

MAX_IMPACTS = 5
MAX_REASONS = 3
MAX_RISKS = 2
MAX_LEVELS = 3
LEVEL_MAX_DISTANCE = 0.25  # support / resistance further than 25 % from the last price are implausible
_SUMMARY_CHARS = 300
_POINT_CHARS = 160
_COMMENT_CHARS = 500
_ERROR_CHARS = 200

Verdict = Literal["confirm", "reduce", "veto"]
ChartStructure = Literal["uptrend", "downtrend", "range", "breakout_up", "breakdown", "reversal_up", "reversal_down"]

# A closed vocabulary instead of a free-text pattern: free text just parroted the prompt's examples (probe).
STRUCTURE_LABELS_DE: Mapping[str, str] = {
    "uptrend": "Aufwärtstrend",
    "downtrend": "Abwärtstrend",
    "range": "Seitwärtsspanne",
    "breakout_up": "Ausbruch nach oben",
    "breakdown": "Bruch nach unten",
    "reversal_up": "Umkehr nach oben",
    "reversal_down": "Umkehr nach unten",
}

# Conviction band of each verdict (the prompt asks for exactly these bands).
CONVICTION_BANDS: Mapping[str, tuple[float, float]] = {
    "confirm": (0.7, 1.0),
    "reduce": (0.4, 0.69),
    "veto": (0.0, 0.39),
}

# Garbled umlauts were seen as CJK / Hangul characters ('pr채gt', 'n盲chste') with the model's default sampling.
_CJK = re.compile("[⺀-꓏가-힯豈-﫿︰-﹏＀-￯\U00020000-\U0003ffff]")


# Links and addresses in the model's free text. None of the job types has a reason to write one, but a hostile
# headline can ask the model to repeat it, and the texts end up in the GUI and in the recommendation mails
# (where mail clients make them clickable). Matched: anything with a scheme ('https://', 'hxxp://' ...) or
# 'www.', e-mail addresses, and host names followed by a path ('evil.example/login'). Plain dotted words such
# as company names ('Amazon.com') and numbers ('1.0845/1.0850') are kept.
_LINK = re.compile(
    r"\b[a-z][a-z0-9+.-]*://\S*"
    r"|\bwww\.\S+"
    r"|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*",
    re.IGNORECASE,
)
LINK_REMOVED_DE = "[Link entfernt]"


class OutputError(ValueError):
    """The model's answer is unusable (wrong structure, empty or garbled text).

    ``str()`` is a short English reason for the logs, ``reason_de`` the German text for the user.
    """

    def __init__(self, reason: str, reason_de: str) -> None:
        super().__init__(reason)
        self.reason_de = reason_de


# ------------------------------------------------------------------------------------------------
# Answer models (what the model writes)
# ------------------------------------------------------------------------------------------------
class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class NewsImpactAnswer(_Answer):
    symbol: str
    sentiment: float


class NewsAnswer(_Answer):
    summary_de: str
    impacts: list[NewsImpactAnswer]
    category: NewsCategory
    horizon: NewsHorizon
    sentiment: float
    relevance: int
    confidence: float


class TradeReviewAnswer(_Answer):
    reasons_de: list[str]
    risks_de: list[str]
    verdict: Verdict
    conviction: float


class ChartAnswer(_Answer):
    structure: ChartStructure
    bias: ChartBias
    strength: int
    support: list[float]
    resistance: list[float]
    comment_de: str


# ------------------------------------------------------------------------------------------------
# Request schemas (what Ollama enforces)
# ------------------------------------------------------------------------------------------------
def _enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties)}


def news_schema(symbols: Sequence[str]) -> dict[str, Any]:
    """Schema of one news classification. ``impacts[].symbol`` is an enum of exactly ``symbols``, so the
    model cannot name any other ticker; without candidates the list must stay empty."""
    symbol = _enum(symbols) if symbols else {"type": "string"}
    impacts = {
        "type": "array",
        "items": _object({"symbol": symbol, "sentiment": {"type": "number"}}),
        "maxItems": MAX_IMPACTS if symbols else 0,
    }
    return _object(
        {
            "summary_de": {"type": "string"},
            "impacts": impacts,
            "category": _enum(get_args(NewsCategory)),
            "horizon": _enum(get_args(NewsHorizon)),
            "sentiment": {"type": "number"},
            "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
            "confidence": {"type": "number"},
        }
    )


def trade_review_schema() -> dict[str, Any]:
    """Schema of a trade review: the arguments first, then the verdict (the model argues before it decides)."""
    point_list = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    return _object(
        {
            "reasons_de": {**point_list, "maxItems": MAX_REASONS},
            "risks_de": {**point_list, "maxItems": MAX_RISKS},
            "verdict": _enum(get_args(Verdict)),
            "conviction": {"type": "number"},
        }
    )


def chart_schema() -> dict[str, Any]:
    """Schema of a chart analysis with 1-3 candidate levels per side (code picks the nearest plausible one)."""
    levels = {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": MAX_LEVELS}
    return _object(
        {
            "structure": _enum(get_args(ChartStructure)),
            "bias": _enum(get_args(ChartBias)),
            "strength": {"type": "integer", "minimum": 1, "maximum": 5},
            "support": levels,
            "resistance": levels,
            "comment_de": {"type": "string"},
        }
    )


# ------------------------------------------------------------------------------------------------
# Conversion into domain objects
# ------------------------------------------------------------------------------------------------
def to_news_analysis(
    data: Mapping[str, Any],
    *,
    candidates: Collection[str],
    news_id: int,
    analyzed_ts: int,
    model: str,
    latency_ms: int | None,
) -> NewsAnalysis:
    """Validate a news answer and map it to ``NewsAnalysis``.

    Scores are clamped into their ranges; impacts naming a symbol outside ``candidates`` are dropped, a
    repeated symbol keeps its first sentiment, at most ``MAX_IMPACTS`` remain. Raises ``OutputError``.
    """
    answer = _validate(NewsAnswer, data)
    symbols: dict[str, float] = {}
    for impact in answer.impacts:
        if impact.symbol in candidates and impact.symbol not in symbols and len(symbols) < MAX_IMPACTS:
            symbols[impact.symbol] = _clamp(impact.sentiment, -1.0, 1.0)
    return NewsAnalysis(
        news_id=news_id,
        analyzed_ts=analyzed_ts,
        model=model,
        relevance=int(_clamp(answer.relevance, 1, 5)),
        sentiment=_clamp(answer.sentiment, -1.0, 1.0),
        category=answer.category,
        horizon=answer.horizon,
        summary_de=german_text(answer.summary_de, "summary_de", _SUMMARY_CHARS),
        confidence=_clamp(answer.confidence, 0.0, 1.0),
        symbols=symbols,
        latency_ms=latency_ms,
    )


def to_trade_review(data: Mapping[str, Any], *, latency_ms: int | None) -> TradeReview:
    """Validate a review answer and map it to ``TradeReview`` (points joined by line breaks).

    The conviction is clamped into the band of the verdict, so that verdict and conviction never
    contradict each other (the verdict is what the engine acts on). Raises ``OutputError``.
    """
    answer = _validate(TradeReviewAnswer, data)
    reasons = _points(answer.reasons_de, "reasons_de", MAX_REASONS)
    risks = _points(answer.risks_de, "risks_de", MAX_RISKS)
    low, high = CONVICTION_BANDS[answer.verdict]
    return TradeReview(
        verdict=answer.verdict,
        conviction=round(_clamp(answer.conviction, low, high), 2),
        rationale_de="\n".join(reasons),
        risks_de="\n".join(risks),
        latency_ms=latency_ms,
    )


def to_chart_analysis(
    data: Mapping[str, Any],
    *,
    symbol: str,
    timeframe: str,
    ts: int,
    model: str,
    last_price: float | None,
    latency_ms: int | None,
) -> ChartAnalysis:
    """Validate a chart answer and map it to ``ChartAnalysis``.

    Support is the nearest proposed level below ``last_price``, resistance the nearest above it; levels
    more than 25 % away (or on the wrong side) are discarded, and without a known last price no level is
    kept (it could not be checked). Raises ``OutputError``.
    """
    answer = _validate(ChartAnswer, data)
    support, resistance = None, None
    if last_price is not None and math.isfinite(last_price) and last_price > 0:
        below = [x for x in answer.support if last_price * (1 - LEVEL_MAX_DISTANCE) <= x < last_price]
        above = [x for x in answer.resistance if last_price < x <= last_price * (1 + LEVEL_MAX_DISTANCE)]
        support = max(below, default=None)
        resistance = min(above, default=None)
    return ChartAnalysis(
        symbol=symbol,
        ts=ts,
        timeframe=timeframe,
        model=model,
        bias=answer.bias,
        strength=int(_clamp(answer.strength, 1, 5)),
        comment_de=german_text(answer.comment_de, "comment_de", _COMMENT_CHARS),
        pattern=STRUCTURE_LABELS_DE[answer.structure],
        support=support,
        resistance=resistance,
        latency_ms=latency_ms,
    )


def german_text(value: str, field: str, limit: int) -> str:
    """Whitespace-normalised text of at most ``limit`` characters (cut with '…') in which every link and e-mail
    address is replaced by '[Link entfernt]' (see ``_LINK``).

    Raises ``OutputError`` if it is empty or contains CJK / Hangul characters (garbled umlauts).
    """
    text = _LINK.sub(LINK_REMOVED_DE, " ".join(value.split()))
    if not text:
        raise OutputError(f"{field} is empty", f"Feld {field} ist leer")
    if _CJK.search(text):
        raise OutputError(f"{field} contains garbled characters", f"Feld {field} enthält verstümmelte Zeichen")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _points(values: Sequence[str], field: str, limit: int) -> list[str]:
    """Non-empty, de-duplicated bullet points (at most ``limit``)."""
    points: list[str] = []
    for value in values:
        if not value.strip():
            continue
        point = german_text(value, field, _POINT_CHARS)
        if point not in points:
            points.append(point)
    return points[:limit]


def _validate[M: _Answer](model: type[M], data: Mapping[str, Any]) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = [(".".join(map(str, error["loc"])) or "$", error["msg"]) for error in exc.errors()]
        problems = "; ".join(f"{loc}: {msg}" for loc, msg in errors)
        fields = ", ".join(dict.fromkeys(loc for loc, _ in errors))
        reason, reason_de = f"validation: {problems}", f"Ungültige Felder: {fields}"
        raise OutputError(reason[:_ERROR_CHARS], reason_de[:_ERROR_CHARS]) from None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)
