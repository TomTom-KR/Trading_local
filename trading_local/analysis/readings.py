"""German, human-readable views of one feature row (pure functions: no I/O, no clock, no state).

The signal engine uses them for

* the analysis panel of the GUI (``analysis_out``: readings, price levels, finite features),
* the deterministic rationale of a signal when the LLM gave none (``rationale_de``),
* the fact sheet and the market note of an LLM trade review (``review_facts``, ``market_note_de``),
* the facts of an LLM chart analysis (``chart_facts``).

A feature row is any mapping with the keys of ``features.FEATURE_COLUMNS`` - a live row or the rounded
snapshot stored with a signal (``Signal.features``). A missing or non-finite value simply leaves out the
reading, level or fact that depends on it.

The thresholds below only choose WORDS for the display (e.g. 'hoch' above 1.5x volume); they never influence
a decision. Decision thresholds live in ``params.EngineParams``.
"""

import math
from collections.abc import Callable, Mapping
from typing import Any, Literal

from trading_local.analysis.features import FEATURE_COLUMNS, feature_float
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.db.repo_derivs import DerivMetrics
from trading_local.dto import AnalysisOut, ChartAnalysisOut, LevelOut, ReadingOut
from trading_local.models import Candidate, Instrument, Signal
from trading_local.presenters import setup_label_de

Stance = Literal["bullish", "bearish", "neutral"]
LevelKind = Literal["support", "resistance", "vwap", "prev_close", "day_high", "day_low", "prev_high", "prev_low"]

_P = ENGINE_PARAMS
_num = feature_float

_FLAT_ATR = 0.1  # EMA spread / VWAP distance (in ATR) below which the display says 'seitwärts' / 'am VWAP'
_LEVEL_MERGE_ATR = 0.1  # price levels closer than this many ATRs are shown once
_RSI_OVERBOUGHT, _RSI_OVERSOLD = 70.0, 30.0  # textbook RSI extremes
_RSI_POSITIVE, _RSI_NEGATIVE = 55.0, 45.0
_RELVOL_HIGH, _RELVOL_LOW = 1.5, 0.7
_RANK_HIGH, _RANK_LOW = 0.8, 0.2
_NEWS_NEUTRAL = 0.1  # |news score| below this is 'ausgeglichen'

COMPONENT_LABELS_DE: dict[str, str] = {
    "trend": "Trend",
    "momentum": "Momentum",
    "volume": "Volumen",
    "structure": "Kursstruktur",
    "market": "Marktumfeld",
    "news": "Nachrichtenlage",
    "volatility": "Volatilität",
}

_DE_SEPARATORS = str.maketrans({",": ".", ".": ","})


# ------------------------------------------------------------------------------------------------
# Formatting
# ------------------------------------------------------------------------------------------------
def de_number(value: float, digits: int = 2, *, signed: bool = False) -> str:
    """German number format: '.' groups thousands, ',' separates decimals ('1.234,56'; signed '+0,40')."""
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    return text.translate(_DE_SEPARATORS)


def de_price(value: float) -> str:
    """A price in German format: four decimals below 10 (FX, pence fractions), otherwise two."""
    return de_number(value, 4 if abs(value) < 10 else 2)


def _sig(value: float, digits: int = 6) -> float:
    """``value`` rounded to ``digits`` significant digits (compact facts for the LLM)."""
    return float(f"{value:.{digits}g}")


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def direction_de(direction: str) -> str:
    return "Long" if direction == "LONG" else "Short"


# ------------------------------------------------------------------------------------------------
# Classifications shared by readings, facts and rationale
# ------------------------------------------------------------------------------------------------
def _ema_trend(row: Mapping[str, Any], fast_key: str, slow_key: str, atr_key: str) -> tuple[str, Stance, float] | None:
    """(word, stance, spread in ATR) of an EMA pair; None when an input is missing."""
    fast, slow, atr = _num(row, fast_key), _num(row, slow_key), _num(row, atr_key)
    if not _finite(fast, slow, atr) or atr <= 0.0:
        return None
    spread = (fast - slow) / atr
    if spread > _FLAT_ATR:
        return "aufwärts", "bullish", spread
    if spread < -_FLAT_ATR:
        return "abwärts", "bearish", spread
    return "seitwärts", "neutral", spread


def _daily_trend(row: Mapping[str, Any]) -> tuple[str, Stance, str] | None:
    """(word, stance, explanation) from yesterday's close and the daily EMAs; None without data."""
    close, fast, slow = _num(row, "prev_day_close"), _num(row, "ema_daily_fast"), _num(row, "ema_daily_slow")
    if not _finite(close, fast):
        return None
    f, s = f"EMA{_P.ema_daily_fast}", f"EMA{_P.ema_daily_slow}"
    if not math.isfinite(slow):
        if close > fast:
            return "aufwärts", "bullish", f"Schluss über {f}"
        return "abwärts", "bearish", f"Schluss unter {f}"
    if close > fast > slow:
        return "aufwärts", "bullish", f"Schluss über {f} über {s}"
    if close < fast < slow:
        return "abwärts", "bearish", f"Schluss unter {f} unter {s}"
    return "uneinheitlich", "neutral", f"Schluss, {f} und {s} nicht geordnet"


def _bench_trend_word(row: Mapping[str, Any]) -> tuple[str, Stance] | None:
    """Benchmark trend word; ``bench_trend`` is already an EMA spread in benchmark ATRs."""
    trend = _num(row, "bench_trend")
    if not math.isfinite(trend):
        return None
    if trend > _FLAT_ATR:
        return "aufwärts", "bullish"
    if trend < -_FLAT_ATR:
        return "abwärts", "bearish"
    return "seitwärts", "neutral"


def _rsi_word(rsi: float) -> tuple[str, Stance]:
    if rsi >= _RSI_OVERBOUGHT:
        return "überkauft", "neutral"
    if rsi <= _RSI_OVERSOLD:
        return "überverkauft", "neutral"
    if rsi > _RSI_POSITIVE:
        return "positiv", "bullish"
    if rsi < _RSI_NEGATIVE:
        return "negativ", "bearish"
    return "neutral", "neutral"


def _level_word(value: float, high: float, low: float, high_word: str, low_word: str) -> str:
    return high_word if value >= high else low_word if value <= low else "normal"


# ------------------------------------------------------------------------------------------------
# Readings
# ------------------------------------------------------------------------------------------------
def _reading(key: str, label: str, value: str, stance: Stance = "neutral") -> ReadingOut:
    return ReadingOut(key=key, label_de=label, value_de=value, stance=stance)


def _trend_reading(row: Mapping[str, Any], key: str, label: str, suffix: str) -> ReadingOut | None:
    trend = _ema_trend(row, f"ema_fast_{suffix}", f"ema_slow_{suffix}", "atr_fast" if suffix == "f" else "atr_slow")
    if trend is None:
        return None
    word, stance, spread = trend
    fast, slow = f"EMA{_P.ema_fast}", f"EMA{_P.ema_slow}"
    if word == "seitwärts":
        return _reading(key, label, f"seitwärts ({fast} nahe {slow})")
    relation = "über" if spread > 0 else "unter"
    return _reading(key, label, f"{word} ({fast} {relation} {slow}, Abstand {de_number(abs(spread), 1)} ATR)", stance)


def _daily_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    trend = _daily_trend(row)
    if trend is None:
        return None
    word, stance, why = trend
    return _reading("trend_daily", "Trend Tag", f"{word} ({why})", stance)


def _momentum_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    rsi = _num(row, "rsi_slow")
    if not math.isfinite(rsi):
        return None
    word, stance = _rsi_word(rsi)
    return _reading("momentum", f"Momentum (RSI {_P.slow_minutes} Min)", f"{rsi:.0f} – {word}", stance)


def _strength_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    adx = _num(row, "adx_slow")
    if not math.isfinite(adx):
        return None
    phase = "Trendphase" if adx >= _P.adx_trend_min else "Seitwärtsphase"
    return _reading("trend_strength", f"Trendstärke (ADX {_P.slow_minutes} Min)", f"{adx:.0f} – {phase}")


def _volume_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    rel = _num(row, "rel_volume")
    if not math.isfinite(rel):
        return None
    word = _level_word(rel, _RELVOL_HIGH, _RELVOL_LOW, "hoch", "niedrig")
    label = f"Volumen ({_P.decision_minutes} Min)"
    return _reading("volume", label, f"{de_number(rel, 1)}-faches des Üblichen – {word}")


def _vwap_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    price, vwap, atr = _num(row, "price"), _num(row, "vwap"), _num(row, "atr_slow")
    if not _finite(price, vwap, atr) or atr <= 0.0:
        return None
    distance = (price - vwap) / atr
    detail = f"{de_number(distance, 2, signed=True)} ATR, VWAP {de_price(vwap)}"
    if distance > _FLAT_ATR:
        return _reading("vwap", "Lage zum VWAP", f"über dem VWAP ({detail})", "bullish")
    if distance < -_FLAT_ATR:
        return _reading("vwap", "Lage zum VWAP", f"unter dem VWAP ({detail})", "bearish")
    return _reading("vwap", "Lage zum VWAP", f"am VWAP ({detail})")


def _volatility_reading(row: Mapping[str, Any]) -> ReadingOut | None:
    rank = _num(row, "atr_pct_rank")
    if not math.isfinite(rank):
        return None
    word = _level_word(rank, _RANK_HIGH, _RANK_LOW, "hoch", "niedrig")
    detail = f"{rank * 100:.0f}. Perzentil der letzten {_P.atr_rank_sessions} Handelstage"
    return _reading("volatility", "Volatilität", f"{word} ({detail})")


def _benchmark_reading(row: Mapping[str, Any], benchmark_name: str | None) -> ReadingOut | None:
    trend = _bench_trend_word(row)
    ret = _num(row, "bench_ret_60")
    if trend is None and not math.isfinite(ret):
        return None
    parts = [] if trend is None else [f"{_P.slow_minutes}-Min-Trend {trend[0]}"]
    if math.isfinite(ret):
        parts.append(f"{_P.return_window_min} Min {de_number(ret, 2, signed=True)} %")
    label = f"Markt ({benchmark_name})" if benchmark_name else "Markt"
    return _reading("benchmark", label, ", ".join(parts), "neutral" if trend is None else trend[1])


def _news_reading(news_score: float | None, news_count: int) -> ReadingOut:
    if news_score is None or news_count == 0 or not math.isfinite(news_score):
        return _reading("news", "Nachrichtenlage", "keine aktuellen Meldungen")
    detail = f"{de_number(news_score, 2, signed=True)} aus {news_count} Meldung{'en' if news_count != 1 else ''}"
    if news_score > _NEWS_NEUTRAL:
        return _reading("news", "Nachrichtenlage", f"positiv ({detail})", "bullish")
    if news_score < -_NEWS_NEUTRAL:
        return _reading("news", "Nachrichtenlage", f"negativ ({detail})", "bearish")
    return _reading("news", "Nachrichtenlage", f"ausgeglichen ({detail})")


def readings(
    row: Mapping[str, Any], *, news_score: float | None, news_count: int, benchmark_name: str | None = None
) -> list[ReadingOut]:
    """The technical readings in display order; a reading whose inputs are missing is left out (news always
    appears)."""
    candidates = [
        _trend_reading(row, "trend_5m", f"Trend {_P.decision_minutes} Min", "f"),
        _trend_reading(row, "trend_15m", f"Trend {_P.slow_minutes} Min", "s"),
        _daily_reading(row),
        _momentum_reading(row),
        _strength_reading(row),
        _volume_reading(row),
        _vwap_reading(row),
        _volatility_reading(row),
        _benchmark_reading(row, benchmark_name),
        _news_reading(news_score, news_count),
    ]
    return [reading for reading in candidates if reading is not None]


# ------------------------------------------------------------------------------------------------
# Price levels
# ------------------------------------------------------------------------------------------------
# (feature, fixed kind or None = support/resistance by position, German label) in priority order: when two
# levels are closer than _LEVEL_MERGE_ATR, the one listed first is kept.
_LEVEL_SOURCES: tuple[tuple[str, LevelKind | None, str], ...] = (
    ("day_high", "day_high", "Tageshoch"),
    ("day_low", "day_low", "Tagestief"),
    ("prev_day_high", "prev_high", "Vortageshoch"),
    ("prev_day_low", "prev_low", "Vortagestief"),
    ("prev_day_close", "prev_close", "Vortagesschluss"),
    ("vwap", "vwap", "VWAP"),
    ("or_high", None, f"Hoch der Eröffnungsspanne ({_P.opening_range_min} Min)"),
    ("or_low", None, f"Tief der Eröffnungsspanne ({_P.opening_range_min} Min)"),
    ("donchian_upper_s", None, f"{_P.donchian_window}-Balken-Hoch ({_P.slow_minutes} Min)"),
    ("donchian_lower_s", None, f"{_P.donchian_window}-Balken-Tief ({_P.slow_minutes} Min)"),
)


def levels(row: Mapping[str, Any]) -> list[LevelOut]:
    """Support / resistance candidates, highest price first.

    Levels closer than 0.1 ATR (slow ATR, else fast ATR; without either only identical prices) to a level
    of higher priority are dropped. Opening-range and channel levels are 'resistance' at or above the
    current price and 'support' below it (they are left out while the price is unknown).
    """
    price = _num(row, "price")
    atr = next((value for value in (_num(row, "atr_slow"), _num(row, "atr_fast")) if _finite(value) and value > 0), 0.0)
    tolerance = _LEVEL_MERGE_ATR * atr
    kept: list[LevelOut] = []
    for key, kind, label in _LEVEL_SOURCES:
        level = _num(row, key)
        if not math.isfinite(level) or level <= 0.0:
            continue
        if kind is None:
            if not math.isfinite(price):
                continue
            kind = "resistance" if level >= price else "support"
        if any(abs(level - other.price) < tolerance or level == other.price for other in kept):
            continue
        kept.append(LevelOut(price=level, kind=kind, label_de=label))
    return sorted(kept, key=lambda item: -item.price)


# ------------------------------------------------------------------------------------------------
# Analysis payload
# ------------------------------------------------------------------------------------------------
def finite_features(row: Mapping[str, Any]) -> dict[str, float]:
    """The numeric features with a finite value (flags, the session string and ``data_ts`` are left out)."""
    values = {key: _num(row, key) for key in FEATURE_COLUMNS if key != "data_ts"}
    return {key: value for key, value in values.items() if math.isfinite(value)}


def analysis_out(
    symbol: str,
    row: Mapping[str, Any],
    *,
    as_of: int,
    news_score: float | None,
    news_count: int,
    benchmark_name: str | None = None,
    chart_analysis: ChartAnalysisOut | None = None,
    note_de: str = "",
) -> AnalysisOut:
    """The GUI analysis of one symbol from its newest feature row (``as_of`` = time the row was computed)."""
    data_ts = _num(row, "data_ts")
    return AnalysisOut(
        symbol=symbol,
        as_of=as_of,
        data_ts=int(data_ts) if math.isfinite(data_ts) else None,
        available=True,
        note_de=note_de,
        features=dict(finite_features(row)),
        readings=readings(row, news_score=news_score, news_count=news_count, benchmark_name=benchmark_name),
        levels=levels(row),
        news_score=None if news_score is None else round(news_score, 4),
        chart_analysis=chart_analysis,
    )


# ------------------------------------------------------------------------------------------------
# Rationale
# ------------------------------------------------------------------------------------------------
def _trend_detail(row: Mapping[str, Any]) -> str | None:
    trend = _ema_trend(row, "ema_fast_s", "ema_slow_s", "atr_slow")
    return None if trend is None else f"{_P.slow_minutes} Min {trend[0]}"


def _momentum_detail(row: Mapping[str, Any]) -> str | None:
    rsi = _num(row, "rsi_slow")
    return f"RSI {rsi:.0f}" if math.isfinite(rsi) else None


def _volume_detail(row: Mapping[str, Any]) -> str | None:
    rel = _num(row, "rel_volume")
    return f"{de_number(rel, 1)}-faches Volumen" if math.isfinite(rel) else None


def _market_detail(row: Mapping[str, Any]) -> str | None:
    ret = _num(row, "bench_ret_60")
    return f"Index {_P.return_window_min} Min {de_number(ret, 1, signed=True)} %" if math.isfinite(ret) else None


def _volatility_detail(row: Mapping[str, Any]) -> str | None:
    rank = _num(row, "atr_pct_rank")
    return f"ATR-Perzentil {rank * 100:.0f}" if math.isfinite(rank) else None


_COMPONENT_DETAILS: dict[str, Callable[[Mapping[str, Any]], str | None]] = {
    "trend": _trend_detail,
    "momentum": _momentum_detail,
    "volume": _volume_detail,
    "market": _market_detail,
    "volatility": _volatility_detail,
}


def _component_de(name: str, value: float, row: Mapping[str, Any] | None) -> str:
    """'Volumen (0,95; 1,8-faches Volumen)' - the component value plus a concrete reading when known."""
    label = COMPONENT_LABELS_DE.get(name, name)
    detail_of = _COMPONENT_DETAILS.get(name)
    detail = None if row is None or detail_of is None else detail_of(row)
    inner = de_number(value, 2) if detail is None else f"{de_number(value, 2)}; {detail}"
    return f"{label} ({inner})"


def rationale_de(signal: Signal, row: Mapping[str, Any] | None = None) -> str:
    """Two short German sentences: the setup with its two strongest score components, then the weakest one.

    ``row`` (the feature row or ``signal.features``) adds concrete readings to the components. Ties are
    broken by component name, so the text is deterministic.
    """
    head = f"{setup_label_de(signal.setup)} ({direction_de(signal.direction)}) bei {signal.symbol}"
    ranked = sorted(
        ((value, name) for name, value in signal.components.items() if math.isfinite(value)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    if not ranked:
        return f"{head}. Keine Score-Komponenten verfügbar."
    strongest = [_component_de(name, value, row) for value, name in ranked[:2]]
    verb = "sprechen" if len(strongest) == 2 else "spricht"
    weakest_value, weakest_name = ranked[-1]
    return (
        f"{head}: Am stärksten {verb} {' und '.join(strongest)} für den Einstieg. "
        f"Schwächster Faktor: {_component_de(weakest_name, weakest_value, row)}."
    )


# ------------------------------------------------------------------------------------------------
# LLM fact sheets
# ------------------------------------------------------------------------------------------------
def _range_position(row: Mapping[str, Any], price: float) -> dict[str, Any]:
    """Position of the price in the opening range (words) and in today's range (percent)."""
    facts: dict[str, Any] = {}
    or_high, or_low = _num(row, "or_high"), _num(row, "or_low")
    if _finite(price, or_high, or_low) and or_high > or_low:
        if price > or_high:
            facts["Lage zur Eröffnungsspanne"] = "darüber"
        elif price < or_low:
            facts["Lage zur Eröffnungsspanne"] = "darunter"
        else:
            facts["Lage zur Eröffnungsspanne"] = f"innerhalb ({(price - or_low) / (or_high - or_low) * 100:.0f} %)"
    day_high, day_low = _num(row, "day_high"), _num(row, "day_low")
    if _finite(price, day_high, day_low) and day_high > day_low:
        facts["Lage in der Tagesspanne %"] = round((price - day_low) / (day_high - day_low) * 100.0)
    return facts


def review_facts(
    candidate: Candidate,
    *,
    instrument: Instrument,
    minutes_to_close: int | None,
    benchmark_name: str | None,
    news_score: float | None,
) -> dict[str, Any]:
    """Compact fact sheet with readable German keys and rounded values for the LLM trade review.

    Built from the candidate and its feature snapshot only; facts whose inputs are missing are left out.
    'Datenalter Min' is the age of the newest bar at decision time, 'Minuten bis Börsenschluss' the trading
    minutes left in the session at decision time.
    """
    row = candidate.features
    price = candidate.ref_price
    facts: dict[str, Any] = {
        "Setup": setup_label_de(candidate.setup),
        "Richtung": "Long (Kauf)" if candidate.direction == "LONG" else "Short (Leerverkauf)",
        "Zeiteinheit": candidate.timeframe,
        "Kurs": _sig(price),
        "Währung": instrument.currency,
        "ATR": _sig(candidate.atr),
        "Stopp-Abstand %": round(candidate.stop_dist / price * 100.0, 2) if price > 0 else None,
        "Ziel-Abstand %": round(candidate.target_dist / price * 100.0, 2) if price > 0 else None,
        "Chance/Risiko": round(candidate.target_dist / candidate.stop_dist, 2) if candidate.stop_dist > 0 else None,
        "Score": round(candidate.score, 2),
    }
    for label, suffix in ((f"Trend {_P.decision_minutes} Min", "f"), (f"Trend {_P.slow_minutes} Min", "s")):
        trend = _ema_trend(row, f"ema_fast_{suffix}", f"ema_slow_{suffix}", "atr_fast" if suffix == "f" else "atr_slow")
        facts[label] = None if trend is None else trend[0]
    daily = _daily_trend(row)
    facts["Trend Tag"] = None if daily is None else daily[0]
    rounded = {
        f"RSI {_P.slow_minutes} Min": ("rsi_slow", 1),
        f"RSI {_P.decision_minutes} Min": ("rsi_fast", 1),
        f"ADX {_P.slow_minutes} Min": ("adx_slow", 1),
        "Relatives Volumen": ("rel_volume", 2),
        "Eröffnungslücke %": ("gap_pct", 2),
        f"Vergleichsindex {_P.return_window_min} Min %": ("bench_ret_60", 2),
    }
    for label, (key, digits) in rounded.items():
        value = _num(row, key)
        facts[label] = round(value, digits) if math.isfinite(value) else None
    vwap, atr_slow = _num(row, "vwap"), _num(row, "atr_slow")
    if _finite(vwap, atr_slow) and atr_slow > 0:
        facts["Abstand zum VWAP (ATR)"] = round((price - vwap) / atr_slow, 2)
    facts.update(_range_position(row, price))
    bench = _bench_trend_word(row)
    facts["Vergleichsindex"] = benchmark_name
    facts["Trend Vergleichsindex"] = None if bench is None else bench[0]
    facts["Nachrichten-Score"] = None if news_score is None else round(news_score, 2)
    facts["Datenalter Min"] = round((candidate.decision_ts - candidate.data_ts) / 60.0)
    facts["Minuten bis Börsenschluss"] = minutes_to_close
    for label, key in _EVENT_FACTS:
        text = row.get(key)
        if isinstance(text, str) and text:
            facts[label] = text
    for label, key in _DERIVATIVE_FACTS:
        value = _num(row, key)
        if math.isfinite(value):
            facts[label] = round(value) if key == "deriv_liq_60m_usd" else value
    return {key: value for key, value in facts.items() if value is not None}


# Agenda facts of the trade review (label, feature key of the signal snapshot written by the engine).
_EVENT_FACTS: tuple[tuple[str, str], ...] = (("Nächster Termin", "event_next"), ("Letzter Termin", "event_last"))


# ------------------------------------------------------------------------------------------------
# Crypto derivatives (Kraken Futures perpetuals)
# ------------------------------------------------------------------------------------------------
# Feature keys of the derivative metrics in the snapshot of a crypto signal: (key, DerivMetrics field, digits).
DERIVATIVE_FEATURES: tuple[tuple[str, str, int], ...] = (
    ("deriv_funding_pa", "funding_annual_pct", 2),
    ("deriv_funding_pred_pa", "funding_pred_annual_pct", 2),
    ("deriv_funding_z", "funding_z", 2),
    ("deriv_oi_1h_pct", "oi_change_1h_pct", 2),
    ("deriv_oi_24h_pct", "oi_change_24h_pct", 2),
    ("deriv_basis_pct", "basis_pct", 4),
    ("deriv_liq_60m_usd", "liq_60m_usd", 0),
    ("deriv_liq_5m_ratio", "liq_5m_ratio", 1),
)
# Facts of the trade review (label, feature key); [INFO] facts for the model to weigh.
_DERIVATIVE_FACTS: tuple[tuple[str, str], ...] = (
    ("Funding p.a. %", "deriv_funding_pa"),
    ("Funding ggü. 30 Tagen (Sigma)", "deriv_funding_z"),
    ("Open Interest 1 Std %", "deriv_oi_1h_pct"),
    ("Open Interest 24 Std %", "deriv_oi_24h_pct"),
    ("Basis Perpetual %", "deriv_basis_pct"),
    ("Liquidationen 1 Std USD", "deriv_liq_60m_usd"),
)
_FUNDING_UNUSUAL_Z = 2.0


def derivative_features(metrics: DerivMetrics | None) -> dict[str, float]:
    """The derivative metrics as feature-snapshot entries (empty without data or with stale data)."""
    if metrics is None or metrics.stale:
        return {}
    found: dict[str, float] = {}
    for key, field, digits in DERIVATIVE_FEATURES:
        value = getattr(metrics, field)
        if value is not None and math.isfinite(value):
            found[key] = round(float(value), digits)
    return found


def derivative_readings(metrics: DerivMetrics | None, base: str) -> list[ReadingOut]:
    """GUI readings of the derivatives picture ('Funding', 'Open Interest', 'Liquidationen', 'Basis');
    ``base`` is the coin ('BTC'). Empty without data."""
    if metrics is None:
        return []
    stale = " (veraltet)" if metrics.stale else ""
    found: list[ReadingOut] = []
    if metrics.funding_annual_pct is not None:
        text = f"{de_number(metrics.funding_annual_pct, 1, signed=True)} % p.a."
        z = metrics.funding_z
        if z is not None:
            word = "ungewöhnlich hoch" if z >= _FUNDING_UNUSUAL_Z else "ungewöhnlich niedrig" if z <= -_FUNDING_UNUSUAL_Z else "üblich"
            text += f", {word} ({de_number(z, 1, signed=True)}σ ggü. 30 Tagen)"
        found.append(_reading("deriv_funding", "Funding (Perpetual)", text + stale))
    if metrics.open_interest is not None:
        parts = [f"{de_number(metrics.open_interest, 0)} {base}"]
        if metrics.oi_change_1h_pct is not None:
            parts.append(f"1 Std {de_number(metrics.oi_change_1h_pct, 1, signed=True)} %")
        if metrics.oi_change_24h_pct is not None:
            parts.append(f"24 Std {de_number(metrics.oi_change_24h_pct, 1, signed=True)} %")
        found.append(_reading("deriv_open_interest", "Open Interest", ", ".join(parts) + stale))
    if metrics.liq_60m is not None:
        text = f"{de_number(metrics.liq_60m, 2 if metrics.liq_60m < 100 else 0)} {base} in 1 Std"
        if metrics.liq_60m_usd is not None:
            text += f" (≈ {de_number(metrics.liq_60m_usd, 0)} USD)"
        found.append(_reading("deriv_liquidations", "Liquidationen", text))
    if metrics.basis_pct is not None:
        found.append(_reading("deriv_basis", "Basis Perpetual", f"{de_number(metrics.basis_pct, 3, signed=True)} %"))
    return found


def market_note_de(row: Mapping[str, Any], benchmark_name: str | None) -> str:
    """One German line about the benchmark, e.g. 'S&P 500: 15-Min-Trend aufwärts, 60 Min +0,40 %'."""
    if not benchmark_name:
        return "Kein Vergleichsindex hinterlegt."
    reading = _benchmark_reading(row, None)
    if reading is None:
        return f"Keine aktuellen Daten zum Vergleichsindex {benchmark_name}."
    return f"{benchmark_name}: {reading.value_de}"


def chart_facts(row: Mapping[str, Any] | None, last_price: float | None = None) -> dict[str, float]:
    """Compact numbers for an LLM chart analysis: last price, EMAs, recent swing high/low and ATR.

    Without a feature row only the last price is given (if known).
    """
    if row is None:
        return {} if last_price is None or not math.isfinite(last_price) else {"Kurs": _sig(last_price)}
    slow = f"{_P.slow_minutes} Min"
    keys = {
        "Kurs": "price",
        f"EMA{_P.ema_fast} ({slow})": "ema_fast_s",
        f"EMA{_P.ema_slow} ({slow})": "ema_slow_s",
        f"Swing-Hoch ({_P.donchian_window} x {slow})": "donchian_upper_s",
        f"Swing-Tief ({_P.donchian_window} x {slow})": "donchian_lower_s",
        f"ATR ({slow})": "atr_slow",
        f"EMA{_P.ema_daily_fast} (Tag)": "ema_daily_fast",
        f"EMA{_P.ema_daily_slow} (Tag)": "ema_daily_slow",
        "Vortageshoch": "prev_day_high",
        "Vortagestief": "prev_day_low",
    }
    values = {label: _num(row, key) for label, key in keys.items()}
    return {label: _sig(value) for label, value in values.items() if math.isfinite(value)}
