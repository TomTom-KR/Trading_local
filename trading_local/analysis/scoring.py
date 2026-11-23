"""Transparent composite score of a setup hit with FIXED a-priori weights.

Seven components, each in 0..1 with 0.5 meaning neutral, are combined with the weights of
``EngineParams.weights``; nothing is fitted to data. Every map is smooth and bounded (tanh / clip) with a
scale chosen a priori, never a threshold tuned on history. The scales below are part of the rules
(``params.RULES_REVISION``).
"""

import math
from collections.abc import Mapping
from typing import Any

from trading_local.analysis.features import NewsContext
from trading_local.analysis.features import feature_float as _num
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.analysis.setups import SetupHit
from trading_local.models import Instrument

NEUTRAL = 0.5
_RSI_SCALE = 100.0  # RSI points per unit of the momentum term
_MACD_SCALE = 2.0  # histogram (in slow ATRs) multiplier inside tanh
_RSI_STRETCH_START = 15.0  # mean reversion: RSI distance from 50 where the stretch starts to count ...
_RSI_STRETCH_SPAN = 25.0  # ... and the distance over which it grows to full strength
_VOLUME_SLOPE = 0.25  # component change per unit of relative volume
_BENCH_RETURN_SCALE = 0.5  # percent of benchmark return that counts like one unit of benchmark trend
_MARKET_DAMPING = 2.0  # divisor inside tanh of the market component


def _squash(x: float) -> float:
    """Map a signed strength to 0..1 (0 -> 0.5)."""
    return NEUTRAL + 0.5 * math.tanh(x)


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _mean_or_neutral(terms: list[float]) -> float:
    finite = [term for term in terms if math.isfinite(term)]
    return sum(finite) / len(finite) if finite else NEUTRAL


def _alignment(sign: float, fast: float, slow: float, scale: float) -> float:
    """``_squash(sign * (fast - slow) / scale)``; NaN when an input is missing or the scale is not positive."""
    if not (math.isfinite(fast) and math.isfinite(slow) and math.isfinite(scale)) or scale <= 0.0:
        return math.nan
    return _squash(sign * (fast - slow) / scale)


def _trend(hit: SetupHit, row: Mapping[str, Any], sign: float) -> float:
    """Alignment of the fast, slow and daily trend with the trade; neutral for mean reversion by definition."""
    if hit.setup == "mean_reversion":
        return NEUTRAL
    return _mean_or_neutral(
        [
            _alignment(sign, _num(row, "ema_fast_f"), _num(row, "ema_slow_f"), _num(row, "atr_fast")),
            _alignment(sign, _num(row, "ema_fast_s"), _num(row, "ema_slow_s"), _num(row, "atr_slow")),
            _alignment(sign, _num(row, "prev_day_close"), _num(row, "ema_daily_fast"), _num(row, "atr_daily")),
        ]
    )


def _momentum(hit: SetupHit, row: Mapping[str, Any], sign: float) -> float:
    """Slow RSI and MACD histogram in the trade direction; for mean reversion the depth of the RSI stretch."""
    if hit.setup == "mean_reversion":
        rsi_fast = _num(row, "rsi_fast")
        if not math.isfinite(rsi_fast):
            return NEUTRAL
        return _clip01((abs(rsi_fast - 50.0) - _RSI_STRETCH_START) / _RSI_STRETCH_SPAN)
    rsi_slow, macd_hist = _num(row, "rsi_slow"), _num(row, "macd_hist_s")
    return _mean_or_neutral(
        [
            _clip01(NEUTRAL + sign * (rsi_slow - 50.0) / _RSI_SCALE) if math.isfinite(rsi_slow) else math.nan,
            _squash(sign * macd_hist * _MACD_SCALE) if math.isfinite(macd_hist) else math.nan,
        ]
    )


def _volume(row: Mapping[str, Any]) -> float:
    rel_volume = _num(row, "rel_volume")
    if not math.isfinite(rel_volume):
        return NEUTRAL
    return _clip01(NEUTRAL + _VOLUME_SLOPE * (rel_volume - 1.0))


_LEVELS_ABOVE = ("day_high", "prev_day_high", "donchian_upper_s")
_LEVELS_BELOW = ("day_low", "prev_day_low", "donchian_lower_s")
_BROKEN_LEVEL: dict[tuple[str, int], str] = {
    ("breakout", 1): "donchian_upper_s",
    ("breakout", -1): "donchian_lower_s",
    ("daily_trend", 1): "prev_day_high",
    ("daily_trend", -1): "prev_day_low",
}


def _structure(hit: SetupHit, row: Mapping[str, Any], sign: int) -> float:
    """Room to the nearest opposing level relative to the target distance (1.0 when nothing is in the way).

    LONG: the nearest of day_high, prev_day_high, donchian_upper_s (and bb_upper_s for mean reversion) that
    lies ABOVE the price; SHORT mirrored with the levels below. The session extreme counts only when it was
    set before the current bucket (day_high > high_f): an extreme made by the current bucket itself is no
    resistance. For breakout and daily_trend the level just broken is ignored.
    """
    price = _num(row, "price")
    if not math.isfinite(price) or hit.target_dist <= 0.0:
        return NEUTRAL
    keys = list(_LEVELS_ABOVE if sign > 0 else _LEVELS_BELOW)
    if hit.setup == "mean_reversion":
        keys.append("bb_upper_s" if sign > 0 else "bb_lower_s")
    broken = _BROKEN_LEVEL.get((hit.setup, sign))
    own_extreme = _num(row, "high_f" if sign > 0 else "low_f")
    distances: list[float] = []
    for key in keys:
        level = _num(row, key)
        if key == broken or not math.isfinite(level):
            continue
        if key in ("day_high", "day_low") and not (math.isfinite(own_extreme) and (level - own_extreme) * sign > 0):
            continue
        distance = (level - price) * sign
        if distance > 0.0:
            distances.append(distance)
    if not distances:
        return 1.0
    return _clip01(min(distances) / hit.target_dist)


def _market(row: Mapping[str, Any], sign: float) -> float:
    """Benchmark trend and 60-minute return in the trade direction (neutral without benchmark data)."""
    trend, ret = _num(row, "bench_trend"), _num(row, "bench_ret_60")
    if not (math.isfinite(trend) or math.isfinite(ret)):
        return NEUTRAL
    strength = (trend if math.isfinite(trend) else 0.0) + (ret / _BENCH_RETURN_SCALE if math.isfinite(ret) else 0.0)
    return _squash(sign * strength / _MARKET_DAMPING)


def _news(news: NewsContext | None, sign: float) -> float:
    if news is None or not math.isfinite(news.score):
        return NEUTRAL
    return _clip01(NEUTRAL + 0.5 * sign * news.score)


def _volatility(row: Mapping[str, Any], params: EngineParams) -> float:
    """1.0 in the middle of the trailing volatility distribution, falling to the floor at both extremes."""
    rank = _num(row, "atr_pct_rank")
    if not math.isfinite(rank):
        return NEUTRAL
    return max(params.volatility_floor, 1.0 - 2.0 * abs(_clip01(rank) - 0.5))


def score_hit(
    hit: SetupHit,
    row: Mapping[str, Any],
    *,
    instrument: Instrument,
    params: EngineParams = ENGINE_PARAMS,
    news: NewsContext | None = None,
) -> tuple[float, dict[str, float]]:
    """(score 0..1, components by name 0..1) of a setup hit on its feature row.

    score = sum(weight_i * component_i) with the fixed weights of ``params``. The score and every component
    are rounded to 4 decimals (deterministic storage); the score is computed from the unrounded components.
    ``instrument`` identifies the symbol; the components depend on the row only.
    """
    sign = 1 if hit.direction == "LONG" else -1
    components = {
        "trend": _trend(hit, row, sign),
        "momentum": _momentum(hit, row, sign),
        "volume": _volume(row),
        "structure": _structure(hit, row, sign),
        "market": _market(row, sign),
        "news": _news(news, sign),
        "volatility": _volatility(row, params),
    }
    score = sum(weight * components[name] for name, weight in params.weights)
    return round(_clip01(score), 4), {name: round(value, 4) for name, value in components.items()}
