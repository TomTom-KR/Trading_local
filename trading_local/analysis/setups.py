"""Deterministic setup detectors: a handful of classic, widely documented trade setups.

Each detector reads ONE feature row (see ``features.FEATURE_COLUMNS``) and decides with plain float
comparisons whether its textbook conditions hold. There is no hidden state and nothing is fitted: every
threshold is a field of ``EngineParams``. A NaN (or missing) value in a feature a detector needs means the
setup does not trigger.

Common guards of the four intraday setups: the bucket started inside a regular session, at least
``no_entry_first_min`` trading minutes have passed since the open (the opening auction noise has settled),
at least ``no_entry_last_min`` remain until the close (the trade has room to work before it is flattened),
and the slow ATR is positive. 24/7 venues have a UTC trading day, so the guards apply around 00:00 UTC.
The caller checks the warm-up (``params.min_bars_slow`` completed slow bars).
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_local.analysis.features import NewsContext, feature_float, news_age_ref_ts
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.models import Direction, Instrument

SETUPS: tuple[str, ...] = ("news_momentum", "breakout", "trend_pullback", "mean_reversion", "daily_trend")


@dataclass(frozen=True, slots=True)
class SetupHit:
    setup: str
    direction: Direction
    timeframe: str  # '5m' | '15m' | '1d'
    intraday: bool  # True: flattened before the session close
    horizon_min: int  # time stop in trading minutes
    atr: float  # ATR the distances are based on (price units)
    stop_dist: float  # entry -> stop (price units, > 0)
    target_dist: float  # entry -> target (price units, > 0)


_num = feature_float


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _flag(row: Mapping[str, Any], key: str) -> bool:
    """True only for a genuine boolean True (NaN or a missing value is False)."""
    value = row.get(key)
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _intraday_guards(row: Mapping[str, Any], params: EngineParams) -> bool:
    since = _num(row, "minutes_since_open")
    to_close = _num(row, "minutes_to_close")
    atr_slow = _num(row, "atr_slow")
    return (
        _flag(row, "in_session")
        and _finite(since, to_close, atr_slow)
        and since >= params.no_entry_first_min
        and to_close >= params.no_entry_last_min
        and atr_slow > 0.0
    )


def _intraday_hit(
    setup: str, direction: Direction, timeframe: str, horizon_min: int, atr: float, stop: float, target: float
) -> SetupHit:
    return SetupHit(setup, direction, timeframe, True, horizon_min, atr, stop, target)


# ------------------------------------------------------------------------------------------------
# Detectors: each returns the hits of ONE setup (at most one per direction)
# ------------------------------------------------------------------------------------------------
def _news_momentum(
    row: Mapping[str, Any], params: EngineParams, news: NewsContext | None, _session_minutes: int
) -> list[SetupHit]:
    """Fresh, relevant news with a clear sentiment and a price reaction in the same direction.

    Rationale: markets digest material news over minutes to hours (post-news drift); trading only when the
    tape confirms the news avoids fading headlines the market ignores. Rules: the newest item qualifies
    (relevance, |sentiment|, age); the bucket closes in the news direction and either moved at least
    ``news_reaction_atr`` slow ATRs versus the previous close or traded at ``news_relvol_min`` x normal
    volume; the price is at most ``news_max_extension_atr`` slow ATRs away from the VWAP (not chased).

    Timing, relative to the end of the bucket (``data_ts``): the news must have been KNOWN (analysed) by
    then - a move that happened before the platform knew the news cannot confirm it - and the item may be at
    most ``news_max_age_min`` minutes old at that time, counted from its first sight
    (``features.news_age_ref_ts``). Instruments whose feed delay is at least ``news_max_age_min`` therefore
    never trigger this setup: their newest bucket cannot show a reaction to news that is still fresh.
    """
    if news is None or news.fresh is None or not _intraday_guards(row, params):
        return []
    fresh = news.fresh
    if fresh.relevance < params.news_min_relevance or abs(fresh.sentiment) < params.news_min_abs_sentiment:
        return []
    bucket_end = _num(row, "data_ts")
    if math.isfinite(bucket_end) and not (
        fresh.analyzed_ts <= bucket_end and bucket_end - news_age_ref_ts(fresh) <= params.news_max_age_min * 60
    ):
        return []
    price, open_f, prev, atr, vwap = (_num(row, key) for key in ("price", "open_f", "prev_price_f", "atr_slow", "vwap"))
    if not _finite(price, open_f, prev, vwap):
        return []
    sign = 1.0 if fresh.sentiment > 0 else -1.0
    rel_volume = _num(row, "rel_volume")
    closes_with_news = (price - open_f) * sign > 0.0
    reaction = (price - prev) * sign >= params.news_reaction_atr * atr or (
        math.isfinite(rel_volume) and rel_volume >= params.news_relvol_min
    )
    not_extended = abs(price - vwap) <= params.news_max_extension_atr * atr
    if not (closes_with_news and reaction and not_extended):
        return []
    stop = params.stop_atr_news * atr
    target = params.reward_risk * stop
    direction: Direction = "LONG" if sign > 0 else "SHORT"
    return [_intraday_hit("news_momentum", direction, "5m", params.horizon_news_min, atr, stop, target)]


def _breakout(
    row: Mapping[str, Any], params: EngineParams, _news: NewsContext | None, _session_minutes: int
) -> list[SetupHit]:
    """First close beyond the 20-bar Donchian channel / opening range, with participation, with the trend.

    Rationale: the Donchian channel breakout (Turtle rules) and the opening-range breakout are the classic
    momentum entries; volume confirmation filters breakouts nobody trades. Rules (LONG; SHORT mirrored):
    level = max(donchian_upper_s, or_high) (or_high ignored while unknown); price > level and the previous
    bucket closed at or below it (the bucket that just closed is the FIRST close beyond the level);
    rel_volume >= breakout_relvol_min - for instruments without volume data (rel_volume NaN: indices, FX,
    zero-volume crypto feeds) the fast ATR must instead be at least ``breakout_atr_ratio_min`` x the slow
    ATR (range expansion as the participation proxy); and the slow trend does not point against the trade.
    """
    if not _intraday_guards(row, params):
        return []
    price, prev, atr = _num(row, "price"), _num(row, "prev_price_f"), _num(row, "atr_slow")
    upper, lower = _num(row, "donchian_upper_s"), _num(row, "donchian_lower_s")
    ema_fast, ema_slow = _num(row, "ema_fast_s"), _num(row, "ema_slow_s")
    if not _finite(price, prev, upper, lower, ema_fast, ema_slow):
        return []
    rel_volume = _num(row, "rel_volume")
    if math.isfinite(rel_volume):
        participation = rel_volume >= params.breakout_relvol_min
    else:
        atr_fast = _num(row, "atr_fast")
        participation = math.isfinite(atr_fast) and atr_fast >= params.breakout_atr_ratio_min * atr
    if not participation:
        return []
    or_high, or_low = _num(row, "or_high"), _num(row, "or_low")
    level_up = max(upper, or_high) if math.isfinite(or_high) else upper
    level_down = min(lower, or_low) if math.isfinite(or_low) else lower
    stop = params.stop_atr_breakout * atr
    target = params.reward_risk * stop
    hits: list[SetupHit] = []
    if price > level_up and prev <= level_up and ema_fast >= ema_slow:
        hits.append(_intraday_hit("breakout", "LONG", "15m", params.horizon_breakout_min, atr, stop, target))
    if price < level_down and prev >= level_down and ema_fast <= ema_slow:
        hits.append(_intraday_hit("breakout", "SHORT", "15m", params.horizon_breakout_min, atr, stop, target))
    return hits


def _trend_pullback(
    row: Mapping[str, Any], params: EngineParams, _news: NewsContext | None, _session_minutes: int
) -> list[SetupHit]:
    """Buy the dip in an established uptrend (sell the rally in a downtrend) when the trend resumes.

    Rationale: entering a trend after a pullback to its moving average gives a better price and a tighter
    stop than chasing; the ADX filter keeps the rule out of sideways markets (Wilder: ADX above ~20 =
    trend). Rules (LONG; SHORT mirrored): EMA20 > EMA50 on the slow timeframe with a rising EMA20 and
    adx_slow >= adx_trend_min; the last buckets dipped into the value zone (pullback_low <= EMA20 +
    pullback_zone_atr x ATR); the current bucket closes above the previous bucket's high and above the
    fast EMA20 (resumption); the price is at most max_extension_atr x ATR above the slow EMA20.
    """
    if not _intraday_guards(row, params):
        return []
    keys = ("price", "atr_slow", "ema_fast_s", "ema_slow_s", "ema_fast_s_slope", "adx_slow", "ema_fast_f")
    price, atr, ema_fast, ema_slow, slope, adx, ema_fast_f = (_num(row, key) for key in keys)
    if not _finite(price, atr, ema_fast, ema_slow, slope, adx, ema_fast_f) or adx < params.adx_trend_min:
        return []
    zone = params.pullback_zone_atr * atr
    extension = params.max_extension_atr * atr
    stop = params.stop_atr_trend * atr
    target = params.reward_risk * stop
    hits: list[SetupHit] = []
    low, prev_high = _num(row, "pullback_low"), _num(row, "prev_high_f")
    if (
        _finite(low, prev_high)
        and ema_fast > ema_slow
        and slope > 0.0
        and low <= ema_fast + zone
        and price > prev_high
        and price > ema_fast_f
        and price - ema_fast <= extension
    ):
        hits.append(_intraday_hit("trend_pullback", "LONG", "15m", params.horizon_trend_min, atr, stop, target))
    high, prev_low = _num(row, "pullback_high"), _num(row, "prev_low_f")
    if (
        _finite(high, prev_low)
        and ema_fast < ema_slow
        and slope < 0.0
        and high >= ema_fast - zone
        and price < prev_low
        and price < ema_fast_f
        and ema_fast - price <= extension
    ):
        hits.append(_intraday_hit("trend_pullback", "SHORT", "15m", params.horizon_trend_min, atr, stop, target))
    return hits


def _mean_reversion(
    row: Mapping[str, Any], params: EngineParams, _news: NewsContext | None, _session_minutes: int
) -> list[SetupHit]:
    """Fade an oversold (overbought) stretch outside the Bollinger band in a range-bound market.

    Rationale: without a trend (low ADX) prices tend to revert to their mean; RSI extremes plus a band
    excursion mark the stretch and a reversal bucket shows that selling (buying) pressure is fading.
    Rules (LONG; SHORT mirrored): adx_slow < adx_range_max; rsi_fast < rsi_oversold; the bucket traded
    below the lower slow band; it closes above its open and above the previous close. The target is the
    band middle, capped at reward_risk x stop; the trade needs at least min_reward_risk_reversion x stop.
    """
    if not _intraday_guards(row, params):
        return []
    keys = ("price", "atr_slow", "adx_slow", "rsi_fast", "open_f", "prev_price_f", "bb_mid_s")
    price, atr, adx, rsi_fast, open_f, prev, mid = (_num(row, key) for key in keys)
    if not _finite(price, atr, adx, rsi_fast, open_f, prev, mid) or adx >= params.adx_range_max:
        return []
    stop = params.stop_atr_reversion * atr
    hits: list[SetupHit] = []
    low, lower = _num(row, "low_f"), _num(row, "bb_lower_s")
    if _finite(low, lower) and rsi_fast < params.rsi_oversold and low < lower and price > open_f and price > prev:
        target = min(mid - price, params.reward_risk * stop)
        if target >= params.min_reward_risk_reversion * stop:
            hits.append(_intraday_hit("mean_reversion", "LONG", "5m", params.horizon_reversion_min, atr, stop, target))
    high, upper = _num(row, "high_f"), _num(row, "bb_upper_s")
    if _finite(high, upper) and rsi_fast > params.rsi_overbought and high > upper and price < open_f and price < prev:
        target = min(price - mid, params.reward_risk * stop)
        if target >= params.min_reward_risk_reversion * stop:
            hits.append(
                _intraday_hit("mean_reversion", "SHORT", "5m", params.horizon_reversion_min, atr, stop, target)
            )
    return hits


def _daily_trend(
    row: Mapping[str, Any], params: EngineParams, _news: NewsContext | None, session_minutes: int
) -> list[SetupHit]:
    """Swing entry in the direction of the daily trend after a pullback, on the break of yesterday's high.

    Rationale: the 50/200-day EMA alignment is the classic long-term trend filter; buying a pullback when
    price takes out the prior day's high is the textbook swing-trading trigger. Rules (LONG; SHORT
    mirrored): prev_day_close > EMA50 > EMA200 (daily, lagged one session); a pullback happened (daily RSI
    between daily_rsi_low and daily_rsi_high, or yesterday's low came within daily_pullback_atr x daily ATR
    of the daily EMA20); the bucket closes above yesterday's high while the previous bucket closed at or
    below it (the first fast close above the level after a close at or below it). Guards: in session and at
    least no_entry_first_min minutes after the open. Not intraday; horizon = horizon_daily_days sessions.
    """
    since = _num(row, "minutes_since_open")
    if not (_flag(row, "in_session") and math.isfinite(since) and since >= params.no_entry_first_min):
        return []
    keys = ("price", "prev_price_f", "prev_day_close", "prev_day_high", "prev_day_low", "atr_daily")
    price, prev, day_close, day_high, day_low, atr = (_num(row, key) for key in keys)
    ema_keys = ("ema_daily_fast", "ema_daily_slow", "ema_daily_pullback")
    ema_fast, ema_slow, ema_pullback = (_num(row, key) for key in ema_keys)
    rsi_daily = _num(row, "rsi_daily")
    if not _finite(price, prev, day_close, day_high, day_low, atr, ema_fast, ema_slow, ema_pullback) or atr <= 0.0:
        return []
    rsi_pullback = math.isfinite(rsi_daily) and params.daily_rsi_low <= rsi_daily <= params.daily_rsi_high
    tolerance = params.daily_pullback_atr * atr
    stop = params.stop_atr_daily * atr
    horizon = params.horizon_daily_days * session_minutes
    hits: list[SetupHit] = []
    if (
        day_close > ema_fast > ema_slow
        and (rsi_pullback or day_low <= ema_pullback + tolerance)
        and price > day_high
        and prev <= day_high
    ):
        hits.append(SetupHit("daily_trend", "LONG", "1d", False, horizon, atr, stop, params.reward_risk * stop))
    if (
        day_close < ema_fast < ema_slow
        and (rsi_pullback or day_high >= ema_pullback - tolerance)
        and price < day_low
        and prev >= day_low
    ):
        hits.append(SetupHit("daily_trend", "SHORT", "1d", False, horizon, atr, stop, params.reward_risk * stop))
    return hits


_DETECTORS: dict[str, Callable[[Mapping[str, Any], EngineParams, NewsContext | None, int], list[SetupHit]]] = {
    "news_momentum": _news_momentum,
    "breakout": _breakout,
    "trend_pullback": _trend_pullback,
    "mean_reversion": _mean_reversion,
    "daily_trend": _daily_trend,
}


def detect(
    row: Mapping[str, Any],
    *,
    instrument: Instrument,
    params: EngineParams = ENGINE_PARAMS,
    news: NewsContext | None = None,
    session_minutes: int | None = None,
) -> list[SetupHit]:
    """All setups that trigger on one feature row, at most ONE hit per direction.

    Setups are evaluated in ``SETUPS`` order and the first hit per direction wins. A setup that would fire
    in both directions at once (impossible with consistent features) is discarded as contradictory.
    ``session_minutes`` is the venue's regular trading minutes per day (daily-trend horizon); None means
    ``params.default_session_minutes``. ``instrument`` identifies the symbol; eligibility (tradable,
    regions, allowed directions) is decided by the caller.
    """
    minutes_per_session = params.default_session_minutes if session_minutes is None else session_minutes
    chosen: dict[str, SetupHit] = {}
    for name in SETUPS:
        hits = _DETECTORS[name](row, params, news, minutes_per_session)
        if len({hit.direction for hit in hits}) > 1:
            continue
        for hit in hits:
            if hit.stop_dist > 0.0 and hit.target_dist > 0.0:
                chosen.setdefault(hit.direction, hit)
    return [chosen[direction] for direction in ("LONG", "SHORT") if direction in chosen]
