"""The one and only parameter set of the signal engine.

Every number that shapes a decision lives in ``EngineParams``: indicator periods, setup thresholds, risk
distances, score weights, the calibration prior and the tracking rules. The values are fixed textbook
defaults chosen BEFORE looking at any stored history; they are never tuned on it. The hash of all values
(``EngineParams.version``) is stored with every signal, so results produced under different parameters are
never mixed up.

The formulas themselves (which feature is compared with which) live in code. ``RULES_REVISION`` is part
of the version string and must be bumped by hand whenever a rule changes in code.
"""

import hashlib
from dataclasses import dataclass, fields
from typing import Any

# e2: payoff-aware expected R, news timing relative to the bucket end (age from first sight), daily
#     features only with the previous session's daily bar, tracking on settled bars only.
RULES_REVISION = "e2"

DEFAULT_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("trend", 0.22),
    ("momentum", 0.14),
    ("volume", 0.14),
    ("structure", 0.14),
    ("market", 0.12),
    ("news", 0.16),
    ("volatility", 0.08),
)


@dataclass(frozen=True, slots=True)
class EngineParams:
    """Frozen engine parameters (minutes are trading minutes unless stated otherwise; ATR multiples are
    in units of the ATR named by the setup)."""

    # --- timeframes ------------------------------------------------------------------------------
    decision_minutes: int = 5  # decision cadence = fast timeframe
    slow_minutes: int = 15  # trend timeframe; a multiple of decision_minutes

    # --- indicator periods -----------------------------------------------------------------------
    ema_fast: int = 20
    ema_slow: int = 50
    ema_daily_fast: int = 50
    ema_daily_slow: int = 200
    ema_daily_pullback: int = 20
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_std: float = 2.0
    donchian_window: int = 20
    opening_range_min: int = 30
    slope_bars: int = 3  # slow bars over which the slope of the fast slow-timeframe EMA is measured
    pullback_buckets: int = 4  # fast buckets (incl. the current one) searched for the pullback extreme
    return_window_min: int = 60  # wall-clock minutes of ret_60 / bench_ret_60
    relvol_sessions: int = 10  # previous sessions for the time-of-day volume median
    relvol_min_sessions: int = 3  # fewer prior sessions with that bucket: fall back to recent buckets
    relvol_fallback_buckets: int = 20
    atr_rank_sessions: int = 10  # trailing sessions of slow buckets for the volatility regime (< engine.WARMUP_SESSIONS)
    atr_rank_min_bars: int = 20  # minimum slow ATR values in that window before a rank is given

    # --- session guards --------------------------------------------------------------------------
    no_entry_first_min: int = 15
    no_entry_last_min: int = 30
    flatten_before_close_min: int = 5

    # --- setup thresholds ------------------------------------------------------------------------
    adx_trend_min: float = 18.0
    adx_range_max: float = 18.0
    pullback_zone_atr: float = 0.25
    max_extension_atr: float = 1.0
    breakout_relvol_min: float = 1.5
    breakout_atr_ratio_min: float = 0.5  # no volume data: atr_fast >= ratio * atr_slow instead of rel_volume
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0
    news_min_relevance: int = 4
    news_min_abs_sentiment: float = 0.5
    news_max_age_min: int = 20  # wall-clock minutes
    news_reaction_atr: float = 0.3
    news_relvol_min: float = 1.3
    news_max_extension_atr: float = 2.0
    daily_rsi_low: float = 40.0
    daily_rsi_high: float = 60.0
    daily_pullback_atr: float = 0.5

    # --- risk distances and holding times --------------------------------------------------------
    stop_atr_trend: float = 1.5
    stop_atr_breakout: float = 1.2
    stop_atr_reversion: float = 1.0
    stop_atr_news: float = 1.5
    stop_atr_daily: float = 2.0
    reward_risk: float = 2.0
    min_reward_risk_reversion: float = 1.2
    horizon_trend_min: int = 240
    horizon_breakout_min: int = 180
    horizon_reversion_min: int = 120
    horizon_news_min: int = 240
    horizon_daily_days: int = 5
    default_session_minutes: int = 390  # used for the daily horizon when the caller knows no session length

    # --- composite score -------------------------------------------------------------------------
    weights: tuple[tuple[str, float], ...] = DEFAULT_WEIGHTS  # pairs keep the dataclass hashable
    volatility_floor: float = 0.2

    # --- calibration (score -> win probability) --------------------------------------------------
    prior_slope: float = 0.40
    prior_strength: float = 40.0  # pseudo-observations of the prior per bin
    calibration_bins: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 1.0000001)
    probability_min: float = 0.02
    probability_max: float = 0.98

    # --- tracking --------------------------------------------------------------------------------
    max_entry_drift: float = 0.5  # in units of the stop distance
    entry_grace_min: int = 10  # wall-clock minutes after issue (plus the feed delay) to get an entry

    # --- news ------------------------------------------------------------------------------------
    news_half_life_min: int = 120  # wall-clock minutes
    news_window_min: int = 360  # wall-clock minutes
    news_max_ids: int = 5

    # --- warm-up ---------------------------------------------------------------------------------
    min_bars_slow: int = 60  # completed slow-timeframe bars required before any signal

    def __post_init__(self) -> None:
        names = [name for name, _ in self.weights]
        if len(set(names)) != len(names) or any(weight < 0 for _, weight in self.weights):
            raise ValueError(f"weights need unique names and non-negative values, got {self.weights}")
        if abs(sum(weight for _, weight in self.weights) - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {sum(weight for _, weight in self.weights)}")
        edges = self.calibration_bins
        if len(edges) < 2 or any(lo >= hi for lo, hi in zip(edges, edges[1:])):
            raise ValueError(f"calibration_bins must be at least two ascending edges, got {edges}")
        if self.decision_minutes < 1 or self.slow_minutes % self.decision_minutes != 0:
            raise ValueError("slow_minutes must be a positive multiple of decision_minutes")

    @property
    def weight_map(self) -> dict[str, float]:
        """Score weights by component name."""
        return dict(self.weights)

    @property
    def version(self) -> str:
        """Engine version tag: rules revision plus the first 8 hex digits of the SHA-1 of all field values.

        ``repr`` of ints, floats, strings and tuples is identical in every process, so the tag is stable.
        """
        values: tuple[Any, ...] = tuple(getattr(self, spec.name) for spec in fields(self))
        digest = hashlib.sha1(repr(values).encode("utf-8")).hexdigest()
        return f"{RULES_REVISION}-{digest[:8]}"


ENGINE_PARAMS = EngineParams()
