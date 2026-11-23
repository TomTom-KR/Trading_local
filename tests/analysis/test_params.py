"""EngineParams: textbook defaults, validation and a stable version hash."""

import dataclasses
import re
import subprocess
import sys

import pytest

from trading_local.analysis.params import ENGINE_PARAMS, RULES_REVISION, EngineParams
from trading_local.config import PROJECT_ROOT


def test_textbook_defaults() -> None:
    p = ENGINE_PARAMS
    assert (p.decision_minutes, p.slow_minutes) == (5, 15)
    assert (p.ema_fast, p.ema_slow, p.ema_daily_fast, p.ema_daily_slow, p.ema_daily_pullback) == (20, 50, 50, 200, 20)
    assert (p.rsi_period, p.atr_period, p.adx_period, p.bb_window, p.bb_std, p.donchian_window) == (14, 14, 14, 20, 2.0, 20)
    assert (p.opening_range_min, p.relvol_sessions, p.min_bars_slow) == (30, 10, 60)
    assert (p.no_entry_first_min, p.no_entry_last_min, p.flatten_before_close_min) == (15, 30, 5)
    assert (p.adx_trend_min, p.adx_range_max, p.pullback_zone_atr, p.max_extension_atr) == (18.0, 18.0, 0.25, 1.0)
    assert (p.breakout_relvol_min, p.rsi_oversold, p.rsi_overbought) == (1.5, 25.0, 75.0)
    assert (p.news_min_relevance, p.news_min_abs_sentiment, p.news_max_age_min) == (4, 0.5, 20)
    assert (p.news_reaction_atr, p.news_max_extension_atr) == (0.3, 2.0)
    assert (p.stop_atr_trend, p.stop_atr_breakout, p.stop_atr_reversion, p.stop_atr_news, p.stop_atr_daily) == (
        1.5,
        1.2,
        1.0,
        1.5,
        2.0,
    )
    assert (p.reward_risk, p.min_reward_risk_reversion) == (2.0, 1.2)
    assert (p.horizon_trend_min, p.horizon_breakout_min, p.horizon_reversion_min, p.horizon_news_min) == (
        240,
        180,
        120,
        240,
    )
    assert p.horizon_daily_days == 5
    assert (p.prior_slope, p.prior_strength) == (0.40, 40.0)
    assert p.calibration_bins == (0.0, 0.5, 0.6, 0.7, 0.8, 1.0000001)
    assert (p.max_entry_drift, p.entry_grace_min, p.news_half_life_min, p.news_window_min) == (0.5, 10, 120, 360)
    assert p.weight_map == {
        "trend": 0.22,
        "momentum": 0.14,
        "volume": 0.14,
        "structure": 0.14,
        "market": 0.12,
        "news": 0.16,
        "volatility": 0.08,
    }


def test_weights_sum_to_one_and_params_are_hashable_and_frozen() -> None:
    assert sum(ENGINE_PARAMS.weight_map.values()) == pytest.approx(1.0, abs=1e-12)
    assert hash(ENGINE_PARAMS) == hash(EngineParams())
    with pytest.raises(dataclasses.FrozenInstanceError):
        ENGINE_PARAMS.ema_fast = 10  # type: ignore[misc]


def test_version_format_and_determinism() -> None:
    version = ENGINE_PARAMS.version
    assert re.fullmatch(rf"{RULES_REVISION}-[0-9a-f]{{8}}", version)
    assert version.startswith("e2-")
    assert EngineParams().version == version


def test_version_is_stable_across_processes() -> None:
    code = "from trading_local.analysis.params import ENGINE_PARAMS; print(ENGINE_PARAMS.version)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip() == ENGINE_PARAMS.version


def test_every_field_changes_the_version() -> None:
    changed = {
        dataclasses.replace(ENGINE_PARAMS, ema_fast=21).version,
        dataclasses.replace(ENGINE_PARAMS, reward_risk=2.5).version,
        dataclasses.replace(ENGINE_PARAMS, min_bars_slow=61).version,
        dataclasses.replace(ENGINE_PARAMS, calibration_bins=(0.0, 0.5, 1.0000001)).version,
    }
    assert ENGINE_PARAMS.version not in changed
    assert len(changed) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"weights": (("trend", 0.5), ("news", 0.4))},
        {"weights": (("trend", 0.5), ("trend", 0.5))},
        {"weights": (("trend", 1.2), ("news", -0.2))},
        {"calibration_bins": (0.0, 0.7, 0.6, 1.0)},
        {"calibration_bins": (0.5,)},
        {"slow_minutes": 7},
        {"decision_minutes": 0},
    ],
)
def test_invalid_parameters_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EngineParams(**overrides)  # type: ignore[arg-type]
