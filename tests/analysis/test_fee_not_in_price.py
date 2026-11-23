"""Proportional fees cost money/R but never move the price levels (regression of the 23.11.2026 live finding:
Kraken's 0.40 % taker fee was folded into the fill price, which put tight crypto stops at the market price and
stopped the signals out in their entry minute)."""

import pandas as pd
import pytest

from trading_local import execution
from trading_local.analysis import tracking
from trading_local.config import DEFAULT_COSTS, CostParams
from trading_local.models import Instrument, Signal
from trading_local.testing import FakeCalendar

KRAKEN_ETH = Instrument("ETH-EUR", "Ethereum", "crypto", "GLOBAL", "KRAKEN", "EUR", timezone="UTC")
T0 = 1_795_400_040 // 60 * 60
MIN = 60


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    index = pd.Index([T0 + k * MIN for k in range(len(rows))], name="ts", dtype="int64")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index).assign(volume=1.0)


def issued_long(stop_dist: float, target_dist: float) -> Signal:
    signal = Signal(
        created_ts=T0 - MIN, data_ts=T0 - MIN, symbol="ETH-EUR", direction="LONG", setup="breakout", timeframe="15m",
        horizon_min=180, ref_price=1580.0, atr=8.0, stop_dist=stop_dist, target_dist=target_dist, score=0.7,
        confidence=0.4, engine_version="test", intraday=True,
    )
    tracking.issue(signal, issued_ts=T0, instrument=KRAKEN_ETH)
    return signal


def test_kraken_cost_split() -> None:
    assert DEFAULT_COSTS.slippage_bps(KRAKEN_ETH) == DEFAULT_COSTS.slippage_bps_kraken  # price: spread only
    assert DEFAULT_COSTS.fee_bps(KRAKEN_ETH) == DEFAULT_COSTS.fee_bps_kraken  # money: taker fee
    assert DEFAULT_COSTS.flat_fee(KRAKEN_ETH) == 0.0
    assert DEFAULT_COSTS.order_fee(10_000.0, KRAKEN_ETH) == pytest.approx(10_000.0 * DEFAULT_COSTS.fee_bps_kraken / 1e4)
    expected = 2 * (DEFAULT_COSTS.slippage_bps_kraken + DEFAULT_COSTS.fee_bps_kraken) / 1e4 * 100.0 / 1.0
    assert DEFAULT_COSTS.round_trip_cost_r(KRAKEN_ETH, 100.0, 1.0) == pytest.approx(expected)


def test_equities_keep_their_flat_fee_and_no_proportional_fee() -> None:
    nvda = Instrument("NVDA", "NVIDIA", "equity", "US", "XNAS", "USD")
    assert DEFAULT_COSTS.fee_bps(nvda) == 0.0
    assert DEFAULT_COSTS.order_fee(5_000.0, nvda) == DEFAULT_COSTS.fee_flat_base
    assert CostParams(fee_pct=0.1).fee_bps(nvda) == pytest.approx(10.0)


def test_a_tight_crypto_stop_is_not_placed_at_the_market_price() -> None:
    # stop 0.53 % below the entry: with the fee in the fill price the stop sat 0.09 % below the market
    signal = issued_long(stop_dist=8.4, target_dist=16.8)
    quiet = bars([(1579.84, 1579.90, 1578.41, 1578.61), (1578.6, 1580.0, 1577.0, 1579.5)])
    tracking.advance(signal, quiet, instrument=KRAKEN_ETH, calendar=FakeCalendar(), costs=DEFAULT_COSTS)
    entry = 1579.84 * (1 + DEFAULT_COSTS.slippage_bps_kraken / 1e4)
    assert signal.entry_price == pytest.approx(entry)
    assert signal.stop_price == pytest.approx(entry - 8.4)
    assert signal.status == "active"  # a 0.1 % wiggle no longer stops the trade out


def test_tracked_result_deducts_the_proportional_fee_in_r() -> None:
    signal = issued_long(stop_dist=8.0, target_dist=16.0)
    run_up = bars([(1580.0, 1581.0, 1579.5, 1580.5), (1581.0, 1600.0, 1580.0, 1599.0)])
    tracking.advance(signal, run_up, instrument=KRAKEN_ETH, calendar=FakeCalendar(), costs=DEFAULT_COSTS)
    assert signal.status == "target"
    gross = (signal.exit_price - signal.entry_price) / 8.0
    fee_r = DEFAULT_COSTS.fee_bps_kraken / 1e4 * (signal.entry_price + signal.exit_price) / 8.0
    assert signal.outcome_r == pytest.approx(gross - fee_r)
    assert signal.outcome_r == pytest.approx(
        execution.result_r("LONG", signal.entry_price, signal.exit_price, 8.0, fee_bps=DEFAULT_COSTS.fee_bps_kraken)
    )
