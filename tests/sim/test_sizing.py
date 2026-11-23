import math

import pytest

from trading_local.sim import sizing
from trading_local.testing import SAMPLE_INSTRUMENTS
from tests.sim.simkit import FUTURE

BY_SYMBOL = {instrument.symbol: instrument for instrument in [*SAMPLE_INSTRUMENTS, FUTURE]}
SAP, SHELL, DAX, BTC, EURUSD = (BY_SYMBOL[s] for s in ("SAP.DE", "SHEL.L", "^GDAXI", "BTC-USD", "EURUSD=X"))


def test_lot_sizes_per_asset_class() -> None:
    assert sizing.lot_size(SAP) == 1.0
    assert sizing.lot_size(DAX) == 0.01
    assert sizing.lot_size(BTC) == 0.000001
    assert sizing.lot_size(EURUSD) is None
    assert sizing.lot_size(FUTURE) is None
    assert sizing.is_tradable_class(SAP) and not sizing.is_tradable_class(FUTURE)


@pytest.mark.parametrize(
    ("symbol", "qty", "expected"),
    [
        ("SAP.DE", 12.99, 12.0),
        ("SAP.DE", 2.9999996, 2.0),  # flooring never rounds up
        ("SAP.DE", 0.99, 0.0),
        ("^GDAXI", 1.239, 1.23),
        ("^GDAXI", 0.03, 0.03),  # 0.03 / 0.01 is 2.9999999999999996 in binary floating point
        ("BTC-USD", 0.12345678, 0.123456),
        ("BTC-USD", 0.0000009, 0.0),
    ],
)
def test_round_qty_floors_to_the_lot(symbol: str, qty: float, expected: float) -> None:
    assert sizing.round_qty(BY_SYMBOL[symbol], qty) == expected


@pytest.mark.parametrize("qty", [-1.0, 0.0, math.nan, math.inf])
def test_round_qty_rejects_invalid_quantities(qty: float) -> None:
    assert sizing.round_qty(SAP, qty) == 0.0


def test_round_qty_is_zero_for_untradable_classes() -> None:
    assert sizing.round_qty(EURUSD, 1000.0) == 0.0
    assert sizing.round_qty(FUTURE, 3.0) == 0.0


def test_qty_from_notional() -> None:
    assert sizing.qty_from_notional(SAP, 1000.0, 101.0, 1.0) == 9.0
    # 2500 pence = 25 GBP = 31.25 EUR per share
    assert sizing.qty_from_notional(SHELL, 10_000.0, 2500.0, 1.25) == 320.0
    # 50,000 USD = 40,000 EUR per bitcoin
    assert sizing.qty_from_notional(BTC, 1000.0, 50_000.0, 0.8) == 0.025
    assert sizing.qty_from_notional(DAX, 10_000.0, 20_003.0, 1.0) == 0.49
    assert sizing.qty_from_notional(SAP, 50.0, 101.0, 1.0) == 0.0
    assert sizing.qty_from_notional(SAP, 1000.0, 0.0, 1.0) == 0.0
    assert sizing.qty_from_notional(SAP, -5.0, 100.0, 1.0) == 0.0


def test_value_base() -> None:
    assert sizing.value_base(100, 2500.0, 0.01, 1.25) == pytest.approx(3125.0)


def test_risk_position_size_by_risk() -> None:
    # 0.5 % of 100,000 = 500 EUR risk; 10 EUR per share between entry and stop
    assert sizing.risk_position_size(100_000, 0.5, 10.0, 100.0, 1.0, 1.0, 10.0, 1e12) == pytest.approx(50.0)
    # USD share: 10 USD stop = 8 EUR per share
    assert sizing.risk_position_size(100_000, 0.5, 10.0, 100.0, 1.0, 0.8, 10.0, 1e12) == pytest.approx(62.5)


def test_risk_position_size_position_cap() -> None:
    # risk alone would allow 500 shares; 10 % of equity = 10,000 EUR = 100 shares
    assert sizing.risk_position_size(100_000, 0.5, 1.0, 100.0, 1.0, 1.0, 10.0, 1e12) == pytest.approx(100.0)
    # pence: risk 50 pence * 0.01 * 1.25 = 0.625 EUR per share -> 800; cap 10,000 / 31.25 = 320
    assert sizing.risk_position_size(100_000, 0.5, 50.0, 2500.0, 0.01, 1.25, 10.0, 1e12) == pytest.approx(320.0)


def test_risk_position_size_buying_power_cap() -> None:
    assert sizing.risk_position_size(100_000, 0.5, 10.0, 100.0, 1.0, 1.0, 10.0, 3000.0) == pytest.approx(30.0)
    assert sizing.risk_position_size(100_000, 0.5, 10.0, 100.0, 1.0, 1.0, 10.0, -50.0) == 0.0


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 0.5, 10.0, 100.0, 1.0, 1.0, 10.0, 1e6),  # no equity
        (100_000, 0.5, 0.0, 100.0, 1.0, 1.0, 10.0, 1e6),  # no stop distance
        (100_000, 0.5, 10.0, 0.0, 1.0, 1.0, 10.0, 1e6),  # no price
        (100_000, 0.5, 10.0, 100.0, 1.0, math.nan, 10.0, 1e6),  # no rate
    ],
)
def test_risk_position_size_invalid_inputs(args: tuple[float, ...]) -> None:
    assert sizing.risk_position_size(*args) == 0.0
