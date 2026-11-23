"""Position sizing for the paper broker: pure functions without I/O.

Units used throughout
* ``price``        quoted price in the units of the data source (London shares in pence)
* ``price_scale``  quoted price * price_scale = price in the instrument currency (0.01 for pence)
* ``fx_rate``      units of the base currency per ONE unit of the instrument currency
* money amounts (equity, buying power, notional) are in the portfolio base currency

Lot sizes: equities trade in whole units, indices in steps of 0.01 and crypto in steps of 0.000001.
FX pairs and futures are context symbols and cannot be traded in the simulation.
"""

import math

from trading_local.models import Instrument

# Smallest tradable quantity per asset class and the number of decimals that represents it exactly.
_LOTS: dict[str, tuple[float, int]] = {
    "equity": (1.0, 0),
    "index": (0.01, 2),
    "crypto": (0.000_001, 6),
}
# Quotients are rounded to this many decimals before flooring: 0.03 / 0.01 is 2.9999999999999996 in binary
# floating point and must count as three lots, while 2.9999996 shares must still be two.
_QUOTIENT_DECIMALS = 9


def lot_size(instrument: Instrument) -> float | None:
    """Smallest tradable quantity of the instrument; None for asset classes the simulation does not trade."""
    lot = _LOTS.get(instrument.asset_class)
    return None if lot is None else lot[0]


def is_tradable_class(instrument: Instrument) -> bool:
    """True for asset classes the simulation trades (equities, indices, crypto)."""
    return instrument.asset_class in _LOTS


def round_qty(instrument: Instrument, qty: float) -> float:
    """Floor ``qty`` to a whole number of lots (never rounds up, so every cap stays respected).

    Returns 0.0 for non-positive or non-finite quantities and for asset classes without a lot size.
    """
    lot = _LOTS.get(instrument.asset_class)
    if lot is None or not math.isfinite(qty) or qty <= 0:
        return 0.0
    size, decimals = lot
    units = math.floor(round(qty / size, _QUOTIENT_DECIMALS))
    return round(units * size, decimals)


def value_base(qty: float, price: float, price_scale: float, fx_rate: float) -> float:
    """Market value in base currency of ``qty`` units at the quoted ``price`` (always >= 0 for qty >= 0)."""
    return qty * price * price_scale * fx_rate


def qty_from_notional(instrument: Instrument, notional_base: float, price: float, fx_rate: float) -> float:
    """Number of units (floored to the lot) that ``notional_base`` buys at the quoted ``price``.

    Returns 0.0 when the amount does not reach one lot or any input is not a positive finite number.
    """
    unit_value = value_base(1.0, price, instrument.price_scale, fx_rate)
    if not (_positive(notional_base) and _positive(unit_value)):
        return 0.0
    return round_qty(instrument, notional_base / unit_value)


def risk_position_size(
    equity: float,
    risk_pct: float,
    stop_dist: float,
    price: float,
    price_scale: float,
    fx_rate: float,
    max_position_pct: float,
    buying_power: float,
) -> float:
    """Unrounded quantity that risks ``risk_pct`` percent of ``equity`` between entry and stop, capped.

    * risk:     equity * risk_pct / 100 / (stop_dist * price_scale * fx_rate)
    * cap 1:    position value <= equity * max_position_pct / 100
    * cap 2:    position value <= buying_power
    ``stop_dist`` is in quoted price units. The result is the smallest of the three and never negative;
    0.0 if any input is invalid (non-positive equity, stop distance, price, scale or rate). Round the result
    with ``round_qty`` - flooring keeps every cap satisfied.
    """
    if not all(_positive(value) for value in (equity, risk_pct, stop_dist, price, price_scale, fx_rate)):
        return 0.0
    unit_value = value_base(1.0, price, price_scale, fx_rate)
    by_risk = equity * risk_pct / 100.0 / (stop_dist * price_scale * fx_rate)
    by_position_cap = max(0.0, equity * max_position_pct / 100.0) / unit_value
    by_buying_power = max(0.0, buying_power) / unit_value
    qty = min(by_risk, by_position_cap, by_buying_power)
    return qty if math.isfinite(qty) else 0.0


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0
