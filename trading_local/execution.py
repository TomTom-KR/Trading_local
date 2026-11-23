"""Order-execution rules shared by the signal tracker, the paper broker and the validation replay.

Keeping these rules in ONE place guarantees that a tracked signal, the AI portfolio, the user's portfolio
and a validation run are all filled by exactly the same, deliberately pessimistic logic:

* Entry: a market order placed at time T fills at the OPEN of the first 1-minute bar whose open time is
  >= T (never at a price that was already known when the decision was made), plus slippage.
* Exit on a bar, checked in this order:
    1. time stop reached            -> market exit at the bar's open
    2. session-close flattening     -> market exit at the bar's open
    3. bar opens beyond the stop    -> market exit at the open (worse than the stop: gap risk is real)
    4. bar opens beyond the target  -> limit fill at the open (better than the target)
    5. stop touched inside the bar  -> market exit at the stop price
    6. target touched inside the bar-> limit fill at the target price
  When a bar touches both stop and target, the STOP wins (the intrabar path is unknown).
* Slippage (``CostParams``) is charged on every market fill, against the trader. Limit fills at the
  target carry no slippage.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from trading_local.clock import from_ts, to_ts
from trading_local.config import CostParams
from trading_local.contracts import MarketCalendar
from trading_local.models import Direction, Instrument

ExitReason = Literal["stop", "target", "time_stop", "session_close"]


@dataclass(frozen=True, slots=True)
class Exit:
    price: float  # quoted price before slippage
    reason: ExitReason
    market: bool  # True: market order (slippage applies); False: resting limit order filled at its price


def entry_fill_price(costs: CostParams, instrument: Instrument, side: Direction, bar_open: float) -> float:
    """Fill price of a market order that OPENS a position at a bar's open (long buys, short sells)."""
    return costs.fill_price(instrument, bar_open, buying=side == "LONG")


def exit_fill_price(costs: CostParams, instrument: Instrument, side: Direction, exit_: Exit) -> float:
    """Fill price of the order that CLOSES a position (long sells, short buys back)."""
    if not exit_.market:
        return exit_.price
    return costs.fill_price(instrument, exit_.price, buying=side == "SHORT")


def drift_exceeded(ref_price: float, bar_open: float, max_drift: float | None) -> bool:
    """True if the market moved more than ``max_drift`` (price units) between decision and fill."""
    return max_drift is not None and abs(bar_open - ref_price) > max_drift


def exit_on_bar(
    *,
    side: Direction,
    bar_ts: int,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    stop: float | None,
    target: float | None,
    time_stop_ts: int | None = None,
    flatten_ts: int | None = None,
) -> Exit | None:
    """Decide whether an open position leaves the market on this 1-minute bar (see module docstring)."""
    if time_stop_ts is not None and bar_ts >= time_stop_ts:
        return Exit(bar_open, "time_stop", True)
    if flatten_ts is not None and bar_ts >= flatten_ts:
        return Exit(bar_open, "session_close", True)
    if side == "LONG":
        if stop is not None and bar_open <= stop:
            return Exit(bar_open, "stop", True)
        if target is not None and bar_open >= target:
            return Exit(bar_open, "target", False)
        if stop is not None and bar_low <= stop:
            return Exit(stop, "stop", True)
        if target is not None and bar_high >= target:
            return Exit(target, "target", False)
    else:
        if stop is not None and bar_open >= stop:
            return Exit(bar_open, "stop", True)
        if target is not None and bar_open <= target:
            return Exit(bar_open, "target", False)
        if stop is not None and bar_high >= stop:
            return Exit(stop, "stop", True)
        if target is not None and bar_low <= target:
            return Exit(target, "target", False)
    return None


def result_r(side: Direction, entry_price: float, exit_price: float, stop_dist: float) -> float:
    """Realised result in R (multiples of the initial risk). Both prices already include slippage."""
    if stop_dist <= 0:
        return 0.0
    sign = 1.0 if side == "LONG" else -1.0
    return (exit_price - entry_price) * sign / stop_dist


def excursion_r(
    side: Direction, entry_price: float, bar_high: float, bar_low: float, stop_dist: float
) -> tuple[float, float]:
    """(favourable, adverse) excursion of one bar in R; both are >= 0 only if the bar reached that side."""
    if stop_dist <= 0:
        return 0.0, 0.0
    if side == "LONG":
        return (bar_high - entry_price) / stop_dist, (entry_price - bar_low) / stop_dist
    return (entry_price - bar_low) / stop_dist, (bar_high - entry_price) / stop_dist


def session_flatten_ts(
    calendar: MarketCalendar, exchange: str, entry: datetime | int, minutes_before_close: int
) -> int | None:
    """Time at which an INTRADAY position opened at ``entry`` must be flattened: ``minutes_before_close``
    before the end of the last regular trading segment of the entry's trading day.

    24/7 venues have a UTC trading day (see ``MarketCalendar``), so intraday crypto positions are flattened
    shortly before 00:00 UTC - also when the calendar reports their trading time as one segment spanning
    several days: segments are cut at every UTC midnight and each piece is attributed to the trading date
    of its start. ``entry`` may be a datetime or epoch seconds (Python or numpy integer). Returns None when
    no close can be determined within the next 36 hours; such positions are only limited by their time stop.
    """
    start = entry if isinstance(entry, datetime) else from_ts(int(entry))
    horizon = start + timedelta(hours=36)
    day = calendar.session_date(exchange, start)
    last_end: datetime | None = None
    for segment in calendar.segments(exchange, start, horizon):
        for piece_start, piece_end in _split_at_utc_midnight(segment.start, segment.end):
            if calendar.session_date(exchange, piece_start) != day:
                return None if last_end is None else to_ts(last_end) - minutes_before_close * 60
            last_end = piece_end
    if last_end is None or last_end >= horizon:
        return None
    return to_ts(last_end) - minutes_before_close * 60


def _split_at_utc_midnight(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """``[start, end)`` cut into consecutive pieces that do not cross a UTC midnight."""
    pieces: list[tuple[datetime, datetime]] = []
    while start < end:
        midnight = from_ts((to_ts(start) // 86_400 + 1) * 86_400)
        pieces.append((start, min(end, midnight)))
        start = min(end, midnight)
    return pieces
