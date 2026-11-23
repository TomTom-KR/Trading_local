"""German formatting helpers for alert mails and chart titles (pure functions, no I/O, no clock).

Conventions
* Numbers: thousands separator '.', decimal comma (``1234.5`` -> ``'1.234,50'``). Non-finite values are
  shown as an en dash. A value that rounds to zero is printed without a sign (never ``'-0,0'``).
* Negative numbers use the ASCII hyphen-minus, which survives every mail client and plain-text search.
* Percent and R values are separated from their unit by a space: ``'58 %'``, ``'+1,9 R'``.
* Times are shown in the user's time zone as ``'20.11.2026 16:05'``.
"""

import math
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo

from trading_local.models import Instrument

MISSING: Final = "–"  # shown for unknown or non-finite values

REGION_LABELS_DE: Final[dict[str, str]] = {"US": "USA", "EU": "Europa", "ASIA": "Asien", "GLOBAL": "Global"}
TIMEFRAME_LABELS_DE: Final[dict[str, str]] = {
    "1m": "1 Min",
    "5m": "5 Min",
    "15m": "15 Min",
    "1h": "1 Std",
    "1d": "1 Tag",
}
MONTHS_DE: Final[tuple[str, ...]] = ("Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez")

_GERMAN_SEPARATORS = str.maketrans({",": ".", ".": ","})
_MAX_DECIMALS = 8
_MINUTES_PER_HOUR = 60
_MINUTES_PER_DAY = 1440


def format_number(value: float, decimals: int = 2) -> str:
    """``value`` in German notation with exactly ``decimals`` decimals (``1234.5, 2`` -> ``'1.234,50'``)."""
    if not math.isfinite(value):
        return MISSING
    rounded = round(value, decimals)
    if rounded == 0:
        rounded = 0.0  # drop the sign of -0.0
    return f"{rounded:,.{decimals}f}".translate(_GERMAN_SEPARATORS)


def format_signed(value: float, decimals: int = 1) -> str:
    """Like ``format_number`` with an explicit '+' for positive values (``1.94`` -> ``'+1,9'``)."""
    text = format_number(value, decimals)
    if text == MISSING or round(value, decimals) <= 0:
        return text
    return f"+{text}"


def price_decimals(price: float) -> int:
    """Number of decimals that shows a price with useful precision, chosen by its magnitude.

    >= 10,000: 0 (indices, Bitcoin); >= 1,000: 1; >= 10: 2 (most shares); >= 1: 4 (FX rates such as 1,0845,
    penny stocks); below 1: four significant digits (at most 8 decimals). 0 and non-finite values: 2.
    """
    magnitude = abs(price)
    if not math.isfinite(magnitude) or magnitude == 0:
        return 2
    if magnitude >= 10_000:
        return 0
    if magnitude >= 1_000:
        return 1
    if magnitude >= 10:
        return 2
    if magnitude >= 1:
        return 4
    return min(_MAX_DECIMALS, 3 - math.floor(math.log10(magnitude)))


def format_price(value: float | None, decimals: int | None = None) -> str:
    """A price in German notation; ``decimals`` defaults to ``price_decimals(value)``. None -> en dash."""
    if value is None:
        return MISSING
    return format_number(value, price_decimals(value) if decimals is None else decimals)


def format_percent(value: float | None, decimals: int = 1, *, signed: bool = False) -> str:
    """A percentage that is already scaled to 0..100 (``58.0`` -> ``'58,0 %'``; ``signed`` adds '+')."""
    if value is None or not math.isfinite(value):
        return MISSING
    number = format_signed(value, decimals) if signed else format_number(value, decimals)
    return f"{number} %"


def format_r(value: float | None, decimals: int = 1) -> str:
    """A result in R multiples with sign (``1.94`` -> ``'+1,9 R'``, ``-1.0`` -> ``'-1,0 R'``)."""
    if value is None or not math.isfinite(value):
        return MISSING
    return f"{format_signed(value, decimals)} R"


def format_ratio(reward_to_risk: float) -> str:
    """Reward-to-risk ratio as 'Chance : Risiko' (``2.0`` -> ``'2,0 : 1'``)."""
    return f"{format_number(reward_to_risk, 1)} : 1"


def local_datetime(ts: int, timezone: str) -> datetime:
    """Epoch seconds as an aware datetime in ``timezone`` (an IANA name)."""
    return datetime.fromtimestamp(ts, ZoneInfo(timezone))


def format_datetime(ts: int | None, timezone: str) -> str:
    """Epoch seconds as ``'20.11.2026 16:05'`` in ``timezone``; None -> en dash."""
    if ts is None:
        return MISSING
    return local_datetime(ts, timezone).strftime("%d.%m.%Y %H:%M")


def format_time(ts: int, timezone: str) -> str:
    """Epoch seconds as ``'16:05'`` in ``timezone``."""
    return local_datetime(ts, timezone).strftime("%H:%M")


def format_duration(minutes: int) -> str:
    """A duration in German: ``45`` -> ``'45 Min.'``, ``390`` -> ``'6 Std. 30 Min.'``, ``2880`` -> ``'2 Tage'``.

    Days are calendar days of 1,440 minutes; remaining minutes below one hour are dropped from day values.
    """
    minutes = max(0, int(minutes))
    if minutes < _MINUTES_PER_HOUR:
        return f"{minutes} Min."
    if minutes < _MINUTES_PER_DAY:
        hours, rest = divmod(minutes, _MINUTES_PER_HOUR)
        return f"{hours} Std." if rest == 0 else f"{hours} Std. {rest} Min."
    days, rest = divmod(minutes, _MINUTES_PER_DAY)
    day_text = "1 Tag" if days == 1 else f"{days} Tage"
    hours = rest // _MINUTES_PER_HOUR
    return day_text if hours == 0 else f"{day_text} {hours} Std."


def price_unit(instrument: Instrument) -> str:
    """Unit of the QUOTED prices of an instrument (prices are shown as the data source quotes them).

    ``'USD'`` for ``price_scale == 1``; London shares quoted in pence (GBP, scale 0.01) give ``'GBp'``;
    any other scale is spelled out, e.g. ``'ZAR × 0,01'``.
    """
    if instrument.price_scale == 1:
        return instrument.currency
    if instrument.currency == "GBP" and instrument.price_scale == 0.01:
        return "GBp"
    return f"{instrument.currency} × {f'{instrument.price_scale:g}'.translate(_GERMAN_SEPARATORS)}"
