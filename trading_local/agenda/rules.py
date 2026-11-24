"""Pure rules of the agenda: reading published values, judging a surprise, matching headlines to calendar
entries, which instruments an event concerns, and when an event blocks a new entry.

No I/O here; ``service`` and the signal engine call these functions.
"""

import math
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol

from trading_local.models import Instrument

EventKind = Literal["macro", "central_bank", "earnings"]

DAY_S: Final = 86_400
BLOCK_MIN_IMPORTANCE: Final = 3  # macro and central-bank events of this importance block entries
MIN_Z_HISTORY: Final = 6  # earlier surprises of the same event needed for a standardised surprise
_MULTIPLIERS: Final = {"": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_VALUE_RE = re.compile(r"^\s*[<>]?\s*([+-]?\d+(?:\.\d+)?)\s*([%KMBT]?)\s*$", re.IGNORECASE)
# Central-bank decisions and statements among the macro events (ForexFactory and FinancialJuice titles).
_CENTRAL_BANK_RE = re.compile(
    r"(?i)\b(?:federal funds rate|interest rate decision|main refinancing rate|deposit facility rate|"
    r"official bank rate|policy rate|cash rate|overnight rate|monetary policy (?:statement|report|summary)|"
    r"fomc (?:statement|press conference|economic projections)|ecb press conference|boj press conference|"
    r"boe (?:monetary|press)|snb press conference|rate statement|mpc official bank rate votes)\b"
)
# A currency's events also concern instruments quoted in these currencies (Hong Kong trades China).
_SPILLOVER: Final[dict[str, tuple[str, ...]]] = {"HKD": ("CNY",)}


# ------------------------------------------------------------------------------------------------
# Values
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Value:
    """A published number: ``number`` in base units (K/M/B/T multiplied out), its unit and its precision."""

    number: float
    unit: str  # '%' or '' (K/M/B/T are multiplied into ``number``)
    step: float  # one unit of the last published digit, in base units ('0.3%' -> 0.1, '21.5K' -> 100)


def parse_value(text: str | None) -> Value | None:
    """'0.3%', '-2', '103.3', '21k', '23.75K', '1.2B', '<0.10%' -> Value; None for '', '-' and other text."""
    if not text:
        return None
    match = _VALUE_RE.match(text.replace(",", ""))
    if match is None:
        return None
    digits, suffix = match.group(1), match.group(2).upper()
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    multiplier = 1.0 if suffix == "%" else _MULTIPLIERS[suffix]
    return Value(float(digits) * multiplier, "%" if suffix == "%" else "", 10.0**-decimals * multiplier)


def is_central_bank(title: str) -> bool:
    return bool(_CENTRAL_BANK_RE.search(title))


# ------------------------------------------------------------------------------------------------
# Surprise
# ------------------------------------------------------------------------------------------------
SurpriseLabel = Literal["above", "below", "inline", ""]
SURPRISE_LABELS_DE: Final[dict[str, str]] = {
    "above": "über Erwartung",
    "below": "unter Erwartung",
    "inline": "wie erwartet",
    "": "",
}


def surprise(actual: Value | None, forecast: Value | None) -> float | None:
    """Actual minus forecast in base units; None when one of them is missing or the units differ."""
    if actual is None or forecast is None or actual.unit != forecast.unit:
        return None
    return actual.number - forecast.number


def surprise_label(actual: Value | None, forecast: Value | None) -> SurpriseLabel:
    """'inline' when the actual equals the forecast at the forecast's published precision, else the side."""
    diff = surprise(actual, forecast)
    if diff is None or forecast is None:
        return ""
    if abs(diff) < forecast.step / 2 * (1 + 1e-9):
        return "inline"
    return "above" if diff > 0 else "below"


def standardised(value: float | None, earlier: Sequence[float]) -> float | None:
    """A surprise in units of the root mean square of the event's earlier surprises (at least
    ``MIN_Z_HISTORY``); None without enough history or when every earlier surprise was zero."""
    if value is None or len(earlier) < MIN_Z_HISTORY:
        return None
    scale = math.sqrt(statistics.fmean(item * item for item in earlier))
    return value / scale if scale > 0 else None


def earnings_label(surprise_pct: float | None) -> SurpriseLabel:
    """Earnings: within +-2 % of the estimate counts as in line."""
    if surprise_pct is None:
        return ""
    if abs(surprise_pct) <= 2.0:
        return "inline"
    return "above" if surprise_pct > 0 else "below"


# ------------------------------------------------------------------------------------------------
# Matching headline values to calendar entries
# ------------------------------------------------------------------------------------------------
_TITLE_SUBSTITUTIONS: Final = (
    (re.compile(r"\bm\s*/\s*m\b|\bmom\b", re.I), " mm "),
    (re.compile(r"\by\s*/\s*y\b|\byoy\b", re.I), " yy "),
    (re.compile(r"\bq\s*/\s*q\b|\bqoq\b", re.I), " qq "),
)
_STOP_WORDS: Final = frozenset(
    {"the", "of", "s", "a", "and", "us", "uk", "ez", "eu", "de", "jp", "cn", "au", "nz", "ca", "ch", "fr", "it",
     "es", "sa", "nsa", "index", "prelim", "flash", "final", "change", "rate", "german", "french", "italian",
     "spanish", "eurozone", "euro", "japanese", "chinese", "british", "swiss", "canadian", "australian"}
)  # fmt: skip
# abbreviations of the headline writers -> the calendar's words
_SYNONYMS: Final = {
    "mfg": "manufacturing", "manuf": "manufacturing", "svcs": "services", "service": "services",
    "comp": "composite", "wkly": "weekly", "conf": "confidence", "emp": "employment", "prod": "production",
    "infl": "inflation", "unemp": "unemployment",
}  # fmt: skip


def title_tokens(title: str) -> frozenset[str]:
    """Normalised words of an event title ('US CPI MoM' and 'CPI m/m' share {'cpi', 'mm'})."""
    text = title.lower()
    for pattern, replacement in _TITLE_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    words = (_SYNONYMS.get(word, word) for word in re.findall(r"[a-z0-9]+", text))
    return frozenset(word for word in words if word not in _STOP_WORDS)


def title_similarity(first: str, second: str) -> float:
    """Jaccard similarity of the normalised titles (0..1)."""
    a, b = title_tokens(first), title_tokens(second)
    return len(a & b) / len(a | b) if a and b else 0.0


def same_value(first: Value | None, second: Value | None) -> bool:
    return (
        first is not None
        and second is not None
        and first.unit == second.unit
        and math.isclose(first.number, second.number, rel_tol=1e-9, abs_tol=1e-12)
    )


def match_score(title: str, forecast: Value | None, previous: Value | None, candidate_title: str,
                candidate_forecast: Value | None, candidate_previous: Value | None) -> float:
    """How well a published value (title, forecast, previous) fits a calendar entry; 0 when it does not.

    Accepted are a title similarity of at least 0.5, or an equal forecast with a similarity of at least 0.15,
    or equal forecast AND previous values; the score orders several accepted entries.
    """  # fmt: skip
    similarity = title_similarity(title, candidate_title)
    equal_forecast = same_value(forecast, candidate_forecast)
    equal_previous = same_value(previous, candidate_previous)
    if similarity >= 0.5 or (equal_forecast and similarity >= 0.15) or (equal_forecast and equal_previous):
        return similarity + 0.5 * equal_forecast + 0.2 * equal_previous
    return 0.0


# ------------------------------------------------------------------------------------------------
# Which instruments an event concerns, and the entry block
# ------------------------------------------------------------------------------------------------
class EventLike(Protocol):
    """The fields the rules read (``repo_agenda.AgendaEvent`` has them)."""

    kind: str
    currency: str
    symbol: str | None
    event_ts: int
    time_known: bool
    importance: int


def affects(event: EventLike, instrument: Instrument) -> bool:
    """Does the event concern the instrument?

    Earnings: only the company itself. Macro data and central-bank decisions: instruments quoted in the event's
    currency (and the spill-overs in ``_SPILLOVER``), FX pairs containing it, and - for USD events only - crypto,
    which trades on global risk appetite and the dollar whatever its quote currency.
    """
    if event.kind == "earnings":
        return event.symbol == instrument.symbol
    currency = event.currency.upper()
    if not currency:
        return False
    if instrument.asset_class == "crypto":
        return currency == "USD"
    if instrument.asset_class == "fx":
        return currency in instrument.symbol.upper()
    quoted = instrument.currency.upper()
    return quoted == currency or currency in _SPILLOVER.get(quoted, ())


def blocking_event[E: EventLike](
    events: Iterable[E], instrument: Instrument, *, t_issue: int, planned_exit_ts: int, before_s: int
) -> E | None:
    """The first event that blocks a new entry into ``instrument`` at ``t_issue``, or None.

    * Macro data and central-bank decisions of importance ``BLOCK_MIN_IMPORTANCE`` with a known time block
      entries from ``before_s`` seconds before the release until the release.
    * The company's own earnings block an entry when the release falls between the entry and the later of the
      planned exit and ``t_issue + before_s`` (no holding through earnings). A release without a known time
      covers its whole day. Intraday trades are flat before the close, so a release after the close does not
      block them.
    """
    window_end = t_issue + max(0, before_s)
    for event in sorted(events, key=lambda item: item.event_ts):
        if not affects(event, instrument):
            continue
        if event.kind == "earnings":
            start = event.event_ts
            end = start + (0 if event.time_known else DAY_S - 1)
            if start <= max(planned_exit_ts, window_end) and end >= t_issue:
                return event
        elif event.time_known and event.importance >= BLOCK_MIN_IMPORTANCE and t_issue <= event.event_ts <= window_end:
            return event
    return None
