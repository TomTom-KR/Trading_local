"""Sources of the agenda (parsing only; the service does the requests).

* ForexFactory weekly calendar ``https://nfs.faireconomy.media/ff_calendar_thisweek.json`` (the site's own
  export, refreshed by them about hourly; polled hourly): macro releases and central-bank decisions of the
  current week (Sunday to Saturday, New York time) with ``impact`` (High / Medium / Low / Holiday), ``forecast``
  and ``previous`` - but no actual value. Checked on 2026-11-24: 74 entries, ``date`` in ISO format with the
  New York offset, ``country`` holds the currency (USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF, CNY). A release at
  New York midnight is an 'all day' / tentative entry without a known time.
* FinancialJuice headlines (already polled once a minute by the news service, stored in ``news``): every
  scheduled release is posted within seconds as ``'<country> <name> Actual <value> (Forecast <value>, Previous
  <value>[, Revised <value>])'``, e.g. 'US CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)'.
  This is the fastest free source of actual values.
* Yahoo Finance earnings dates (``MarketData.earnings``): past and upcoming releases with EPS estimate, the
  reported EPS and the surprise, also for European and Asian companies.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from zoneinfo import ZoneInfo

from trading_local.agenda import rules
from trading_local.agenda.rules import Value, parse_value
from trading_local.db.repo_agenda import AgendaEvent
from trading_local.models import EarningsDate, Instrument

FOREXFACTORY_URL: Final = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
SOURCE_FOREXFACTORY: Final = "forexfactory"
SOURCE_FINANCIALJUICE: Final = "financialjuice"
SOURCE_YAHOO: Final = "yahoo"
FINANCIALJUICE_FEED: Final = "financialjuice"  # feed id in the news registry
EARNINGS_TITLE: Final = "Quartalszahlen"
EARNINGS_HISTORY_DAYS: Final = 400  # past releases kept: the last four to five quarters
_NEW_YORK: Final = ZoneInfo("America/New_York")
_IMPACT: Final = {"high": 3, "medium": 2, "low": 1}


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number_or_none(value: Value | None) -> float | None:
    return None if value is None else value.number


# ------------------------------------------------------------------------------------------------
# ForexFactory
# ------------------------------------------------------------------------------------------------
def parse_forexfactory(body: Any, *, now_ts: int) -> list[AgendaEvent]:
    """The macro and central-bank events of a ForexFactory weekly answer (holidays and unknown impact levels
    are left out). Raises ``ValueError`` when the answer is not a list."""
    if not isinstance(body, list):
        raise ValueError("ForexFactory: the answer is not a list")
    events: list[AgendaEvent] = []
    for raw in body:
        if not isinstance(raw, dict):
            continue
        title, currency, stamp = _text(raw.get("title")), _text(raw.get("country")).upper(), _text(raw.get("date"))
        importance = _IMPACT.get(_text(raw.get("impact")).lower())
        if not title or not currency or not stamp or importance is None:
            continue
        try:
            moment = datetime.fromisoformat(stamp)
        except ValueError:
            continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_NEW_YORK)
        local = moment.astimezone(_NEW_YORK)
        forecast_text, previous_text = _text(raw.get("forecast")), _text(raw.get("previous"))
        forecast, previous = parse_value(forecast_text), parse_value(previous_text)
        unit = (forecast or previous).unit if (forecast or previous) else ""
        events.append(
            AgendaEvent(
                uid=f"ff:{currency}:{title}:{local.date().isoformat()}",
                kind="central_bank" if rules.is_central_bank(title) else "macro",
                source=SOURCE_FOREXFACTORY,
                title=title,
                event_ts=int(moment.timestamp()),
                first_seen_ts=now_ts,
                updated_ts=now_ts,
                currency=currency,
                time_known=not (local.hour == 0 and local.minute == 0),
                importance=importance,
                unit=unit,
                forecast=_number_or_none(forecast),
                previous=_number_or_none(previous),
                forecast_text=forecast_text,
                previous_text=previous_text,
            )
        )
    return events


# ------------------------------------------------------------------------------------------------
# FinancialJuice actual values
# ------------------------------------------------------------------------------------------------
_ACTUAL_RE = re.compile(
    r"^(?P<name>.+?)\s+Actual\s+(?P<actual>[^\s(]+)\s*\(\s*(?:Forecast|Expected|Consensus)\s+(?P<forecast>[^,)]*?)"
    r"\s*,\s*Previous\s+(?P<previous>[^,)]*?)\s*(?:,[^)]*)?\)\s*$",
    re.IGNORECASE,
)
# leading country word of a FinancialJuice headline -> currency
_COUNTRY_WORDS: Final[dict[str, str]] = {
    "US": "USD", "USA": "USD", "UK": "GBP", "GB": "GBP", "BRITISH": "GBP", "EZ": "EUR", "EU": "EUR",
    "EUROZONE": "EUR", "DE": "EUR", "GERMAN": "EUR", "GERMANY": "EUR", "FR": "EUR", "FRENCH": "EUR", "FRANCE": "EUR",
    "IT": "EUR", "ITALIAN": "EUR", "ITALY": "EUR", "ES": "EUR", "SPANISH": "EUR", "SPAIN": "EUR", "JP": "JPY",
    "JAPAN": "JPY", "JAPANESE": "JPY", "CN": "CNY", "CHINA": "CNY", "CHINESE": "CNY", "AU": "AUD", "AUSTRALIA": "AUD",
    "AUSTRALIAN": "AUD", "NZ": "NZD", "CA": "CAD", "CANADA": "CAD", "CANADIAN": "CAD", "CH": "CHF", "SWISS": "CHF",
    "HK": "HKD",
}  # fmt: skip
# institutions that identify the currency when there is no country word ('Richmond Fed Comp. Index')
_INSTITUTION_CURRENCY: Final = (
    (re.compile(r"\b(?:Fed|FOMC|Philly|Chicago PMI|ISM|NFP|ADP)\b", re.I), "USD"),
    (re.compile(r"\b(?:ECB|Ifo|ZEW|HCOB)\b", re.I), "EUR"),
    (re.compile(r"\bBoE\b", re.I), "GBP"),
    (re.compile(r"\bBoJ\b", re.I), "JPY"),
    (re.compile(r"\bSNB\b", re.I), "CHF"),
    (re.compile(r"\bRBA\b", re.I), "AUD"),
    (re.compile(r"\bRBNZ\b", re.I), "NZD"),
    (re.compile(r"\bBoC\b", re.I), "CAD"),
    (re.compile(r"\bPBoC\b", re.I), "CNY"),
)


@dataclass(frozen=True, slots=True)
class PublishedValue:
    """An actual value as posted in a headline."""

    title: str  # the release's name without the country word
    currency: str  # '' when the headline does not reveal it
    actual: Value
    actual_text: str
    forecast: Value | None
    forecast_text: str
    previous: Value | None
    previous_text: str


def parse_actual_headline(headline: str) -> PublishedValue | None:
    """The actual value of a FinancialJuice 'Actual ... (Forecast ..., Previous ...)' headline; None for other
    headlines and for actual values that are not numbers."""
    match = _ACTUAL_RE.match(headline.strip())
    if match is None:
        return None
    actual = parse_value(match["actual"])
    if actual is None:
        return None
    name = match["name"].strip()
    words = name.split(maxsplit=1)
    currency = _COUNTRY_WORDS.get(words[0].upper().strip(".,:"), "")
    if currency and len(words) == 2:
        name = words[1]
    if not currency:
        currency = next((code for pattern, code in _INSTITUTION_CURRENCY if pattern.search(name)), "")
    forecast_text = match["forecast"].strip()
    previous_text = match["previous"].strip()
    return PublishedValue(
        title=name,
        currency=currency,
        actual=actual,
        actual_text=match["actual"].strip(),
        forecast=parse_value(forecast_text),
        forecast_text="" if forecast_text == "-" else forecast_text,
        previous=parse_value(previous_text),
        previous_text="" if previous_text == "-" else previous_text,
    )


def best_match(value: PublishedValue, candidates: Sequence[AgendaEvent]) -> AgendaEvent | None:
    """The calendar entry the published value belongs to (same currency when known), or None."""
    scored: list[tuple[float, AgendaEvent]] = []
    for event in candidates:
        if event.kind == "earnings" or (value.currency and event.currency != value.currency):
            continue
        forecast = parse_value(event.forecast_text)
        previous = parse_value(event.previous_text)
        score = rules.match_score(value.title, value.forecast, value.previous, event.title, forecast, previous)
        if score > 0:
            scored.append((score, event))
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def event_from_headline(
    value: PublishedValue, *, news_id: int, ts: int, released_ts: int | None = None
) -> AgendaEvent:
    """A calendar entry for a published value without a scheduled entry (importance 1: such releases are minor).
    ``ts`` is when the value became known here, ``released_ts`` when it was published (default ``ts``)."""
    surprise = rules.surprise(value.actual, value.forecast)
    return AgendaEvent(
        uid=f"fj:{news_id}",
        kind="central_bank" if rules.is_central_bank(value.title) else "macro",
        source=SOURCE_FINANCIALJUICE,
        title=value.title,
        event_ts=(ts if released_ts is None else released_ts) // 60 * 60,
        first_seen_ts=ts,
        updated_ts=ts,
        currency=value.currency,
        importance=1,
        unit=value.actual.unit,
        forecast=_number_or_none(value.forecast),
        previous=_number_or_none(value.previous),
        actual=value.actual.number,
        forecast_text=value.forecast_text,
        previous_text=value.previous_text,
        actual_text=value.actual_text,
        surprise=surprise,
        actual_ts=ts,
    )


# ------------------------------------------------------------------------------------------------
# Earnings
# ------------------------------------------------------------------------------------------------
def _eps_text(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def earnings_events(instrument: Instrument, releases: Sequence[EarningsDate], *, now_ts: int) -> list[AgendaEvent]:
    """Calendar entries of a company's earnings releases of the last ``EARNINGS_HISTORY_DAYS`` days and the
    upcoming ones (the source may answer with years of history). A release without a known time is dated at
    the start of its local day (venue time zone); its uid is the local date, so a time change keeps the entry."""
    zone = ZoneInfo(instrument.timezone)
    events: list[AgendaEvent] = []
    for release in releases:
        if release.ts < now_ts - EARNINGS_HISTORY_DAYS * 86_400:
            continue
        local = datetime.fromtimestamp(release.ts, zone)
        if not release.time_known:
            day_start = datetime.fromtimestamp(release.ts, _NEW_YORK).date()
            local = datetime(day_start.year, day_start.month, day_start.day, tzinfo=zone)
        reported = release.eps_actual is not None
        events.append(
            AgendaEvent(
                uid=f"earn:{instrument.symbol}:{local.date().isoformat()}",
                kind="earnings",
                source=SOURCE_YAHOO,
                title=EARNINGS_TITLE,
                event_ts=int(local.timestamp()),
                first_seen_ts=now_ts,
                updated_ts=now_ts,
                symbol=instrument.symbol,
                time_known=release.time_known,
                importance=3,
                forecast=release.eps_estimate,
                actual=release.eps_actual,
                forecast_text=_eps_text(release.eps_estimate),
                actual_text=_eps_text(release.eps_actual),
                surprise=release.surprise_pct if reported else None,
                actual_ts=now_ts if reported else None,
            )
        )
    return events
