"""Presenters: pure functions that turn domain objects into API models (no I/O, no clock, no state).

Fields with the same name on the domain object and on the API model are copied by pydantic
(``from_attributes``); the functions here only add what is derived or comes from a second object.
Objects that carry a database id must have been stored: presenting one without an id raises ``ValueError``.
"""

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from email.utils import getaddresses
from typing import Any, Final

from trading_local import clock
from trading_local.dto import (
    ChartAnalysisOut,
    InstrumentOut,
    NewsAnalysisOut,
    NewsImpactOut,
    NewsOut,
    QuoteOut,
    SignalBrief,
    SignalDetailOut,
    SignalOut,
)
from trading_local.models import ChartAnalysis, Instrument, NewsAnalysis, NewsItem, Quote, SessionState, Signal

SETUP_LABELS_DE: Final[dict[str, str]] = {
    "trend_pullback": "Trend-Rücksetzer",
    "breakout": "Ausbruch",
    "mean_reversion": "Gegenbewegung",
    "news_momentum": "Nachrichten-Impuls",
    "daily_trend": "Tagestrend",
}


def setup_label_de(setup: str) -> str:
    """German label of a setup; an unknown setup keeps its raw name."""
    return SETUP_LABELS_DE.get(setup, setup)


def to_ts(dt: datetime | None) -> int | None:
    """Epoch seconds of an optional timezone-aware datetime (for the nullable ``SessionState`` fields)."""
    return None if dt is None else clock.to_ts(dt)


def mask_email(address: str) -> str:
    """Mask e-mail addresses for display: 'max@example.org' becomes 'm***@example.org'.

    Handles what a recipient setting may hold: display names ('Max <max@example.org>') are dropped and
    several addresses separated by ',' or ';' are masked one by one (a stray separator at either end is
    ignored). Text that is not an address is replaced by '***' as a whole, so the input never appears in
    full; nothing but blanks and separators gives ''.
    """
    recipients = address.replace(";", ",").strip(", \t\r\n")
    if not recipients:
        return ""
    addresses = [parsed for _, parsed in getaddresses([recipients]) if parsed]
    return ", ".join(_mask_address(parsed) for parsed in addresses) or "***"


def address_masker(settings_values: Iterable[str]) -> Callable[[str], str]:
    """A function that masks every occurrence of the given e-mail addresses inside a free text.

    ``settings_values`` are recipient/sender settings as ``mask_email`` accepts them (display names, several
    addresses separated by ',' or ';'); values without an address are ignored. Matching ignores case
    ('Max@Example.org' in a server reply is the same mailbox). The masked form ('m***@example.org') never
    contains the address again, so masking twice changes nothing. Without any address the text is returned
    unchanged.
    """
    found: set[str] = set()
    for value in settings_values:
        recipients = value.replace(";", ",")
        found.update(parsed for _, parsed in getaddresses([recipients]) if "@" in parsed)
    if not found:
        return _unchanged
    pattern = re.compile("|".join(re.escape(address) for address in sorted(found, key=len, reverse=True)), re.IGNORECASE)
    return lambda text: pattern.sub(lambda match: _mask_address(match.group(0)), text)


# ------------------------------------------------------------------------------------------------
# Market data
# ------------------------------------------------------------------------------------------------
def quote_out(quote: Quote) -> QuoteOut:
    return QuoteOut.model_validate(quote)


def instrument_out(
    instrument: Instrument,
    *,
    quote: Quote | None = None,
    state: SessionState | None = None,
    open_signal: Signal | None = None,
    exchange_name: str = "",
) -> InstrumentOut:
    """Watchlist entry. Without a session ``state`` the venue is shown as closed."""
    derived: dict[str, Any] = {"exchange_name": exchange_name}
    if state is not None:
        derived.update(phase=state.phase, is_open=state.is_open)
    if quote is not None:
        derived["quote"] = quote_out(quote)
    if open_signal is not None:
        _require_id(open_signal.id, "signal")
        derived["open_signal"] = SignalBrief.model_validate(open_signal)
    return InstrumentOut.model_validate(instrument).model_copy(update=derived)


# ------------------------------------------------------------------------------------------------
# News
# ------------------------------------------------------------------------------------------------
def news_out(
    item: NewsItem, analysis: NewsAnalysis | None, *, source_names: Mapping[str, str] | None = None
) -> NewsOut:
    """News item with its analysis.

    ``source_names`` maps feed ids to display names; a feed that is not listed is shown by its id.
    The affected symbols are ordered by the strength of their sentiment, strongest first.
    """
    _require_id(item.id, "news item")
    derived = {
        "source_name": (source_names or {}).get(item.source, item.source),
        "analysis": None if analysis is None else _news_analysis_out(analysis),
    }
    return NewsOut.model_validate(item).model_copy(update=derived)


def _news_analysis_out(analysis: NewsAnalysis) -> NewsAnalysisOut:
    impacts = sorted(analysis.symbols.items(), key=lambda impact: (-abs(impact[1]), impact[0]))
    return NewsAnalysisOut(
        analyzed_ts=analysis.analyzed_ts,
        relevance=analysis.relevance,
        sentiment=analysis.sentiment,
        confidence=analysis.confidence,
        category=analysis.category,
        horizon=analysis.horizon,
        summary_de=analysis.summary_de,
        symbols=[NewsImpactOut(symbol=symbol, sentiment=sentiment) for symbol, sentiment in impacts],
    )


# ------------------------------------------------------------------------------------------------
# Signals and chart analyses
# ------------------------------------------------------------------------------------------------
def signal_out(signal: Signal, instrument: Instrument | None, *, quote: Quote | None = None) -> SignalOut:
    """Signal for lists and live updates.

    Adds the instrument's name, region, currency and feed delay (left at their defaults when the
    instrument is unknown), the reward-to-risk ratio and the German setup label. With a ``quote``,
    ``last_price`` is its price and ``open_r`` the unrealised result in R of an ACTIVE signal with a
    known entry: ``(last - entry) * sign / stop_dist`` (before costs).
    """
    _require_id(signal.id, "signal")
    return SignalOut.model_validate(signal).model_copy(update=_derived_signal_fields(signal, instrument, quote))


def signal_detail_out(
    signal: Signal, instrument: Instrument | None, *, news: Sequence[NewsOut] = (), quote: Quote | None = None
) -> SignalDetailOut:
    """Like ``signal_out``, plus the score components, the feature snapshot and the news behind the signal."""
    _require_id(signal.id, "signal")
    derived = _derived_signal_fields(signal, instrument, quote)
    derived["news"] = list(news)
    return SignalDetailOut.model_validate(signal).model_copy(update=derived)


def chart_analysis_out(analysis: ChartAnalysis) -> ChartAnalysisOut:
    _require_id(analysis.id, "chart analysis")
    return ChartAnalysisOut.model_validate(analysis)


def _derived_signal_fields(signal: Signal, instrument: Instrument | None, quote: Quote | None) -> dict[str, Any]:
    """The ``SignalOut`` fields that are not attributes of the signal itself."""
    derived: dict[str, Any] = {"setup_label_de": setup_label_de(signal.setup)}
    if instrument is not None:
        derived.update(
            name=instrument.name,
            region=instrument.region,
            currency=instrument.currency,
            delay_min=instrument.delay_min,
        )
    if quote is not None:
        derived.update(last_price=quote.price, open_r=_open_r(signal, quote.price))
    return derived


def _open_r(signal: Signal, last_price: float) -> float | None:
    """Unrealised result in R at ``last_price``; None unless the signal is active and has an entry price."""
    if signal.status != "active" or signal.entry_price is None or signal.stop_dist <= 0:
        return None
    return (last_price - signal.entry_price) * signal.sign / signal.stop_dist


# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
def _require_id(value: int | None, what: str) -> None:
    if value is None:
        raise ValueError(f"{what} has no id: only stored objects can be presented")


def _unchanged(text: str) -> str:
    return text


def _mask_address(address: str) -> str:
    """One bare address with everything but the first character of the local part hidden."""
    local, at, domain = address.rpartition("@")
    if not at or not domain:
        return "***"
    return f"{local[:1]}***@{domain}"
