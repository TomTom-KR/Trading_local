"""Agenda sources: ForexFactory calendar, FinancialJuice actual values and Yahoo earnings dates."""

from datetime import UTC, datetime

import pytest

from trading_local.agenda import sources
from trading_local.agenda.sources import best_match, earnings_events, parse_actual_headline, parse_forexfactory
from trading_local.models import EarningsDate
from trading_local.testing import SAMPLE_INSTRUMENTS

NOW_TS = 1_795_557_000

# Entries as answered on 2026-11-24 (plus a holiday, an all-day entry and junk).
WEEK = [
    {"title": "CPI m/m", "country": "AUD", "date": "2026-11-24T19:30:00-05:00", "impact": "High", "forecast": "0.4%",
     "previous": "0.8%"},
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.3%"},
    {"title": "CB Consumer Confidence", "country": "USD", "date": "2026-11-24T10:00:00-05:00", "impact": "High",
     "forecast": "92.4", "previous": "91.7"},
    {"title": "Federal Funds Rate", "country": "USD", "date": "2026-12-09T14:00:00-05:00", "impact": "High",
     "forecast": "4.50%", "previous": "4.75%"},
    {"title": "Bank Holiday", "country": "USD", "date": "2026-11-26T00:00:00-05:00", "impact": "Holiday",
     "forecast": "", "previous": ""},
    {"title": "German Prelim CPI m/m", "country": "EUR", "date": "2026-11-27T00:00:00-05:00", "impact": "Medium",
     "forecast": "0.1%", "previous": "0.3%"},
    {"title": "", "country": "USD", "date": "2026-11-24T10:00:00-05:00", "impact": "Low"},
    {"title": "Broken date", "country": "USD", "date": "soon", "impact": "Low"},
    "junk",
]  # fmt: skip


def test_parse_forexfactory() -> None:
    events = {event.title: event for event in parse_forexfactory(WEEK, now_ts=NOW_TS)}
    assert set(events) == {"CPI m/m", "Core PCE Price Index m/m", "CB Consumer Confidence", "Federal Funds Rate",
                           "German Prelim CPI m/m"}  # fmt: skip
    pce = events["Core PCE Price Index m/m"]
    assert pce.uid == "ff:USD:Core PCE Price Index m/m:2026-11-25" and pce.kind == "macro"
    assert pce.event_ts == int(datetime(2026, 11, 25, 13, 30, tzinfo=UTC).timestamp()) and pce.time_known
    assert (pce.importance, pce.unit, pce.forecast, pce.previous) == (3, "%", 0.3, 0.3)
    assert (pce.forecast_text, pce.first_seen_ts, pce.source) == ("0.3%", NOW_TS, "forexfactory")
    assert events["Federal Funds Rate"].kind == "central_bank"
    assert events["CPI m/m"].currency == "AUD" and events["CPI m/m"].uid.endswith(":2026-11-24")
    german = events["German Prelim CPI m/m"]
    assert german.time_known is False and german.importance == 2  # all-day entry at New York midnight
    with pytest.raises(ValueError):
        parse_forexfactory({"error": "rate limited"}, now_ts=NOW_TS)


@pytest.mark.parametrize(
    ("headline", "title", "currency", "actual", "forecast", "previous"),
    [
        ("US CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)",
         "CB Consumer Confidence", "USD", 103.3, 92.4, 91.7),
        ("Richmond Fed Comp. Index Actual 3 (Forecast -2, Previous -2)", "Richmond Fed Comp. Index", "USD", 3, -2, -2),
        ("US ADP Wkly Employment Change Actual 21k (Forecast -, Previous 23.75k, Revised 23k)",
         "ADP Wkly Employment Change", "USD", 21_000, None, 23_750),
        ("US CaseShiller 20 YoY Actual 2.56% (Forecast 2.71%, Previous 2.7%)", "CaseShiller 20 YoY", "USD", 2.56,
         2.71, 2.7),
        ("German Ifo Business Climate Actual 88.1 (Forecast 87.5, Previous 87.3)", "Ifo Business Climate", "EUR",
         88.1, 87.5, 87.3),
        ("Swiss GDP QoQ Actual 0.4% (Expected 0.3%, Previous 0.2%)", "GDP QoQ", "CHF", 0.4, 0.3, 0.2),
    ],
)  # fmt: skip
def test_parse_actual_headline(
    headline: str, title: str, currency: str, actual: float, forecast: float | None, previous: float | None
) -> None:
    value = parse_actual_headline(headline)
    assert value is not None
    assert (value.title, value.currency, value.actual.number) == (title, currency, pytest.approx(actual))
    assert (value.forecast.number if value.forecast else None) == (None if forecast is None else pytest.approx(forecast))
    assert (value.previous.number if value.previous else None) == (None if previous is None else pytest.approx(previous))
    if forecast is None:
        assert value.forecast_text == ""


@pytest.mark.parametrize(
    "headline",
    [
        "BoC's Gov. Macklem: Economy has shown more resilience than we expected.",
        "Abercrombie & Fitch forecasts FY2026 EPS $13.70-$13.90",
        "US Fed Chair Actual remarks (Forecast none, Previous none)",
    ],
)
def test_other_headlines_carry_no_actual_value(headline: str) -> None:
    assert parse_actual_headline(headline) is None


def test_best_match_prefers_the_same_release() -> None:
    week = parse_forexfactory(WEEK, now_ts=NOW_TS)
    for event in week:
        event.id = hash(event.uid)
    value = parse_actual_headline("US CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)")
    assert value is not None
    match = best_match(value, week)
    assert match is not None and match.title == "CB Consumer Confidence"
    elsewhere = parse_actual_headline("UK CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)")
    assert elsewhere is not None and best_match(elsewhere, week) is None  # another currency
    unrelated = parse_actual_headline("US Redbook YoY Actual 11.3% (Forecast -, Previous 9.5%)")
    assert unrelated is not None and best_match(unrelated, week) is None
    created = sources.event_from_headline(unrelated, news_id=42, ts=NOW_TS + 30)
    assert (created.uid, created.importance, created.event_ts, created.actual) == ("fj:42", 1, NOW_TS, 11.3)
    assert created.surprise is None and created.actual_ts == NOW_TS + 30


def test_earnings_events() -> None:
    sap = SAMPLE_INSTRUMENTS[1]
    releases = [
        EarningsDate("SAP.DE", int(datetime(2027, 1, 28, 16, 0, tzinfo=UTC).timestamp()), True, 1.98),
        EarningsDate("SAP.DE", int(datetime(2026, 10, 21, 5, 0, tzinfo=UTC).timestamp()), False, 1.83, 1.89, 3.09),
    ]
    ancient = EarningsDate("SAP.DE", int(datetime(2021, 1, 28, 16, 0, tzinfo=UTC).timestamp()), True, 1.0, 1.1, 10.0)
    upcoming, reported = earnings_events(sap, [*releases, ancient], now_ts=NOW_TS)  # years-old releases dropped
    assert upcoming.uid == "earn:SAP.DE:2027-01-28" and upcoming.kind == "earnings" and upcoming.importance == 3
    assert (upcoming.forecast_text, upcoming.actual, upcoming.actual_ts) == ("1.98", None, None)
    assert reported.time_known is False
    assert reported.event_ts == int(datetime(2026, 10, 20, 22, 0, tzinfo=UTC).timestamp())  # Berlin midnight
    assert (reported.actual, reported.surprise, reported.actual_ts) == (1.89, 3.09, NOW_TS)
    assert reported.uid == "earn:SAP.DE:2026-10-21"
