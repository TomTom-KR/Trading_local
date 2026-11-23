"""Holidays, early closes, the override tables and the estimate beyond the library's data."""

import logging
from datetime import date, timedelta

import pytest

from tests.markets.helpers import BERLIN, HONG_KONG, NEW_YORK, ONE_SECOND, SEOUL, SHANGHAI, TOKYO, local, spans, utc
from trading_local.markets import ExchangeCalendar
from trading_local.markets import calendar as calendar_module


def trades_on(calendar: ExchangeCalendar, key: str, day: date) -> bool:
    """True if the venue has a regular segment on its local calendar date ``day``."""
    start = utc(day.year, day.month, day.day) - timedelta(hours=14)
    return any(
        calendar.session_date(key, segment.start) == day
        for segment in calendar.segments(key, start, start + timedelta(hours=40))
    )


def test_tokyo_labour_thanksgiving_day(calendar: ExchangeCalendar) -> None:
    holiday = local(TOKYO, 2026, 11, 23, 10)
    state = calendar.state("XTKS", holiday)
    assert not calendar.is_open("XTKS", holiday)
    assert (state.phase, state.note) == ("closed", "Feiertag")
    assert state.next_open == local(TOKYO, 2026, 11, 24, 9)
    assert state.since == local(TOKYO, 2026, 11, 20, 15, 30)


def test_new_york_thanksgiving_has_no_session_and_no_extended_hours(calendar: ExchangeCalendar) -> None:
    for hour in (5, 10, 17):
        state = calendar.state("XNYS", local(NEW_YORK, 2026, 11, 26, hour))
        assert (state.phase, state.note) == ("closed", "Feiertag")
        assert state.next_open == local(NEW_YORK, 2026, 11, 27, 9, 30)
    assert not trades_on(calendar, "XNAS", date(2026, 11, 26))


def test_new_york_early_close_day_after_thanksgiving(calendar: ExchangeCalendar) -> None:
    close = local(NEW_YORK, 2026, 11, 27, 13)
    assert close == utc(2026, 11, 27, 18)
    before = calendar.state("XNYS", close - ONE_SECOND)
    assert (before.phase, before.until, before.note) == ("open", close, "Handel (verkürzter Handelstag)")
    assert calendar.minutes_to_close("XNYS", local(NEW_YORK, 2026, 11, 27, 9, 30)) == 210
    after = calendar.state("XNYS", close)
    # after an early close the post window starts at the close and ends at 17:00 New York
    assert (after.phase, after.since, after.until) == ("post", close, local(NEW_YORK, 2026, 11, 27, 17))
    assert calendar.state("XNYS", local(NEW_YORK, 2026, 11, 27, 17)).phase == "closed"


@pytest.mark.parametrize("day", [date(2026, 12, 24), date(2026, 12, 25), date(2026, 12, 31), date(2027, 1, 1)])
def test_xetra_closed_over_christmas_and_new_year(calendar: ExchangeCalendar, day: date) -> None:
    assert not trades_on(calendar, "XETR", day)
    assert not trades_on(calendar, "XFRA", day)
    state = calendar.state("XETR", local(BERLIN, day.year, day.month, day.day, 12))
    assert state.note == "Feiertag"


def test_xetra_early_close_on_december_30(calendar: ExchangeCalendar) -> None:
    assert trades_on(calendar, "XETR", date(2026, 12, 28))
    assert spans(calendar.segments("XETR", utc(2026, 12, 30), utc(2026, 12, 31))) == [
        (local(BERLIN, 2026, 12, 30, 9), local(BERLIN, 2026, 12, 30, 14))
    ]


def test_hong_kong_half_day_has_no_afternoon(calendar: ExchangeCalendar) -> None:
    assert spans(calendar.segments("XHKG", utc(2026, 12, 24), utc(2026, 12, 25))) == [
        (local(HONG_KONG, 2026, 12, 24, 9, 30), local(HONG_KONG, 2026, 12, 24, 12))
    ]
    assert calendar.state("XHKG", local(HONG_KONG, 2026, 12, 24, 11)).note == "Handel (verkürzter Handelstag)"
    assert calendar.state("XHKG", local(HONG_KONG, 2026, 12, 24, 12, 30)).phase == "closed"


@pytest.mark.parametrize("day", [date(2027, 2, 2), date(2027, 2, 3), date(2027, 4, 6), date(2027, 12, 24)])
def test_taiwan_closures_missing_in_the_library(calendar: ExchangeCalendar, day: date) -> None:
    assert not trades_on(calendar, "XTAI", day)


def test_taiwan_trades_around_the_added_closures(calendar: ExchangeCalendar) -> None:
    assert trades_on(calendar, "XTAI", date(2027, 2, 1))
    assert trades_on(calendar, "XTAI", date(2027, 4, 7))


@pytest.mark.parametrize("day", [date(2026, 7, 17), date(2027, 7, 19)])
def test_korea_constitution_day(calendar: ExchangeCalendar, day: date) -> None:
    assert not trades_on(calendar, "XKRX", day)
    assert trades_on(calendar, "XKRX", day + timedelta(days=-1 if day.weekday() == 4 else 1))


def test_korea_late_open_on_the_first_trading_day_of_2027(calendar: ExchangeCalendar) -> None:
    assert not trades_on(calendar, "XKRX", date(2026, 12, 31))
    assert calendar.state("XKRX", local(SEOUL, 2027, 1, 4, 12)).since == local(SEOUL, 2027, 1, 4, 10)


@pytest.mark.parametrize("key", ["XSHG", "XSHE"])
def test_china_2027_built_from_researched_holidays(calendar: ExchangeCalendar, key: str) -> None:
    assert trades_on(calendar, key, date(2026, 12, 31))  # last library session
    assert not trades_on(calendar, key, date(2027, 1, 1))
    assert trades_on(calendar, key, date(2027, 1, 4))
    for day in (5, 8, 9, 10, 11, 12):
        assert not trades_on(calendar, key, date(2027, 2, day))
    assert not trades_on(calendar, key, date(2027, 2, 14))  # make-up workday on a Sunday: never a trading day
    assert trades_on(calendar, key, date(2027, 2, 15))
    assert calendar.state(key, local(SHANGHAI, 2027, 10, 5, 10)).note == "Feiertag"


@pytest.mark.parametrize("key", ["XSHG", "XSHE"])
def test_china_trading_continues_seamlessly_into_2027(calendar: ExchangeCalendar, key: str) -> None:
    last_close_2026 = local(SHANGHAI, 2026, 12, 31, 15)
    assert calendar.add_trading_minutes(key, last_close_2026 - timedelta(minutes=10), 20) == local(
        SHANGHAI, 2027, 1, 4, 9, 40
    )
    assert calendar.last_completed_session(key, local(SHANGHAI, 2027, 1, 4, 10)) == date(2026, 12, 31)


def test_india_uses_the_bse_calendar_and_researched_2027_holidays(calendar: ExchangeCalendar) -> None:
    assert not trades_on(calendar, "XNSE", date(2026, 11, 24))  # library (XBOM)
    assert trades_on(calendar, "XNSE", date(2026, 11, 25))
    assert trades_on(calendar, "XNSE", date(2027, 1, 1))  # NSE trades on 1 January
    assert not trades_on(calendar, "XNSE", date(2027, 1, 26))  # tentative 2027 list


def test_weekday_estimate_beyond_the_researched_years(caplog: pytest.LogCaptureFixture) -> None:
    fresh = ExchangeCalendar()
    with caplog.at_level(logging.WARNING, logger="trading_local.markets.calendar"):
        wednesday = local(SHANGHAI, 2028, 3, 1, 10)
        assert fresh.is_open("XSHG", wednesday)
        assert fresh.state("XSHG", wednesday).until == local(SHANGHAI, 2028, 3, 1, 11, 30)
        assert not fresh.is_open("XSHG", local(SHANGHAI, 2028, 3, 4, 10))  # Saturday
        fresh.is_open("XSHG", local(SHANGHAI, 2031, 3, 4, 10))  # extends the window: no second warning
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "XSHG" in warnings[0].getMessage() and "estimated" in warnings[0].getMessage()


def test_library_failure_falls_back_to_weekday_estimate(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("library unavailable")

    monkeypatch.setattr(calendar_module.xcals, "get_calendar", broken)
    fresh = ExchangeCalendar()
    with caplog.at_level(logging.WARNING, logger="trading_local.markets.calendar"):
        assert fresh.is_open("XETR", local(BERLIN, 2026, 11, 23, 10))
        assert not fresh.is_open("XETR", local(BERLIN, 2026, 11, 22, 10))  # Sunday
        assert fresh.state("XETR", local(BERLIN, 2026, 11, 23, 10)).until == local(BERLIN, 2026, 11, 23, 17, 30)
    assert any("library unavailable" in record.getMessage() for record in caplog.records)
