"""Library data ranges, cross-method invariants, DST switches of the rule venues and concurrent rebuilds."""

import logging
import random
import threading
from datetime import UTC, date, datetime, timedelta

import exchange_calendars as xcals
import pytest

from tests.markets.helpers import HONG_KONG, NEW_YORK, ONE_SECOND, SYDNEY, TOKYO, local, spans, utc
from trading_local.markets import ExchangeCalendar, sessions_overview
from trading_local.markets import calendar as calendar_module
from trading_local.markets.venues import VENUES
from trading_local.models import Instrument

LIBRARY_VENUES = [key for key, venue in VENUES.items() if venue.rule == "library"]


# ------------------------------------------------------------------------------------------------
# Library data range
# ------------------------------------------------------------------------------------------------
def test_library_ranges_in_the_venue_table_match_the_library() -> None:
    """library_from / library_until must equal the calendars' bound_min / bound_max (detects library upgrades)."""
    # The library has no public accessor for a calendar class; its factory table avoids building 15 calendars.
    factories = xcals.calendar_utils._default_calendar_factories
    for key in LIBRARY_VENUES:
        venue = VENUES[key]
        library = factories[xcals.resolve_alias(venue.library_code or key)]
        low, high = library.bound_min(), library.bound_max()
        assert venue.library_from == (None if low is None else low.date()), key
        assert venue.library_until == (None if high is None else high.date()), key


def test_queries_beyond_the_library_range_keep_the_library_data_of_other_days() -> None:
    """A query after XHKG's last library year or before XTKS's first one must not discard current holidays."""
    fresh = ExchangeCalendar()
    christmas, half_day = local(HONG_KONG, 2026, 12, 25, 10), local(HONG_KONG, 2026, 12, 24, 14)
    assert not fresh.is_open("XHKG", christmas)
    future_wednesday = local(HONG_KONG, 2050, 3, 2, 10)  # beyond the library (2049-12-31): weekday estimate
    assert fresh.is_open("XHKG", future_wednesday)
    assert fresh.state("XHKG", local(HONG_KONG, 2050, 3, 2, 12, 30)).phase == "break"
    assert not fresh.is_open("XHKG", christmas)
    assert not fresh.is_open("XHKG", half_day)
    assert fresh.state("XHKG", christmas).note == "Feiertag"

    assert not fresh.is_open("XTKS", local(TOKYO, 2026, 11, 23, 10))
    assert fresh.is_open("XTKS", local(TOKYO, 1996, 3, 1, 10))  # before the library (1997-01-01): estimate
    assert not fresh.is_open("XTKS", local(TOKYO, 2026, 11, 23, 10))
    assert not fresh.is_open("XTKS", local(TOKYO, 2003, 1, 2, 10))  # library data inside the new window

    far = fresh.add_trading_minutes("XHKG", local(HONG_KONG, 2049, 12, 31, 11, 50), 20)  # half day, then 2050
    assert far == local(HONG_KONG, 2050, 1, 3, 9, 40)  # estimate: Monday 2050-01-03 is a weekday
    assert not fresh.is_open("XHKG", half_day)


def test_estimate_outside_the_library_range_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    fresh = ExchangeCalendar()
    with caplog.at_level(logging.WARNING, logger="trading_local.markets.calendar"):
        assert fresh.is_open("XTKS", local(TOKYO, 1996, 3, 1, 10))
        assert not fresh.is_open("XTKS", local(TOKYO, 1996, 3, 2, 10))  # Saturday
        assert fresh.is_open("XTKS", local(TOKYO, 1996, 3, 4, 13))
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(messages) == 1
    assert "XTKS" in messages[0] and "..1996-12-31" in messages[0]


def test_windows_that_overlap_the_library_range_by_a_few_days(caplog: pytest.LogCaptureFixture) -> None:
    """The library request is widened to a minimum span but never beyond the library's data range."""
    at = utc(2026, 12, 28, 3) + timedelta(days=calendar_module.HISTORY_DAYS)  # window starts 2026-12-28
    fresh = ExchangeCalendar()
    with caplog.at_level(logging.WARNING, logger="trading_local.markets.calendar"):
        assert fresh.is_open("XSHG", at - timedelta(hours=1))  # Tuesday 2032-12-28 10:00 Shanghai (estimate)
        assert fresh.is_open("XTKS", local(TOKYO, 1995, 12, 1, 10))  # window ends 1997-01-04
    messages = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert not any("failed" in message for message in messages), messages
    assert any("XSHG" in message and "data for 2027-01-01.." in message for message in messages), messages
    assert any("XTKS" in message and "..1996-12-31:" in message for message in messages), messages
    assert fresh.last_completed_session("XTKS", local(TOKYO, 1997, 1, 6, 16)) == date(1997, 1, 6)


# ------------------------------------------------------------------------------------------------
# Rule venues across the US daylight-saving switches
# ------------------------------------------------------------------------------------------------
def test_fx_week_across_the_spring_switch(calendar: ExchangeCalendar) -> None:
    friday_close = local(NEW_YORK, 2027, 3, 12, 17)  # EST
    sunday_open = local(NEW_YORK, 2027, 3, 14, 17)  # EDT (switch at 02:00 that morning)
    assert (friday_close, sunday_open) == (utc(2027, 3, 12, 22), utc(2027, 3, 14, 21))
    assert calendar.is_open("FX", friday_close - ONE_SECOND)
    assert not calendar.is_open("FX", friday_close)
    assert not calendar.is_open("FX", sunday_open - ONE_SECOND)
    assert calendar.is_open("FX", sunday_open)
    assert calendar.state("FX", utc(2027, 3, 13, 12)).until == sunday_open


def test_cme_week_across_the_autumn_switch(calendar: ExchangeCalendar) -> None:
    assert not calendar.is_open("CME", utc(2026, 10, 30, 21))  # Friday 17:00 EDT
    assert calendar.is_open("CME", utc(2026, 10, 30, 20, 59, 59))
    sunday_open = local(NEW_YORK, 2026, 11, 1, 18)  # EST, the day of the switch
    assert sunday_open == utc(2026, 11, 1, 23)
    assert calendar.state("CME", utc(2026, 11, 1, 22)).next_open == sunday_open
    assert calendar.is_open("CME", sunday_open)
    assert calendar.state("CME", utc(2026, 11, 2, 22, 30)).phase == "break"  # Monday 17:30 EST


# ------------------------------------------------------------------------------------------------
# Further boundary facts
# ------------------------------------------------------------------------------------------------
def test_us_pre_market_follows_daylight_saving(calendar: ExchangeCalendar) -> None:
    assert calendar.state("XNYS", utc(2026, 11, 2, 9)).phase == "pre"  # 04:00 EST
    assert calendar.state("XNYS", utc(2026, 11, 2, 9) - ONE_SECOND).phase == "closed"
    assert calendar.state("XNYS", utc(2027, 3, 15, 8)).phase == "pre"  # 04:00 EDT
    pre = calendar.state("XNYS", local(NEW_YORK, 2026, 11, 27, 8))  # early-close day: normal pre-market
    assert (pre.phase, pre.since) == ("pre", local(NEW_YORK, 2026, 11, 27, 4))
    assert (pre.until, pre.next_close) == (local(NEW_YORK, 2026, 11, 27, 9, 30), local(NEW_YORK, 2026, 11, 27, 13))


def test_us_overnight_session_is_not_a_trading_phase(calendar: ExchangeCalendar) -> None:
    night = local(NEW_YORK, 2026, 12, 8, 22)  # after the 23x5 launch of 2026-12-06
    state = calendar.state("XNYS", night)
    assert (state.phase, state.is_open) == ("closed", False)
    assert calendar.segments("XNAS", night, night + timedelta(hours=5), extended=True) == []


def test_segments_clip_at_sub_second_boundaries(calendar: ExchangeCalendar) -> None:
    close, open_ = utc(2026, 11, 24, 21), utc(2026, 11, 24, 14, 30)
    half = timedelta(microseconds=500_000)
    assert spans(calendar.segments("XNYS", close - half, close + timedelta(hours=1))) == [(close - half, close)]
    assert spans(calendar.segments("XNYS", open_ - timedelta(hours=1), open_ + half)) == [(open_, open_ + half)]
    assert calendar.segments("XNYS", close + half, close + timedelta(hours=1)) == []
    assert calendar.segments("XNYS", open_ - timedelta(hours=1), open_ - half) == []


def test_sydney_last_completed_session_across_utc_midnight(calendar: ExchangeCalendar) -> None:
    close = local(SYDNEY, 2026, 11, 23, 16)  # 05:00 UTC; the session opened 23:00 UTC the day before
    assert calendar.last_completed_session("XASX", close - ONE_SECOND) == date(2026, 11, 20)
    assert calendar.last_completed_session("XASX", close) == date(2026, 11, 23)
    assert calendar.minutes_since_open("XASX", utc(2026, 11, 23, 1)) == 120
    assert calendar.minutes_to_close("XASX", utc(2026, 11, 23, 1)) == 240


def test_overview_ignores_pseudo_venues_and_unknown_exchanges(calendar: ExchangeCalendar) -> None:
    instruments = [
        Instrument("BTC-USD", "Bitcoin", "crypto", "GLOBAL", "CRYPTO", "USD"),
        Instrument("EURUSD=X", "EUR/USD", "fx", "GLOBAL", "FX", "USD"),
        Instrument("ES=F", "S&P 500 Future", "future", "US", "CME", "USD"),
        Instrument("ODD", "Unbekannt", "equity", "EU", "XWAR", "PLN"),
        Instrument("0700.HK", "Tencent", "equity", "ASIA", "XHKG", "HKD"),
    ]
    result = sessions_overview(calendar, instruments, utc(2026, 11, 24, 3))
    assert [[exchange.exchange for exchange in region.exchanges] for region in result.regions] == [
        ["XHKG"], ["XETR"], ["XNYS"]
    ]
    assert result.regions[0].is_open  # Hong Kong afternoon session
    assert {segment.exchange for segment in result.timeline} <= {"XHKG", "XETR", "XNYS"}


# ------------------------------------------------------------------------------------------------
# Cross-method invariants on random instants
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", [key for key, venue in VENUES.items() if venue.rule != "always_open"])
def test_methods_agree_with_each_other(calendar: ExchangeCalendar, key: str) -> None:
    rng = random.Random(key)
    low, high = int(utc(2025, 6, 1).timestamp()), int(utc(2027, 12, 1).timestamp())
    for _ in range(60):
        at = datetime.fromtimestamp(rng.randrange(low, high), UTC)
        state = calendar.state(key, at)
        assert state.is_open is calendar.is_open(key, at) is (state.phase == "open"), at
        assert state.since <= at < state.until, at
        assert calendar.state(key, state.since).phase == state.phase, at
        assert calendar.state(key, state.since - ONE_SECOND).phase != state.phase, at
        assert calendar.state(key, state.until - ONE_SECOND).phase == state.phase, at
        assert calendar.state(key, state.until).phase != state.phase, at
        assert at < state.next_open and calendar.is_open(key, state.next_open), at
        assert at < state.next_close and not calendar.is_open(key, state.next_close), at
        assert calendar.is_open(key, state.next_close - ONE_SECOND), at
        if not state.is_open:
            assert calendar.segments(key, at, state.next_open) == [], at
        elapsed, remaining = calendar.minutes_since_open(key, at), calendar.minutes_to_close(key, at)
        assert (elapsed is None) is (remaining is None) is (not state.is_open), at
        completed = calendar.last_completed_session(key, at)
        assert completed is not None and completed <= calendar.session_date(key, at), at
        minutes = rng.choice([1, 13, 61, 389, 1000])
        reached = calendar.add_trading_minutes(key, at, minutes)
        counted = sum((s.end - s.start for s in calendar.segments(key, at, reached)), timedelta())
        assert counted == timedelta(minutes=minutes), (at, minutes)
        assert calendar.is_open(key, reached - ONE_SECOND), (at, minutes)


# ------------------------------------------------------------------------------------------------
# Concurrency
# ------------------------------------------------------------------------------------------------
def test_readers_see_correct_answers_while_other_threads_extend_the_window() -> None:
    fresh = ExchangeCalendar()
    trading, thanksgiving = utc(2026, 11, 20, 15), local(NEW_YORK, 2026, 11, 26, 11)
    fresh.is_open("XNYS", trading)
    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                assert fresh.is_open("XNYS", trading)
                assert not fresh.is_open("XNYS", thanksgiving)
                assert fresh.state("XNYS", thanksgiving).note == "Feiertag"
        except BaseException as exc:  # re-raised by the main thread
            errors.append(exc)

    def extender(year: int) -> None:
        try:
            assert fresh.last_completed_session("XNYS", local(NEW_YORK, year, 3, 2, 17)) == date(year, 3, 2)
        except BaseException as exc:
            errors.append(exc)

    readers = [threading.Thread(target=reader) for _ in range(3)]
    extenders = [threading.Thread(target=extender, args=(year,)) for year in (2034, 2016, 2035, 2015)]
    for thread in readers + extenders:
        thread.start()
    for thread in extenders:
        thread.join(timeout=60)
    stop.set()
    for thread in readers:
        thread.join(timeout=10)
    assert not errors
