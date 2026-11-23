"""Rule-based venues: CRYPTO and KRAKEN (24/7), FX (Sunday 17:00 - Friday 17:00 New York) and CME Globex."""

from datetime import date, timedelta

import pytest

from tests.markets.helpers import NEW_YORK, ONE_SECOND, local, spans, utc
from trading_local.clock import to_ts
from trading_local.execution import session_flatten_ts
from trading_local.markets import ExchangeCalendar
from trading_local.markets.venues import VENUES


# ------------------------------------------------------------------------------------------------
# 24/7 venues: CRYPTO (Yahoo's crypto quotes) and KRAKEN (the Kraken exchange) follow the same rule
# ------------------------------------------------------------------------------------------------
ALWAYS_OPEN = ("CRYPTO", "KRAKEN")


@pytest.mark.parametrize("key", ALWAYS_OPEN)
@pytest.mark.parametrize(
    "at", [utc(2026, 11, 22, 3), utc(2026, 12, 25, 12), utc(2027, 1, 1), utc(2026, 11, 21, 23, 59, 59)]
)
def test_crypto_is_always_open(calendar: ExchangeCalendar, at, key: str) -> None:
    assert calendar.is_open(key, at)
    state = calendar.state(key, at)
    assert (state.phase, state.is_open, state.note) == ("open", True, "Rund um die Uhr")
    assert (state.since, state.until, state.next_open, state.next_close) == (None, None, None, None)


@pytest.mark.parametrize("key", ALWAYS_OPEN)
def test_crypto_trading_day_is_the_utc_day(calendar: ExchangeCalendar, key: str) -> None:
    assert spans(calendar.segments(key, utc(2026, 11, 21, 18), utc(2026, 11, 23, 6))) == [
        (utc(2026, 11, 21, 18), utc(2026, 11, 22)),
        (utc(2026, 11, 22), utc(2026, 11, 23)),
        (utc(2026, 11, 23), utc(2026, 11, 23, 6)),
    ]
    assert calendar.session_date(key, utc(2026, 11, 22, 23, 59)) == date(2026, 11, 22)
    assert calendar.session_date(key, local(NEW_YORK, 2026, 11, 22, 20)) == date(2026, 11, 23)
    assert calendar.last_completed_session(key, utc(2026, 11, 22)) == date(2026, 11, 21)
    assert calendar.last_completed_session(key, utc(2026, 11, 22, 23, 59)) == date(2026, 11, 21)
    assert calendar.minutes_since_open(key, utc(2026, 11, 22, 10)) == 600
    assert calendar.minutes_to_close(key, utc(2026, 11, 22, 10)) == 840
    assert calendar.minutes_since_open(key, utc(2026, 11, 22)) == 0


@pytest.mark.parametrize("key", ALWAYS_OPEN)
def test_crypto_add_minutes_runs_through_midnight(calendar: ExchangeCalendar, key: str) -> None:
    assert calendar.add_trading_minutes(key, utc(2026, 11, 22, 23, 50), 20) == utc(2026, 11, 23, 0, 10)
    assert calendar.add_trading_minutes(key, utc(2026, 11, 22, 23, 50), 10) == utc(2026, 11, 23)


def test_kraken_is_a_global_24_7_venue_named_kraken(calendar: ExchangeCalendar) -> None:
    venue = VENUES["KRAKEN"]
    assert (venue.name_de, venue.timezone, venue.region, venue.rule) == ("Kraken", "UTC", "GLOBAL", "always_open")
    assert calendar.display_name("KRAKEN") == "Kraken" and "KRAKEN" in calendar.exchanges()
    # a weekend and a holiday of the stock exchanges trade like any other day
    for at in (utc(2026, 11, 28, 12), utc(2026, 12, 25, 0, 30), utc(2027, 1, 1, 23, 59)):
        assert calendar.is_open("KRAKEN", at)
    assert len(calendar.segments("KRAKEN", utc(2026, 11, 27), utc(2026, 11, 30))) == 3  # one per UTC day


def test_kraken_intraday_positions_are_flattened_before_utc_midnight(calendar: ExchangeCalendar) -> None:
    """execution.session_flatten_ts treats KRAKEN like CRYPTO: the UTC day is the trading day."""
    for key in ALWAYS_OPEN:
        assert session_flatten_ts(calendar, key, utc(2026, 11, 22, 15, 7), 15) == to_ts(utc(2026, 11, 22, 23, 45))
        assert session_flatten_ts(calendar, key, utc(2026, 11, 22, 23, 50), 15) == to_ts(utc(2026, 11, 22, 23, 45))


# ------------------------------------------------------------------------------------------------
# FX
# ------------------------------------------------------------------------------------------------
def test_fx_friday_close_and_sunday_open_in_winter(calendar: ExchangeCalendar) -> None:
    close, reopen = local(NEW_YORK, 2026, 11, 20, 17), local(NEW_YORK, 2026, 11, 22, 17)
    assert (close, reopen) == (utc(2026, 11, 20, 22), utc(2026, 11, 22, 22))
    assert calendar.is_open("FX", close - ONE_SECOND)
    assert not calendar.is_open("FX", close)
    assert not calendar.is_open("FX", reopen - ONE_SECOND)
    assert calendar.is_open("FX", reopen)
    weekend = calendar.state("FX", utc(2026, 11, 21, 12))
    assert (weekend.phase, weekend.note, weekend.since, weekend.until) == ("closed", "Wochenende", close, reopen)
    assert weekend.next_close == utc(2026, 11, 27, 22)


def test_fx_hours_follow_new_york_daylight_saving(calendar: ExchangeCalendar) -> None:
    # Friday 2026-10-30 17:00 EDT = 21:00 UTC; Sunday 2026-11-01 (after the switch) 17:00 EST = 22:00 UTC
    assert calendar.is_open("FX", utc(2026, 10, 30, 20, 59))
    assert not calendar.is_open("FX", utc(2026, 10, 30, 21))
    assert not calendar.is_open("FX", utc(2026, 11, 1, 21, 59))
    assert calendar.is_open("FX", utc(2026, 11, 1, 22))


def test_fx_midnight_split_is_not_a_phase_change(calendar: ExchangeCalendar) -> None:
    state = calendar.state("FX", utc(2026, 11, 24))
    assert (state.phase, state.note) == ("open", "Handel")
    assert state.since == utc(2026, 11, 22, 22)
    assert state.until == state.next_close == utc(2026, 11, 27, 22)
    assert state.next_open == utc(2026, 11, 29, 22)
    assert spans(calendar.segments("FX", utc(2026, 11, 22, 20), utc(2026, 11, 23, 3))) == [
        (utc(2026, 11, 22, 22), utc(2026, 11, 23)),
        (utc(2026, 11, 23), utc(2026, 11, 23, 3)),
    ]


def test_fx_trading_dates_and_minutes(calendar: ExchangeCalendar) -> None:
    assert calendar.minutes_since_open("FX", utc(2026, 11, 22, 23)) == 60  # Sunday's short UTC day
    assert calendar.minutes_to_close("FX", utc(2026, 11, 22, 23)) == 60
    assert calendar.minutes_since_open("FX", utc(2026, 11, 23, 1)) == 60
    assert calendar.minutes_to_close("FX", utc(2026, 11, 23, 1)) == 23 * 60
    assert calendar.minutes_to_close("FX", utc(2026, 11, 27, 21)) == 60  # Friday closes at 22:00 UTC
    assert calendar.session_date("FX", utc(2026, 11, 22, 23)) == date(2026, 11, 22)
    assert calendar.last_completed_session("FX", utc(2026, 11, 21, 12)) == date(2026, 11, 20)
    assert calendar.last_completed_session("FX", utc(2026, 11, 23, 10)) == date(2026, 11, 22)
    assert calendar.add_trading_minutes("FX", utc(2026, 11, 20, 21, 50), 20) == utc(2026, 11, 22, 22, 10)


# ------------------------------------------------------------------------------------------------
# CME Globex
# ------------------------------------------------------------------------------------------------
def test_cme_week_and_daily_break_in_winter(calendar: ExchangeCalendar) -> None:
    sunday_open = local(NEW_YORK, 2026, 11, 22, 18)
    assert sunday_open == utc(2026, 11, 22, 23)
    assert not calendar.is_open("CME", sunday_open - ONE_SECOND)
    assert calendar.is_open("CME", sunday_open)
    pause_start, pause_end = utc(2026, 11, 24, 22), utc(2026, 11, 24, 23)  # 17:00-18:00 New York
    assert calendar.is_open("CME", pause_start - ONE_SECOND)
    pause = calendar.state("CME", pause_start)
    assert (pause.phase, pause.note, pause.is_open) == ("break", "Handelspause", False)
    assert (pause.since, pause.until, pause.next_open) == (pause_start, pause_end, pause_end)
    assert pause.next_close == utc(2026, 11, 25, 22)
    assert calendar.is_open("CME", pause_end)
    friday_close = utc(2026, 11, 27, 22)
    assert calendar.is_open("CME", friday_close - ONE_SECOND)
    weekend = calendar.state("CME", friday_close)
    assert (weekend.phase, weekend.note, weekend.until) == ("closed", "Wochenende", utc(2026, 11, 29, 23))


def test_cme_daily_break_in_summer_and_holidays_ignored(calendar: ExchangeCalendar) -> None:
    assert calendar.state("CME", utc(2027, 7, 13, 21, 30)).phase == "break"  # 17:30 EDT
    assert calendar.is_open("CME", utc(2027, 7, 13, 22))
    assert calendar.is_open("CME", utc(2026, 11, 26, 15))  # US Thanksgiving


def test_cme_minutes_exclude_the_daily_break(calendar: ExchangeCalendar) -> None:
    evening = utc(2026, 11, 24, 23, 30)  # after the break; the UTC day keeps counting from midnight
    assert calendar.minutes_since_open("CME", evening) == 22 * 60 + 30
    assert calendar.minutes_to_close("CME", evening) == 30
    assert calendar.minutes_since_open("CME", utc(2026, 11, 24, 22, 30)) is None
    assert calendar.add_trading_minutes("CME", utc(2026, 11, 24, 21, 50), 20) == utc(2026, 11, 24, 23, 10)
    assert calendar.last_completed_session("CME", utc(2026, 11, 28, 12)) == date(2026, 11, 27)


def test_cme_segments_split_at_the_break_and_at_midnight(calendar: ExchangeCalendar) -> None:
    assert spans(calendar.segments("CME", utc(2026, 11, 24, 20), utc(2026, 11, 25, 2))) == [
        (utc(2026, 11, 24, 20), utc(2026, 11, 24, 22)),
        (utc(2026, 11, 24, 23), utc(2026, 11, 25)),
        (utc(2026, 11, 25), utc(2026, 11, 25, 2)),
    ]
    assert calendar.state("CME", utc(2026, 11, 25, 1)).since == utc(2026, 11, 24, 23)
    assert calendar.state("CME", utc(2026, 11, 25, 1)).next_open == utc(2026, 11, 25, 23)
    assert calendar.segments("CME", utc(2026, 11, 28), utc(2026, 11, 29)) == []  # Saturday


def test_pseudo_venue_timezones_match_their_utc_trading_date(calendar: ExchangeCalendar) -> None:
    for key in ("CRYPTO", "KRAKEN", "FX", "CME"):
        assert calendar.timezone(key) == "UTC"
    assert calendar.session_date("CME", utc(2026, 11, 24, 23, 30)) == date(2026, 11, 24)
    assert calendar.session_date("FX", utc(2026, 11, 24) - timedelta(seconds=1)) == date(2026, 11, 23)
