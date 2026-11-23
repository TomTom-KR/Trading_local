"""Venue table, protocol conformance, key and input validation."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from tests.markets.helpers import utc
from trading_local.contracts import MarketCalendar
from trading_local.markets import ExchangeCalendar, build_calendar
from trading_local.markets.venues import OVERVIEW_REGIONS, REGION_LABELS_DE, VENUES, venues_of_region

EXPECTED_NAMES = {
    "XTKS": "Tokio", "XHKG": "Hongkong", "XSHG": "Shanghai", "XSHE": "Shenzhen", "XKRX": "Seoul",
    "XTAI": "Taipeh", "XNSE": "Mumbai", "XASX": "Sydney", "XETR": "Xetra", "XFRA": "Frankfurt",
    "XLON": "London", "XPAR": "Paris", "XAMS": "Amsterdam", "XSWX": "Zürich", "XCSE": "Kopenhagen",
    "XNYS": "New York", "XNAS": "Nasdaq", "CRYPTO": "Krypto", "KRAKEN": "Kraken", "FX": "Devisen",
    "CME": "CME Globex",
}
LIBRARY_VENUES = [key for key, venue in VENUES.items() if venue.rule == "library"]
NORMAL_WEDNESDAY = (2026, 11, 18)  # a trading day at every venue


def test_build_calendar_returns_a_market_calendar() -> None:
    calendar = build_calendar()
    assert isinstance(calendar, ExchangeCalendar)
    assert isinstance(calendar, MarketCalendar)


def test_exchanges_lists_every_key_in_table_order(calendar: ExchangeCalendar) -> None:
    assert calendar.exchanges() == list(EXPECTED_NAMES)


@pytest.mark.parametrize(("key", "name"), EXPECTED_NAMES.items())
def test_display_names_are_german(calendar: ExchangeCalendar, key: str, name: str) -> None:
    assert calendar.display_name(key) == name


@pytest.mark.parametrize(
    ("key", "zone"),
    [("XNYS", "America/New_York"), ("XETR", "Europe/Berlin"), ("XLON", "Europe/London"), ("XTKS", "Asia/Tokyo"),
     ("XHKG", "Asia/Hong_Kong"), ("XSHE", "Asia/Shanghai"), ("XNSE", "Asia/Kolkata"), ("XASX", "Australia/Sydney"),
     ("XSWX", "Europe/Zurich"), ("CRYPTO", "UTC"), ("KRAKEN", "UTC"), ("FX", "UTC"), ("CME", "UTC")],
)
def test_timezones(calendar: ExchangeCalendar, key: str, zone: str) -> None:
    assert calendar.timezone(key) == zone


def test_every_timezone_is_a_valid_iana_zone() -> None:
    for venue in VENUES.values():
        ZoneInfo(venue.timezone)


def test_regions_and_their_venue_order() -> None:
    assert OVERVIEW_REGIONS == ("ASIA", "EU", "US")
    assert dict(REGION_LABELS_DE) == {"ASIA": "Asien", "EU": "Europa", "US": "USA"}
    assert venues_of_region("ASIA") == ("XTKS", "XHKG", "XSHG", "XSHE", "XKRX", "XTAI", "XNSE", "XASX")
    assert venues_of_region("EU") == ("XETR", "XFRA", "XLON", "XPAR", "XAMS", "XSWX", "XCSE")
    assert venues_of_region("US") == ("XNYS", "XNAS")
    assert venues_of_region("GLOBAL") == ("CRYPTO", "KRAKEN", "FX", "CME")


def test_only_us_venues_have_extended_hours() -> None:
    assert {key for key, venue in VENUES.items() if venue.extended is not None} == {"XNYS", "XNAS"}


@pytest.mark.parametrize("key", LIBRARY_VENUES)
def test_library_hours_agree_with_the_venue_table(calendar: ExchangeCalendar, key: str) -> None:
    """The library's segments of a normal day equal the regular hours documented in venues.py."""
    venue = VENUES[key]
    zone = ZoneInfo(venue.timezone)
    day_start = datetime(*NORMAL_WEDNESDAY, tzinfo=zone)
    segments = calendar.segments(key, day_start, day_start + timedelta(days=1))
    local_hours = [(s.start.astimezone(zone).time(), s.end.astimezone(zone).time()) for s in segments]
    assert local_hours == list(venue.regular_hours)


CALLS: dict[str, Callable[[ExchangeCalendar, str], Any]] = {
    "display_name": lambda c, key: c.display_name(key),
    "timezone": lambda c, key: c.timezone(key),
    "state": lambda c, key: c.state(key, utc(2026, 11, 20, 15)),
    "is_open": lambda c, key: c.is_open(key, utc(2026, 11, 20, 15)),
    "segments": lambda c, key: c.segments(key, utc(2026, 11, 20), utc(2026, 11, 21)),
    "session_date": lambda c, key: c.session_date(key, utc(2026, 11, 20, 15)),
    "last_completed_session": lambda c, key: c.last_completed_session(key, utc(2026, 11, 20, 15)),
    "minutes_since_open": lambda c, key: c.minutes_since_open(key, utc(2026, 11, 20, 15)),
    "minutes_to_close": lambda c, key: c.minutes_to_close(key, utc(2026, 11, 20, 15)),
    "add_trading_minutes": lambda c, key: c.add_trading_minutes(key, utc(2026, 11, 20, 15), 5),
}


@pytest.mark.parametrize("method", CALLS)
@pytest.mark.parametrize("key", ["XXXX", "xnys", "XBOM", ""])
def test_unknown_key_raises_key_error(calendar: ExchangeCalendar, method: str, key: str) -> None:
    with pytest.raises(KeyError, match="unknown exchange calendar key"):
        CALLS[method](calendar, key)


@pytest.mark.parametrize("method", [m for m in CALLS if m not in ("display_name", "timezone")])
def test_naive_datetimes_are_rejected(calendar: ExchangeCalendar, method: str) -> None:
    naive = datetime(2026, 11, 20, 15)
    calls: dict[str, Callable[[], Any]] = {
        "state": lambda: calendar.state("XNYS", naive),
        "is_open": lambda: calendar.is_open("XNYS", naive),
        "segments": lambda: calendar.segments("XNYS", naive, naive + timedelta(days=1)),
        "session_date": lambda: calendar.session_date("XNYS", naive),
        "last_completed_session": lambda: calendar.last_completed_session("XNYS", naive),
        "minutes_since_open": lambda: calendar.minutes_since_open("XNYS", naive),
        "minutes_to_close": lambda: calendar.minutes_to_close("XNYS", naive),
        "add_trading_minutes": lambda: calendar.add_trading_minutes("XNYS", naive, 5),
    }
    with pytest.raises(ValueError, match="naive datetime"):
        calls[method]()


def test_results_are_utc_even_for_non_utc_input(calendar: ExchangeCalendar) -> None:
    berlin_time = datetime(2026, 11, 20, 16, 0, tzinfo=ZoneInfo("Europe/Berlin"))  # 15:00 UTC
    state = calendar.state("XNYS", berlin_time)
    assert state.since == utc(2026, 11, 20, 14, 30)
    assert state.since.tzinfo is UTC
    assert calendar.add_trading_minutes("XNYS", berlin_time, 0) == utc(2026, 11, 20, 15)
