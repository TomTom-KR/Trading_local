"""Regular hours, lunch breaks and daylight-saving switches pinned with real dates."""

from datetime import timedelta

import pytest

from tests.markets.helpers import BERLIN, HONG_KONG, LONDON, NEW_YORK, SHANGHAI, SYDNEY, TOKYO, local, spans, utc
from trading_local.markets import ExchangeCalendar


def day_spans(calendar: ExchangeCalendar, key: str, year: int, month: int, day: int) -> list:
    """Regular segments of one UTC day (all venues below trade within a single UTC day)."""
    start = utc(year, month, day)
    return spans(calendar.segments(key, start, start + timedelta(days=1)))


@pytest.mark.parametrize("key", ["XNYS", "XNAS"])
def test_new_york_winter_and_summer(calendar: ExchangeCalendar, key: str) -> None:
    # 09:30-16:00 New York = 14:30-21:00 UTC in winter, 13:30-20:00 UTC in summer
    assert day_spans(calendar, key, 2026, 11, 23) == [(utc(2026, 11, 23, 14, 30), utc(2026, 11, 23, 21))]
    assert day_spans(calendar, key, 2026, 10, 28) == [(utc(2026, 10, 28, 13, 30), utc(2026, 10, 28, 20))]
    assert day_spans(calendar, key, 2027, 7, 14) == [(utc(2027, 7, 14, 13, 30), utc(2027, 7, 14, 20))]


def test_xetra_winter_and_summer(calendar: ExchangeCalendar) -> None:
    # 09:00-17:30 Berlin
    assert day_spans(calendar, "XETR", 2026, 11, 23) == [(utc(2026, 11, 23, 8), utc(2026, 11, 23, 16, 30))]
    assert day_spans(calendar, "XETR", 2026, 10, 23) == [(utc(2026, 10, 23, 7), utc(2026, 10, 23, 15, 30))]


def test_frankfurt_keeps_xetra_aligned_hours(calendar: ExchangeCalendar) -> None:
    assert day_spans(calendar, "XFRA", 2026, 11, 23) == day_spans(calendar, "XETR", 2026, 11, 23)


def test_london_winter_and_summer(calendar: ExchangeCalendar) -> None:
    # 08:00-16:30 London
    assert day_spans(calendar, "XLON", 2026, 11, 23) == [(utc(2026, 11, 23, 8), utc(2026, 11, 23, 16, 30))]
    assert day_spans(calendar, "XLON", 2027, 3, 30) == [(utc(2027, 3, 30, 7), utc(2027, 3, 30, 15, 30))]


def test_tokyo_lunch_break_and_1530_close(calendar: ExchangeCalendar) -> None:
    # 09:00-11:30 and 12:30-15:30 Tokyo (UTC+9, no daylight saving)
    assert day_spans(calendar, "XTKS", 2026, 11, 24) == [
        (local(TOKYO, 2026, 11, 24, 9), local(TOKYO, 2026, 11, 24, 11, 30)),
        (local(TOKYO, 2026, 11, 24, 12, 30), local(TOKYO, 2026, 11, 24, 15, 30)),
    ]


def test_tokyo_closed_at_1500_before_november_2024(calendar: ExchangeCalendar) -> None:
    assert day_spans(calendar, "XTKS", 2024, 11, 1)[-1][1] == local(TOKYO, 2024, 11, 1, 15)
    assert day_spans(calendar, "XTKS", 2024, 11, 5)[-1][1] == local(TOKYO, 2024, 11, 5, 15, 30)


def test_hong_kong_lunch_break(calendar: ExchangeCalendar) -> None:
    assert day_spans(calendar, "XHKG", 2026, 11, 23) == [
        (local(HONG_KONG, 2026, 11, 23, 9, 30), local(HONG_KONG, 2026, 11, 23, 12)),
        (local(HONG_KONG, 2026, 11, 23, 13), local(HONG_KONG, 2026, 11, 23, 16)),
    ]


@pytest.mark.parametrize("key", ["XSHG", "XSHE"])
@pytest.mark.parametrize("day", [(2026, 11, 23), (2027, 1, 4), (2027, 6, 1)])  # library data and 2027 estimate
def test_shanghai_and_shenzhen_lunch_break(calendar: ExchangeCalendar, key: str, day: tuple[int, int, int]) -> None:
    assert day_spans(calendar, key, *day) == [
        (local(SHANGHAI, *day, 9, 30), local(SHANGHAI, *day, 11, 30)),
        (local(SHANGHAI, *day, 13), local(SHANGHAI, *day, 15)),
    ]


def test_us_switch_week_autumn(calendar: ExchangeCalendar) -> None:
    """EU leaves summer time on 2026-10-25, the US on 2026-11-01: one week with a 5 hour difference."""
    assert calendar.state("XETR", utc(2026, 10, 26, 8)).since == utc(2026, 10, 26, 8)
    assert calendar.state("XNYS", utc(2026, 10, 26, 14)).since == utc(2026, 10, 26, 13, 30)
    assert local(BERLIN, 2026, 10, 26, 14, 30) == utc(2026, 10, 26, 13, 30)  # NYSE opens 14:30 in Berlin
    assert calendar.state("XNYS", utc(2026, 11, 2, 15)).since == utc(2026, 11, 2, 14, 30)


def test_us_switch_week_spring(calendar: ExchangeCalendar) -> None:
    """US enters summer time on 2027-03-14, the EU on 2027-03-28."""
    assert calendar.state("XNYS", utc(2027, 3, 12, 15)).since == utc(2027, 3, 12, 14, 30)
    assert calendar.state("XNYS", utc(2027, 3, 15, 15)).since == utc(2027, 3, 15, 13, 30)
    assert calendar.state("XETR", utc(2027, 3, 15, 9)).since == utc(2027, 3, 15, 8)
    assert calendar.state("XETR", utc(2027, 3, 30, 9)).since == utc(2027, 3, 30, 7)  # 03-29 is Easter Monday
    assert calendar.state("XLON", utc(2027, 3, 30, 9)).since == utc(2027, 3, 30, 7)


def test_dst_sunday_itself_is_closed(calendar: ExchangeCalendar) -> None:
    for key, at in (("XNYS", utc(2026, 11, 1, 15)), ("XETR", utc(2026, 10, 25, 10)), ("XNYS", utc(2027, 3, 14, 15))):
        state = calendar.state(key, at)
        assert (state.phase, state.note) == ("closed", "Wochenende")


def test_sydney_session_starts_on_the_previous_utc_day(calendar: ExchangeCalendar) -> None:
    """ASX 10:00-16:00 Sydney; in the southern summer (UTC+11) the session opens at 23:00 UTC the day before."""
    opens = utc(2026, 11, 22, 23)
    assert local(SYDNEY, 2026, 11, 23, 10) == opens
    assert calendar.is_open("XASX", opens)
    assert calendar.session_date("XASX", opens).isoformat() == "2026-11-23"
    assert calendar.state("XASX", opens).until == local(SYDNEY, 2026, 11, 23, 16)
    assert calendar.state("XASX", utc(2027, 6, 15, 1)).since == utc(2027, 6, 15)  # winter, UTC+10


@pytest.mark.parametrize(
    ("key", "minutes"), [("XNYS", 390), ("XETR", 510), ("XLON", 510), ("XTKS", 330), ("XHKG", 330), ("XSHG", 240)]
)
def test_trading_minutes_per_regular_session(calendar: ExchangeCalendar, key: str, minutes: int) -> None:
    first_open = calendar.segments(key, utc(2026, 11, 23, 22), utc(2026, 11, 25))[0].start  # all closed at 22:00
    assert calendar.minutes_to_close(key, first_open) == minutes
    assert calendar.minutes_since_open(key, first_open) == 0


def test_london_and_new_york_seen_from_berlin(calendar: ExchangeCalendar) -> None:
    assert calendar.state("XLON", local(BERLIN, 2026, 11, 23, 9)).since == local(LONDON, 2026, 11, 23, 8)
    assert calendar.state("XNYS", local(BERLIN, 2026, 11, 23, 15, 30)).since == local(NEW_YORK, 2026, 11, 23, 9, 30)
