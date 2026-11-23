"""Every protocol method at and around the session boundaries (open, close, lunch break)."""

from datetime import date, timedelta

import pytest

from tests.markets.helpers import NEW_YORK, ONE_SECOND, TOKYO, local, spans, utc
from trading_local.markets import ExchangeCalendar

# Tuesday 2026-11-24: New York 09:30-16:00 = 14:30-21:00 UTC; Tokyo 00:00-02:30 and 03:30-06:30 UTC.
NY_OPEN = utc(2026, 11, 24, 14, 30)
NY_CLOSE = utc(2026, 11, 24, 21)
TK_OPEN = utc(2026, 11, 24, 0)
TK_LUNCH = utc(2026, 11, 24, 2, 30)
TK_AFTERNOON = utc(2026, 11, 24, 3, 30)
TK_CLOSE = utc(2026, 11, 24, 6, 30)


# ------------------------------------------------------------------------------------------------
# is_open / state
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("key", "at", "expected"),
    [
        ("XNYS", NY_OPEN - ONE_SECOND, False), ("XNYS", NY_OPEN, True), ("XNYS", NY_OPEN + ONE_SECOND, True),
        ("XNYS", NY_CLOSE - ONE_SECOND, True), ("XNYS", NY_CLOSE, False), ("XNYS", NY_CLOSE + ONE_SECOND, False),
        ("XTKS", TK_OPEN - ONE_SECOND, False), ("XTKS", TK_OPEN, True),
        ("XTKS", TK_LUNCH - ONE_SECOND, True), ("XTKS", TK_LUNCH, False), ("XTKS", TK_LUNCH + ONE_SECOND, False),
        ("XTKS", TK_AFTERNOON - ONE_SECOND, False), ("XTKS", TK_AFTERNOON, True),
        ("XTKS", TK_CLOSE - ONE_SECOND, True), ("XTKS", TK_CLOSE, False),
    ],
)
def test_is_open_at_the_boundaries(calendar: ExchangeCalendar, key: str, at, expected: bool) -> None:
    assert calendar.is_open(key, at) is expected
    assert calendar.state(key, at).is_open is expected


def test_state_new_york_through_a_trading_day(calendar: ExchangeCalendar) -> None:
    pre_start, post_end = local(NEW_YORK, 2026, 11, 24, 4), local(NEW_YORK, 2026, 11, 24, 20)
    next_open, next_close = utc(2026, 11, 25, 14, 30), utc(2026, 11, 25, 21)
    night = calendar.state("XNYS", pre_start - ONE_SECOND)
    assert (night.phase, night.note, night.until) == ("closed", "Geschlossen", pre_start)
    assert night.since == local(NEW_YORK, 2026, 11, 23, 20)  # end of Monday's after-hours window
    pre = calendar.state("XNYS", pre_start)
    assert (pre.phase, pre.note, pre.since, pre.until) == ("pre", "Vorbörse", pre_start, NY_OPEN)
    assert (pre.next_open, pre.next_close) == (NY_OPEN, NY_CLOSE)
    assert calendar.state("XNYS", NY_OPEN - ONE_SECOND).phase == "pre"
    regular = calendar.state("XNYS", NY_OPEN)
    assert (regular.phase, regular.note, regular.since, regular.until) == ("open", "Handel", NY_OPEN, NY_CLOSE)
    assert (regular.next_open, regular.next_close) == (next_open, NY_CLOSE)
    post = calendar.state("XNYS", NY_CLOSE)
    assert (post.phase, post.note, post.since, post.until) == ("post", "Nachbörse", NY_CLOSE, post_end)
    assert (post.next_open, post.next_close) == (next_open, next_close)
    late = calendar.state("XNYS", post_end)
    assert (late.phase, late.since, late.until) == ("closed", post_end, local(NEW_YORK, 2026, 11, 25, 4))


def test_state_tokyo_lunch_break(calendar: ExchangeCalendar) -> None:
    morning = calendar.state("XTKS", TK_LUNCH - ONE_SECOND)
    assert (morning.phase, morning.since, morning.until) == ("open", TK_OPEN, TK_LUNCH)
    assert (morning.next_open, morning.next_close) == (TK_AFTERNOON, TK_LUNCH)
    lunch = calendar.state("XTKS", TK_LUNCH)
    assert (lunch.phase, lunch.note, lunch.is_open) == ("break", "Mittagspause", False)
    assert (lunch.since, lunch.until) == (TK_LUNCH, TK_AFTERNOON)
    assert (lunch.next_open, lunch.next_close) == (TK_AFTERNOON, TK_CLOSE)
    afternoon = calendar.state("XTKS", TK_AFTERNOON)
    assert (afternoon.phase, afternoon.since, afternoon.until) == ("open", TK_AFTERNOON, TK_CLOSE)
    assert afternoon.next_open == utc(2026, 11, 25, 0)
    evening = calendar.state("XTKS", TK_CLOSE)
    assert (evening.phase, evening.note) == ("closed", "Geschlossen")
    assert (evening.since, evening.until) == (TK_CLOSE, utc(2026, 11, 25))
    assert calendar.state("XTKS", TK_CLOSE).next_close == utc(2026, 11, 25, 2, 30)


def test_state_on_the_weekend(calendar: ExchangeCalendar) -> None:
    saturday = calendar.state("XETR", utc(2026, 11, 21, 12))
    assert (saturday.phase, saturday.note) == ("closed", "Wochenende")
    assert (saturday.since, saturday.until) == (utc(2026, 11, 20, 16, 30), utc(2026, 11, 23, 8))
    assert saturday.next_close == utc(2026, 11, 23, 16, 30)
    friday_night = calendar.state("XNYS", local(NEW_YORK, 2026, 11, 20, 21))
    assert (friday_night.phase, friday_night.note) == ("closed", "Geschlossen")
    assert friday_night.until == local(NEW_YORK, 2026, 11, 23, 4)  # Monday's pre-market
    assert calendar.state("XNYS", local(NEW_YORK, 2026, 11, 22, 12)).note == "Wochenende"


# ------------------------------------------------------------------------------------------------
# segments
# ------------------------------------------------------------------------------------------------
def test_segments_are_clipped_and_ascending(calendar: ExchangeCalendar) -> None:
    result = calendar.segments("XTKS", utc(2026, 11, 24, 1), utc(2026, 11, 25, 1))
    assert spans(result) == [
        (utc(2026, 11, 24, 1), TK_LUNCH),
        (TK_AFTERNOON, TK_CLOSE),
        (utc(2026, 11, 25), utc(2026, 11, 25, 1)),
    ]
    assert all(segment.kind == "regular" and segment.exchange == "XTKS" for segment in result)


def test_segments_are_half_open(calendar: ExchangeCalendar) -> None:
    assert calendar.segments("XNYS", NY_CLOSE, NY_CLOSE + timedelta(hours=1)) == []
    assert calendar.segments("XNYS", NY_OPEN - timedelta(hours=1), NY_OPEN) == []
    assert spans(calendar.segments("XNYS", NY_CLOSE - ONE_SECOND, NY_CLOSE)) == [(NY_CLOSE - ONE_SECOND, NY_CLOSE)]


def test_segments_with_an_empty_or_inverted_range(calendar: ExchangeCalendar) -> None:
    assert calendar.segments("XNYS", NY_OPEN, NY_OPEN) == []
    assert calendar.segments("XNYS", NY_CLOSE, NY_OPEN) == []


def test_segments_extended_adds_pre_and_post_windows(calendar: ExchangeCalendar) -> None:
    result = calendar.segments("XNYS", utc(2026, 11, 24), utc(2026, 11, 25), extended=True)
    assert [(segment.kind, segment.start, segment.end) for segment in result] == [
        ("post", utc(2026, 11, 24), local(NEW_YORK, 2026, 11, 23, 20)),
        ("pre", local(NEW_YORK, 2026, 11, 24, 4), NY_OPEN),
        ("regular", NY_OPEN, NY_CLOSE),
        ("post", NY_CLOSE, utc(2026, 11, 25)),
    ]
    assert calendar.segments("XETR", utc(2026, 11, 24), utc(2026, 11, 25), extended=True) == calendar.segments(
        "XETR", utc(2026, 11, 24), utc(2026, 11, 25)
    )


def test_segments_over_a_week_skip_weekend_and_holiday(calendar: ExchangeCalendar) -> None:
    result = calendar.segments("XNYS", utc(2026, 11, 23), utc(2026, 11, 30))
    assert [calendar.session_date("XNYS", segment.start) for segment in result] == [
        date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25), date(2026, 11, 27)
    ]


# ------------------------------------------------------------------------------------------------
# session_date / last_completed_session
# ------------------------------------------------------------------------------------------------
def test_session_date_is_the_local_calendar_date(calendar: ExchangeCalendar) -> None:
    assert calendar.session_date("XNYS", utc(2026, 11, 25, 3)) == date(2026, 11, 24)  # 22:00 in New York
    assert calendar.session_date("XTKS", utc(2026, 11, 23, 23)) == date(2026, 11, 24)  # 08:00 in Tokyo
    assert calendar.session_date("XETR", utc(2026, 11, 21, 12)) == date(2026, 11, 21)  # on a Saturday, too


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        (NY_OPEN - timedelta(hours=1), date(2026, 11, 23)),  # before the open
        (NY_OPEN + timedelta(hours=2), date(2026, 11, 23)),  # during the session
        (NY_CLOSE - ONE_SECOND, date(2026, 11, 23)),
        (NY_CLOSE, date(2026, 11, 24)),  # right at the close
        (NY_CLOSE + ONE_SECOND, date(2026, 11, 24)),
        (utc(2026, 11, 22, 12), date(2026, 11, 20)),  # Sunday -> Friday
        (utc(2026, 11, 23, 14), date(2026, 11, 20)),  # Monday before the open -> Friday
        (local(NEW_YORK, 2026, 11, 27, 10), date(2026, 11, 25)),  # skips Thanksgiving
        (local(NEW_YORK, 2026, 11, 27, 13), date(2026, 11, 27)),  # early close at 13:00
    ],
)
def test_last_completed_session_new_york(calendar: ExchangeCalendar, at, expected: date) -> None:
    assert calendar.last_completed_session("XNYS", at) == expected


def test_last_completed_session_waits_for_the_afternoon_segment(calendar: ExchangeCalendar) -> None:
    assert calendar.last_completed_session("XTKS", TK_LUNCH) == date(2026, 11, 20)  # Monday 11-23 was a holiday
    assert calendar.last_completed_session("XTKS", TK_CLOSE) == date(2026, 11, 24)


# ------------------------------------------------------------------------------------------------
# minutes_since_open / minutes_to_close
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("at", "since", "to_close"),
    [
        (NY_OPEN, 0, 390),
        (NY_OPEN + timedelta(minutes=1, seconds=30), 1, 388),
        (NY_CLOSE - ONE_SECOND, 389, 0),
        (NY_OPEN - ONE_SECOND, None, None),
        (NY_CLOSE, None, None),
    ],
)
def test_minutes_new_york(calendar: ExchangeCalendar, at, since: int | None, to_close: int | None) -> None:
    assert calendar.minutes_since_open("XNYS", at) == since
    assert calendar.minutes_to_close("XNYS", at) == to_close


@pytest.mark.parametrize(
    ("at", "since", "to_close"),
    [
        (TK_OPEN, 0, 330),
        (TK_LUNCH - timedelta(minutes=1), 149, 181),
        (TK_LUNCH, None, None),  # lunch break
        (TK_AFTERNOON - ONE_SECOND, None, None),
        (TK_AFTERNOON, 150, 180),  # the lunch break does not count
        (TK_CLOSE - timedelta(minutes=1), 329, 1),
    ],
)
def test_minutes_tokyo_exclude_the_lunch_break(calendar: ExchangeCalendar, at, since, to_close) -> None:
    assert calendar.minutes_since_open("XTKS", at) == since
    assert calendar.minutes_to_close("XTKS", at) == to_close


# ------------------------------------------------------------------------------------------------
# add_trading_minutes
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("key", "start", "minutes", "expected"),
    [
        # across the lunch break: 11:20 + 20 trading minutes = 12:40 Tokyo
        ("XTKS", local(TOKYO, 2026, 11, 24, 11, 20), 20, local(TOKYO, 2026, 11, 24, 12, 40)),
        # ending exactly with the morning segment: its end, not the afternoon open
        ("XTKS", local(TOKYO, 2026, 11, 24, 11), 30, local(TOKYO, 2026, 11, 24, 11, 30)),
        # started during the lunch break: counting begins at 12:30
        ("XTKS", local(TOKYO, 2026, 11, 24, 12), 10, local(TOKYO, 2026, 11, 24, 12, 40)),
        # overnight
        ("XNYS", local(NEW_YORK, 2026, 11, 24, 15, 50), 20, local(NEW_YORK, 2026, 11, 25, 9, 40)),
        # over the weekend
        ("XNYS", local(NEW_YORK, 2026, 11, 20, 15, 50), 20, local(NEW_YORK, 2026, 11, 23, 9, 40)),
        # over Thanksgiving, then over the early close of the day after
        ("XNYS", local(NEW_YORK, 2026, 11, 25, 15, 50), 20, local(NEW_YORK, 2026, 11, 27, 9, 40)),
        ("XNYS", local(NEW_YORK, 2026, 11, 27, 12, 50), 20, local(NEW_YORK, 2026, 11, 30, 9, 40)),
        # over the Tokyo holiday on Monday 2026-11-23
        ("XTKS", local(TOKYO, 2026, 11, 20, 15, 20), 20, local(TOKYO, 2026, 11, 24, 9, 10)),
        # start outside a session (Saturday): counting begins at Monday's open
        ("XETR", utc(2026, 11, 21, 12), 1, utc(2026, 11, 23, 8, 1)),
        # one full week of New York sessions from Monday's open ends at Friday's close
        ("XNYS", utc(2026, 11, 16, 14, 30), 5 * 390, utc(2026, 11, 20, 21)),
    ],
)
def test_add_trading_minutes(calendar: ExchangeCalendar, key: str, start, minutes: int, expected) -> None:
    assert calendar.add_trading_minutes(key, start, minutes) == expected


def test_add_zero_minutes_returns_start_even_outside_a_session(calendar: ExchangeCalendar) -> None:
    saturday = utc(2026, 11, 21, 12, 0, 7)
    assert calendar.add_trading_minutes("XNYS", saturday, 0) == saturday


def test_add_trading_minutes_keeps_sub_second_precision(calendar: ExchangeCalendar) -> None:
    start = NY_OPEN + timedelta(microseconds=500_000)
    assert calendar.add_trading_minutes("XNYS", start, 1) == start + timedelta(minutes=1)
    near_close = NY_CLOSE - timedelta(minutes=1) + timedelta(microseconds=500_000)
    next_open = utc(2026, 11, 25, 14, 30)
    assert calendar.add_trading_minutes("XNYS", near_close, 1) == next_open + timedelta(microseconds=500_000)


def test_add_negative_minutes_is_rejected(calendar: ExchangeCalendar) -> None:
    with pytest.raises(ValueError, match="negative"):
        calendar.add_trading_minutes("XNYS", NY_OPEN, -1)


def test_add_many_minutes_extends_the_schedule_lazily() -> None:
    fresh = ExchangeCalendar()
    start = utc(2026, 11, 24, 14, 30)
    later = fresh.add_trading_minutes("XNYS", start, 390 * 600)  # roughly 2.4 years of sessions
    assert later.year == 2029
    assert fresh.is_open("XNYS", later - ONE_SECOND)
    halfway = fresh.add_trading_minutes("XNYS", start, 390 * 300)
    assert fresh.add_trading_minutes("XNYS", halfway, 390 * 300) == later
