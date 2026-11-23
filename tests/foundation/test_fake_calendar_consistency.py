"""Review tests for ``FakeCalendar``: its methods must tell ONE consistent story.

Other modules are tested against this double, each through a different method (``is_open``, ``segments``,
``state``, ``last_completed_session``, ``add_trading_minutes`` ...). The tests here cross-check the
methods against each other and against the ``MarketCalendar`` protocol.
"""

import inspect
import random
from datetime import UTC, date, datetime, timedelta

import pytest

from trading_local.contracts import MarketCalendar
from trading_local.models import SessionSegment
from trading_local.testing import ALWAYS_OPEN_EXCHANGES, DEFAULT_TEST_WINDOWS, FakeCalendar

MINUTE = timedelta(minutes=1)
DAY = timedelta(days=1)
MONDAY = datetime(2026, 11, 16, tzinfo=UTC)
HOLIDAYS = {date(2026, 11, 19), date(2026, 11, 27)}  # a Thursday and the Friday of the following week
VENUES = ["XETR", "XNAS", "XTKS"]  # every distinct shape of DEFAULT_TEST_WINDOWS (XLON = XETR, XNYS = XNAS)


def calendar_for(exchange: str) -> FakeCalendar:
    return FakeCalendar(holidays={exchange: HOLIDAYS})


def trading_time(calendar: FakeCalendar, exchange: str, start: datetime, end: datetime) -> timedelta:
    """Time inside regular segments between two instants, taken from ``segments``."""
    return sum((segment.end - segment.start for segment in calendar.segments(exchange, start, end)), timedelta())


def day_segments(calendar: FakeCalendar, exchange: str, day: date) -> list[SessionSegment]:
    midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return calendar.segments(exchange, midnight, midnight + DAY)


def test_fake_calendar_has_the_signatures_of_the_protocol() -> None:
    protocol_methods = {
        name: member
        for name, member in inspect.getmembers(MarketCalendar, inspect.isfunction)
        if not name.startswith("_")
    }
    assert sorted(protocol_methods) == [
        "add_trading_minutes",
        "display_name",
        "exchanges",
        "is_open",
        "last_completed_session",
        "minutes_since_open",
        "minutes_to_close",
        "segments",
        "session_date",
        "state",
        "timezone",
    ]

    for name, declared in protocol_methods.items():
        expected = inspect.signature(declared)
        actual = inspect.signature(getattr(FakeCalendar, name))
        described = [(p.name, p.kind, p.default) for p in actual.parameters.values()]
        assert described == [(p.name, p.kind, p.default) for p in expected.parameters.values()], name
        assert actual.return_annotation == expected.return_annotation, name


@pytest.mark.parametrize("exchange", VENUES)
def test_is_open_state_and_segments_agree_minute_by_minute(exchange: str) -> None:
    calendar = calendar_for(exchange)
    end = MONDAY + 14 * DAY
    segments = calendar.segments(exchange, MONDAY, end)
    open_minutes = {
        segment.start + n * MINUTE for segment in segments for n in range((segment.end - segment.start) // MINUTE)
    }
    assert all(first.end < second.start for first, second in zip(segments, segments[1:]))  # ascending, separated

    for n in range(14 * 1440):
        moment = MONDAY + n * MINUTE
        expected = moment in open_minutes
        assert calendar.is_open(exchange, moment) is expected, moment
        assert calendar.is_open(exchange, moment + timedelta(seconds=59)) is expected, moment
    for moment in (MONDAY + n * timedelta(minutes=30) for n in range(14 * 48)):
        state = calendar.state(exchange, moment)
        assert state.is_open is (moment in open_minutes) and (state.phase == "open") is state.is_open
        # the phase run reported by state() contains no boundary of a regular segment
        assert trading_time(calendar, exchange, state.since, state.until) == (
            state.until - state.since if state.is_open else timedelta()
        )
        # next_open / next_close are the next segment boundaries listed by segments()
        around = calendar.segments(exchange, moment - DAY, moment + 7 * DAY)
        following = next(segment for segment in around if segment.start > moment)
        assert state.next_open == following.start
        assert state.next_close == (state.until if state.is_open else following.end)


@pytest.mark.parametrize("exchange", VENUES)
def test_minutes_since_open_and_to_close_add_up_to_the_session_length(exchange: str) -> None:
    calendar = calendar_for(exchange)
    session_minutes = sum(end - start for start, end in DEFAULT_TEST_WINDOWS[exchange])

    for n in range(0, 14 * 1440, 7):
        moment = MONDAY + n * MINUTE
        since_open = calendar.minutes_since_open(exchange, moment)
        to_close = calendar.minutes_to_close(exchange, moment)

        if calendar.is_open(exchange, moment):
            assert since_open is not None and to_close is not None
            assert since_open + to_close == session_minutes
            midnight = moment.replace(hour=0, minute=0)
            assert trading_time(calendar, exchange, midnight, moment) == since_open * MINUTE
            assert trading_time(calendar, exchange, moment, midnight + DAY) == to_close * MINUTE
        else:
            assert since_open is None and to_close is None


@pytest.mark.parametrize("exchange", VENUES)
def test_last_completed_session_agrees_with_the_segments(exchange: str) -> None:
    calendar = calendar_for(exchange)
    previous: date | None = None

    for n in range(0, 14 * 1440, 15):
        for moment in (MONDAY + n * MINUTE - timedelta(seconds=1), MONDAY + n * MINUTE):
            completed = calendar.last_completed_session(exchange, moment)

            assert completed is not None
            session = day_segments(calendar, exchange, completed)
            assert session and session[-1].end <= moment  # a trading day whose LAST segment has ended
            # ... and the newest such day: every later day up to today is no trading day or still running
            day = completed + DAY
            while day <= calendar.session_date(exchange, moment):
                later = day_segments(calendar, exchange, day)
                assert not later or later[-1].end > moment, (moment, day)
                day += DAY
            assert completed not in HOLIDAYS and completed.weekday() < 5
            assert previous is None or completed >= previous  # never goes back in time
            previous = completed


@pytest.mark.parametrize("exchange", [*VENUES, *ALWAYS_OPEN_EXCHANGES])
def test_add_trading_minutes_is_additive_and_measures_trading_time(exchange: str) -> None:
    calendar = calendar_for(exchange) if exchange in DEFAULT_TEST_WINDOWS else FakeCalendar()
    rng = random.Random(len(exchange) * 31 + ord(exchange[1]))

    for _ in range(60):
        start = MONDAY + timedelta(seconds=rng.randrange(0, 12 * 86_400))
        first, second = rng.randrange(1, 700), rng.randrange(1, 700)

        middle = calendar.add_trading_minutes(exchange, start, first)
        end = calendar.add_trading_minutes(exchange, middle, second)

        assert end == calendar.add_trading_minutes(exchange, start, first + second)
        assert start < middle < end
        assert trading_time(calendar, exchange, start, middle) == first * MINUTE
        assert trading_time(calendar, exchange, middle, end) == second * MINUTE
        # the result is the EARLIEST instant with that much trading time: the venue traded right up to it
        assert calendar.is_open(exchange, middle - timedelta(microseconds=1))
        assert calendar.is_open(exchange, end - timedelta(microseconds=1))


def test_always_open_venues_tell_the_same_story_through_every_method() -> None:
    calendar = FakeCalendar()

    for exchange in ALWAYS_OPEN_EXCHANGES:
        for n in range(0, 7 * 1440, 97):
            moment = MONDAY + n * MINUTE + timedelta(seconds=n % 60)
            assert calendar.is_open(exchange, moment)
            assert calendar.state(exchange, moment).is_open
            assert calendar.segments(exchange, moment, moment + DAY)[0].start == moment
            since_open = calendar.minutes_since_open(exchange, moment)
            to_close = calendar.minutes_to_close(exchange, moment)
            assert since_open is not None and to_close is not None
            assert since_open == n % 1440
            assert since_open + to_close in (1439, 1440)  # whole minutes on both sides of `moment`
            assert calendar.last_completed_session(exchange, moment) == calendar.session_date(exchange, moment) - DAY
            assert calendar.add_trading_minutes(exchange, moment, n) == moment + n * MINUTE
