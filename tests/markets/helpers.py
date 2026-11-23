"""Time helpers for the trading-calendar tests."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

ONE_SECOND = timedelta(seconds=1)
NEW_YORK = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")
LONDON = ZoneInfo("Europe/London")
TOKYO = ZoneInfo("Asia/Tokyo")
HONG_KONG = ZoneInfo("Asia/Hong_Kong")
SHANGHAI = ZoneInfo("Asia/Shanghai")
SEOUL = ZoneInfo("Asia/Seoul")
SYDNEY = ZoneInfo("Australia/Sydney")


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def local(
    zone: ZoneInfo, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """A wall-clock time in ``zone``, converted to UTC (the form the calendar returns)."""
    return datetime(year, month, day, hour, minute, second, tzinfo=zone).astimezone(UTC)


def spans(segments: list) -> list[tuple[datetime, datetime]]:
    """(start, end) pairs of SessionSegment objects."""
    return [(segment.start, segment.end) for segment in segments]
