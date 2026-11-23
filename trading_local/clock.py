"""Time source abstraction.

All components take a ``Clock`` instead of calling ``datetime.now()`` / ``time.time()`` directly.
Live operation uses ``SystemClock``; tests and the replay/validation engine use ``ManualClock``.
This is what makes it possible to run the exact same engine code on historical data without
look-ahead: the engine can only ever ask "what time is it?" and "what was known at that time?".
"""

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol, runtime_checkable


def from_ts(ts: int | float) -> datetime:
    """Epoch seconds -> timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=UTC)


def to_ts(dt: datetime) -> int:
    """Timezone-aware datetime -> epoch seconds (floored). Naive datetimes are rejected."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime: all datetimes in this code base must be timezone-aware")
    return int(dt.timestamp())


def floor_ts(ts: int | float, step: int = 60) -> int:
    """Floor an epoch timestamp to a multiple of ``step`` seconds (default: start of the minute)."""
    return int(ts) // step * step


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Current time as timezone-aware UTC datetime."""
        ...

    def ts(self) -> int:
        """Current time as epoch seconds (UTC)."""
        ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def ts(self) -> int:
        return int(time.time())


class ManualClock:
    """Clock that only moves when told to. Used by tests and by the replay engine."""

    def __init__(self, start: datetime | int | float) -> None:
        self._now = from_ts(start) if isinstance(start, (int, float)) else self._check(start)

    @staticmethod
    def _check(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("ManualClock needs a timezone-aware datetime")
        return dt.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def ts(self) -> int:
        return int(self._now.timestamp())

    def set(self, value: datetime | int | float) -> None:
        self._now = from_ts(value) if isinstance(value, (int, float)) else self._check(value)

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)
