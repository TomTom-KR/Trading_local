"""Shared helpers for the test suites of all modules (not used at runtime).

Import from here instead of copying fixtures between test packages:

    from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_bars, make_settings, seed_bars
"""

from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from trading_local.clock import floor_ts
from trading_local.config import Settings
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.models import Instrument, SessionSegment, SessionState

# A small, representative universe: real-time and delayed feeds, a pence-quoted share,
# an Asian share with a lunch break, a 24/7 instrument, indices and the FX pairs needed
# to convert all of them into EUR.
SAMPLE_INSTRUMENTS: list[Instrument] = [
    Instrument("NVDA", "NVIDIA", "equity", "US", "XNAS", "USD", timezone="America/New_York",
               benchmark="^GSPC", aliases=("Nvidia", "NVIDIA"), sort_order=1),
    Instrument("SAP.DE", "SAP", "equity", "EU", "XETR", "EUR", timezone="Europe/Berlin", delay_min=15,
               benchmark="^GDAXI", aliases=("SAP",), sort_order=2),
    Instrument("SHEL.L", "Shell", "equity", "EU", "XLON", "GBP", price_scale=0.01, timezone="Europe/London",
               delay_min=15, benchmark="^GDAXI", aliases=("Shell",), sort_order=3),
    Instrument("7203.T", "Toyota Motor", "equity", "ASIA", "XTKS", "JPY", timezone="Asia/Tokyo", delay_min=20,
               benchmark=None, aliases=("Toyota",), sort_order=4),
    Instrument("BTC-USD", "Bitcoin", "crypto", "GLOBAL", "CRYPTO", "USD", timezone="UTC",
               aliases=("Bitcoin", "BTC"), sort_order=5),
    Instrument("^GDAXI", "DAX", "index", "EU", "XETR", "EUR", timezone="Europe/Berlin", delay_min=15,
               aliases=("DAX",), sort_order=6),
    Instrument("^GSPC", "S&P 500", "index", "US", "XNYS", "USD", timezone="America/New_York",
               aliases=("S&P 500",), sort_order=7),
    Instrument("EURUSD=X", "EUR/USD", "fx", "GLOBAL", "FX", "USD", timezone="UTC", tradable=False, sort_order=8),
    Instrument("EURGBP=X", "EUR/GBP", "fx", "GLOBAL", "FX", "GBP", timezone="UTC", tradable=False, sort_order=9),
    Instrument("EURJPY=X", "EUR/JPY", "fx", "GLOBAL", "FX", "JPY", timezone="UTC", tradable=False, sort_order=10),
]


def make_settings(data_dir: Path, **overrides: Any) -> Settings:
    """Settings for tests: no .env file, data in ``data_dir``, mails only written to files, no network features."""
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "mail_mode": "file",
        "mail_to": "empfaenger@example.invalid",
        "mail_from": "absender@example.invalid",
        "smtp_user": "",
        "open_browser": False,
        "offline": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


# ------------------------------------------------------------------------------------------------
# Market data
# ------------------------------------------------------------------------------------------------
def make_bars(
    start_ts: int,
    minutes: int,
    *,
    start_price: float = 100.0,
    seed: int = 1,
    drift: float = 0.0,
    vol: float = 0.0008,
    base_volume: float = 1000.0,
) -> pd.DataFrame:
    """Deterministic random-walk 1-minute bars in the canonical format.

    ``minutes`` consecutive bars, the first one opening at ``start_ts`` floored to a full minute.
    ``drift`` and ``vol`` are the mean and the standard deviation of the log return per bar. The first
    bar opens at ``start_price``, every other bar at the previous close; high and low enclose open and
    close; all prices are positive and the volume varies around ``base_volume``. The same arguments
    always produce the same frame.
    """
    if minutes < 0:
        raise ValueError(f"minutes must not be negative, got {minutes}")
    if start_price <= 0 or vol < 0 or base_volume < 0:
        raise ValueError("start_price must be positive, vol and base_volume must not be negative")
    rng = np.random.default_rng(seed)
    close = start_price * np.exp(np.cumsum(drift + vol * rng.standard_normal(minutes)))
    open_ = np.concatenate(([start_price], close))[:minutes]
    high = np.maximum(open_, close) * np.exp(vol / 2 * np.abs(rng.standard_normal(minutes)))
    low = np.minimum(open_, close) * np.exp(-vol / 2 * np.abs(rng.standard_normal(minutes)))
    volume = base_volume * rng.uniform(0.5, 1.5, minutes)
    ts = floor_ts(start_ts) + repo_market.BAR_SECONDS * np.arange(minutes, dtype=np.int64)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=pd.Index(ts, name="ts")
    )


def seed_instruments(db: Database, instruments: Iterable[Instrument] = SAMPLE_INSTRUMENTS) -> int:
    """Store instruments exactly as given (including their active/tradable flags); returns the row count."""
    return repo_market.upsert_instruments(db, instruments, keep_user_flags=False)


def seed_bars(db: Database, symbol: str, frame: pd.DataFrame, source: str = "rest") -> int:
    """Store a canonical 1-minute frame (see ``make_bars``); returns the number of rows written."""
    return repo_market.upsert_bars_1m(db, symbol, frame, source)


# ------------------------------------------------------------------------------------------------
# Trading calendar
# ------------------------------------------------------------------------------------------------
# Regular sessions of the venues used by SAMPLE_INSTRUMENTS as (start, end) minute of the UTC day.
DEFAULT_TEST_WINDOWS: Mapping[str, tuple[tuple[int, int], ...]] = MappingProxyType(
    {
        "XNAS": ((870, 1260),),  # 14:30-21:00 UTC
        "XNYS": ((870, 1260),),  # 14:30-21:00 UTC
        "XETR": ((480, 990),),  # 08:00-16:30 UTC
        "XLON": ((480, 990),),  # 08:00-16:30 UTC
        "XTKS": ((0, 150), (210, 390)),  # 00:00-02:30 and 03:30-06:30 UTC (lunch break)
    }
)
# Venues that FakeCalendar lists in ``exchanges()`` although they have no windows (open around the clock).
ALWAYS_OPEN_EXCHANGES: tuple[str, ...] = ("CRYPTO", "FX", "KRAKEN")

_DISPLAY_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "XNAS": "Nasdaq",
        "XNYS": "New York",
        "XETR": "Xetra",
        "XLON": "London",
        "XTKS": "Tokio",
        "CRYPTO": "Krypto",
        "FX": "Devisen",
        "KRAKEN": "Kraken",
    }
)
_MINUTES_PER_DAY = 24 * 60
_ONE_DAY = timedelta(days=1)
_ONE_MINUTE = timedelta(minutes=1)

type _Segment = tuple[datetime, datetime]  # one regular trading window [start, end) in UTC


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("naive datetime: all datetimes in this code base must be timezone-aware")
    return value.astimezone(UTC)


def _midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _checked_windows(exchange: str, windows: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Validate the daily windows of one exchange: inside the day, ascending and separated by a gap."""
    checked = tuple((int(start), int(end)) for start, end in windows)
    if not checked:
        raise ValueError(f"{exchange}: needs at least one window (leave an always-open exchange out of `windows`)")
    previous_end = -1
    for start, end in checked:
        if not (previous_end < start < end <= _MINUTES_PER_DAY):
            raise ValueError(f"{exchange}: windows must be ascending, separated minute-of-day pairs, got {checked}")
        previous_end = end
    return checked


def _checked_holidays(exchange: str, days: Collection[date]) -> frozenset[date]:
    if not all(type(day) is date for day in days):
        raise ValueError(f"{exchange}: holidays must be datetime.date objects")
    return frozenset(days)


class FakeCalendar:
    """Deterministic ``MarketCalendar`` for tests: fixed UTC trading windows instead of real exchange calendars.

    ``windows`` maps an exchange key to its regular segments of every trading day as
    ``(start_minute, end_minute)`` of the UTC day; two pairs describe a day with a lunch break. The
    default is ``DEFAULT_TEST_WINDOWS``. An exchange with windows trades on every day that is not one of
    its ``holidays`` and, with ``weekends_closed``, not a Saturday or Sunday (UTC).

    Every exchange WITHOUT an entry in ``windows`` (like 'CRYPTO' and 'FX' by default) is open around the
    clock on every day; unlike a real calendar the double therefore never raises ``KeyError``.

    Conventions
    * A segment is the half-open interval [start, end): the venue is open at its start and no longer
      open at its end. Windows that touch midnight on consecutive days are separate, adjacent sessions.
    * The trading date is the UTC date. A session is completed once its last segment has ended; the
      "session" of an always-open venue is the UTC day, which is completed at the next midnight.
    * ``minutes_since_open`` / ``minutes_to_close`` count whole minutes (rounded down).
    * All datetimes must be timezone-aware; results are UTC.
    """

    def __init__(
        self,
        windows: Mapping[str, Sequence[tuple[int, int]]] | None = None,
        *,
        weekends_closed: bool = True,
        holidays: Mapping[str, Collection[date]] | None = None,
    ) -> None:
        source = DEFAULT_TEST_WINDOWS if windows is None else windows
        self._windows = {exchange: _checked_windows(exchange, pairs) for exchange, pairs in source.items()}
        self._weekends_closed = weekends_closed
        self._holidays = {exchange: _checked_holidays(exchange, days) for exchange, days in (holidays or {}).items()}
        unused = sorted(set(self._holidays) - set(self._windows))
        if unused:
            raise ValueError(f"holidays given for exchanges without trading windows (always open): {unused}")

    # -- description ----------------------------------------------------------------------------
    def exchanges(self) -> list[str]:
        return sorted({*self._windows, *ALWAYS_OPEN_EXCHANGES})

    def display_name(self, exchange: str) -> str:
        return _DISPLAY_NAMES.get(exchange, exchange)

    def timezone(self, exchange: str) -> str:
        return "UTC"

    # -- sessions -------------------------------------------------------------------------------
    def state(self, exchange: str, at: datetime) -> SessionState:
        at = _as_utc(at)
        if exchange not in self._windows:
            return SessionState(
                exchange, "open", is_open=True, since=None, until=None, next_open=None, next_close=None, note="Handel"
            )
        upcoming = self._next_segment(exchange, at)
        current = self._current_segment(exchange, at)
        if current is not None:
            return SessionState(
                exchange,
                "open",
                is_open=True,
                since=current[0],
                until=current[1],
                next_open=upcoming[0],
                next_close=current[1],
                note="Handel",
            )
        previous = self._previous_segment(exchange, at)
        in_break = previous[0].date() == upcoming[0].date()  # between two segments of the same trading day
        return SessionState(
            exchange,
            "break" if in_break else "closed",
            is_open=False,
            since=previous[1],
            until=upcoming[0],
            next_open=upcoming[0],
            next_close=upcoming[1],
            note="Mittagspause" if in_break else self._closed_note(exchange, at.date()),
        )

    def is_open(self, exchange: str, at: datetime) -> bool:
        at = _as_utc(at)
        return exchange not in self._windows or self._current_segment(exchange, at) is not None

    def segments(
        self, exchange: str, start: datetime, end: datetime, *, extended: bool = False
    ) -> list[SessionSegment]:
        """Regular segments clipped to ``[start, end)``; without pre/post market ``extended`` changes nothing."""
        start, end = _as_utc(start), _as_utc(end)
        if start >= end:
            return []
        if exchange not in self._windows:
            return [SessionSegment(exchange, start, end)]
        clipped: list[SessionSegment] = []
        for segment_start, segment_end in self._segments_from(exchange, start.date()):
            if segment_start >= end:
                break
            if segment_end > start:
                clipped.append(SessionSegment(exchange, max(segment_start, start), min(segment_end, end)))
        return clipped

    def session_date(self, exchange: str, at: datetime) -> date:
        return _as_utc(at).date()

    def last_completed_session(self, exchange: str, at: datetime) -> date | None:
        at = _as_utc(at)
        day = at.date()
        if exchange not in self._windows:
            return day - _ONE_DAY
        while True:
            day_segments = self._day_segments(exchange, day)
            if day_segments and day_segments[-1][1] <= at:
                return day
            day -= _ONE_DAY

    def minutes_since_open(self, exchange: str, at: datetime) -> int | None:
        at = _as_utc(at)
        if exchange not in self._windows:
            return (at - _midnight(at.date())) // _ONE_MINUTE
        if self._current_segment(exchange, at) is None:
            return None
        day_segments = self._day_segments(exchange, at.date())
        elapsed = sum((min(end, at) - start for start, end in day_segments if start <= at), timedelta())
        return elapsed // _ONE_MINUTE

    def minutes_to_close(self, exchange: str, at: datetime) -> int | None:
        at = _as_utc(at)
        if exchange not in self._windows:
            return (_midnight(at.date()) + _ONE_DAY - at) // _ONE_MINUTE
        if self._current_segment(exchange, at) is None:
            return None
        day_segments = self._day_segments(exchange, at.date())
        remaining = sum((end - max(start, at) for start, end in day_segments if end > at), timedelta())
        return remaining // _ONE_MINUTE

    def add_trading_minutes(self, exchange: str, start: datetime, minutes: int) -> datetime:
        """The instant at which ``minutes`` trading minutes have passed after ``start``.

        Time outside the regular segments does not count. Trading time that ends exactly with a segment
        yields the end of that segment, not the start of the next one; ``minutes=0`` returns ``start``.
        """
        if minutes < 0:
            raise ValueError(f"minutes must not be negative, got {minutes}")
        start = _as_utc(start)
        remaining = timedelta(minutes=minutes)
        if minutes == 0 or exchange not in self._windows:
            return start + remaining
        upcoming = self._segments_from(exchange, start.date())
        while True:
            segment_start, segment_end = next(upcoming)
            begin = max(segment_start, start)
            if begin >= segment_end:  # this segment had ended by `start`
                continue
            if remaining <= segment_end - begin:
                return begin + remaining
            remaining -= segment_end - begin

    # -- internals ------------------------------------------------------------------------------
    def _is_trading_day(self, exchange: str, day: date) -> bool:
        if self._weekends_closed and day.weekday() >= 5:
            return False
        return day not in self._holidays.get(exchange, ())

    def _closed_note(self, exchange: str, day: date) -> str:
        if self._weekends_closed and day.weekday() >= 5:
            return "Wochenende"
        if day in self._holidays.get(exchange, ()):
            return "Feiertag"
        return "Geschlossen"

    def _day_segments(self, exchange: str, day: date) -> list[_Segment]:
        """Regular segments of an exchange with windows on one UTC day (none on weekends and holidays)."""
        if not self._is_trading_day(exchange, day):
            return []
        midnight = _midnight(day)
        return [
            (midnight + timedelta(minutes=start), midnight + timedelta(minutes=end))
            for start, end in self._windows[exchange]
        ]

    # The two generators are endless. Searches in them terminate because every exchange has at least one
    # window and only finitely many holidays: a trading day always follows (and precedes) any day.
    def _segments_from(self, exchange: str, day: date) -> Iterator[_Segment]:
        """All segments from UTC day ``day`` on, ascending."""
        while True:
            yield from self._day_segments(exchange, day)
            day += _ONE_DAY

    def _segments_until(self, exchange: str, day: date) -> Iterator[_Segment]:
        """All segments up to and including UTC day ``day``, descending."""
        while True:
            yield from reversed(self._day_segments(exchange, day))
            day -= _ONE_DAY

    def _current_segment(self, exchange: str, at: datetime) -> _Segment | None:
        """The segment that contains ``at``."""
        return next((s for s in self._day_segments(exchange, at.date()) if s[0] <= at < s[1]), None)

    def _next_segment(self, exchange: str, at: datetime) -> _Segment:
        """The first segment that starts strictly after ``at``."""
        return next(s for s in self._segments_from(exchange, at.date()) if s[0] > at)

    def _previous_segment(self, exchange: str, at: datetime) -> _Segment:
        """The last segment that ended at or before ``at``."""
        return next(s for s in self._segments_until(exchange, at.date()) if s[1] <= at)
