"""Thread safety, lazy window extension, speed and the interplay with execution.session_flatten_ts."""

import threading
import time
from datetime import date, timedelta

from tests.markets.helpers import NEW_YORK, ONE_SECOND, local, utc
from trading_local.clock import to_ts
from trading_local.execution import session_flatten_ts
from trading_local.markets import ExchangeCalendar

THREAD_VENUES = ("XNYS", "XNAS", "XETR", "XSHG", "XSHE", "FX", "CME", "CRYPTO")
THREAD_TIMES = [utc(2026, 11, 16) + timedelta(minutes=37 * step) for step in range(300)]


def snapshot(calendar: ExchangeCalendar) -> list:
    return [
        (calendar.state(key, at), calendar.minutes_to_close(key, at), calendar.last_completed_session(key, at))
        for key in THREAD_VENUES
        for at in THREAD_TIMES
    ]


def test_concurrent_first_use_from_many_threads() -> None:
    """Eight threads hit a fresh calendar at once (lazy builds race) and all see the same answers."""
    fresh = ExchangeCalendar()
    barrier = threading.Barrier(8)
    results: list[list] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            results.append(snapshot(fresh))
        except BaseException as exc:  # collected and re-raised by the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert len(results) == 8
    reference = snapshot(ExchangeCalendar())
    assert all(result == reference for result in results)


def test_window_extends_lazily_in_both_directions() -> None:
    fresh = ExchangeCalendar()
    assert fresh.is_open("XNYS", utc(2026, 11, 20, 15))
    assert fresh.is_open("XNYS", local(NEW_YORK, 2033, 3, 1, 10))  # beyond the first window
    assert fresh.last_completed_session("XNYS", local(NEW_YORK, 2013, 3, 1, 10)) == date(2013, 2, 28)
    assert not fresh.is_open("XNYS", local(NEW_YORK, 2015, 12, 25, 10))  # Christmas, long before the first window
    assert fresh.is_open("XNYS", utc(2026, 11, 20, 15))  # the old range still answers


def test_is_open_is_fast_after_warm_up(calendar: ExchangeCalendar) -> None:
    at = utc(2026, 11, 20, 15)
    calendar.is_open("XNYS", at)
    calls = 20_000
    started = time.perf_counter()
    for step in range(calls):
        calendar.is_open("XNYS", at + timedelta(seconds=step))
    microseconds = (time.perf_counter() - started) / calls * 1e6
    assert microseconds < 50, f"is_open took {microseconds:.1f} us per call"


def test_state_of_all_venues_is_fast_after_warm_up(calendar: ExchangeCalendar) -> None:
    at = utc(2026, 11, 20, 15)
    keys = calendar.exchanges()
    for key in keys:
        calendar.state(key, at)
    started = time.perf_counter()
    for _ in range(100):
        for key in keys:
            calendar.state(key, at)
    microseconds = (time.perf_counter() - started) / (100 * len(keys)) * 1e6
    assert microseconds < 100, f"state took {microseconds:.1f} us per call"


def test_session_flatten_ts_uses_the_session_close(calendar: ExchangeCalendar) -> None:
    entry = utc(2026, 11, 24, 15)
    assert session_flatten_ts(calendar, "XNYS", entry, 5) == to_ts(utc(2026, 11, 24, 21) - timedelta(minutes=5))
    early = local(NEW_YORK, 2026, 11, 27, 10)
    assert session_flatten_ts(calendar, "XNYS", early, 5) == to_ts(local(NEW_YORK, 2026, 11, 27, 12, 55))
    tokyo_morning = utc(2026, 11, 24, 1)  # the afternoon segment belongs to the same trading day
    assert session_flatten_ts(calendar, "XTKS", tokyo_morning, 5) == to_ts(utc(2026, 11, 24, 6, 25))
    before_midnight = to_ts(utc(2026, 11, 25) - timedelta(minutes=5))  # 24h venues: the UTC day ends
    assert session_flatten_ts(calendar, "CRYPTO", utc(2026, 11, 24, 15), 5) == before_midnight
    assert session_flatten_ts(calendar, "FX", utc(2026, 11, 24, 15), 5) == before_midnight


def test_minute_by_minute_replay_matches_segments(calendar: ExchangeCalendar) -> None:
    """is_open sampled every minute of a week agrees with the segment list (no gaps, no overlaps)."""
    start, end = utc(2026, 11, 23), utc(2026, 11, 30)
    for key in ("XNYS", "XTKS", "XETR", "CME"):
        open_minutes = {
            to_ts(start + timedelta(minutes=step))
            for step in range(7 * 24 * 60)
            if calendar.is_open(key, start + timedelta(minutes=step))
        }
        from_segments = {
            ts
            for segment in calendar.segments(key, start, end)
            for ts in range(to_ts(segment.start), to_ts(segment.end), 60)
        }
        assert open_minutes == from_segments, key
        assert all(calendar.is_open(key, segment.end - ONE_SECOND) for segment in calendar.segments(key, start, end))
