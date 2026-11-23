"""Fill and exit rules of trading_local.execution (shared by tracker, broker and validation)."""

from datetime import UTC, date, datetime

import numpy as np
import pytest

from tests.analysis.test_no_lookahead import SydneyLikeCalendar
from trading_local import execution
from trading_local.clock import from_ts, to_ts
from trading_local.config import CostParams
from trading_local.execution import Exit, exit_on_bar
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
NVDA = INSTRUMENTS["NVDA"]  # US equity: 2 bps slippage
COSTS = CostParams()


def utc(day: int, hour: int, minute: int = 0) -> int:
    return to_ts(datetime(2026, 11, day, hour, minute, tzinfo=UTC))


# ------------------------------------------------------------------------------------------------
# Fill prices
# ------------------------------------------------------------------------------------------------
def test_entry_fill_price_charges_slippage_against_the_trader() -> None:
    assert execution.entry_fill_price(COSTS, NVDA, "LONG", 100.0) == pytest.approx(100.02)
    assert execution.entry_fill_price(COSTS, NVDA, "SHORT", 100.0) == pytest.approx(99.98)


def test_exit_fill_price_market_and_limit() -> None:
    assert execution.exit_fill_price(COSTS, NVDA, "LONG", Exit(100.0, "stop", True)) == pytest.approx(99.98)
    assert execution.exit_fill_price(COSTS, NVDA, "SHORT", Exit(100.0, "stop", True)) == pytest.approx(100.02)
    assert execution.exit_fill_price(COSTS, NVDA, "LONG", Exit(100.0, "target", False)) == 100.0
    assert execution.exit_fill_price(COSTS, NVDA, "SHORT", Exit(100.0, "target", False)) == 100.0


def test_drift_exceeded() -> None:
    assert not execution.drift_exceeded(100.0, 100.5, 0.5)
    assert execution.drift_exceeded(100.0, 100.51, 0.5)
    assert execution.drift_exceeded(100.0, 99.4, 0.5)
    assert not execution.drift_exceeded(100.0, 150.0, None)


# ------------------------------------------------------------------------------------------------
# exit_on_bar truth table
# ------------------------------------------------------------------------------------------------
# LONG with stop 95 and target 110; SHORT with stop 105 and target 90. Bars: (open, high, low).
LONG_CASES = [
    ("no touch", (100.0, 104.0, 96.0), None),
    ("stop touched", (100.0, 104.0, 94.0), Exit(95.0, "stop", True)),
    ("stop exactly", (100.0, 104.0, 95.0), Exit(95.0, "stop", True)),
    ("target touched", (100.0, 111.0, 96.0), Exit(110.0, "target", False)),
    ("target exactly", (100.0, 110.0, 96.0), Exit(110.0, "target", False)),
    ("both touched: stop wins", (100.0, 111.0, 94.0), Exit(95.0, "stop", True)),
    ("gap through stop", (93.0, 97.0, 92.0), Exit(93.0, "stop", True)),
    ("gap through target", (112.0, 113.0, 94.0), Exit(112.0, "target", False)),
    ("opens exactly at stop", (95.0, 99.0, 95.0), Exit(95.0, "stop", True)),
    ("opens exactly at target", (110.0, 111.0, 109.0), Exit(110.0, "target", False)),
    ("gap below the stop although the bar reaches the target", (94.0, 111.0, 93.0), Exit(94.0, "stop", True)),
    ("one cent inside both levels", (100.0, 109.99, 95.01), None),
]
SHORT_CASES = [
    ("no touch", (100.0, 104.0, 91.0), None),
    ("stop touched", (100.0, 106.0, 96.0), Exit(105.0, "stop", True)),
    ("target touched", (100.0, 104.0, 89.0), Exit(90.0, "target", False)),
    ("target exactly", (100.0, 104.0, 90.0), Exit(90.0, "target", False)),
    ("both touched: stop wins", (100.0, 106.0, 89.0), Exit(105.0, "stop", True)),
    ("gap through stop", (107.0, 108.0, 101.0), Exit(107.0, "stop", True)),
    ("gap through target", (88.0, 106.0, 87.0), Exit(88.0, "target", False)),
    ("stop exactly", (100.0, 105.0, 96.0), Exit(105.0, "stop", True)),
    ("opens exactly at stop", (105.0, 105.5, 101.0), Exit(105.0, "stop", True)),
    ("opens exactly at target", (90.0, 91.0, 89.0), Exit(90.0, "target", False)),
    ("gap above the stop although the bar reaches the target", (106.0, 107.0, 89.0), Exit(106.0, "stop", True)),
    ("one cent inside both levels", (100.0, 104.99, 90.01), None),
]


@pytest.mark.parametrize(("name", "bar", "expected"), LONG_CASES, ids=[case[0] for case in LONG_CASES])
def test_exit_on_bar_long(name: str, bar: tuple[float, float, float], expected: Exit | None) -> None:
    open_, high, low = bar
    got = exit_on_bar(side="LONG", bar_ts=0, bar_open=open_, bar_high=high, bar_low=low, stop=95.0, target=110.0)
    assert got == expected


@pytest.mark.parametrize(("name", "bar", "expected"), SHORT_CASES, ids=[case[0] for case in SHORT_CASES])
def test_exit_on_bar_short(name: str, bar: tuple[float, float, float], expected: Exit | None) -> None:
    open_, high, low = bar
    got = exit_on_bar(side="SHORT", bar_ts=0, bar_open=open_, bar_high=high, bar_low=low, stop=105.0, target=90.0)
    assert got == expected


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_time_stop_and_flatten_exit_at_the_open_and_take_precedence(side: str) -> None:
    common = {"side": side, "bar_open": 100.0, "bar_high": 120.0, "bar_low": 80.0, "stop": 95.0, "target": 110.0}
    if side == "SHORT":
        common.update(stop=105.0, target=90.0)
    assert exit_on_bar(bar_ts=600, time_stop_ts=600, **common) == Exit(100.0, "time_stop", True)
    assert exit_on_bar(bar_ts=600, flatten_ts=540, **common) == Exit(100.0, "session_close", True)
    assert exit_on_bar(bar_ts=600, time_stop_ts=600, flatten_ts=600, **common) == Exit(100.0, "time_stop", True)
    # one minute earlier neither is due and the wide bar hits the stop
    assert exit_on_bar(bar_ts=540, time_stop_ts=600, flatten_ts=600, **common).reason == "stop"  # type: ignore[union-attr]


def test_exit_on_bar_without_levels() -> None:
    assert exit_on_bar(side="LONG", bar_ts=0, bar_open=1.0, bar_high=9.0, bar_low=0.1, stop=None, target=None) is None
    got = exit_on_bar(side="LONG", bar_ts=0, bar_open=1.0, bar_high=9.0, bar_low=0.1, stop=None, target=5.0)
    assert got == Exit(5.0, "target", False)


def test_result_r_and_excursions() -> None:
    assert execution.result_r("LONG", 100.0, 104.0, 2.0) == pytest.approx(2.0)
    assert execution.result_r("SHORT", 100.0, 104.0, 2.0) == pytest.approx(-2.0)
    assert execution.result_r("LONG", 100.0, 104.0, 0.0) == 0.0
    assert execution.excursion_r("LONG", 100.0, 103.0, 99.0, 2.0) == pytest.approx((1.5, 0.5))
    assert execution.excursion_r("SHORT", 100.0, 103.0, 99.0, 2.0) == pytest.approx((0.5, 1.5))
    assert execution.excursion_r("LONG", 100.0, 103.0, 99.0, 0.0) == (0.0, 0.0)


# ------------------------------------------------------------------------------------------------
# session_flatten_ts
# ------------------------------------------------------------------------------------------------
def test_flatten_regular_venue() -> None:
    calendar = FakeCalendar()
    assert execution.session_flatten_ts(calendar, "XNAS", utc(20, 15), 5) == utc(20, 20, 55)
    entry = datetime(2026, 11, 20, 15, tzinfo=UTC)
    assert execution.session_flatten_ts(calendar, "XNAS", entry, 5) == utc(20, 20, 55)


def test_flatten_accepts_numpy_integers() -> None:
    calendar = FakeCalendar()
    assert execution.session_flatten_ts(calendar, "XNAS", np.int64(utc(20, 15)), 5) == utc(20, 20, 55)


def test_flatten_lunch_break_venue_uses_the_last_segment_of_the_day() -> None:
    calendar = FakeCalendar()  # XTKS: 00:00-02:30 and 03:30-06:30 UTC
    assert execution.session_flatten_ts(calendar, "XTKS", utc(20, 1), 5) == utc(20, 6, 25)
    assert execution.session_flatten_ts(calendar, "XTKS", utc(20, 3), 5) == utc(20, 6, 25)  # in the break
    assert execution.session_flatten_ts(calendar, "XTKS", utc(20, 4), 5) == utc(20, 6, 25)  # last segment


def test_flatten_in_the_last_minutes_lies_before_the_entry() -> None:
    calendar = FakeCalendar()
    assert execution.session_flatten_ts(calendar, "XNAS", utc(20, 20, 58), 5) == utc(20, 20, 55)


def test_flatten_24_7_venue_before_utc_midnight() -> None:
    calendar = FakeCalendar()  # CRYPTO is one endless segment in the fake calendar
    assert execution.session_flatten_ts(calendar, "CRYPTO", utc(20, 15), 5) == utc(20, 23, 55)
    assert execution.session_flatten_ts(calendar, "CRYPTO", utc(21, 0, 1), 5) == utc(21, 23, 55)


def test_flatten_after_the_close_finds_no_close_of_that_day() -> None:
    calendar = FakeCalendar()
    assert execution.session_flatten_ts(calendar, "XNAS", utc(20, 22), 5) is None  # next segment is Monday


def test_flatten_respects_holidays() -> None:
    calendar = FakeCalendar(holidays={"XNAS": [date(2026, 11, 20)]})
    assert execution.session_flatten_ts(calendar, "XNAS", utc(20, 15), 5) is None


def test_flatten_session_across_utc_midnight_uses_the_local_trading_date() -> None:
    """A Sydney-like session (23:00-05:00 UTC, one local trading date) is NOT cut at UTC midnight."""
    calendar = SydneyLikeCalendar()
    assert execution.session_flatten_ts(calendar, "XASX", utc(19, 23, 30), 5) == utc(20, 4, 55)
    assert execution.session_flatten_ts(calendar, "XASX", utc(20, 2, 0), 5) == utc(20, 4, 55)
    assert execution.session_flatten_ts(calendar, "XASX", utc(20, 4, 58), 5) == utc(20, 4, 55)


def test_flatten_24_7_venue_reported_as_daily_segments() -> None:
    """Calendars may report a 24/7 venue as one segment per UTC day: the same flatten time results."""
    calendar = FakeCalendar({"CRYPTO": ((0, 1440),)}, weekends_closed=False)
    assert execution.session_flatten_ts(calendar, "CRYPTO", utc(20, 15), 5) == utc(20, 23, 55)
    assert execution.session_flatten_ts(calendar, "CRYPTO", utc(21, 0, 0), 5) == utc(21, 23, 55)


def reference_flatten_ts(calendar: FakeCalendar, exchange: str, entry: int, minutes_before_close: int) -> int | None:
    """Brute force: the end of the last open minute of the entry's trading date within 36 hours."""
    day = calendar.session_date(exchange, from_ts(entry))
    end = None
    for minute in range(entry // 60 * 60, entry + 36 * 3600, 60):
        at = from_ts(minute)
        if calendar.is_open(exchange, at) and calendar.session_date(exchange, at) == day:
            end = minute + 60
    if end is None or end >= entry + 36 * 3600:
        return None
    return end - minutes_before_close * 60


@pytest.mark.parametrize(
    ("calendar", "exchange"),
    [(FakeCalendar(), "XNAS"), (FakeCalendar(), "XTKS"), (FakeCalendar(), "CRYPTO"), (SydneyLikeCalendar(), "XASX")],
    ids=["regular", "lunch-break", "24-7", "across-utc-midnight"],
)
def test_flatten_agrees_with_a_brute_force_scan(calendar: FakeCalendar, exchange: str) -> None:
    rng = np.random.default_rng(len(exchange))
    found = set()
    for entry in (utc(16, 0) + 60 * rng.integers(0, 7 * 1440, size=40)).tolist():
        expected = reference_flatten_ts(calendar, exchange, int(entry), 5)
        assert execution.session_flatten_ts(calendar, exchange, int(entry), 5) == expected, from_ts(int(entry))
        found.add(expected is None)
    assert False in found  # the comparison is not vacuous


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_round_trip_result_includes_slippage_on_both_market_fills(side: str) -> None:
    """Entry at 100 and a market exit at 100 lose exactly two slippages; a limit exit loses only one."""
    entry = execution.entry_fill_price(COSTS, NVDA, side, 100.0)  # type: ignore[arg-type]
    market = execution.exit_fill_price(COSTS, NVDA, side, Exit(100.0, "time_stop", True))  # type: ignore[arg-type]
    limit = execution.exit_fill_price(COSTS, NVDA, side, Exit(100.0, "target", False))  # type: ignore[arg-type]
    assert execution.result_r(side, entry, market, 1.0) == pytest.approx(-2 * 100.0 * 0.0002)  # type: ignore[arg-type]
    assert execution.result_r(side, entry, limit, 1.0) == pytest.approx(-100.0 * 0.0002)  # type: ignore[arg-type]
