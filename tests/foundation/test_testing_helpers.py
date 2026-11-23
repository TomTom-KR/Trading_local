import random
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from trading_local.clock import to_ts
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS
from trading_local.models import Instrument, SessionSegment, SessionState
from trading_local.testing import (
    ALWAYS_OPEN_EXCHANGES,
    DEFAULT_TEST_WINDOWS,
    SAMPLE_INSTRUMENTS,
    FakeCalendar,
    make_bars,
    make_settings,
    seed_bars,
    seed_instruments,
)

BERLIN = ZoneInfo("Europe/Berlin")


def at(day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """A UTC time in November 2026: the 16th is a Monday, the 20th a Friday, the 26th a Thursday."""
    return datetime(2026, 11, day, hour, minute, second, tzinfo=UTC)


T0 = to_ts(at(20, 14, 30))


# ------------------------------------------------------------------------------------------------
# make_settings / SAMPLE_INSTRUMENTS (unchanged helpers)
# ------------------------------------------------------------------------------------------------
def test_make_settings_keeps_tests_offline_and_inside_the_given_directory(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, log_level="DEBUG")

    assert settings.data_dir == tmp_path
    assert settings.db_path.parent == tmp_path and settings.log_dir.parent == tmp_path
    assert settings.offline is True and settings.open_browser is False
    assert settings.mail_mode == "file" and settings.smtp_user == ""
    assert settings.log_level == "DEBUG"


def test_sample_instruments_cover_every_kind_of_venue_and_feed() -> None:
    symbols = "NVDA SAP.DE SHEL.L 7203.T BTC-USD ^GDAXI ^GSPC EURUSD=X EURGBP=X EURJPY=X"
    assert [instrument.symbol for instrument in SAMPLE_INSTRUMENTS] == symbols.split()
    exchanges = {"XNAS", "XNYS", "XETR", "XLON", "XTKS", "CRYPTO", "FX"}
    assert {instrument.exchange for instrument in SAMPLE_INSTRUMENTS} == exchanges
    assert {instrument.delay_min for instrument in SAMPLE_INSTRUMENTS} == {0, 15, 20}


# ------------------------------------------------------------------------------------------------
# make_bars
# ------------------------------------------------------------------------------------------------
def test_make_bars_returns_a_canonical_frame_of_consecutive_minutes() -> None:
    frame = make_bars(T0, 240)

    assert frame.index.name == "ts"
    assert frame.index.dtype == np.int64
    assert list(frame.columns) == list(BAR_COLUMNS)
    assert all(dtype == np.float64 for dtype in frame.dtypes)
    assert frame.index.tolist() == [T0 + 60 * n for n in range(240)]
    assert not frame.isna().any().any()


def test_make_bars_floors_the_start_to_a_full_minute() -> None:
    assert make_bars(T0 + 59, 3).index.tolist() == [T0, T0 + 60, T0 + 120]
    assert make_bars(T0 + 60, 1).index.tolist() == [T0 + 60]
    assert make_bars(np.int64(T0 + 1), 1).index.tolist() == [T0]


def test_make_bars_is_deterministic_per_seed() -> None:
    pd.testing.assert_frame_equal(make_bars(T0, 500), make_bars(T0, 500), check_exact=True)
    pd.testing.assert_frame_equal(make_bars(T0, 500, seed=7), make_bars(T0, 500, seed=7), check_exact=True)
    pd.testing.assert_frame_equal(make_bars(T0, 500, seed=1), make_bars(T0, 500), check_exact=True)  # default seed
    assert not make_bars(T0, 500, seed=1)["close"].equals(make_bars(T0, 500, seed=2)["close"])
    assert not make_bars(T0, 500, seed=1)["volume"].equals(make_bars(T0, 500, seed=2)["volume"])


@pytest.mark.parametrize("seed", [1, 2, 3, 99])
@pytest.mark.parametrize("vol", [0.0, 0.0008, 0.05])
def test_make_bars_prices_are_consistent(seed: int, vol: float) -> None:
    frame = make_bars(T0, 2000, start_price=42.0, seed=seed, vol=vol, drift=-0.0001)

    assert frame["open"].iloc[0] == 42.0
    assert (frame["open"].to_numpy()[1:] == frame["close"].to_numpy()[:-1]).all()  # opens at the previous close
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
    assert (frame[["open", "high", "low", "close"]] > 0).all().all()
    assert np.isfinite(frame.to_numpy()).all()
    assert ((frame["volume"] >= 500.0) & (frame["volume"] <= 1500.0)).all()  # positive, around base_volume


def test_make_bars_volatility_and_drift_shape_the_walk() -> None:
    returns = np.log(make_bars(T0, 20_000, vol=0.002)["close"]).diff().dropna()
    assert returns.std() == pytest.approx(0.002, rel=0.05)
    assert abs(returns.mean()) < 0.0001

    trend = make_bars(T0, 300, start_price=50.0, drift=0.001, vol=0.0)
    assert trend["close"].is_monotonic_increasing
    np.testing.assert_allclose(trend["close"], 50.0 * np.exp(0.001 * np.arange(1, 301)), rtol=1e-12)
    assert (trend["high"] == trend["close"]).all() and (trend["low"] == trend["open"]).all()  # no wicks without vol

    assert make_bars(T0, 2000, vol=0.01)["close"].std() > make_bars(T0, 2000, vol=0.001)["close"].std()


def test_make_bars_scales_the_volume_with_base_volume() -> None:
    assert make_bars(T0, 100, base_volume=1_000_000.0)["volume"].between(500_000.0, 1_500_000.0).all()
    assert (make_bars(T0, 100, base_volume=0.0)["volume"] == 0.0).all()  # e.g. for an index


def test_make_bars_with_zero_minutes_is_an_empty_canonical_frame() -> None:
    frame = make_bars(T0, 0)

    assert frame.empty
    pd.testing.assert_frame_equal(frame, repo_market.empty_bars_1m())


@pytest.mark.parametrize(
    "arguments",
    [
        {"minutes": -1},
        {"minutes": 5, "start_price": 0.0},
        {"minutes": 5, "vol": -0.1},
        {"minutes": 5, "base_volume": -1.0},
    ],
)
def test_make_bars_rejects_impossible_arguments(arguments: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_bars(T0, **arguments)


# ------------------------------------------------------------------------------------------------
# seed helpers
# ------------------------------------------------------------------------------------------------
def test_seed_instruments_stores_the_sample_universe_exactly(db: Database) -> None:
    assert seed_instruments(db) == len(SAMPLE_INSTRUMENTS)
    assert repo_market.list_instruments(db, active_only=False) == SAMPLE_INSTRUMENTS

    repo_market.set_instrument_flags(db, "NVDA", active=False, tradable=False)
    seed_instruments(db)  # seeding again restores the flags of the samples
    assert repo_market.get_instrument(db, "NVDA") == SAMPLE_INSTRUMENTS[0]


def test_seed_instruments_accepts_a_custom_universe(db: Database) -> None:
    custom = [Instrument("AAPL", "Apple", "equity", "US", "XNAS", "USD", active=False)]

    assert seed_instruments(db, custom) == 1
    assert repo_market.list_instruments(db, active_only=False) == custom


def test_seed_bars_stores_the_frame_as_rest_data_by_default(db: Database) -> None:
    frame = make_bars(T0, 30)

    assert seed_bars(db, "NVDA", frame) == 30
    assert seed_bars(db, "BTC-USD", frame.head(5), "quote") == 5

    pd.testing.assert_frame_equal(repo_market.read_bars_1m(db, "NVDA"), frame, check_exact=True)
    sources = db.query("SELECT symbol, source, COUNT(*) FROM bars_1m GROUP BY symbol, source ORDER BY symbol")
    assert [tuple(row) for row in sources] == [("BTC-USD", "quote", 5), ("NVDA", "rest", 30)]


# ------------------------------------------------------------------------------------------------
# FakeCalendar: description
# ------------------------------------------------------------------------------------------------
def test_fake_calendar_implements_the_market_calendar_protocol() -> None:
    assert isinstance(FakeCalendar(), MarketCalendar)


def test_default_windows_match_the_documented_sessions() -> None:
    assert dict(DEFAULT_TEST_WINDOWS) == {
        "XNAS": ((14 * 60 + 30, 21 * 60),),
        "XNYS": ((14 * 60 + 30, 21 * 60),),
        "XETR": ((8 * 60, 16 * 60 + 30),),
        "XLON": ((8 * 60, 16 * 60 + 30),),
        "XTKS": ((0, 2 * 60 + 30), (3 * 60 + 30, 6 * 60 + 30)),
    }
    assert ALWAYS_OPEN_EXCHANGES == ("CRYPTO", "FX", "KRAKEN")
    with pytest.raises(TypeError):
        DEFAULT_TEST_WINDOWS["XNAS"] = ((0, 60),)  # type: ignore[index]


def test_fake_calendar_knows_every_exchange_of_the_sample_instruments() -> None:
    calendar = FakeCalendar()

    assert calendar.exchanges() == ["CRYPTO", "FX", "KRAKEN", "XETR", "XLON", "XNAS", "XNYS", "XTKS"]
    assert {instrument.exchange for instrument in SAMPLE_INSTRUMENTS} <= set(calendar.exchanges())
    assert FakeCalendar({"XHKG": [(90, 480)]}).exchanges() == ["CRYPTO", "FX", "KRAKEN", "XHKG"]


def test_fake_calendar_display_names_and_time_zone() -> None:
    calendar = FakeCalendar()

    assert calendar.display_name("XETR") == "Xetra"
    assert calendar.display_name("XNYS") == "New York"
    assert calendar.display_name("XTKS") == "Tokio"
    assert calendar.display_name("XHKG") == "XHKG"  # unknown venues are shown by their key
    assert {calendar.timezone(exchange) for exchange in calendar.exchanges()} == {"UTC"}


# ------------------------------------------------------------------------------------------------
# FakeCalendar: state
# ------------------------------------------------------------------------------------------------
NY_OPEN, NY_CLOSE = (14, 30), (21, 0)


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        # Friday around the open
        (at(20, 14, 29, 59), ("closed", at(19, 21), at(20, 14, 30), at(20, 14, 30), at(20, 21), "Geschlossen")),
        (at(20, 14, 30), ("open", at(20, 14, 30), at(20, 21), at(23, 14, 30), at(20, 21), "Handel")),
        (at(20, 17, 45), ("open", at(20, 14, 30), at(20, 21), at(23, 14, 30), at(20, 21), "Handel")),
        # Friday around the close
        (at(20, 20, 59, 59), ("open", at(20, 14, 30), at(20, 21), at(23, 14, 30), at(20, 21), "Handel")),
        (at(20, 21), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Geschlossen")),
        (at(20, 23, 59, 59), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Geschlossen")),
        # weekend
        (at(21, 0), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Wochenende")),
        (at(21, 15), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Wochenende")),
        (at(22, 23, 59, 59), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Wochenende")),
        # Monday morning: the weekend is over, the venue is simply not open yet
        (at(23, 0), ("closed", at(20, 21), at(23, 14, 30), at(23, 14, 30), at(23, 21), "Geschlossen")),
        (at(23, 14, 30), ("open", at(23, 14, 30), at(23, 21), at(24, 14, 30), at(23, 21), "Handel")),
        # ordinary night between two trading days
        (at(18, 3), ("closed", at(17, 21), at(18, 14, 30), at(18, 14, 30), at(18, 21), "Geschlossen")),
    ],
)
def test_state_of_a_venue_with_one_session_per_day(moment: datetime, expected: tuple[object, ...]) -> None:
    phase, since, until, next_open, next_close, note = expected

    assert FakeCalendar().state("XNAS", moment) == SessionState(
        "XNAS", phase, phase == "open", since, until, next_open, next_close, note  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (at(19, 23, 59, 59), ("closed", at(19, 6, 30), at(20, 0), at(20, 0), at(20, 2, 30), "Geschlossen")),
        # morning session starts at midnight UTC
        (at(20, 0), ("open", at(20, 0), at(20, 2, 30), at(20, 3, 30), at(20, 2, 30), "Handel")),
        (at(20, 2, 29, 59), ("open", at(20, 0), at(20, 2, 30), at(20, 3, 30), at(20, 2, 30), "Handel")),
        # lunch break
        (at(20, 2, 30), ("break", at(20, 2, 30), at(20, 3, 30), at(20, 3, 30), at(20, 6, 30), "Mittagspause")),
        (at(20, 3, 29, 59), ("break", at(20, 2, 30), at(20, 3, 30), at(20, 3, 30), at(20, 6, 30), "Mittagspause")),
        # afternoon session: the next open is Monday's morning session
        (at(20, 3, 30), ("open", at(20, 3, 30), at(20, 6, 30), at(23, 0), at(20, 6, 30), "Handel")),
        (at(20, 6, 29, 59), ("open", at(20, 3, 30), at(20, 6, 30), at(23, 0), at(20, 6, 30), "Handel")),
        (at(20, 6, 30), ("closed", at(20, 6, 30), at(23, 0), at(23, 0), at(23, 2, 30), "Geschlossen")),
        (at(22, 23, 59, 59), ("closed", at(20, 6, 30), at(23, 0), at(23, 0), at(23, 2, 30), "Wochenende")),
        (at(23, 0), ("open", at(23, 0), at(23, 2, 30), at(23, 3, 30), at(23, 2, 30), "Handel")),
    ],
)
def test_state_of_a_venue_with_a_lunch_break(moment: datetime, expected: tuple[object, ...]) -> None:
    phase, since, until, next_open, next_close, note = expected

    assert FakeCalendar().state("XTKS", moment) == SessionState(
        "XTKS", phase, phase == "open", since, until, next_open, next_close, note  # type: ignore[arg-type]
    )


def test_state_on_and_around_a_holiday() -> None:
    calendar = FakeCalendar(holidays={"XNAS": [date(2026, 11, 26)]})  # Thursday

    holiday = calendar.state("XNAS", at(26, 15))
    assert holiday == SessionState(
        "XNAS", "closed", False, at(25, 21), at(27, 14, 30), at(27, 14, 30), at(27, 21), "Feiertag"
    )
    evening_before = calendar.state("XNAS", at(25, 22))
    assert (evening_before.note, evening_before.until) == ("Geschlossen", at(27, 14, 30))
    morning_after = calendar.state("XNAS", at(27, 10))
    assert (morning_after.note, morning_after.since) == ("Geschlossen", at(25, 21))
    assert calendar.state("XNAS", at(25, 15)).next_open == at(27, 14, 30)
    assert calendar.state("XNYS", at(26, 15)).is_open  # the holiday belongs to XNAS only


def test_state_across_a_holiday_next_to_a_weekend() -> None:
    calendar = FakeCalendar(holidays={"XNAS": {date(2026, 11, 27), date(2026, 11, 28)}})  # Friday and Saturday

    thursday_evening = calendar.state("XNAS", at(26, 22))
    assert (thursday_evening.since, thursday_evening.until) == (at(26, 21), at(30, 14, 30))
    assert thursday_evening.note == "Geschlossen"
    assert calendar.state("XNAS", at(27, 15)).note == "Feiertag"
    assert calendar.state("XNAS", at(28, 15)).note == "Wochenende"  # a holiday on a weekend is a weekend
    assert calendar.state("XNAS", at(29, 15)).since == at(26, 21)


def test_state_of_an_exchange_without_windows_is_always_open() -> None:
    calendar = FakeCalendar()

    for exchange in ("CRYPTO", "FX", "SOME-UNKNOWN-VENUE"):
        always_open = SessionState(exchange, "open", True, None, None, None, None, "Handel")
        for moment in (at(20, 15), at(21, 3), at(22, 23, 59, 59)):
            assert calendar.state(exchange, moment) == always_open
            assert calendar.is_open(exchange, moment)


def test_custom_windows_replace_the_defaults_completely() -> None:
    calendar = FakeCalendar({"CRYPTO": [(60, 120)]})

    assert calendar.is_open("CRYPTO", at(20, 1, 30))
    assert not calendar.is_open("CRYPTO", at(20, 3))
    assert not calendar.is_open("CRYPTO", at(21, 1, 30))  # with windows the weekend applies
    assert calendar.is_open("XNAS", at(21, 3))  # no entry any more: around the clock
    assert FakeCalendar({}).is_open("XETR", at(22, 4))


def test_weekends_can_be_trading_days() -> None:
    calendar = FakeCalendar(weekends_closed=False, holidays={"XNAS": [date(2026, 11, 22)]})

    saturday = calendar.state("XNAS", at(21, 15))
    assert saturday == SessionState(
        "XNAS", "open", True, at(21, 14, 30), at(21, 21), at(23, 14, 30), at(21, 21), "Handel"
    )
    assert calendar.state("XNAS", at(22, 15)).note == "Feiertag"  # Sunday is closed as a holiday only
    assert calendar.state("XNAS", at(21, 22)).note == "Geschlossen"


def test_windows_touching_midnight_are_adjacent_sessions() -> None:
    calendar = FakeCalendar({"CME": [(0, 1440)]})

    tuesday = calendar.state("CME", at(17, 12))
    assert tuesday == SessionState("CME", "open", True, at(17, 0), at(18, 0), at(18, 0), at(18, 0), "Handel")
    assert calendar.state("CME", at(18, 0)).since == at(18, 0)
    assert calendar.state("CME", at(20, 23, 59, 59)).until == at(21, 0)
    saturday = calendar.state("CME", at(21, 0))
    assert saturday == SessionState("CME", "closed", False, at(21, 0), at(23, 0), at(23, 0), at(24, 0), "Wochenende")
    assert calendar.last_completed_session("CME", at(18, 0)) == date(2026, 11, 17)
    assert calendar.add_trading_minutes("CME", at(20, 23, 30), 60) == at(23, 0, 30)


def test_times_in_other_zones_are_converted_and_results_are_utc() -> None:
    calendar = FakeCalendar()
    berlin_afternoon = datetime(2026, 11, 20, 16, 0, tzinfo=BERLIN)  # 15:00 UTC

    state = calendar.state("XNAS", berlin_afternoon)
    assert state == calendar.state("XNAS", at(20, 15))
    assert state.since is not None and state.since.tzinfo is UTC
    assert calendar.is_open("XNAS", berlin_afternoon)
    assert calendar.session_date("XNAS", datetime(2026, 11, 21, 0, 30, tzinfo=BERLIN)) == date(2026, 11, 20)
    assert calendar.add_trading_minutes("XNAS", berlin_afternoon, 30) == at(20, 15, 30)
    assert calendar.segments("XNAS", berlin_afternoon, berlin_afternoon + timedelta(hours=1)) == [
        SessionSegment("XNAS", at(20, 15), at(20, 16))
    ]


def test_naive_datetimes_are_rejected() -> None:
    calendar = FakeCalendar()
    naive = datetime(2026, 11, 20, 15, 0)

    for exchange in ("XNAS", "CRYPTO"):
        for call in (
            lambda: calendar.state(exchange, naive),
            lambda: calendar.is_open(exchange, naive),
            lambda: calendar.segments(exchange, naive, at(21)),
            lambda: calendar.segments(exchange, at(20), naive),
            lambda: calendar.session_date(exchange, naive),
            lambda: calendar.last_completed_session(exchange, naive),
            lambda: calendar.minutes_since_open(exchange, naive),
            lambda: calendar.minutes_to_close(exchange, naive),
            lambda: calendar.add_trading_minutes(exchange, naive, 5),
        ):
            with pytest.raises(ValueError, match="naive"):
                call()


def phase_by_arithmetic(windows: tuple[tuple[int, int], ...], holidays: set[date], moment: datetime) -> str:
    """The phase straight from the definition of the windows (independent of FakeCalendar's segment search)."""
    if moment.weekday() >= 5 or moment.date() in holidays:
        return "closed"
    minute = (moment - datetime(moment.year, moment.month, moment.day, tzinfo=UTC)) / timedelta(minutes=1)
    if any(start <= minute < end for start, end in windows):
        return "open"
    return "break" if windows[0][0] <= minute < windows[-1][1] else "closed"


@pytest.mark.parametrize("exchange", ["XNAS", "XETR", "XTKS"])
def test_state_agrees_with_the_window_definition_at_every_boundary(exchange: str) -> None:
    holidays = {date(2026, 11, 19), date(2026, 11, 26)}
    calendar = FakeCalendar(holidays={exchange: holidays})
    windows = DEFAULT_TEST_WINDOWS[exchange]
    second = timedelta(seconds=1)

    for step in range(0, 14 * 1440, 5):  # two weeks in steps of five minutes: hits every window boundary
        for moment in (at(16) + timedelta(minutes=step), at(16) + timedelta(minutes=step, seconds=-1)):
            state = calendar.state(exchange, moment)
            phase = phase_by_arithmetic(windows, holidays, moment)
            assert state.phase == phase, moment
            assert state.is_open == (phase == "open") == calendar.is_open(exchange, moment)
            assert state.since is not None and state.until is not None
            assert state.since <= moment < state.until
            # [since, until) is exactly the run of the current phase
            assert phase_by_arithmetic(windows, holidays, state.since) == phase
            assert phase_by_arithmetic(windows, holidays, state.until - second) == phase
            assert phase_by_arithmetic(windows, holidays, state.since - second) != phase
            assert phase_by_arithmetic(windows, holidays, state.until) != phase
            # next_open / next_close are the nearest boundaries of regular segments
            assert state.next_open is not None and state.next_close is not None
            assert state.next_open > moment and state.next_close > moment
            assert phase_by_arithmetic(windows, holidays, state.next_open) == "open"
            assert phase_by_arithmetic(windows, holidays, state.next_open - second) != "open"
            assert phase_by_arithmetic(windows, holidays, state.next_close - second) == "open"
            assert phase_by_arithmetic(windows, holidays, state.next_close) != "open"
            segment_start = moment if state.is_open else state.next_open
            assert state.next_close == calendar.state(exchange, segment_start).until


# ------------------------------------------------------------------------------------------------
# FakeCalendar: segments
# ------------------------------------------------------------------------------------------------
def test_segments_lists_the_regular_sessions_of_a_week() -> None:
    calendar = FakeCalendar()

    week = calendar.segments("XNAS", at(16), at(23))

    assert week == [SessionSegment("XNAS", at(day, 14, 30), at(day, 21), "regular") for day in (16, 17, 18, 19, 20)]
    assert calendar.segments("XNAS", at(16), at(23), extended=True) == week  # no pre/post market in the double


def test_segments_are_clipped_to_the_requested_range() -> None:
    calendar = FakeCalendar()

    assert calendar.segments("XNAS", at(20, 15), at(20, 16)) == [SessionSegment("XNAS", at(20, 15), at(20, 16))]
    assert calendar.segments("XNAS", at(20, 20), at(23, 15)) == [
        SessionSegment("XNAS", at(20, 20), at(20, 21)),
        SessionSegment("XNAS", at(23, 14, 30), at(23, 15)),
    ]
    assert calendar.segments("XNAS", at(20, 10), at(20, 14, 30)) == []  # ends when the session starts
    assert calendar.segments("XNAS", at(20, 21), at(21, 12)) == []  # starts when the session ends
    assert calendar.segments("XNAS", at(20, 14, 30), at(20, 14, 30, 1)) == [
        SessionSegment("XNAS", at(20, 14, 30), at(20, 14, 30, 1))
    ]
    assert calendar.segments("XNAS", at(21), at(23)) == []  # weekend
    assert calendar.segments("XNAS", at(20, 16), at(20, 16)) == []
    assert calendar.segments("XNAS", at(20, 16), at(20, 15)) == []


def test_segments_of_a_day_with_lunch_break_and_of_a_holiday() -> None:
    calendar = FakeCalendar(holidays={"XTKS": [date(2026, 11, 19)]})

    assert calendar.segments("XTKS", at(20), at(21)) == [
        SessionSegment("XTKS", at(20, 0), at(20, 2, 30)),
        SessionSegment("XTKS", at(20, 3, 30), at(20, 6, 30)),
    ]
    assert calendar.segments("XTKS", at(20, 1), at(20, 4)) == [
        SessionSegment("XTKS", at(20, 1), at(20, 2, 30)),
        SessionSegment("XTKS", at(20, 3, 30), at(20, 4)),
    ]
    assert calendar.segments("XTKS", at(19), at(20)) == []


def test_segments_of_an_always_open_venue_is_the_range_itself() -> None:
    calendar = FakeCalendar()

    assert calendar.segments("CRYPTO", at(20, 22), at(23, 3)) == [SessionSegment("CRYPTO", at(20, 22), at(23, 3))]
    assert calendar.segments("FX", at(21), at(21)) == []


# ------------------------------------------------------------------------------------------------
# FakeCalendar: sessions
# ------------------------------------------------------------------------------------------------
def test_session_date_is_the_utc_date() -> None:
    calendar = FakeCalendar()

    assert calendar.session_date("XNAS", at(20, 15)) == date(2026, 11, 20)
    assert calendar.session_date("XTKS", at(20, 23, 59, 59)) == date(2026, 11, 20)
    assert calendar.session_date("CRYPTO", at(21, 0)) == date(2026, 11, 21)


@pytest.mark.parametrize(
    ("exchange", "moment", "expected"),
    [
        ("XNAS", at(20, 0), date(2026, 11, 19)),
        ("XNAS", at(20, 14, 30), date(2026, 11, 19)),
        ("XNAS", at(20, 20, 59, 59), date(2026, 11, 19)),
        ("XNAS", at(20, 21), date(2026, 11, 20)),  # completed at the very instant of the close
        ("XNAS", at(21, 12), date(2026, 11, 20)),
        ("XNAS", at(22, 23), date(2026, 11, 20)),
        ("XNAS", at(23, 14, 29), date(2026, 11, 20)),
        ("XNAS", at(23, 21), date(2026, 11, 23)),
        ("XTKS", at(20, 0), date(2026, 11, 19)),
        ("XTKS", at(20, 3), date(2026, 11, 19)),  # lunch break: the morning alone does not complete the day
        ("XTKS", at(20, 6, 29, 59), date(2026, 11, 19)),
        ("XTKS", at(20, 6, 30), date(2026, 11, 20)),
        ("XETR", at(16, 9), date(2026, 11, 13)),  # Monday morning: the previous Friday
        ("CRYPTO", at(20, 0), date(2026, 11, 19)),
        ("CRYPTO", at(20, 23, 59, 59), date(2026, 11, 19)),
        ("CRYPTO", at(22, 12), date(2026, 11, 21)),  # always open: every UTC day is a session
    ],
)
def test_last_completed_session(exchange: str, moment: datetime, expected: date) -> None:
    assert FakeCalendar().last_completed_session(exchange, moment) == expected


def test_last_completed_session_skips_holidays() -> None:
    calendar = FakeCalendar(holidays={"XNAS": [date(2026, 11, 26), date(2026, 11, 27)]})

    assert calendar.last_completed_session("XNAS", at(27, 23)) == date(2026, 11, 25)
    assert calendar.last_completed_session("XNAS", at(30, 14)) == date(2026, 11, 25)
    assert calendar.last_completed_session("XNAS", at(30, 21)) == date(2026, 11, 30)
    assert calendar.last_completed_session("XNYS", at(27, 23)) == date(2026, 11, 27)


@pytest.mark.parametrize(
    ("exchange", "moment", "since_open", "to_close"),
    [
        ("XNAS", at(20, 14, 29, 59), None, None),
        ("XNAS", at(20, 14, 30), 0, 390),
        ("XNAS", at(20, 14, 30, 59), 0, 389),  # whole minutes, rounded down
        ("XNAS", at(20, 15), 30, 360),
        ("XNAS", at(20, 20, 59, 59), 389, 0),
        ("XNAS", at(20, 21), None, None),
        ("XNAS", at(21, 15), None, None),  # weekend
        ("XTKS", at(20, 0), 0, 330),
        ("XTKS", at(20, 1), 60, 270),  # the lunch break does not count
        ("XTKS", at(20, 2, 30), None, None),  # lunch break
        ("XTKS", at(20, 3, 30), 150, 180),
        ("XTKS", at(20, 4), 180, 150),
        ("XTKS", at(20, 6, 30), None, None),
        ("CRYPTO", at(21, 0), 0, 1440),  # always open: the session is the UTC day
        ("CRYPTO", at(21, 15, 0, 30), 900, 539),
    ],
)
def test_minutes_since_open_and_to_close(
    exchange: str, moment: datetime, since_open: int | None, to_close: int | None
) -> None:
    calendar = FakeCalendar()

    assert calendar.minutes_since_open(exchange, moment) == since_open
    assert calendar.minutes_to_close(exchange, moment) == to_close


def test_trading_minutes_are_none_on_a_holiday() -> None:
    calendar = FakeCalendar(holidays={"XNAS": [date(2026, 11, 26)]})

    assert calendar.minutes_since_open("XNAS", at(26, 15)) is None
    assert calendar.minutes_to_close("XNAS", at(26, 15)) is None
    assert calendar.minutes_since_open("XNYS", at(26, 15)) == 30


# ------------------------------------------------------------------------------------------------
# FakeCalendar: add_trading_minutes
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("exchange", "start", "minutes", "expected"),
    [
        ("XNAS", at(20, 15), 30, at(20, 15, 30)),
        ("XNAS", at(20, 15, 0, 30), 1, at(20, 15, 1, 30)),
        ("XNAS", at(20, 14, 30), 390, at(20, 21)),  # a whole session ends at its close ...
        ("XNAS", at(20, 20, 30), 30, at(20, 21)),
        ("XNAS", at(20, 20, 30), 31, at(23, 14, 31)),  # ... one minute more continues after the weekend
        ("XNAS", at(20, 20, 30), 60, at(23, 15)),
        ("XNAS", at(20, 15), 390, at(23, 15)),
        ("XNAS", at(20, 15), 780, at(24, 15)),
        ("XNAS", at(20, 21), 1, at(23, 14, 31)),  # starting at the close
        ("XNAS", at(20, 22), 10, at(23, 14, 40)),  # starting while closed
        ("XNAS", at(21, 12), 10, at(23, 14, 40)),  # starting on the weekend
        ("XNAS", at(18, 3), 400, at(19, 14, 40)),  # over a night
        ("XNAS", at(21, 12), 0, at(21, 12)),  # nothing to wait for
        ("XNAS", at(20, 15), 0, at(20, 15)),
        ("XTKS", at(20, 2), 30, at(20, 2, 30)),  # up to the lunch break
        ("XTKS", at(20, 2), 60, at(20, 4)),  # over the lunch break
        ("XTKS", at(20, 3), 10, at(20, 3, 40)),  # starting in the lunch break
        ("XTKS", at(20, 6), 60, at(23, 0, 30)),  # over the weekend into a session that starts at midnight
        ("CRYPTO", at(21, 12), 90, at(21, 13, 30)),
        ("CRYPTO", at(20, 23, 30), 24 * 60, at(21, 23, 30)),
    ],
)
def test_add_trading_minutes(exchange: str, start: datetime, minutes: int, expected: datetime) -> None:
    assert FakeCalendar().add_trading_minutes(exchange, start, minutes) == expected


def test_add_trading_minutes_skips_holidays() -> None:
    calendar = FakeCalendar(holidays={"XNAS": [date(2026, 11, 26)]})

    assert calendar.add_trading_minutes("XNAS", at(25, 20, 30), 60) == at(27, 15)
    assert calendar.add_trading_minutes("XNYS", at(25, 20, 30), 60) == at(26, 15)


def test_add_trading_minutes_rejects_negative_durations() -> None:
    with pytest.raises(ValueError, match="minutes"):
        FakeCalendar().add_trading_minutes("XNAS", at(20, 15), -1)


@pytest.mark.parametrize("exchange", ["XNAS", "XTKS"])
def test_add_trading_minutes_counts_exactly_the_open_minutes(exchange: str) -> None:
    calendar = FakeCalendar(holidays={exchange: [date(2026, 11, 19)]})
    rng = random.Random(7)

    for _ in range(40):
        start = at(16) + timedelta(minutes=rng.randrange(0, 7 * 1440))
        minutes = rng.randrange(1, 900)

        end = calendar.add_trading_minutes(exchange, start, minutes)

        elapsed = int((end - start) / timedelta(minutes=1))
        open_minutes = sum(calendar.is_open(exchange, start + timedelta(minutes=n)) for n in range(elapsed))
        assert open_minutes == minutes
        assert calendar.is_open(exchange, end - timedelta(minutes=1))  # the last counted minute was a trading minute


# ------------------------------------------------------------------------------------------------
# FakeCalendar: configuration errors
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "windows",
    [
        [],
        [(600, 600)],
        [(700, 600)],
        [(-1, 60)],
        [(600, 1441)],
        [(0, 100), (50, 200)],  # overlapping
        [(0, 100), (100, 200)],  # no gap: this is one window
        [(300, 400), (100, 200)],  # descending
    ],
)
def test_invalid_windows_are_rejected(windows: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError, match="XNAS"):
        FakeCalendar({"XNAS": windows})


def test_invalid_holidays_are_rejected() -> None:
    with pytest.raises(ValueError, match="CRYPTO"):
        FakeCalendar(holidays={"CRYPTO": [date(2026, 11, 26)]})  # always open: a holiday would be ignored
    with pytest.raises(ValueError, match="XNAS"):
        FakeCalendar(holidays={"XNAS": ["2026-11-26"]})  # type: ignore[list-item]
    with pytest.raises(ValueError, match="XNAS"):
        FakeCalendar(holidays={"XNAS": [at(26)]})  # a datetime never equals a date


def test_calendar_keeps_its_own_copy_of_the_configuration() -> None:
    windows = {"XNAS": [(870, 1260)]}
    holidays = {"XNAS": [date(2026, 11, 26)]}
    calendar = FakeCalendar(windows, holidays=holidays)

    windows["XNAS"].append((1300, 1400))
    windows["XETR"] = [(480, 990)]
    holidays["XNAS"].clear()

    assert not calendar.is_open("XNAS", at(20, 22))
    assert calendar.is_open("XETR", at(20, 3))  # still without windows: always open
    assert not calendar.is_open("XNAS", at(26, 15))
