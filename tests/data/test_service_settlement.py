"""Data-side support for settled bars: hole repair, prompt daily bars of UTC-day venues, start-up off the loop.

* RUN-1: a live bar that leaves a traded hole behind the previous stored bar makes the symbol due for the
  REST reconciliation at once, with a window reaching back to the hole (consumers wait at the hole); the
  repairs of one outage go out as one request, so the mass-failure check still works.
* INT-6 / CAUS-6: CRYPTO, FX and CME trade through midnight UTC; the completed day's daily bar is fetched
  right after 00:00 UTC and retried until stored, not only 30 minutes later (the request at the regular
  offset still follows and stores the final values).
* RUN-4: ``prepare`` runs its calendar look-ups in the database thread, never on the event loop.

FakeCalendar (UTC): XNAS/XNYS 14:30-21:00, XETR/XLON 08:00-16:30 on weekdays; CRYPTO and FX always open.
"""

import threading
from dataclasses import replace
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from tests.data.conftest import TEST_OPTIONS, Env, EnvFactory
from tests.data.fakes import daily_frame, minute_frame
from trading_local.barstore import BarReader
from trading_local.clock import to_ts
from trading_local.data.provider import Tick
from trading_local.db import repo_market

MIN = 60
T0 = to_ts(datetime(2026, 11, 20, 15, 0, tzinfo=UTC))  # Friday, New York and Xetra open
MONDAY_OPEN = to_ts(datetime(2026, 11, 23, 14, 30, tzinfo=UTC))
MIDNIGHT = to_ts(datetime(2026, 11, 21, 0, 0, tzinfo=UTC))  # end of Friday the 20th (UTC)
UTC_DAY_SYMBOLS = {"BTC-USD", "EURUSD=X", "EURGBP=X", "EURJPY=X"}  # CRYPTO and FX: open around the clock


def no_rest_bars(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame | None:
    return None


def calls_at(env: Env, symbol: str, kind: str = "minutes") -> list[int]:
    return [call.at for call in env.provider.calls_of(kind) if symbol in call.symbols]


def tick(env: Env, symbol: str, minute: int, price: float = 100.0) -> None:
    """One stream tick in the middle of ``minute`` (it arrives one second after its source time)."""
    env.service.on_tick(Tick(symbol, float(minute + 30), price, float(minute + 31)))


async def refresh_at(env: Env, ts: int) -> list[str]:
    env.clock.set(ts)
    return await env.service.refresh(env.clock.now())


# ------------------------------------------------------------------------------------------------
# Hole repair (RUN-1)
# ------------------------------------------------------------------------------------------------
async def test_a_hole_left_by_stream_bars_makes_the_symbol_due_at_once(env: Env) -> None:
    gap: set[int] = set()

    def rest_confirms_the_gap(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame | None:
        """REST has bars around the stream gap but none inside it: the gap is a genuine one (no trades)."""
        if symbol != "NVDA":
            return None
        frame = minute_frame(start_ts, env.clock.ts() // MIN * MIN)
        return frame[~frame.index.isin(sorted(gap))]

    env.provider.minutes_for = rest_confirms_the_gap
    env.clock.set(T0 + 3)
    await env.service.prepare()
    fetched: int | None = None  # refresh at which the round-robin fetched NVDA while it was streaming
    detected: int | None = None  # refresh that wrote the first bar after the gap
    for k in range(1, 20):
        minute = T0 + (k - 1) * MIN
        now = T0 + k * MIN + 3
        if minute not in gap:
            tick(env, "NVDA", minute)
        if detected is not None and now == detected + MIN:
            # a late tick corrects the bar after the gap: it is rewritten, but reports no second hole
            env.service.on_tick(Tick("NVDA", float(minute - MIN + 45), 100.5, float(now - 5)))
        before = len(calls_at(env, "NVDA"))
        await refresh_at(env, now)
        if fetched is None and k >= 2 and len(calls_at(env, "NVDA")) > before:
            fetched = now
            gap = {now - 3, now - 3 + MIN}  # no trades reach the stream for the next two minutes
        elif gap and detected is None and now - 3 == max(gap) + 2 * MIN:
            detected = now
    assert fetched is not None and detected == fetched + 3 * MIN

    nvda = calls_at(env, "NVDA")
    # requested right when the hole showed (the round-robin alone would wait until fetched + 4 min) ...
    assert detected in nvda
    repair = next(call for call in env.provider.calls_of("minutes") if call.at == detected and "NVDA" in call.symbols)
    assert repair.start_ts <= min(gap)
    # ... and only once: REST confirmed the gap, so only the round-robin asks again, 4 minutes later
    later = [at for at in nvda if at > detected]
    assert later and later[0] >= detected + 4 * MIN
    stored = repo_market.read_bars_1m(env.db, "NVDA")
    assert not set(stored.index.tolist()) & gap and len(stored) >= 10


async def test_the_repair_window_reaches_back_to_an_old_hole(env: Env) -> None:
    last = T0 - 3 * 3_600 - MIN  # BTC-USD was stored until 3 hours ago (the app was down since)
    repo_market.upsert_bars_1m(env.db, "BTC-USD", minute_frame(last - 30 * MIN, last + MIN), "rest")
    env.clock.set(T0 + 3)
    await env.service.prepare()
    tick(env, "BTC-USD", T0)

    await refresh_at(env, T0 + MIN + 3)  # the first live bar after the restart reveals the downtime

    btc = [call for call in env.provider.calls_of("minutes") if "BTC-USD" in call.symbols]
    assert btc and btc[0].at == T0 + MIN + 3
    assert btc[0].start_ts == last  # the bar before the hole (the plain window would start 20 minutes back)
    stored = repo_market.read_bars_1m(env.db, "BTC-USD", start_ts=last, end_ts=T0 - 2 * MIN)
    assert stored.index.tolist() == list(range(last, T0 - 2 * MIN, MIN))  # the downtime is filled

    for k in range(2, 12):
        await refresh_at(env, T0 + k * MIN + 3)
    later = [call for call in env.provider.calls_of("minutes") if "BTC-USD" in call.symbols][1:]
    assert later and all(call.start_ts > last + 2 * 3_600 for call in later)  # the hole is repaired


async def test_a_settled_bar_consumer_examines_the_whole_downtime_after_a_restart(env: Env) -> None:
    """RUN-1 end to end: after a three-hour downtime the first live bar is written before the downtime is
    stored. A consumer with a high-water mark over ``BarReader.settled_bars_after`` (like the broker's
    protective-exit check) waits at the hole, the hole repair fills it within the settle horizon, and the
    consumer examines every minute exactly once, in order."""
    last = T0 - 3 * 3_600 - MIN  # BTC-USD was stored and examined until 3 hours ago
    repo_market.upsert_bars_1m(env.db, "BTC-USD", minute_frame(last - 30 * MIN, last + MIN), "rest")
    env.clock.set(T0 + 3)
    await env.service.prepare()
    checked = last
    examined: list[int] = []

    for k in range(1, 21):
        tick(env, "BTC-USD", T0 + (k - 1) * MIN)
        now = T0 + k * MIN + 3
        await refresh_at(env, now)
        reader = BarReader(env.db, env.instruments, env.calendar, now)
        frame = reader.settled_bars_after("BTC-USD", checked, start_ts=last - 30 * MIN)
        examined += frame.index.tolist()
        if not frame.empty:
            checked = int(frame.index[-1])

    assert examined[0] == last + MIN and examined == list(range(last + MIN, checked + MIN, MIN))
    assert checked >= T0 + 10 * MIN  # the consumer keeps up with the live data


async def test_the_repairs_of_one_outage_share_a_request_and_survive_a_silent_refusal(env: Env) -> None:
    """After a restart every symbol's hole starts at a slightly different minute. The repairs still go out
    as ONE request. A source that hides a refusal behind empty answers must not make the crypto hole count
    as repaired: it stays marked and is repaired once the source answers again. Empty FX answers are
    accepted (FX pairs really have phases without bars), so they neither trip the breaker nor stay marked."""
    down = T0 - 3 * 3_600  # the app was down for three hours
    lasts = {symbol: down - number * 4 * MIN for number, symbol in enumerate(sorted(UTC_DAY_SYMBOLS))}
    for symbol, last in lasts.items():
        repo_market.upsert_bars_1m(env.db, symbol, minute_frame(last - 30 * MIN, last + MIN), "rest")
    answers = env.provider.minutes_for
    env.provider.minutes_for = no_rest_bars  # the source refuses silently
    env.clock.set(T0 - 10 * MIN)
    await env.service.prepare()  # every symbol is due for the round-robin by the first refresh
    env.clock.set(T0 + 3)
    for symbol in UTC_DAY_SYMBOLS:
        tick(env, symbol, T0)

    await refresh_at(env, T0 + MIN + 3)  # the first live bars after the restart reveal four holes

    repairs = [call for call in env.provider.calls_of("minutes") if set(call.symbols) & UTC_DAY_SYMBOLS]
    assert len(repairs) == 1 and set(repairs[0].symbols) == UTC_DAY_SYMBOLS
    assert repairs[0].start_ts == min(lasts.values())  # the oldest of the four starts
    assert env.provider.trips == []  # one reliable symbol is too few for the mass-failure check

    env.provider.minutes_for = answers  # the source answers again
    for k in range(2, 14):
        await refresh_at(env, T0 + k * MIN + 3)

    last = lasts["BTC-USD"]  # the crypto hole stayed marked and is repaired once the source answers
    stored = repo_market.read_bars_1m(env.db, "BTC-USD", start_ts=last, end_ts=T0)
    assert stored.index.tolist() == list(range(last, T0, MIN))
    plain = [call for call in env.provider.calls_of("minutes") if not set(call.symbols) & UTC_DAY_SYMBOLS]
    # no hole: the plain reconciliation window (20 minutes plus the feed delay), never the old hole start
    assert plain and all(call.start_ts >= call.at - 40 * MIN for call in plain)


@pytest.mark.parametrize(
    ("live_minute", "repaired"),
    [(MONDAY_OPEN, False), (MONDAY_OPEN + MIN, True)],
    ids=["weekend-is-no-hole", "missing-open-minute-is-a-hole"],
)
async def test_only_minutes_in_which_the_venue_traded_make_a_hole(
    env: Env, live_minute: int, repaired: bool
) -> None:
    friday_close = to_ts(datetime(2026, 11, 20, 20, 59, tzinfo=UTC))
    repo_market.upsert_bars_1m(env.db, "NVDA", minute_frame(friday_close - 10 * MIN, friday_close + MIN), "rest")
    env.provider.minutes_for = no_rest_bars
    env.clock.set(MONDAY_OPEN + 3)
    await env.service.prepare()  # NVDA counts as just fetched

    for k in (1, 2):
        minute = MONDAY_OPEN + (k - 1) * MIN
        if minute == live_minute:
            tick(env, "NVDA", minute)
        await refresh_at(env, MONDAY_OPEN + k * MIN + 3)

    # the weekend is no hole; a missing first minute of the session (14:30) is one
    assert repo_market.last_bar(env.db, "NVDA").ts == live_minute  # type: ignore[union-attr]
    assert (MONDAY_OPEN + 2 * MIN + 3 in calls_at(env, "NVDA")) is repaired


# ------------------------------------------------------------------------------------------------
# Daily bars of UTC-day venues (INT-6 / CAUS-6)
# ------------------------------------------------------------------------------------------------
async def run_until(env: Env, start: int, end: int) -> None:
    for now in range(start, end, MIN):
        await refresh_at(env, now)


async def test_utc_day_venues_fetch_the_completed_day_right_after_midnight(env: Env) -> None:
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20", "2026-11-21"])
    env.clock.set(MIDNIGHT - 5 * MIN + 3)
    await env.service.prepare()

    await run_until(env, MIDNIGHT - 5 * MIN + 3, MIDNIGHT + 10 * MIN)

    daily = env.provider.calls_of("daily")
    assert {symbol for call in daily for symbol in call.symbols} == UTC_DAY_SYMBOLS
    assert {call.at for call in daily} == {MIDNIGHT + MIN + 3}  # one minute after midnight, not 00:31
    for symbol in UTC_DAY_SYMBOLS:
        assert repo_market.last_day(env.db, symbol) == "2026-11-20"  # the running day 21 is never stored


async def test_a_missing_daily_bar_of_a_utc_day_venue_is_retried_until_stored(env: Env) -> None:
    attempts = {"BTC-USD": 0}

    def daily(symbol: str, start_ts: int) -> pd.DataFrame:
        if symbol != "BTC-USD":
            return daily_frame(["2026-11-19", "2026-11-20"])
        attempts["BTC-USD"] += 1  # the source rolls the day over late: the first two answers lack it
        return daily_frame(["2026-11-19", "2026-11-20"] if attempts["BTC-USD"] >= 3 else ["2026-11-19"])

    env.provider.daily_for = daily
    env.clock.set(MIDNIGHT - MIN + 3)
    await env.service.prepare()

    await run_until(env, MIDNIGHT - MIN + 3, MIDNIGHT + 30 * MIN)

    first = MIDNIGHT + MIN + 3
    assert calls_at(env, "BTC-USD", "daily") == [first, first + MIN, first + 3 * MIN]  # pauses 1 and 2 minutes
    assert calls_at(env, "EURUSD=X", "daily") == [first]  # stored at once: no retry
    assert repo_market.last_day(env.db, "BTC-USD") == "2026-11-20"


async def test_the_prompt_daily_bar_is_replaced_by_the_final_one_at_the_regular_offset(env: Env) -> None:
    """The prompt request may get the source's provisional values of the day's last minutes; a second
    request at the regular 30-minute offset stores the final values (the replay reads those as well)."""
    final_after = MIDNIGHT + 30 * MIN

    def daily(symbol: str, start_ts: int) -> pd.DataFrame:
        frame = daily_frame(["2026-11-19", "2026-11-20"])
        if env.clock.ts() < final_after:  # the last minutes of the 20th are not in the source's bar yet
            frame.loc["2026-11-20", "close"] = frame.loc["2026-11-20", "close"] - 1.0
        return frame

    env.provider.daily_for = daily
    env.clock.set(MIDNIGHT - MIN + 3)
    await env.service.prepare()

    await run_until(env, MIDNIGHT - MIN + 3, MIDNIGHT + 50 * MIN)

    for symbol in UTC_DAY_SYMBOLS:
        assert calls_at(env, symbol, "daily") == [MIDNIGHT + MIN + 3, final_after + 3]  # prompt, then final
    stored = repo_market.read_bars_1d(env.db, "BTC-USD")
    assert stored.loc["2026-11-20", "close"] == daily_frame(["2026-11-19", "2026-11-20"]).loc["2026-11-20", "close"]
    assert calls_at(env, "SAP.DE", "daily") == []  # venues with a real close are not affected


async def test_daily_retries_give_up_after_the_limit(make_env: EnvFactory) -> None:
    env = make_env(options=replace(TEST_OPTIONS, daily_retry_limit=3))
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19"])  # never delivers the 20th
    env.clock.set(MIDNIGHT - MIN + 3)
    await env.service.prepare()

    await run_until(env, MIDNIGHT - MIN + 3, MIDNIGHT + 40 * MIN)

    first = MIDNIGHT + MIN + 3
    # three attempts with growing pauses, then only the regular (confirming) request of the day
    assert calls_at(env, "BTC-USD", "daily") == [first, first + MIN, first + 3 * MIN, MIDNIGHT + 30 * MIN + 3]
    assert repo_market.last_day(env.db, "BTC-USD") == "2026-11-19"


async def test_a_daily_bar_found_missing_at_start_is_retried_if_the_backfill_did_not_store_it(env: Env) -> None:
    saturday = to_ts(datetime(2026, 11, 21, 10, 0, tzinfo=UTC))
    repo_market.upsert_bars_1d(env.db, "BTC-USD", daily_frame(["2026-11-18", "2026-11-19"]))
    repo_market.upsert_bars_1d(env.db, "EURUSD=X", daily_frame(["2026-11-19", "2026-11-20"]))
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20", "2026-11-21"])
    env.clock.set(saturday + 3)
    await env.service.prepare()

    await run_until(env, saturday + 3, saturday + 40 * MIN)

    assert calls_at(env, "BTC-USD", "daily") == [saturday + 30 * MIN + 3]  # the backfill gets 30 minutes
    assert calls_at(env, "EURUSD=X", "daily") == []  # complete: nothing to do
    assert repo_market.last_day(env.db, "BTC-USD") == "2026-11-20"


async def test_venues_with_a_real_close_keep_the_30_minute_offset(env: Env) -> None:
    close = to_ts(datetime(2026, 11, 20, 16, 30, tzinfo=UTC))  # Xetra/London, feed delay 15 minutes
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20"])
    env.clock.set(close + 3)
    await env.service.prepare()

    await run_until(env, close + 3, close + 50 * MIN)

    assert set(calls_at(env, "SAP.DE", "daily")) == {close + 45 * MIN + 3}


# ------------------------------------------------------------------------------------------------
# Start-up off the event loop (RUN-4)
# ------------------------------------------------------------------------------------------------
async def test_prepare_keeps_calendar_work_off_the_event_loop(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.get_ident()
    threads: list[int] = []
    original = env.calendar.last_completed_session

    def recording(exchange: str, at: datetime) -> date | None:
        threads.append(threading.get_ident())
        return original(exchange, at)

    monkeypatch.setattr(env.calendar, "last_completed_session", recording)
    env.clock.set(T0)

    await env.service.prepare()

    assert threads  # the schedules were initialised ...
    assert loop_thread not in threads  # ... without blocking the loop
