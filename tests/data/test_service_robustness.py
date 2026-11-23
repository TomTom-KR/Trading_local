"""Edge cases of MarketDataService found in review: delayed feeds, REST pauses, downtime gaps, failure isolation."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tests.data.conftest import Env, EnvFactory
from tests.data.fakes import daily_frame, minute_frame
from tests.data.test_service_backfill import DAYS, OLDEST, sparse_minutes, wire
from trading_local.clock import floor_ts, to_ts
from trading_local.data.provider import RateLimitError
from trading_local.db import repo_market
from trading_local.db.repo_market import BAR_COLUMNS
from trading_local.testing import make_settings, seed_bars

T0 = to_ts(datetime(2026, 11, 20, 15, 0, tzinfo=UTC))  # Friday
DAY = 86_400
SNAPSHOT_PRICE = 555.0


def yahoo_live(env: Env, symbol: str, start_ts: int, end_ts: int | None) -> pd.DataFrame:
    """Like Yahoo for a live market: closed bars up to the minute that contains the source's newest price
    (now - feed delay) and, for that minute, a flat zero-volume snapshot instead of a bar."""
    instrument = env.instruments.require(symbol)
    live_minute = floor_ts(env.clock.ts() - instrument.delay_min * 60)
    end = live_minute if end_ts is None else min(end_ts, live_minute)
    closed = minute_frame(start_ts, end)
    if end_ts is not None and end_ts <= live_minute:
        return closed
    snapshot = pd.DataFrame(
        [[SNAPSHOT_PRICE] * 4 + [0.0]], index=pd.Index([live_minute], dtype=np.int64, name="ts"), columns=list(BAR_COLUMNS)
    )
    return pd.concat([closed, snapshot])


async def run_minutes(env: Env, start_ts: int, minutes: int, *, offset: int = 3) -> None:
    for minute in range(minutes):
        env.clock.set(start_ts + minute * 60 + offset)
        await env.service.refresh(env.clock.now())


# ------------------------------------------------------------------------------------------------
# Delayed feeds
# ------------------------------------------------------------------------------------------------
async def test_forming_bar_of_a_delayed_feed_is_never_stored(env: Env) -> None:
    env.provider.minutes_for = lambda symbol, start, end, prepost: yahoo_live(env, symbol, start, end)
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 10)
    for symbol in ("SAP.DE", "SHEL.L", "^GDAXI", "NVDA", "BTC-USD"):
        bars = repo_market.read_bars_1m(env.db, symbol)
        assert not bars.empty, symbol
        assert (bars["close"] != SNAPSHOT_PRICE).all(), symbol  # the flat snapshot row is no bar


async def test_reconciliation_window_covers_twenty_minutes_of_delayed_data(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 6)
    for call in env.provider.calls_of("minutes"):
        delays = {env.instruments.require(symbol).delay_min for symbol in call.symbols}
        assert len(delays) == 1  # one window per feed delay
        assert call.start_ts == floor_ts(call.at) - 1_200 - delays.pop() * 60


# ------------------------------------------------------------------------------------------------
# REST pause
# ------------------------------------------------------------------------------------------------
async def test_session_final_and_daily_bars_survive_a_rest_pause(env: Env) -> None:
    close = to_ts(datetime(2026, 11, 20, 16, 30, tzinfo=UTC))
    start = close + 33 * 60  # 17:03: the final is due at 17:05, the daily bars at 17:15
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20"])
    env.provider.paused = close + 80 * 60  # REST paused until 17:50
    env.clock.set(start)
    await env.service.prepare()
    await run_minutes(env, start, 60)  # 17:03 .. 18:02
    finals = [call for call in env.provider.calls_of("minutes") if call.prepost]
    assert [set(call.symbols) for call in finals] == [{"SAP.DE", "^GDAXI", "SHEL.L"}]
    assert finals[0].at >= env.provider.paused and finals[0].start_ts == close - 1_800
    assert {"SAP.DE", "^GDAXI", "SHEL.L"} <= set(env.provider.requested("daily"))
    assert repo_market.last_day(env.db, "SAP.DE") == "2026-11-20"


async def test_jobs_refused_by_a_new_rate_limit_are_retried_after_the_pause(env: Env) -> None:
    close = to_ts(datetime(2026, 11, 20, 16, 30, tzinfo=UTC))
    start = close + 35 * 60  # 17:05: the session-final fetch is due now
    env.clock.set(start)
    await env.service.prepare()
    original = env.provider.fetch_minutes

    async def limited_once(symbols, *, start_ts, end_ts=None, prepost=False):  # type: ignore[no-untyped-def]
        if prepost and not env.provider.trips:
            env.provider.trip("429")
            raise RateLimitError("limited")
        return await original(symbols, start_ts=start_ts, end_ts=end_ts, prepost=prepost)

    env.provider.fetch_minutes = limited_once  # type: ignore[method-assign]
    await run_minutes(env, start, 15)  # the pause lasts 10 minutes
    finals = [call for call in env.provider.calls_of("minutes") if call.prepost]
    assert [set(call.symbols) for call in finals] == [{"SAP.DE", "^GDAXI", "SHEL.L"}]


# ------------------------------------------------------------------------------------------------
# Backfill after downtime and failure isolation
# ------------------------------------------------------------------------------------------------
async def test_backfill_fills_the_gap_left_by_a_long_downtime(env: Env) -> None:
    wire(env)
    seed_bars(env.db, "NVDA", sparse_minutes(OLDEST, floor_ts(T0) - 12 * DAY))  # the app was off for 12 days
    repo_market.upsert_bars_1d(env.db, "NVDA", daily_frame(DAYS[:-1]))
    first, last = repo_market.bar_range(env.db, "NVDA")  # type: ignore[misc]
    await env.service.run_backfill()
    windows = sorted((c.start_ts, c.end_ts or c.at) for c in env.provider.calls_of("minutes") if "NVDA" in c.symbols)
    covered = last
    for start, end in windows:
        if start <= covered:
            covered = max(covered, end)
    assert covered >= floor_ts(T0) - 60, "a hole remains between the old data and the recent window"
    assert all(start >= first - 600 for start, _ in windows)  # nothing older than the stored data is re-fetched


async def test_a_delayed_backfill_leaves_no_hole(env: Env) -> None:
    wire(env)
    base = env.provider.minutes_for

    def slow(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        frame = base(symbol, start_ts, end_ts, prepost)
        env.clock.advance(1_800)  # every request "takes" half an hour (rate-limit pauses in between)
        return frame

    env.provider.minutes_for = slow
    await env.service.run_backfill()
    for symbol in ("BTC-USD", "EURUSD=X", "EURJPY=X"):  # 24/7 instruments: any hole is a real one
        windows = sorted((c.start_ts, c.end_ts or c.at) for c in env.provider.calls_of("minutes") if symbol in c.symbols)
        covered = windows[0][0]
        assert covered == OLDEST
        for start, end in windows:
            assert start <= covered, f"{symbol}: nothing fetched between {covered} and {start}"
            covered = max(covered, end)


async def test_one_failing_history_job_does_not_stop_the_backfill(env: Env) -> None:
    wire(env)

    def daily(symbol: str, start_ts: int) -> pd.DataFrame:
        if symbol == "NVDA":
            raise KeyError("unexpected answer shape")
        return daily_frame(DAYS)

    env.provider.daily_for = daily
    await env.service.run_backfill()
    assert {s for s in env.provider.requested("minutes")} == {i.symbol for i in env.instruments.all()}
    assert repo_market.last_day(env.db, "BTC-USD") == "2026-11-19"
    assert env.service.status().backfill_progress is None


async def test_lookup_never_raises(env: Env) -> None:
    env.provider.fail = RuntimeError("unexpected")
    assert await env.service.lookup("AAPL") is None


async def test_offline_lookup_makes_no_request(make_env: EnvFactory, tmp_path: Path) -> None:
    env = make_env(settings=make_settings(tmp_path, offline=True))
    assert await env.service.lookup("AAPL") is None
    assert env.provider.calls == []


# ------------------------------------------------------------------------------------------------
# Event-loop hygiene and life cycle
# ------------------------------------------------------------------------------------------------
async def test_refresh_after_stop_does_nothing(make_env: EnvFactory) -> None:
    env = make_env()
    env.clock.set(T0)
    await env.service.prepare()
    await env.service.stop()
    env.clock.set(T0 + 63)
    assert await env.service.refresh(env.clock.now()) == []
    assert env.provider.calls == []



async def test_refresh_does_not_wait_for_a_slow_snapshot_request(make_env: EnvFactory) -> None:
    env = make_env()
    release = asyncio.Event()
    original = env.provider.fetch_quotes

    async def slow(symbols):  # type: ignore[no-untyped-def]
        await release.wait()
        return await original(symbols)

    env.provider.fetch_quotes = slow  # type: ignore[method-assign]
    env.clock.set(T0)
    await env.service.start()
    await asyncio.sleep(0.01)  # the background quote loop is now waiting for its answer
    env.clock.set(T0 + 63)
    try:
        await asyncio.wait_for(env.service.refresh(env.clock.now()), timeout=1.0)
    finally:
        release.set()
