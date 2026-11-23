"""Background history download: ordering, skipping of covered history, restart safety, ensure_history."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from tests.data.conftest import Env, EnvFactory
from tests.data.fakes import daily_frame
from trading_local.clock import floor_ts, to_ts
from trading_local.data.provider import RateLimitError
from trading_local.db import repo_market, repo_ops
from trading_local.db.repo_market import BAR_COLUMNS
from trading_local.models import Instrument
from trading_local.testing import seed_bars

T0 = to_ts(datetime(2026, 11, 20, 15, 0, tzinfo=UTC))  # Friday
DAY = 86_400
WINDOW = 8 * DAY - 60
OLDEST = floor_ts(T0) - 29 * DAY
DAYS = ["2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20"]


def sparse_minutes(start_ts: int, end_ts: int) -> pd.DataFrame:
    """A few bars per window (first minute, every 6 hours, the last two minutes) keep the tests fast."""
    first = -(-start_ts // 60) * 60
    ts = np.unique(np.r_[np.arange(first, end_ts, 6 * 3600), end_ts - 120, end_ts - 60]).astype(np.int64)
    ts = ts[(ts >= first) & (ts < end_ts)]
    values = np.full((len(ts), len(BAR_COLUMNS)), 100.0)
    return pd.DataFrame(values, index=pd.Index(ts, name="ts"), columns=list(BAR_COLUMNS))


def wire(env: Env) -> None:
    def minutes(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        end = end_ts if end_ts is not None else floor_ts(env.clock.ts()) + 60  # includes the forming bar
        return sparse_minutes(start_ts, end)

    env.provider.minutes_for = minutes
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(DAYS)
    env.clock.set(T0)


async def test_backfill_order_windows_and_completion(env: Env) -> None:
    wire(env)
    await env.service.run_backfill()
    kinds = ["daily" if call.kind == "daily" else ("recent" if call.end_ts is None else "older") for call in env.provider.calls]
    assert kinds == sorted(kinds, key=["daily", "recent", "older"].index)  # daily, then recent, then older
    daily = env.provider.calls_of("daily")
    assert {s for call in daily for s in call.symbols} == {i.symbol for i in env.instruments.all()}
    assert all(call.start_ts <= T0 - 5 * 365 * DAY for call in daily)
    recent = [call for call in env.provider.calls_of("minutes") if call.end_ts is None]
    assert recent[0].symbols == ("NVDA", "SAP.DE", "SHEL.L", "7203.T")  # tradable first, then sort order
    assert {s for call in recent for s in call.symbols} == {i.symbol for i in env.instruments.all()}
    for call in env.provider.calls_of("minutes"):
        end = call.end_ts if call.end_ts is not None else call.at
        assert end - call.start_ts <= WINDOW and call.start_ts >= OLDEST
        assert call.start_ts % 60 == 0 and len(call.symbols) <= 4
    btc_windows = sorted((c.start_ts, c.end_ts or c.at) for c in env.provider.calls_of("minutes") if "BTC-USD" in c.symbols)
    assert btc_windows[0][0] == OLDEST and len(btc_windows) == 4  # 29 days in four windows
    for (_, earlier_end), (later_start, _) in zip(btc_windows, btc_windows[1:]):
        assert later_start < earlier_end  # overlapping boundaries lose no bar
    span = repo_market.bar_range(env.db, "BTC-USD")
    assert span == (OLDEST, floor_ts(T0) - 60)  # the forming bar was not stored
    assert repo_market.last_day(env.db, "NVDA") == "2026-11-19"  # New York's Friday session is still running
    assert repo_market.last_day(env.db, "7203.T") == "2026-11-20"  # Tokyo closed at 06:30 UTC
    assert env.service.status().backfill_progress is None
    messages = [entry.message for entry in repo_ops.list_activity(env.db, category="data")]
    assert messages and messages[0].startswith("Historische Kursdaten geladen")


async def test_restart_skips_what_the_database_covers(env: Env) -> None:
    wire(env)
    await env.service.run_backfill()
    first_run = sum(call.requests for call in env.provider.calls)
    env.provider.calls.clear()
    await env.service.run_backfill()
    assert first_run > 30 and env.provider.calls == []
    env.clock.set(T0 + 3_600)  # one hour later: only the recent tails of venues that traded meanwhile
    await env.service.run_backfill()
    tails = {s for call in env.provider.calls for s in call.symbols}
    assert tails == {"NVDA", "^GSPC", "SAP.DE", "^GDAXI", "SHEL.L", "BTC-USD", "EURUSD=X", "EURGBP=X", "EURJPY=X"}
    assert all(call.kind == "minutes" and call.end_ts is None for call in env.provider.calls)
    assert all(call.start_ts >= floor_ts(T0) - 60 - 600 for call in env.provider.calls)


async def test_partially_covered_symbol_only_gets_the_missing_windows(env: Env) -> None:
    wire(env)
    seed_bars(env.db, "NVDA", sparse_minutes(OLDEST, floor_ts(T0)))
    repo_market.upsert_bars_1d(env.db, "NVDA", daily_frame(["2026-11-18", "2026-11-19"]))
    await env.service.run_backfill()
    assert "NVDA" not in env.provider.requested("daily") and "NVDA" not in env.provider.requested("minutes")


async def test_ensure_history_for_a_new_symbol(make_env: EnvFactory) -> None:
    env = make_env()
    wire(env)
    repo_market.upsert_instruments(env.db, [Instrument("MSFT", "Microsoft", "equity", "US", "XNAS", "USD")])
    env.instruments.reload()
    await env.service.ensure_history("MSFT")
    assert {s for call in env.provider.calls for s in call.symbols} == {"MSFT"}
    assert [call.kind for call in env.provider.calls][0] == "daily"
    assert repo_market.bar_range(env.db, "MSFT") is not None and repo_market.last_day(env.db, "MSFT") == "2026-11-19"
    await env.service.ensure_history("UNKNOWN")  # not in the book: logged, nothing requested


async def test_refused_history_job_is_skipped_without_spinning(env: Env) -> None:
    wire(env)
    env.provider.fail = RateLimitError("refused")  # refused although no pause is reported
    await env.service.run_backfill()
    assert env.provider.calls == []
    assert env.service.status().backfill_progress is None
