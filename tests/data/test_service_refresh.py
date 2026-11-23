"""MarketDataService.refresh scheduling, storage rules and events against a ManualClock and the FakeCalendar.

FakeCalendar (UTC): XNAS/XNYS 14:30-21:00, XETR/XLON 08:00-16:30, XTKS 00:00-02:30 + 03:30-06:30, CRYPTO and FX
always open; weekends closed. The clock starts on Friday 2026-11-20 15:00 UTC.
"""

import asyncio
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.data.conftest import Env, EnvFactory
from tests.data.fakes import daily_frame, minute_frame
from trading_local.clock import floor_ts, from_ts, to_ts
from trading_local.data.provider import RateLimitError, Tick
from trading_local.db import repo_market, repo_ops
from trading_local.events import TOPIC_BAR, TOPIC_QUOTES, Event, Subscription
from trading_local.models import Instrument, Quote
from trading_local.testing import make_settings
from trading_local.universe import DEFAULT_INSTRUMENTS

T0 = to_ts(datetime(2026, 11, 20, 15, 0, tzinfo=UTC))
OPEN_SYMBOLS = {"NVDA", "SAP.DE", "SHEL.L", "BTC-USD", "^GDAXI", "^GSPC", "EURUSD=X", "EURGBP=X", "EURJPY=X"}


async def run_minutes(env: Env, start_ts: int, minutes: int, *, offset: int = 3) -> list[list[str]]:
    """Call refresh once per minute (``offset`` seconds after the boundary); returns the 'updated' lists."""
    results = []
    for minute in range(minutes):
        env.clock.set(start_ts + minute * 60 + offset)
        results.append(await env.service.refresh(env.clock.now()))
    return results


async def drain(subscription: Subscription) -> list[Event]:
    """All events delivered so far (publishing from the database thread needs a loop turn)."""
    events = []
    while True:
        try:
            events.append(await asyncio.wait_for(subscription.get(), timeout=0.05))
        except TimeoutError:
            return events


def stored(env: Env, symbol: str) -> pd.DataFrame:
    return repo_market.read_bars_1m(env.db, symbol)


def source_of(env: Env, symbol: str, ts: int) -> str:
    return env.db.scalar("SELECT source FROM bars_1m WHERE symbol = ? AND ts = ?", (symbol, ts))


# ------------------------------------------------------------------------------------------------
# Which symbols are requested when
# ------------------------------------------------------------------------------------------------
async def test_round_robin_covers_open_venues_every_five_minutes(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 10)
    counts = Counter(env.provider.requested("minutes"))
    assert set(counts) == OPEN_SYMBOLS  # 7203.T: Tokyo closed since 06:30
    assert all(1 <= count <= 3 for count in counts.values())
    for call in env.provider.calls_of("minutes"):
        assert len(call.symbols) <= 2  # ceil(9 eligible / 5 minutes)
        delay_s = env.instruments.require(call.symbols[0]).delay_min * 60  # the last 20 minutes of the feed
        assert call.start_ts == floor_ts(call.at) - 1_200 - delay_s and isinstance(call.start_ts, int)
        assert call.end_ts is None and not call.prepost
    assert len(env.provider.calls_of("quotes")) == 10  # one snapshot per refresh here (cadence 15 s)
    assert set(env.provider.calls_of("quotes")[0].symbols) == {i.symbol for i in env.instruments.all()}


async def test_closed_venues_are_left_alone(env: Env) -> None:
    saturday = to_ts(datetime(2026, 11, 21, 12, 0, tzinfo=UTC))
    env.clock.set(saturday)
    await env.service.prepare()
    await run_minutes(env, saturday, 8)
    assert set(env.provider.requested("minutes")) == {"BTC-USD", "EURUSD=X", "EURGBP=X", "EURJPY=X"}
    assert env.provider.calls_of("daily") == []


async def test_venues_stay_eligible_for_30_minutes_plus_delay_after_the_close(env: Env) -> None:
    after_close = to_ts(datetime(2026, 11, 20, 16, 50, tzinfo=UTC))  # Xetra/London closed 16:30, delay 15
    env.clock.set(after_close)
    await env.service.prepare()
    await run_minutes(env, after_close, 10)  # 16:50 .. 16:59
    regular = [s for call in env.provider.calls_of("minutes") if not call.prepost for s in call.symbols]
    assert {"SAP.DE", "^GDAXI", "SHEL.L"} <= set(regular)
    env.provider.calls.clear()
    late = to_ts(datetime(2026, 11, 20, 17, 16, tzinfo=UTC))  # 16:30 + 30 + 15 = 17:15
    await run_minutes(env, late, 10)
    regular = [s for call in env.provider.calls_of("minutes") if not call.prepost for s in call.symbols]
    assert not {"SAP.DE", "^GDAXI", "SHEL.L"} & set(regular)


async def test_session_final_fetch_with_closing_print(env: Env) -> None:
    close = to_ts(datetime(2026, 11, 20, 16, 30, tzinfo=UTC))
    start = close + 15 * 60 + 18 * 60  # 17:03

    def yahoo_like(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        frame = env.provider._default_minutes(symbol, start_ts, end_ts, prepost)
        # regular bars end before the close; pre/post data adds the closing print and after-hours rows
        return frame if prepost else frame[frame.index < close]

    env.provider.minutes_for = yahoo_like
    env.clock.set(start)
    await env.service.prepare()
    await run_minutes(env, start, 6)  # 17:03 .. 17:08: due at 16:30 + 15 + 20 = 17:05
    finals = [call for call in env.provider.calls_of("minutes") if call.prepost]
    assert len(finals) == 1
    assert set(finals[0].symbols) == {"SAP.DE", "^GDAXI", "SHEL.L"}
    assert finals[0].start_ts == close - 1_800 and finals[0].at == close + 35 * 60 + 3
    sap = stored(env, "SAP.DE")
    assert sap.index.max() == close  # the closing print stamped at the session end, nothing after it
    await run_minutes(env, start + 6 * 60, 30)
    assert len([call for call in env.provider.calls_of("minutes") if call.prepost]) == 1  # once per close


# ------------------------------------------------------------------------------------------------
# Request budget
# ------------------------------------------------------------------------------------------------
def big_universe() -> list[Instrument]:
    instruments = []
    for index in range(45):
        instruments.append(Instrument(f"US{index}", f"US {index}", "equity", "US", "XNYS", "USD", timezone="America/New_York"))
    for index in range(45):
        instruments.append(
            Instrument(f"DE{index}", f"DE {index}", "equity", "EU", "XETR", "EUR", timezone="Europe/Berlin", delay_min=15)
        )
    for index in range(10):
        instruments.append(Instrument(f"C{index}-USD", f"C {index}", "crypto", "GLOBAL", "CRYPTO", "USD"))
    for index in range(15):
        instruments.append(Instrument(f"EURX{index}=X", f"FX {index}", "fx", "GLOBAL", "FX", "USD", tradable=False))
    return instruments


async def test_request_budget_is_never_exceeded(make_env: EnvFactory) -> None:
    env = make_env(big_universe())
    sparse = minute_frame  # one small frame per symbol keeps the test fast

    def few_bars(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        now_minute = floor_ts(env.clock.ts())
        return sparse(now_minute - 120, now_minute + 60)

    env.provider.minutes_for = few_bars
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20"])
    start = to_ts(datetime(2026, 11, 20, 16, 0, tzinfo=UTC))
    env.clock.set(start)
    await env.service.prepare()
    await run_minutes(env, start, 100)  # 16:00 .. 17:39: Xetra close, finals, daily bars, US session
    per_minute: Counter[int] = Counter()
    for call in env.provider.calls:
        per_minute[call.at // 60] += call.requests
    assert max(per_minute.values()) <= 26
    finals = {s for call in env.provider.calls_of("minutes") if call.prepost for s in call.symbols}
    assert finals == {f"DE{index}" for index in range(45)}  # a venue larger than one minute's budget
    assert {s for s in env.provider.requested("daily")} >= {f"DE{index}" for index in range(45)}


# ------------------------------------------------------------------------------------------------
# Storage rules and the 'updated' result
# ------------------------------------------------------------------------------------------------
async def test_only_closed_bars_are_stored_and_updated_lists_new_bars(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.clock.set(T0 + 3)
    updated = await env.service.refresh(env.clock.now())
    requested = env.provider.requested("minutes")
    assert requested and updated == sorted(requested)
    for symbol in requested:
        bars = stored(env, symbol)
        assert bars.index.max() == T0 - 60  # the forming 15:00 bar was dropped
        assert source_of(env, symbol, T0 - 60) == "rest"
    env.clock.set(T0 + 63)
    second = await env.service.refresh(env.clock.now())
    assert not set(second) & set(requested)  # nothing new for them: not reported again


async def test_fx_and_crypto_rest_bars_carry_no_volume(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 6)
    assert (stored(env, "BTC-USD")["volume"] == 0).all()
    assert (stored(env, "EURUSD=X")["volume"] == 0).all()
    assert (stored(env, "NVDA")["volume"] > 0).all()


async def test_outage_extends_the_reconciliation_window(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 6)
    first = {s: call for call in env.provider.calls_of("minutes") for s in call.symbols}
    env.provider.calls.clear()
    env.provider.paused = T0 + 6 * 60 + 2_400  # 40 minutes without REST
    await run_minutes(env, T0 + 6 * 60, 45)
    resumed = env.provider.calls_of("minutes")
    assert resumed and all(call.at >= env.provider.paused for call in resumed)
    btc = next(call for call in resumed if "BTC-USD" in call.symbols)
    assert btc.start_ts <= first["BTC-USD"].at - 600 + 60  # from the last good window minus the overlap


# ------------------------------------------------------------------------------------------------
# Stream, quotes and precedence
# ------------------------------------------------------------------------------------------------
async def test_stream_bars_are_replaced_by_rest_bars(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.provider.minutes_for = lambda symbol, start, end, prepost: None  # REST has nothing yet
    service = env.service
    env.clock.set(T0 + 10)
    service.on_tick(Tick("NVDA", T0 + 5, 100.0, T0 + 6, day_volume=1_000))
    service.on_tick(Tick("NVDA", T0 + 30, 101.0, T0 + 31, day_volume=1_500))
    env.clock.set(T0 + 66)
    service.on_tick(Tick("NVDA", T0 + 65, 102.0, T0 + 66, day_volume=1_700))
    env.clock.set(T0 + 70)
    assert await service.refresh(env.clock.now()) == ["NVDA"]
    bar = stored(env, "NVDA").loc[T0]
    assert source_of(env, "NVDA", T0) == "stream" and bar["volume"] == 500 and bar["close"] == 101.0

    rest_bar = minute_frame(T0, T0 + 60, price=200.0)
    env.provider.minutes_for = lambda symbol, start, end, prepost: rest_bar if symbol == "NVDA" else None
    for minute in range(1, 8):
        env.clock.set(T0 + minute * 60 + 15)
        await service.refresh(env.clock.now())
        if "NVDA" in env.provider.requested():
            break
    assert source_of(env, "NVDA", T0) == "rest" and stored(env, "NVDA").loc[T0, "open"] == pytest.approx(200.0)
    service.on_tick(Tick("NVDA", T0 + 50, 999.0, env.clock.ts()))  # late stream correction of minute T0
    await service.refresh(env.clock.now())
    assert source_of(env, "NVDA", T0) == "rest" and stored(env, "NVDA").loc[T0, "high"] < 999.0


async def test_quote_sampled_bars_for_symbols_without_stream_and_stream_precedence(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.provider.minutes_for = lambda symbol, start, end, prepost: None
    service = env.service
    for offset, price in ((5, 5_000.0), (25, 5_010.0), (50, 4_990.0), (65, 5_005.0)):
        env.provider.quotes = {"^GSPC": Quote("^GSPC", T0 + offset, T0 + offset + 1, price, prev_close=4_950.0)}
        env.clock.set(T0 + offset + 1)
        await service.poll_quotes(env.clock.ts())
    env.clock.set(T0 + 70)
    await service.refresh(env.clock.now())
    bar = stored(env, "^GSPC").loc[T0]
    assert source_of(env, "^GSPC", T0) == "quote"
    assert (bar["open"], bar["high"], bar["low"], bar["close"]) == (5_000.0, 5_010.0, 4_990.0, 4_990.0)
    assert bar["volume"] == 0.0  # index: no session volume
    # a stream bar for the same minute ranks higher than the quote-sampled bar
    service.on_tick(Tick("^GSPC", T0 + 40, 5_020.0, env.clock.ts()))
    env.clock.set(T0 + 130)
    await service.refresh(env.clock.now())
    assert source_of(env, "^GSPC", T0) == "stream"
    persisted = repo_market.load_quotes(env.db)["^GSPC"]  # the older tick did not overwrite the newer quote
    assert (persisted.ts, persisted.price) == (T0 + 65, 5_005.0)
    assert persisted.change_pct == pytest.approx((5_005.0 / 4_950.0 - 1) * 100)


async def test_old_replays_and_closed_venue_ticks_build_no_bars(env: Env) -> None:
    env.clock.set(T0 + 10)
    await env.service.prepare()
    env.provider.minutes_for = lambda symbol, start, end, prepost: None
    service = env.service
    service.on_tick(Tick("7203.T", T0 - 3_600, 2_900.0, T0 + 5))  # Tokyo closed at that time
    service.on_tick(Tick("NVDA", T0 - 2 * 86_400, 170.0, T0 + 5))  # replay with a 2-day-old timestamp
    service.on_tick(Tick("UNKNOWN", T0 + 1, 1.0, T0 + 5))
    env.clock.set(T0 + 300)
    assert await service.refresh(env.clock.now()) == []
    assert stored(env, "7203.T").empty and stored(env, "NVDA").empty
    assert service.quote("7203.T") is not None and service.quote("UNKNOWN") is None


async def test_quotes_are_published_coalesced(env: Env) -> None:
    env.bus.bind_loop()
    subscription = env.bus.subscribe()
    env.provider.quotes = {
        "NVDA": Quote("NVDA", T0, T0 + 1, 180.0, prev_close=178.0),
        "SAP.DE": Quote("SAP.DE", T0, T0 + 1, 200.0),
    }
    await env.service.poll_quotes(T0)
    assert env.service.publish_quotes() == 2
    assert env.service.publish_quotes() == 0
    event = await asyncio.wait_for(subscription.get(), timeout=1)
    assert event.topic == TOPIC_QUOTES
    nvda = next(item for item in event.data["items"] if item["symbol"] == "NVDA")
    assert nvda["price"] == 180.0 and nvda["change_pct"] == pytest.approx((180 / 178 - 1) * 100)
    subscription.close()


async def test_bar_events_for_stored_bars(env: Env) -> None:
    env.bus.bind_loop()
    subscription = env.bus.subscribe()
    env.clock.set(T0)
    await env.service.prepare()
    env.clock.set(T0 + 3)
    await env.service.refresh(env.clock.now())
    bars = [event for event in await drain(subscription) if event.topic == TOPIC_BAR]
    assert bars and all(event.data["bar"][0] + 60 <= T0 + 3 for event in bars)
    assert {event.data["source"] for event in bars} == {"rest"}
    assert len(bars[0].data["bar"]) == 6
    subscription.close()


# ------------------------------------------------------------------------------------------------
# Circuit breaker, quarantine, daily bars
# ------------------------------------------------------------------------------------------------
async def test_rate_limit_pause_stops_requests_and_is_logged(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.provider.paused = T0 + 600
    await run_minutes(env, T0, 5)
    assert env.provider.calls == []
    env.clock.set(T0 + 601)
    await env.service.refresh(env.clock.now())
    assert env.provider.calls_of("minutes")
    messages = [entry.message for entry in repo_ops.list_activity(env.db, category="data")]
    assert any("pausiert bis" in message for message in messages)
    assert messages[0] == "Kursdaten-Abrufe bei Yahoo laufen wieder."
    assert env.service.status().rate_limited_until is None


async def test_rate_limit_during_reconciliation_stops_the_round(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.provider.fail = RateLimitError("limited")
    await env.service.refresh(from_ts(T0 + 3))
    assert env.provider.calls == []


async def test_empty_answers_for_most_open_markets_trip_the_breaker(make_env: EnvFactory) -> None:
    env = make_env(big_universe())
    env.provider.minutes_for = lambda symbol, start, end, prepost: None
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 3)
    assert env.provider.trips == ["no minute bars for most open-market symbols"]
    assert env.service.status().rate_limited_until == env.provider.paused


async def test_empty_fx_and_futures_do_not_trip_the_breaker(make_env: EnvFactory) -> None:
    """Sunday evening: FX is nominally open, but exotic pairs (INR, CNY, KRW ...) have no minute bars at all.
    Crypto delivers. Empty FX/futures answers say nothing about the data source's health."""
    instruments = [Instrument(f"C{i}-USD", f"C {i}", "crypto", "GLOBAL", "CRYPTO", "USD") for i in range(3)]
    instruments += [Instrument(f"EURX{i}=X", f"FX {i}", "fx", "GLOBAL", "FX", "USD", tradable=False) for i in range(15)]
    instruments += [Instrument(f"F{i}=F", f"Future {i}", "future", "US", "CME", "USD", tradable=False) for i in range(5)]
    env = make_env(instruments)

    def crypto_only(symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame | None:
        if not symbol.startswith("C"):
            return None
        now_minute = floor_ts(env.clock.ts())
        return minute_frame(now_minute - 300, now_minute)

    env.provider.minutes_for = crypto_only
    env.clock.set(T0)
    await env.service.prepare()
    await run_minutes(env, T0, 3)
    assert env.provider.trips == []
    assert env.service.status().rate_limited_until is None


async def test_unknown_symbols_are_quarantined_until_they_reappear(env: Env) -> None:
    env.clock.set(T0)
    await env.service.prepare()
    env.provider.quotes = {s: Quote(s, T0, T0, 10.0) for s in OPEN_SYMBOLS - {"SHEL.L"}}
    await run_minutes(env, T0, 2, offset=20)
    env.provider.calls.clear()
    await run_minutes(env, T0 + 120, 8, offset=20)
    assert "SHEL.L" not in env.provider.requested("minutes")
    env.provider.quotes["SHEL.L"] = Quote("SHEL.L", T0, T0, 3_500.0)
    await run_minutes(env, T0 + 600, 8, offset=20)
    assert "SHEL.L" in env.provider.requested("minutes")


async def test_daily_bars_only_for_completed_sessions(env: Env) -> None:
    start = to_ts(datetime(2026, 11, 20, 17, 0, tzinfo=UTC))
    env.clock.set(start)
    await env.service.prepare()
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-18", "2026-11-19", "2026-11-20", "2026-11-23"])
    await run_minutes(env, start, 20)  # 17:00 .. 17:19; Xetra/London session done 16:30 + 30 + 15 = 17:15
    daily_calls = env.provider.calls_of("daily")
    assert {s for call in daily_calls for s in call.symbols} == {"SAP.DE", "^GDAXI", "SHEL.L"}
    assert all(call.at >= to_ts(datetime(2026, 11, 20, 17, 15, tzinfo=UTC)) for call in daily_calls)
    assert all(call.start_ts == call.at - 10 * 86_400 for call in daily_calls)
    assert repo_market.last_day(env.db, "SAP.DE") == "2026-11-20"  # 2026-11-23 is not completed
    assert repo_market.last_day(env.db, "NVDA") is None  # New York still trading


async def test_crypto_daily_bar_of_the_running_utc_day_is_dropped(env: Env) -> None:
    start = to_ts(datetime(2026, 11, 21, 0, 0, tzinfo=UTC))
    env.clock.set(start)
    await env.service.prepare()
    env.provider.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-19", "2026-11-20", "2026-11-21"])
    await run_minutes(env, start, 32)
    assert "BTC-USD" in env.provider.requested("daily")
    assert repo_market.last_day(env.db, "BTC-USD") == "2026-11-20"


# ------------------------------------------------------------------------------------------------
# Status, lookup, life cycle
# ------------------------------------------------------------------------------------------------
async def test_status_counts_stale_symbols_of_open_venues(env: Env) -> None:
    env.clock.set(T0)
    status = env.service.status()
    assert status.symbols_active == 10 and status.symbols_stale == 9  # all but Tokyo (closed)
    assert not status.stream_connected and status.backfill_progress is None
    env.provider.quotes = {"NVDA": Quote("NVDA", T0 - 30, T0, 180.0), "SAP.DE": Quote("SAP.DE", T0 - 900, T0, 200.0)}
    await env.service.poll_quotes(T0)
    status = env.service.status()
    assert status.symbols_stale == 7  # SAP.DE is 15 minutes behind, within its 15 min delay + 5
    assert status.last_rest_ok_ts == T0


async def test_lookup_validates_and_quarantines_unknown_symbols(env: Env) -> None:
    env.provider.instruments = {
        "AZN.L": Instrument("AZN.L", "AstraZeneca PLC", "equity", "EU", "XLON", "GBP", price_scale=0.01,
                            timezone="Europe/London", delay_min=15, benchmark="^STOXX50E"),
        "MC.PA": Instrument("MC.PA", "LVMH", "equity", "EU", "XPAR", "EUR", timezone="Europe/Paris"),
    }  # fmt: skip
    instrument = await env.service.lookup(" azn.l ")
    assert instrument is not None and instrument.sort_order == 2_900 and instrument.price_scale == 0.01
    assert await env.service.lookup("MC.PA") is None  # venue unknown to this calendar
    assert await env.service.lookup("NOPE") is None
    assert await env.service.lookup("NOPE") is None
    assert await env.service.lookup("NOPE") is None
    assert env.provider.requested("lookup").count("NOPE") == 2


async def test_start_subscribes_the_stream_and_logs_state_changes(make_env: EnvFactory, tmp_path: Path) -> None:
    env = make_env(settings=make_settings(tmp_path, offline=False, enable_stream=True))
    env.clock.set(T0)
    await env.service.start()
    stream = env.provider.stream
    assert stream is not None and stream.connect_calls == 1
    assert set(stream.subscriptions[-1]) == {i.symbol for i in env.instruments.all()}
    stream.set_connected(True)
    assert env.service.status().stream_connected
    stream.set_connected(False)
    await env.service.stop()
    assert stream.closed and env.provider.closed
    messages = [entry.message for entry in repo_ops.list_activity(env.db, category="data")]
    assert "Kursstream Yahoo verbunden." in messages  # the source is named (several sources run side by side)
    assert any(message.startswith("Kursstream Yahoo getrennt") for message in messages)


async def test_offline_start_seeds_the_universe_and_loads_quotes(make_env: EnvFactory, tmp_path: Path) -> None:
    env = make_env([], settings=make_settings(tmp_path, offline=True))
    repo_market.upsert_quotes(env.db, [Quote("NVDA", T0, T0, 180.0)])
    await env.service.start()
    assert len(env.instruments.all()) == len(DEFAULT_INSTRUMENTS)
    assert env.service.quote("NVDA").price == 180.0  # type: ignore[union-attr]
    assert env.provider.calls == [] and env.provider.stream is None
    assert await env.service.refresh(from_ts(T0 + 60)) == []


async def test_refresh_never_raises(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database gone")

    monkeypatch.setattr(env.service, "_write_rest_frames", broken)
    env.clock.set(T0 + 3)
    assert await env.service.refresh(env.clock.now()) == []


def test_frames_helpers_are_canonical() -> None:
    frame = minute_frame(T0 + 30, T0 + 181)
    assert frame.index.tolist() == [T0 + 60, T0 + 120, T0 + 180] and frame.index.dtype == np.int64
