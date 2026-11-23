"""The Kraken lane of the market-data service (candle source) and the routing between Yahoo and Kraken."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.data.fakes import FakeProvider, FakeStream, daily_frame
from trading_local.barstore import BarReader
from trading_local.clock import ManualClock, floor_ts, from_ts, to_ts
from trading_local.data.provider import BarHandler, CandleProvider, StateHandler, Tick, TickHandler
from trading_local.data.service import (
    KRAKEN_OPTIONS,
    UTC_DAY_VENUES,
    DataOptions,
    InstrumentView,
    Lane,
    MarketDataService,
    RoutedMarketData,
)
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS, InstrumentBook
from trading_local.events import EventBus
from trading_local.markets import ExchangeCalendar
from trading_local.models import Bar, Instrument, Quote
from trading_local.testing import SAMPLE_INSTRUMENTS, make_settings, seed_instruments
from trading_local.universe import DEFAULT_INSTRUMENTS

T0 = to_ts(datetime(2026, 11, 22, 12, 0, tzinfo=UTC))  # a Sunday: only the 24/7 venues trade
DAY = 86_400
KRAKEN_TEST = replace(KRAKEN_OPTIONS, rest_offset_s=0, backfill_pause_s=0)
YAHOO_TEST = DataOptions(rest_offset_s=0, backfill_pause_s=0)
BTC, ETH = (next(i for i in DEFAULT_INSTRUMENTS if i.symbol == s) for s in ("BTC-EUR", "ETH-EUR"))
CALENDAR = ExchangeCalendar()


def is_kraken(instrument: Instrument) -> bool:
    return instrument.exchange == "KRAKEN"


def is_yahoo(instrument: Instrument) -> bool:
    return not is_kraken(instrument)


class FakeCandleStream(FakeStream):
    def __init__(self, on_tick: TickHandler, on_bar: BarHandler, on_state: StateHandler) -> None:
        super().__init__(on_tick, on_state)
        self.on_bar = on_bar

    def emit_bar(self, bar: Bar) -> None:
        self.on_bar(bar)


class FakeCandleProvider(FakeProvider):
    """A ``CandleProvider`` double (Kraken-like): REST answers carry only committed minutes."""

    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.minutes_for = self._committed

    def _committed(self, symbol: str, start_ts: int, end_ts: int | None, prepost: bool) -> pd.DataFrame:
        end = end_ts if end_ts is not None else floor_ts(self.clock.ts())  # no forming minute
        first = -(-start_ts // 60) * 60
        ts = np.arange(first, end, 60, dtype=np.int64)
        values = np.column_stack([np.full(len(ts), 50_000.0)] * 4 + [np.full(len(ts), 0.25)])
        return pd.DataFrame(values, index=pd.Index(ts, name="ts"), columns=list(BAR_COLUMNS))

    def open_candle_stream(self, on_tick: TickHandler, on_bar: BarHandler, on_state: StateHandler) -> FakeCandleStream:
        self.stream = FakeCandleStream(on_tick, on_bar, on_state)
        return self.stream


@dataclass
class Routed:
    data: RoutedMarketData
    yahoo: FakeProvider
    kraken: FakeCandleProvider
    db: Database
    clock: ManualClock
    book: InstrumentBook


@pytest.fixture
async def make_routed(tmp_path: Path, db: Database, clock: ManualClock, bus: EventBus) -> AsyncIterator[Callable[..., Routed]]:
    built: list[RoutedMarketData] = []

    async def no_backfill() -> None:
        return None

    def build(*, stream: bool = False, instruments: list[Instrument] | None = None) -> Routed:
        seed_instruments(db, instruments if instruments is not None else [*SAMPLE_INSTRUMENTS, BTC, ETH])
        book = InstrumentBook(db)
        settings = make_settings(tmp_path, offline=False, enable_stream=stream)
        yahoo, kraken = FakeProvider(clock), FakeCandleProvider(clock)
        common = {"settings": settings, "db": db, "clock": clock, "bus": bus, "calendar": CALENDAR}
        lanes = [
            Lane("Yahoo", MarketDataService(**common, instruments=InstrumentView(book, is_yahoo), provider=yahoo,
                                            options=YAHOO_TEST), is_yahoo),
            Lane("Kraken", MarketDataService(**common, instruments=InstrumentView(book, is_kraken), provider=kraken,
                                             options=KRAKEN_TEST), is_kraken, lambda s: s.endswith("-EUR")),
        ]  # fmt: skip
        for lane in lanes:  # start() must not race the tests with a background history download
            lane.service.run_backfill = no_backfill  # type: ignore[method-assign]
        data = RoutedMarketData(lanes, book)
        built.append(data)
        clock.set(T0)
        return Routed(data, yahoo, kraken, db, clock, book)

    yield build
    for data in built:
        await data.stop()


def kraken_lane(routed: Routed) -> MarketDataService:
    return routed.data.lanes[1].service


def stored(db: Database, symbol: str) -> list[tuple[int, float, str]]:
    frame, sources = repo_market.read_bars_1m_with_sources(db, symbol)
    return [(int(ts), float(volume), str(source)) for ts, volume, source in zip(frame.index, frame["volume"], sources, strict=True)]


# ------------------------------------------------------------------------------------------------
# The candle source
# ------------------------------------------------------------------------------------------------
def test_kraken_is_a_utc_day_venue_without_closing_auction() -> None:
    assert "KRAKEN" in UTC_DAY_VENUES
    assert isinstance(FakeCandleProvider(ManualClock(T0)), CandleProvider)
    assert not isinstance(FakeProvider(ManualClock(T0)), CandleProvider)


async def test_closed_exchange_candles_become_stream_bars_and_ticks_build_none(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(stream=True)
    await routed.data.start()
    stream = routed.kraken.stream
    assert isinstance(stream, FakeCandleStream)
    minute = floor_ts(T0) - 60
    stream.emit(Tick("BTC-EUR", T0 - 30, 50_100.0, T0 - 29, bid=50_099.0, ask=50_101.0))
    stream.emit_bar(Bar("BTC-EUR", minute, 50_000.0, 50_200.0, 49_900.0, 50_100.0, 1.5))
    stream.emit_bar(Bar("BTC-EUR", floor_ts(T0), 50_100.0, 50_100.0, 50_100.0, 50_100.0, 0.1))  # still forming
    stream.emit_bar(Bar("BTC-EUR", minute - 2 * 3_600, 1.0, 1.0, 1.0, 1.0, 1.0))  # an old snapshot candle
    stream.emit_bar(Bar("NVDA", minute, 1.0, 1.0, 1.0, 1.0, 1.0))  # not a Kraken instrument
    routed.clock.set(T0 + 3)
    updated = await routed.data.refresh(from_ts(T0 + 3))
    assert "BTC-EUR" in updated
    assert stored(routed.db, "BTC-EUR") == [(minute, 1.5, "stream")]  # real volume, no tick-built bar
    assert all(source != "stream" for _, _, source in stored(routed.db, "NVDA"))  # not routed to Kraken
    quote = routed.data.quote("BTC-EUR")
    assert quote is not None and (quote.price, quote.bid, quote.ask, quote.source) == (50_100.0, 50_099.0, 50_101.0, "stream")


async def test_a_candle_closed_by_the_exchange_a_moment_early_waits_for_the_local_minute_end(
    make_routed: Callable[..., Routed],
) -> None:
    """The local clock may run a little behind Kraken's: the closed candle arrives while the local clock still
    shows the last second of its minute. It must not be dropped, only written once the minute is over locally."""
    routed = make_routed(stream=True)
    await routed.data.start()
    stream = routed.kraken.stream
    assert isinstance(stream, FakeCandleStream)
    routed.kraken.trip("no REST in this test")  # only the stream writes
    routed.clock.set(T0 + 59)
    stream.emit_bar(Bar("ETH-EUR", T0, 1_570.0, 1_571.0, 1_569.0, 1_570.5, 2.0))
    await routed.data.refresh(from_ts(T0 + 59))
    assert stored(routed.db, "ETH-EUR") == []
    routed.clock.set(T0 + 63)
    await routed.data.refresh(from_ts(T0 + 63))
    assert stored(routed.db, "ETH-EUR") == [(T0, 2.0, "stream")]
    stream.emit_bar(Bar("ETH-EUR", T0 + 3 * 60, 1.0, 1.0, 1.0, 1.0, 1.0))  # a minute in the future: bogus
    routed.clock.set(T0 + 123)
    await routed.data.refresh(from_ts(T0 + 123))
    assert [ts for ts, _, _ in stored(routed.db, "ETH-EUR")] == [T0]


async def test_snapshot_quotes_are_never_sampled_into_bars(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed()
    service = kraken_lane(routed)
    await service.prepare()
    routed.kraken.quotes = {"BTC-EUR": Quote("BTC-EUR", T0, T0, 50_000.0, prev_close=49_000.0)}
    await service.poll_quotes(T0)
    routed.clock.set(T0 + 120)
    await service.refresh(from_ts(T0 + 120))
    assert routed.data.quote("BTC-EUR") is not None
    assert all(source != "quote" for _, _, source in stored(routed.db, "BTC-EUR"))


async def test_rest_reconciliation_confirms_stream_bars_so_they_settle_at_once(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(stream=True)
    await routed.data.start()
    stream = routed.kraken.stream
    assert isinstance(stream, FakeCandleStream)
    minute = floor_ts(T0) - 60
    stream.emit_bar(Bar("BTC-EUR", minute, 50_000.0, 50_000.0, 50_000.0, 50_000.0, 0.5))
    routed.clock.set(T0 + 3)
    await routed.data.refresh(from_ts(T0 + 3))
    reader = BarReader(routed.db, routed.book, CALENDAR, T0 + 3)
    assert reader.settled_bars_after("BTC-EUR", None, start_ts=minute).empty  # a young stream bar is provisional
    for minutes in (4, 5):  # one symbol per minute (2 symbols, every 3 minutes): ETH first, then BTC
        routed.clock.set(T0 + minutes * 60)
        await routed.data.refresh(from_ts(T0 + minutes * 60))
    requests = routed.kraken.calls_of("minutes")
    assert requests and all(set(call.symbols) <= {"BTC-EUR", "ETH-EUR"} for call in requests)
    assert all(T0 - 12 * 3_600 < call.start_ts for call in requests)  # inside the OHLC range
    bars = dict((ts, (volume, source)) for ts, volume, source in stored(routed.db, "BTC-EUR"))
    assert bars[minute] == (0.25, "rest")  # authoritative REST replaced the stream bar, volume kept for crypto
    newest = max(bars)
    assert newest == floor_ts(T0 + 5 * 60) - 60  # the newest committed minute is stored at once
    reader = BarReader(routed.db, routed.book, CALENDAR, T0 + 5 * 60)
    settled = reader.settled_bars_after("BTC-EUR", None, start_ts=minute)
    assert minute in settled.index and settled.index[-1] == newest  # settled by source, not by age


async def test_backfill_uses_ohlc_for_the_recent_window_and_resumes_older_windows(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(instruments=[BTC])
    routed.kraken.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-20", "2026-11-21", "2026-11-22"])
    service = kraken_lane(routed)
    del service.run_backfill  # the real backfill (the fixture disables it for start())
    progress: list[float | None] = []
    original = routed.kraken.fetch_minutes

    async def watched(symbols, *, start_ts, end_ts=None, prepost=False):  # type: ignore[no-untyped-def]
        progress.append(routed.data.status().backfill_progress)
        if len(routed.kraken.calls_of("minutes")) == 3:
            raise asyncio.CancelledError  # the platform stops in the middle of the backfill
        return await original(symbols, start_ts=start_ts, end_ts=end_ts, prepost=prepost)

    routed.kraken.fetch_minutes = watched  # type: ignore[method-assign]
    with pytest.raises(asyncio.CancelledError):
        await service.run_backfill()
    first = routed.kraken.calls_of("minutes")
    recent = first[0]
    assert recent.end_ts is None and recent.start_ts >= floor_ts(T0) - KRAKEN_OPTIONS.minute_window_s
    assert all(c.end_ts is not None and c.end_ts - c.start_ts <= KRAKEN_OPTIONS.minute_window_s for c in first[1:])
    assert all(p is not None and 0 <= p < 1 for p in progress)
    assert routed.kraken.calls_of("daily")[0].start_ts <= T0 - 2 * 365 * DAY
    covered = repo_market.bar_range(routed.db, "BTC-EUR")
    assert covered is not None and covered[1] == floor_ts(T0) - 60

    # restart: only what bars_1m does not cover yet, back to BACKFILL_DAYS
    routed.kraken.fetch_minutes = original  # type: ignore[method-assign]
    routed.kraken.calls.clear()
    await service.run_backfill()
    resumed = routed.kraken.calls_of("minutes")
    assert resumed and all(call.end_ts is not None for call in resumed)  # the recent window is complete
    assert max(call.end_ts for call in resumed if call.end_ts) <= covered[0] + KRAKEN_OPTIONS.recon_overlap_s
    oldest = floor_ts(T0) - 14 * DAY
    assert min(call.start_ts for call in resumed) == oldest
    assert repo_market.bar_range(routed.db, "BTC-EUR") == (oldest, floor_ts(T0) - 60)
    routed.kraken.calls.clear()
    await service.run_backfill()
    assert routed.kraken.calls == []  # nothing left to do


# ------------------------------------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------------------------------------
async def test_each_lane_serves_only_its_instruments(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(stream=True)
    await routed.data.start()
    assert routed.yahoo.stream is not None and routed.kraken.stream is not None
    assert set(routed.kraken.stream.subscriptions[-1]) == {"BTC-EUR", "ETH-EUR"}
    assert "BTC-EUR" not in routed.yahoo.stream.subscriptions[-1] and "NVDA" in routed.yahoo.stream.subscriptions[-1]
    routed.clock.set(T0 + 5 * 60)
    await routed.data.refresh(from_ts(T0 + 5 * 60))
    assert set(routed.kraken.requested("minutes")) <= {"BTC-EUR", "ETH-EUR"}
    assert not {"BTC-EUR", "ETH-EUR"} & set(routed.yahoo.requested("minutes")) and routed.yahoo.requested("minutes")
    assert set(routed.kraken.requested("quotes")) <= {"BTC-EUR", "ETH-EUR"}
    assert not {"BTC-EUR", "ETH-EUR"} & set(routed.yahoo.requested("quotes"))


async def test_a_kraken_outage_does_not_stop_yahoo_and_status_shows_it(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(stream=True)
    await routed.data.start()
    routed.kraken.trip("rate limit")
    routed.clock.set(T0 + 5 * 60)
    routed.yahoo.calls.clear()
    routed.kraken.calls.clear()
    await routed.data.refresh(from_ts(T0 + 5 * 60))
    assert routed.yahoo.calls_of("minutes") and not routed.kraken.calls_of("minutes")
    status = routed.data.status()
    assert status.rate_limited_until == routed.kraken.paused
    assert status.symbols_active == len(SAMPLE_INSTRUMENTS) + 2
    assert not status.stream_connected
    assert routed.yahoo.stream is not None and routed.kraken.stream is not None
    routed.yahoo.stream.set_connected(True)
    assert not routed.data.status().stream_connected  # the Kraken stream is still down
    routed.kraken.stream.set_connected(True)
    assert routed.data.status().stream_connected


async def test_a_lane_without_active_symbols_does_not_count_for_the_status(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed(stream=True, instruments=SAMPLE_INSTRUMENTS)  # no Kraken instrument at all
    await routed.data.start()
    assert routed.yahoo.stream is not None
    routed.yahoo.stream.set_connected(True)
    assert routed.data.status().stream_connected


async def test_quotes_lookup_and_history_are_routed(make_routed: Callable[..., Routed]) -> None:
    routed = make_routed()
    await routed.data.lanes[0].service.prepare()
    await routed.data.lanes[1].service.prepare()
    routed.kraken.quotes = {"BTC-EUR": Quote("BTC-EUR", T0, T0, 50_000.0)}
    routed.yahoo.quotes = {"NVDA": Quote("NVDA", T0, T0, 180.0)}
    await routed.data.lanes[0].service.poll_quotes(T0)
    await routed.data.lanes[1].service.poll_quotes(T0)
    assert set(routed.data.quotes()) == {"NVDA", "BTC-EUR"}
    assert routed.data.quote("BTC-EUR").price == 50_000.0  # type: ignore[union-attr]
    ltc = Instrument("LTC-EUR", "Litecoin", "crypto", "GLOBAL", "KRAKEN", "EUR", timezone="UTC")
    routed.kraken.instruments["LTC-EUR"] = ltc
    routed.yahoo.instruments["AAPL"] = Instrument("AAPL", "Apple", "equity", "US", "XNAS", "USD")
    assert (await routed.data.lookup("ltc-eur")) is not None and routed.kraken.requested("lookup") == ["LTC-EUR"]
    assert (await routed.data.lookup("AAPL")) is not None and routed.yahoo.requested("lookup") == ["AAPL"]
    routed.kraken.daily_for = lambda symbol, start_ts: daily_frame(["2026-11-21"])
    await routed.data.ensure_history("ETH-EUR")
    assert {s for call in routed.kraken.calls for s in call.symbols if call.kind != "quotes"} >= {"ETH-EUR"}
    assert "ETH-EUR" not in routed.yahoo.requested("daily")


async def test_a_failing_lane_never_stops_the_refresh_of_the_other(
    make_routed: Callable[..., Routed], monkeypatch: pytest.MonkeyPatch
) -> None:
    routed = make_routed()

    async def broken(now: datetime) -> list[str]:
        raise RuntimeError("lane crashed")

    monkeypatch.setattr(routed.data.lanes[1].service, "refresh", broken)
    routed.clock.set(T0 + 5 * 60)
    assert isinstance(await routed.data.refresh(from_ts(T0 + 5 * 60)), list)
    assert routed.yahoo.calls_of("minutes")
