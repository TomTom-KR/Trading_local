"""KrakenStream: decoding of recorded WS v2 messages, candle closing, subscriptions, reconnect and liveness."""

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from trading_local.clock import ManualClock
from trading_local.data.kraken import KRAKEN_PAIRS
from trading_local.data.kraken_stream import (
    CandleTracker,
    KrakenStream,
    KrakenStreamOptions,
    backoff_delay,
    decode_candle,
    decode_ticker,
    rfc3339_ts,
)
from trading_local.data.provider import Tick
from trading_local.models import Bar

FIXTURES = Path(__file__).parent / "fixtures" / "kraken"
SESSION = json.loads((FIXTURES / "ws_session.json").read_text(encoding="utf-8"))
WS_TO_SYMBOL = {pair.ws_symbol: symbol for symbol, pair in KRAKEN_PAIRS.items()}
FAST = KrakenStreamOptions(
    poll_s=0.03, idle_timeout_s=0.4, ping_interval_s=0.05, pong_timeout_s=0.3, backoff_initial_s=0.01,
    backoff_max_s=0.02, close_grace_s=2.0,
)  # fmt: skip


def messages(channel: str, kind: str | None = None) -> list[dict[str, Any]]:
    return [
        item["msg"] for item in SESSION
        if item["msg"].get("channel") == channel and (kind is None or item["msg"].get("type") == kind)
    ]  # fmt: skip


def symbol_of(ws_symbol: Any) -> str | None:
    return WS_TO_SYMBOL.get(ws_symbol)


def resolve(symbol: str) -> str | None:
    pair = KRAKEN_PAIRS.get(symbol)
    return None if pair is None else pair.ws_symbol


# ------------------------------------------------------------------------------------------------
# Pure decoding
# ------------------------------------------------------------------------------------------------
def test_rfc3339_with_any_fraction() -> None:
    assert rfc3339_ts("2026-11-22T23:13:00.000000000Z") == 1_795_389_180.0
    assert rfc3339_ts("2026-11-22T23:26:16.393564Z") == pytest.approx(1_795_389_976.393564)
    assert rfc3339_ts("2026-11-22T23:26:16Z") == 1_795_389_976.0
    assert rfc3339_ts("2026-11-23T00:26:16+01:00") == 1_795_389_976.0
    assert rfc3339_ts("yesterday") is None and rfc3339_ts(None) is None


def test_recorded_candles_decode_to_bars_at_their_interval_start() -> None:
    snapshot = messages("ohlc", "snapshot")[0]["data"]
    bars = [decode_candle(item, symbol_of) for item in snapshot]
    assert all(bar is not None for bar in bars)
    first = bars[0]
    assert first is not None and first.source == "stream" and first.ts % 60 == 0
    assert first.ts == int(rfc3339_ts(snapshot[0]["interval_begin"]) or 0)
    assert (first.open, first.high, first.low, first.close, first.volume) == tuple(
        float(snapshot[0][key]) for key in ("open", "high", "low", "close", "volume")
    )
    item = dict(snapshot[0])
    assert decode_candle({**item, "trades": 0}, symbol_of) is None  # a minute without trades is no bar
    assert decode_candle({**item, "interval": 5}, symbol_of) is None
    assert decode_candle({**item, "symbol": "NOPE/EUR"}, symbol_of) is None
    assert decode_candle({**item, "high": item["low"] - 1}, symbol_of) is None
    legacy = {key: value for key, value in item.items() if key != "interval_begin"}  # deprecated 'timestamp' = end
    assert decode_candle(legacy, symbol_of) == first


def test_recorded_ticker_decodes_to_a_tick_with_last_bid_ask_and_kraken_time() -> None:
    item = messages("ticker", "snapshot")[0]["data"][0]
    tick = decode_ticker(item, 1_795_390_000.0, symbol_of)
    assert tick is not None and tick.symbol == WS_TO_SYMBOL[item["symbol"]]
    assert (tick.price, tick.bid, tick.ask) == (item["last"], item["bid"], item["ask"])
    assert tick.ts == rfc3339_ts(item["timestamp"]) and tick.received_ts == 1_795_390_000.0
    # rolling 24 h figures stay out of the quote book (it keeps the UTC-day values of the REST ticker)
    assert (tick.day_volume, tick.day_high, tick.day_low, tick.change_pct, tick.prev_close) == (None,) * 5
    assert decode_ticker({**item, "last": 0}, 1.0, symbol_of) is None


def test_backoff_grows_with_jitter_and_is_capped() -> None:
    rng = random.Random(1)
    delays = [backoff_delay(attempt, 1.0, 60.0, rng) for attempt in range(10)]
    assert all(0.5 * min(60.0, 2.0**n) <= d <= min(60.0, 2.0**n) for n, d in enumerate(delays))


# ------------------------------------------------------------------------------------------------
# Candle closing
# ------------------------------------------------------------------------------------------------
def bar(ts: int, close: float, symbol: str = "BTC-EUR") -> Bar:
    return Bar(symbol, ts, close, close + 1, close - 1, close, 1.0, "stream")


def test_a_candle_is_emitted_once_its_minute_is_over() -> None:
    tracker = CandleTracker(close_grace_s=2.0)
    assert tracker.update(bar(600, 10.0), now=630.0) == []  # forming
    assert tracker.update(bar(600, 11.0), now=650.0) == []  # updated in place
    assert tracker.due(661.9) == []
    assert tracker.due(662.0) == [bar(600, 11.0)]  # minute end + grace
    assert tracker.due(700.0) == []


def test_a_newer_candle_closes_the_previous_one_and_late_updates_are_corrections() -> None:
    tracker = CandleTracker()
    tracker.update(bar(600, 10.0), now=630.0)
    assert tracker.update(bar(660, 12.0), now=660.5) == [bar(600, 10.0)]
    assert tracker.update(bar(600, 10.5), now=661.0) == [bar(600, 10.5)]  # correction of an emitted minute
    assert tracker.update(bar(600, 10.5), now=661.5) == []  # identical: not again
    assert tracker.update(bar(540, 9.0), now=662.0) == [bar(540, 9.0)]  # an older snapshot candle fills in


def test_a_reconnect_snapshot_repeats_nothing_already_emitted_and_unfinished_candles_are_dropped() -> None:
    tracker = CandleTracker()
    snapshot = [bar(480, 8.0), bar(540, 9.0), bar(600, 10.0)]
    emitted = [b for candle in snapshot for b in tracker.update(candle, now=630.0)]
    assert emitted == [bar(480, 8.0), bar(540, 9.0)]  # the forming 600 candle waits
    tracker.drop_open()  # connection lost in minute 600
    again = [b for candle in [*snapshot[1:], bar(660, 11.0)] for b in tracker.update(candle, now=665.0)]
    assert again == [bar(600, 10.0)]  # 540 is not repeated; 600 is emitted from the fresh snapshot


def test_symbols_are_tracked_separately() -> None:
    tracker = CandleTracker()
    tracker.update(bar(600, 10.0, "BTC-EUR"), now=601.0)
    assert tracker.update(bar(660, 20.0, "LTC-EUR"), now=661.0) == []
    assert {b.symbol for b in tracker.due(800.0)} == {"BTC-EUR", "LTC-EUR"}


# ------------------------------------------------------------------------------------------------
# Connection behaviour against a scripted server
# ------------------------------------------------------------------------------------------------
class FakeSocket:
    def __init__(self, *, heartbeat_s: float | None = None, answer_pings: bool = True) -> None:
        self.sent: list[dict[str, Any]] = []
        self.inbox: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.heartbeat_s = heartbeat_s
        self.answer_pings = answer_pings

    async def send(self, text: str) -> None:
        message = json.loads(text)
        self.sent.append(message)
        if message.get("method") == "ping" and self.answer_pings:
            self.push({"method": "pong", "req_id": message["req_id"]})

    async def recv(self) -> str:
        if self.heartbeat_s is not None and self.inbox.empty():
            await asyncio.sleep(self.heartbeat_s)
            if self.inbox.empty():
                return json.dumps({"channel": "heartbeat"})
        item = await self.inbox.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def push(self, message: dict[str, Any] | BaseException) -> None:
        self.inbox.put_nowait(message if isinstance(message, BaseException) else json.dumps(message))

    def requests(self, method: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("method") == method]


class FakeServer:
    """Connector double: every connection gets a new FakeSocket built by ``make``."""

    def __init__(self, make: Callable[[], FakeSocket] = lambda: FakeSocket(heartbeat_s=0.01)) -> None:
        self.make = make
        self.sockets: list[FakeSocket] = []
        self.refuse = 0  # connection attempts to refuse first

    def __call__(self, options: KrakenStreamOptions) -> Any:
        @asynccontextmanager
        async def connection() -> AsyncIterator[FakeSocket]:
            if self.refuse:
                self.refuse -= 1
                raise OSError("connection refused")
            socket = self.make()
            self.sockets.append(socket)
            yield socket

        return connection()


async def until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.005)


class Recorder:
    def __init__(self) -> None:
        self.ticks: list[Tick] = []
        self.bars: list[Bar] = []
        self.states: list[bool] = []


def make_stream(server: FakeServer, clock: ManualClock, rec: Recorder, options: KrakenStreamOptions = FAST,
                candles: bool = True) -> KrakenStream:  # fmt: skip
    return KrakenStream(
        rec.ticks.append, rec.bars.append if candles else None, rec.states.append, clock=clock, resolve=resolve,
        options=options, connector=server, rng=random.Random(0),
    )  # fmt: skip


async def test_subscribes_both_channels_and_delivers_quotes_and_closed_candles(clock: ManualClock) -> None:
    server, rec = FakeServer(), Recorder()
    stream = make_stream(server, clock, rec)
    await stream.subscribe(["BTC-EUR", "LTC-EUR", "NVDA"])  # NVDA has no Kraken pair: ignored
    await stream.connect()
    await until(lambda: stream.connected and len(server.sockets[0].requests("subscribe")) == 2)
    ohlc, ticker = server.sockets[0].requests("subscribe")
    assert ohlc["params"] == {"channel": "ohlc", "symbol": ["BTC/EUR", "LTC/EUR"], "interval": 1, "snapshot": True}
    assert ticker["params"] == {"channel": "ticker", "symbol": ["BTC/EUR", "LTC/EUR"], "snapshot": True}
    assert rec.states == [True]

    snapshot = next(m for m in messages("ohlc", "snapshot") if m["data"][0]["symbol"] == "BTC/EUR")
    forming = snapshot["data"][-1]
    forming_ts = int(rfc3339_ts(forming["interval_begin"]) or 0)
    clock.set(forming_ts + 30)  # inside the forming candle's minute
    socket = server.sockets[0]
    socket.push(snapshot)
    socket.push(messages("ticker", "snapshot")[0])
    await until(lambda: len(rec.bars) == len(snapshot["data"]) - 1 and rec.ticks)
    assert all(b.ts < forming_ts and b.source == "stream" for b in rec.bars)
    assert rec.ticks[0].symbol in ("BTC-EUR", "LTC-EUR")
    clock.set(forming_ts + 62)  # minute over + grace: the forming candle closes without a further message
    await until(lambda: len(rec.bars) == len(snapshot["data"]))
    assert rec.bars[-1].ts == forming_ts
    await stream.close()
    assert not stream.connected and rec.states == [True]  # no state callback while closing


async def test_reconnects_with_backoff_and_subscribes_again(clock: ManualClock) -> None:
    server, rec = FakeServer(), Recorder()
    server.refuse = 2  # the first two attempts fail
    stream = make_stream(server, clock, rec)
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: stream.connected)
    assert stream.connects == 1 and len(server.sockets) == 1
    server.sockets[0].push(ConnectionError("server went away"))
    await until(lambda: len(server.sockets) == 2 and stream.connected)
    assert [m["params"]["channel"] for m in server.sockets[1].requests("subscribe")] == ["ohlc", "ticker"]
    assert rec.states == [True, False, True]
    await stream.close()


async def test_watchdog_reconnects_a_silent_connection(clock: ManualClock) -> None:
    server, rec = FakeServer(lambda: FakeSocket(heartbeat_s=None)), Recorder()
    stream = make_stream(server, clock, rec)
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: len(server.sockets) >= 2, timeout=5.0)  # nothing arrived for idle_timeout_s
    await stream.close()


async def test_missing_pong_forces_a_reconnect_and_answered_pings_keep_the_line(clock: ManualClock) -> None:
    server = FakeServer(lambda: FakeSocket(heartbeat_s=0.01, answer_pings=False))
    stream = make_stream(server, clock, Recorder())
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: len(server.sockets) >= 2, timeout=5.0)
    assert server.sockets[0].requests("ping")  # application-level pings were sent
    await stream.close()

    healthy = FakeServer(lambda: FakeSocket(heartbeat_s=0.01, answer_pings=True))
    stream = make_stream(healthy, clock, Recorder())
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await asyncio.sleep(0.8)  # well past pong_timeout_s and idle_timeout_s
    assert len(healthy.sockets) == 1 and len(healthy.sockets[0].requests("ping")) >= 3
    await stream.close()


async def test_subscription_changes_are_applied_on_the_live_connection(clock: ManualClock) -> None:
    server = FakeServer()
    stream = make_stream(server, clock, Recorder())
    await stream.subscribe(["BTC-EUR", "LTC-EUR"])
    await stream.connect()
    await until(lambda: stream.connected)
    await stream.subscribe(["LTC-EUR", "DOT-EUR"])
    socket = server.sockets[0]
    await until(lambda: len(socket.requests("unsubscribe")) == 2 and len(socket.requests("subscribe")) == 4)
    assert {m["params"]["channel"]: m["params"]["symbol"] for m in socket.requests("unsubscribe")} == {
        "ohlc": ["BTC/EUR"], "ticker": ["BTC/EUR"]
    }
    assert socket.requests("unsubscribe")[0]["params"]["interval"] == 1
    assert [m["params"]["symbol"] for m in socket.requests("subscribe")[2:]] == [["DOT/EUR"], ["DOT/EUR"]]
    await stream.close()


async def test_without_symbols_no_connection_is_opened(clock: ManualClock) -> None:
    server = FakeServer()
    stream = make_stream(server, clock, Recorder())
    await stream.subscribe(["NVDA"])  # nothing Kraken knows
    await stream.connect()
    await asyncio.sleep(0.1)
    assert server.sockets == [] and not stream.connected
    await stream.subscribe(["ETH-EUR"])
    await until(lambda: stream.connected)
    await stream.subscribe([])  # the last pair was deactivated: the connection is closed on purpose
    await until(lambda: not stream.connected)
    assert len(server.sockets) == 1
    await stream.close()


async def test_a_refused_pair_is_logged_once_and_does_not_reconnect(
    clock: ManualClock, caplog: pytest.LogCaptureFixture
) -> None:
    server = FakeServer()
    stream = make_stream(server, clock, Recorder())
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: stream.connected)
    refused = next(item["msg"] for item in SESSION if item["msg"].get("success") is False)
    with caplog.at_level(logging.WARNING, logger="trading_local.data.kraken_stream"):
        server.sockets[0].push(refused)
        server.sockets[0].push(refused)
        await asyncio.sleep(0.1)
    assert sum("refused" in record.getMessage() for record in caplog.records) == 1
    assert len(server.sockets) == 1 and stream.connected
    await stream.close()


async def test_ticker_only_stream_without_candle_handler(clock: ManualClock) -> None:
    server, rec = FakeServer(), Recorder()
    stream = make_stream(server, clock, rec, candles=False)
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: stream.connected and server.sockets[0].requests("subscribe"))
    assert [m["params"]["channel"] for m in server.sockets[0].requests("subscribe")] == ["ticker"]
    server.sockets[0].push(messages("ohlc", "snapshot")[0])  # ignored without a candle handler
    await asyncio.sleep(0.05)
    assert rec.bars == []
    await stream.close()


async def test_handler_errors_do_not_stop_the_stream(clock: ManualClock) -> None:
    server = FakeServer()

    def broken(tick: Tick) -> None:
        raise RuntimeError("handler bug")

    stream = KrakenStream(broken, None, lambda connected: None, clock=clock, resolve=resolve, options=FAST,
                          connector=server)  # fmt: skip
    await stream.subscribe(["BTC-EUR"])
    await stream.connect()
    await until(lambda: stream.connected)
    for message in messages("ticker"):
        server.sockets[0].push(message)
    await asyncio.sleep(0.1)
    assert stream.connected and len(server.sockets) == 1
    await stream.close()
