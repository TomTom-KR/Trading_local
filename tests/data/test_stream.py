"""Yahoo stream client against a local WebSocket server that speaks the same protocol (offline)."""

import asyncio
import base64
import json
import random
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field, replace

import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from yfinance.pricing_pb2 import PricingData

from trading_local.clock import SystemClock
from trading_local.data.provider import Tick
from trading_local.data.stream import StreamOptions, YahooStream, backoff_delay, decode_message, shard_symbols


def frame(symbol: str, price: float, time_ms: int, **fields: object) -> str:
    data = PricingData(id=symbol, price=price, time=time_ms, **fields)  # type: ignore[arg-type]
    return json.dumps({"type": "pricing", "message": base64.b64encode(data.SerializeToString()).decode()})


def test_decode_message_fields_units_and_defaults() -> None:
    raw = frame(
        "BTC-USD", 61_174.54, 1_795_367_029_000, price_hint=2, market_hours=1, change_percent=-0.028,
        day_volume=1_917_434_240, day_high=61_500.0, previous_close=60_000.0,
    )  # fmt: skip
    tick = decode_message(raw, received_ts=1_795_367_030.5)
    assert tick is not None
    assert tick.symbol == "BTC-USD" and tick.ts == 1_795_367_029.0 and tick.received_ts == 1_795_367_030.5
    assert tick.price == 61_174.54  # the 32-bit float is rounded to the price hint
    assert tick.market_state == "REGULAR" and tick.change_pct == pytest.approx(-0.028)
    assert tick.day_volume == 1_917_434_240 and tick.prev_close == 60_000.0
    assert tick.bid is None and tick.ask is None and tick.day_open is None  # proto defaults are absent
    no_hours = decode_message(frame("NVDA", 180.5, 1_795_367_029_000, price_hint=2), 0.0)
    assert no_hours is not None and no_hours.market_state == "PRE"  # market_hours absent = 0 = PRE_MARKET
    assert decode_message("not json", 0.0) is None
    assert decode_message(json.dumps({"type": "pricing"}), 0.0) is None
    assert decode_message(json.dumps({"message": "%%%"}), 0.0) is None
    assert decode_message(frame("X", 0.0, 1_000), 0.0) is None  # no price


def test_shards_are_balanced_and_each_has_a_canary() -> None:
    symbols = [f"S{index}" for index in range(108)] + ["S5"]
    shards = shard_symbols(symbols, 70, ("BTC-USD", "ETH-USD"))
    assert [len(shard) for shard in shards] == [55, 55]
    assert shards[0][-1] == "BTC-USD" and shards[1][-1] == "ETH-USD"
    assert sorted(s for shard in shards for s in shard if s.startswith("S")) == sorted(set(symbols))
    with_canary = shard_symbols(["A", "ETH-USD", "B"], 70, ("BTC-USD", "ETH-USD"))
    assert with_canary == [["A", "ETH-USD", "B"]]
    assert shard_symbols([], 70, ("BTC-USD",)) == [["BTC-USD"]]
    assert all(len(shard) <= 71 for shard in shard_symbols([f"X{i}" for i in range(300)], 70, ("BTC-USD",)))


def test_backoff_grows_exponentially_with_jitter_and_a_cap() -> None:
    rng = random.Random(1)
    delays = [backoff_delay(attempt, 1.0, 60.0, rng) for attempt in range(10)]
    assert 0.5 <= delays[0] <= 1.0 and 4.0 <= delays[3] <= 8.0
    assert all(30.0 <= delay <= 60.0 for delay in delays[6:])


# ------------------------------------------------------------------------------------------------
# Live tests against a local server
# ------------------------------------------------------------------------------------------------
FAST = StreamOptions(
    url="",
    shard_size=70,
    canaries=("BTC-USD",),
    resubscribe_s=30.0,
    idle_timeout_s=0.6,
    canary_timeout_s=0.6,
    backoff_initial_s=0.01,
    backoff_max_s=0.05,
    open_timeout_s=2.0,
    close_timeout_s=0.5,
    poll_s=0.05,
)


@dataclass
class FakeYahoo:
    """Server double: records subscriptions per connection and streams frames from ``send``."""

    send: Callable[[ServerConnection, int], list[str]]  # (connection, connection number) -> frames per beat
    connections: int = 0
    subscriptions: list[tuple[int, list[str]]] = field(default_factory=list)
    unsubscriptions: list[tuple[int, list[str]]] = field(default_factory=list)
    close_after: dict[int, float] = field(default_factory=dict)  # connection number -> close cleanly after seconds

    async def handler(self, ws: ServerConnection) -> None:
        self.connections += 1
        number = self.connections
        started = time.monotonic()
        receiver = asyncio.create_task(self._receive(ws, number))
        try:
            while not receiver.done():  # the receiver ends when the client goes away
                if number in self.close_after and time.monotonic() - started > self.close_after[number]:
                    await ws.close()
                    return
                for message in self.send(ws, number):
                    await ws.send(message)
                await asyncio.sleep(0.02)
        except Exception:  # noqa: BLE001 - the client went away
            return
        finally:
            receiver.cancel()

    async def _receive(self, ws: ServerConnection, number: int) -> None:
        async for message in ws:
            payload = json.loads(message)
            if "subscribe" in payload:
                self.subscriptions.append((number, payload["subscribe"]))
            if "unsubscribe" in payload:
                self.unsubscriptions.append((number, payload["unsubscribe"]))


type ServerFactory = Callable[[FakeYahoo], Coroutine[None, None, str]]


@pytest.fixture
async def server_factory() -> AsyncIterator[ServerFactory]:
    servers: list[Server] = []

    async def start(fake: FakeYahoo) -> str:
        server = await serve(fake.handler, "127.0.0.1", 0)
        servers.append(server)
        port = server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    yield start
    for server in servers:
        server.close()
        await server.wait_closed()


async def wait_for(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


def beat(*symbols: str) -> Callable[[ServerConnection, int], list[str]]:
    def send(ws: ServerConnection, number: int) -> list[str]:
        now_ms = int(time.time() * 1000)
        return [frame(symbol, 100.0 + number, now_ms, price_hint=2, market_hours=1) for symbol in symbols]

    return send


def make_stream(url: str, ticks: list[Tick], states: list[bool], **options: object) -> YahooStream:
    opts = replace(FAST, url=url, **options)  # type: ignore[arg-type]
    return YahooStream(ticks.append, states.append, clock=SystemClock(), options=opts, rng=random.Random(7))


async def test_ticks_arrive_and_a_clean_server_close_is_survived(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("BTC-USD", "NVDA"), close_after={1: 0.3})
    url = await server_factory(fake)
    ticks: list[Tick] = []
    states: list[bool] = []
    stream = make_stream(url, ticks, states)
    await stream.subscribe(["NVDA"])
    await stream.connect()
    await wait_for(lambda: fake.connections >= 2 and any(t.price == 102.0 for t in ticks))
    assert states[:3] == [True, False, True]
    assert fake.subscriptions[0] == (1, ["NVDA", "BTC-USD"])
    assert any(number == 2 and symbols == ["NVDA", "BTC-USD"] for number, symbols in fake.subscriptions)
    started = time.monotonic()
    await stream.close()
    assert time.monotonic() - started < 2.0
    count = fake.connections
    await asyncio.sleep(0.2)
    assert fake.connections == count and not stream.connected
    assert states[-1] is True  # a deliberate close is not reported as a disconnect


async def test_silent_canary_forces_a_reconnect(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("NVDA"))  # the canary never ticks, other data does
    url = await server_factory(fake)
    stream = make_stream(url, [], [])
    await stream.subscribe(["NVDA"])
    await stream.connect()
    await wait_for(lambda: fake.connections >= 2)
    await stream.close()


async def test_idle_connection_is_reconnected(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=lambda ws, number: [])
    url = await server_factory(fake)
    stream = make_stream(url, [], [], canary_timeout_s=60.0)
    await stream.subscribe(["NVDA"])
    await stream.connect()
    await wait_for(lambda: fake.connections >= 2)
    await stream.close()


async def test_shards_each_get_their_own_connection_and_canary(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("BTC-USD"))
    url = await server_factory(fake)
    ticks: list[Tick] = []
    stream = make_stream(url, ticks, [], shard_size=2)
    await stream.subscribe(["A", "B", "C"])
    await stream.connect()
    await wait_for(lambda: fake.connections == 2 and len(fake.subscriptions) >= 2 and stream.connected)
    subscribed = sorted(symbols for _, symbols in fake.subscriptions[:2])
    assert subscribed == [["A", "B", "BTC-USD"], ["C", "BTC-USD"]]
    await wait_for(lambda: len(ticks) >= 3)
    times = [tick.ts for tick in ticks]
    assert len(times) == len(set(times))  # the canary on both connections is delivered once
    await stream.close()


async def test_each_shard_watches_its_own_canary(server_factory: ServerFactory) -> None:
    fake: FakeYahoo

    def send(ws: ServerConnection, number: int) -> list[str]:
        subscribed = {symbol for n, symbols in fake.subscriptions if n == number for symbol in symbols}
        now_ms = int(time.time() * 1000)
        if "C" in subscribed:  # only this connection delivers the canary ...
            return [frame("BTC-USD", 100.0, now_ms, price_hint=2)]
        if "A" in subscribed:  # ... the other one delivers data but its canary stays silent
            return [frame("A", 10.0, now_ms, price_hint=2)]
        return []

    fake = FakeYahoo(send=send)
    url = await server_factory(fake)
    stream = make_stream(url, [], [], shard_size=2, idle_timeout_s=10.0)
    await stream.subscribe(["A", "B", "C"])
    await stream.connect()

    def connections_with(symbol: str) -> set[int]:
        return {number for number, symbols in fake.subscriptions if symbol in symbols}

    await wait_for(lambda: len(connections_with("A")) >= 2)  # the shard with the silent canary reconnects
    assert len(connections_with("C")) == 1  # the healthy shard keeps its connection
    await stream.close()


async def test_subscription_changes_are_applied_live(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("BTC-USD"))
    url = await server_factory(fake)
    stream = make_stream(url, [], [])
    await stream.subscribe(["NVDA", "SAP.DE"])
    await stream.connect()
    await wait_for(lambda: len(fake.subscriptions) >= 1)
    await stream.subscribe(["NVDA", "AAPL"])
    await wait_for(lambda: fake.unsubscriptions == [(1, ["SAP.DE"])])
    await wait_for(lambda: fake.subscriptions[-1] == (1, ["NVDA", "AAPL", "BTC-USD"]))
    assert fake.connections == 1
    await stream.close()


async def test_concurrent_resharding_leaves_no_orphaned_connections(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("BTC-USD"))
    url = await server_factory(fake)
    stream = make_stream(url, [], [], shard_size=2)
    await stream.subscribe(["A"])
    await stream.connect()
    await wait_for(lambda: stream.connected)
    # two callers change the shard count at the same time (e.g. refresh and ensure_history)
    await asyncio.gather(stream.subscribe(["A", "B", "C"]), stream.subscribe(["A", "B", "C", "D", "E"]))
    await wait_for(lambda: stream.connected)
    await stream.close()
    running = [task for task in asyncio.all_tasks() if task.get_name().startswith("yahoo-stream-") and not task.done()]
    assert running == []


async def test_handler_errors_do_not_break_the_stream(server_factory: ServerFactory) -> None:
    fake = FakeYahoo(send=beat("BTC-USD"))
    url = await server_factory(fake)
    seen: list[Tick] = []

    def handler(tick: Tick) -> None:
        seen.append(tick)
        raise RuntimeError("bug in the consumer")

    stream = YahooStream(handler, lambda connected: None, clock=SystemClock(), options=replace(FAST, url=url))
    await stream.subscribe([])
    await stream.connect()
    await wait_for(lambda: len(seen) >= 3)
    assert fake.connections == 1
    await stream.close()
