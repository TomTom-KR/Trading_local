"""WebSocket hub: unit tests with a scripted socket, plus end-to-end checks through the app."""

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import trading_local
from trading_local.clock import ManualClock
from trading_local.db import repo_market
from trading_local.dto import QuoteOut
from trading_local.events import TOPIC_BAR, TOPIC_NEWS, TOPIC_QUOTES, TOPIC_SIGNAL, TOPIC_STATUS
from trading_local.services import Services
from trading_local.web.deps import ApiContext
from trading_local.web.status import venue_phases
from trading_local.web.ws import Hub, HubTiming

from .conftest import BASE_URL, NOW_TS, WS_URL

QUIET = HubTiming(status_interval_s=3600, ping_interval_s=3600, quote_flush_s=0.05, send_timeout_s=5.0)


class ScriptedSocket:
    """The part of Starlette's WebSocket the hub uses; sends can be held back to simulate a slow client."""

    def __init__(self, *, blocked: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.close_codes: list[int] = []
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.gate = asyncio.Event()
        if not blocked:
            self.gate.set()

    async def accept(self) -> None: ...

    async def send_text(self, text: str) -> None:
        await self.gate.wait()
        self.sent.append(orjson.loads(text))

    async def receive(self) -> dict[str, Any]:
        return await self.incoming.get()

    async def close(self, code: int = 1000) -> None:
        self.close_codes.append(code)
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": code})

    def says(self, text: str) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "text": text})

    def of_type(self, message_type: str) -> list[dict[str, Any]]:
        return [message for message in self.sent if message["type"] == message_type]


async def eventually(condition: Callable[[], bool], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


@pytest.fixture
async def make_hub(services: Services) -> AsyncIterator[Callable[[HubTiming], Hub]]:
    services.bus.bind_loop()
    ctx = ApiContext(services)
    hubs: list[Hub] = []

    def factory(timing: HubTiming = QUIET) -> Hub:
        hubs.append(Hub(ctx, timing))
        return hubs[-1]

    yield factory
    for hub in hubs:
        await hub.stop()
    await ctx.aclose()


async def connect(hub: Hub, socket: ScriptedSocket) -> asyncio.Task[None]:
    if not hub.running:
        await hub.start()
    task = asyncio.create_task(hub.serve(socket))  # type: ignore[arg-type]
    await eventually(lambda: hub.client_count == 1 and (bool(socket.sent) or not socket.gate.is_set()))
    return task


# ------------------------------------------------------------------------------------------------
# Hub unit tests
# ------------------------------------------------------------------------------------------------
async def test_hello_then_broadcast_of_bus_events(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    await connect(hub, socket)
    assert socket.sent[0] == {"type": "hello", "data": {"server_time": NOW_TS, "version": trading_local.__version__}}
    services.bus.publish(TOPIC_NEWS, {"id": 7, "title": "Schlagzeile"})
    services.bus.publish(TOPIC_SIGNAL, {"id": 3})
    await eventually(lambda: len(socket.sent) == 3)
    assert socket.sent[1:] == [
        {"type": "news", "data": {"id": 7, "title": "Schlagzeile"}},
        {"type": "signal", "data": {"id": 3}},
    ]


async def test_quotes_are_coalesced_per_symbol(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    await connect(hub, socket)
    services.bus.publish(TOPIC_QUOTES, {"items": [{"symbol": "NVDA", "price": 1.0}]})
    services.bus.publish(
        TOPIC_QUOTES, {"items": [{"symbol": "NVDA", "price": 2.0}, {"symbol": "SAP.DE", "price": 5.0}]}
    )
    services.bus.publish(TOPIC_QUOTES, {"items": [{"symbol": "NVDA", "price": 3.0}, "broken", {"price": 9}]})
    await eventually(lambda: bool(socket.of_type("quotes")))
    await asyncio.sleep(0.15)  # further flush ticks send nothing new
    assert socket.of_type("quotes") == [
        {"type": "quotes", "data": {"items": [{"symbol": "NVDA", "price": 3.0}, {"symbol": "SAP.DE", "price": 5.0}]}}
    ]


async def test_bars_only_reach_subscribers_of_the_symbol(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    await connect(hub, socket)

    async def bars_after_publishing(*symbols: str) -> list[str]:
        socket.sent.clear()
        for symbol in symbols:
            services.bus.publish(TOPIC_BAR, {"symbol": symbol, "bar": [NOW_TS, 1, 2, 0.5, 1.5, 10], "source": "rest"})
        services.bus.publish(TOPIC_NEWS, {"id": 0})  # marker: everything before it has been dispatched
        await eventually(lambda: bool(socket.of_type("news")))
        return [message["data"]["symbol"] for message in socket.of_type("bar")]

    assert await bars_after_publishing("NVDA") == []
    socket.says(json.dumps({"type": "subscribe", "symbols": ["nvda", "NVDA", 5, "", "X" * 30]}))
    await asyncio.sleep(0.02)
    assert await bars_after_publishing("NVDA", "SAP.DE") == ["NVDA"]
    socket.says(json.dumps({"type": "subscribe", "symbols": ["SAP.DE"]}))  # replaces the subscription
    await asyncio.sleep(0.02)
    assert await bars_after_publishing("NVDA", "SAP.DE") == ["SAP.DE"]
    socket.says(json.dumps({"type": "subscribe", "symbols": [f"S{i}" for i in range(25)]}))
    await asyncio.sleep(0.02)
    assert await bars_after_publishing("S0", "S19", "S20", "S24") == ["S0", "S19"]  # at most 20 symbols


async def test_unknown_client_messages_are_ignored(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    task = await connect(hub, socket)
    for text in ("not json", "[1, 2]", json.dumps({"type": "dance"}), json.dumps({"type": "subscribe"}), "x" * 5000):
        socket.says(text)
    socket.incoming.put_nowait({"type": "websocket.receive", "bytes": b"\x00\x01"})
    await asyncio.sleep(0.02)
    services.bus.publish(TOPIC_NEWS, {"id": 1})
    await eventually(lambda: bool(socket.of_type("news")))
    assert not task.done()


async def test_ping_and_status_are_sent_periodically(make_hub: Callable[..., Hub]) -> None:
    hub = make_hub(HubTiming(status_interval_s=0.05, ping_interval_s=0.05, quote_flush_s=0.05))
    socket = ScriptedSocket()
    await connect(hub, socket)
    await eventually(lambda: bool(socket.of_type("ping")) and bool(socket.of_type("status")))
    assert socket.of_type("ping")[0] == {"type": "ping"}
    status = socket.of_type("status")[0]["data"]
    assert (status["engine_version"], status["base_currency"]) == ("engine-test-1", "EUR")
    assert status["server_time"] == NOW_TS
    assert status["mail"]["recipient_masked"] == "e***@example.invalid"


async def test_session_phase_changes_are_announced_once(
    make_hub: Callable[..., Hub], clock: ManualClock
) -> None:
    hub = make_hub(HubTiming(status_interval_s=0.02, ping_interval_s=3600, quote_flush_s=0.05))
    socket = ScriptedSocket()
    await connect(hub, socket)
    await eventually(lambda: len(socket.of_type("status")) >= 2)
    assert socket.of_type("sessions") == []  # the first push only learns the phases
    clock.advance(90 * 60)  # Friday 16:30 UTC: Xetra and London close (test calendar)
    await eventually(lambda: len(socket.of_type("sessions")) >= 2)
    pushes = len(socket.of_type("status"))
    await eventually(lambda: len(socket.of_type("status")) >= pushes + 2)  # later pushes repeat nothing
    assert sorted((m["data"]["exchange"], m["data"]["phase"]) for m in socket.of_type("sessions")) == [
        ("XETR", "closed"),
        ("XLON", "closed"),
    ]


def test_venue_phases_cover_every_venue_with_an_instrument(services: Services) -> None:
    repo_market.set_instrument_flags(services.db, "7203.T", active=False)
    services.instruments.reload()
    phases = venue_phases(services)  # Friday 15:00 UTC
    assert phases == {
        "CRYPTO": "open", "FX": "open", "XETR": "open", "XLON": "open", "XNAS": "open", "XNYS": "open",
        "XTKS": "closed",  # inactive instruments count as well: the GUI lists them too
    }


def test_venue_phases_skip_venues_the_calendar_does_not_know(services: Services) -> None:
    calendar = services.calendar

    class PartialCalendar:
        def state(self, exchange: str, at: datetime) -> Any:
            if exchange == "XTKS":
                raise KeyError(exchange)
            return calendar.state(exchange, at)

    phases = venue_phases(dataclasses.replace(services, calendar=PartialCalendar()))  # type: ignore[arg-type]
    assert "XTKS" not in phases and phases["XNAS"] == "open"


async def test_status_events_from_the_bus_are_not_forwarded(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    await connect(hub, socket)
    services.bus.publish(TOPIC_STATUS, {"forged": True})
    services.bus.publish(TOPIC_NEWS, {"id": 1})
    await eventually(lambda: bool(socket.of_type("news")))
    assert socket.of_type("status") == []


async def test_payloads_are_serialised_or_dropped(make_hub: Callable[..., Hub], services: Services) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    await connect(hub, socket)
    services.bus.publish(TOPIC_SIGNAL, {"broken": object()})
    services.bus.publish(TOPIC_NEWS, {"quote": QuoteOut(symbol="NVDA", price=1.5, ts=NOW_TS)})
    await eventually(lambda: bool(socket.of_type("news")))
    assert socket.of_type("signal") == []
    assert socket.of_type("news")[0]["data"]["quote"]["price"] == 1.5


async def test_a_slow_client_loses_its_oldest_messages_without_blocking_the_hub(
    make_hub: Callable[..., Hub], services: Services
) -> None:
    hub = make_hub(HubTiming(status_interval_s=3600, ping_interval_s=3600, quote_flush_s=0.05, outbox_size=3))
    slow = ScriptedSocket(blocked=True)
    fast = ScriptedSocket()
    await connect(hub, slow)  # its hello is stuck in send_text
    task = asyncio.create_task(hub.serve(fast))  # type: ignore[arg-type]
    await eventually(lambda: hub.client_count == 2 and bool(fast.sent))
    for n in range(10):
        services.bus.publish(TOPIC_NEWS, {"n": n})
        await asyncio.sleep(0.005)
    assert [message["data"]["n"] for message in fast.of_type("news")] == list(range(10))  # not held up
    slow.gate.set()
    await eventually(lambda: len(slow.sent) == 4)
    assert [message["data"]["n"] for message in slow.of_type("news")] == [7, 8, 9]
    assert not task.done()


async def test_a_client_that_stays_blocked_is_disconnected(make_hub: Callable[..., Hub]) -> None:
    hub = make_hub(HubTiming(status_interval_s=3600, ping_interval_s=3600, send_timeout_s=0.05))
    socket = ScriptedSocket(blocked=True)
    task = await connect(hub, socket)
    await asyncio.wait_for(task, timeout=1.0)
    assert socket.close_codes == [1013]
    assert hub.client_count == 0


async def test_a_disconnecting_client_is_removed(make_hub: Callable[..., Hub]) -> None:
    hub = make_hub()
    socket = ScriptedSocket()
    task = await connect(hub, socket)
    socket.incoming.put_nowait({"type": "websocket.disconnect", "code": 1000})
    await asyncio.wait_for(task, timeout=1.0)
    assert hub.client_count == 0 and socket.close_codes == []


async def test_stop_closes_every_connection(make_hub: Callable[..., Hub]) -> None:
    hub = make_hub()
    sockets = [ScriptedSocket(), ScriptedSocket()]
    tasks = [await connect(hub, sockets[0])]
    tasks.append(asyncio.create_task(hub.serve(sockets[1])))  # type: ignore[arg-type]
    await eventually(lambda: hub.client_count == 2)
    await hub.stop()
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert [socket.close_codes for socket in sockets] == [[1001], [1001]]
    assert not hub.running
    late = ScriptedSocket()
    await hub.serve(late)  # type: ignore[arg-type]
    assert late.close_codes == [1013] and late.sent == []


# ------------------------------------------------------------------------------------------------
# Through the app (TestClient)
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def live_app(app: FastAPI) -> FastAPI:
    app.state.hub.timing = HubTiming(status_interval_s=0.05, ping_interval_s=0.05, quote_flush_s=0.05)
    return app


def receive_until(websocket: Any, message_type: str, limit: int = 200) -> dict[str, Any]:
    for _ in range(limit):
        message: dict[str, Any] = websocket.receive_json()
        if message["type"] == message_type:
            return message
    raise AssertionError(f"no {message_type!r} message")


def test_events_published_from_other_threads_reach_the_browser(live_app: FastAPI, services: Services) -> None:
    with TestClient(live_app, base_url=BASE_URL) as client, client.websocket_connect(WS_URL) as websocket:
        assert websocket.receive_json()["type"] == "hello"
        services.bus.publish(TOPIC_SIGNAL, {"id": 42})  # from the test thread, like a worker thread would
        assert receive_until(websocket, "signal")["data"] == {"id": 42}
        services.bus.publish(TOPIC_QUOTES, {"items": [{"symbol": "NVDA", "price": 181.0}]})
        assert receive_until(websocket, "quotes")["data"]["items"] == [{"symbol": "NVDA", "price": 181.0}]
        assert receive_until(websocket, "ping") == {"type": "ping"}
        assert receive_until(websocket, "status")["data"]["version"] == trading_local.__version__


def test_bar_subscription_through_the_app(live_app: FastAPI, services: Services) -> None:
    with TestClient(live_app, base_url=BASE_URL) as client, client.websocket_connect(WS_URL) as websocket:
        websocket.send_json({"type": "subscribe", "symbols": ["NVDA"]})
        websocket.receive_json()  # hello; the subscription is applied while the next messages arrive
        receive_until(websocket, "ping")
        services.bus.publish(TOPIC_BAR, {"symbol": "SAP.DE", "bar": [NOW_TS, 1, 1, 1, 1, 0], "source": "rest"})
        services.bus.publish(TOPIC_BAR, {"symbol": "NVDA", "bar": [NOW_TS, 2, 2, 2, 2, 0], "source": "rest"})
        assert receive_until(websocket, "bar")["data"]["symbol"] == "NVDA"


def test_shutdown_disconnects_open_websockets(live_app: FastAPI) -> None:
    with TestClient(live_app, base_url=BASE_URL) as client:
        with client.websocket_connect(WS_URL) as websocket:
            assert websocket.receive_json()["type"] == "hello"
        assert live_app.state.hub.client_count == 0
