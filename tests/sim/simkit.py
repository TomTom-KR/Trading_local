"""Test doubles and helpers for the paper-trading tests."""

import math
import threading
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from pytest import approx

from trading_local.clock import ManualClock
from trading_local.contracts import Broker
from trading_local.db import repo_market, repo_signals
from trading_local.db.database import Database
from trading_local.dto import OrderRequest, PositionOut
from trading_local.events import EventBus
from trading_local.models import Bar, BarSource, Direction, Instrument, Quote, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS

T0 = int(datetime(2026, 11, 20, 15, 0, tzinfo=UTC).timestamp())  # Friday: Xetra and Nasdaq are open
MIN = 60
MONDAY_0000 = int(datetime(2026, 11, 23, 0, 0, tzinfo=UTC).timestamp())

# EUR is the base currency: 1 EUR = 1.25 USD = 0.80 GBP = 160 JPY
FX_QUOTES: dict[str, float] = {"EURUSD=X": 1.25, "EURGBP=X": 0.80, "EURJPY=X": 160.0}
FUTURE = Instrument("ES=F", "E-mini S&P 500", "future", "US", "CME", "USD", timezone="America/Chicago", delay_min=10)
_PRICE_SCALES: dict[str, float] = {i.symbol: i.price_scale for i in [*SAMPLE_INSTRUMENTS, FUTURE]}
SETTLE_S = 600  # barstore.SETTLE_S: a provisional bar of minute ts settles at ts + 60 + delay + SETTLE_S

# Default CostParams: one-way slippage in basis points and the flat fee per order
SLIP_EU, SLIP_US, SLIP_ASIA, SLIP_INDEX, SLIP_CRYPTO = 4e-4, 2e-4, 6e-4, 1.5e-4, 5e-4
FEE = 1.0


class FakeMarketData:
    """The part of ``contracts.MarketData`` the broker and the FX table use: an in-memory quote book."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}

    def set_quote(self, symbol: str, price: float, ts: int) -> None:
        with self._lock:
            self._quotes[symbol] = Quote(symbol=symbol, ts=ts, received_ts=ts, price=price)

    def drop_quote(self, symbol: str) -> None:
        with self._lock:
            self._quotes.pop(symbol, None)

    def quote(self, symbol: str) -> Quote | None:
        with self._lock:
            return self._quotes.get(symbol)

    def quotes(self) -> dict[str, Quote]:
        with self._lock:
            return dict(self._quotes)


class RecordingBus(EventBus):
    """Event bus that records every publication instead of delivering it."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, data: dict[str, Any] | None = None) -> None:
        self.events.append((topic, data or {}))

    def of(self, topic: str) -> list[dict[str, Any]]:
        return [data for event_topic, data in self.events if event_topic == topic]


def add_bar(
    db: Database,
    symbol: str,
    ts: int,
    open_: float,
    *,
    high: float | None = None,
    low: float | None = None,
    close: float | None = None,
    source: BarSource = "rest",
) -> None:
    """Store one 1-minute bar; omitted prices default to the open (storage widens high/low around open/close).

    The default source 'rest' is authoritative (settled at once); 'stream' and 'quote' bars are provisional."""
    close = open_ if close is None else close
    high = max(open_, close) if high is None else high
    low = min(open_, close) if low is None else low
    repo_market.upsert_bar_rows(db, [Bar(symbol, ts, open_, high, low, close, 1000.0, source)])


def add_flat_bars(db: Database, symbol: str, start_ts: int, count: int, price: float, spread: float = 0.1) -> None:
    """``count`` consecutive quiet bars around ``price`` (high = price + spread, low = price - spread)."""
    index = pd.Index([start_ts + MIN * i for i in range(count)], name="ts", dtype="int64")
    frame = pd.DataFrame(
        {"open": price, "high": price + spread, "low": price - spread, "close": price, "volume": 1000.0},
        index=index,
    )
    repo_market.upsert_bars_1m(db, symbol, frame, "rest")


def add_daily(db: Database, symbol: str, day: str, close: float) -> None:
    frame = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 0.0},
        index=pd.Index([day], name="day", dtype="str"),
    )
    repo_market.upsert_bars_1d(db, symbol, frame)


def next_minute(ts: int) -> int:
    """Open time of the first bar a market order placed at ``ts`` can fill on."""
    return -(-ts // MIN) * MIN


def open_position(
    broker: Broker,
    db: Database,
    data: FakeMarketData,
    clock: ManualClock,
    symbol: str,
    side: Direction = "LONG",
    *,
    qty: float | None = None,
    amount: float | None = None,
    price: float = 100.0,
    key: str = "me",
    stop: float | None = None,
    target: float | None = None,
) -> PositionOut:
    """Place a manual order at the clock time, let it fill on the next bar (opening at ``price``) and return
    the open position. Leaves the clock at the close of the fill bar."""
    data.set_quote(symbol, price, clock.ts())
    broker.place_order(
        key,
        OrderRequest(symbol=symbol, side=side, qty=qty, amount_base=amount, stop_price=stop, target_price=target),
    )
    bar_ts = next_minute(clock.ts())
    add_bar(db, symbol, bar_ts, price)
    clock.set(bar_ts + MIN)
    broker.process(clock.now())
    return next(p for p in broker.detail(key).positions if p.symbol == symbol)


def make_signal(
    db: Database,
    symbol: str = "SAP.DE",
    direction: Direction = "LONG",
    *,
    created_ts: int = T0,
    ref_price: float = 100.0,
    stop_dist: float = 2.0,
    target_dist: float = 4.0,
    horizon_min: int = 240,
    intraday: bool = False,
    status: str = "pending_entry",
    actionable: bool = True,
    llm_verdict: str | None = "confirm",
    expected_r: float = 0.5,
) -> Signal:
    """A stored signal (orders reference signals by foreign key). Not issued: ``issued_ts`` and
    ``entry_deadline_ts`` are None unless the test calls ``tracking.issue``."""
    signal = Signal(
        created_ts=created_ts,
        data_ts=created_ts,
        symbol=symbol,
        direction=direction,
        setup="breakout",
        timeframe="15m",
        horizon_min=horizon_min,
        ref_price=ref_price,
        atr=1.0,
        stop_dist=stop_dist,
        target_dist=target_dist,
        score=0.7,
        confidence=0.6,
        engine_version="test",
        status=status,  # type: ignore[arg-type]
        intraday=intraday,
        llm_verdict=llm_verdict,  # type: ignore[arg-type]
        actionable=actionable,
        expected_r=expected_r,
    )
    return repo_signals.insert_signal(db, signal)


def held_value(position: PositionOut) -> tuple[float, float]:
    """``(equity value, result before fees)`` of an open position, recomputed from its fields.

    The current rate is recovered from the marked exposure. A LONG is worth its market value. A SHORT keeps its
    sale proceeds in the instrument currency: against the proceeds booked at the entry rate it owes
    entry_value + qty * scale * (mark - entry) * fx_now, i.e. only its result is exposed to the exchange rate.
    """
    scale = _PRICE_SCALES[position.symbol]
    mark, exposure = position.last_price or 0.0, position.exposure_base or 0.0
    fx_now = abs(exposure) / (position.qty * mark * scale)
    entry_value = position.qty * scale * position.entry_price * position.entry_fx
    if position.side == "LONG":
        return exposure, position.qty * scale * (mark * fx_now - position.entry_price * position.entry_fx)
    result = -position.qty * scale * (mark - position.entry_price) * fx_now
    return -entry_value + result, result


def check_accounting(broker: Broker, key: str) -> None:
    """The identities every state of a portfolio must satisfy."""
    detail = broker.detail(key)
    summary = detail.summary
    held = 0.0
    for position in detail.positions:
        value, result = held_value(position)
        held += value
        assert position.unrealized_pnl == approx(result - position.fees_base, abs=1e-6)
    assert summary.equity == approx(summary.cash + held, abs=1e-6)
    assert summary.equity - summary.start_cash == approx(summary.realized_pnl + summary.unrealized_pnl, abs=1e-6)
    closed = broker.trades(key, 100_000)
    assert summary.fees_paid == approx(sum(p.fees_base for p in closed) + sum(p.fees_base for p in detail.positions))
    assert summary.long_exposure == approx(sum(p.exposure_base or 0.0 for p in detail.positions if p.side == "LONG"))
    assert summary.short_exposure == approx(-sum(p.exposure_base or 0.0 for p in detail.positions if p.side == "SHORT"))
    assert math.isfinite(summary.buying_power) and summary.buying_power >= 0
