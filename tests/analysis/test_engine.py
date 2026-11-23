"""End-to-end tests of the live signal engine.

The engineered market: Nasdaq hours of the FakeCalendar (14:30-21:00 UTC), twelve and more sessions of a
gently rising random walk, then on Fri 2026-11-20 a rally, a pullback and at 16:30-16:35 UTC a high-volume
close above the 20-bar channel - a textbook LONG breakout and the ONLY setup that fires all day. After the
breakout the price either rises to the target ('up', about 17:00) or falls to the stop ('down', 16:47).
All bars are stored up front; the engine sees only what had closed at the ManualClock time (BarReader).
"""

import asyncio
import functools
import logging
import math
import re
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_local.analysis import build_engine
from trading_local.analysis import engine as engine_module
from trading_local.analysis.engine import (
    NOTE_MARKET_CLOSED,
    NOTE_TOO_LITTLE_DATA,
    NOTE_UNKNOWN_SYMBOL,
    SignalEngine,
    closed_ts,
    sample_note_de,
    warmup_start_ts,
)
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.analysis.setups import SETUPS
from trading_local.clock import ManualClock, to_ts
from trading_local.config import DEFAULT_COSTS, Settings
from trading_local.contracts import Engine, TradeReview, TradeReviewContext
from trading_local.db import repo_market, repo_news, repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import AnalysisOut, OrderOut
from trading_local.events import TOPIC_CHART_ANALYSIS, TOPIC_SIGNAL, EventBus
from trading_local.models import Bar, ChartAnalysis, Instrument, NewsAnalysis, NewsItem, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_bars


def utc(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """Epoch seconds of a time in November 2026 (the 20th is a Friday)."""
    return to_ts(datetime(2026, 11, day, hour, minute, second, tzinfo=UTC))


DECISION = utc(20, 16, 35)  # end of the engineered breakout bucket
PNG = b"\x89PNG-fake-chart"
VERSION = ENGINE_PARAMS.version

NVDA = SAMPLE_INSTRUMENTS[0]
SPX = replace(next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "^GSPC"), tradable=False)


def twin(symbol: str, sort_order: int) -> Instrument:
    """Another Nasdaq share like NVDA (same benchmark)."""
    return replace(NVDA, symbol=symbol, name=f"Firma {symbol}", aliases=(), sort_order=sort_order)


# ------------------------------------------------------------------------------------------------
# Engineered market
# ------------------------------------------------------------------------------------------------
def ramp(
    start_ts: int, minutes: int, start_price: float, step: float, *, wiggle: float = 0.0015,
    volume: float = 1000.0, seed: int = 0,
) -> pd.DataFrame:  # fmt: skip
    """Deterministic 1-minute bars: the close moves by ``step`` (log) per minute plus a small zigzag."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, minutes + 1)
    close = start_price * np.exp(step * k + wiggle * np.sin(k * 1.7) * 0.5)
    open_ = np.r_[start_price, close[:-1]]
    spread = start_price * wiggle * (0.6 + 0.4 * rng.random(minutes))
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.full(minutes, volume),
        },
        index=pd.Index(start_ts + 60 * np.arange(minutes, dtype=np.int64), name="ts"),
    )


@functools.cache
def market(path: str = "up", volume_factor: float = 3.0, seed: int = 7, start: float = 100.0) -> pd.DataFrame:
    """Nasdaq sessions 2026-11-02 .. 2026-11-20 with the breakout at 16:30-16:35 UTC on the 20th."""
    frames = []
    price = start
    past_days = [day for day in range(2, 20) if date(2026, 11, day).weekday() < 5]
    for k, day in enumerate(past_days):
        frame = make_bars(utc(day, 14, 30), 390, start_price=price, seed=seed * 100 + k, drift=0.00004, vol=0.0012)
        frames.append(frame)
        price = float(frame["close"].iloc[-1])
    pieces = [
        (utc(20, 14, 30), 75, 0.00030, 0.0015, 1.0),  # rally
        (utc(20, 15, 45), 45, -0.00012, 0.0015, 1.0),  # pullback
        (utc(20, 16, 30), 5, 0.0030, 0.0005, volume_factor),  # breakout bucket
        (utc(20, 16, 35), 265, 0.0006 if path == "up" else -0.0006, 0.0008, 1.0),  # follow-through
    ]
    for k, (start_ts, minutes, step, wiggle, volume) in enumerate(pieces):
        frame = ramp(start_ts, minutes, price, step, wiggle=wiggle, volume=1000.0 * volume, seed=k + 1)
        frames.append(frame)
        price = float(frame["close"].iloc[-1])
    return pd.concat(frames)


def daily_from(minutes: pd.DataFrame, before_day: str = "2026-11-20", history: int = 260) -> pd.DataFrame:
    """Daily bars of the stored sessions before ``before_day``, preceded by ``history`` days of an uptrend."""
    days = np.datetime_as_string(minutes.index.to_numpy().astype("datetime64[s]"), unit="D")
    recent = minutes.groupby(days).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    )  # fmt: skip
    recent = recent[recent.index < before_day]
    first = date.fromisoformat(str(recent.index[0]))
    rng = np.random.default_rng(5)
    closes = float(recent["open"].iloc[0]) * np.exp(-np.cumsum(0.004 + 0.012 * rng.standard_normal(history)))[::-1]
    old_days = [(first - timedelta(days=history - i)).isoformat() for i in range(history)]
    old = pd.DataFrame(
        {"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes, "volume": 1e6}, index=old_days
    )
    daily = pd.concat([old, recent])
    daily.index = pd.Index(daily.index.astype(str), dtype="str", name="day")
    return daily.astype("float64")


# ------------------------------------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------------------------------------
class RecordingBus(EventBus):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[tuple[str, dict[str, Any]]] = []

    def publish(self, topic: str, data: dict[str, Any] | None = None) -> None:
        self.events.append((topic, data or {}))
        super().publish(topic, data)

    def topics(self, topic: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == topic]


class FakeLlm:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.verdict = "confirm"
        self.fail = False
        self.review_gate: asyncio.Event | None = None
        self.chart_gate: asyncio.Event | None = None
        self.on_review: Callable[[], None] | None = None
        self.reviews: list[tuple[TradeReviewContext, float]] = []
        self.charts: list[tuple[str, str, dict[str, Any]]] = []

    async def review_trade(self, ctx: TradeReviewContext, *, timeout_s: float) -> TradeReview:
        self.reviews.append((ctx, timeout_s))
        if self.fail:
            raise RuntimeError("model crashed")
        if self.review_gate is not None:
            await self.review_gate.wait()
        if self.on_review is not None:
            self.on_review()
        if self.verdict == "unavailable":
            return TradeReview(verdict="unavailable", latency_ms=45_000)
        return TradeReview(
            verdict=self.verdict, conviction=0.7, rationale_de="KI: Ausbruch mit Volumen.",
            risks_de="KI: Index schwach.", latency_ms=1500,
        )  # fmt: skip

    async def analyze_chart(
        self, instrument: Instrument, timeframe: str, chart_png: bytes, facts: Mapping[str, Any]
    ) -> ChartAnalysis | None:
        self.charts.append((instrument.symbol, timeframe, dict(facts)))
        if self.chart_gate is not None:
            await self.chart_gate.wait()
        if self.fail:
            raise RuntimeError("model crashed")
        return ChartAnalysis(
            symbol=instrument.symbol, ts=self.clock.ts(), timeframe=timeframe, model="fake", bias="bullish",
            strength=3, comment_de="Aufwärtstrend intakt.",
        )  # fmt: skip


class FakeCharts:
    def __init__(self) -> None:
        self.result: bytes | None = PNG
        self.fail = False
        self.calls: list[tuple[str, str, int | None, int | None]] = []

    async def render(
        self, symbol: str, timeframe: str = "15m", *, asof_ts: int | None = None, bars: int = 120,
        signal: Signal | None = None, width: int = 1024, height: int = 576,
    ) -> bytes | None:  # fmt: skip
        self.calls.append((symbol, timeframe, asof_ts, None if signal is None else signal.id))
        if self.fail:
            raise RuntimeError("renderer crashed")
        return self.result


class FakeMailer:
    """Records mails; like the real mailer it marks delivered entry mails in the database and on the object."""

    def __init__(self, db: Database, clock: ManualClock) -> None:
        self.db = db
        self.clock = clock
        self.fail = False
        self.entries: list[tuple[int | None, bytes | None]] = []
        self.exits: list[tuple[int | None, str]] = []

    async def send_signal(self, signal: Signal, instrument: Instrument, chart_png: bytes | None = None) -> bool:
        self.entries.append((signal.id, chart_png))
        if self.fail:
            raise RuntimeError("smtp down")
        assert signal.id is not None
        repo_signals.mark_mailed(self.db, signal.id, self.clock.ts())
        signal.mailed_ts = self.clock.ts()
        return True

    async def send_exit(self, signal: Signal, instrument: Instrument) -> bool:
        self.exits.append((signal.id, signal.status))
        if self.fail:
            raise RuntimeError("smtp down")
        return True


class FakeBroker:
    def __init__(self) -> None:
        self.fail = False
        self.executed: list[tuple[int | None, str, bool]] = []

    def execute_signal(self, signal: Signal) -> OrderOut | None:
        self.executed.append((signal.id, signal.status, signal.actionable))
        if self.fail:
            raise RuntimeError("broker crashed")
        return None


@dataclass
class World:
    settings: Settings
    db: Database
    clock: ManualClock
    bus: RecordingBus
    calendar: FakeCalendar
    book: InstrumentBook
    store: SettingsStore
    llm: FakeLlm
    charts: FakeCharts
    mailer: FakeMailer
    broker: FakeBroker
    engines: list[SignalEngine] = field(default_factory=list)

    def seed(self, instrument: Instrument, frame: pd.DataFrame, *, daily: bool = True) -> None:
        repo_market.upsert_instruments(self.db, [instrument], keep_user_flags=False)
        repo_market.upsert_bars_1m(self.db, instrument.symbol, frame, "rest")
        if daily:
            repo_market.upsert_bars_1d(self.db, instrument.symbol, daily_from(frame))
        self.book.reload()

    def seed_nvda(self, path: str = "up", *, until: int | None = None) -> None:
        frame = market(path)
        self.seed(NVDA, frame if until is None else frame[frame.index < until])

    def engine(self, *, calendar: Any = None, runtime_settings: Any = None, **settings: Any) -> SignalEngine:
        engine = build_engine(
            settings=self.settings.model_copy(update=settings),
            db=self.db,
            clock=self.clock,
            bus=self.bus,
            calendar=calendar or self.calendar,
            instruments=self.book,
            runtime_settings=runtime_settings or self.store,
            costs=DEFAULT_COSTS,
            llm=self.llm,
            charts=self.charts,
            mailer=self.mailer,
            broker=self.broker,
        )
        assert isinstance(engine, SignalEngine)
        self.engines.append(engine)
        return engine

    def prefs(self, **patch: Any) -> None:
        self.store.update(patch, now_ts=self.clock.ts())

    def signals(self, symbol: str | None = None) -> list[Signal]:
        """All stored signals, oldest first."""
        return repo_signals.list_signals(self.db, symbol=symbol, limit=10_000)[::-1]

    def only_signal(self, symbol: str = "NVDA") -> Signal:
        signals = self.signals(symbol)
        assert len(signals) == 1, [(s.id, s.status, s.direction, s.created_ts) for s in signals]
        return signals[0]

    def activity(self) -> list[str]:
        return [entry.message for entry in reversed(repo_ops.list_activity(self.db, category="signal"))]


@pytest.fixture
async def world(settings: Settings, db: Database, clock: ManualClock) -> AsyncIterator[World]:
    bus = RecordingBus()
    bus.bind_loop(asyncio.get_running_loop())
    w = World(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        calendar=FakeCalendar(),
        book=InstrumentBook(db),
        store=SettingsStore(db),
        llm=FakeLlm(clock),
        charts=FakeCharts(),
        mailer=FakeMailer(db, clock),
        broker=FakeBroker(),
    )
    w.seed(SPX, market(seed=11, start=6000.0), daily=False)
    w.prefs(chart_scan_enabled=False)
    yield w
    for engine in w.engines:
        await engine.stop()


async def run(engine: SignalEngine, clock: ManualClock, first: int, last: int, *, drain: bool = True) -> None:
    """One minute cycle per minute from ``first`` to ``last`` (inclusive), each 3 s after the minute."""
    for minute in range(first, last + 1, 60):
        clock.set(minute + 3)
        await engine.on_minute(clock.now(), [])
        if drain:
            await engine.drain()


async def run_until(engine: SignalEngine, clock: ManualClock, first: int, done: Callable[[], bool], limit: int) -> None:
    """Minute cycles from ``first`` until ``done()`` holds (fails after ``limit``)."""
    minute = first
    while not done():
        assert minute <= limit, "condition not reached in time"
        clock.set(minute + 3)
        await engine.on_minute(clock.now(), [])
        await engine.drain()
        minute += 60


async def until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


async def detect(world: World, engine: SignalEngine, *, drain: bool = True) -> None:
    """Initialise the engine shortly before the breakout bucket and run through it."""
    await run(engine, world.clock, DECISION - 120, DECISION, drain=drain)


def stored_signal(db: Database, **fields: Any) -> Signal:
    values: dict[str, Any] = {
        "created_ts": DECISION - 3600, "data_ts": DECISION - 3600, "symbol": "NVDA", "direction": "LONG",
        "setup": "breakout", "timeframe": "15m", "horizon_min": 180, "ref_price": 100.0, "atr": 1.0,
        "stop_dist": 1.0, "target_dist": 2.0, "score": 0.7, "confidence": 0.45, "engine_version": VERSION,
    }  # fmt: skip
    values.update(fields)
    return repo_signals.insert_signal(db, Signal(**values))


# ------------------------------------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------------------------------------
async def test_build_engine_implements_the_protocol(world: World) -> None:
    engine = world.engine()
    assert isinstance(engine, Engine)
    assert engine.version == VERSION


def test_warmup_covers_twelve_sessions_of_each_venue_type() -> None:
    calendar = FakeCalendar()
    now = DECISION + 3
    assert warmup_start_ts(calendar, "XNAS", now) == utc(5, 14, 30)  # 12 weekdays incl. Fri 20th
    assert warmup_start_ts(calendar, "CRYPTO", now) == utc(9, 0, 0)  # 12 UTC days
    assert warmup_start_ts(calendar, "XTKS", utc(20, 1, 0)) == utc(5, 0, 0)
    assert warmup_start_ts(calendar, "XNAS", utc(21, 12, 0)) == utc(5, 14, 30)  # Saturday: no new session
    assert warmup_start_ts(calendar, "XNAS", utc(23, 15, 0)) == utc(6, 14, 30)  # Monday after the open


def test_closed_ts_and_sample_note() -> None:
    base = Signal(1, 1, "X", "LONG", "breakout", "15m", 1, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, "v", status="cancelled")
    assert closed_ts(replace(base, entry_deadline_ts=50)) == 50
    assert closed_ts(replace(base, checked_ts=40, entry_deadline_ts=50)) == 40
    assert closed_ts(replace(base, exit_ts=70, checked_ts=40)) == 70
    assert sample_note_de(12) == "Erst 12 abgeschlossene Signale – statistisch noch nicht belastbar."
    assert sample_note_de(1).startswith("Erst 1 abgeschlossenes Signal")
    assert "keine Aussage" in sample_note_de(0) and "95" in sample_note_de(150)


# ------------------------------------------------------------------------------------------------
# Full life cycle
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("path", "status"), [("up", "target"), ("down", "stopped")])
async def test_confirmed_candidate_is_issued_traded_mailed_and_closed(world: World, path: str, status: str) -> None:
    world.seed_nvda(path)
    engine = world.engine()
    await detect(world, engine)

    signal = world.only_signal()
    assert (signal.setup, signal.direction, signal.timeframe) == ("breakout", "LONG", "15m")
    assert signal.created_ts == DECISION + 3 and signal.data_ts == DECISION
    assert signal.status == "pending_entry" and signal.issued_ts == DECISION + 3
    assert signal.actionable and signal.shadow_reason == ""
    assert signal.llm_verdict == "confirm" and signal.llm_conviction == 0.7 and signal.llm_latency_ms == 1500
    assert signal.rationale_de == "KI: Ausbruch mit Volumen." and signal.risks_de == "KI: Index schwach."
    assert 0.8 < signal.score < 0.9 and signal.expected_r > 0
    assert signal.confidence == pytest.approx(1 / 3 + ENGINE_PARAMS.prior_slope * (signal.score - 0.5), abs=1e-4)
    assert signal.entry_deadline_ts == DECISION + 3 + ENGINE_PARAMS.entry_grace_min * 60
    assert world.broker.executed == [(signal.id, "pending_entry", True)]
    assert world.mailer.entries == [(signal.id, PNG)]  # the review chart is reused for the mail
    assert world.charts.calls == [("NVDA", "15m", DECISION + 3, signal.id)]
    assert signal.mailed_ts == DECISION + 3
    confidence = round(signal.confidence * 100)
    assert world.activity() == [f"Empfehlung: LONG NVDA (Ausbruch), Konfidenz {confidence} %"]

    await run(engine, world.clock, DECISION + 60, DECISION + 120)
    signal = world.only_signal()
    assert signal.status == "active" and signal.entry_ts == DECISION + 60  # first bar opening after the issue
    assert world.activity()[-1].startswith("Einstieg: LONG NVDA (Ausbruch), Signal #")

    await run_until(engine, world.clock, DECISION + 180, lambda: world.only_signal().status == status, utc(20, 18))
    signal = world.only_signal()
    assert (signal.outcome_r or 0.0) > 0 if status == "target" else (signal.outcome_r or 0.0) < -0.9
    assert world.mailer.exits == [(signal.id, status)]
    assert world.activity()[-1].startswith("Ziel erreicht" if status == "target" else "Stopp ausgelöst")
    assert world.broker.executed == [(signal.id, "pending_entry", True)]
    statuses = [data["status"] for data in world.bus.topics(TOPIC_SIGNAL)]
    assert statuses[:3] == ["review", "pending_entry", "active"] and statuses[-1] == status

    await run(engine, world.clock, world.clock.ts() // 60 * 60 + 60, utc(20, 18, 0))
    assert len(world.signals()) == 1 and len(world.mailer.exits) == 1  # cooldown, no second exit mail


async def test_a_mailed_signal_that_reaches_its_time_stop_gets_an_exit_mail(world: World) -> None:
    world.seed_nvda()
    entry_ts = DECISION + 60
    entry = float(market().loc[entry_ts, "open"])
    stored = stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="active", actionable=True,
        ref_price=entry, stop_dist=entry * 0.2, target_dist=entry * 0.4, entry_ts=entry_ts, entry_price=entry,
        stop_price=entry * 0.8, target_price=entry * 1.4, expires_ts=entry_ts + 180, checked_ts=entry_ts,
        mailed_ts=DECISION + 5, mfe_r=0.0, mae_r=0.0,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, entry_ts + 60, entry_ts + 300)
    signal = world.only_signal()
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "time_stop", entry_ts + 180)
    assert world.mailer.exits == [(stored.id, "expired")]
    assert world.activity()[-1].startswith("Zeitablauf (Zeitstopp): LONG NVDA (Ausbruch)")


def bar_frame(rows: Mapping[int, tuple[float, float, float, float]]) -> pd.DataFrame:
    """Canonical 1-minute bars from {ts: (open, high, low, close)}."""
    index = sorted(rows)
    return pd.DataFrame(
        {
            "open": [rows[ts][0] for ts in index],
            "high": [rows[ts][1] for ts in index],
            "low": [rows[ts][2] for ts in index],
            "close": [rows[ts][3] for ts in index],
            "volume": [1000.0] * len(index),
        },
        index=pd.Index(index, dtype="int64", name="ts"),
    )


def stream_bar(db: Database, ts: int, open_: float, high: float, low: float, close: float) -> None:
    """A provisional bar as the live stream writes it (REST replaces it later)."""
    repo_market.upsert_bar_rows(db, [Bar("NVDA", ts, open_, high, low, close, 1000.0, "stream")])


def active_breakout(world: World, entry_ts: int, *, stop_pct: float = 0.01, target_pct: float = 0.05) -> Signal:
    """A mailed, actionable LONG that entered at the open of ``entry_ts`` and has processed that bar."""
    entry = float(market().loc[entry_ts, "open"])
    return stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="active", actionable=True,
        ref_price=entry, stop_dist=entry * stop_pct, target_dist=entry * target_pct, entry_ts=entry_ts,
        entry_price=entry, stop_price=entry * (1 - stop_pct), target_price=entry * (1 + target_pct),
        expires_ts=entry_ts + 180 * 60, checked_ts=entry_ts, mailed_ts=DECISION + 5, mfe_r=0.0, mae_r=0.0,
    )  # fmt: skip


async def test_bars_filled_in_behind_a_newer_bar_are_still_tracked(world: World) -> None:
    """RUN-1 / CAUS-1: after a restart the first cycle stores the CURRENT minute's stream bar, the backfill
    fills the downtime minutes later. The tracker waits at the hole and then checks the late bars - one of
    them went through the stop."""
    entry_ts = DECISION + 60
    world.seed_nvda(until=entry_ts + 120)  # REST history up to the minute after the entry
    stored = active_breakout(world, entry_ts)
    entry = stored.entry_price or 0.0
    now_minute = entry_ts + 9 * 60  # the process was down; the current minute arrives first
    stream_bar(world.db, now_minute, entry * 1.001, entry * 1.002, entry * 1.0005, entry * 1.001)
    engine = world.engine()
    await run(engine, world.clock, now_minute + 60, now_minute + 60)
    signal = world.only_signal()
    assert signal.status == "active" and signal.checked_ts == entry_ts + 60  # waits at the young hole

    gap = {
        ts: (entry * 1.001, entry * 1.002, entry * (0.98 if ts == entry_ts + 300 else 1.0005), entry * 1.001)
        for ts in range(entry_ts + 120, now_minute + 60, 60)
    }
    repo_market.upsert_bars_1m(world.db, "NVDA", bar_frame(gap), "rest")  # the backfill arrives
    await run(engine, world.clock, now_minute + 120, now_minute + 120)
    signal = world.only_signal()
    assert (signal.status, signal.exit_ts) == ("stopped", entry_ts + 300)
    assert world.mailer.exits == [(stored.id, "stopped")]


async def test_a_provisional_stream_bar_is_judged_by_its_rest_correction(world: World) -> None:
    """CAUS-1: the stream bar shows an extreme through the stop that the authoritative REST bar does not
    contain. The tracker acts on the final bar only."""
    entry_ts = DECISION + 60
    world.seed_nvda(until=entry_ts + 120)
    stored = active_breakout(world, entry_ts)
    entry = stored.entry_price or 0.0
    spike = entry_ts + 120
    stream_bar(world.db, spike, entry * 1.001, entry * 1.002, entry * 0.985, entry * 1.001)
    engine = world.engine()
    await run(engine, world.clock, spike + 60, spike + 120)
    signal = world.only_signal()
    assert signal.status == "active" and signal.checked_ts == entry_ts + 60  # provisional: not acted on

    corrected = {spike: (entry * 1.001, entry * 1.002, entry * 0.999, entry * 1.001)}
    repo_market.upsert_bars_1m(world.db, "NVDA", bar_frame(corrected), "rest")
    await run(engine, world.clock, spike + 180, spike + 180)
    signal = world.only_signal()
    assert signal.status == "active" and signal.checked_ts == spike  # the final bar never touched the stop
    assert world.mailer.exits == []


async def test_a_pending_signal_is_not_cancelled_while_its_entry_bar_may_still_arrive(world: World) -> None:
    """RUN-1: restarted 13 minutes after the issue, nothing of the entry window is stored yet. The wall
    clock alone would cancel the signal ('kein Einstieg') although the backfill delivers the entry bar."""
    world.seed_nvda(until=DECISION)
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.05,
        entry_deadline_ts=DECISION + 3 + ENGINE_PARAMS.entry_grace_min * 60, mailed_ts=DECISION + 5,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, DECISION + 13 * 60, DECISION + 13 * 60)
    assert world.only_signal().status == "pending_entry"

    frame = market()
    window = frame[(frame.index >= DECISION) & (frame.index < DECISION + 14 * 60)]
    repo_market.upsert_bars_1m(world.db, "NVDA", window, "rest")
    await run(engine, world.clock, DECISION + 14 * 60, DECISION + 14 * 60)
    signal = world.only_signal()
    assert signal.status == "active" and signal.entry_ts == DECISION + 60


async def test_after_a_long_downtime_a_pending_signal_waits_for_the_backfill(world: World) -> None:
    """RUN-1: restarted 30 minutes after the issue. Every minute of the entry window is older than the settle
    horizon, yet none of them could arrive while the platform was down: the first cycle after the restart
    must not cancel the signal before the backfill has delivered the entry bar."""
    world.seed_nvda(until=DECISION)
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.05,
        entry_deadline_ts=DECISION + 3 + ENGINE_PARAMS.entry_grace_min * 60, mailed_ts=DECISION + 5,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, DECISION + 30 * 60, DECISION + 30 * 60)  # first cycle after the restart
    assert world.only_signal().status == "pending_entry" and world.mailer.exits == []

    frame = market()
    window = frame[(frame.index >= DECISION) & (frame.index < DECISION + 31 * 60)]
    repo_market.upsert_bars_1m(world.db, "NVDA", window, "rest")  # the backfill
    await run(engine, world.clock, DECISION + 31 * 60, DECISION + 31 * 60)
    signal = world.only_signal()
    assert signal.status == "active" and signal.entry_ts == DECISION + 60


async def test_after_an_overnight_downtime_an_active_signal_is_closed_on_the_real_bars(world: World) -> None:
    """RUN-1: the laptop slept from just after the entry until the next morning. At the wake-up the venue is
    closed and the time stop long overdue, but the session's bars are not stored yet: no market exit at the
    stale last price - the backfill delivers the bars and the time stop fills on its own bar."""
    entry_ts = DECISION + 60
    world.seed_nvda(until=entry_ts + 120)
    stored = active_breakout(world, entry_ts, target_pct=0.5)  # expires_ts = entry + 180 min, before the close
    engine = world.engine()
    wake = utc(21, 10)  # Saturday morning
    await run(engine, world.clock, wake, wake + 5 * 60)
    assert world.only_signal().status == "active" and world.mailer.exits == []

    frame = market()
    session = frame[(frame.index >= entry_ts) & (frame.index < utc(20, 21))]
    repo_market.upsert_bars_1m(world.db, "NVDA", session, "rest")  # the backfill
    await run(engine, world.clock, wake + 6 * 60, wake + 6 * 60)
    signal = world.only_signal()
    expires = stored.expires_ts or 0
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "time_stop", expires)
    assert signal.exit_price == pytest.approx(
        DEFAULT_COSTS.fill_price(NVDA, float(frame.loc[expires, "open"]), buying=False)
    )
    assert world.mailer.exits == [(stored.id, "expired")]


async def test_after_a_laptop_sleep_the_running_engine_waits_for_the_hole_repair(world: World) -> None:
    """RUN-1: the process keeps running but the laptop sleeps for 25 minutes right after the issue. The first
    cycle after the wake-up must not cancel the signal before the hole repair has stored the entry bar."""
    world.seed_nvda(until=DECISION)
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.05,
        entry_deadline_ts=DECISION + 3 + ENGINE_PARAMS.entry_grace_min * 60, mailed_ts=DECISION + 5,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, DECISION - 15 * 60, DECISION)  # running long enough to be past any start-up
    await run(engine, world.clock, DECISION + 25 * 60, DECISION + 25 * 60)  # first cycle after the wake-up
    assert world.only_signal().status == "pending_entry"

    frame = market()
    window = frame[(frame.index >= DECISION) & (frame.index < DECISION + 26 * 60)]
    repo_market.upsert_bars_1m(world.db, "NVDA", window, "rest")  # the hole repair
    await run(engine, world.clock, DECISION + 26 * 60, DECISION + 26 * 60)
    signal = world.only_signal()
    assert signal.status == "active" and signal.entry_ts == DECISION + 60


async def test_without_any_bar_the_wall_clock_closes_signals_once_the_downtime_had_time_to_settle(
    world: World,
) -> None:
    """The restart guard only defers the wall-clock fallback: when no bar ever arrives, a pending signal is
    cancelled ('kein Einstieg') once the settle time after the first cycle has passed."""
    world.seed_nvda(until=DECISION)
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.05,
        entry_deadline_ts=DECISION + 3 + ENGINE_PARAMS.entry_grace_min * 60,
    )  # fmt: skip
    engine = world.engine()
    first = DECISION + 30 * 60
    await run(engine, world.clock, first, first + engine_module.SETTLE_S - 60)
    assert world.only_signal().status == "pending_entry"
    await run(engine, world.clock, first + engine_module.SETTLE_S, first + engine_module.SETTLE_S)
    signal = world.only_signal()
    assert (signal.status, signal.status_reason) == ("cancelled", "kein_einstieg")


async def test_a_mailed_recommendation_that_is_cancelled_gets_the_cancellation_mail(world: World) -> None:
    """INT-7: whoever acted on the entry mail must learn that the recommendation was dropped ('ENTFÄLLT')."""
    world.seed_nvda()
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored = stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open * 0.98, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.02,
        entry_deadline_ts=DECISION + 603, mailed_ts=DECISION + 5,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, DECISION + 60, DECISION + 120)
    signal = world.only_signal()
    assert (signal.status, signal.status_reason) == ("cancelled", "kurs_entlaufen")
    assert world.mailer.exits == [(stored.id, "cancelled")]
    assert world.activity()[-1].startswith("Kein Einstieg: LONG NVDA (Ausbruch)")


async def test_a_cancelled_signal_that_was_never_mailed_gets_no_exit_mail(world: World) -> None:
    world.seed_nvda()
    entry_open = float(market().loc[DECISION + 60, "open"])
    stored_signal(
        world.db, created_ts=DECISION, data_ts=DECISION, issued_ts=DECISION + 3, status="pending_entry",
        actionable=True, ref_price=entry_open * 0.98, stop_dist=entry_open * 0.01, target_dist=entry_open * 0.02,
        entry_deadline_ts=DECISION + 603,
    )  # fmt: skip
    engine = world.engine()
    await run(engine, world.clock, DECISION + 60, DECISION + 120)
    assert world.only_signal().status == "cancelled" and world.mailer.exits == []


async def test_time_stop_winners_do_not_count_as_full_targets_in_the_expected_value(world: World) -> None:
    """MONEY-2: half of this version's earlier breakouts won - but only +0.1 R each, closed by the time stop.
    The binary formula would value every win at +2 R and issue the candidate; its real expectancy is negative."""
    world.seed_nvda()
    for k in range(200):
        exit_ts = DECISION - 7200 - k * 60
        won = k % 2 == 0
        stored_signal(
            world.db, score=0.86, status="expired" if won else "stopped", outcome_r=0.1 if won else -1.0,
            exit_ts=exit_ts, issued_ts=exit_ts - 600, created_ts=exit_ts - 660, data_ts=exit_ts - 720,
        )  # fmt: skip
    engine = world.engine()
    await detect(world, engine)
    signal = world.signals()[-1]
    assert signal.status == "pending_entry" and signal.setup == "breakout"
    p, rr = signal.confidence, signal.rr
    assert p * rr - (1 - p) - 0.1 > 0.0  # what the binary formula would have said, even with 0.1 R of costs
    assert signal.expected_r < 0.0 and signal.shadow_reason == "erwartungswert" and not signal.actionable
    assert world.broker.executed == [] and world.mailer.entries == []


async def test_review_context_carries_facts_news_market_note_and_chart(world: World) -> None:
    world.seed_nvda()
    item = repo_news.insert_news(
        world.db,
        [NewsItem(uid="n1", title_key="nvidia beats", source="test", url="https://example.invalid/n1",
                  title="Nvidia beats estimates", fetched_ts=DECISION - 3700)],
    )[0]  # fmt: skip
    assert item.id is not None
    repo_news.save_analysis(
        world.db,
        NewsAnalysis(news_id=item.id, analyzed_ts=DECISION - 3600, model="fake", relevance=4, sentiment=0.8,
                     category="product", horizon="days", summary_de="Nvidia übertrifft die Erwartungen.",
                     symbols={"NVDA": 0.8}),
    )  # fmt: skip
    engine = world.engine()
    await detect(world, engine)

    signal = world.only_signal()
    assert signal.news_ids == [item.id] and signal.components["news"] > 0.5
    [(ctx, timeout_s)] = world.llm.reviews
    assert timeout_s == 45.0
    assert ctx.instrument.symbol == "NVDA" and ctx.candidate.score == signal.score and ctx.chart_png == PNG
    assert ctx.headlines == [("Nvidia übertrifft die Erwartungen.", 0.8)]
    assert ctx.market_note.startswith("S&P 500: 15-Min-Trend aufwärts")
    facts = ctx.facts
    assert facts["Setup"] == "Ausbruch" and facts["Richtung"] == "Long (Kauf)" and facts["Währung"] == "USD"
    assert facts["Minuten bis Börsenschluss"] == 264 and facts["Datenalter Min"] == 0
    assert facts["Chance/Risiko"] == 2.0 and facts["Relatives Volumen"] > 1.5
    assert facts["Trend 15 Min"] == "aufwärts" and facts["Vergleichsindex"] == "S&P 500"
    assert all(isinstance(key, str) for key in facts)


# ------------------------------------------------------------------------------------------------
# Decision paths
# ------------------------------------------------------------------------------------------------
async def test_veto_gives_a_shadow_signal_that_is_tracked_without_order_or_mail(world: World) -> None:
    world.seed_nvda()
    world.llm.verdict = "veto"
    engine = world.engine()
    await detect(world, engine)

    signal = world.only_signal()
    assert not signal.actionable and signal.shadow_reason == "llm_veto" and signal.llm_verdict == "veto"
    assert signal.status == "pending_entry"
    assert world.broker.executed == [] and world.mailer.entries == []
    assert world.activity() == ["Kandidat verworfen: LONG NVDA (Ausbruch) – Veto der KI-Prüfung"]

    await run_until(engine, world.clock, DECISION + 60, lambda: world.only_signal().status == "target", utc(20, 18))
    assert world.only_signal().outcome_r is not None
    assert world.mailer.exits == [] and len(world.activity()) == 1  # shadow signals stay out of the activity log


@pytest.mark.parametrize(
    ("min_actionable", "actionable", "reason"), [(0.75, True, ""), (0.80, False, "llm_nicht_verfuegbar")]
)
async def test_unavailable_review_needs_the_extra_margin(
    world: World, min_actionable: float, actionable: bool, reason: str
) -> None:
    world.seed_nvda()
    world.llm.verdict = "unavailable"
    world.prefs(min_score_actionable=min_actionable)
    engine = world.engine()
    await detect(world, engine)

    signal = world.only_signal()  # score ~0.86: >= 0.75 + 0.08 but < 0.80 + 0.08
    assert signal.llm_verdict == "unavailable" and signal.actionable is actionable and signal.shadow_reason == reason
    assert signal.rationale_de.startswith("Ausbruch (Long) bei NVDA: Am stärksten sprechen")
    assert len(world.broker.executed) == int(actionable)


@pytest.mark.parametrize(("min_actionable", "actionable"), [(0.75, True), (0.80, False)])
async def test_an_llm_that_ignores_its_timeout_counts_as_unavailable(
    world: World, monkeypatch: pytest.MonkeyPatch, min_actionable: float, actionable: bool
) -> None:
    world.seed_nvda()
    world.llm.review_gate = asyncio.Event()  # never answers
    world.prefs(min_score_actionable=min_actionable)
    monkeypatch.setattr(engine_module, "REVIEW_GRACE_S", 0.05 - 45)  # the engine gives up after 0.05 s
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()
    assert signal.llm_verdict == "unavailable" and signal.actionable is actionable
    assert signal.shadow_reason == ("" if actionable else "llm_nicht_verfuegbar")


JUNK_FIELDS: dict[str, Any] = {"conviction": math.nan, "rationale_de": None, "risks_de": None, "latency_ms": -5}


@pytest.mark.parametrize(
    ("answer", "verdict"),
    [
        (TradeReview(verdict="Confirm", rationale_de="KI: sieht gut aus."), "unavailable"),  # type: ignore[arg-type]
        (None, "unavailable"),
        (TradeReview(verdict="confirm", **JUNK_FIELDS), "confirm"),
    ],
)
async def test_a_malformed_llm_answer_is_stored_safely(world: World, answer: Any, verdict: str) -> None:
    world.seed_nvda()

    async def sloppy(ctx: TradeReviewContext, *, timeout_s: float) -> Any:
        return answer

    world.llm.review_trade = sloppy  # type: ignore[method-assign]
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()  # finalised at once, not left in 'review' until the crash recovery
    assert signal.status == "pending_entry" and signal.llm_verdict == verdict
    assert signal.llm_conviction is None and signal.llm_latency_ms is None and signal.risks_de == ""
    assert signal.rationale_de.startswith("Ausbruch (Long) bei NVDA: Am stärksten")
    assert signal.actionable  # 'confirm', or unavailable with 0.86 >= 0.60 + 0.08


async def test_a_hanging_chart_renderer_does_not_block_review_or_mail(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.seed_nvda()
    never = asyncio.Event()

    async def hanging(symbol: str, timeframe: str = "15m", **kwargs: Any) -> bytes | None:
        await never.wait()
        return PNG

    world.charts.render = hanging  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "CHART_TIMEOUT_S", 0.05)
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()
    [(ctx, _)] = world.llm.reviews
    assert ctx.chart_png is None and signal.actionable and world.mailer.entries == [(signal.id, None)]


@pytest.mark.parametrize(
    ("patch", "actionable", "reason"),
    [
        ({"min_score_review": 0.95}, True, ""),  # 0.86 >= 0.60 + 0.08
        ({"min_score_review": 0.95, "min_score_actionable": 0.85}, False, "unter_schwelle"),
        ({"min_score_review": 0.95, "min_score_actionable": 0.95}, False, "unter_schwelle"),
    ],
)
async def test_below_the_review_threshold_the_llm_is_skipped(
    world: World, patch: dict[str, float], actionable: bool, reason: str
) -> None:
    world.seed_nvda()
    world.prefs(**patch)
    engine = world.engine()
    await detect(world, engine)

    signal = world.only_signal()
    assert signal.llm_verdict is None and world.llm.reviews == []
    assert signal.actionable is actionable and signal.shadow_reason == reason and signal.status == "pending_entry"
    if actionable:  # no review chart: the mail chart is rendered at issue time
        assert world.mailer.entries == [(signal.id, PNG)]
        assert world.charts.calls == [("NVDA", "15m", DECISION + 3, signal.id)]
    else:
        assert world.mailer.entries == [] and "unter" in world.activity()[-1]


async def test_disabled_llm_or_review_finalises_without_asking(world: World) -> None:
    world.seed_nvda()
    engine = world.engine(enable_llm=False)
    await detect(world, engine)
    assert world.only_signal().llm_verdict == "unavailable" and world.llm.reviews == []


async def test_review_switched_off_by_the_user_counts_as_skipped(world: World) -> None:
    world.seed_nvda()
    world.prefs(llm_review=False)
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()
    assert signal.llm_verdict is None and signal.actionable and world.llm.reviews == []


async def test_expected_value_after_costs_must_be_positive(world: World) -> None:
    world.seed_nvda()
    expensive = replace(DEFAULT_COSTS, slippage_bps_us_equity=60.0)
    engine = build_engine(
        settings=world.settings, db=world.db, clock=world.clock, bus=world.bus, calendar=world.calendar,
        instruments=world.book, runtime_settings=world.store, costs=expensive, llm=world.llm, charts=world.charts,
        mailer=world.mailer, broker=world.broker,
    )  # fmt: skip
    assert isinstance(engine, SignalEngine)
    world.engines.append(engine)
    await detect(world, engine)
    signal = world.only_signal()
    assert signal.expected_r < 0 and signal.shadow_reason == "erwartungswert" and not signal.actionable


# ------------------------------------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "patch",
    [
        {"disabled_setups": ["breakout"]},
        {"allow_long": False},
        {"signal_regions": ["EU", "ASIA"]},
        {"signals_enabled": False},
    ],
)
async def test_filtered_hits_are_dropped_silently(world: World, patch: dict[str, Any]) -> None:
    world.seed_nvda()
    world.prefs(**patch)
    engine = world.engine()
    await run(engine, world.clock, DECISION - 120, DECISION + 600)
    assert world.signals() == [] and world.llm.reviews == [] and world.activity() == []


async def test_one_open_signal_per_symbol(world: World) -> None:
    world.seed_nvda()
    other = stored_signal(world.db, direction="SHORT", status="review", created_ts=DECISION - 60)
    engine = world.engine()
    await detect(world, engine)
    assert [signal.id for signal in world.signals()] == [other.id]


@pytest.mark.parametrize(
    ("direction", "closed_min_before", "cooldown_min", "blocked"),
    [
        ("LONG", 10, 30, True),
        ("LONG", 40, 30, False),
        ("SHORT", 10, 30, False),
        ("LONG", 10, 0, False),
    ],
)
async def test_cooldown_per_symbol_and_direction(
    world: World, direction: str, closed_min_before: int, cooldown_min: int, blocked: bool
) -> None:
    world.seed_nvda()
    exit_ts = DECISION - closed_min_before * 60
    stored_signal(
        world.db, direction=direction, status="stopped", issued_ts=exit_ts - 900, exit_ts=exit_ts, outcome_r=-1.0
    )
    world.prefs(signal_cooldown_min=cooldown_min)
    engine = world.engine()
    await detect(world, engine)
    assert len(world.signals()) == (1 if blocked else 2)


@pytest.mark.parametrize(("seconds_before_decision", "blocked"), [(1800, True), (1801, False)])
async def test_the_cooldown_includes_its_last_second(world: World, seconds_before_decision: int, blocked: bool) -> None:
    world.seed_nvda()
    exit_ts = DECISION + 3 - seconds_before_decision  # the decision is taken at DECISION + 3
    stored_signal(world.db, status="stopped", issued_ts=exit_ts - 900, exit_ts=exit_ts, outcome_r=-1.0)
    engine = world.engine()
    await detect(world, engine)
    assert len(world.signals()) == (1 if blocked else 2)


@pytest.mark.parametrize(("seconds_after_decision", "evaluated"), [(600, True), (601, False)])
async def test_staleness_boundary(world: World, seconds_after_decision: int, evaluated: bool) -> None:
    world.seed_nvda(until=DECISION)  # the newest bucket ends at DECISION
    engine = world.engine()
    await run(engine, world.clock, DECISION - 120, DECISION - 120)
    world.clock.set(DECISION + seconds_after_decision)  # 10 minutes (+1 s) after the bucket end
    await engine.on_minute(world.clock.now(), [])
    await engine.drain()
    signals = world.signals()
    assert len(signals) == int(evaluated)
    if evaluated:
        assert signals[0].actionable  # the newest bar is not stale at issue time either


class ClosedNasdaq(FakeCalendar):
    def is_open(self, exchange: str, at: datetime) -> bool:
        return False if exchange == "XNAS" else super().is_open(exchange, at)


async def test_instruments_of_a_closed_venue_are_not_scanned(world: World) -> None:
    world.seed_nvda()
    engine = world.engine(calendar=ClosedNasdaq())
    await detect(world, engine)
    assert world.signals() == []


async def test_venue_closing_during_the_review_gives_a_shadow_signal(world: World) -> None:
    world.seed_nvda()
    world.llm.on_review = lambda: world.clock.set(utc(20, 21, 0, 30))
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()
    assert signal.llm_verdict == "confirm" and signal.shadow_reason == "markt_geschlossen"
    assert signal.issued_ts == utc(20, 21, 0, 30) and world.broker.executed == []


async def test_a_stale_bucket_is_not_evaluated(world: World) -> None:
    world.seed_nvda(until=DECISION)  # the feed stops with the breakout bucket
    engine = world.engine()
    await run(engine, world.clock, DECISION - 120, DECISION - 120)
    await run(engine, world.clock, DECISION + 11 * 60, DECISION + 12 * 60)
    assert world.signals() == []


async def test_data_gone_stale_during_the_review_gives_a_shadow_signal(world: World) -> None:
    world.seed_nvda(until=DECISION)
    world.llm.on_review = lambda: world.clock.advance(15 * 60)
    engine = world.engine()
    await detect(world, engine)
    signal = world.only_signal()
    assert signal.shadow_reason == "daten_veraltet" and not signal.actionable


# ------------------------------------------------------------------------------------------------
# Restart and recovery
# ------------------------------------------------------------------------------------------------
async def test_a_restarted_engine_does_not_evaluate_the_bucket_it_finds(world: World) -> None:
    world.seed_nvda()
    engine = world.engine()
    await run(engine, world.clock, DECISION, DECISION + 600)  # first cycle right after the breakout bucket
    assert world.signals() == []


async def test_a_restarted_engine_continues_tracking_and_sends_the_exit_mail(world: World) -> None:
    world.seed_nvda()
    first = world.engine()
    await detect(world, first)
    await first.stop()
    issued = world.only_signal()
    assert issued.status == "pending_entry" and issued.mailed_ts == DECISION + 3

    second = world.engine()
    await run_until(second, world.clock, DECISION + 60, lambda: world.only_signal().status == "target", utc(20, 18))
    assert world.only_signal().entry_ts == DECISION + 60
    assert world.mailer.exits == [(issued.id, "target")]
    assert world.broker.executed == [(issued.id, "pending_entry", True)]


async def test_signal_left_in_review_by_a_crash_becomes_a_shadow_signal(world: World) -> None:
    world.seed_nvda()
    world.llm.review_gate = asyncio.Event()  # the review never finishes
    crashed = world.engine()
    await detect(world, crashed, drain=False)
    await until(lambda: len(world.llm.reviews) == 1)
    await crashed.stop()
    assert world.only_signal().status == "review"

    restarted = world.engine()
    await run(restarted, world.clock, DECISION + 9 * 60, DECISION + 9 * 60)
    assert world.only_signal().status == "review"  # not yet ten minutes
    await run(restarted, world.clock, DECISION + 11 * 60, DECISION + 11 * 60)
    signal = world.only_signal()
    assert signal.status == "pending_entry" and signal.issued_ts == DECISION + 11 * 60 + 3
    assert signal.llm_verdict == "unavailable" and signal.shadow_reason == "llm_nicht_verfuegbar"
    assert not signal.actionable and world.broker.executed == []
    assert world.activity() == ["Kandidat verworfen: LONG NVDA (Ausbruch) – KI-Prüfung nicht abgeschlossen"]


async def test_a_review_hanging_in_this_process_is_recovered_once(world: World) -> None:
    world.seed_nvda()
    gate = world.llm.review_gate = asyncio.Event()
    engine = world.engine()
    await run(engine, world.clock, DECISION - 120, DECISION + 11 * 60, drain=False)
    signal = world.only_signal()
    assert signal.status != "review" and signal.llm_verdict == "unavailable" and not signal.actionable
    gate.set()
    await engine.drain()
    again = world.only_signal()
    assert again.llm_verdict == "unavailable" and again.issued_ts == signal.issued_ts
    assert len(world.activity()) == 1


async def test_stopping_during_the_finalisation_still_places_the_order(world: World) -> None:
    """An issued recommendation must never exist without the AI portfolio's order, even on shutdown."""
    world.seed_nvda()
    inside, release = threading.Event(), threading.Event()
    publish = world.bus.publish

    def blocking_publish(topic: str, data: dict[str, Any] | None = None) -> None:
        if topic == TOPIC_SIGNAL and (data or {}).get("status") == "pending_entry" and not release.is_set():
            inside.set()  # the worker thread is in the middle of the finalisation
            release.wait(5)
        publish(topic, data)

    world.bus.publish = blocking_publish  # type: ignore[method-assign]
    engine = world.engine()
    await detect(world, engine, drain=False)
    await until(inside.is_set)
    stopping = asyncio.create_task(engine.stop())
    await until(lambda: not engine._tasks)  # the review task has been cancelled
    release.set()
    await stopping
    signal = world.only_signal()
    assert signal.status == "pending_entry" and signal.actionable
    assert world.broker.executed == [(signal.id, "pending_entry", True)]


def slow_entry_mails(world: World) -> asyncio.Event:
    """Make the mailer wait (SMTP pacing / transaction) until the returned event is set."""
    gate = asyncio.Event()
    send = world.mailer.send_signal

    async def slow_send(signal: Signal, instrument: Instrument, chart_png: bytes | None = None) -> bool:
        await gate.wait()
        return await send(signal, instrument, chart_png)

    world.mailer.send_signal = slow_send  # type: ignore[method-assign]
    return gate


def mail_activity(world: World) -> list[tuple[str, str, int | None]]:
    return [(e.level, e.message, e.ref_id) for e in reversed(repo_ops.list_activity(world.db, category="mail"))]


async def test_stop_lets_an_entry_mail_in_flight_finish(world: World) -> None:
    """RUN-2: Ctrl+C while the entry mail is in the SMTP transaction must not lose the recommendation."""
    world.seed_nvda()
    gate = slow_entry_mails(world)
    engine = world.engine()
    await detect(world, engine, drain=False)
    await until(lambda: any(task.get_name().startswith("mail-signal-") for task in engine._tasks))
    asyncio.get_running_loop().call_later(0.05, gate.set)  # the transaction ends during the shutdown
    await engine.stop()
    signal = world.only_signal()
    assert world.mailer.entries == [(signal.id, PNG)] and signal.mailed_ts is not None
    assert mail_activity(world) == []


async def test_stop_records_the_mails_it_had_to_cancel(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """RUN-2: a mail that does not finish within the grace period is cancelled and noted in the activity log,
    so the user knows the recommendation never reached the inbox."""
    world.seed_nvda()
    slow_entry_mails(world)  # never released
    monkeypatch.setattr(engine_module, "MAIL_STOP_GRACE_S", 0.05)
    engine = world.engine()
    await detect(world, engine, drain=False)
    await until(lambda: any(task.get_name().startswith("mail-signal-") for task in engine._tasks))
    started = time.monotonic()
    await engine.stop()
    assert time.monotonic() - started < 2.0
    signal = world.only_signal()
    assert world.mailer.entries == [] and signal.mailed_ts is None
    message = (
        "Einstiegs-Mail beim Beenden der Plattform abgebrochen – möglicherweise nicht verschickt: "
        f"LONG NVDA (Ausbruch), Signal #{signal.id}"
    )
    assert mail_activity(world) == [("warning", message, signal.id)]


async def test_stop_records_an_exit_mail_it_had_to_cancel(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    world.seed_nvda()
    stored = active_breakout(world, DECISION + 60, stop_pct=0.001, target_pct=0.001)  # closes within minutes
    never = asyncio.Event()

    async def hanging_exit(signal: Signal, instrument: Instrument) -> bool:
        await never.wait()
        return True

    world.mailer.send_exit = hanging_exit  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "MAIL_STOP_GRACE_S", 0.05)
    engine = world.engine()
    await run(engine, world.clock, DECISION + 120, DECISION + 900, drain=False)
    assert world.only_signal().status in ("stopped", "target")
    await engine.stop()
    [(level, message, ref_id)] = mail_activity(world)
    assert level == "warning" and ref_id == stored.id
    assert message.startswith("Ausstiegs-Mail beim Beenden der Plattform abgebrochen – möglicherweise nicht")


@pytest.mark.parametrize(
    ("volumes", "unreviewed"),
    [
        ((2.5, 3.0, 2.0, 1.6), "DDD"),  # the newcomer is the weakest: finalised at once
        ((1.6, 2.5, 3.0, 2.0), "AAA"),  # a waiting candidate is the weakest: it is evicted
    ],
)
async def test_review_backlog_is_capped_at_three(world: World, volumes: tuple[float, ...], unreviewed: str) -> None:
    for k, (symbol, volume) in enumerate(zip(("AAA", "BBB", "CCC", "DDD"), volumes)):
        world.seed(twin(symbol, 20 + k), market(volume_factor=volume))
    gate = world.llm.review_gate = asyncio.Event()
    engine = world.engine()
    await detect(world, engine, drain=False)
    await until(lambda: len(world.llm.reviews) == 3)

    by_symbol = {signal.symbol: signal for signal in world.signals()}
    assert len(by_symbol) == 4
    assert sorted(by_symbol, key=lambda symbol: by_symbol[symbol].score)[0] == unreviewed
    assert by_symbol[unreviewed].status == "pending_entry" and by_symbol[unreviewed].llm_verdict == "unavailable"
    reviewed = {ctx.instrument.symbol for ctx, _ in world.llm.reviews}
    assert reviewed == set(by_symbol) - {unreviewed}
    assert all(by_symbol[symbol].status == "review" for symbol in reviewed)

    gate.set()
    await engine.drain()
    final = {signal.symbol: signal for signal in world.signals()}
    assert {symbol: final[symbol].llm_verdict for symbol in reviewed} == dict.fromkeys(reviewed, "confirm")
    assert final[unreviewed].llm_verdict == "unavailable"


# ------------------------------------------------------------------------------------------------
# Chart analyses
# ------------------------------------------------------------------------------------------------
async def test_chart_scan_picks_the_oldest_instrument_and_runs_one_at_a_time(world: World) -> None:
    world.seed_nvda()
    world.seed(twin("AAA", 2), market())
    world.prefs(chart_scan_enabled=True, signals_enabled=False, chart_scan_interval_min=30)
    gate = world.llm.chart_gate = asyncio.Event()
    engine = world.engine()

    await run(engine, world.clock, utc(20, 16, 0), utc(20, 16, 0), drain=False)
    await until(lambda: len(world.llm.charts) == 1)
    await run(engine, world.clock, utc(20, 16, 1), utc(20, 16, 1), drain=False)  # single flight
    assert [symbol for symbol, _, _ in world.llm.charts] == ["NVDA"]
    symbol, timeframe, facts = world.llm.charts[0]
    assert timeframe == "15m" and facts["Kurs"] > 0 and "ATR (15 Min)" in facts

    gate.set()
    await engine.drain()
    stored = repo_signals.latest_chart_analysis(world.db, "NVDA")
    assert stored is not None and stored.ts == utc(20, 16, 1, 3)
    assert [data["symbol"] for data in world.bus.topics(TOPIC_CHART_ANALYSIS)] == ["NVDA"]

    await run(engine, world.clock, utc(20, 16, 2), utc(20, 16, 3))
    assert [symbol for symbol, _, _ in world.llm.charts] == ["NVDA", "AAA"]  # then nothing is due
    await run(engine, world.clock, utc(20, 16, 31), utc(20, 16, 32))
    assert [symbol for symbol, _, _ in world.llm.charts] == ["NVDA", "AAA", "NVDA", "AAA"]
    await run(engine, world.clock, utc(20, 21, 30), utc(20, 21, 31))  # venue closed
    assert len(world.llm.charts) == 4


async def test_a_chart_analysis_that_never_returns_does_not_block_the_scan(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.seed_nvda()
    world.seed(twin("AAA", 2), market())
    world.prefs(chart_scan_enabled=True, signals_enabled=False)
    world.llm.chart_gate = asyncio.Event()  # the model never answers
    monkeypatch.setattr(engine_module, "CHART_ANALYSIS_TIMEOUT_S", 0.05)
    engine = world.engine()
    await run(engine, world.clock, utc(20, 16, 0), utc(20, 16, 1))
    assert [symbol for symbol, _, _ in world.llm.charts] == ["NVDA", "AAA"]
    assert repo_signals.latest_chart_analysis(world.db, "NVDA") is None


async def test_chart_scan_waits_for_the_interval_after_a_failed_render(world: World) -> None:
    world.seed_nvda()
    world.prefs(chart_scan_enabled=True, signals_enabled=False)
    world.charts.result = None
    engine = world.engine()
    await run(engine, world.clock, utc(20, 16, 0), utc(20, 16, 10))
    assert len(world.charts.calls) == 1 and world.llm.charts == []


async def test_request_chart_analysis(world: World) -> None:
    world.seed_nvda()
    few = twin("FEW", 3)
    world.seed(few, market().iloc[-400:-390], daily=False)
    world.clock.set(DECISION + 3)
    disabled = world.engine(enable_llm=False)
    assert not await disabled.request_chart_analysis("NVDA")
    engine = world.engine()
    assert not await engine.request_chart_analysis("NOPE")
    assert not await engine.request_chart_analysis("NVDA", "2h")
    assert not await engine.request_chart_analysis("FEW")

    gate = world.llm.chart_gate = asyncio.Event()
    assert await engine.request_chart_analysis("NVDA", "15m")
    await until(lambda: len(world.llm.charts) == 1)
    assert await engine.request_chart_analysis("NVDA", "15m")  # already running: nothing new queued
    assert not await engine.request_chart_analysis("NVDA", "1d")  # another timeframe is running: not queued
    gate.set()
    await engine.drain()
    assert len(world.llm.charts) == 1
    assert await engine.request_chart_analysis("NVDA", "1d")
    await engine.drain()
    assert [row.timeframe for row in repo_signals.list_chart_analyses(world.db, "NVDA")] == ["1d", "15m"]
    assert llm_notes(world) == []  # results arrived: no failure notes


def llm_notes(world: World) -> list[tuple[str, str]]:
    return [(e.level, e.message) for e in reversed(repo_ops.list_activity(world.db, category="llm"))]


async def test_a_requested_chart_analysis_without_result_ends_with_a_german_note(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INT-9: the GUI waits for the outcome of 'Neu analysieren'; it must never stay without an answer."""
    world.seed_nvda()
    world.clock.set(DECISION + 3)
    engine = world.engine()

    world.charts.result = None  # the renderer gives no chart
    assert await engine.request_chart_analysis("NVDA")
    await engine.drain()
    assert llm_notes(world) == [
        ("warning", "Chartanalyse NVDA (15m) ohne Ergebnis: es konnte kein Chart erstellt werden.")
    ]

    world.charts.result = PNG

    async def no_answer(instrument: Instrument, timeframe: str, png: bytes, facts: Mapping[str, Any]) -> None:
        return None  # stale in the LLM queue or an invalid answer

    world.llm.analyze_chart = no_answer  # type: ignore[method-assign,assignment]
    assert await engine.request_chart_analysis("NVDA", "1h")
    await engine.drain()
    assert llm_notes(world)[-1] == (
        "warning", "Chartanalyse NVDA (1h) ohne Ergebnis: die KI hat keine verwertbare Antwort geliefert."
    )

    never = asyncio.Event()

    async def silent(instrument: Instrument, timeframe: str, png: bytes, facts: Mapping[str, Any]) -> None:
        await never.wait()

    world.llm.analyze_chart = silent  # type: ignore[method-assign,assignment]
    monkeypatch.setattr(engine_module, "CHART_ANALYSIS_TIMEOUT_S", 0.05)
    assert await engine.request_chart_analysis("NVDA", "5m")
    await engine.drain()
    assert llm_notes(world)[-1] == (
        "warning", "Chartanalyse NVDA (5m) ohne Ergebnis: die KI hat nicht rechtzeitig geantwortet."
    )
    assert repo_signals.list_chart_analyses(world.db, "NVDA") == []


async def test_a_failing_routine_scan_is_only_logged_unless_the_gui_joined_it(world: World) -> None:
    world.seed_nvda()
    world.prefs(chart_scan_enabled=True, signals_enabled=False, chart_scan_interval_min=30)
    gate = asyncio.Event()

    async def late_none(instrument: Instrument, timeframe: str, png: bytes, facts: Mapping[str, Any]) -> None:
        await gate.wait()

    world.llm.analyze_chart = late_none  # type: ignore[method-assign,assignment]
    engine = world.engine()
    await run(engine, world.clock, utc(20, 16, 0), utc(20, 16, 0), drain=False)
    gate.set()
    await engine.drain()
    assert llm_notes(world) == []  # routine scan: no note for nobody

    gate.clear()
    await run(engine, world.clock, utc(20, 16, 31), utc(20, 16, 31), drain=False)  # the next routine scan
    await until(lambda: "NVDA" in engine._chart_inflight)
    assert await engine.request_chart_analysis("NVDA", "15m")  # the user asks for the running one
    gate.set()
    await engine.drain()
    assert llm_notes(world) == [
        ("warning", "Chartanalyse NVDA (15m) ohne Ergebnis: die KI hat keine verwertbare Antwort geliefert.")
    ]


async def test_a_chart_request_that_cannot_be_served_returns_false(world: World) -> None:
    world.seed_nvda()
    world.clock.set(DECISION + 3)
    engine = world.engine()
    await engine.stop()  # worker gone: nothing would be produced
    assert not await engine.request_chart_analysis("NVDA")
    assert llm_notes(world) == []


# ------------------------------------------------------------------------------------------------
# Analysis and statistics
# ------------------------------------------------------------------------------------------------
async def test_analysis_payload(world: World) -> None:
    world.seed_nvda()
    world.seed(twin("FEW", 3), market().iloc[-400:-390], daily=False)
    engine = world.engine()
    world.clock.set(DECISION + 3)

    out = engine.analysis("NVDA")  # computed on demand: no cycle has run
    assert out.available and out.note_de == "" and out.as_of == DECISION + 3 and out.data_ts == DECISION
    assert [r.key for r in out.readings] == [
        "trend_5m", "trend_15m", "trend_daily", "momentum", "trend_strength", "volume", "vwap", "volatility",
        "benchmark", "news",
    ]  # fmt: skip
    readings_by_key = {reading.key: reading for reading in out.readings}
    assert readings_by_key["trend_15m"].stance == "bullish"
    assert readings_by_key["news"].value_de == "keine aktuellen Meldungen"
    assert readings_by_key["benchmark"].label_de == "Markt (S&P 500)"
    prices = [level.price for level in out.levels]
    assert prices == sorted(prices, reverse=True)
    assert {"day_high", "vwap", "prev_close"} <= {level.kind for level in out.levels}
    assert out.features["price"] == pytest.approx(float(market().loc[DECISION - 60, "close"]))
    assert all(np.isfinite(value) for value in out.features.values() if value is not None)
    assert out.news_score == 0.0 and out.chart_analysis is None

    world.clock.set(DECISION + 5 * 60 + 3)
    assert engine.analysis("NVDA").as_of == DECISION + 3  # cached: 5 trading minutes old
    world.clock.set(DECISION + 20 * 60 + 3)
    assert engine.analysis("NVDA").as_of == DECISION + 20 * 60 + 3  # recomputed

    repo_signals.insert_chart_analysis(
        world.db, ChartAnalysis("NVDA", DECISION, "15m", "fake", "bullish", 4, "Ausbruch bestätigt.")
    )
    assert engine.analysis("NVDA").chart_analysis is not None

    unknown = engine.analysis("NOPE")
    assert not unknown.available and unknown.note_de == NOTE_UNKNOWN_SYMBOL
    few = engine.analysis("FEW")
    assert not few.available and few.note_de == NOTE_TOO_LITTLE_DATA

    world.clock.set(utc(21, 12, 0))  # Saturday
    weekend = engine.analysis("NVDA")
    assert weekend.available and weekend.note_de == NOTE_MARKET_CLOSED and weekend.data_ts == utc(20, 21, 0)


async def test_analysis_uses_the_row_cached_by_the_detection(world: World) -> None:
    world.seed_nvda()
    engine = world.engine()
    await detect(world, engine)
    world.clock.set(DECISION + 4 * 60)
    out = engine.analysis("NVDA")
    assert out.as_of == DECISION + 3 and out.data_ts == DECISION


async def test_analysis_from_api_threads_while_the_cycle_runs(world: World) -> None:
    world.seed_nvda()
    engine = world.engine()
    world.clock.set(DECISION - 120 + 3)
    results: list[AnalysisOut] = []

    async def ask() -> None:
        for _ in range(6):
            results.append(await asyncio.to_thread(engine.analysis, "NVDA"))

    await asyncio.gather(detect(world, engine), ask())
    assert len(results) == 6 and all(out.available for out in results)
    assert world.only_signal().setup == "breakout"


async def test_stats_payload(world: World) -> None:
    world.seed_nvda()
    rows = [
        # setup, direction, verdict, actionable, outcome, exit offset (minutes before DECISION)
        ("breakout", "LONG", "confirm", True, 2.0, 300),
        ("breakout", "LONG", "confirm", True, -1.0, 290),
        ("breakout", "SHORT", "reduce", True, 1.5, 280),
        ("trend_pullback", "LONG", "veto", False, -1.0, 270),
        ("trend_pullback", "SHORT", None, False, 2.0, 260),
        ("mean_reversion", "LONG", "unavailable", False, 0.4, 250),
        ("daily_trend", "LONG", "confirm", True, -0.2, 240),
        ("news_momentum", "SHORT", None, False, -1.0, 230),
        ("breakout", "LONG", "confirm", True, 2.0, 220),
        ("breakout", "LONG", "confirm", True, 2.0, 210),
        ("trend_pullback", "LONG", "confirm", True, -1.0, 200),
        ("breakout", "SHORT", None, False, 1.0, 190),
    ]
    for setup, direction, verdict, actionable, outcome, minutes in rows:
        exit_ts = DECISION - minutes * 60
        stored_signal(
            world.db, setup=setup, direction=direction, llm_verdict=verdict, actionable=actionable,
            outcome_r=outcome, status="target" if outcome > 0 else "stopped", exit_ts=exit_ts, issued_ts=exit_ts - 600,
            score=0.55 + 0.02 * (minutes % 7),
        )  # fmt: skip
    stored_signal(world.db, engine_version="e0-old", status="target", exit_ts=DECISION - 100, outcome_r=3.0)
    engine = world.engine()
    world.clock.set(DECISION + 3)

    stats = engine.stats()
    assert stats.engine_version == VERSION and stats.since_ts is None
    assert stats.actionable.n == 7 and stats.shadow.n == 5
    assert stats.actionable.total_r == pytest.approx(2.0 - 1.0 + 1.5 - 0.2 + 2.0 + 2.0 - 1.0)
    assert stats.note_de == "Erst 12 abgeschlossene Signale – statistisch noch nicht belastbar."
    setups = {group.key: group for group in stats.by_setup}
    assert setups["breakout"].label_de == "Ausbruch" and setups["breakout"].stats.n == 6
    assert stats.by_setup[0].key == "breakout"  # largest group first
    assert {group.label_de: group.stats.n for group in stats.by_region} == {"USA": 12}
    assert {group.label_de: group.stats.n for group in stats.by_direction} == {"Long": 8, "Short": 4}
    verdicts = {group.label_de: group.stats.n for group in stats.by_llm_verdict}
    assert verdicts == {"bestätigt": 6, "reduziert": 1, "Veto": 1, "ohne Prüfung": 4}
    assert len(stats.calibration) == len(ENGINE_PARAMS.calibration_bins) - 1
    assert sum(b.n for b in stats.calibration) == 12

    later = engine.stats(since_ts=DECISION - 220 * 60)  # inclusive: the last four exits
    assert later.actionable.n + later.shadow.n == 4 and later.since_ts == DECISION - 220 * 60
    assert later.note_de == "Erst 4 abgeschlossene Signale – statistisch noch nicht belastbar."


# ------------------------------------------------------------------------------------------------
# Robustness and speed
# ------------------------------------------------------------------------------------------------
async def test_on_minute_never_raises_when_every_dependency_fails(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.seed_nvda()
    world.llm.fail = world.charts.fail = world.mailer.fail = world.broker.fail = True
    engine = world.engine()
    with caplog.at_level(logging.ERROR, logger="trading_local.analysis.engine"):
        await detect(world, engine)
        await run(engine, world.clock, DECISION + 60, DECISION + 600)
    signal = world.only_signal()
    assert signal.llm_verdict == "unavailable" and signal.actionable  # 0.86 >= 0.60 + 0.08
    assert len(world.broker.executed) == 1 and len(world.mailer.entries) == 1
    assert signal.status == "active"  # tracking goes on
    assert any("broker failed" in record.getMessage() for record in caplog.records)


class BrokenCalendar(FakeCalendar):
    def is_open(self, exchange: str, at: datetime) -> bool:
        raise RuntimeError("calendar broken")

    def segments(self, exchange: str, start: datetime, end: datetime, *, extended: bool = False) -> list[Any]:
        raise RuntimeError("calendar broken")


class BrokenStore:
    def get(self) -> Any:
        raise RuntimeError("database locked")


async def test_on_minute_and_analysis_survive_a_broken_calendar_or_settings_store(world: World) -> None:
    world.seed_nvda()
    engine = world.engine(calendar=BrokenCalendar())
    await detect(world, engine)
    assert world.signals() == []
    assert not engine.analysis("NVDA").available
    assert engine.stats().note_de != ""
    without_settings = world.engine(runtime_settings=BrokenStore())
    await detect(world, without_settings)
    assert world.signals() == []


async def test_on_minute_never_raises_on_a_naive_time_or_after_stop(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.seed_nvda()
    engine = world.engine()
    with caplog.at_level(logging.ERROR, logger="trading_local.analysis.engine"):
        await engine.on_minute(datetime(2026, 11, 20, 16, 0), [])  # naive: rejected and logged
        await engine.stop()
        await run(engine, world.clock, DECISION - 120, DECISION)  # worker gone: logged, not raised
    assert world.signals() == [] and caplog.records
    assert not await engine.request_chart_analysis("NVDA")


async def test_a_hundred_symbol_cycle_stays_within_the_budget(
    world: World, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = market()
    frame = frame[(frame.index >= utc(5, 14, 30)) & (frame.index < DECISION + 300)]
    daily = daily_from(market())
    symbols = [twin(f"T{k:03d}", 100 + k) for k in range(100)]
    repo_market.upsert_instruments(world.db, symbols, keep_user_flags=False)
    for instrument in symbols:
        repo_market.upsert_bars_1m(world.db, instrument.symbol, frame, "rest")
        repo_market.upsert_bars_1d(world.db, instrument.symbol, daily)
    world.book.reload()
    world.prefs(disabled_setups=list(SETUPS))  # measure detection, not 100 finalisations
    engine = world.engine()
    await run(engine, world.clock, DECISION - 120, DECISION - 60)

    world.clock.set(DECISION + 3)
    interval = sys.getswitchinterval()
    sys.setswitchinterval(0.0005)  # as the runtime sets it at start-up
    try:
        with caplog.at_level(logging.DEBUG, logger="trading_local.analysis.engine"):
            started = time.perf_counter()
            await engine.on_minute(world.clock.now(), [])
            elapsed = time.perf_counter() - started
    finally:
        sys.setswitchinterval(interval)
    passes = [record.getMessage() for record in caplog.records if "detection pass" in record.getMessage()]
    assert passes and re.search(r"detection pass: 100 feature rows", passes[-1]), passes
    with capsys.disabled():
        print(f"\n100-symbol minute cycle: {elapsed:.2f} s")
    # Target ~3 s (measured on this machine: ~3.1 s steady state, ~3.9 s for the first pass of a new engine);
    # the bound leaves room for other test suites running on the same machine.
    assert elapsed < 6.0, elapsed
