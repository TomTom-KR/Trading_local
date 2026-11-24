"""Price trigger: baseline, measurement, the trigger rule and the MoveWatch service."""

import math
from collections.abc import Collection
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from trading_local.analysis import movers
from trading_local.analysis.movers import Baseline, Move, MoveWatch, compute_baseline, is_unusual, measure
from trading_local.clock import ManualClock
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.testing import FakeCalendar, make_bars, seed_bars, seed_instruments

DAY = 86_400


def jump(start_ts: int, price: float, *, bars: int = 5, step: float = 0.005, volume: float = 5000.0) -> pd.DataFrame:
    """``bars`` 1-minute bars from ``start_ts`` that each rise by ``step`` (relative)."""
    opens = price * (1.0 + step) ** np.arange(bars)
    closes = opens * (1.0 + step)
    ts = start_ts + 60 * np.arange(bars, dtype=np.int64)
    return pd.DataFrame(
        {"open": opens, "high": closes, "low": opens, "close": closes, "volume": np.full(bars, volume)},
        index=pd.Index(ts, name="ts"),
    )


def crypto_history(end_ts: int, *, days: int = 11, base_volume: float = 1000.0) -> pd.DataFrame:
    """24/7 random-walk bars (log-return sd 0.001 per minute) for ``days`` days that close at ``end_ts``."""
    minutes = days * 1440
    return make_bars(end_ts - minutes * 60, minutes, vol=0.001, base_volume=base_volume, seed=7)


# ------------------------------------------------------------------------------------------------
# Pure part
# ------------------------------------------------------------------------------------------------
def utc_ts(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def test_baseline_measures_the_spread_of_five_minute_returns_before_today() -> None:
    end_ts = utc_ts("2026-11-20 15:00")
    history = crypto_history(end_ts)  # from 2026-11-09 15:00
    today = date(2026, 11, 20)
    baseline = compute_baseline(history, timezone="UTC", before_day=today, computed_ts=end_ts)
    assert baseline is not None
    assert 0.0018 < baseline.sigma < 0.0027  # 0.001 * sqrt(5) = 0.00224
    assert baseline.buckets == 9 * 12 + 10 * 288  # 9 hours of the first day, ten full days, nothing of today
    assert 900 < baseline.volume_median < 1100 * 5 and set(baseline.volume_by_bucket) == set(range(288))
    assert compute_baseline(history.tail(100), timezone="UTC", before_day=today, computed_ts=end_ts) is None


def test_a_large_move_needs_large_volume_to_trigger() -> None:
    baseline = Baseline(0, sigma=0.002, volume_by_bucket={}, volume_median=5000.0, buckets=1000)
    start = utc_ts("2026-11-20 14:00")
    loud = measure("X", jump(start, 100.0, volume=4000.0), baseline, timezone="UTC")
    assert loud is not None and is_unusual(loud)
    assert loud.minutes == 5 and loud.ts == start + 300
    assert math.isclose(loud.change_pct, (1.005**5 - 1) * 100) and loud.z == pytest.approx(5 * math.log(1.005) / 0.002)
    assert loud.rel_volume == pytest.approx(20_000 / 5000)
    quiet = measure("X", jump(start, 100.0, volume=1000.0), baseline, timezone="UTC")
    assert quiet is not None and not is_unusual(quiet)  # volume only x1
    small = measure("X", jump(start, 100.0, step=0.0002, volume=4000.0), baseline, timezone="UTC")
    assert small is not None and not is_unusual(small)  # 0.5 standard deviations
    moderate = measure("X", jump(start, 100.0, step=0.0018, volume=4000.0), baseline, timezone="UTC")
    assert moderate is not None and 4 < moderate.z < 6 and not is_unusual(moderate)  # fat tails: 6 sigma needed
    falling = measure("X", jump(start, 100.0, step=-0.005, volume=4000.0), baseline, timezone="UTC")
    assert falling is not None and falling.z < 0 and is_unusual(falling)
    assert measure("X", jump(start, 100.0, bars=2), baseline, timezone="UTC") is None


def test_opening_volume_is_compared_with_the_usual_opening_volume() -> None:
    days = [date(2026, 11, d) for d in (9, 10, 11, 12, 13, 16, 17, 18, 19)]
    frames = []
    for i, day in enumerate(days):
        open_ts = utc_ts(day.isoformat() + " 14:30")
        session = make_bars(open_ts, 390, vol=0.0008, base_volume=1000.0, seed=i + 1)
        session.loc[session.index < open_ts + 15 * 60, "volume"] *= 4.0  # a busy opening every day
        frames.append(session)
    history = pd.concat(frames)
    baseline = compute_baseline(history, timezone="UTC", before_day=date(2026, 11, 20), computed_ts=0)
    assert baseline is not None
    opening = utc_ts("2026-11-20 14:30")
    at_open = measure("X", jump(opening, 100.0, volume=4000.0), baseline, timezone="UTC")
    midday = measure("X", jump(opening + 3 * 3600, 100.0, volume=4000.0), baseline, timezone="UTC")
    assert at_open is not None and midday is not None
    assert at_open.rel_volume == pytest.approx(1.0, rel=0.25) and not is_unusual(at_open)  # usual for the open
    assert midday.rel_volume == pytest.approx(4.0, rel=0.25) and is_unusual(midday)  # four times the usual


def test_instruments_without_volume_never_trigger() -> None:
    end_ts = utc_ts("2026-11-20 15:00")
    baseline = compute_baseline(
        crypto_history(end_ts, base_volume=0.0), timezone="UTC", before_day=date(2026, 11, 20), computed_ts=end_ts
    )
    assert baseline is not None and baseline.volume_median == 0.0
    move = measure("X", jump(end_ts, 100.0, volume=0.0), baseline, timezone="UTC")
    assert move is not None and move.rel_volume is None and not is_unusual(move)


def test_activity_message() -> None:
    move = Move("NVDA", 0, 5, 2.43, 5.12, 4.2)
    assert movers.move_message_de(move, 0) == (
        "Ungewöhnliche Kursbewegung: NVDA +2,4 % in 5 Min (5,1σ), Volumen 4,2-fach – Nachrichten vorgezogen"
    )
    assert movers.move_message_de(move, 1).endswith(", 1 Meldung erneut eingereiht")
    assert movers.move_message_de(Move("BTC-EUR", 0, 4, -3.0, -6.0, 3.5), 3).startswith(
        "Ungewöhnliche Kursbewegung: BTC-EUR -3,0 % in 4 Min (6,0σ)"
    )


# ------------------------------------------------------------------------------------------------
# Service
# ------------------------------------------------------------------------------------------------
class RecordingNews:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def prioritize(self, symbols: Collection[str], now: datetime) -> int:
        self.calls.append(list(symbols))
        return 2


@pytest.fixture
def watch_parts(db: Database, clock: ManualClock, bus: EventBus) -> tuple[MoveWatch, RecordingNews, SettingsStore]:
    seed_instruments(db)
    store = SettingsStore(db)
    news = RecordingNews()
    watch = MoveWatch(
        db=db,
        clock=clock,
        bus=bus,
        calendar=FakeCalendar(),
        instruments=InstrumentBook(db),
        runtime_settings=store,
        news=news,  # type: ignore[arg-type]
    )
    return watch, news, store


async def test_service_triggers_once_and_pulls_the_news_forward(
    watch_parts: tuple[MoveWatch, RecordingNews, SettingsStore], db: Database, clock: ManualClock
) -> None:
    watch, news, _ = watch_parts
    now_ts = clock.ts()
    history = crypto_history(now_ts - 5 * 60)
    seed_bars(db, "BTC-USD", history)
    seed_bars(db, "BTC-USD", jump(now_ts - 5 * 60, float(history["close"].iloc[-1]), volume=20_000.0))

    moves = await watch.on_minute(clock.now(), ["BTC-USD", "NO-SUCH-SYMBOL"])
    assert [move.symbol for move in moves] == ["BTC-USD"] and moves[0].z > movers.MOVE_Z_MIN
    assert news.calls == [["BTC-USD"]]
    messages = [entry.message for entry in repo_ops.list_activity(db, category="data")]
    assert len(messages) == 1 and messages[0].startswith("Ungewöhnliche Kursbewegung: BTC-USD +2,5 % in 5 Min")
    assert messages[0].endswith("2 Meldungen erneut eingereiht")

    # the same window, and a new one within the cooldown, never trigger again
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []
    clock.advance(60)
    seed_bars(db, "BTC-USD", jump(now_ts, 105.0 * 1.0, bars=1, step=0.02, volume=50_000.0))
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []
    assert news.calls == [["BTC-USD"]]
    await watch.stop()


async def test_service_respects_the_setting_and_ignores_quiet_or_stale_data(
    watch_parts: tuple[MoveWatch, RecordingNews, SettingsStore], db: Database, clock: ManualClock
) -> None:
    watch, news, store = watch_parts
    now_ts = clock.ts()
    history = crypto_history(now_ts)
    seed_bars(db, "BTC-USD", history)
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []  # an ordinary random walk

    store.update({"price_trigger_enabled": False}, now_ts)
    clock.advance(300)
    seed_bars(db, "BTC-USD", jump(now_ts, float(history["close"].iloc[-1]), volume=20_000.0))
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []  # switched off

    store.update({"price_trigger_enabled": True}, now_ts)
    clock.advance(3600)  # the jump is an hour old now: stale
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []
    assert news.calls == []
    await watch.stop()
    assert await watch.on_minute(clock.now(), ["BTC-USD"]) == []  # stopped
