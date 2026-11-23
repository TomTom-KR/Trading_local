"""Boundaries, no look-ahead of signal levels and odd inputs of the chart renderer (review tests)."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from trading_local.charts import render as render_module
from trading_local.charts.render import ChartRenderer, ChartSpec, SignalLevels, render_candles
from trading_local.clock import ManualClock
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.models import Signal
from trading_local.testing import FakeCalendar, make_bars, seed_bars, seed_instruments


def utc(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


SESSION_OPEN = utc("2026-11-20T14:30")


def meta(png: bytes) -> dict[str, Any]:
    return json.loads(Image.open(BytesIO(png)).text["Description"])


@pytest.fixture
def small_db(db: Database) -> Database:
    seed_instruments(db)
    seed_bars(db, "NVDA", make_bars(SESSION_OPEN, 200, start_price=150.0, seed=5, base_volume=4e4))
    return db


@pytest.fixture
def renderer(small_db: Database, clock: ManualClock) -> Iterator[ChartRenderer]:
    instance = ChartRenderer(db=small_db, clock=clock, calendar=FakeCalendar(), instruments=InstrumentBook(small_db),
                             timezone="Europe/Berlin")
    yield instance
    instance.close()


def signal(**overrides: Any) -> Signal:
    values: dict[str, Any] = {
        "created_ts": SESSION_OPEN + 3600, "data_ts": SESSION_OPEN + 3600, "symbol": "NVDA", "direction": "LONG",
        "setup": "trend_pullback", "timeframe": "1m", "horizon_min": 120, "ref_price": 150.0, "atr": 0.5,
        "stop_dist": 0.8, "target_dist": 1.6, "score": 0.6, "confidence": 0.55, "engine_version": "test",
    }
    values.update(overrides)
    return Signal(**values)


async def test_a_one_minute_bar_counts_from_the_second_it_has_closed(renderer: ChartRenderer) -> None:
    last_open = SESSION_OPEN + 99 * 60
    at_close = await renderer.render("NVDA", "1m", asof_ts=last_open + 60, bars=60)
    just_before = await renderer.render("NVDA", "1m", asof_ts=last_open + 59, bars=60)
    assert at_close is not None and just_before is not None
    assert meta(at_close)["last_bar"] == last_open
    assert meta(just_before)["last_bar"] == last_open - 60


async def test_entry_levels_are_drawn_only_once_the_entry_was_known(renderer: ChartRenderer) -> None:
    """A chart as of a time before the entry must not show the entry price (it was not known yet)."""
    entered = signal(entry_ts=SESSION_OPEN + 150 * 60, entry_price=150.6, stop_price=149.8, target_price=152.2,
                     status="active")
    before = await renderer.render("NVDA", "1m", asof_ts=SESSION_OPEN + 150 * 60, bars=60, signal=entered)
    after = await renderer.render("NVDA", "1m", asof_ts=SESSION_OPEN + 151 * 60, bars=60, signal=entered)
    assert before is not None and after is not None
    assert meta(before)["levels"] == {"ref": 150.0, "stop": 149.2, "target": 151.6}
    assert meta(after)["levels"] == {"ref": 150.6, "stop": 149.8, "target": 152.2}
    assert SignalLevels.from_signal(entered, SESSION_OPEN + 150 * 60).ref_label == "Referenz"
    assert SignalLevels.from_signal(entered).ref_label == "Einstieg"


async def test_render_after_close_returns_none(small_db: Database, clock: ManualClock) -> None:
    instance = ChartRenderer(db=small_db, clock=clock, calendar=FakeCalendar(), instruments=InstrumentBook(small_db),
                             timezone="Europe/Berlin")
    instance.close()
    assert await instance.render("NVDA", "1m", asof_ts=SESSION_OPEN + 120 * 60) is None


async def test_invalid_arguments_give_none(renderer: ChartRenderer) -> None:
    asof = SESSION_OPEN + 120 * 60
    assert await renderer.render("NVDA", "1m", asof_ts=asof, bars="viele") is None  # type: ignore[arg-type]
    assert await renderer.render("NVDA", "", asof_ts=asof) is None


@pytest.mark.parametrize(
    ("start_price", "volume"),
    [(0.000123, 1e9), (97_345.0, 2.5), (1.0845, 0.0), (2650.0, 1e6)],
)
def test_extreme_price_magnitudes_render(start_price: float, volume: float) -> None:
    frame = make_bars(SESSION_OPEN, 80, start_price=start_price, seed=9, base_volume=volume)
    spec = ChartSpec(symbol="X", name="Test", timeframe="1m", timezone="Europe/Berlin", unit="USD",
                     levels=SignalLevels(start_price, start_price * 0.99, start_price * 1.02, "Referenz"))
    png = render_candles(frame, spec, bars=60)
    assert Image.open(BytesIO(png)).size == (1024, 576)
    assert meta(png)["volume_panel"] is (volume > 0)


def test_flat_prices_and_missing_volume_render() -> None:
    index = pd.Index(np.arange(SESSION_OPEN, SESSION_OPEN + 40 * 60, 60, dtype=np.int64), name="ts")
    frame = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": np.nan}, index=index)
    spec = ChartSpec(symbol="X", name="Flach", timeframe="1m", timezone="Europe/Berlin", unit="EUR")
    png = render_candles(frame, spec, bars=40)
    assert Image.open(BytesIO(png)).size == (1024, 576)
    assert meta(png)["volume_panel"] is False


def test_x_axis_crosses_the_daylight_saving_change() -> None:
    """25 Oct 2026: Berlin falls back from CEST to CET; the labels follow the local clock."""
    index = pd.Index(
        [ts for day in ("2026-10-23", "2026-10-26") for ts in range(utc(f"{day}T13:30"), utc(f"{day}T20:00"), 900)],
        name="ts",
    )
    ticks = render_module._time_ticks(index, "15m", "Europe/Berlin", px_per_bar=18.0)
    assert "26.10. 14:30" in ticks.labels  # 13:30 UTC: 15:30 on Friday (CEST), 14:30 on Monday (CET)
    assert any(label in ticks.labels for label in ("16:00", "17:00", "18:00"))
    assert ticks.day_breaks == [26]
