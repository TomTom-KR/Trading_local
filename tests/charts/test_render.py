import asyncio
import gc
import json
import statistics
import threading
import time
import weakref
from collections.abc import Iterator
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pandas as pd
import pytest
from matplotlib.figure import Figure
from PIL import Image

from trading_local.charts import build_chart_renderer
from trading_local.charts import render as render_module
from trading_local.charts.render import ChartRenderer, ChartSpec, SignalLevels, render_candles
from trading_local.clock import ManualClock
from trading_local.contracts import ChartRenderer as ChartRendererProtocol
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.models import Signal
from trading_local.testing import FakeCalendar, make_bars, seed_bars, seed_instruments

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SESSION_DAYS = ("2026-11-16", "2026-11-17", "2026-11-18", "2026-11-19", "2026-11-20")


def utc(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


FRIDAY_CLOSE = utc("2026-11-20T21:00")  # end of the last seeded NVDA session


def info(png: bytes) -> tuple[tuple[int, int], dict[str, Any], str]:
    """(pixel size, Description metadata, Title) of a rendered chart."""
    assert png.startswith(PNG_SIGNATURE)
    image = Image.open(BytesIO(png))
    return image.size, json.loads(image.text["Description"]), image.text["Title"]


def make_signal(**overrides: Any) -> Signal:
    values: dict[str, Any] = {
        "created_ts": FRIDAY_CLOSE, "data_ts": FRIDAY_CLOSE, "symbol": "NVDA", "direction": "LONG",
        "setup": "trend_pullback", "timeframe": "15m", "horizon_min": 390, "ref_price": 150.0, "atr": 1.0,
        "stop_dist": 1.5, "target_dist": 3.0, "score": 0.6, "confidence": 0.55, "engine_version": "test",
    }
    values.update(overrides)
    return Signal(**values)


@pytest.fixture
def chart_db(db: Database) -> Database:
    seed_instruments(db)
    sessions = []
    price = 142.0
    for number, day in enumerate(SESSION_DAYS):  # 5 New York sessions of 390 one-minute bars
        frame = make_bars(utc(f"{day}T14:30"), 390, start_price=price, seed=number + 1, vol=0.0012, base_volume=5e4)
        price = float(frame["close"].iloc[-1])
        sessions.append(frame)
    seed_bars(db, "NVDA", pd.concat(sessions))
    seed_bars(db, "^GDAXI", make_bars(utc("2026-11-20T08:00"), 510, start_price=23456.0, seed=7, base_volume=0.0))
    days = pd.bdate_range("2026-03-02", "2026-11-19")
    daily = make_bars(0, len(days), start_price=2650.0, seed=4, vol=0.012, base_volume=1e6)
    daily.index = pd.Index([day.strftime("%Y-%m-%d") for day in days], name="day")
    repo_market.upsert_bars_1d(db, "SHEL.L", daily)
    seed_bars(db, "BTC-USD", make_bars(FRIDAY_CLOSE - 20 * 60, 20, start_price=97000.0, seed=3, base_volume=2.0))
    return db


@pytest.fixture
def renderer(chart_db: Database, clock: ManualClock) -> Iterator[ChartRenderer]:
    instance = ChartRenderer(
        db=chart_db, clock=clock, calendar=FakeCalendar(), instruments=InstrumentBook(chart_db),
        timezone="Europe/Berlin",
    )
    yield instance
    instance.close()


# ------------------------------------------------------------------------------------------------
# Output format
# ------------------------------------------------------------------------------------------------
def test_factory_returns_a_chart_renderer(chart_db: Database, clock: ManualClock) -> None:
    instance = build_chart_renderer(db=chart_db, clock=clock, calendar=FakeCalendar(),
                                    instruments=InstrumentBook(chart_db))
    try:
        assert isinstance(instance, ChartRendererProtocol)
    finally:
        assert isinstance(instance, ChartRenderer)
        instance.close()


async def test_png_with_exact_default_size_and_title(renderer: ChartRenderer) -> None:
    png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE)
    assert png is not None
    size, meta, title = info(png)
    assert size == (1024, 576)
    assert meta["bars"] == 120 and meta["timeframe"] == "15m" and meta["symbol"] == "NVDA"
    assert meta["last_bar"] == FRIDAY_CLOSE - 900
    assert title == "NVDA · NVIDIA · 15 Min · Stand 20.11.2026 22:00 CET"  # the user's time zone


@pytest.mark.parametrize(
    ("requested", "expected"),
    [((800, 450), (800, 450)), ((1023, 577), (1023, 577)), ((1280, 720), (1280, 720)), ((100, 100), (320, 180)),
     ((5000, 3000), (2560, 1440))],
)
async def test_exact_pixel_size(
    renderer: ChartRenderer, requested: tuple[int, int], expected: tuple[int, int]
) -> None:
    png = await renderer.render("NVDA", "5m", asof_ts=FRIDAY_CLOSE, width=requested[0], height=requested[1])
    assert png is not None
    assert info(png)[0] == expected


async def test_title_follows_the_configured_time_zone(chart_db: Database, clock: ManualClock) -> None:
    instance = ChartRenderer(db=chart_db, clock=clock, calendar=FakeCalendar(), instruments=InstrumentBook(chart_db),
                             timezone="America/New_York")
    try:
        png = await instance.render("NVDA", "1h", asof_ts=FRIDAY_CLOSE)
    finally:
        instance.close()
    assert png is not None
    assert info(png)[2] == "NVDA · NVIDIA · 1 Std · Stand 20.11.2026 16:00 EST"


@pytest.mark.parametrize(("timeframe", "minutes"), [("1m", 1), ("5m", 5), ("15m", 15), ("1h", 60)])
async def test_every_intraday_timeframe(renderer: ChartRenderer, timeframe: str, minutes: int) -> None:
    asof = utc("2026-11-20T18:07")
    png = await renderer.render("NVDA", timeframe, asof_ts=asof, bars=40)
    assert png is not None
    _, meta, _ = info(png)
    assert meta["timeframe"] == timeframe
    assert 30 <= meta["bars"] <= 40
    assert meta["last_bar"] + minutes * 60 <= asof  # only complete buckets
    assert meta["last_bar"] % (minutes * 60) == 0


async def test_daily_timeframe_uses_completed_sessions_only(renderer: ChartRenderer) -> None:
    png = await renderer.render("SHEL.L", "1d", asof_ts=utc("2026-11-19T12:00"))
    assert png is not None
    _, meta, title = info(png)
    assert meta["last_bar"] == "2026-11-18"  # the London session of 19.11. was still running
    assert title == "SHEL.L · Shell · 1 Tag · Stand 18.11.2026"
    later = await renderer.render("SHEL.L", "1d", asof_ts=utc("2026-11-19T17:00"))
    assert later is not None and info(later)[1]["last_bar"] == "2026-11-19"


async def test_volume_panel_only_when_the_instrument_has_volume(renderer: ChartRenderer) -> None:
    with_volume = await renderer.render("NVDA", "5m", asof_ts=FRIDAY_CLOSE)
    without = await renderer.render("^GDAXI", "5m", asof_ts=utc("2026-11-20T16:30"))
    assert with_volume is not None and without is not None
    assert info(with_volume)[1]["volume_panel"] is True
    assert info(without)[1]["volume_panel"] is False


# ------------------------------------------------------------------------------------------------
# Signal lines
# ------------------------------------------------------------------------------------------------
async def test_signal_lines_from_reference_and_distances(renderer: ChartRenderer) -> None:
    plain = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE)
    long_png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=make_signal())
    short_png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=make_signal(direction="SHORT"))
    assert plain is not None and long_png is not None and short_png is not None
    assert info(plain)[1]["levels"] is None
    assert info(long_png)[1]["levels"] == {"ref": 150.0, "stop": 148.5, "target": 153.0}
    assert info(short_png)[1]["levels"] == {"ref": 150.0, "stop": 151.5, "target": 147.0}
    assert long_png != plain and info(long_png)[0] == (1024, 576)


async def test_signal_lines_use_stored_levels_after_the_entry(renderer: ChartRenderer) -> None:
    signal = make_signal(entry_price=150.4, stop_price=148.9, target_price=153.4, status="active")
    png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=signal)
    assert png is not None
    assert info(png)[1]["levels"] == {"ref": 150.4, "stop": 148.9, "target": 153.4}
    assert SignalLevels.from_signal(signal).ref_label == "Einstieg"
    assert SignalLevels.from_signal(make_signal()).ref_label == "Referenz"


async def test_signal_of_another_symbol_is_ignored(renderer: ChartRenderer) -> None:
    png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=make_signal(symbol="SAP.DE"))
    assert png is not None and info(png)[1]["levels"] is None


# ------------------------------------------------------------------------------------------------
# Missing data and errors
# ------------------------------------------------------------------------------------------------
async def test_too_little_data_gives_none(renderer: ChartRenderer) -> None:
    assert await renderer.render("BTC-USD", "1m", asof_ts=FRIDAY_CLOSE) is None  # 20 bars
    assert await renderer.render("NVDA", "15m", asof_ts=utc("2026-11-16T21:00")) is None  # 26 complete buckets
    assert await renderer.render("NVDA", "15m", asof_ts=utc("2026-11-16T22:00")) is None
    assert await renderer.render("NVDA", "5m", asof_ts=utc("2026-11-16T14:30")) is None  # nothing yet
    assert await renderer.render("SAP.DE", "15m", asof_ts=FRIDAY_CLOSE) is None  # no bars at all


async def test_unknown_symbol_and_timeframe_give_none(renderer: ChartRenderer) -> None:
    assert await renderer.render("NOPE", asof_ts=FRIDAY_CLOSE) is None
    assert await renderer.render("NVDA", "2h", asof_ts=FRIDAY_CLOSE) is None


async def test_render_never_raises(
    renderer: ChartRenderer, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(*args: Any, **kwargs: Any) -> bytes:
        raise RuntimeError("matplotlib exploded")

    monkeypatch.setattr(render_module, "render_candles", broken)
    assert await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE) is None
    assert "rendering failed" in caplog.text


# ------------------------------------------------------------------------------------------------
# No look-ahead
# ------------------------------------------------------------------------------------------------
async def test_bars_after_asof_are_never_drawn(renderer: ChartRenderer, chart_db: Database) -> None:
    asof = utc("2026-11-19T18:00")
    before = {tf: await renderer.render("NVDA", tf, asof_ts=asof, signal=make_signal()) for tf in ("1m", "5m", "15m")}
    later = await renderer.render("NVDA", "15m", asof_ts=FRIDAY_CLOSE)
    # remove everything that was not yet known at `asof` and render again: the images must be identical
    chart_db.execute("DELETE FROM bars_1m WHERE symbol = 'NVDA' AND ts >= ?", (asof - 59,))
    for timeframe, png in before.items():
        assert png is not None
        assert png == await renderer.render("NVDA", timeframe, asof_ts=asof, signal=make_signal()), timeframe
    assert later is not None and later != before["15m"]


async def test_default_asof_is_the_clock(renderer: ChartRenderer, clock: ManualClock) -> None:
    png = await renderer.render("NVDA", "15m", bars=30)  # clock: Friday 15:00 UTC, the session opened 14:30
    assert png is not None
    assert info(png)[1]["last_bar"] == utc("2026-11-20T14:45")
    clock.set(utc("2026-11-20T15:20"))
    later = await renderer.render("NVDA", "15m", bars=30)
    assert later is not None and info(later)[1]["last_bar"] == utc("2026-11-20T15:00")


# ------------------------------------------------------------------------------------------------
# Threading, leaks, speed
# ------------------------------------------------------------------------------------------------
async def test_renders_are_serialised_on_one_worker_thread(
    renderer: ChartRenderer, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[str] = []
    active = 0
    max_active = 0
    lock = threading.Lock()
    original = render_module.render_candles

    def tracking(*args: Any, **kwargs: Any) -> bytes:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            threads.append(threading.current_thread().name)
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(render_module, "render_candles", tracking)
    results = await asyncio.gather(*(renderer.render("NVDA", tf, asof_ts=FRIDAY_CLOSE, bars=40, width=480, height=270)
                                     for tf in ("1m", "5m", "15m", "1h", "5m", "1m")))
    assert all(result is not None for result in results)
    assert max_active == 1
    assert len(set(threads)) == 1 and threads[0].startswith("chart-render")


def _small_frame() -> pd.DataFrame:
    return make_bars(utc("2026-11-20T14:30"), 60, seed=11)


def test_repeated_renders_do_not_leak_figures_or_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated renders leave no figure behind and do not grow the number of live Python objects.

    120 renders (a 200-render run passes as well but takes ~19 s alone, beyond the test-time budget).
    tracemalloc would slow matplotlib down about four-fold; one leaked figure per render would add
    thousands of live objects per render, so the object count is a sharp leak detector.

    Figures must also go away WITHOUT a full collection: their reference cycles would otherwise keep up to
    14 figures (and their pixel buffers, +40 MB at 1024 x 576) alive until the cyclic GC gets round to them.
    """
    spec = ChartSpec(symbol="NVDA", name="NVIDIA", timeframe="1m", timezone="Europe/Berlin", unit="USD",
                     width=320, height=180, levels=SignalLevels(100.0, 99.0, 102.0, "Referenz"))
    frame = _small_frame()
    created: list[weakref.ref[Figure]] = []

    def tracked(*args: Any, **kwargs: Any) -> Figure:
        figure = Figure(*args, **kwargs)
        created.append(weakref.ref(figure))
        return figure

    def live() -> tuple[int, int]:
        gc.collect()
        objects = gc.get_objects()
        return sum(isinstance(obj, Figure) for obj in objects), len(objects)

    for _ in range(20):  # warm-up: font cache, text layout cache
        render_candles(frame, spec, bars=40)
    figures_before, objects_before = live()
    monkeypatch.setattr(render_module, "Figure", tracked)
    most_alive = 0
    for _ in range(100):
        render_candles(frame, spec, bars=40)
        most_alive = max(most_alive, sum(ref() is not None for ref in created))
    figures_after, objects_after = live()
    assert most_alive <= 1
    assert figures_after <= figures_before
    assert objects_after - objects_before < 2_000, f"{objects_after - objects_before} more live objects"


async def test_typical_render_time(renderer: ChartRenderer) -> None:
    signal = make_signal()
    await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=signal)  # warm-up
    durations = []
    for _ in range(5):
        started = time.perf_counter()
        png = await renderer.render("NVDA", asof_ts=FRIDAY_CLOSE, signal=signal)
        durations.append(time.perf_counter() - started)
        assert png is not None
    median = statistics.median(durations)
    print(f"\nchart render (15m, 120 bars, signal lines, 1024x576, incl. DB read): median {median * 1000:.0f} ms")
    assert median < 0.4


# ------------------------------------------------------------------------------------------------
# X axis
# ------------------------------------------------------------------------------------------------
def test_intraday_ticks_are_in_the_users_time_zone() -> None:
    index = pd.Index(
        [ts for day in ("2026-11-19", "2026-11-20") for ts in range(utc(f"{day}T14:30"), utc(f"{day}T21:00"), 900)],
        name="ts",
    )
    ticks = render_module._time_ticks(index, "15m", "Europe/Berlin", px_per_bar=18.0)
    assert ticks.day_breaks == [26]
    assert "20.11. 15:30" in ticks.labels  # 14:30 UTC is 15:30 in Berlin
    assert all(label.endswith(":00") or label == "20.11. 15:30" for label in ticks.labels)
    gaps = [b - a for a, b in zip(ticks.positions, ticks.positions[1:])]
    assert min(gaps) >= 7  # 125 px label spacing at 18 px per bar
    many_days = render_module._time_ticks(index, "15m", "Europe/Berlin", px_per_bar=2.0)
    assert many_days.labels == ["20.11."]


def test_daily_ticks_show_months() -> None:
    days = pd.Index([day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-05-04", "2026-11-19")], name="day")
    ticks = render_module._time_ticks(days, "1d", "Europe/Berlin", px_per_bar=6.0)
    assert ticks.labels[-3:] == ["Sep 2026", "Okt 2026", "Nov 2026"]
    assert ticks.day_breaks == []
