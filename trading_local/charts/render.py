"""Candlestick chart PNGs for the vision LLM and the alert mails.

Drawing uses matplotlib's object-oriented API only (``Figure`` + ``FigureCanvasAgg``, no pyplot, no global
state): one LineCollection for all wicks and one PolyCollection each for candle bodies and volume bars
(the stack probe measured ~126 ms for 390 bars this way, mplfinance ~350 ms).

Design rules (the image is read by a vision model and embedded in mails)
* exact pixel size ``width x height``, white background, high-contrast ink, large axis numbers, no
  decoration; every word is German.
* prices on the axis and in the labels use a decimal POINT: the vision model compares them with the
  numeric facts it receives, which are plain floats. Prices are shown as the data source quotes them
  (London shares in pence); the unit is printed in the top right corner.
* x axis: ordinal bar positions, so nights, weekends and holidays leave no gaps; tick labels and the
  'Stand' time in the title are in the USER's time zone; thin dashed lines mark day changes.
* colours: up/down candles green/red with a large lightness difference (separable with colour-vision
  deficiency); EMA 20 blue and EMA 50 orange with a legend; stop red and target green with labelled tags;
  reference/entry price neutral dashed; the last price as a dark tag on the price axis.

``ChartRenderer`` serialises every database read and render on ONE dedicated worker thread (matplotlib is
not thread-safe) and never blocks the event loop.
"""

import asyncio
import functools
import gc
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from trading_local.alerts.formatting import (
    MONTHS_DE,
    TIMEFRAME_LABELS_DE,
    format_datetime,
    local_datetime,
    price_decimals,
    price_unit,
)
from trading_local.analysis.indicators import ema
from trading_local.barstore import BarReader
from trading_local.clock import Clock
from trading_local.contracts import MarketCalendar
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.models import Signal

log = logging.getLogger(__name__)

INTRADAY_MINUTES: Final[dict[str, int]] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
DAILY: Final = "1d"
TIMEFRAMES: Final[tuple[str, ...]] = (*INTRADAY_MINUTES, DAILY)
MIN_BARS: Final = 30  # fewer available bars: no chart
MAX_BARS: Final = 1000
EMA_SPANS: Final[tuple[int, int]] = (20, 50)
EMA_WARMUP_BARS: Final = 150  # extra history so that EMA 50 is settled at the first drawn bar
WIDTH_RANGE: Final = (320, 2560)
HEIGHT_RANGE: Final = (180, 1440)

DPI: Final = 100
_BAR_SECONDS = 60
_BASE_SIZE = (1024, 576)  # fonts are designed for this size and scale with the image
_LABEL_SPACING_PX = 125  # minimum distance between two x tick labels
_LABEL_HALF_WIDTH_PX = 48  # a label centred closer than this to the left plot edge would be cut off
_NICE_STEPS_MIN = (1, 2, 5, 10, 15, 30, 60, 120, 180, 240, 360, 720)

# Palette (validated for colour-vision deficiency against the white surface; text uses ink, not series colours).
BG: Final = "#ffffff"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"
UP: Final = "#34a853"
DOWN: Final = "#a61b1b"
EMA_COLORS: Final[tuple[str, str]] = ("#2a78d6", "#eb6834")
STOP_COLOR: Final = "#a61b1b"
TARGET_COLOR: Final = "#15803d"
REF_COLOR: Final = INK_SECONDARY


@dataclass(frozen=True, slots=True)
class SignalLevels:
    """Horizontal lines of a signal: reference (or entry) price, stop and target, in quoted price units."""

    ref: float
    stop: float
    target: float
    ref_label: str  # 'Einstieg' once the entry is known, otherwise 'Referenz'

    @classmethod
    def from_signal(cls, signal: Signal, asof_ts: int | None = None) -> "SignalLevels":
        """Entry-based levels once the entry is known, otherwise reference price plus the planned distances.

        With ``asof_ts`` the entry counts as known only if its 1-minute bar had closed by then
        (``entry_ts + 60 <= asof_ts``), so a chart as of an earlier time never shows a later entry. Stored
        stop/target prices are used with the entry, derived ones otherwise.
        """
        entry_known = asof_ts is None or signal.entry_ts is None or signal.entry_ts + _BAR_SECONDS <= asof_ts
        base = signal.entry_price
        if base is None or not entry_known:
            ref = signal.ref_price
            return cls(ref, ref - signal.sign * signal.stop_dist, ref + signal.sign * signal.target_dist, "Referenz")
        stop = signal.stop_price if signal.stop_price is not None else base - signal.sign * signal.stop_dist
        target = signal.target_price if signal.target_price is not None else base + signal.sign * signal.target_dist
        return cls(base, stop, target, "Einstieg")


@dataclass(frozen=True, slots=True)
class ChartSpec:
    symbol: str
    name: str
    timeframe: str  # one of TIMEFRAMES
    timezone: str  # IANA zone of the reader (x labels, title time)
    unit: str  # unit of the quoted prices, e.g. 'USD' or 'GBp'
    width: int = 1024
    height: int = 576
    levels: SignalLevels | None = None


type _LevelLine = tuple[float, str, str, str | tuple[int, tuple[int, int]]]  # price, label, colour, line style


@dataclass(frozen=True, slots=True)
class _Ticks:
    positions: list[int]
    labels: list[str]
    day_breaks: list[int]  # first bar of each new local day (intraday only)


# ------------------------------------------------------------------------------------------------
# Pure rendering
# ------------------------------------------------------------------------------------------------
def render_candles(frame: pd.DataFrame, spec: ChartSpec, *, bars: int) -> bytes:
    """PNG of the newest ``bars`` rows of a canonical bar frame (intraday: int epoch index of bar OPEN times;
    daily: 'YYYY-MM-DD' index). EMAs are computed over the whole frame, so extra older rows serve as warm-up.

    The image is exactly ``spec.width x spec.height`` pixels. It carries the PNG text chunks 'Title' (the chart
    title) and 'Description' (JSON: symbol, timeframe, bars, first_bar, last_bar, volume_panel, levels).
    Raises ``ValueError`` for an empty frame or ``bars < 1``.

    Figure, axes and artists reference each other, so a finished figure (with its pixel buffer) would wait for
    the next full cyclic garbage collection - measured: up to 14 figures alive and +40 MB over 200 renders.
    A young-generation collection after every render (about 2 ms, what ``pyplot.close`` does) frees it at once.
    """
    if frame.empty or bars < 1:
        raise ValueError(f"nothing to render: {len(frame)} rows, bars={bars}")
    try:
        return _draw_png(frame, spec, bars)
    finally:
        gc.collect(1)


def _draw_png(frame: pd.DataFrame, spec: ChartSpec, bars: int) -> bytes:
    emas = [ema(frame["close"], span).to_numpy()[-bars:] for span in EMA_SPANS]
    view = frame.iloc[-bars:]
    n = len(view)
    o, h, lo, c, v = (view[column].to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close", "volume"))
    has_volume = bool(np.nansum(v) > 0)
    scale = min(max(min(spec.width / _BASE_SIZE[0], spec.height / _BASE_SIZE[1]), 0.55), 2.5)
    y_lo, y_hi = _price_range(lo, h, emas, spec.levels)
    price_ticks = MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]).tick_values(y_lo, y_hi)
    price_ticks = price_ticks[(price_ticks >= y_lo) & (price_ticks <= y_hi)]
    tick_decimals = _tick_decimals(price_ticks)
    last_close = float(c[-1])
    last_label = f"{last_close:.{price_decimals(last_close)}f}"
    label_chars = max([len(last_label) + 1, *(len(f"{t:.{tick_decimals}f}") for t in price_ticks)])

    fig = Figure(figsize=(spec.width / DPI, spec.height / DPI), dpi=DPI, facecolor=BG)
    FigureCanvasAgg(fig)
    price_rect, volume_rect = _layout(spec, scale, label_chars, has_volume)
    ax = fig.add_axes(price_rect)
    axv = fig.add_axes(volume_rect, sharex=ax) if volume_rect is not None else None
    level_lines = [] if spec.levels is None else _level_lines(spec.levels)
    pad_right = _free_zone_bars(level_lines, n, price_rect[2] * spec.width, scale)
    ax.set_xlim(-0.7, n - 0.3 + pad_right)
    ax.set_ylim(y_lo, y_hi)

    _style_axes(ax, scale)
    _draw_candles(ax, o, h, lo, c, scale)
    _draw_emas(ax, emas, scale)
    _set_price_ticks(ax, price_ticks, tick_decimals, last_close, price_rect[3] * spec.height / (y_hi - y_lo), scale)
    if level_lines:
        _draw_levels(ax, level_lines, n, pad_right, scale)
    _draw_last_price(ax, last_close, last_label, scale)
    if axv is not None:
        _draw_volume(axv, o, c, v, scale)
    ticks = _time_ticks(view.index, spec.timeframe, spec.timezone, price_rect[2] * spec.width / (n + pad_right))
    _set_time_axis(ax, axv, ticks, scale)
    title = _title(spec, view.index)
    _draw_titles(fig, title, f"Kurs in {spec.unit}", price_rect, spec.height, scale)

    description = {  # what the chart shows, for reproducing an LLM analysis and for tests
        "symbol": spec.symbol,
        "timeframe": spec.timeframe,
        "bars": n,
        "first_bar": _plain(view.index[0]),
        "last_bar": _plain(view.index[-1]),
        "volume_panel": has_volume,
        "levels": None
        if spec.levels is None
        else {"ref": spec.levels.ref, "stop": spec.levels.stop, "target": spec.levels.target},
    }
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=BG,
                metadata={"Title": title, "Description": json.dumps(description, ensure_ascii=False)})
    return buffer.getvalue()


def _plain(value: object) -> object:
    """numpy scalars (intraday index) as plain Python values for JSON; day strings stay as they are."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _price_range(
    low: np.ndarray, high: np.ndarray, emas: list[np.ndarray], levels: SignalLevels | None
) -> tuple[float, float]:
    """Y limits that contain all candles, the visible EMA values and the signal lines, plus headroom."""
    values = [float(np.min(low)), float(np.max(high))]
    for line in emas:
        finite = line[np.isfinite(line)]
        if finite.size:
            values += [float(finite.min()), float(finite.max())]
    if levels is not None:
        values += [levels.ref, levels.stop, levels.target]
    y_min, y_max = min(values), max(values)
    span = y_max - y_min
    if span <= 0:
        span = abs(y_max) * 0.01 or 1.0
    return y_min - span * 0.05, y_max + span * 0.12  # more room on top for the legend


def _tick_decimals(ticks: np.ndarray) -> int:
    """Fewest decimals that print every tick exactly (a 2.5 step must not show as 62, 65, 68)."""
    for decimals in range(7):
        scaled = ticks * 10**decimals
        if np.allclose(scaled, np.round(scaled), atol=1e-6):
            return decimals
    return 7


def _layout(
    spec: ChartSpec, scale: float, label_chars: int, has_volume: bool
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float] | None]:
    """Axes rectangles (figure fractions) for the price panel and the optional volume panel."""
    tick_px = 11 * scale * DPI / 72
    left_px = 10 * scale
    right_px = label_chars * 0.64 * tick_px + 18 * scale  # price labels and the last-price tag
    top_px = 36 * scale
    bottom_px = 30 * scale
    gap_px = 6 * scale
    left = left_px / spec.width
    width = 1.0 - (left_px + right_px) / spec.width
    bottom = bottom_px / spec.height
    usable = 1.0 - (top_px + bottom_px) / spec.height
    if not has_volume:
        return (left, bottom, width, usable), None
    volume_h = usable * 0.2
    price_bottom = bottom + volume_h + gap_px / spec.height
    return (left, price_bottom, width, 1.0 - top_px / spec.height - price_bottom), (left, bottom, width, volume_h)


def _style_axes(ax: Axes, scale: float) -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.7 * scale, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.tick_right()
    ax.tick_params(colors=INK, labelsize=11 * scale, length=3 * scale, width=0.8 * scale, pad=3 * scale)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("right", "bottom"))
        spine.set_color(AXIS)
        spine.set_linewidth(0.8 * scale)


def _draw_candles(ax: Axes, o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, scale: float) -> None:
    x = np.arange(len(c), dtype=np.float64)
    colors = np.where(c >= o, UP, DOWN)
    wicks = np.stack([np.column_stack([x, lo]), np.column_stack([x, h])], axis=1)
    ax.add_collection(LineCollection(wicks, colors=colors, linewidths=1.0 * scale, zorder=2))
    half = 0.34
    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    bodies = np.stack(
        [np.column_stack([x - half, body_lo]), np.column_stack([x - half, body_hi]),
         np.column_stack([x + half, body_hi]), np.column_stack([x + half, body_lo])],
        axis=1,
    )
    # the edge in the body colour keeps flat (doji) bodies visible as a line
    ax.add_collection(PolyCollection(bodies, facecolors=colors, edgecolors=colors, linewidths=0.6 * scale, zorder=2))


def _draw_emas(ax: Axes, emas: list[np.ndarray], scale: float) -> None:
    """EMA lines (NaN during their warm-up) and their legend in the upper left corner."""
    x = np.arange(len(emas[0]))
    for values, color in zip(emas, EMA_COLORS, strict=True):
        ax.plot(x, values, color=color, linewidth=1.6 * scale, zorder=3)
    ax.legend(
        handles=[Line2D([], [], color=color, linewidth=2.2 * scale, label=f"EMA {span}")
                 for span, color in zip(EMA_SPANS, EMA_COLORS, strict=True)],
        loc="upper left", fontsize=10 * scale, frameon=True, framealpha=0.92, facecolor=BG, edgecolor=GRID,
        labelcolor=INK, borderpad=0.4, handlelength=1.6,
    )


def _set_price_ticks(
    ax: Axes, ticks: np.ndarray, decimals: int, last_close: float, px_per_price: float, scale: float
) -> None:
    """Fixed price ticks with exact labels; a label that would sit under the last-price tag is left out."""
    clearance_px = 13 * scale * DPI / 72

    def label(value: float, _position: int) -> str:
        return "" if abs(value - last_close) * px_per_price < clearance_px else f"{value:.{decimals}f}"

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(label))


def _set_time_axis(ax: Axes, axv: Axes | None, ticks: _Ticks, scale: float) -> None:
    """Day-change lines in both panels and the time labels under the lowest panel."""
    panels = [ax] if axv is None else [ax, axv]
    for panel in panels:
        for position in ticks.day_breaks:
            panel.axvline(position - 0.5, color=AXIS, linewidth=0.8 * scale, linestyle=(0, (4, 3)), zorder=1)
    if axv is not None:
        ax.tick_params(labelbottom=False)
    panels[-1].xaxis.set_major_locator(FixedLocator(ticks.positions))
    panels[-1].set_xticklabels(ticks.labels)


def _draw_titles(
    fig: Figure, title: str, unit_text: str, price_rect: tuple[float, float, float, float], height: int, scale: float
) -> None:
    """Title (left) and price unit (right) as figure texts at fixed positions.

    ``Axes.set_title`` would compute tight bounding boxes on every draw (about 20 ms per chart).
    """
    y = 1.0 - 9 * scale / height
    fig.text(price_rect[0], y, title, ha="left", va="top", fontsize=13 * scale, color=INK, fontweight="bold")
    fig.text(price_rect[0] + price_rect[2], y, unit_text, ha="right", va="top", fontsize=11 * scale,
             color=INK_SECONDARY)


def _draw_volume(axv: Axes, o: np.ndarray, c: np.ndarray, v: np.ndarray, scale: float) -> None:
    _style_axes(axv, scale)
    x = np.arange(len(v), dtype=np.float64)
    half = 0.34
    volume = np.nan_to_num(v, nan=0.0)
    zeros = np.zeros_like(volume)
    rects = np.stack(
        [np.column_stack([x - half, zeros]), np.column_stack([x - half, volume]),
         np.column_stack([x + half, volume]), np.column_stack([x + half, zeros])],
        axis=1,
    )
    axv.add_collection(PolyCollection(rects, facecolors=np.where(c >= o, UP, DOWN), edgecolors="none", alpha=0.55))
    axv.set_ylim(0.0, float(volume.max()) * 1.15 or 1.0)
    axv.yaxis.set_major_locator(MaxNLocator(nbins=2, integer=True, min_n_ticks=1))
    axv.yaxis.set_minor_locator(NullLocator())
    axv.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: _compact_volume(value)))
    axv.text(0.004, 0.94, "Volumen", transform=axv.transAxes, ha="left", va="top", fontsize=10 * scale,
             color=INK_SECONDARY)


def _compact_volume(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f}G".replace(".0G", "G")
    if value >= 1e6:
        return f"{value / 1e6:.1f}M".replace(".0M", "M")
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def _level_lines(levels: SignalLevels) -> list[_LevelLine]:
    decimals = price_decimals(levels.ref)
    return [
        (levels.target, f"Ziel {levels.target:.{decimals}f}", TARGET_COLOR, "-"),
        (levels.ref, f"{levels.ref_label} {levels.ref:.{decimals}f}", REF_COLOR, (0, (5, 3))),
        (levels.stop, f"Stopp {levels.stop:.{decimals}f}", STOP_COLOR, "-"),
    ]


def _free_zone_bars(lines: list[_LevelLine], n: int, plot_px: float, scale: float) -> int:
    """Bars of empty space right of the last candle: wide enough for the widest level tag (1 without tags)."""
    if not lines:
        return 1
    font_px = 10 * scale * DPI / 72
    need_px = max(len(label) for _, label, _, _ in lines) * 0.66 * font_px + 1.2 * font_px + 6 * scale
    return max(4, math.ceil(need_px * n / max(plot_px - need_px, 1.0)) + 1)


def _draw_levels(ax: Axes, lines: list[_LevelLine], n: int, pad_right: int, scale: float) -> None:
    """Reference, stop and target lines with tags in the free zone right of the last candle."""
    tag_px = 10 * scale * DPI / 72 * 1.9  # height of one tag including its padding
    transform = ax.transData
    pixel_y = sorted((transform.transform((0.0, price))[1], index) for index, (price, *_rest) in enumerate(lines))
    placed: dict[int, float] = {}
    previous = -math.inf
    for y_px, index in pixel_y:  # push overlapping tags apart, bottom to top
        y_px = max(y_px, previous + tag_px)
        placed[index] = y_px
        previous = y_px
    x_tag = n - 0.3 + pad_right - 0.2
    for index, (price, label, color, style) in enumerate(lines):
        ax.axhline(price, color=color, linewidth=1.4 * scale, linestyle=style, zorder=4)
        y = transform.inverted().transform((0.0, placed[index]))[1]
        ax.text(
            x_tag, y, label, ha="right", va="center", fontsize=10 * scale, color="#ffffff", fontweight="bold",
            zorder=6, bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": color},
        )


def _draw_last_price(ax: Axes, price: float, label: str, scale: float) -> None:
    ax.axhline(price, color=INK_SECONDARY, linewidth=0.9 * scale, linestyle=(0, (1.5, 2.0)), zorder=3)
    ax.annotate(
        label, xy=(1.0, price), xycoords=ax.get_yaxis_transform(), xytext=(3 * scale, 0), textcoords="offset points",
        ha="left", va="center", fontsize=10.5 * scale, color="#ffffff", fontweight="bold", annotation_clip=False,
        zorder=7, bbox={"boxstyle": "square,pad=0.25", "facecolor": INK, "edgecolor": INK},
    )


def _title(spec: ChartSpec, index: pd.Index) -> str:
    timeframe = TIMEFRAME_LABELS_DE.get(spec.timeframe, spec.timeframe)
    last = index[-1]
    if spec.timeframe == DAILY:
        stamp = date.fromisoformat(str(last)).strftime("%d.%m.%Y")
    else:
        close_ts = int(last) + INTRADAY_MINUTES[spec.timeframe] * 60
        zone = local_datetime(close_ts, spec.timezone).strftime("%Z")
        stamp = f"{format_datetime(close_ts, spec.timezone)} {zone}".rstrip()
    return f"{spec.symbol} · {spec.name} · {timeframe} · Stand {stamp}"


# ------------------------------------------------------------------------------------------------
# X axis ticks (ordinal positions, labels in the user's time zone)
# ------------------------------------------------------------------------------------------------
def _time_ticks(index: pd.Index, timeframe: str, timezone: str, px_per_bar: float) -> _Ticks:
    n = len(index)
    px_per_bar = max(px_per_bar, 1e-6)
    min_gap = max(1, math.ceil(_LABEL_SPACING_PX / px_per_bar))
    first_allowed = math.ceil(_LABEL_HALF_WIDTH_PX / px_per_bar)
    if timeframe == DAILY:
        return _daily_ticks(np.array(index.tolist(), dtype="datetime64[D]"), min_gap, first_allowed)
    local = pd.to_datetime(index.to_numpy(dtype=np.int64), unit="s", utc=True).tz_convert(ZoneInfo(timezone))
    minute_of_day = (local.hour * 60 + local.minute).to_numpy()
    day_key = (local.year * 10_000 + local.month * 100 + local.day).to_numpy()
    new_day = np.r_[True, day_key[1:] != day_key[:-1]]
    day_starts = np.flatnonzero(new_day)
    if len(day_starts) * min_gap > n:  # too many days for time labels: one label per (some) day(s)
        chosen = _thin(day_starts[day_starts >= first_allowed], np.array([], dtype=np.int64), min_gap)
        labels = [local[i].strftime("%d.%m.") for i in chosen]
    else:
        period = INTRADAY_MINUTES[timeframe]
        step = next((s for s in _NICE_STEPS_MIN if s >= period * min_gap), _NICE_STEPS_MIN[-1])
        grid = np.flatnonzero(minute_of_day % step == 0)
        chosen = _thin(day_starts[day_starts >= first_allowed], grid[grid >= first_allowed], min_gap)
        starts = set(day_starts.tolist())
        labels = [local[i].strftime("%d.%m. %H:%M" if i in starts else "%H:%M") for i in chosen]
    return _Ticks(chosen, labels, [int(i) for i in day_starts[1:]])


def _daily_ticks(days: np.ndarray, min_gap: int, first_allowed: int) -> _Ticks:
    months = days.astype("datetime64[M]")
    month_starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])[1:]
    if len(month_starts) >= 3:
        chosen = _thin(month_starts[month_starts >= first_allowed], np.array([], dtype=np.int64), min_gap)
        labels = [_month_label(days[i]) for i in chosen]
    else:
        week = (days.astype(np.int64) + 3) // 7  # 1970-01-01 was a Thursday: weeks start on Monday
        week_starts = np.flatnonzero(np.r_[True, week[1:] != week[:-1]])[1:]
        chosen = _thin(week_starts[week_starts >= first_allowed], np.array([], dtype=np.int64), min_gap)
        labels = [days[i].astype(date).strftime("%d.%m.") for i in chosen]
    return _Ticks(chosen, labels, [])


def _month_label(day: np.datetime64) -> str:
    value: date = day.astype(date)
    return f"{MONTHS_DE[value.month - 1]} {value.year}"


def _thin(preferred: np.ndarray, others: np.ndarray, min_gap: int) -> list[int]:
    """Tick positions at least ``min_gap`` bars apart: ``preferred`` ones first, then ``others`` that fit."""
    chosen: list[int] = []
    for candidate in preferred.tolist():
        if not chosen or candidate - chosen[-1] >= min_gap:
            chosen.append(candidate)
    for candidate in others.tolist():
        position = int(np.searchsorted(chosen, candidate))
        before_ok = position == 0 or candidate - chosen[position - 1] >= min_gap
        after_ok = position == len(chosen) or chosen[position] - candidate >= min_gap
        if before_ok and after_ok:
            chosen.insert(position, candidate)
    return chosen


# ------------------------------------------------------------------------------------------------
# The renderer service
# ------------------------------------------------------------------------------------------------
class ChartRenderer:
    """``contracts.ChartRenderer``: reads bars as of a decision time and renders them on one worker thread."""

    def __init__(
        self, *, db: Database, clock: Clock, calendar: MarketCalendar, instruments: InstrumentBook, timezone: str
    ) -> None:
        ZoneInfo(timezone)  # fail fast on a misconfigured time zone
        self._db = db
        self._clock = clock
        self._calendar = calendar
        self._instruments = instruments
        self._timezone = timezone
        # ONE thread: matplotlib is not thread-safe, and it keeps the number of database readers small.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chart-render")

    async def render(
        self,
        symbol: str,
        timeframe: str = "15m",
        *,
        asof_ts: int | None = None,
        bars: int = 120,
        signal: Signal | None = None,
        width: int = 1024,
        height: int = 576,
    ) -> bytes | None:
        """PNG chart of the bars that were closed at ``asof_ts`` (default: now); see the module docstring.

        ``bars`` is clamped to 30..1000, ``width`` to 320..2560 and ``height`` to 180..1440 pixels.
        Returns None for an unknown symbol or timeframe, with fewer than 30 bars, or on any error (logged).
        """
        try:
            cutoff = self._clock.ts() if asof_ts is None else int(asof_ts)
            job = functools.partial(
                self._render_blocking, symbol, timeframe, cutoff,
                _clamp(bars, MIN_BARS, MAX_BARS), signal, _clamp(width, *WIDTH_RANGE), _clamp(height, *HEIGHT_RANGE),
            )
            return await asyncio.get_running_loop().run_in_executor(self._executor, job)
        except Exception:  # noqa: BLE001 - never raises: a missing chart must not stop the caller
            log.exception("chart %s %s: rendering failed", symbol, timeframe)
            return None

    def close(self) -> None:
        """Stop the worker thread after the queued renders (for shutdown and tests)."""
        self._executor.shutdown(wait=True)

    def _render_blocking(
        self, symbol: str, timeframe: str, asof_ts: int, bars: int, signal: Signal | None, width: int, height: int
    ) -> bytes | None:
        started = time.perf_counter()
        instrument = self._instruments.get(symbol)
        if instrument is None:
            log.info("chart %s: unknown symbol", symbol)
            return None
        if timeframe not in TIMEFRAMES:
            log.warning("chart %s: unsupported timeframe %r", symbol, timeframe)
            return None
        reader = BarReader(self._db, self._instruments, self._calendar, asof_ts)
        need = bars + EMA_WARMUP_BARS
        if timeframe == DAILY:
            frame = reader.daily(symbol, limit=need)
        else:
            frame = reader.resampled(symbol, INTRADAY_MINUTES[timeframe], lookback_bars=need)
        if len(frame) < MIN_BARS:
            log.debug("chart %s %s: only %d bars as of %d", symbol, timeframe, len(frame), asof_ts)
            return None
        levels = None
        if signal is not None:
            if signal.symbol == symbol:
                levels = SignalLevels.from_signal(signal, asof_ts)
            else:
                log.warning("chart %s: ignoring the lines of signal %s for %s", symbol, signal.id, signal.symbol)
        spec = ChartSpec(
            symbol=symbol, name=instrument.name, timeframe=timeframe, timezone=self._timezone,
            unit=price_unit(instrument), width=width, height=height, levels=levels,
        )
        png = render_candles(frame, spec, bars=bars)
        log.debug("chart %s %s rendered in %.0f ms", symbol, timeframe, (time.perf_counter() - started) * 1000)
        return png


def _clamp(value: int, low: int, high: int) -> int:
    return min(max(int(value), low), high)
