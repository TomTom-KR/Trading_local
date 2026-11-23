"""Synthetic chart PNGs for the Ollama integration test (same visual language as trading_local.charts)."""

from io import BytesIO

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.figure import Figure

from trading_local.barstore import resample
from trading_local.testing import make_bars

UP, DOWN, EMA20, EMA50 = "#34a853", "#a61b1b", "#2a78d6", "#eb6834"


def trend_bars(
    start_ts: int, *, bars: int = 120, minutes: int = 15, drift: float = 0.0004, seed: int = 3
) -> pd.DataFrame:
    """``bars`` candles of ``minutes`` minutes from a deterministic random walk (``drift`` per 1-minute bar)."""
    return resample(make_bars(start_ts, bars * minutes, start_price=100.0, seed=seed, drift=drift, vol=0.0012), minutes)


def chart_facts(frame: pd.DataFrame) -> dict[str, float]:
    """Facts shaped like analysis.readings.chart_facts (German keys, rounded)."""
    close = frame["close"]
    return {
        "Kurs": round(float(close.iloc[-1]), 2),
        "EMA20 (15 Min)": round(float(close.ewm(span=20, adjust=False).mean().iloc[-1]), 2),
        "EMA50 (15 Min)": round(float(close.ewm(span=50, adjust=False).mean().iloc[-1]), 2),
        "Swing-Hoch (20 x 15 Min)": round(float(frame["high"].iloc[-20:].max()), 2),
        "Swing-Tief (20 x 15 Min)": round(float(frame["low"].iloc[-20:].min()), 2),
        "ATR (15 Min)": round(float((frame["high"] - frame["low"]).iloc[-14:].mean()), 2),
    }


def render_png(frame: pd.DataFrame, title: str, width: int = 1024, height: int = 576) -> bytes:
    """Candles, EMA 20 (blue) / EMA 50 (orange) with legend, volume panel and a dark last-price tag."""
    o, h, l, c, v = (frame[col].to_numpy() for col in ("open", "high", "low", "close", "volume"))
    x = np.arange(len(frame))
    colors = np.where(c >= o, UP, DOWN)
    fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.05, 0.28, 0.85, 0.64))
    axv = fig.add_axes((0.05, 0.08, 0.85, 0.17), sharex=ax)
    ax.add_collection(LineCollection([((i, lo), (i, hi)) for i, lo, hi in zip(x, l, h)], colors=colors, linewidths=1))
    bodies = [((i - 0.35, a), (i + 0.35, a), (i + 0.35, b), (i - 0.35, b)) for i, a, b in zip(x, o, c)]
    ax.add_collection(PolyCollection(bodies, facecolors=colors, edgecolors=colors))
    close = frame["close"]
    for span, color in ((20, EMA20), (50, EMA50)):
        line = close.ewm(span=span, adjust=False).mean().to_numpy()
        ax.plot(x, line, color=color, linewidth=1.6, label=f"EMA {span}")
    ax.legend(loc="upper left")
    ax.set_xlim(-1, len(frame) + 1)
    ax.set_ylim(l.min() * 0.995, h.max() * 1.005)
    ax.yaxis.tick_right()
    ax.grid(True, color="#e1e0d9")
    ax.tick_params(labelbottom=False)
    ax.annotate(f"{c[-1]:.2f}", xy=(1.0, c[-1]), xycoords=("axes fraction", "data"), color="white",
                bbox={"boxstyle": "square", "fc": "#0b0b0b"}, va="center")
    ax.axhline(c[-1], color="#0b0b0b", linewidth=0.8, linestyle=(0, (4, 3)))
    axv.bar(x, v, color=colors, alpha=0.55)
    axv.yaxis.tick_right()
    fig.text(0.05, 0.97, title, fontsize=13, fontweight="bold", va="top")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, facecolor="white")
    return buffer.getvalue()
