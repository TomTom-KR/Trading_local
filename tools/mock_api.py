"""Development mock of the Trading Local REST/WebSocket API with deterministic German demo data.

    python tools/mock_api.py [--port 8765] [--host 127.0.0.1] [--now 2026-11-19T15:12:00Z | --now real] [--offline]

Serves ``trading_local/web/static`` at ``/`` and implements the complete API contract of the GUI. Every
payload is built from the pydantic models in ``trading_local.dto`` (and ``config.RuntimeSettings``), so a
shape mismatch fails loudly here instead of silently in the browser.

The demo clock runs in real time but starts at ``--now`` (default: a Thursday afternoon, when Europe and
the USA trade and Asia's next session is inside the 24-hour ribbon). ``--now real`` uses the wall clock.
Prices are deterministic random walks on the real trading windows of each venue; signals, the AI
portfolio and the statistics are derived from those prices, so charts, markers and results agree.
State-changing calls (orders, settings, reset, validation runs ...) work in memory.
"""

import argparse
import asyncio
import contextlib
import math
import mimetypes
import sys
import time
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow 'python tools/mock_api.py' from the project root
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException, Query, Request, WebSocket  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from trading_local import dto  # noqa: E402
from trading_local.config import DEFAULT_COSTS, RuntimeSettings  # noqa: E402
from trading_local.models import Instrument  # noqa: E402
from trading_local.presenters import SETUP_LABELS_DE  # noqa: E402
from trading_local.sim import sizing  # noqa: E402

STATIC_DIR = ROOT / "trading_local" / "web" / "static"
DEFAULT_DEMO_NOW = datetime(2026, 11, 19, 15, 12, tzinfo=UTC)
VERSION = "0.1.0 (Demo-Server)"
ENGINE_VERSION = "engine-2026.11.1"
BASE = "EUR"
START_CASH = 100_000.0
HISTORY_DAYS = 9  # intraday history before the demo start
FUTURE_DAYS = 7  # prices are prepared this far ahead (the mock is meant for sessions shorter than a week)
DAILY_HISTORY = 260
MAX_BAR_SYMBOLS = 20
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; "
    "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)

Region = Literal["US", "EU", "ASIA", "GLOBAL"]


# ------------------------------------------------------------------------------------------------
# Demo clock
# ------------------------------------------------------------------------------------------------
class DemoClock:
    """Real-time clock that starts at ``anchor`` (None: the wall clock)."""

    def __init__(self, anchor: datetime | None) -> None:
        self._offset = 0.0 if anchor is None else anchor.timestamp() - time.time()

    def now(self) -> float:
        return time.time() + self._offset

    def ts(self) -> int:
        return int(self.now())


# ------------------------------------------------------------------------------------------------
# Trading calendar (fixed regular windows per venue, weekends closed, no holidays)
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Venue:
    key: str
    name: str
    region: Region
    tz: str
    sessions: tuple[tuple[str, str], ...]
    pre: tuple[str, str] | None = None
    post: tuple[str, str] | None = None
    weekdays_only: bool = True
    always_open: bool = False


VENUES: dict[str, Venue] = {
    v.key: v
    for v in (
        Venue("XTKS", "Tokio", "ASIA", "Asia/Tokyo", (("09:00", "11:30"), ("12:30", "15:30"))),
        Venue("XHKG", "Hongkong", "ASIA", "Asia/Hong_Kong", (("09:30", "12:00"), ("13:00", "16:00"))),
        Venue("XKRX", "Seoul", "ASIA", "Asia/Seoul", (("09:00", "15:30"),)),
        Venue("XTAI", "Taipeh", "ASIA", "Asia/Taipei", (("09:00", "13:30"),)),
        Venue("XETR", "Xetra", "EU", "Europe/Berlin", (("09:00", "17:30"),)),
        Venue("XLON", "London", "EU", "Europe/London", (("08:00", "16:30"),)),
        Venue("XPAR", "Paris", "EU", "Europe/Paris", (("09:00", "17:30"),)),
        Venue("XAMS", "Amsterdam", "EU", "Europe/Amsterdam", (("09:00", "17:30"),)),
        Venue("XSWX", "Zürich", "EU", "Europe/Zurich", (("09:00", "17:30"),)),
        Venue("XNYS", "New York", "US", "America/New_York", (("09:30", "16:00"),), ("04:00", "09:30"), ("16:00", "20:00")),
        Venue("XNAS", "Nasdaq", "US", "America/New_York", (("09:30", "16:00"),), ("04:00", "09:30"), ("16:00", "20:00")),
        Venue("CRYPTO", "Krypto", "GLOBAL", "UTC", (), weekdays_only=False, always_open=True),
        Venue("KRAKEN", "Kraken", "GLOBAL", "UTC", (), weekdays_only=False, always_open=True),
        Venue("FX", "Devisen", "GLOBAL", "UTC", (("00:00", "24:00"),)),
    )
}
REGION_LABELS: dict[str, str] = {"ASIA": "Asien", "EU": "Europa", "US": "USA", "GLOBAL": "Global"}

type Segment = tuple[int, int, str]  # (start_ts, end_ts, kind)


def _local_ts(day: date, hhmm: str, tz: ZoneInfo) -> int:
    if hhmm == "24:00":
        return int(datetime.combine(day + timedelta(days=1), dtime(0, 0), tz).timestamp())
    return int(datetime.combine(day, dtime.fromisoformat(hhmm), tz).timestamp())


def venue_segments(venue: Venue, start: int, end: int, *, extended: bool = True) -> list[Segment]:
    """Trading windows of ``venue`` intersecting [start, end), clipped and ascending."""
    if start >= end:
        return []
    if venue.always_open:
        return [(start, end, "regular")]
    tz = ZoneInfo(venue.tz)
    day = datetime.fromtimestamp(start, tz).date() - timedelta(days=1)
    last = datetime.fromtimestamp(end, tz).date() + timedelta(days=1)
    out: list[Segment] = []
    while day <= last:
        if not (venue.weekdays_only and day.weekday() >= 5):
            windows = [(a, b, "regular") for a, b in venue.sessions]
            if extended and venue.pre:
                windows.append((*venue.pre, "pre"))
            if extended and venue.post:
                windows.append((*venue.post, "post"))
            for a, b, kind in windows:
                s, e = _local_ts(day, a, tz), _local_ts(day, b, tz)
                if e > start and s < end:
                    out.append((max(s, start), min(e, end), kind))
        day += timedelta(days=1)
    return sorted(out)


def venue_state(venue: Venue, now: int) -> dto.ExchangeSessionOut:
    """Session phase of ``venue`` at ``now`` with the surrounding boundaries."""
    base = {"exchange": venue.key, "name": venue.name, "region": venue.region, "timezone": venue.tz}
    if venue.always_open:
        return dto.ExchangeSessionOut(**base, phase="open", is_open=True, note="Handel rund um die Uhr")
    segs = venue_segments(venue, now - 6 * 86400, now + 7 * 86400, extended=True)
    regular = [(s, e) for s, e, k in segs if k == "regular"]
    current = next((r for r in regular if r[0] <= now < r[1]), None)
    upcoming = next((r for r in regular if r[0] > now), None)
    previous = next((r for r in reversed(regular) if r[1] <= now), None)
    ext = next(((s, e, k) for s, e, k in segs if k != "regular" and s <= now < e), None)
    nxt_open = upcoming[0] if upcoming else None
    if current:
        return dto.ExchangeSessionOut(
            **base, phase="open", is_open=True, since=current[0], until=current[1], next_open=nxt_open,
            next_close=current[1], note="Handel",
        )
    nxt_close = upcoming[1] if upcoming else None
    if ext:
        return dto.ExchangeSessionOut(
            **base, phase=ext[2], is_open=False, since=ext[0], until=ext[1], next_open=nxt_open, next_close=nxt_close,
            note="Vorbörse" if ext[2] == "pre" else "Nachbörse",
        )
    tz = ZoneInfo(venue.tz)
    same_day = (
        previous is not None and upcoming is not None
        and datetime.fromtimestamp(previous[0], tz).date() == datetime.fromtimestamp(upcoming[0], tz).date()
    )
    if same_day:
        return dto.ExchangeSessionOut(
            **base, phase="break", is_open=False, since=previous[1], until=upcoming[0], next_open=nxt_open,
            next_close=nxt_close, note="Mittagspause",
        )
    next_change = min((s for s, _, _ in segs if s > now), default=None)
    weekend = datetime.fromtimestamp(now, tz).weekday() >= 5
    return dto.ExchangeSessionOut(
        **base, phase="closed", is_open=False, since=previous[1] if previous else None, until=next_change,
        next_open=nxt_open, next_close=nxt_close, note="Wochenende" if weekend else "Geschlossen",
    )


# ------------------------------------------------------------------------------------------------
# Universe
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Spec:
    symbol: str
    name: str
    asset_class: Literal["equity", "index", "crypto", "fx", "future"]
    exchange: str
    currency: str
    price: float  # price at the demo start, in source units
    vol: float  # standard deviation of the 1-minute log return
    volume: float  # mean 1-minute volume (0: the source has no volume)
    price_scale: float = 1.0
    delay_min: int = 0
    tradable: bool = True
    benchmark: str | None = None

    @property
    def region(self) -> Region:
        return VENUES[self.exchange].region


UNIVERSE: tuple[Spec, ...] = (
    Spec("7203.T", "Toyota Motor", "equity", "XTKS", "JPY", 3312, 0.0009, 42_000, delay_min=20, benchmark="^N225"),
    Spec("6758.T", "Sony Group", "equity", "XTKS", "JPY", 4312, 0.0010, 31_000, delay_min=20, benchmark="^N225"),
    Spec("9984.T", "SoftBank Group", "equity", "XTKS", "JPY", 3025, 0.0016, 55_000, delay_min=20, benchmark="^N225"),
    Spec("0700.HK", "Tencent", "equity", "XHKG", "HKD", 344.0, 0.0011, 60_000, delay_min=15, benchmark="^HSI"),
    Spec("9988.HK", "Alibaba", "equity", "XHKG", "HKD", 83.35, 0.0014, 120_000, delay_min=15, benchmark="^HSI"),
    Spec("005930.KS", "Samsung Electronics", "equity", "XKRX", "KRW", 183_100, 0.0010, 90_000, delay_min=20),
    Spec("2330.TW", "TSMC", "equity", "XTAI", "TWD", 2000, 0.0011, 35_000, delay_min=20),
    Spec("^N225", "Nikkei 225", "index", "XTKS", "JPY", 56_436.65, 0.0005, 0, delay_min=20),
    Spec("^HSI", "Hang Seng", "index", "XHKG", "HKD", 22_410.8, 0.0006, 0, delay_min=15),
    Spec("SAP.DE", "SAP", "equity", "XETR", "EUR", 183.68, 0.0009, 9_000, delay_min=15, benchmark="^GDAXI"),
    Spec("SIE.DE", "Siemens", "equity", "XETR", "EUR", 254.65, 0.0008, 6_500, delay_min=15, benchmark="^GDAXI"),
    Spec("ALV.DE", "Allianz", "equity", "XETR", "EUR", 451.8, 0.0006, 3_200, delay_min=15, benchmark="^GDAXI"),
    Spec("RHM.DE", "Rheinmetall", "equity", "XETR", "EUR", 693.4, 0.0014, 4_100, delay_min=15, benchmark="^GDAXI"),
    Spec("IFX.DE", "Infineon", "equity", "XETR", "EUR", 45.44, 0.0012, 21_000, delay_min=15, benchmark="^GDAXI"),
    Spec("ASML.AS", "ASML Holding", "equity", "XAMS", "EUR", 1168.0, 0.0011, 2_800, benchmark="^STOXX50E"),
    Spec("MC.PA", "LVMH", "equity", "XPAR", "EUR", 381.95, 0.0009, 3_600, delay_min=15, benchmark="^STOXX50E"),
    Spec("NESN.SW", "Nestlé", "equity", "XSWX", "CHF", 85.55, 0.0006, 12_000, delay_min=15),
    Spec("SHEL.L", "Shell", "equity", "XLON", "GBP", 3574.5, 0.0008, 24_000, price_scale=0.01, delay_min=15),
    Spec("AZN.L", "AstraZeneca", "equity", "XLON", "GBP", 13_412.0, 0.0007, 5_200, price_scale=0.01, delay_min=15),
    Spec("^GDAXI", "DAX", "index", "XETR", "EUR", 24_802.78, 0.0004, 0, delay_min=15),
    Spec("^STOXX50E", "Euro Stoxx 50", "index", "XETR", "EUR", 5941.95, 0.0004, 0),
    Spec("AAPL", "Apple", "equity", "XNAS", "USD", 342.56, 0.0008, 160_000, benchmark="^GSPC"),
    Spec("MSFT", "Microsoft", "equity", "XNAS", "USD", 484.52, 0.0007, 95_000, benchmark="^GSPC"),
    Spec("NVDA", "NVIDIA", "equity", "XNAS", "USD", 125.53, 0.0013, 420_000, benchmark="^GSPC"),
    Spec("AMZN", "Amazon", "equity", "XNAS", "USD", 246.82, 0.0009, 140_000, benchmark="^GSPC"),
    Spec("TSLA", "Tesla", "equity", "XNAS", "USD", 345.82, 0.0017, 260_000, benchmark="^GSPC"),
    Spec("META", "Meta Platforms", "equity", "XNAS", "USD", 560.84, 0.0010, 70_000, benchmark="^GSPC"),
    Spec("JPM", "JPMorgan Chase", "equity", "XNYS", "USD", 310.53, 0.0007, 48_000, benchmark="^GSPC"),
    Spec("^GSPC", "S&P 500", "index", "XNYS", "USD", 6910.39, 0.0004, 0),
    Spec("^NDX", "Nasdaq 100", "index", "XNAS", "USD", 25_028.98, 0.0005, 0),
    Spec("BTC-EUR", "Bitcoin", "crypto", "KRAKEN", "EUR", 55_643.0, 0.0011, 35.0),
    Spec("ETH-EUR", "Ethereum", "crypto", "KRAKEN", "EUR", 1569.61, 0.0014, 420.0, benchmark="BTC-EUR"),
    Spec("EURUSD=X", "EUR/USD", "fx", "FX", "USD", 1.097454, 0.00012, 0, tradable=False),
    Spec("EURGBP=X", "EUR/GBP", "fx", "FX", "GBP", 0.8381, 0.00010, 0, tradable=False),
    Spec("EURJPY=X", "EUR/JPY", "fx", "FX", "JPY", 173.904, 0.00014, 0, tradable=False),
    Spec("EURCHF=X", "EUR/CHF", "fx", "FX", "CHF", 0.92081, 0.00009, 0, tradable=False),
    Spec("EURHKD=X", "EUR/HKD", "fx", "FX", "HKD", 8.5969, 0.00012, 0, tradable=False),
    Spec("EURKRW=X", "EUR/KRW", "fx", "FX", "KRW", 1475.7, 0.00015, 0, tradable=False),
    Spec("EURTWD=X", "EUR/TWD", "fx", "FX", "TWD", 36.189, 0.00013, 0, tradable=False),
)
SPECS: dict[str, Spec] = {s.symbol: s for s in UNIVERSE}
EUR_PER_UNIT: dict[str, float] = {
    "EUR": 1.0, "USD": 1 / 1.097454, "GBP": 1 / 0.8381, "JPY": 1 / 173.904, "CHF": 1 / 0.92081, "HKD": 1 / 8.5969,
    "KRW": 1 / 1475.7, "TWD": 1 / 36.189,
}


def to_instrument(spec: Spec) -> Instrument:
    return Instrument(
        spec.symbol, spec.name, spec.asset_class, spec.region, spec.exchange, spec.currency,
        price_scale=spec.price_scale, timezone=VENUES[spec.exchange].tz, delay_min=spec.delay_min, tradable=spec.tradable,
    )


def de_num(value: float, digits: int = 2, signed: bool = False) -> str:
    """German number: 1.234,56 (with '+' / '−' when ``signed``)."""
    text = f"{abs(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if signed and round(value, digits) != 0:
        return ("+" if value > 0 else "−") + text
    return ("−" if value < 0 and round(value, digits) != 0 else "") + text


# ------------------------------------------------------------------------------------------------
# Market simulation
# ------------------------------------------------------------------------------------------------
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


@dataclass(slots=True)
class Series:
    """1-minute bars of one symbol (including bars prepared for the future) plus daily history."""

    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    session: np.ndarray  # session number per bar (consecutive)
    daily: pd.DataFrame  # index 'day' (session date 00:00 UTC as int), columns open, high, low, close, volume


def _minutes(spec: Spec, start: int, end: int) -> np.ndarray:
    segs = venue_segments(VENUES[spec.exchange], start, end, extended=False)
    parts = [np.arange(s - s % 60 + (60 if s % 60 else 0), e, 60, dtype=np.int64) for s, e, _ in segs]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _session_ids(spec: Spec, ts: np.ndarray) -> np.ndarray:
    if VENUES[spec.exchange].always_open or spec.exchange == "FX":
        return (ts // 86400).astype(np.int64)
    return np.concatenate(([0], np.cumsum(np.diff(ts) > 60))).astype(np.int64)


def build_series(spec: Spec, anchor: int) -> Series:
    """Deterministic random walk on the venue's trading minutes; the close at ``anchor`` equals ``spec.price``."""
    rng = np.random.default_rng(zlib.crc32(spec.symbol.encode()))
    ts = _minutes(spec, anchor - HISTORY_DAYS * 86400, anchor + FUTURE_DAYS * 86400)
    n = len(ts)
    session = _session_ids(spec, ts)
    kernel = 0.985 ** np.arange(240)
    drift = np.convolve(rng.standard_normal(n), kernel)[:n] / np.sqrt((kernel**2).sum())
    returns = spec.vol * (rng.standard_normal(n) + 0.22 * drift)
    starts = np.flatnonzero(np.diff(session, prepend=-1))
    gaps = np.zeros(n)
    gaps[starts[1:]] = spec.vol * 9 * rng.standard_normal(len(starts) - 1)
    log_close = np.cumsum(returns + gaps)
    idx_anchor = min(int(np.searchsorted(ts, anchor, side="right")) - 1, n - 1)
    close = spec.price * np.exp(log_close - log_close[max(idx_anchor, 0)])
    open_ = np.concatenate(([close[0] / math.exp(returns[0])], close[:-1])) * np.exp(gaps)
    wick = spec.vol * 0.7
    high = np.maximum(open_, close) * np.exp(wick * np.abs(rng.standard_normal(n)))
    low = np.minimum(open_, close) * np.exp(-wick * np.abs(rng.standard_normal(n)))
    if spec.volume > 0:
        pos = np.arange(n) - starts[np.searchsorted(starts, np.arange(n), side="right") - 1]
        length = np.diff(np.append(starts, n))[np.searchsorted(starts, np.arange(n), side="right") - 1]
        u_shape = 0.6 + 1.6 * ((pos / np.maximum(length, 1)) - 0.5) ** 2 * 4
        volume = spec.volume * u_shape * rng.lognormal(0, 0.45, n)
    else:
        volume = np.zeros(n)
    digits = 6 if spec.asset_class == "fx" else 4
    o, h, lo, c = (np.round(a, digits) for a in (open_, high, low, close))
    daily = _daily_history(spec, ts, o, h, lo, c, volume, session, rng)
    return Series(ts, o, h, lo, c, np.round(volume), session, daily)


def _session_day(spec: Spec, ts: int) -> int:
    """Session date 00:00 UTC (as epoch seconds) of a bar."""
    tz = ZoneInfo(VENUES[spec.exchange].tz)
    day = datetime.fromtimestamp(ts, tz).date()
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


def _daily_history(spec: Spec, ts: np.ndarray, o: np.ndarray, h: np.ndarray, lo: np.ndarray, c: np.ndarray, v: np.ndarray,
                   session: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Daily bars: aggregates of the intraday sessions (all prepared minutes), preceded by a daily random walk."""
    _, first_idx, counts = np.unique(session, return_index=True, return_counts=True)
    days = np.array([_session_day(spec, int(ts[i])) for i in first_idx])
    frame = pd.DataFrame({"day": days.repeat(counts), "o": o, "h": h, "l": lo, "c": c, "v": v})
    intraday = frame.groupby("day").agg(open=("o", "first"), high=("h", "max"), low=("l", "min"), close=("c", "last"), volume=("v", "sum"))
    first_day = int(intraday.index[0])
    weekdays = spec.exchange not in ("CRYPTO", "KRAKEN")
    hist_days: list[int] = []
    d = datetime.fromtimestamp(first_day, UTC).date()
    while len(hist_days) < DAILY_HISTORY:
        d -= timedelta(days=1)
        if weekdays and d.weekday() >= 5:
            continue
        hist_days.append(int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()))
    hist_days.reverse()
    dvol = spec.vol * math.sqrt(390) * 0.8
    rets = dvol * rng.standard_normal(DAILY_HISTORY) + 0.0003
    closes = intraday["open"].iloc[0] / np.exp(np.cumsum(rets[::-1]))[::-1]
    opens = closes / np.exp(dvol * 0.4 * rng.standard_normal(DAILY_HISTORY))
    highs = np.maximum(opens, closes) * np.exp(dvol * 0.5 * np.abs(rng.standard_normal(DAILY_HISTORY)))
    lows = np.minimum(opens, closes) * np.exp(-dvol * 0.5 * np.abs(rng.standard_normal(DAILY_HISTORY)))
    volume = spec.volume * 390 * rng.lognormal(0, 0.3, DAILY_HISTORY) if spec.volume else np.zeros(DAILY_HISTORY)
    history = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume}, index=pd.Index(hist_days, name="day"))
    return pd.concat([history, intraday]).round(6 if spec.asset_class == "fx" else 4)


def ema(values: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def resample(ts: np.ndarray, o, h, lo, c, v, seconds: int) -> tuple[np.ndarray, ...]:
    keys = ts // seconds * seconds
    starts = np.flatnonzero(np.diff(keys, prepend=keys[0] - 1))
    ends = np.append(starts[1:], len(ts)) - 1
    return (keys[starts], o[starts], np.maximum.reduceat(h, starts), np.minimum.reduceat(lo, starts), c[ends],
            np.add.reduceat(v, starts))


class Market:
    """All series plus the as-of views the API serves (closed bars of the delayed feed only)."""

    def __init__(self, clock: DemoClock, anchor: int) -> None:
        self.clock = clock
        self.series = {s.symbol: build_series(s, anchor) for s in UNIVERSE}

    def effective(self, symbol: str) -> float:
        """The instant the (possibly delayed) feed of ``symbol`` has reached."""
        return self.clock.now() - SPECS[symbol].delay_min * 60

    def closed_count(self, symbol: str, at: float | None = None) -> int:
        eff = self.effective(symbol) if at is None else at
        return int(np.searchsorted(self.series[symbol].ts, eff - 60, side="right"))

    def forming(self, symbol: str) -> tuple[int, float, float, float, float, float] | None:
        """The 1-minute bar in progress at the feed's current instant (None outside the session)."""
        s = self.series[symbol]
        eff = self.effective(symbol)
        i = self.closed_count(symbol)
        if i >= len(s.ts) or not (s.ts[i] <= eff < s.ts[i] + 60):
            return None
        frac = (eff - s.ts[i]) / 60
        wobble = SPECS[symbol].vol * s.o[i] * 0.4 * math.sin(eff * 1.7)
        close = s.o[i] + (s.c[i] - s.o[i]) * frac + wobble * (1 - frac)
        high = max(s.o[i], close, s.o[i] + (s.h[i] - s.o[i]) * frac)
        low = min(s.o[i], close, s.o[i] - (s.o[i] - s.l[i]) * frac)
        return int(s.ts[i]), float(s.o[i]), float(high), float(low), float(close), float(s.v[i] * frac)

    def last_price(self, symbol: str) -> float:
        bar = self.forming(symbol)
        if bar:
            return bar[4]
        i = self.closed_count(symbol)
        return float(self.series[symbol].c[max(i - 1, 0)])

    def quote(self, symbol: str) -> dto.QuoteOut:
        s = self.series[symbol]
        spec = SPECS[symbol]
        i = max(self.closed_count(symbol) - 1, 0)
        price = self.last_price(symbol)
        sess = s.session[i]
        first = int(np.searchsorted(s.session, sess))
        prev_close = float(s.c[first - 1]) if first > 0 else float(s.daily["close"].iloc[DAILY_HISTORY - 1])
        forming = self.forming(symbol)
        hi = max(float(s.h[first:i + 1].max()), forming[2] if forming else -math.inf)
        lo = min(float(s.l[first:i + 1].min()), forming[3] if forming else math.inf)
        state = venue_state(VENUES[spec.exchange], int(self.clock.now()))
        spread = price * (0.0001 if spec.asset_class != "fx" else 0.00002)
        digits = 6 if spec.asset_class == "fx" else 4
        return dto.QuoteOut(
            symbol=symbol, price=round(price, digits), ts=int(self.effective(symbol)) if forming else int(s.ts[i]) + 60,
            change_pct=round((price / prev_close - 1) * 100, 3), prev_close=prev_close, day_open=float(s.o[first]),
            day_high=hi, day_low=lo, day_volume=float(s.v[first:i + 1].sum()) or None,
            bid=round(price - spread, digits), ask=round(price + spread, digits),
            market_state={"open": "REGULAR", "pre": "PRE", "post": "POST"}.get(state.phase, "CLOSED"),
            source="stream" if forming else "quote",
        )

    def bars(self, symbol: str, tf: str, limit: int, before: int | None) -> dto.BarsOut:
        spec = SPECS[symbol]
        s = self.series[symbol]
        if tf == "1d":
            daily = self.daily_asof(symbol)
            ts = daily.index.to_numpy(dtype=np.int64)
            cols = [daily[k].to_numpy() for k in ("open", "high", "low", "close", "volume")]
            overlays_spec = {"ema20": 20, "ema50": 50, "ema200": 200}
            vwap = None
        else:
            n = self.closed_count(symbol)
            ts, o, h, lo, c, v = s.ts[:n], s.o[:n], s.h[:n], s.l[:n], s.c[:n], s.v[:n]
            sess = s.session[:n]
            if tf != "1m" and n:
                keys = ts // TF_SECONDS[tf] * TF_SECONDS[tf]
                starts = np.flatnonzero(np.diff(keys, prepend=keys[0] - 1))
                sess = sess[starts]
                ts, o, h, lo, c, v = resample(ts, o, h, lo, c, v, TF_SECONDS[tf])
            cols = [o, h, lo, c, v]
            overlays_spec = {"ema20": 20, "ema50": 50}
            vwap = _vwap(h, lo, c, v, sess) if spec.volume > 0 and n else None
        keep = np.flatnonzero(ts < before) if before else np.arange(len(ts))
        keep = keep[-limit:]
        digits = 6 if spec.asset_class == "fx" else 4
        bars = [(int(ts[k]), *(round(float(col[k]), digits) for col in cols[:4]), round(float(cols[4][k]), 0)) for k in keep]
        overlays: dict[str, list[tuple[int, float]]] = {}
        if len(ts):
            for name, span in overlays_spec.items():
                if len(ts) >= span // 2:
                    values = ema(cols[3], span)
                    overlays[name] = [(int(ts[k]), round(float(values[k]), digits)) for k in keep if k >= span // 2]
            if vwap is not None:
                overlays["vwap"] = [(int(ts[k]), round(float(vwap[k]), digits)) for k in keep]
        return dto.BarsOut(
            symbol=symbol, timeframe=tf, currency=spec.currency, price_scale=spec.price_scale, delay_min=spec.delay_min,
            bars=bars, overlays=overlays,
        )

    def daily_asof(self, symbol: str) -> pd.DataFrame:
        """Daily bars known now: completed sessions plus today's session built from the closed minutes only."""
        s = self.series[symbol]
        spec = SPECS[symbol]
        n = self.closed_count(symbol)
        if n == 0:
            return s.daily[s.daily.index < int(s.ts[0])]
        sess = s.session[n - 1]
        first = int(np.searchsorted(s.session, sess))
        today = _session_day(spec, int(s.ts[first]))
        past = s.daily[s.daily.index < today]
        partial = pd.DataFrame(
            {"open": [s.o[first]], "high": [s.h[first:n].max()], "low": [s.l[first:n].min()], "close": [s.c[n - 1]],
             "volume": [s.v[first:n].sum()]},
            index=pd.Index([today], name="day"),
        )
        return pd.concat([past, partial])

    def sparkline(self, symbol: str) -> list[tuple[int, float]]:
        s = self.series[symbol]
        n = self.closed_count(symbol)
        if n == 0:
            return []
        first = int(np.searchsorted(s.session, s.session[n - 1]))
        idx = np.arange(first, n)
        step = max(1, math.ceil(len(idx) / 80))
        picked = idx[::step]
        if picked[-1] != n - 1:
            picked = np.append(picked, n - 1)
        return [(int(s.ts[k]), float(s.c[k])) for k in picked]

    def price_at(self, symbol: str, ts: int) -> float:
        """Close of the last 1-minute bar opened before ``ts`` (daily close before the intraday history)."""
        s = self.series[symbol]
        i = int(np.searchsorted(s.ts, ts - 60, side="right")) - 1
        if i >= 0:
            return float(s.c[i])
        daily = s.daily[s.daily.index < ts - 86400]
        return float(daily["close"].iloc[-1]) if len(daily) else float(s.daily["open"].iloc[0])


def _vwap(h, lo, c, v, sess) -> np.ndarray:
    frame = pd.DataFrame({"pv": (h + lo + c) / 3 * v, "v": v, "s": sess})
    grouped = frame.groupby("s")
    return (grouped["pv"].cumsum() / grouped["v"].cumsum().replace(0, np.nan)).ffill().fillna(pd.Series(c)).to_numpy()


# ------------------------------------------------------------------------------------------------
# Signals
# ------------------------------------------------------------------------------------------------
RATIONALES: dict[str, dict[str, str]] = {
    "trend_pullback": {
        "LONG": "Aufwärtstrend im 15-Minuten-Chart (EMA 20 über EMA 50), Rücksetzer bis an die EMA 20 bei nachlassendem Verkaufsvolumen. RSI 14 bei 48 – Luft nach oben.",
        "SHORT": "Abwärtstrend im 15-Minuten-Chart (EMA 20 unter EMA 50), Erholung bis an die EMA 20 ohne Volumen. Die Gegenbewegung wirkt erschöpft.",
    },
    "breakout": {
        "LONG": "Ausbruch über das Hoch der letzten zwei Stunden bei 1,6-fachem Durchschnittsvolumen; der Tagestrend zeigt ebenfalls nach oben.",
        "SHORT": "Bruch des Tagestiefs mit steigendem Volumen; der Markt insgesamt schwächer.",
    },
    "mean_reversion": {
        "LONG": "Kurs 2,2 ATR unter dem VWAP, RSI 14 bei 24; ähnliche Überdehnungen bildeten sich zuletzt zurück.",
        "SHORT": "Kurs 2,3 ATR über dem VWAP, RSI 14 bei 76; ähnliche Überdehnungen bildeten sich zuletzt zurück.",
    },
    "news_momentum": {
        "LONG": "Positive Meldung (Relevanz 4 von 5) mit anziehendem Volumen; der Kurs hält sich über dem Eröffnungsniveau.",
        "SHORT": "Negative Meldung (Relevanz 4 von 5), Abgaben bei hohem Volumen; keine Erholung bis zum VWAP.",
    },
    "daily_trend": {
        "LONG": "Tagestrend intakt (Schlusskurs über der EMA 50 der Tageskerzen), Eröffnung über dem Vortageshoch.",
        "SHORT": "Tagestrend abwärts (Schlusskurs unter der EMA 50 der Tageskerzen), Eröffnung unter dem Vortagestief.",
    },
}
RISKS = (
    "Wichtige US-Konjunkturdaten in der nächsten Stunde; der Spread kann sich ausweiten.",
    "Der Rücksetzer könnte sich bis zum VWAP ausdehnen, bevor der Trend weiterläuft.",
    "Geringe Liquidität am Nachmittag; Stopps können über den Kurs rutschen.",
    "Die Meldung ist bereits teilweise eingepreist.",
    "Marktbreite schwach – die Bewegung trägt nur wenige Werte.",
)
COMPONENT_KEYS = ("trend", "momentum", "volume", "structure", "market", "news", "volatility")  # as in the engine


@dataclass(slots=True)
class SimResult:
    status: str
    entry_ts: int | None
    entry_price: float | None
    exit_ts: int | None
    exit_price: float | None
    outcome_r: float | None
    mfe_r: float | None
    mae_r: float | None
    status_reason: str


def _cost_r(spec: Spec, price: float, stop_dist: float) -> float:
    return DEFAULT_COSTS.round_trip_cost_r(to_instrument(spec), price, stop_dist)


def simulate(market: Market, spec: Spec, i: int, direction: str, stop_dist: float, target_dist: float,
             horizon_min: int) -> SimResult:
    """Follow a signal decided at the close of bar ``i`` through the closed bars that exist now.

    Like the real tracker, ``entry_ts`` and ``exit_ts`` are the OPEN times of the entry and exit bars.
    """
    s = market.series[spec.symbol]
    n = market.closed_count(spec.symbol)
    sign = 1 if direction == "LONG" else -1
    if i + 1 >= n:
        return SimResult("pending_entry", None, None, None, None, None, None, None, "")
    entry_ts, entry = int(s.ts[i + 1]), float(s.o[i + 1])
    stop, target = entry - sign * stop_dist, entry + sign * target_dist
    expires = entry_ts + horizon_min * 60
    cost = _cost_r(spec, entry, stop_dist)
    best = worst = 0.0
    for j in range(i + 1, n):
        hi, lo = float(s.h[j]), float(s.l[j])
        fav, adv = ((hi - entry), (entry - lo)) if sign > 0 else ((entry - lo), (hi - entry))
        best, worst = max(best, fav / stop_dist), max(worst, adv / stop_dist)
        if (sign > 0 and lo <= stop) or (sign < 0 and hi >= stop):
            return SimResult("stopped", entry_ts, entry, int(s.ts[j]), stop, -1 - cost, best, -worst, "stop")
        if (sign > 0 and hi >= target) or (sign < 0 and lo <= target):
            r = target_dist / stop_dist - cost
            return SimResult("target", entry_ts, entry, int(s.ts[j]), target, r, best, -worst, "target")
        if s.ts[j] + 60 >= expires:
            exit_price = float(s.c[j])
            r = (exit_price - entry) * sign / stop_dist - cost
            return SimResult("expired", entry_ts, entry, int(s.ts[j]), exit_price, r, best, -worst, "time_stop")
    return SimResult("active", entry_ts, entry, None, None, None, best, -worst, "")


class SignalFactory:
    """Creates internally consistent signals from the simulated prices."""

    def __init__(self, market: Market, rng: np.random.Generator) -> None:
        self.market = market
        self.rng = rng
        self.next_id = 101

    def create(self, spec: Spec, i: int, *, setup: str, direction: str, actionable: bool, score: float,
               verdict: str | None, shadow_reason: str = "", news: list[dto.NewsOut] | None = None) -> dto.SignalDetailOut:
        s = self.market.series[spec.symbol]
        rng = self.rng
        lo = max(0, i - 60)
        atr = float(np.mean(s.h[lo:i + 1] - s.l[lo:i + 1])) * math.sqrt(15) * 0.9
        stop_dist = round(1.5 * atr, 6)
        rr = float(rng.choice([1.5, 1.8, 2.0, 2.2, 2.5]))
        target_dist = round(rr * stop_dist, 6)
        horizon = int(rng.choice([90, 120, 180, 240]))
        created = int(s.ts[i]) + 60
        sim = simulate(self.market, spec, i, direction, stop_dist, target_dist, horizon)
        sign = 1 if direction == "LONG" else -1
        confidence = round(min(0.78, max(0.38, 0.3 + score * 0.55 + rng.normal(0, 0.03))), 3)
        conviction = None if verdict in (None, "unavailable") else round(float(rng.uniform(0.45, 0.85)), 2)
        ref = float(s.c[i])
        sid = self.next_id
        self.next_id += 1
        components = {k: round(float(np.clip(rng.normal(0.5 + (score - 0.5) * 1.4, 0.12), 0.05, 0.98)), 2) for k in COMPONENT_KEYS}
        rsi = round(float(rng.uniform(35, 70)), 1)
        return dto.SignalDetailOut(
            id=sid, created_ts=created, issued_ts=created + 40 if actionable else None, data_ts=created,
            symbol=spec.symbol, name=spec.name, region=spec.region, currency=spec.currency, delay_min=spec.delay_min,
            direction=direction, setup=setup, setup_label_de=SETUP_LABELS_DE.get(setup, setup), timeframe="15m",
            intraday=setup != "daily_trend", horizon_min=horizon, ref_price=ref, atr=round(atr, 6), stop_dist=stop_dist,
            target_dist=target_dist, rr=rr, score=round(score, 3), confidence=confidence,
            expected_r=round(confidence * rr - (1 - confidence) - 0.05, 3), llm_verdict=verdict, llm_conviction=conviction,
            rationale_de=RATIONALES[setup][direction], risks_de=str(rng.choice(RISKS)), actionable=actionable,
            shadow_reason=shadow_reason, status=sim.status, status_reason=sim.status_reason, entry_ts=sim.entry_ts,
            entry_price=sim.entry_price,
            stop_price=None if sim.entry_price is None else round(sim.entry_price - sign * stop_dist, 6),
            target_price=None if sim.entry_price is None else round(sim.entry_price + sign * target_dist, 6),
            expires_ts=(sim.entry_ts or created) + horizon * 60, exit_ts=sim.exit_ts, exit_price=sim.exit_price,
            outcome_r=None if sim.outcome_r is None else round(sim.outcome_r, 3),
            mfe_r=None if sim.mfe_r is None else round(sim.mfe_r, 2), mae_r=None if sim.mae_r is None else round(sim.mae_r, 2),
            mailed_ts=created + 45 if actionable else None, engine_version=ENGINE_VERSION,
            components=components,
            features=self._features(spec, i, ref, atr, rsi),
            news=news or [],
        )

    def _features(self, spec: Spec, i: int, ref: float, atr: float, rsi: float) -> dict[str, Any]:
        """A plausible subset of the engine's feature snapshot (same keys as analysis.features.FEATURE_COLUMNS)."""
        s = self.market.series[spec.symbol]
        rng = self.rng
        first = int(np.searchsorted(s.session, s.session[i]))
        last = int(np.searchsorted(s.session, s.session[i], side="right")) - 1
        day = datetime.fromtimestamp(_session_day(spec, int(s.ts[first])), UTC).date().isoformat()
        return {
            "price": ref, "session": day, "minutes_since_open": float(i - first + 1), "minutes_to_close": float(last - i),
            "in_session": 1.0, "atr_slow": round(atr, 6), "rsi_slow": rsi, "adx_slow": round(float(rng.uniform(14, 38)), 2),
            "ema_fast_s_slope": round(float(rng.normal(0, 0.15)), 4), "vwap": round(ref * (1 + float(rng.normal(0, 0.002))), 4),
            "rel_volume": round(float(rng.uniform(0.8, 2.1)), 3), "ret_60": round(float(rng.normal(0, 0.006)), 5),
            "rel_strength_60": round(float(rng.normal(0, 0.004)), 5), "atr_pct_rank": round(float(rng.uniform(0.1, 0.95)), 3),
        }


def build_signals(market: Market, news: list[dto.NewsOut], now: int) -> dict[int, dto.SignalDetailOut]:
    """About 40 signals: closed ones over the last days, open ones of every status, and shadow candidates."""
    rng = np.random.default_rng(7)
    factory = SignalFactory(market, rng)
    candidates = [s for s in UNIVERSE if s.asset_class in ("equity", "crypto")]
    setups = list(RATIONALES)
    made: list[dto.SignalDetailOut] = []
    wanted = {"target": 6, "stopped": 6, "expired": 3, "active": 3}
    shadow_wanted = 9
    news_by_symbol = {n.analysis.symbols[0].symbol: n for n in news if n.analysis and n.analysis.symbols}
    for _ in range(600):
        spec = candidates[int(rng.integers(len(candidates)))]
        s = market.series[spec.symbol]
        n = market.closed_count(spec.symbol)
        if n < 200:
            continue
        i = int(rng.integers(max(120, n - 2400), n - 2))
        if s.session[i] != s.session[min(i + 1, n - 1)]:
            continue
        trend = ema(s.c[max(0, i - 200):i + 1], 50)
        direction = "LONG" if trend[-1] >= trend[max(0, len(trend) - 20)] else "SHORT"
        if rng.random() < 0.25:
            direction = "SHORT" if direction == "LONG" else "LONG"
        setup = setups[int(rng.integers(len(setups)))]
        shadow = shadow_wanted > 0 and rng.random() < 0.3
        score = float(rng.uniform(0.53, 0.595)) if shadow else float(rng.uniform(0.6, 0.82))
        verdict = str(rng.choice(["confirm", "confirm", "reduce", "unavailable"]))
        reason = ""
        if shadow:
            veto = rng.random() < 0.35
            verdict = "veto" if veto else verdict
            reason = "llm_veto" if veto else "unter_schwelle"  # the engine's codes (analysis.engine.SHADOW_*)
        linked = [news_by_symbol[spec.symbol]] if spec.symbol in news_by_symbol and setup == "news_momentum" else []
        sig = factory.create(spec, i, setup=setup, direction=direction, actionable=not shadow, score=score, verdict=verdict,
                             shadow_reason=reason, news=linked)
        if shadow:
            if sig.status in ("target", "stopped", "expired", "active"):
                made.append(sig)
                shadow_wanted -= 1
            continue
        if wanted.get(sig.status, 0) > 0:
            wanted[sig.status] -= 1
            made.append(sig)
        if not any(wanted.values()) and shadow_wanted <= 0:
            break
    made.extend(_special_signals(market, factory, now))
    made.sort(key=lambda x: x.created_ts)
    out: dict[int, dto.SignalDetailOut] = {}
    for new_id, sig in enumerate(made, start=101):
        out[new_id] = sig.model_copy(update={"id": new_id})
    return out


def _special_signals(market: Market, factory: SignalFactory, now: int) -> list[dto.SignalDetailOut]:
    """One signal under review, one waiting for entry and two cancelled ones."""
    out: list[dto.SignalDetailOut] = []
    open_specs = [s for s in UNIVERSE if s.asset_class == "equity" and venue_state(VENUES[s.exchange], now).is_open]
    closed_specs = [s for s in UNIVERSE if s.asset_class == "equity" and not venue_state(VENUES[s.exchange], now).is_open]
    if open_specs:
        spec = open_specs[0]
        n = market.closed_count(spec.symbol)
        sig = factory.create(spec, n - 1, setup="breakout", direction="LONG", actionable=True, score=0.66, verdict=None)
        out.append(sig.model_copy(update={"status": "review", "issued_ts": None, "mailed_ts": None, "entry_ts": None,
                                          "entry_price": None, "stop_price": None, "target_price": None, "exit_ts": None,
                                          "exit_price": None, "outcome_r": None, "mfe_r": None, "mae_r": None}))
    if closed_specs:
        spec = closed_specs[0]
        n = market.closed_count(spec.symbol)
        sig = factory.create(spec, n - 1, setup="daily_trend", direction="LONG", actionable=True, score=0.64, verdict="confirm")
        out.append(sig.model_copy(update={"status": "pending_entry", "status_reason": ""}))
        spec2 = closed_specs[min(3, len(closed_specs) - 1)]
        n2 = market.closed_count(spec2.symbol)
        sig2 = factory.create(spec2, n2 - 30, setup="trend_pullback", direction="SHORT", actionable=True, score=0.61, verdict="reduce")
        out.append(sig2.model_copy(update={"status": "cancelled", "status_reason": "kein_einstieg",
                                           "entry_ts": None, "entry_price": None, "stop_price": None, "target_price": None,
                                           "exit_ts": None, "exit_price": None, "outcome_r": None, "mfe_r": None, "mae_r": None}))
    return out


def wilson(wins: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = 1.96
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def stat_block(results: list[float]) -> dto.StatBlock:
    n = len(results)
    if n == 0:
        return dto.StatBlock()
    arr = np.array(results)
    wins = int((arr > 0).sum())
    lo, hi = wilson(wins, n)
    rng = np.random.default_rng(n)
    boots = rng.choice(arr, size=(400, n)).mean(axis=1) if n > 1 else arr
    gains, losses = arr[arr > 0].sum(), -arr[arr < 0].sum()
    return dto.StatBlock(
        n=n, wins=wins, win_rate=wins / n, win_rate_lo=lo, win_rate_hi=hi, avg_r=float(arr.mean()),
        avg_r_lo=float(np.percentile(boots, 2.5)), avg_r_hi=float(np.percentile(boots, 97.5)), total_r=float(arr.sum()),
        profit_factor=float(gains / losses) if losses > 0 else None,
    )


def group_stats(items: list[tuple[str, str, float]]) -> list[dto.GroupStat]:
    groups: dict[str, tuple[str, list[float]]] = {}
    for key, label, r in items:
        groups.setdefault(key, (label, []))[1].append(r)
    return [dto.GroupStat(key=k, label_de=label, stats=stat_block(rs)) for k, (label, rs) in sorted(groups.items())]


def calibration(pairs: list[tuple[float, float]]) -> list[dto.CalibrationBin]:
    bins = []
    for lo in (0.3, 0.4, 0.5, 0.6, 0.7):
        inside = [(p, r) for p, r in pairs if lo <= p < lo + 0.1]
        observed = sum(r > 0 for _, r in inside) / len(inside) if inside else None
        predicted = float(np.mean([p for p, _ in inside])) if inside else lo + 0.05
        bins.append(dto.CalibrationBin(lo=lo, hi=lo + 0.1, n=len(inside), predicted=predicted, observed=observed))
    return bins


VERDICT_LABELS = {"confirm": "bestätigt", "reduce": "reduziert", "veto": "Veto", "unavailable": "ohne Prüfung", "none": "ohne Prüfung"}


def signal_stats(signals: list[dto.SignalDetailOut]) -> dto.SignalStatsOut:
    closed = [s for s in signals if s.status in ("target", "stopped", "expired") and s.outcome_r is not None]
    act = [s for s in closed if s.actionable]
    shadow = [s for s in closed if not s.actionable]
    n = len(act)
    note = (
        f"Erst {n} abgeschlossene Empfehlungen: Die Intervalle sind noch sehr breit. Ob die Engine besser als Zufall ist, "
        "lässt sich frühestens nach etwa 100 Signalen sagen – bis dahin bitte keine Schlüsse aus der Trefferquote ziehen."
    )
    return dto.SignalStatsOut(
        engine_version=ENGINE_VERSION, actionable=stat_block([s.outcome_r for s in act]),
        shadow=stat_block([s.outcome_r for s in shadow]),
        by_setup=group_stats([(s.setup, s.setup_label_de, s.outcome_r) for s in act]),
        by_region=group_stats([(s.region, REGION_LABELS[s.region], s.outcome_r) for s in act]),
        by_direction=group_stats([(s.direction, "Long" if s.direction == "LONG" else "Short", s.outcome_r) for s in act]),
        by_llm_verdict=group_stats([(s.llm_verdict or "none", VERDICT_LABELS[s.llm_verdict or "none"], s.outcome_r) for s in act]),
        calibration=calibration([(s.confidence, s.outcome_r) for s in act]), note_de=note,
    )


# ------------------------------------------------------------------------------------------------
# News, feeds, chart analyses
# ------------------------------------------------------------------------------------------------
FEEDS: tuple[tuple[str, str, str, str, str], ...] = (
    ("finanzen_net_news", "finanzen.net News", "DE", "de", "markets"),
    ("hb_unternehmen", "Handelsblatt Unternehmen", "DE", "de", "companies"),
    ("fn_adhoc", "finanznachrichten.de Ad-hoc", "DE", "de", "press_release"),
    ("ecb_press", "ECB press releases", "EU", "en", "macro"),
    ("cash_ch", "cash.ch", "EU", "de", "markets"),
    ("cnbc_earnings", "CNBC Earnings", "US", "en", "companies"),
    ("bls_cpi", "BLS CPI", "US", "en", "macro"),
    ("scmp_business", "SCMP Business", "ASIA", "en", "markets"),
    ("et_markets", "Economic Times Markets", "ASIA", "en", "markets"),
    ("ft_markets", "FT Markets", "GLOBAL", "en", "markets"),
    ("bbg_markets", "Bloomberg Markets", "GLOBAL", "en", "markets"),
    ("coindesk", "CoinDesk", "GLOBAL", "en", "crypto"),
    ("forexlive", "ForexLive / investingLive", "GLOBAL", "en", "macro"),
    ("nikkei_asia", "Nikkei Asia Markets", "ASIA", "en", "markets"),
)
FAILING_FEED = "nikkei_asia"
FEED_NAMES = {f[0]: f[1] for f in FEEDS}

# (minutes ago, source, lang, title, summary_de, relevance, sentiment, category, horizon, impacts, status)
NEWS_SEED: tuple[tuple[Any, ...], ...] = (
    (3, "cnbc_earnings", "en", "Chipmaker demand outlook lifted as data-centre orders keep climbing",
     "Ein Branchenbericht hebt die Nachfrageprognose für Rechenzentrums-Chips an; Auftragseingänge steigen weiter. Positiv für NVIDIA, leicht positiv für ASML.",
     4, 0.55, "guidance", "days", {"NVDA": 0.6, "ASML.AS": 0.3}, "analyzed"),
    (6, "finanzen_net_news", "de", "Rüstungswerte gefragt: Neue Rahmenverträge in Aussicht",
     "Mehrere europäische Regierungen stellen größere Rahmenverträge in Aussicht. Rheinmetall profitiert laut Analysten am stärksten.",
     4, 0.48, "product", "days", {"RHM.DE": 0.55}, "analyzed"),
    (9, "bls_cpi", "en", "Consumer prices rise 0.2% in October, core inflation steady",
     "US-Verbraucherpreise steigen im Oktober um 0,2 %; die Kernrate bleibt stabil. Keine Überraschung – Zinssenkungserwartungen unverändert.",
     3, 0.1, "macro", "intraday", {"^GSPC": 0.15, "^NDX": 0.1}, "analyzed"),
    (12, "hb_unternehmen", "de", "SAP verlängert Partnerschaft mit großem Cloud-Anbieter",
     "SAP verlängert eine Cloud-Partnerschaft um fünf Jahre. Finanzielle Details nennt der Konzern nicht; Analysten sehen einen leicht positiven Effekt.",
     3, 0.3, "product", "weeks", {"SAP.DE": 0.35}, "analyzed"),
    (15, "coindesk", "en", "Bitcoin ETF inflows slow for third straight session",
     "Die Zuflüsse in Bitcoin-ETFs schwächen sich den dritten Handelstag in Folge ab. Kurzfristig leicht belastend für Bitcoin und Ethereum.",
     3, -0.35, "other", "intraday", {"BTC-EUR": -0.4, "ETH-EUR": -0.25}, "analyzed"),
    (2, "ft_markets", "en", "European banks rally as bond yields edge higher", "", 0, 0, "", "", {}, "queued"),
    (1, "cnbc_earnings", "en", "Retailer cuts full-year forecast, shares slide in pre-market", "", 0, 0, "", "", {}, "queued"),
    (21, "fn_adhoc", "de", "Ad-hoc: Infineon passt Umsatzprognose für das Gesamtjahr an",
     "Infineon senkt die Umsatzprognose leicht wegen schwächerer Nachfrage aus der Autoindustrie; die Marge soll stabil bleiben.",
     5, -0.62, "guidance", "days", {"IFX.DE": -0.7}, "analyzed"),
    (28, "bbg_markets", "en", "Tesla delivery estimates trimmed by two brokers",
     "Zwei Broker senken ihre Auslieferungsschätzungen für Tesla im vierten Quartal. Leicht negativ für die Aktie.",
     3, -0.4, "analyst", "days", {"TSLA": -0.45}, "analyzed"),
    (35, "ecb_press", "en", "ECB publishes account of monetary policy meeting",
     "Die EZB veröffentlicht das Protokoll ihrer letzten Sitzung: Mehrheit sieht keinen Grund für schnelle weitere Zinsschritte.",
     2, 0.05, "macro", "weeks", {"^GDAXI": 0.05}, "analyzed"),
    (44, "scmp_business", "en", "Tencent unveils new AI assistant for enterprise customers",
     "Tencent stellt einen KI-Assistenten für Unternehmenskunden vor. Analysten erwarten erst mittelfristig Umsatzbeiträge.",
     3, 0.32, "product", "weeks", {"0700.HK": 0.35}, "analyzed"),
    (52, "et_markets", "en", "Asian chip stocks mixed after export data", "", 0, 0, "", "", {}, "skipped"),
    (61, "cash_ch", "de", "Nestlé: Neuer Konzernchef bestätigt Mittelfristziele",
     "Der neue Nestlé-Chef bestätigt die Mittelfristziele und kündigt ein Sparprogramm an. Reaktion der Aktie verhalten positiv.",
     3, 0.25, "management", "weeks", {"NESN.SW": 0.3}, "analyzed"),
    (75, "cnbc_earnings", "en", "Apple supplier signals stronger holiday quarter",
     "Ein wichtiger Zulieferer signalisiert ein stärkeres Weihnachtsquartal. Positiv für Apple.",
     4, 0.5, "earnings", "days", {"AAPL": 0.5}, "analyzed"),
    (88, "forexlive", "en", "Dollar steadies ahead of US data", "", 0, 0, "", "", {}, "skipped"),
    (96, "hb_unternehmen", "de", "Siemens gewinnt Großauftrag für Bahntechnik",
     "Siemens Mobility erhält einen Großauftrag über Züge und Signaltechnik. Positiv, aber im Konzernmaßstab überschaubar.",
     3, 0.4, "product", "weeks", {"SIE.DE": 0.4}, "analyzed"),
    (118, "bbg_markets", "en", "Meta faces new EU inquiry over advertising model",
     "Die EU-Kommission leitet eine neue Prüfung des Werbemodells von Meta ein. Ausgang offen, kurzfristig belastend.",
     4, -0.45, "regulation_legal", "weeks", {"META": -0.5}, "analyzed"),
    (140, "scmp_business", "en", "Alibaba cloud unit posts faster growth",
     "Die Cloud-Sparte von Alibaba wächst schneller als erwartet. Positiv für die Aktie.",
     4, 0.52, "earnings", "days", {"9988.HK": 0.55}, "analyzed"),
    (170, "finanzen_net_news", "de", "Allianz: Analysten erhöhen Kursziel nach Kapitalmarkttag",
     "Mehrere Analysten erhöhen nach dem Kapitalmarkttag ihr Kursziel für die Allianz; die Dividendenpolitik überzeugt.",
     3, 0.38, "analyst", "weeks", {"ALV.DE": 0.4}, "analyzed"),
    (210, "ft_markets", "en", "Shell to accelerate buybacks after trading update",
     "Shell beschleunigt nach einem Zwischenbericht die Aktienrückkäufe. Positiv für die Aktie.",
     4, 0.45, "earnings", "days", {"SHEL.L": 0.5}, "analyzed"),
    (260, "et_markets", "en", "Samsung Electronics wins memory supply deal",
     "Samsung Electronics schließt einen großen Liefervertrag für Speicherchips ab.",
     3, 0.42, "product", "days", {"005930.KS": 0.45}, "analyzed"),
    (330, "bbg_markets", "en", "Toyota raises output plan for hybrids",
     "Toyota erhöht den Produktionsplan für Hybridfahrzeuge. Leicht positiv.",
     3, 0.3, "product", "weeks", {"7203.T": 0.35}, "analyzed"),
    (420, "coindesk", "en", "Ethereum network upgrade date confirmed",
     "Der Termin für das nächste Ethereum-Netzwerk-Upgrade steht fest. Leicht positiv für Ethereum.",
     2, 0.2, "product", "weeks", {"ETH-EUR": 0.25}, "analyzed"),
    (520, "cnbc_earnings", "en", "Microsoft expands AI partnership in Europe",
     "Microsoft erweitert seine KI-Partnerschaften in Europa; Investitionen in Rechenzentren steigen.",
     3, 0.35, "product", "weeks", {"MSFT": 0.4}, "analyzed"),
    (640, "hb_unternehmen", "de", "LVMH: Nachfrage aus China bleibt schwach",
     "LVMH meldet anhaltend schwache Nachfrage aus China; die Aktie gerät unter Druck.",
     4, -0.5, "earnings", "days", {"MC.PA": -0.55}, "analyzed"),
    (800, "ft_markets", "en", "AstraZeneca drug wins approval in key market",
     "Ein Medikament von AstraZeneca erhält die Zulassung in einem wichtigen Markt. Positiv.",
     4, 0.55, "regulation_legal", "weeks", {"AZN.L": 0.6}, "analyzed"),
)


def build_news(now: int) -> list[dto.NewsOut]:
    """News newest first; ids descend with age."""
    items = []
    seed = sorted(NEWS_SEED, key=lambda row: row[0])
    for k, (ago, source, lang, title, summary, rel, sent, cat, horizon, impacts, status) in enumerate(seed):
        fetched = now - ago * 60
        analysis = None
        if status == "analyzed":
            analysis = dto.NewsAnalysisOut(
                analyzed_ts=fetched + 40, relevance=rel, sentiment=sent, confidence=0.7, category=cat, horizon=horizon,
                summary_de=summary,
                symbols=[dto.NewsImpactOut(symbol=s, sentiment=v) for s, v in sorted(impacts.items(), key=lambda x: -abs(x[1]))],
            )
        items.append(dto.NewsOut(
            id=900 - k, source=source, source_name=FEED_NAMES[source], url=f"https://example.org/demo/news/{900 - k}",
            title=title, summary="", lang=lang, published_ts=fetched - 120, fetched_ts=fetched, status=status,
            hinted_symbols=sorted(impacts), analysis=analysis,
        ))
    return sorted(items, key=lambda n: -n.id)


LIVE_NEWS: tuple[tuple[str, str, str, str, dict[str, float], int, float, str], ...] = (
    ("finanzen_net_news", "de", "Deutsche Industrieaufträge überraschend kräftig",
     "Die Auftragseingänge der deutschen Industrie steigen stärker als erwartet. Positiv für zyklische DAX-Werte.",
     {"^GDAXI": 0.35, "SIE.DE": 0.3}, 3, 0.35, "macro"),
    ("cnbc_earnings", "en", "Amazon web services wins multi-year government contract",
     "Die Cloud-Sparte von Amazon gewinnt einen mehrjährigen Regierungsauftrag. Positiv für Amazon.",
     {"AMZN": 0.5}, 4, 0.5, "product"),
    ("coindesk", "en", "Large Bitcoin transfer to exchange sparks selling worries",
     "Eine große Bitcoin-Überweisung an eine Börse weckt Sorgen vor Verkäufen. Kurzfristig belastend.",
     {"BTC-EUR": -0.4}, 3, -0.4, "other"),
    ("bbg_markets", "en", "JPMorgan raises net interest income guidance",
     "JPMorgan hebt die Prognose für den Zinsüberschuss an. Positiv für die Aktie.",
     {"JPM": 0.5}, 4, 0.5, "guidance"),
)


def build_feeds(now: int) -> list[dto.FeedStateOut]:
    out = []
    for k, (fid, name, region, lang, cat) in enumerate(FEEDS):
        failing = fid == FAILING_FEED
        out.append(dto.FeedStateOut(
            feed_id=fid, name=name, region=region, language=lang, category=cat,
            last_ok_ts=now - (5 * 3600 if failing else 40 + 7 * k), last_error="HTTP 503 – Dienst nicht verfügbar" if failing else "",
            fail_count=6 if failing else 0, items_total=120 + 37 * k,
        ))
    return out


CHART_COMMENTS = {
    "bullish": "Höhere Tiefs seit dem Vormittag, der Kurs läuft an der oberen Bandbreite entlang. Solange die Unterstützung hält, bleibt das Bild freundlich.",
    "bearish": "Die Erholungen werden kürzer, mehrere Ausbruchsversuche sind gescheitert. Ein Bruch der Unterstützung würde weitere Abgaben öffnen.",
    "neutral": "Seitwärtsphase ohne klare Richtung; Volumen rückläufig. Ein Ausbruch aus der Spanne steht noch aus.",
}


def build_chart_analyses(market: Market, now: int) -> dict[str, list[dto.ChartAnalysisOut]]:
    rng = np.random.default_rng(11)
    out: dict[str, list[dto.ChartAnalysisOut]] = {}
    next_id = 501
    for symbol in ("NVDA", "SAP.DE", "BTC-EUR", "AAPL", "7203.T", "RHM.DE", "TSLA", "ASML.AS"):
        items = []
        for k in range(2):
            bias = str(rng.choice(["bullish", "bearish", "neutral"]))
            price = market.last_price(symbol)
            items.append(dto.ChartAnalysisOut(
                id=next_id, symbol=symbol, ts=now - (k * 90 + int(rng.integers(5, 40))) * 60, timeframe="15m",
                model="qwen3.5:9b", bias=bias, strength=int(rng.integers(2, 5)),
                pattern=str(rng.choice(["Aufwärtstrendkanal", "Doppelboden", "Flagge", "Seitwärtsrange", ""])),
                support=round(price * 0.985, 2), resistance=round(price * 1.018, 2), comment_de=CHART_COMMENTS[bias],
                latency_ms=int(rng.integers(6000, 14000)),
            ))
            next_id += 1
        out[symbol] = items
    return out


def build_analysis(market: Market, symbol: str, chart: dto.ChartAnalysisOut | None, news_score: float | None) -> dto.AnalysisOut:
    """Technical readings computed from the closed 15-minute bars."""
    spec = SPECS[symbol]
    bars = market.bars(symbol, "15m", 400, None).bars
    if len(bars) < 30:
        return dto.AnalysisOut(symbol=symbol, available=False, note_de="Noch zu wenig Kursdaten für eine Einschätzung (mindestens 30 Viertelstundenkerzen).")
    arr = np.array(bars, dtype=float)
    c, h, lo, v = arr[:, 4], arr[:, 2], arr[:, 3], arr[:, 5]
    e20, e50 = ema(c, 20), ema(c, 50)
    delta = np.diff(c)
    gain = pd.Series(np.clip(delta, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    loss = pd.Series(np.clip(-delta, 0, None)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 100.0
    atr = float(np.mean((h - lo)[-14:]))
    quote = market.quote(symbol)
    daily = market.bars(symbol, "1d", 80, None).bars
    dc = np.array([b[4] for b in daily])
    d20, d50 = ema(dc, 20)[-1], ema(dc, 50)[-1]
    vwap_line = market.bars(symbol, "15m", 1, None).overlays.get("vwap")
    vwap = vwap_line[-1][1] if vwap_line else None
    vol_ratio = float(v[-4:].mean() / v[-80:].mean()) if v[-80:].mean() > 0 else None
    readings = [
        dto.ReadingOut(key="trend_15m", label_de="Trend 15 Min", value_de="aufwärts (EMA 20 über EMA 50)" if e20[-1] > e50[-1] else "abwärts (EMA 20 unter EMA 50)",
                       stance="bullish" if e20[-1] > e50[-1] else "bearish"),
        dto.ReadingOut(key="trend_1d", label_de="Tagestrend", value_de="aufwärts" if d20 > d50 else "abwärts", stance="bullish" if d20 > d50 else "bearish"),
        dto.ReadingOut(key="rsi14", label_de="RSI 14", value_de=de_num(rsi, 1) + (" (überkauft)" if rsi > 70 else " (überverkauft)" if rsi < 30 else ""),
                       stance="bearish" if rsi > 70 else "bullish" if rsi < 30 else "neutral"),
        dto.ReadingOut(key="atr", label_de="Schwankung (ATR 14)", value_de=f"{de_num(atr / c[-1] * 100, 2)} %"),
        dto.ReadingOut(key="day_change", label_de="Tagesveränderung", value_de=f"{de_num(quote.change_pct or 0, 2, signed=True)} %",
                       stance="bullish" if (quote.change_pct or 0) > 0.3 else "bearish" if (quote.change_pct or 0) < -0.3 else "neutral"),
    ]
    if vwap:
        dist = (c[-1] - vwap) / atr if atr else 0.0
        readings.append(dto.ReadingOut(key="vwap", label_de="Abstand zum VWAP", value_de=f"{de_num(dist, 1, signed=True)} ATR",
                                       stance="bullish" if dist > 0.3 else "bearish" if dist < -0.3 else "neutral"))
    if vol_ratio:
        readings.append(dto.ReadingOut(key="volume", label_de="Volumen", value_de=f"{de_num(vol_ratio, 1)}× Durchschnitt"))
    if news_score is not None:
        word = "positiv" if news_score > 0.15 else "negativ" if news_score < -0.15 else "neutral"
        readings.append(dto.ReadingOut(key="news", label_de="Nachrichtenlage", value_de=f"{word} ({de_num(news_score, 2, signed=True)})",
                                       stance="bullish" if news_score > 0.15 else "bearish" if news_score < -0.15 else "neutral"))
    levels = [
        dto.LevelOut(price=float(h[-40:].max()), kind="resistance", label_de="Widerstand (Hoch 10 Std)"),
        dto.LevelOut(price=float(lo[-40:].min()), kind="support", label_de="Unterstützung (Tief 10 Std)"),
    ]
    if quote.prev_close:
        levels.append(dto.LevelOut(price=quote.prev_close, kind="prev_close", label_de="Vortagesschluss"))
    if quote.day_high:
        levels.append(dto.LevelOut(price=quote.day_high, kind="day_high", label_de="Tageshoch"))
        levels.append(dto.LevelOut(price=quote.day_low, kind="day_low", label_de="Tagestief"))
    if vwap:
        levels.append(dto.LevelOut(price=vwap, kind="vwap", label_de="VWAP"))
    return dto.AnalysisOut(
        symbol=symbol, as_of=int(market.clock.now()) // 60 * 60, data_ts=int(bars[-1][0]) + 900, available=True,
        note_de=f"Daten {spec.delay_min} Min verzögert" if spec.delay_min else "",
        features={"rsi14": round(rsi, 2), "atr": round(atr, 4), "ema20": round(float(e20[-1]), 4), "ema50": round(float(e50[-1]), 4)},
        readings=readings, levels=levels, news_score=news_score, chart_analysis=chart,
    )


# ------------------------------------------------------------------------------------------------
# Paper portfolios
# ------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Pos:
    id: int
    key: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    entry_fx: float
    opened_ts: int
    stop_price: float | None = None
    target_price: float | None = None
    signal_id: int | None = None
    status: str = "open"
    closed_ts: int | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    fees_base: float = 0.0
    realized_pnl: float | None = None


@dataclass(slots=True)
class Ord:
    id: int
    key: str
    created_ts: int
    symbol: str
    intent: str
    side: str
    origin: str
    status: str
    qty: float | None = None
    notional_base: float | None = None
    ref_price: float | None = None
    stop_price: float | None = None
    target_price: float | None = None
    status_reason: str = ""
    filled_ts: int | None = None
    fill_price: float | None = None
    fill_fx: float | None = None
    fee_base: float | None = None
    signal_id: int | None = None
    position_id: int | None = None


@dataclass(slots=True)
class Book:
    key: str
    name: str
    start_cash: float
    reset_ts: int
    cash: float = 0.0
    fees_paid: float = 0.0
    curve: list[tuple[int, float]] = field(default_factory=list)


def unit_value_base(spec: Spec, price: float) -> float:
    return price * spec.price_scale * EUR_PER_UNIT[spec.currency]


class Broker:
    """In-memory paper broker for 'ai' and 'me' with the same fee and slippage rules."""

    FEE = DEFAULT_COSTS.fee_flat_base

    def __init__(self, market: Market, signals: dict[int, dto.SignalDetailOut], now: int) -> None:
        self.market = market
        self.positions: dict[int, Pos] = {}
        self.orders: dict[int, Ord] = {}
        self._pid = 1
        self._oid = 1
        reset = now - 24 * 86400
        self.books = {"ai": Book("ai", "KI-Depot", START_CASH, reset, START_CASH), "me": Book("me", "Mein Depot", START_CASH, reset, START_CASH)}
        self._seed_ai(signals)
        self._seed_me(signals, now)
        for book in self.books.values():
            book.curve = self._synthetic_curve(book, now)

    # -- helpers ------------------------------------------------------------------------------
    def _new_pos(self, **kw: Any) -> Pos:
        pos = Pos(id=self._pid, **kw)
        self._pid += 1
        self.positions[pos.id] = pos
        return pos

    def _new_order(self, **kw: Any) -> Ord:
        order = Ord(id=self._oid, **kw)
        self._oid += 1
        self.orders[order.id] = order
        return order

    def _close(self, pos: Pos, price: float, ts: int, reason: str) -> None:
        spec = SPECS[pos.symbol]
        sign = 1 if pos.side == "LONG" else -1
        fx = EUR_PER_UNIT[spec.currency]
        pnl = (price - pos.entry_price) * sign * pos.qty * spec.price_scale * fx - self.FEE
        pos.status, pos.closed_ts, pos.exit_price, pos.exit_reason = "closed", ts, price, reason
        pos.fees_base += self.FEE
        pos.realized_pnl = pnl
        book = self.books[pos.key]
        book.cash += pnl
        book.fees_paid += self.FEE

    def _open(self, key: str, symbol: str, side: str, qty: float, price: float, ts: int, **kw: Any) -> Pos:
        spec = SPECS[symbol]
        book = self.books[key]
        book.cash -= self.FEE
        book.fees_paid += self.FEE
        return self._new_pos(key=key, symbol=symbol, side=side, qty=qty, entry_price=price, entry_fx=EUR_PER_UNIT[spec.currency],
                             opened_ts=ts, fees_base=self.FEE, **kw)

    def _seed_ai(self, signals: dict[int, dto.SignalDetailOut]) -> None:
        equity = START_CASH
        for sig in sorted(signals.values(), key=lambda s: s.created_ts):
            if not sig.actionable:
                continue
            spec = SPECS[sig.symbol]
            risk_units = sig.stop_dist * spec.price_scale * EUR_PER_UNIT[spec.currency]
            qty = equity * 0.005 / risk_units
            cap = equity * 0.10 / unit_value_base(spec, sig.ref_price)
            qty = min(qty, cap)
            qty = round(qty, 4) if spec.asset_class == "crypto" else float(max(1, math.floor(qty)))
            if sig.status == "pending_entry":
                self._new_order(key="ai", created_ts=sig.issued_ts or sig.created_ts, symbol=sig.symbol, intent="open", side=sig.direction,
                                origin="signal", status="pending", qty=qty, ref_price=sig.ref_price,
                                status_reason="Markt geschlossen – Ausführung zur Eröffnung", signal_id=sig.id)
                continue
            if sig.entry_ts is None or sig.entry_price is None:
                continue
            pos = self._open("ai", sig.symbol, sig.direction, qty, sig.entry_price, sig.entry_ts, stop_price=sig.stop_price,
                             target_price=sig.target_price, signal_id=sig.id)
            self._new_order(key="ai", created_ts=sig.created_ts + 40, symbol=sig.symbol, intent="open", side=sig.direction, origin="signal",
                            status="filled", qty=qty, ref_price=sig.ref_price, stop_price=sig.stop_price, target_price=sig.target_price,
                            filled_ts=sig.entry_ts, fill_price=sig.entry_price, fill_fx=pos.entry_fx, fee_base=self.FEE, signal_id=sig.id,
                            position_id=pos.id)
            if sig.exit_ts and sig.exit_price is not None:
                reason = {"target": "target", "stopped": "stop", "expired": "time_stop"}[sig.status]
                self._close(pos, sig.exit_price, sig.exit_ts, reason)
                self._new_order(key="ai", created_ts=sig.exit_ts - 60, symbol=sig.symbol, intent="close", side=sig.direction,
                                origin=reason, status="filled", qty=qty, filled_ts=sig.exit_ts, fill_price=sig.exit_price,
                                fill_fx=pos.entry_fx, fee_base=self.FEE, signal_id=sig.id, position_id=pos.id)
                equity += pos.realized_pnl or 0.0

    def _seed_me(self, signals: dict[int, dto.SignalDetailOut], now: int) -> None:
        m = self.market

        def price_ago(symbol: str, minutes: int) -> float:
            return m.price_at(symbol, now - minutes * 60)

        trades = (("ALV.DE", "LONG", 20, 6 * 1440, 5 * 1440, "target"), ("TSLA", "SHORT", 15, 4 * 1440, 3 * 1440 + 300, "stop"),
                  ("MSFT", "LONG", 12, 3 * 1440, 2 * 1440, "manual"), ("ETH-EUR", "LONG", 2.5, 2 * 1440, 1440 + 200, "manual"),
                  ("SHEL.L", "LONG", 300, 1440 + 600, 1440 + 100, "stop"))
        for symbol, side, qty, opened_ago, closed_ago, reason in trades:
            pos = self._open("me", symbol, side, qty, price_ago(symbol, opened_ago), now - opened_ago * 60)
            self._close(pos, price_ago(symbol, closed_ago), now - closed_ago * 60, reason)
        sap = self._open("me", "SAP.DE", "LONG", 40, price_ago("SAP.DE", 2 * 1440), now - 2 * 1440 * 60)
        sap.stop_price = round(sap.entry_price * 0.96, 2)
        nv = self._open("me", "NVDA", "SHORT", 60, price_ago("NVDA", 1440), now - 1440 * 60)
        nv.stop_price, nv.target_price = round(nv.entry_price * 1.035, 2), round(nv.entry_price * 0.93, 2)
        follow = next((s for s in signals.values() if s.status == "active" and s.actionable), None)
        if follow and follow.entry_price:
            spec = SPECS[follow.symbol]
            qty = round(2500 / unit_value_base(spec, follow.entry_price), 4 if spec.asset_class == "crypto" else 0) or 1
            self._open("me", follow.symbol, follow.direction, float(qty), follow.entry_price, follow.entry_ts or now,
                       stop_price=follow.stop_price, target_price=follow.target_price, signal_id=follow.id)
        for pos in list(self.positions.values()):
            if pos.key == "me":
                self._new_order(key="me", created_ts=pos.opened_ts - 50, symbol=pos.symbol, intent="open", side=pos.side, origin="manual",
                                status="filled", qty=pos.qty, ref_price=pos.entry_price, filled_ts=pos.opened_ts, fill_price=pos.entry_price,
                                fill_fx=pos.entry_fx, fee_base=self.FEE, signal_id=pos.signal_id, position_id=pos.id)
        self._new_order(key="me", created_ts=now - 3600, symbol="7203.T", intent="open", side="LONG", origin="manual", status="pending",
                        qty=300, ref_price=m.last_price("7203.T"), status_reason="Markt geschlossen – Ausführung zur Eröffnung in Tokio")
        self._new_order(key="me", created_ts=now - 2 * 86400, symbol="AMZN", intent="open", side="LONG", origin="manual",
                        status="cancelled", notional_base=5000, status_reason="vom Nutzer storniert")
        self._new_order(key="me", created_ts=now - 3 * 86400, symbol="RHM.DE", intent="open", side="LONG", origin="manual",
                        status="rejected", notional_base=150_000, status_reason="Kaufkraft reicht nicht: benötigt 150.000 EUR")

    def _synthetic_curve(self, book: Book, now: int) -> list[tuple[int, float]]:
        """Equity every 30 minutes since the reset: a Brownian bridge to today's equity."""
        times = np.arange(book.reset_ts, now, 1800, dtype=np.int64)
        rng = np.random.default_rng(zlib.crc32(book.key.encode()))
        steps = rng.normal(0, 140 if book.key == "ai" else 190, len(times))
        weekend = np.array([datetime.fromtimestamp(int(t), UTC).weekday() >= 5 for t in times])
        steps[weekend] = 0
        walk = np.cumsum(steps)
        target = self.summary(book.key, now, with_curve=False).equity - book.start_cash
        progress = np.cumsum(~weekend) / max(1, int((~weekend).sum()))  # the correction moves only on trading days
        bridge = walk - progress * (walk[-1] - target)
        return [(int(t), round(float(book.start_cash + b), 2)) for t, b in zip(times, bridge, strict=True)]

    # -- views --------------------------------------------------------------------------------
    def position_out(self, pos: Pos) -> dto.PositionOut:
        spec = SPECS[pos.symbol]
        sign = 1 if pos.side == "LONG" else -1
        base = {k: getattr(pos, k) for k in ("id", "symbol", "side", "qty", "entry_price", "entry_fx", "opened_ts", "stop_price",
                                             "target_price", "signal_id", "status", "closed_ts", "exit_price", "exit_reason", "fees_base",
                                             "realized_pnl")}
        out = dto.PositionOut(portfolio_key=pos.key, name=spec.name, currency=spec.currency, **base)
        cost = pos.entry_price * pos.qty * spec.price_scale * pos.entry_fx
        if pos.status == "open":
            last = self.market.last_price(pos.symbol)
            fx = EUR_PER_UNIT[spec.currency]
            pnl = (last - pos.entry_price) * sign * pos.qty * spec.price_scale * fx
            return out.model_copy(update={"last_price": last, "last_ts": int(self.market.effective(pos.symbol)),
                                          "exposure_base": sign * pos.qty * unit_value_base(spec, last), "unrealized_pnl": pnl,
                                          "unrealized_pct": (last / pos.entry_price - 1) * 100 * sign})
        return out.model_copy(update={"realized_pct": (pos.realized_pnl or 0) / cost * 100 if cost else None})

    def order_out(self, o: Ord) -> dto.OrderOut:
        return dto.OrderOut(portfolio_key=o.key, name=SPECS[o.symbol].name, filled_qty=o.qty if o.status == "filled" else None,
                            **{k: getattr(o, k) for k in Ord.__slots__ if k not in ("key",)})

    def open_positions(self, key: str) -> list[Pos]:
        return sorted((p for p in self.positions.values() if p.key == key and p.status == "open"), key=lambda p: -p.opened_ts)

    def summary(self, key: str, now: int, *, with_curve: bool = True) -> dto.PortfolioSummaryOut:
        book = self.books[key]
        outs = [self.position_out(p) for p in self.open_positions(key)]
        unreal = sum(p.unrealized_pnl or 0 for p in outs)
        long_exp = sum(p.exposure_base for p in outs if (p.exposure_base or 0) > 0)
        short_exp = -sum(p.exposure_base for p in outs if (p.exposure_base or 0) < 0)
        equity = book.cash + unreal
        closed = [p for p in self.positions.values() if p.key == key and p.status == "closed"]
        pnls = [p.realized_pnl or 0 for p in closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        curve = book.curve if with_curve else []
        day_start = int(datetime.fromtimestamp(now, ZoneInfo("Europe/Berlin")).replace(hour=0, minute=0, second=0).timestamp())
        start_equity = next((v for t, v in reversed(curve) if t <= day_start), book.start_cash)
        drawdown = sharpe = None
        if len(curve) > 2:
            values = np.array([v for _, v in curve] + [equity])
            peak = np.maximum.accumulate(values)
            drawdown = float(((values - peak) / peak).min() * -100)
            daily = pd.Series(values[::48])
            rets = daily.pct_change().dropna()
            if len(rets) >= 15 and rets.std() > 0:
                sharpe = float(rets.mean() / rets.std() * math.sqrt(252))
        pending = [o for o in self.orders.values() if o.key == key and o.status == "pending"]
        return dto.PortfolioSummaryOut(
            key=key, name=book.name, base_currency=BASE, start_cash=book.start_cash, cash=book.cash, equity=equity,
            buying_power=max(0.0, equity * 1.0 - (long_exp + short_exp)), long_exposure=long_exp, short_exposure=short_exp,
            gross_leverage=(long_exp + short_exp) / equity if equity else 0.0, unrealized_pnl=unreal, realized_pnl=sum(pnls),
            fees_paid=book.fees_paid, total_return_pct=(equity / book.start_cash - 1) * 100, day_pnl=equity - start_equity,
            day_return_pct=(equity / start_equity - 1) * 100 if start_equity else 0.0, open_positions=len(outs),
            pending_orders=len(pending), closed_trades=len(closed), wins=len(wins), win_rate=len(wins) / len(closed) if closed else None,
            profit_factor=sum(wins) / -sum(losses) if losses else None, avg_win=float(np.mean(wins)) if wins else None,
            avg_loss=float(np.mean(losses)) if losses else None, max_drawdown_pct=drawdown, sharpe=sharpe, reset_ts=book.reset_ts,
            as_of_ts=now,
        )

    # -- actions ------------------------------------------------------------------------------
    def place(self, req: dto.OrderRequest, now: int) -> Ord:
        spec = SPECS.get(req.symbol.upper()) or SPECS.get(req.symbol)
        if spec is None:
            raise HTTPException(400, f"Unbekanntes Symbol „{req.symbol}“ – nur Werte der Beobachtungsliste sind handelbar.")
        if not spec.tradable:
            raise HTTPException(400, f"{spec.name} ist nicht handelbar (nur Umrechnungskurs).")
        if (req.amount_base is None) == (req.qty is None):
            raise HTTPException(400, "Bitte entweder einen Betrag in EUR oder eine Stückzahl angeben – nicht beides.")
        price = self.market.last_price(spec.symbol)
        sign = 1 if req.side == "LONG" else -1
        unit = price_text(spec, price)
        if req.stop_price is not None and (req.stop_price - price) * sign >= 0:
            side = "unter" if sign > 0 else "über"
            raise HTTPException(400, f"Der Stopp muss bei {'Long' if sign > 0 else 'Short'} {side} dem aktuellen Kurs liegen ({unit}).")
        if req.target_price is not None and (req.target_price - price) * sign <= 0:
            side = "über" if sign > 0 else "unter"
            raise HTTPException(400, f"Das Ziel muss bei {'Long' if sign > 0 else 'Short'} {side} dem aktuellen Kurs liegen ({unit}).")
        value = req.amount_base if req.amount_base is not None else (req.qty or 0) * unit_value_base(spec, price)
        summary = self.summary("me", now)
        if value > summary.buying_power:
            raise HTTPException(400, f"Kaufkraft reicht nicht: benötigt {de_num(value, 0)} EUR, verfügbar {de_num(summary.buying_power, 0)} EUR.")
        raw_qty = req.qty if req.qty is not None else value / unit_value_base(spec, price)
        qty = sizing.round_qty(to_instrument(spec), raw_qty)  # whole lots, like the paper broker
        if qty <= 0:
            lot = sizing.lot_size(to_instrument(spec)) or 1.0
            raise HTTPException(400, "Der Betrag reicht nicht für ein Stück." if lot >= 1
                                else f"Die Menge ist kleiner als die kleinste handelbare Einheit von {de_num(lot, 6).rstrip('0').rstrip(',')} Stück.")
        open_now = venue_state(VENUES[spec.exchange], now).is_open
        return self._new_order(key="me", created_ts=now, symbol=spec.symbol, intent="open", side=req.side, origin="manual", status="pending",
                               qty=qty, notional_base=req.amount_base, ref_price=price, stop_price=req.stop_price,
                               target_price=req.target_price, signal_id=req.signal_id,
                               status_reason="" if open_now else "Markt geschlossen – Ausführung zur Eröffnung")

    def try_fill(self, order: Ord, now: int) -> bool:
        """Fill a pending order at the current price plus slippage if its venue is open."""
        spec = SPECS[order.symbol]
        if order.status != "pending" or not venue_state(VENUES[spec.exchange], now).is_open:
            return False
        price = self.market.last_price(order.symbol)
        buying = (order.side == "LONG") == (order.intent == "open")
        fill = DEFAULT_COSTS.fill_price(to_instrument(spec), price, buying=buying)
        order.status, order.filled_ts, order.fill_price, order.fill_fx, order.fee_base = "filled", now, fill, EUR_PER_UNIT[spec.currency], self.FEE
        order.status_reason = ""
        if order.intent == "open":
            pos = self._open(order.key, order.symbol, order.side, order.qty or 0, fill, now, stop_price=order.stop_price,
                             target_price=order.target_price, signal_id=order.signal_id)
            order.position_id = pos.id
        elif order.position_id in self.positions:
            self._close(self.positions[order.position_id], fill, now, order.origin if order.origin != "manual" else "manual")
        return True

    def cancel(self, order_id: int) -> Ord:
        order = self.orders.get(order_id)
        if order is None or order.key != "me":
            raise HTTPException(404, f"Order #{order_id} gibt es nicht.")
        if order.status != "pending":
            raise HTTPException(400, f"Order #{order_id} ist nicht mehr offen und kann nicht storniert werden.")
        order.status, order.status_reason = "cancelled", "vom Nutzer storniert"
        return order

    def close_position(self, position_id: int, now: int) -> Ord:
        pos = self._my_open(position_id)
        if any(o.position_id == pos.id and o.status == "pending" and o.intent == "close" for o in self.orders.values()):
            raise HTTPException(400, f"Für Position #{position_id} liegt bereits eine Schließorder vor.")
        open_now = venue_state(VENUES[SPECS[pos.symbol].exchange], now).is_open
        return self._new_order(key="me", created_ts=now, symbol=pos.symbol, intent="close", side=pos.side, origin="manual", status="pending",
                               qty=pos.qty, ref_price=self.market.last_price(pos.symbol), position_id=pos.id,
                               status_reason="" if open_now else "Markt geschlossen – Ausführung zur Eröffnung")

    def update_levels(self, position_id: int, req: dto.PositionUpdateRequest) -> Pos:
        pos = self._my_open(position_id)
        price = self.market.last_price(pos.symbol)
        sign = 1 if pos.side == "LONG" else -1
        if req.stop_price is not None and (req.stop_price - price) * sign >= 0:
            raise HTTPException(400, f"Der Stopp liegt auf der falschen Seite des aktuellen Kurses ({price_text(SPECS[pos.symbol], price)}).")
        if req.target_price is not None and (req.target_price - price) * sign <= 0:
            raise HTTPException(400, f"Das Ziel liegt auf der falschen Seite des aktuellen Kurses ({price_text(SPECS[pos.symbol], price)}).")
        pos.stop_price, pos.target_price = req.stop_price, req.target_price
        return pos

    def _my_open(self, position_id: int) -> Pos:
        pos = self.positions.get(position_id)
        if pos is None or pos.key != "me":
            raise HTTPException(404, f"Position #{position_id} gibt es nicht in Ihrem Depot.")
        if pos.status != "open":
            raise HTTPException(400, f"Position #{position_id} ist bereits geschlossen.")
        return pos

    def reset(self, capital: float | None, now: int) -> None:
        for pos in [p for p in self.positions.values() if p.status == "open"]:
            self._close(pos, self.market.last_price(pos.symbol), now, "reset")
        for order in self.orders.values():
            if order.status == "pending":
                order.status, order.status_reason = "cancelled", "beim Neustart storniert"
        for book in self.books.values():
            book.start_cash = capital or book.start_cash
            book.cash = book.start_cash
            book.fees_paid = 0.0
            book.reset_ts = now
            book.curve = [(now, book.start_cash)]
        self.positions.clear()
        self.orders.clear()

    def equity_curves(self, since: int | None, max_points: int, now: int) -> dto.EquityCurvesOut:
        start = max(since or 0, self.books["ai"].reset_ts)
        series: dict[str, list[tuple[int, float]]] = {}
        for key, book in self.books.items():
            points = [p for p in book.curve if p[0] >= start] + [(now, round(self.summary(key, now, with_curve=False).equity, 2))]
            series[key] = _thin(points, max_points)
        times = [t for t, _ in series["ai"]]
        capital = self.books["ai"].start_cash
        benchmarks = {}
        for symbol in ("^GDAXI", "^GSPC"):
            base_price = self.market.price_at(symbol, times[0]) if times else 1.0
            benchmarks[symbol] = [(t, round(capital * self.market.price_at(symbol, t) / base_price, 2)) for t in times]
        return dto.EquityCurvesOut(since_ts=start, base_currency=BASE, start_cash=self.books["ai"].start_cash, ai=series["ai"],
                                   me=series["me"], benchmarks=benchmarks)


def _thin(points: list[tuple[int, float]], max_points: int) -> list[tuple[int, float]]:
    if len(points) <= max_points:
        return points
    step = math.ceil(len(points) / max_points)
    return points[::step][: max_points - 1] + [points[-1]]


def price_text(spec: Spec, price: float) -> str:
    unit = "GBp" if spec.price_scale == 0.01 and spec.currency == "GBP" else spec.currency
    return f"{de_num(price, 4 if spec.asset_class == 'fx' else 2)} {unit}"


# ------------------------------------------------------------------------------------------------
# Validation, operations
# ------------------------------------------------------------------------------------------------
def build_validation_result(days: int, now: int) -> dto.ValidationResultOut:
    """A replay report with an honest, unremarkable result (no edge beyond chance)."""
    rng = np.random.default_rng(1000 + days)
    n = max(30, days * 6)
    start = now - days * 86400
    ts = np.sort(rng.integers(start, now, n))
    setups = rng.choice(list(RATIONALES), n)
    regions = rng.choice(["ASIA", "EU", "US", "GLOBAL"], n, p=[0.2, 0.35, 0.35, 0.1])
    directions = rng.choice(["LONG", "SHORT"], n, p=[0.6, 0.4])
    conf = rng.uniform(0.35, 0.78, n)
    wins = rng.random(n) < (0.36 + 0.12 * (conf - 0.35))
    r = np.where(wins, rng.uniform(1.4, 2.4, n), -1.0) - rng.uniform(0.03, 0.08, n)
    results = [float(x) for x in r]
    overall = stat_block(results)
    folds = []
    for k, idx in enumerate(np.array_split(np.arange(n), 6)):
        folds.append(dto.FoldOut(index=k, start_ts=int(ts[idx[0]]), end_ts=int(ts[idx[-1]]), stats=stat_block([results[i] for i in idx])))
    sims = rng.normal(overall.avg_r - 0.07, 0.09, 1000)  # the replay's edge stays inside the random spread
    p_value = float((sims >= overall.avg_r).mean())
    trading_days = int(days * 5 / 7)
    warnings = []
    if trading_days < 40:
        warnings.append(f"Nur {trading_days} Handelstage – die Aussagekraft ist begrenzt.")
    negative = [f.index + 1 for f in folds if (f.stats.avg_r or 0) < 0]
    if negative:
        warnings.append(f"Zeitblock {', '.join(map(str, negative))} mit negativem Ø R: Das Ergebnis ist zeitlich nicht stabil.")
    if overall.n < 100:
        warnings.append(f"Nur {overall.n} Trades – erst ab etwa 100 werden die Intervalle eng genug für eine Aussage.")
    verdict = (
        f"Kein belastbarer Vorteil nachweisbar: Ø {de_num(overall.avg_r or 0, 2, signed=True)} R liegt innerhalb der Spanne zufälliger Einstiege (p = {de_num(p_value, 2)})."
        if p_value >= 0.05
        else f"Die Engine schlägt zufällige Einstiege (p = {de_num(p_value, 2)}); das Ergebnis sollte sich in weiteren Zeiträumen bestätigen."
    )
    labels = {k: SETUP_LABELS_DE[k] for k in RATIONALES}
    return dto.ValidationResultOut(
        period_start_ts=start, period_end_ts=now, trading_days=trading_days, symbols=31, decisions=n * 37, overall=overall,
        by_setup=group_stats([(s, labels[s], x) for s, x in zip(setups, results, strict=True)]),
        by_region=group_stats([(g, REGION_LABELS[g], x) for g, x in zip(regions, results, strict=True)]),
        by_direction=group_stats([(d, "Long" if d == "LONG" else "Short", x) for d, x in zip(directions, results, strict=True)]),
        folds=folds, calibration=calibration(list(zip(conf.tolist(), results, strict=True))),
        baseline=dto.BaselineOut(n_sims=1000, mean_avg_r=float(sims.mean()), p95_avg_r=float(np.percentile(sims, 95)), p_value=p_value),
        equity_r=[(int(t), round(float(v), 2)) for t, v in zip(ts, np.cumsum(r), strict=True)], warnings_de=warnings, verdict_de=verdict,
    )


ACTIVITY_SEED: tuple[tuple[int, str, str, str], ...] = (
    (1, "info", "signal", "Neues Signal NVDA Long (Ausbruch) zur KI-Prüfung gegeben."),
    (2, "info", "llm", "Signalprüfung NVDA: bestätigt (Überzeugung 72 %), 6,8 s."),
    (4, "warning", "news", "Feed „Nikkei Asia Markets“ antwortet mit HTTP 503 – nächster Versuch in 10 Minuten."),
    (6, "info", "sim", "KI-Depot: Order für SAP.DE ausgeführt, 27 Stück zu 183,42 EUR."),
    (9, "info", "mail", "Alarm-Mail zu Signal #128 als Datei gespeichert."),
    (12, "info", "data", "Xetra geöffnet: 9 Werte im Minutentakt."),
    (15, "warning", "llm", "Chartanalyse TSLA abgebrochen: Zeitüberschreitung nach 45 s."),
    (19, "info", "news", "54 neue Meldungen in der letzten Stunde, 21 analysiert."),
    (26, "info", "signal", "Signal #117 RHM.DE geschlossen: Ziel erreicht, +2,1 R."),
    (33, "error", "data", "Abruf der Minutenkerzen für 9984.T fehlgeschlagen (HTTP 429) – Abrufe 60 s gedrosselt."),
    (41, "info", "system", "Tageswartung abgeschlossen in 3,2 s."),
    (58, "info", "sim", "Mein Depot: Position MSFT geschlossen, +184,20 EUR."),
    (75, "info", "data", "London geöffnet."),
    (96, "info", "signal", "Verworfener Kandidat AMZN Short: Score 0,55 unter der Schwelle 0,60."),
    (130, "warning", "mail", "Mail-Stundenlimit erreicht (12) – weitere Alarme übersprungen."),
    (180, "info", "system", "Trading Local gestartet (Version 0.1.0)."),
)


def build_ops(now: int) -> tuple[list[dto.ActivityOut], list[dto.LlmCallOut], list[dto.MailLogOut]]:
    activity = [dto.ActivityOut(id=400 - k, ts=now - m * 60, level=lvl, category=cat, message=msg)
                for k, (m, lvl, cat, msg) in enumerate(ACTIVITY_SEED)]
    rng = np.random.default_rng(5)
    calls = []
    for k in range(40):
        kind = str(rng.choice(["news", "news", "trade_review", "chart"]))
        ok = rng.random() > 0.06
        calls.append(dto.LlmCallOut(
            id=800 - k, ts=now - k * 97, kind=kind, ok=bool(ok), latency_ms=int(rng.integers(1800, 9500)) if ok else 45000,
            queue_wait_ms=int(rng.integers(0, 4000)), prompt_tokens=int(rng.integers(600, 2400)) if ok else None,
            output_tokens=int(rng.integers(80, 420)) if ok else None, error="" if ok else "Zeitüberschreitung nach 45 s",
        ))
    mails = [
        dto.MailLogOut(id=60, ts=now - 9 * 60, kind="signal", signal_id=128, recipient="t***@example.org",
                       subject="▲ Long NVDA – Ausbruch, Konfidenz 61 %", status="file", detail="data/outbox/2026-11-19_signal_128.eml"),
        dto.MailLogOut(id=59, ts=now - 52 * 60, kind="exit", signal_id=117, recipient="t***@example.org",
                       subject="Ausstieg RHM.DE – Ziel erreicht (+2,1 R)", status="file", detail=""),
        dto.MailLogOut(id=58, ts=now - 130 * 60, kind="signal", signal_id=115, recipient="t***@example.org",
                       subject="▼ Short TSLA – Trend-Rücksetzer", status="skipped", detail="Stundenlimit erreicht (12)"),
        dto.MailLogOut(id=57, ts=now - 26 * 3600, kind="test", recipient="t***@example.org", subject="Trading Local: Testmail",
                       status="file", detail=""),
        dto.MailLogOut(id=56, ts=now - 30 * 3600, kind="signal", signal_id=109, recipient="t***@example.org",
                       subject="▲ Long SAP.DE – Trend-Rücksetzer", status="failed", detail="SMTP-Anmeldung fehlgeschlagen (535)"),
    ]
    return activity, calls, mails


JOBS = (
    ("minute_cycle", "Minutentakt", 60, 1240, True, ""),
    ("news_poll", "Nachrichtenabruf", 60, 3100, True, "14 Feeds, 1 gestört"),
    ("equity_snapshot", "Depotwerte festhalten", 60, 35, True, ""),
    ("chart_scan", "Chartanalysen", 1800, 61_000, True, "4 Werte analysiert"),
    ("backfill", "Historie nachladen", 3600, 18_400, True, "alle Werte vollständig"),
    ("daily_maintenance", "Tageswartung", 86400, 3200, False, "VACUUM übersprungen: Datenbank in Benutzung"),
    ("movers", "Kursauslöser (Nachrichten vorziehen)", 60, 140, True, ""),
    ("derivatives", "Krypto-Derivate (Kraken Futures)", 60, 2300, True, "11 neue Einträge"),
    ("agenda", "Terminkalender", 60, 420, True, "2 neue Einträge"),
)


# ------------------------------------------------------------------------------------------------
# Agenda
# ------------------------------------------------------------------------------------------------
# (minutes from the demo start, kind, currency or symbol, title, importance, forecast, previous, actual, z)
AGENDA_DEMO: tuple[tuple[int, str, str, str, int, str, str, str, float | None], ...] = (
    (-26 * 60, "macro", "EUR", "German Ifo Business Climate", 2, "87.5", "87.3", "88.1", 1.2),
    (-5 * 60 - 12, "macro", "USD", "Unemployment Claims", 3, "221K", "219K", "213K", -1.4),
    (-3 * 60 - 12, "macro", "USD", "CB Consumer Confidence", 3, "92.4", "91.7", "103.3", 2.6),
    (-3 * 60 - 12, "macro", "USD", "Richmond Manufacturing Index", 1, "-2", "-2", "3", None),
    (18, "macro", "USD", "Core PCE Price Index m/m", 3, "0.3%", "0.3%", "", None),
    (18, "macro", "USD", "Prelim GDP q/q", 3, "3.5%", "3.4%", "", None),
    (3 * 60 + 48, "earnings", "NVDA", "Quartalszahlen", 3, "1.25", "", "", None),
    (16 * 60 + 18, "macro", "JPY", "Tokyo Core CPI y/y", 2, "1.9%", "2.0%", "", None),
    (20 * 60, "macro", "EUR", "German Prelim CPI m/m", 2, "0.1%", "0.3%", "", None),
    (26 * 60 + 48, "central_bank", "EUR", "Main Refinancing Rate", 3, "2.15%", "2.15%", "", None),
    (27 * 60 + 18, "central_bank", "EUR", "ECB Press Conference", 3, "", "", "", None),
    (6 * 1440 + 18, "central_bank", "USD", "Federal Funds Rate", 3, "4.50%", "4.75%", "", None),
    (9 * 1440 - 4 * 60, "earnings", "SAP.DE", "Quartalszahlen", 3, "1.98", "", "", None),
    (-2 * 1440 - 5 * 60, "earnings", "AAPL", "Quartalszahlen", 3, "1.98", "", "2.10", None),
)
AGENDA_KINDS_DE = {"macro": "Konjunkturdaten", "central_bank": "Notenbank", "earnings": "Quartalszahlen"}
AGENDA_BLOCK_MIN = 30


def _agenda_number(text: str) -> float | None:
    try:
        value = float(text.rstrip("%KkMB"))
    except ValueError:
        return None
    return value * (1000.0 if text[-1:] in "Kk" else 1.0)


def build_agenda(now: int) -> list[dto.AgendaEventOut]:
    """Deterministic demo calendar around the demo start (released entries carry actual and surprise)."""
    out = []
    start = now // 60 * 60
    for i, (minutes, kind, key, title, importance, forecast, previous, actual, z) in enumerate(AGENDA_DEMO, 1):
        ts = start + minutes * 60
        earnings = kind == "earnings"
        actual_value, forecast_value = _agenda_number(actual), _agenda_number(forecast)
        surprise = None
        label: Literal["above", "below", "inline", ""] = ""
        if actual_value is not None and forecast_value is not None:
            surprise = (actual_value / forecast_value - 1) * 100 if earnings else actual_value - forecast_value
            label = "inline" if abs(surprise) < 1e-9 else "above" if surprise > 0 else "below"
        result = ""
        if actual:
            result = f"EPS {actual} statt {forecast}" if earnings else f"{actual} statt {forecast}"
            result += {"above": " – über Erwartung", "below": " – unter Erwartung", "inline": " – wie erwartet", "": ""}[label]
        currency = "" if earnings else key
        affected = 1 if earnings else sum(
            1 for spec in UNIVERSE if spec.tradable and (spec.currency == currency or (currency == "USD" and spec.asset_class == "crypto"))
        )
        blocking = not earnings and importance == 3
        block_from = ts - AGENDA_BLOCK_MIN * 60 if blocking else None
        out.append(dto.AgendaEventOut(
            id=i, kind=kind, kind_de=AGENDA_KINDS_DE[kind], source="yahoo" if earnings else "forexfactory", currency=currency,
            symbol=key if earnings else None, name=SPECS[key].name if earnings else "", title=title, event_ts=ts,
            time_known=True, importance=importance, forecast_text=forecast, previous_text=previous, actual_text=actual,
            surprise=surprise, surprise_z=z, surprise_label=label, result_de=result, affected=affected,
            block_from_ts=block_from, blocks_entries=block_from is not None and block_from <= now <= ts,
        ))
    return out


# ------------------------------------------------------------------------------------------------
# State container
# ------------------------------------------------------------------------------------------------
class MockState:
    def __init__(self, clock: DemoClock, offline: bool = False) -> None:
        self.clock = clock
        self.offline = offline
        now = clock.ts()
        self.started_ts = now - 3 * 3600
        self.market = Market(clock, now)
        self.instruments_active = {s.symbol: True for s in UNIVERSE}
        self.instruments_tradable = {s.symbol: s.tradable for s in UNIVERSE}
        self.news = build_news(now)
        self.signals = build_signals(self.market, self.news, now)
        self.broker = Broker(self.market, self.signals, now)
        self.chart_analyses = build_chart_analyses(self.market, now)
        self.feeds = build_feeds(now)
        self.agenda = build_agenda(now)
        self.activity, self.llm_calls, self.mail_log = build_ops(now)
        self.settings = RuntimeSettings()
        self.validation_runs = [
            dto.ValidationRunOut(id=3, ts=now - 2 * 86400, status="done", engine_version=ENGINE_VERSION, params={"days": 30},
                                 result=build_validation_result(30, now - 2 * 86400)),
            dto.ValidationRunOut(id=2, ts=now - 6 * 86400, status="failed", engine_version=ENGINE_VERSION, params={"days": 5},
                                 message="Zu wenig Daten: Für 5 Tage liegen erst 2 vollständige Handelstage vor."),
            dto.ValidationRunOut(id=1, ts=now - 9 * 86400, status="done", engine_version="engine-2026.10.4", params={"days": 10},
                                 result=build_validation_result(10, now - 9 * 86400)),
        ]
        self.live_news_index = 0
        self.chart_pending: set[str] = set()  # symbols with a chart analysis in progress

    def now(self) -> int:
        return self.clock.ts()

    def instrument_out(self, spec: Spec) -> dto.InstrumentOut:
        venue = VENUES[spec.exchange]
        state = venue_state(venue, self.now())
        open_signal = next((s for s in sorted(self.signals.values(), key=lambda x: -x.created_ts)
                            if s.symbol == spec.symbol and s.actionable and s.status in ("review", "pending_entry", "active")), None)
        return dto.InstrumentOut(
            symbol=spec.symbol, name=spec.name, asset_class=spec.asset_class, region=spec.region, exchange=spec.exchange,
            exchange_name=venue.name, currency=spec.currency, price_scale=spec.price_scale, timezone=venue.tz, delay_min=spec.delay_min,
            tradable=self.instruments_tradable[spec.symbol], active=self.instruments_active[spec.symbol], benchmark=spec.benchmark,
            sort_order=UNIVERSE.index(spec), phase=state.phase, is_open=state.is_open, quote=self.market.quote(spec.symbol),
            open_signal=None if open_signal is None else dto.SignalBrief.model_validate(open_signal),
        )

    def signal_out(self, sig: dto.SignalDetailOut) -> dto.SignalOut:
        data = sig.model_dump(exclude={"components", "features", "news"})
        if sig.status == "active" and sig.entry_price is not None:
            last = self.market.last_price(sig.symbol)
            sign = 1 if sig.direction == "LONG" else -1
            data.update(last_price=last, open_r=(last - sig.entry_price) * sign / sig.stop_dist)
        return dto.SignalOut(**data)

    def sessions(self) -> dto.SessionsOut:
        now = self.now()
        start, end = now - 6 * 3600, now + 18 * 3600
        regions, timeline = [], []
        for region in ("ASIA", "EU", "US"):
            venues = [v for v in VENUES.values() if v.region == region]
            states = [venue_state(v, now) for v in venues]
            regions.append(dto.RegionSessionOut(region=region, label=REGION_LABELS[region], is_open=any(s.is_open for s in states), exchanges=states))
            for v in venues:
                timeline += [dto.TimelineSegmentOut(exchange=v.key, region=region, start=s, end=e, kind=k)
                             for s, e, k in venue_segments(v, start, end)]
        return dto.SessionsOut(now=now, regions=regions, timeline_start=start, timeline_end=end, timeline=timeline)

    def status(self) -> dto.SystemStatusOut:
        now = self.now()
        tick = now // 7
        n_open = sum(1 for s in UNIVERSE if venue_state(VENUES[s.exchange], now).is_open)
        return dto.SystemStatusOut(
            server_time=now, started_ts=self.started_ts, version=VERSION, engine_version=ENGINE_VERSION, base_currency=BASE,
            offline=self.offline,
            llm=dto.LlmStatusOut(model="qwen3.5:9b", enabled=True, available=True, busy=bool(tick % 3), queue_depth=int(tick % 4),
                                 last_latency_ms=4200 + int(tick % 5) * 600, calls_last_hour=46, errors_last_hour=2,
                                 last_error="Zeitüberschreitung nach 45 s (Chartanalyse TSLA)"),
            data=dto.DataStatusOut(stream_connected=True, last_tick_ts=now - 1, last_rest_ok_ts=now - 38, symbols_active=len(UNIVERSE),
                                   symbols_stale=1 if n_open else 0),
            news=dto.NewsStatusOut(feeds_total=len(FEEDS), feeds_failing=1, items_last_hour=54, analyzed_last_hour=21, backlog=3,
                                   last_poll_ts=now - 22),
            mail=dto.MailStatusOut(mode="file", configured=True, recipient_masked="t***@example.org", sent_last_hour=2),
            db=dto.DbStatusOut(size_mb=412.6, bars_1m=4_812_340, first_bar_ts=now - 41 * 86400, last_bar_ts=now - 60, news=18_204,
                               signals=len(self.signals) + 312),
            jobs=self.jobs(),
        )

    def jobs(self) -> list[dto.JobStatusOut]:
        now = self.now()
        return [dto.JobStatusOut(job=j, label_de=label, last_started_ts=now - (now % every) - (0 if every < 3600 else 600),
                                 last_duration_ms=ms, ok=ok, message=msg) for j, label, every, ms, ok, msg in JOBS]

    def add_activity(self, level: str, category: str, message: str, ref_type: str | None = None, ref_id: int | None = None) -> dto.ActivityOut:
        entry = dto.ActivityOut(id=max(a.id for a in self.activity) + 1, ts=self.now(), level=level, category=category, message=message,
                                ref_type=ref_type, ref_id=ref_id)
        self.activity.insert(0, entry)
        return entry


# ------------------------------------------------------------------------------------------------
# WebSocket hub
# ------------------------------------------------------------------------------------------------
class Hub:
    def __init__(self) -> None:
        self.clients: set[asyncio.Queue[dict[str, Any]]] = set()

    def publish(self, kind: str, data: Any) -> None:
        message = {"type": kind, "data": data.model_dump(mode="json") if hasattr(data, "model_dump") else data}
        for queue in list(self.clients):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)


async def producer(state: MockState, hub: Hub) -> None:
    """Occasional global events: news (queued, then analysed), new signals, fills of pending orders, and a
    'sessions' message whenever a venue's phase changes (like the real hub)."""
    tick = 0
    phases = {key: venue_state(venue, state.now()).phase for key, venue in VENUES.items()}
    while True:
        await asyncio.sleep(1)
        tick += 1
        now = state.now()
        for key, venue in VENUES.items():
            phase = venue_state(venue, now).phase
            if phase != phases[key]:
                phases[key] = phase
                hub.publish("sessions", {"exchange": key, "phase": phase})
        for order in [o for o in state.broker.orders.values() if o.status == "pending" and now - o.created_ts >= 2]:
            if state.broker.try_fill(order, now):
                hub.publish("order", state.broker.order_out(order))
                hub.publish("portfolio", {"key": order.key})
                hub.publish("activity", state.add_activity("info", "sim", f"{'KI-Depot' if order.key == 'ai' else 'Mein Depot'}: Order #{order.id} {order.symbol} ausgeführt zu {price_text(SPECS[order.symbol], order.fill_price or 0)}."))
        if tick % 40 == 0:
            await _live_news(state, hub)
        if tick % 55 == 0:
            await _live_signal(state, hub)


async def _live_news(state: MockState, hub: Hub) -> None:
    source, lang, title, summary, impacts, rel, sent, cat = LIVE_NEWS[state.live_news_index % len(LIVE_NEWS)]
    state.live_news_index += 1
    now = state.now()
    item = dto.NewsOut(id=max(n.id for n in state.news) + 1, source=source, source_name=FEED_NAMES[source],
                       url=f"https://example.org/demo/news/live-{state.live_news_index}", title=title, lang=lang, published_ts=now - 60,
                       fetched_ts=now, status="queued", hinted_symbols=sorted(impacts))
    state.news.insert(0, item)
    hub.publish("news", item)
    await asyncio.sleep(8)
    analysed = item.model_copy(update={"status": "analyzed", "analysis": dto.NewsAnalysisOut(
        analyzed_ts=state.now(), relevance=rel, sentiment=sent, confidence=0.7, category=cat, horizon="days", summary_de=summary,
        symbols=[dto.NewsImpactOut(symbol=s, sentiment=v) for s, v in impacts.items()])})
    state.news[state.news.index(item)] = analysed
    hub.publish("news", analysed)


async def _live_signal(state: MockState, hub: Hub) -> None:
    now = state.now()
    candidates = [s for s in UNIVERSE if s.asset_class in ("equity", "crypto") and venue_state(VENUES[s.exchange], now).is_open]
    if not candidates:
        return
    spec = candidates[now % len(candidates)]
    factory = SignalFactory(state.market, np.random.default_rng(now))
    n = state.market.closed_count(spec.symbol)
    base = factory.create(spec, n - 1, setup="trend_pullback", direction="LONG" if now % 2 else "SHORT", actionable=True, score=0.65, verdict=None)
    sid = max(state.signals) + 1
    sig = base.model_copy(update={"id": sid, "status": "review", "issued_ts": None, "mailed_ts": None, "entry_ts": None, "entry_price": None,
                                  "stop_price": None, "target_price": None, "exit_ts": None, "exit_price": None, "outcome_r": None,
                                  "mfe_r": None, "mae_r": None})
    state.signals[sid] = sig
    hub.publish("signal", state.signal_out(sig))
    hub.publish("activity", state.add_activity("info", "signal", f"Neues Signal {spec.symbol} {'Long' if sig.direction == 'LONG' else 'Short'} (Trend-Rücksetzer) zur KI-Prüfung gegeben.", "signal", sid))
    await asyncio.sleep(6)
    issued = sig.model_copy(update={"status": "pending_entry", "llm_verdict": "confirm", "llm_conviction": 0.68, "issued_ts": state.now(),
                                    "mailed_ts": state.now()})
    state.signals[sid] = issued
    hub.publish("signal", state.signal_out(issued))


async def client_ticker(ws: WebSocket, state: MockState, subs: set[str]) -> None:
    """Per-connection periodic messages: quotes every second, bars of subscribed symbols, status, ping."""
    tick = 0
    while True:
        tick += 1
        now = state.now()
        open_syms = [s.symbol for s in UNIVERSE if venue_state(VENUES[s.exchange], now).is_open]
        if open_syms:
            k = tick % max(1, len(open_syms))
            chosen = (open_syms[k:] + open_syms[:k])[:8]
            await ws.send_json({"type": "quotes", "data": {"items": [state.market.quote(s).model_dump(mode="json") for s in chosen]}})
        if tick % 3 == 0:
            for symbol in list(subs):
                bar = state.market.forming(symbol)
                if bar:
                    await ws.send_json({"type": "bar", "data": {"symbol": symbol, "bar": list(bar), "source": "stream"}})
        if tick % 5 == 0:
            await ws.send_json({"type": "status", "data": state.status().model_dump(mode="json")})
        if tick % 20 == 0:
            await ws.send_json({"type": "ping", "data": {}})
        await asyncio.sleep(1)


# ------------------------------------------------------------------------------------------------
# App
# ------------------------------------------------------------------------------------------------
def _spec_or_404(state: MockState, symbol: str) -> Spec:
    spec = SPECS.get(symbol)
    if spec is None:
        raise HTTPException(404, f"„{symbol}“ ist nicht in der Beobachtungsliste.")
    return spec


def _german_validation(exc: RequestValidationError | ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path"))
        msg = err.get("msg", "ungültig")
        ctx = err.get("ctx") or {}
        kind = err.get("type", "")
        if kind in ("greater_than", "greater_than_equal"):
            msg = f"muss {'größer als' if kind == 'greater_than' else 'mindestens'} {ctx.get('gt', ctx.get('ge'))} sein"
        elif kind in ("less_than_equal", "less_than"):
            msg = f"darf höchstens {ctx.get('le', ctx.get('lt'))} sein"
        elif kind == "missing":
            msg = "fehlt"
        elif kind == "extra_forbidden":
            msg = "ist kein bekanntes Feld"
        elif "type" in kind or "parsing" in kind:
            msg = "hat den falschen Typ"
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "Ungültige Eingabe – " + "; ".join(parts)


def create_app(demo_now: datetime | None = DEFAULT_DEMO_NOW, *, offline: bool = False, live: bool = True) -> FastAPI:
    """The mock API. ``demo_now=None`` uses the wall clock; ``live=False`` starts no background events (tests)."""
    for ext, kind in ((".js", "text/javascript"), (".mjs", "text/javascript"), (".css", "text/css"), (".svg", "image/svg+xml"),
                      (".woff2", "font/woff2"), (".json", "application/json")):
        mimetypes.add_type(kind, ext)
    state = MockState(DemoClock(demo_now), offline=offline)
    hub = Hub()
    background: set[asyncio.Task[None]] = set()

    def spawn(coro: Any) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        background.add(task)
        task.add_done_callback(background.discard)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if live:
            spawn(producer(state, hub))
        try:
            yield
        finally:
            pending = list(background)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(title="Trading Local – Demo-API", lifespan=lifespan)
    app.state.mock = state

    @app.middleware("http")
    async def headers(request: Request, call_next: Callable[..., Any]) -> Any:
        if request.url.path.startswith("/api/") and request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not request.headers.get("content-type", "").startswith("application/json"):
                return JSONResponse({"detail": "Anfragen müssen als JSON gesendet werden (Content-Type: application/json)."}, status_code=415)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": _german_validation(exc)}, status_code=422)

    # -- market -----------------------------------------------------------------------------------
    @app.get("/api/status")
    def get_status() -> dto.SystemStatusOut:
        return state.status()

    @app.get("/api/sessions")
    def get_sessions() -> dto.SessionsOut:
        return state.sessions()

    @app.get("/api/instruments")
    def get_instruments(include_inactive: bool = False) -> list[dto.InstrumentOut]:
        return [state.instrument_out(s) for s in UNIVERSE if include_inactive or state.instruments_active[s.symbol]]

    @app.post("/api/instruments", status_code=201)
    def add_instrument(body: dto.InstrumentCreate) -> dto.InstrumentOut:
        symbol = body.symbol.strip().upper()
        if symbol in SPECS:
            raise HTTPException(409, f"{symbol} ist bereits in der Beobachtungsliste.")
        raise HTTPException(404, f"Die Datenquelle kennt das Symbol „{symbol}“ nicht. Schreibweise wie bei Yahoo Finance verwenden, z. B. ASML.AS oder 7203.T.")

    @app.patch("/api/instruments/{symbol}")
    def patch_instrument(symbol: str, body: dto.InstrumentUpdate) -> dto.InstrumentOut:
        spec = _spec_or_404(state, symbol)
        if body.tradable and spec.asset_class not in ("equity", "index", "crypto"):  # sim.sizing.is_tradable_class
            role = "ein Umrechnungskurs" if spec.asset_class == "fx" else "ein Vergleichswert"
            raise HTTPException(400, f"{spec.symbol} ({spec.name}) ist {role} und kann im Papierdepot nicht gehandelt werden; "
                                     "„Handelbar“ lässt sich deshalb nicht einschalten.")
        if body.active is not None:
            state.instruments_active[symbol] = body.active
        if body.tradable is not None:
            state.instruments_tradable[symbol] = body.tradable
        return state.instrument_out(spec)

    @app.get("/api/quotes")
    def get_quotes() -> list[dto.QuoteOut]:
        return [state.market.quote(s.symbol) for s in UNIVERSE if state.instruments_active[s.symbol]]

    @app.get("/api/sparklines")
    def get_sparklines() -> dto.SparklinesOut:
        return dto.SparklinesOut(series={s.symbol: state.market.sparkline(s.symbol) for s in UNIVERSE})

    @app.get("/api/bars/{symbol}")
    def get_bars(symbol: str, tf: Literal["1m", "5m", "15m", "1h", "1d"] = "15m", limit: int = Query(500, ge=1, le=5000),
                 before: int | None = None) -> dto.BarsOut:
        _spec_or_404(state, symbol)
        return state.market.bars(symbol, tf, limit, before)

    @app.get("/api/analysis/{symbol}")
    def get_analysis(symbol: str) -> dto.AnalysisOut:
        _spec_or_404(state, symbol)
        charts = state.chart_analyses.get(symbol, [])
        impacts = [i.sentiment for n in state.news if n.analysis for i in n.analysis.symbols if i.symbol == symbol]
        return build_analysis(state.market, symbol, charts[0] if charts else None, round(float(np.mean(impacts)), 2) if impacts else None)

    @app.post("/api/analysis/{symbol}/chart", status_code=202)
    async def request_chart(symbol: str) -> dict[str, Any]:
        spec = _spec_or_404(state, symbol)
        if symbol in state.chart_pending:  # like the engine: one analysis per symbol at a time
            return {"queued": True, "detail_de": "Chartanalyse angefordert – das Ergebnis erscheint hier, sobald die KI geantwortet hat."}
        if state.market.closed_count(symbol) < 30 * 15:  # the engine needs 30 bars of the 15-minute timeframe
            return {"queued": False, "detail_de": "Chartanalyse derzeit nicht möglich: Für diesen Wert liegen zu wenige "
                                                  "Kursdaten vor oder es läuft bereits eine Analyse in einem anderen Zeitrahmen."}
        state.chart_pending.add(symbol)

        async def finish() -> None:
            await asyncio.sleep(5)
            items = state.chart_analyses.setdefault(symbol, [])
            price = state.market.last_price(symbol)
            ca = dto.ChartAnalysisOut(id=max([c.id for v in state.chart_analyses.values() for c in v] + [500]) + 1, symbol=symbol,
                                      ts=state.now(), timeframe="15m", model="qwen3.5:9b", bias="neutral", strength=3, pattern="Seitwärtsrange",
                                      support=round(price * 0.99, 2), resistance=round(price * 1.012, 2), comment_de=CHART_COMMENTS["neutral"],
                                      latency_ms=8800)
            items.insert(0, ca)
            state.chart_pending.discard(symbol)
            hub.publish("chart_analysis", ca)
            hub.publish("activity", state.add_activity("info", "llm", f"Chartanalyse {spec.symbol} fertig (8,8 s)."))

        spawn(finish())
        return {"queued": True, "detail_de": "Chartanalyse angefordert – das Ergebnis erscheint hier, sobald die KI geantwortet hat."}

    @app.get("/api/chart-analyses/{symbol}")
    def get_chart_analyses(symbol: str, limit: int = Query(10, ge=1, le=100)) -> list[dto.ChartAnalysisOut]:
        _spec_or_404(state, symbol)
        return state.chart_analyses.get(symbol, [])[:limit]

    # -- news -------------------------------------------------------------------------------------
    @app.get("/api/news")
    def get_news(limit: int = Query(50, ge=1, le=500), before_id: int | None = None, symbol: str | None = None,
                 min_relevance: int | None = Query(None, ge=1, le=5), analyzed_only: bool = False) -> list[dto.NewsOut]:
        items = state.news
        if before_id is not None:
            items = [n for n in items if n.id < before_id]
        if symbol:
            items = [n for n in items if symbol in n.hinted_symbols or (n.analysis and any(i.symbol == symbol for i in n.analysis.symbols))]
        if min_relevance:
            items = [n for n in items if n.analysis and n.analysis.relevance >= min_relevance]
        if analyzed_only:
            items = [n for n in items if n.analysis]
        return items[:limit]

    # -- agenda -----------------------------------------------------------------------------------
    @app.get("/api/agenda")
    def get_agenda(from_ts: int | None = Query(None, ge=0), to_ts: int | None = Query(None, ge=0),
                   min_importance: int = Query(1, ge=1, le=3),
                   kind: Literal["macro", "central_bank", "earnings"] | None = None,
                   symbol: str | None = None) -> list[dto.AgendaEventOut]:
        now = state.now()
        start = now - 86400 if from_ts is None else from_ts
        end = now + 7 * 86400 if to_ts is None else to_ts
        items = [e for e in state.agenda if start <= e.event_ts < end and e.importance >= min_importance]
        if kind:
            items = [e for e in items if e.kind == kind]
        if symbol:
            spec = _spec_or_404(state, symbol)
            items = [e for e in items if e.symbol == symbol or (e.kind != "earnings" and (
                e.currency == spec.currency or (e.currency == "USD" and spec.asset_class == "crypto")))]
        return sorted(items, key=lambda e: (e.event_ts, e.id))

    # -- signals ----------------------------------------------------------------------------------
    @app.get("/api/signals")
    def get_signals(scope: Literal["open", "closed", "all"] = "all", actionable: Literal["true", "false", "all"] = "all",
                    symbol: str | None = None, limit: int = Query(50, ge=1, le=500), before_id: int | None = None) -> list[dto.SignalOut]:
        items = sorted(state.signals.values(), key=lambda s: -s.id)
        open_states = ("review", "pending_entry", "active")
        if scope != "all":
            items = [s for s in items if (s.status in open_states) == (scope == "open")]
        if actionable != "all":
            items = [s for s in items if s.actionable == (actionable == "true")]
        if symbol:
            items = [s for s in items if s.symbol == symbol]
        if before_id is not None:
            items = [s for s in items if s.id < before_id]
        return [state.signal_out(s) for s in items[:limit]]

    @app.get("/api/signals/stats")
    def get_signal_stats(since_ts: int | None = None) -> dto.SignalStatsOut:
        items = [s for s in state.signals.values() if since_ts is None or s.created_ts >= since_ts]
        return signal_stats(items).model_copy(update={"since_ts": since_ts})

    @app.get("/api/signals/{signal_id}")
    def get_signal(signal_id: int) -> dto.SignalDetailOut:
        sig = state.signals.get(signal_id)
        if sig is None:
            raise HTTPException(404, f"Signal #{signal_id} gibt es nicht.")
        return dto.SignalDetailOut(**state.signal_out(sig).model_dump(), components=sig.components, features=sig.features, news=sig.news)

    # -- portfolios -------------------------------------------------------------------------------
    def key_or_404(key: str) -> str:
        if key not in ("ai", "me"):
            raise HTTPException(404, f"Ein Depot „{key}“ gibt es nicht (ai oder me).")
        return key

    @app.get("/api/portfolios")
    def get_portfolios() -> list[dto.PortfolioSummaryOut]:
        return [state.broker.summary(k, state.now()) for k in ("ai", "me")]

    @app.get("/api/portfolios/equity")
    def get_equity(since_ts: int | None = None, max_points: int = Query(600, ge=10, le=5000)) -> dto.EquityCurvesOut:
        return state.broker.equity_curves(since_ts, max_points, state.now())

    @app.get("/api/portfolios/{key}")
    def get_portfolio(key: str) -> dto.PortfolioDetailOut:
        key_or_404(key)
        b = state.broker
        return dto.PortfolioDetailOut(summary=b.summary(key, state.now()), positions=[b.position_out(p) for p in b.open_positions(key)],
                                      pending_orders=[b.order_out(o) for o in b.orders.values() if o.key == key and o.status == "pending"])

    @app.get("/api/portfolios/{key}/trades")
    def get_trades(key: str, limit: int = Query(200, ge=1, le=1000)) -> list[dto.PositionOut]:
        key_or_404(key)
        closed = sorted((p for p in state.broker.positions.values() if p.key == key and p.status == "closed"), key=lambda p: -(p.closed_ts or 0))
        return [state.broker.position_out(p) for p in closed[:limit]]

    @app.get("/api/portfolios/{key}/orders")
    def get_orders(key: str, limit: int = Query(200, ge=1, le=1000)) -> list[dto.OrderOut]:
        key_or_404(key)
        orders = sorted((o for o in state.broker.orders.values() if o.key == key), key=lambda o: -o.created_ts)
        return [state.broker.order_out(o) for o in orders[:limit]]

    @app.post("/api/portfolios/me/orders", status_code=201)
    def post_order(body: dto.OrderRequest) -> dto.OrderOut:
        order = state.broker.place(body, state.now())
        hub.publish("order", state.broker.order_out(order))
        hub.publish("activity", state.add_activity("info", "sim", f"Mein Depot: Order #{order.id} {order.symbol} angenommen."))
        return state.broker.order_out(order)

    @app.delete("/api/portfolios/me/orders/{order_id}")
    def delete_order(order_id: int) -> dto.OrderOut:
        order = state.broker.cancel(order_id)
        hub.publish("order", state.broker.order_out(order))
        hub.publish("portfolio", {"key": "me"})
        return state.broker.order_out(order)

    @app.post("/api/portfolios/me/positions/{position_id}/close")
    def close_position(position_id: int) -> dto.OrderOut:
        order = state.broker.close_position(position_id, state.now())
        hub.publish("order", state.broker.order_out(order))
        return state.broker.order_out(order)

    @app.patch("/api/portfolios/me/positions/{position_id}")
    def patch_position(position_id: int, body: dto.PositionUpdateRequest) -> dto.PositionOut:
        pos = state.broker.update_levels(position_id, body)
        hub.publish("portfolio", {"key": "me"})
        return state.broker.position_out(pos)

    @app.post("/api/portfolios/reset")
    def reset(body: dto.ResetRequest) -> list[dto.PortfolioSummaryOut]:
        if not body.confirm:
            raise HTTPException(400, "Bitte den Neustart ausdrücklich bestätigen (confirm: true).")
        state.broker.reset(body.start_capital, state.now())
        hub.publish("portfolio", {"key": "ai"})
        hub.publish("portfolio", {"key": "me"})
        hub.publish("activity", state.add_activity("warning", "sim", "Beide Depots wurden neu gestartet."))
        return [state.broker.summary(k, state.now()) for k in ("ai", "me")]

    # -- settings, mail ---------------------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> RuntimeSettings:
        return state.settings

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any] = Body(...)) -> RuntimeSettings:
        unknown = sorted(set(body) - set(RuntimeSettings.model_fields))
        if unknown:
            raise HTTPException(422, f"Unbekannte Einstellung: {', '.join(unknown)}.")
        try:
            state.settings = RuntimeSettings.model_validate({**state.settings.model_dump(), **body})
        except ValidationError as exc:
            raise HTTPException(422, _german_validation(exc)) from exc
        return state.settings

    @app.post("/api/mail/test")
    def mail_test() -> dto.MailTestOut:
        entry = dto.MailLogOut(id=max(m.id for m in state.mail_log) + 1, ts=state.now(), kind="test", recipient="t***@example.org",
                               subject="Trading Local: Testmail", status="file", detail="data/outbox/testmail.eml")
        state.mail_log.insert(0, entry)
        return dto.MailTestOut(ok=True, status="file", detail_de="Testmail als Datei gespeichert: data/outbox/testmail.eml (Modus „Datei“).")

    @app.get("/api/mail/log")
    def get_mail_log(limit: int = Query(50, ge=1, le=500)) -> list[dto.MailLogOut]:
        return state.mail_log[:limit]

    # -- validation -------------------------------------------------------------------------------
    @app.get("/api/validation")
    def get_validation() -> list[dto.ValidationRunOut]:
        return sorted(state.validation_runs, key=lambda r: -r.id)

    @app.post("/api/validation/run", status_code=202)
    async def run_validation(body: dto.ValidationRequest) -> dto.ValidationRunOut:
        if any(r.status == "running" for r in state.validation_runs):
            raise HTTPException(409, "Eine Validierung läuft bereits.")
        run = dto.ValidationRunOut(id=max(r.id for r in state.validation_runs) + 1, ts=state.now(), status="running",
                                   engine_version=ENGINE_VERSION, params={"days": body.days})
        state.validation_runs.append(run)

        async def finish() -> None:
            await asyncio.sleep(8)
            done = run.model_copy(update={"status": "done", "result": build_validation_result(body.days, state.now())})
            state.validation_runs[state.validation_runs.index(run)] = done
            hub.publish("validation", {"id": run.id, "status": "done"})
            hub.publish("activity", state.add_activity("info", "system", f"Validierung #{run.id} über {body.days} Tage fertig."))

        spawn(finish())
        return run

    # -- operations -------------------------------------------------------------------------------
    @app.get("/api/activity")
    def get_activity(limit: int = Query(100, ge=1, le=1000), level: Literal["info", "warning", "error"] | None = None,
                     category: str | None = None, before_id: int | None = None) -> list[dto.ActivityOut]:
        items = [a for a in state.activity if (level is None or a.level == level) and (category is None or a.category == category)
                 and (before_id is None or a.id < before_id)]
        return items[:limit]

    @app.get("/api/system/llm-calls")
    def get_llm_calls() -> list[dto.LlmCallOut]:
        return state.llm_calls

    @app.get("/api/system/feeds")
    def get_feeds() -> list[dto.FeedStateOut]:
        return state.feeds

    @app.get("/api/system/jobs")
    def get_jobs() -> list[dto.JobStatusOut]:
        return state.jobs()

    # -- websocket --------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        hub.clients.add(queue)
        subs: set[str] = set()
        await ws.send_json({"type": "hello", "data": {"server_time": state.now(), "version": VERSION}})

        async def receive() -> None:
            while True:
                msg = await ws.receive_json()
                if isinstance(msg, dict) and msg.get("type") == "subscribe" and isinstance(msg.get("symbols"), list):
                    subs.clear()
                    subs.update(s for s in msg["symbols"][:MAX_BAR_SYMBOLS] if isinstance(s, str) and s in SPECS)

        async def forward() -> None:
            while True:
                await ws.send_json(await queue.get())

        tasks = [asyncio.create_task(receive()), asyncio.create_task(forward())]
        if live:
            tasks.append(asyncio.create_task(client_ticker(ws, state, subs)))
        try:
            # the first task to stop (disconnect, failed send) ends the connection
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            hub.clients.discard(queue)
            for task in tasks:
                task.cancel()

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _parse_now(value: str) -> datetime | None:
    if value == "real":
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--now braucht eine Zeitzone, z. B. 2026-11-19T15:12:00Z")
    return parsed.astimezone(UTC)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python tools/mock_api.py", description="Demo-Server für die Oberfläche von Trading Local")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--now", type=_parse_now, default=DEFAULT_DEMO_NOW,
                        help="Startzeit der Demo-Uhr (ISO mit Zeitzone) oder 'real' für die echte Uhr")
    parser.add_argument("--offline", action="store_true", help="Status meldet den Offline-Modus (Banner testen)")
    args = parser.parse_args(argv)
    app = create_app(args.now, offline=args.offline)
    print(f"Trading Local Demo-Server: http://{args.host}:{args.port}/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
