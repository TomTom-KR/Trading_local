"""Price trigger: an unusual price move WITH unusual volume pulls the instrument's news forward.

News explain moves - and a move that nobody has explained yet is the moment when the analysis of the
instrument's headlines matters most. So, once a minute after the market data step, every active instrument
that received new closed 1-minute bars is checked on its newest bars:

  move     log return of the last ``WINDOW_MIN`` closed 1-minute bars (open of the first -> close of the
           last), in units of the instrument's standard deviation of 5-minute returns (``z``)
  volume   the volume of those bars relative to the median volume of the same time of day (``rel_volume``)
  trigger  |z| >= ``MOVE_Z_MIN`` AND rel_volume >= ``REL_VOLUME_MIN``

Baseline (``compute_baseline``): the 1-minute bars of the newest ``BASELINE_SESSIONS`` trading dates BEFORE
today (venue local date), cut into 5-minute buckets of the venue's local day. The spread is the sample standard
deviation of the bucket returns; overnight gaps never enter it, because a bucket only holds bars of one
trading date.

Thresholds, checked on the stored history of 2026-11-19 to 2026-11-24 (all active instruments with volume,
three trading days each): a robust spread (median absolute deviation) and 4 sigma fired 93 times a day - five-
minute returns have fat tails, and the thinly traded EUR crypto pairs have many flat buckets that shrink a
robust spread. The sample standard deviation with 6 sigma and three times the usual volume fired 15 times a
day, about once per instrument and week (a 99.9 % quantile of the past moves gave the same 16).
The volume reference of a time of day is the median volume of that bucket over the previous sessions (at
least ``MIN_BUCKET_DATES`` of them; otherwise the median of all buckets), so the busy first and last minutes
of a session are compared with busy first and last minutes. Baselines are cached for ``BASELINE_TTL_S``; at
most ``MAX_BASELINES_PER_SCAN`` are built per minute, so the first minutes after a start spread the work.

Guards: the window lies inside one regular session (its first and newest bar fall into open trading time and
it spans at most ``WINDOW_MIN`` minutes, so a lunch break or an overnight gap never counts as a move), at
least ``MIN_WINDOW_BARS`` bars, data not stale, and one trigger per instrument per ``COOLDOWN_S``.
Instruments without volume data (indices, FX pairs) are never triggered: an unusual move needs volume.

A trigger calls ``News.prioritize`` for the instrument and writes a German activity-log entry. It changes
nothing in the signal engine: the engine's decisions and the validation replay are unaffected.

Threading: all database and pandas work runs in one dedicated worker thread; ``on_minute`` only orchestrates.
"""

import asyncio
import functools
import logging
import math
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trading_local.analysis.engine import warmup_start_ts
from trading_local.analysis.readings import de_number
from trading_local.barstore import BarReader
from trading_local.clock import Clock, from_ts, to_ts
from trading_local.contracts import MarketCalendar, News
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS, InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.models import Instrument

log = logging.getLogger(__name__)

WINDOW_MIN: Final = 5  # minutes of the move window (newest closed 1-minute bars)
MIN_WINDOW_BARS: Final = 3  # fewer bars in the window (illiquid minutes): no measurement
BUCKET_MIN: Final = 5  # bucket length of the baseline (equal to the window)
MOVE_Z_MIN: Final = 6.0  # standard deviations of a 5-minute return
REL_VOLUME_MIN: Final = 3.0  # times the usual volume of that time of day
BASELINE_SESSIONS: Final = 10  # previous trading dates behind the baseline
BASELINE_TTL_S: Final = 1800
MIN_BASELINE_BUCKETS: Final = 60  # fewer bucket returns: no baseline (about one session of a US share)
MIN_BUCKET_DATES: Final = 3  # sessions needed for the time-of-day volume reference of a bucket
MAX_BASELINES_PER_SCAN: Final = 12
COOLDOWN_S: Final = 1800  # one trigger per instrument per half hour
STALE_GRACE_MIN: Final = 10  # newest bar older than feed delay + this: stale, not checked


# ------------------------------------------------------------------------------------------------
# Pure part
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Baseline:
    """What is usual for one instrument (see the module docstring)."""

    computed_ts: int
    sigma: float  # standard deviation of 5-minute log returns (> 0)
    volume_by_bucket: Mapping[int, float]  # local minute of the day // BUCKET_MIN -> median bucket volume
    volume_median: float  # median volume of all buckets; 0 means the instrument has no volume data
    buckets: int  # bucket returns behind ``sigma``


@dataclass(frozen=True, slots=True)
class Move:
    """The newest window of one instrument, measured against its baseline."""

    symbol: str
    ts: int  # close time of the newest bar of the window
    minutes: int  # wall-clock minutes from the first bar's open to the newest bar's close
    change_pct: float  # price change over the window in percent
    z: float  # log return in standard deviations of a 5-minute return (signed)
    rel_volume: float | None  # window volume / usual volume of that time of day (None: no volume data)


def _bucket_of_day(local: pd.DatetimeIndex) -> np.ndarray:
    return (np.asarray(local.hour) * 60 + np.asarray(local.minute)) // BUCKET_MIN


def compute_baseline(minutes: pd.DataFrame, *, timezone: str, before_day: date, computed_ts: int) -> Baseline | None:
    """The baseline from canonical 1-minute bars, using only trading dates before ``before_day``.

    Dates are the local dates in ``timezone`` (the venue's zone; UTC for 24/7 venues). None when fewer than
    ``MIN_BASELINE_BUCKETS`` usable bucket returns exist or their spread is zero.
    """
    if minutes.empty:
        return None
    local = pd.DatetimeIndex(pd.to_datetime(minutes.index.to_numpy(dtype=np.int64), unit="s", utc=True))
    local = local.tz_convert(timezone)
    frame = pd.DataFrame(
        {
            "day": local.date,
            "bucket": _bucket_of_day(local),
            "open": minutes["open"].to_numpy(dtype=float),
            "close": minutes["close"].to_numpy(dtype=float),
            "volume": minutes["volume"].to_numpy(dtype=float),
        }
    )
    frame = frame[frame["day"] < before_day]
    if frame.empty:
        return None
    buckets = frame.groupby(["day", "bucket"], sort=True).agg(
        open=("open", "first"), close=("close", "last"), volume=("volume", "sum")
    )
    valid = (buckets["open"] > 0) & (buckets["close"] > 0)
    returns = np.log(buckets.loc[valid, "close"].to_numpy() / buckets.loc[valid, "open"].to_numpy())
    returns = returns[np.isfinite(returns)]
    if len(returns) < MIN_BASELINE_BUCKETS:
        return None
    sigma = float(np.std(returns, ddof=1))
    if not (math.isfinite(sigma) and sigma > 0):
        return None
    volumes = buckets["volume"].clip(lower=0.0)
    per_bucket = volumes.groupby(level="bucket").agg(["median", "count"])
    usable = per_bucket[per_bucket["count"] >= MIN_BUCKET_DATES]
    return Baseline(
        computed_ts=computed_ts,
        sigma=sigma,
        volume_by_bucket=MappingProxyType({int(key): float(value) for key, value in usable["median"].items()}),
        volume_median=float(np.median(volumes.to_numpy())),
        buckets=len(returns),
    )


def measure(symbol: str, window: pd.DataFrame, baseline: Baseline, *, timezone: str) -> Move | None:
    """Measure the window (canonical 1-minute bars, oldest first) against the baseline; None with fewer than
    ``MIN_WINDOW_BARS`` bars or without positive prices."""
    if len(window) < MIN_WINDOW_BARS:
        return None
    first_open = float(window["open"].iloc[0])
    last_close = float(window["close"].iloc[-1])
    if not (first_open > 0 and last_close > 0):
        return None
    first_ts, last_ts = int(window.index[0]), int(window.index[-1])
    end_ts = last_ts + BAR_SECONDS
    middle = from_ts((first_ts + end_ts) // 2).astimezone(ZoneInfo(timezone))
    usual = baseline.volume_by_bucket.get((middle.hour * 60 + middle.minute) // BUCKET_MIN, baseline.volume_median)
    volume = float(window["volume"].clip(lower=0.0).sum())
    return Move(
        symbol=symbol,
        ts=end_ts,
        minutes=(end_ts - first_ts) // 60,
        change_pct=(last_close / first_open - 1.0) * 100.0,
        z=math.log(last_close / first_open) / baseline.sigma,
        rel_volume=volume / usual if usual > 0 else None,
    )


def is_unusual(move: Move) -> bool:
    """The trigger rule: a large move with large volume (no volume data: never)."""
    return abs(move.z) >= MOVE_Z_MIN and move.rel_volume is not None and move.rel_volume >= REL_VOLUME_MIN


def move_message_de(move: Move, requeued: int) -> str:
    """Activity-log text, e.g. 'Ungewöhnliche Kursbewegung: NVDA +2,4 % in 5 Min (5,1σ), Volumen 4,2-fach –
    Nachrichten vorgezogen, 3 Meldungen erneut eingereiht'."""
    volume = "" if move.rel_volume is None else f", Volumen {de_number(move.rel_volume, 1)}-fach"
    text = (
        f"Ungewöhnliche Kursbewegung: {move.symbol} {de_number(move.change_pct, 1, signed=True)} % in "
        f"{move.minutes} Min ({de_number(abs(move.z), 1)}σ){volume} – Nachrichten vorgezogen"
    )
    if requeued == 1:
        return text + ", 1 Meldung erneut eingereiht"
    if requeued > 1:
        return text + f", {requeued} Meldungen erneut eingereiht"
    return text


# ------------------------------------------------------------------------------------------------
# Service
# ------------------------------------------------------------------------------------------------
class MoveWatch:
    """The price trigger. ``on_minute`` and ``stop`` run on the event loop."""

    def __init__(
        self,
        *,
        db: Database,
        clock: Clock,
        bus: EventBus,
        calendar: MarketCalendar,
        instruments: InstrumentBook,
        runtime_settings: SettingsStore,
        news: News,
    ) -> None:
        self._db = db
        self._clock = clock
        self._bus = bus
        self._calendar = calendar
        self._instruments = instruments
        self._runtime_settings = runtime_settings
        self._news = news
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="price-trigger")
        self._closed = False
        # worker-thread state
        self._baselines: dict[str, tuple[int, Baseline | None]] = {}  # symbol -> (built at, baseline or None)
        self._last_seen: dict[str, int] = {}  # symbol -> open time of the newest bar already checked
        self._last_trigger: dict[str, int] = {}

    async def on_minute(self, now: datetime, updated: Collection[str]) -> list[Move]:
        """Check the instruments with new bars and act on unusual moves. Never raises; returns the triggers."""
        if self._closed or not updated:
            return []
        try:
            if not self._runtime_settings.get().price_trigger_enabled:
                return []
            now_ts = to_ts(now)
            moves = await self._run(self._scan, now_ts, sorted(set(updated)))
        except Exception:
            log.exception("price trigger: the scan failed")
            return []
        for move in moves:
            try:
                requeued = await self._news.prioritize([move.symbol], now)
                await self._run(self._record, move, requeued, now_ts)
            except Exception:
                log.exception("price trigger: acting on the move of %s failed", move.symbol)
        return moves

    async def stop(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def _run[T](self, func: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(func, *args))

    # -- worker thread --------------------------------------------------------------------------
    def _scan(self, now_ts: int, symbols: list[str]) -> list[Move]:
        reader = BarReader(self._db, self._instruments, self._calendar, now_ts)
        budget = MAX_BASELINES_PER_SCAN
        moves: list[Move] = []
        for symbol in symbols:
            instrument = self._instruments.get(symbol)
            if instrument is None or not instrument.active:
                continue
            if now_ts - self._last_trigger.get(symbol, -COOLDOWN_S) < COOLDOWN_S:
                continue
            try:
                move, budget = self._check(instrument, reader, now_ts, budget)
            except Exception:
                log.exception("price trigger: checking %s failed", symbol)
                continue
            if move is not None:
                self._last_trigger[symbol] = now_ts
                moves.append(move)
                log.info(
                    "price trigger: %s %+.2f %% in %d min (z %.1f, volume x%.1f)",
                    symbol, move.change_pct, move.minutes, move.z, move.rel_volume or 0.0,
                )
        return moves

    def _check(self, instrument: Instrument, reader: BarReader, now_ts: int, budget: int) -> tuple[Move | None, int]:
        """The unusual move of one instrument, or None; also returns the remaining baseline budget."""
        symbol = instrument.symbol
        last = reader.last_bar(symbol)
        if last is None or last.ts <= self._last_seen.get(symbol, -1):
            return None, budget
        if last.ts + BAR_SECONDS < now_ts - (instrument.delay_min + STALE_GRACE_MIN) * 60:
            return None, budget
        if not self._open(instrument, last.ts):
            return None, budget
        cached = self._baselines.get(symbol)
        if cached is None or now_ts - cached[0] >= BASELINE_TTL_S:
            if budget <= 0:
                return None, budget  # checked in a later minute (the bar is not marked as seen)
            budget -= 1
            cached = (now_ts, self._baseline(instrument, reader, now_ts))
            self._baselines[symbol] = cached
        self._last_seen[symbol] = last.ts
        baseline = cached[1]
        if baseline is None or baseline.volume_median <= 0:
            return None, budget
        window = reader.minutes(symbol, start_ts=last.ts - (WINDOW_MIN - 1) * BAR_SECONDS)
        if window.empty or not self._open(instrument, int(window.index[0])):
            return None, budget
        move = measure(symbol, window, baseline, timezone=instrument.timezone)
        return (move if move is not None and is_unusual(move) else None), budget

    def _baseline(self, instrument: Instrument, reader: BarReader, now_ts: int) -> Baseline | None:
        start_ts = warmup_start_ts(self._calendar, instrument.exchange, now_ts, sessions=BASELINE_SESSIONS + 1)
        minutes = reader.minutes(instrument.symbol, start_ts=start_ts)
        today = from_ts(now_ts).astimezone(ZoneInfo(instrument.timezone)).date()
        return compute_baseline(minutes, timezone=instrument.timezone, before_day=today, computed_ts=now_ts)

    def _open(self, instrument: Instrument, bar_ts: int) -> bool:
        try:
            return self._calendar.is_open(instrument.exchange, from_ts(bar_ts))
        except KeyError:
            return False

    def _record(self, move: Move, requeued: int, now_ts: int) -> None:
        repo_ops.log_activity(
            self._db,
            ts=now_ts,
            level="info",
            category="data",
            message=move_message_de(move, requeued),
            bus=self._bus,
        )
