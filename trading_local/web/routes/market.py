"""Market data for the GUI: quotes, watchlist sparklines and chart bars with indicator overlays."""

import logging
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, Request

from trading_local import presenters
from trading_local.analysis import indicators
from trading_local.barstore import BarReader
from trading_local.clock import from_ts, to_ts
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_market
from trading_local.db.repo_market import BAR_COLUMNS, BAR_SECONDS
from trading_local.dto import BarsOut, QuoteOut, SparklinesOut
from trading_local.models import Instrument
from trading_local.services import Services
from trading_local.web.deps import MAX_TS, ApiContext, Ctx, TtlCache, require_instrument

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

Timeframe = Literal["1m", "5m", "15m", "1h", "1d"]
_INTRADAY_MINUTES: dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
_EMA_SPANS_INTRADAY = (20, 50)
_EMA_SPANS_DAILY = (20, 50, 200)
_WARMUP_FACTOR = 3  # extra history per EMA span: the initial value's weight drops below 0.3 % (span 50)
_LABEL_STEP_S = 900  # session dates change at local midnight, which is always a multiple of 15 minutes in UTC

SPARKLINE_TTL_S = 60
SPARKLINE_BUCKET_S = 300
SPARKLINE_MAX_POINTS = 80


@router.get("/quotes")
async def list_quotes(ctx: Ctx) -> list[QuoteOut]:
    """Latest quotes from memory (no I/O), by symbol."""
    quotes = ctx.services.data.quotes()
    return [presenters.quote_out(quotes[symbol]) for symbol in sorted(quotes)]


async def get_sparkline_cache(request: Request) -> TtlCache[SparklinesOut]:
    cache: TtlCache[SparklinesOut] = request.app.state.sparklines
    return cache


@router.get("/sparklines")
async def get_sparklines(
    ctx: Ctx, cache: Annotated[TtlCache[SparklinesOut], Depends(get_sparkline_cache)]
) -> SparklinesOut:
    return await cache.get(lambda: ctx.run(compute_sparklines, ctx.services))


@router.get("/bars/{symbol}")
async def get_bars(
    ctx: Ctx,
    symbol: str,
    tf: Timeframe = "15m",
    limit: Annotated[int, Query(ge=1, le=2000)] = 500,
    before: Annotated[int | None, Query(ge=0, le=MAX_TS)] = None,
) -> BarsOut:
    """Closed bars only (``BarReader`` as of now, or as of ``before`` when paging back)."""
    instrument = require_instrument(ctx.services, symbol)
    return await ctx.run(load_bars, ctx.services, instrument, tf, limit, before)


# ------------------------------------------------------------------------------------------------
# Sparklines
# ------------------------------------------------------------------------------------------------
# One statement for all symbols: every (symbol, window) row is answered by one primary-key range seek, and
# SQLite returns the close of the newest bar of each 5-minute bucket (bare column next to MAX()).
_SPARKLINE_SQL = """
WITH windows (symbol, tag, start_ts, end_ts) AS (VALUES {values})
SELECT w.symbol, w.tag, b.ts / {step} * {step} AS bucket, b.close, MAX(b.ts)
FROM windows AS w
JOIN bars_1m AS b ON b.symbol = w.symbol AND b.ts >= w.start_ts AND b.ts < w.end_ts
GROUP BY w.symbol, w.tag, bucket
ORDER BY w.symbol, w.tag, bucket
"""
_CURRENT, _PREVIOUS = 0, 1  # window tags

type _Points = list[tuple[int, float]]  # [(bucket start, close)]


def compute_sparklines(services: Services) -> SparklinesOut:
    """Per active instrument the closes of its current trading session (or, while that has no closed bar
    yet or the venue is closed, of its last completed session) in 5-minute buckets, newest 80."""
    now = services.clock.now()
    closed_before = to_ts(now) - BAR_SECONDS + 1  # bars that had closed at `now`
    instruments = services.instruments.all()
    exchanges = {instrument.exchange for instrument in instruments}
    windows_by_exchange = {exchange: _sparkline_windows(services.calendar, exchange, now) for exchange in exchanges}
    rows: list[tuple[str, int, int, int]] = []
    for instrument in instruments:
        for tag, (start_ts, end_ts) in windows_by_exchange[instrument.exchange].items():
            rows.append((instrument.symbol, tag, start_ts, min(end_ts, closed_before)))
    buckets = _read_buckets(services, rows)
    series: dict[str, _Points] = {}
    for instrument in instruments:
        tagged = buckets.get(instrument.symbol, {})
        points = tagged.get(_CURRENT) or tagged.get(_PREVIOUS) or []
        series[instrument.symbol] = points[-SPARKLINE_MAX_POINTS:]
    return SparklinesOut(series=series)


def _read_buckets(services: Services, rows: Sequence[tuple[str, int, int, int]]) -> dict[str, dict[int, _Points]]:
    """symbol -> window tag -> [(bucket start, close)] ascending, from one query (one fetchall)."""
    if not rows:
        return {}
    sql = _SPARKLINE_SQL.format(values=", ".join(["(?, ?, ?, ?)"] * len(rows)), step=SPARKLINE_BUCKET_S)
    params = [value for row in rows for value in row]
    result: dict[str, dict[int, _Points]] = {}
    for symbol, tag, bucket, close, _ in services.db.query(sql, params):
        result.setdefault(symbol, {}).setdefault(tag, []).append((bucket, close))
    return result


def _sparkline_windows(calendar: MarketCalendar, exchange: str, now: datetime) -> dict[int, tuple[int, int]]:
    """[start, end) of the current session (only while the venue is in it) and of the last completed one."""
    try:
        days: dict[int, date | None] = {_PREVIOUS: calendar.last_completed_session(exchange, now)}
        if calendar.state(exchange, now).phase in ("open", "break"):
            days[_CURRENT] = calendar.session_date(exchange, now)
    except KeyError:
        log.warning("instrument venue %r is unknown to the trading calendar", exchange)
        return {}
    windows: dict[int, tuple[int, int]] = {}
    for tag, day in days.items():
        window = None if day is None else session_window(calendar, exchange, day, now)
        if window is not None:
            windows[tag] = window
    return windows


def session_window(calendar: MarketCalendar, exchange: str, day: date, now: datetime) -> tuple[int, int] | None:
    """Epoch seconds [start, end) spanned by the regular segments of trading date ``day`` up to ``now``.

    Segments are searched from 12 h before to 36 h after the UTC midnight of ``day`` (covers every time
    zone) and split at UTC midnights, so venues dated by the UTC day (24/7 markets) are cut to that day.
    """
    midnight = datetime.combine(day, time(), UTC)
    search_end = min(now, midnight + timedelta(hours=36))
    segments = calendar.segments(exchange, midnight - timedelta(hours=12), search_end)
    starts: list[datetime] = []
    ends: list[datetime] = []
    for segment in segments:
        for start, end in _split_at_midnights(segment.start, segment.end):
            if calendar.session_date(exchange, start) == day:
                starts.append(start)
                ends.append(end)
    if not starts:
        return None
    return to_ts(min(starts)), to_ts(max(ends))


def _split_at_midnights(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    pieces = []
    while start < end:
        next_midnight = datetime.combine(start.date() + timedelta(days=1), time(), UTC)
        pieces.append((start, min(end, next_midnight)))
        start = next_midnight
    return pieces


# ------------------------------------------------------------------------------------------------
# Chart bars
# ------------------------------------------------------------------------------------------------
def load_bars(services: Services, instrument: Instrument, tf: str, limit: int, before: int | None) -> BarsOut:
    """The newest ``limit`` closed bars of the timeframe (before ``before`` when given) with EMA overlays
    (warmed up on 3x the longest span of extra history) and the session VWAP for intraday timeframes.

    Daily bars carry the session date at 00:00 UTC as ``ts``. Without ``before`` the daily series ends with
    the current session, built from its closed 1-minute bars while it runs (see ``_with_current_session``),
    so the GUI's live updates have today's candle to extend. NaN values never appear in the overlays.
    """
    now_ts = services.clock.ts()
    asof_ts = now_ts if before is None else min(now_ts, before)
    reader = BarReader(services.db, services.instruments, services.calendar, asof_ts)
    spans = _EMA_SPANS_DAILY if tf == "1d" else _EMA_SPANS_INTRADAY
    lookback = limit + _WARMUP_FACTOR * max(spans)
    if tf == "1d":
        frame = reader.daily(instrument.symbol, limit=lookback)
        if before is None:
            frame = _with_current_session(services.calendar, reader, instrument, frame)
        ts = pd.to_datetime(frame.index, format="%Y-%m-%d", utc=True).as_unit("s").asi8
    else:
        frame = reader.resampled(instrument.symbol, _INTRADAY_MINUTES[tf], lookback_bars=lookback)
        ts = frame.index.to_numpy(dtype=np.int64)
    overlays = {f"ema{span}": indicators.ema(frame["close"], span).to_numpy() for span in spans}
    if tf != "1d" and len(frame):
        truncated = len(frame) >= lookback
        overlays["vwap"] = _session_vwap(services.calendar, instrument.exchange, frame, ts, truncated=truncated)
    visible = slice(max(len(frame) - limit, 0), None)
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy()[visible]
    return BarsOut(
        symbol=instrument.symbol,
        timeframe=tf,
        currency=instrument.currency,
        price_scale=instrument.price_scale,
        delay_min=instrument.delay_min,
        bars=list(zip(ts[visible].tolist(), *values.T.tolist(), strict=True)),
        overlays={name: _line(ts[visible], line[visible]) for name, line in overlays.items()},
    )


def _with_current_session(
    calendar: MarketCalendar, reader: BarReader, instrument: Instrument, daily: pd.DataFrame
) -> pd.DataFrame:
    """``daily`` plus one row for the venue's current trading date when the reader has no daily bar for it yet.

    ``BarReader.daily`` returns completed sessions only. The current session (still running, or just ended
    but its daily bar not stored yet) is aggregated from its closed 1-minute bars inside the session window:
    open = first open, high = max, low = min, close = last close, volume = sum. Display only; the decision
    code never sees this row. Without a closed minute bar in the session nothing is added.
    """
    now = from_ts(reader.asof_ts)
    try:
        day = calendar.session_date(instrument.exchange, now)
    except KeyError:
        return daily
    key = day.isoformat()
    if len(daily) and str(daily.index[-1]) >= key:
        return daily
    window = session_window(calendar, instrument.exchange, day, now)
    if window is None:
        return daily
    start_ts, end_ts = window
    minutes = reader.minutes(instrument.symbol, start_ts=start_ts)
    minutes = minutes[minutes.index < end_ts]
    if minutes.empty:
        return daily
    _, values = repo_market.bar_arrays(minutes)
    row = pd.DataFrame(
        [[values[0, 0], values[:, 1].max(), values[:, 2].min(), values[-1, 3], values[:, 4].sum()]],
        index=pd.Index([key], dtype="str", name=daily.index.name),
        columns=list(BAR_COLUMNS),
    )
    return pd.concat([daily, row]) if len(daily) else row


def _session_vwap(
    calendar: MarketCalendar, exchange: str, frame: pd.DataFrame, ts: np.ndarray, *, truncated: bool
) -> np.ndarray:
    """Session VWAP per bar. When the read was cut by the row limit, the oldest session may be incomplete:
    its values are blanked instead of showing a VWAP that misses the session's first bars."""
    labels = _session_labels(calendar, exchange, ts)
    vwap = indicators.session_vwap(frame, pd.Series(labels, index=frame.index)).to_numpy(copy=True)
    if truncated:
        vwap[labels == labels[0]] = np.nan
    return vwap


def _session_labels(calendar: MarketCalendar, exchange: str, ts: np.ndarray) -> np.ndarray:
    """Trading date ('YYYY-MM-DD') of every bar, asking the calendar once per quarter hour that has bars."""
    _, first_rows, inverse = np.unique(ts // _LABEL_STEP_S, return_index=True, return_inverse=True)
    labels = np.array([calendar.session_date(exchange, from_ts(int(ts[row]))).isoformat() for row in first_rows])
    return labels[inverse]


def _line(ts: np.ndarray, values: np.ndarray) -> list[tuple[int, float]]:
    finite = np.isfinite(values)
    return list(zip(ts[finite].tolist(), values[finite].tolist(), strict=True))


def build_sparkline_cache(ctx: ApiContext) -> TtlCache[SparklinesOut]:
    """The per-app sparkline cache (stored on ``app.state.sparklines`` by ``create_app``)."""
    return TtlCache(ctx.services.clock, SPARKLINE_TTL_S)
