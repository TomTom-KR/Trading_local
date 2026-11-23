"""Causal feature rows for the signal engine (one code path for live operation and validation).

``build_features`` turns closed 1-minute bars plus daily bars into one row per completed fast-timeframe
bucket. The row of decision time t (the END of the bucket) is built only from

* 1-minute bars that had closed at t (``ts + 60 <= t``),
* slow-timeframe buckets that had ended at t,
* daily bars of trading dates strictly before the trading date of the row (one-session lag) - and only
  when the venue's previous session is among them; otherwise the daily columns are NaN (a missing or not
  yet fetched daily bar never lets the day before pass for "yesterday"),
* benchmark bars that had closed at t,

so computing the features on a full history and reading row t gives the same values as computing them on
the history truncated at t. ``tests/analysis/test_no_lookahead.py`` proves this by perturbing the future.

Sessions come from the ``MarketCalendar``: the regular trading time around the frame is fetched once, cut
into pieces that each belong to one trading date (24/7 venues: the UTC date), and every row is placed in
those pieces with ``numpy.searchsorted`` - no per-row calendar calls.

``news_context`` condenses the analysed news of one symbol into a time-decayed, shrunk sentiment score
(rows filtered on the time their analysis became known, ages counted from the first sight of the item).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_local.analysis import indicators as ind
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.barstore import resample
from trading_local.clock import from_ts, to_ts
from trading_local.contracts import MarketCalendar
from trading_local.db.repo_market import BAR_SECONDS, bar_arrays
from trading_local.db.repo_news import SymbolNews
from trading_local.models import Instrument

FEATURE_COLUMNS: tuple[str, ...] = (
    "price",
    "data_ts",
    "session",
    "minutes_since_open",
    "minutes_to_close",
    "in_session",
    "atr_fast",
    "atr_slow",
    "atr_daily",
    "ema_fast_f",
    "ema_slow_f",
    "ema_fast_s",
    "ema_slow_s",
    "ema_fast_s_slope",
    "rsi_fast",
    "rsi_slow",
    "rsi_daily",
    "macd_hist_s",
    "adx_slow",
    "bb_mid_s",
    "bb_upper_s",
    "bb_lower_s",
    "vwap",
    "prev_high_f",
    "prev_low_f",
    "prev_price_f",
    "high_f",
    "low_f",
    "open_f",
    "pullback_low",
    "pullback_high",
    "donchian_upper_s",
    "donchian_lower_s",
    "day_high",
    "day_low",
    "or_high",
    "or_low",
    "prev_day_high",
    "prev_day_low",
    "prev_day_close",
    "gap_pct",
    "ema_daily_fast",
    "ema_daily_slow",
    "ema_daily_pullback",
    "rel_volume",
    "atr_pct_rank",
    "ret_60",
    "bench_trend",
    "bench_ret_60",
    "rel_strength_60",
)

_DAY_SECONDS = 86_400
_EPOCH = date(1970, 1, 1)
# Calendar days of sessions fetched before the first bar: enough to know the session before the first row's
# trading date across the longest regular closures (Chinese New Year, Golden Week: about 9 days).
_SESSION_LOOKBACK_DAYS = 14
_RANK_CHUNK_CELLS = 2_000_000  # bounds the memory of the volatility-rank window comparison


# ------------------------------------------------------------------------------------------------
# Small vectorised helpers
# ------------------------------------------------------------------------------------------------
def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise quotient; NaN where the denominator is not a positive number."""
    out = np.full(np.broadcast(numerator, denominator).shape, np.nan)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``values[index]`` with NaN where ``index`` is negative (nothing known yet)."""
    if values.size == 0:
        return np.full(index.shape, np.nan)
    out = values[np.clip(index, 0, None)].astype(np.float64)
    out[index < 0] = np.nan
    return out


def _asof_close(ts: np.ndarray, close: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Close of the newest 1-minute bar that had closed at each instant of ``at`` (NaN if none)."""
    return _take(close, np.searchsorted(ts, at - BAR_SECONDS, side="right") - 1)


def _pct_change(now: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Return from ``before`` to ``now`` in percent (NaN without a positive reference)."""
    return (_ratio(now, before) - 1.0) * 100.0


def _day_number(value: date) -> int:
    return (value - _EPOCH).days


def feature_float(row: Mapping[str, Any], key: str) -> float:
    """A feature of a row (dict or pandas Series) as float; NaN when missing or not a real number.

    Booleans are not numbers here (``in_session`` is read as a flag, never as 0/1).
    """
    value = row.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        return math.nan
    return float(value)


# ------------------------------------------------------------------------------------------------
# Sessions from the calendar
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Sessions:
    """Regular trading time around a frame, cut into pieces that each belong to ONE trading date.

    All arrays are aligned and ascending in time; times are epoch seconds, trading dates are day numbers
    (days since 1970-01-01).
    """

    start: np.ndarray  # piece start
    end: np.ndarray  # piece end (exclusive)
    day: np.ndarray  # trading date of the piece
    before: np.ndarray  # trading seconds of the same trading date before the piece
    total: np.ndarray  # trading seconds of the whole trading date

    def locate(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(piece index, inside) for each instant: the last piece starting at or before it, and whether
        the instant lies inside that piece. The index is clipped to 0 where no piece precedes."""
        if self.start.size == 0:
            return np.zeros(ts.shape, dtype=np.int64), np.zeros(ts.shape, dtype=bool)
        index = np.searchsorted(self.start, ts, side="right") - 1
        safe = np.clip(index, 0, None)
        return safe, (index >= 0) & (ts < self.end[safe])

    def elapsed(self, piece: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Trading seconds of the piece's trading date that have passed at ``ts`` (clipped to the piece end)."""
        return self.before[piece] + np.minimum(ts, self.end[piece]) - self.start[piece]

    def trading_day(self, ts: np.ndarray, timezone: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(trading date, piece index, inside) of each instant.

        Inside a regular segment the trading date is the one the calendar gave the piece; outside it is the
        local calendar date in the venue's time zone.
        """
        piece, inside = self.locate(ts)
        day = _local_days(ts, timezone)
        day[inside] = self.day[piece[inside]]
        return day, piece, inside


def _sessions(calendar: MarketCalendar, exchange: str, first_ts: int, last_ts: int) -> _Sessions:
    """Fetch the regular segments from ``_SESSION_LOOKBACK_DAYS`` days before ``first_ts`` to two days after
    ``last_ts``.

    Segments are cut at every UTC midnight and each piece gets the trading date of its start, so a 24/7
    venue whose calendar reports one long segment still gets one trading date per UTC day, while a venue
    whose session crosses UTC midnight (e.g. Sydney) keeps one trading date for both pieces. The margins
    make sure the sessions of the first and last rows are complete (not clipped by the query range) and
    that the trading session BEFORE the first row's date is known, even after a long holiday closure
    (``_daily_columns`` checks that the daily bar it uses is that session's).
    """
    query_start = (first_ts // _DAY_SECONDS - _SESSION_LOOKBACK_DAYS) * _DAY_SECONDS
    query_end = (last_ts // _DAY_SECONDS + 2) * _DAY_SECONDS
    starts: list[int] = []
    ends: list[int] = []
    days: list[int] = []
    for segment in calendar.segments(exchange, from_ts(query_start), from_ts(query_end)):
        begin, finish = to_ts(segment.start), to_ts(segment.end)
        while begin < finish:
            cut = min(finish, (begin // _DAY_SECONDS + 1) * _DAY_SECONDS)
            starts.append(begin)
            ends.append(cut)
            days.append(_day_number(calendar.session_date(exchange, from_ts(begin))))
            begin = cut
    start = np.asarray(starts, dtype=np.int64)
    end = np.asarray(ends, dtype=np.int64)
    day = np.asarray(days, dtype=np.int64)
    if start.size == 0:
        return _Sessions(start, end, day, start.copy(), start.copy())
    duration = end - start
    first_of_day = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
    group = np.cumsum(np.r_[True, day[1:] != day[:-1]]) - 1
    passed = np.cumsum(duration) - duration  # trading seconds of all earlier pieces
    before = passed - passed[first_of_day][group]
    total = np.add.reduceat(duration, first_of_day)[group]
    return _Sessions(start, end, day, before, total)


def _local_days(ts: np.ndarray, timezone: str) -> np.ndarray:
    """Calendar date (day number) of each instant in the venue's time zone."""
    stamps = pd.to_datetime(ts, unit="s", utc=True).tz_convert(timezone).tz_localize(None)
    return stamps.to_numpy().astype("datetime64[D]").astype(np.int64)


# ------------------------------------------------------------------------------------------------
# Column groups
# ------------------------------------------------------------------------------------------------
def _fast_columns(fast: pd.DataFrame, params: EngineParams) -> dict[str, np.ndarray]:
    """Bucket prices and fast-timeframe indicators."""
    close = fast["close"]
    lookback = params.pullback_buckets
    return {
        "price": close.to_numpy(),
        "open_f": fast["open"].to_numpy(),
        "high_f": fast["high"].to_numpy(),
        "low_f": fast["low"].to_numpy(),
        "prev_high_f": fast["high"].shift(1).to_numpy(),
        "prev_low_f": fast["low"].shift(1).to_numpy(),
        "prev_price_f": close.shift(1).to_numpy(),
        "pullback_low": fast["low"].rolling(lookback, min_periods=lookback).min().to_numpy(),
        "pullback_high": fast["high"].rolling(lookback, min_periods=lookback).max().to_numpy(),
        "atr_fast": ind.atr(fast, params.atr_period).to_numpy(),
        "ema_fast_f": ind.ema(close, params.ema_fast).to_numpy(),
        "ema_slow_f": ind.ema(close, params.ema_slow).to_numpy(),
        "rsi_fast": ind.rsi(close, params.rsi_period).to_numpy(),
    }


def _slow_columns(
    minutes: pd.DataFrame,
    bucket_start: np.ndarray,
    decision: np.ndarray,
    sessions: _Sessions,
    timezone: str,
    params: EngineParams,
) -> dict[str, np.ndarray]:
    """Slow-timeframe indicators as of each decision time.

    Values come from the newest slow bucket that had ENDED at the decision time. The Donchian channel is
    read at the slow bucket that CONTAINS the fast bucket: it spans the ``donchian_window`` slow buckets
    before it, all complete, and never includes the bars of the current fast bucket.
    """
    slow = resample(minutes, params.slow_minutes)  # the newest bucket may still be forming; it is never read
    step = params.slow_minutes * BAR_SECONDS
    slow_start = slow.index.to_numpy(dtype=np.int64)
    close = slow["close"]
    atr_slow = ind.atr(slow, params.atr_period).to_numpy()
    ema_fast_series = ind.ema(close, params.ema_fast)
    ema_fast = ema_fast_series.to_numpy()
    ema_slow = ind.ema(close, params.ema_slow).to_numpy()
    slope = _ratio((ema_fast_series - ema_fast_series.shift(params.slope_bars)).to_numpy(), atr_slow)
    macd_hist = ind.macd(close, params.macd_fast, params.macd_slow, params.macd_signal)["hist"].to_numpy()
    bands = ind.bollinger(close, params.bb_window, params.bb_std)
    channel = ind.donchian(slow, params.donchian_window)
    slow_day, _, _ = sessions.trading_day(slow_start, timezone)
    rank = _volatility_rank(atr_slow, slow_day, params.atr_rank_sessions, params.atr_rank_min_bars)

    ended = np.searchsorted(slow_start + step, decision, side="right") - 1
    containing = np.searchsorted(slow_start, bucket_start // step * step)
    return {
        "atr_slow": _take(atr_slow, ended),
        "ema_fast_s": _take(ema_fast, ended),
        "ema_slow_s": _take(ema_slow, ended),
        "ema_fast_s_slope": _take(slope, ended),
        "rsi_slow": _take(ind.rsi(close, params.rsi_period).to_numpy(), ended),
        "macd_hist_s": _take(_ratio(macd_hist, atr_slow), ended),
        "adx_slow": _take(ind.adx(slow, params.adx_period).to_numpy(), ended),
        "bb_mid_s": _take(bands["mid"].to_numpy(), ended),
        "bb_upper_s": _take(bands["upper"].to_numpy(), ended),
        "bb_lower_s": _take(bands["lower"].to_numpy(), ended),
        "donchian_upper_s": _take(channel["upper"].to_numpy(), containing),
        "donchian_lower_s": _take(channel["lower"].to_numpy(), containing),
        "atr_pct_rank": _take(rank, ended),
    }


def _volatility_rank(atr: np.ndarray, day: np.ndarray, sessions_window: int, min_count: int) -> np.ndarray:
    """Percentile rank (0..1, ties count half) of each ATR value among the ATR values of the slow buckets
    of the trailing ``sessions_window`` trading dates up to and including the current bucket.

    NaN while fewer than ``min_count`` values are available. Vectorised with a sliding window; rows are
    processed in chunks so that memory stays bounded for long frames.
    """
    n = atr.size
    out = np.full(n, np.nan)
    if n == 0:
        return out
    # Dense rank of the trading date; the running maximum keeps it ascending even if a bar outside the
    # regular session carried an earlier local date.
    _, day_rank = np.unique(np.maximum.accumulate(day), return_inverse=True)
    first = np.searchsorted(day_rank, day_rank - (sessions_window - 1), side="left")
    width = int((np.arange(n) - first).max()) + 1
    windows = sliding_window_view(np.r_[np.full(width - 1, np.nan), atr], width)  # row i ends at atr[i]
    chunk = max(1, _RANK_CHUNK_CELLS // width)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        rows = np.arange(lo, hi)
        block = windows[lo:hi]
        # offset k of row i holds atr[i - width + 1 + k]; it belongs to the window if that index >= first[i]
        member = np.arange(width)[None, :] >= (first[lo:hi] - rows + width - 1)[:, None]
        current = atr[lo:hi, None]
        valid = member & ~np.isnan(block)
        count = valid.sum(axis=1)
        less = (member & (block < current)).sum(axis=1)
        equal = (member & (block == current)).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rank = (less + 0.5 * equal) / count
        rank[(count < min_count) | np.isnan(atr[lo:hi])] = np.nan
        out[lo:hi] = rank
    return out


def _session_columns(
    bucket_start: np.ndarray, decision: np.ndarray, sessions: _Sessions, timezone: str, params: EngineParams
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Trading date, session flags and trading minutes of each row.

    Returns the columns plus the trading date (day number) and the time-of-day slot of every row
    (-1 outside regular sessions). Inside a session the trading date is the date of the calendar piece
    (``calendar.session_date``); outside it is the local date in the venue's time zone.

    ``minutes_since_open`` / ``minutes_to_close`` are measured at the decision time (bucket end) in trading
    minutes of the row's trading date; for the last bucket of a segment the decision falls on the segment
    end, where the count continues (``MarketCalendar.minutes_since_open`` would report None at that instant).
    """
    day, piece, inside = sessions.trading_day(bucket_start, timezone)
    since = np.full(bucket_start.shape, np.nan)
    to_close = np.full(bucket_start.shape, np.nan)
    slot = np.full(bucket_start.shape, -1, dtype=np.int64)
    if inside.any():
        p = piece[inside]
        elapsed = sessions.elapsed(p, decision[inside])
        since[inside] = elapsed / 60.0
        to_close[inside] = (sessions.total[p] - elapsed) / 60.0
        slot[inside] = sessions.elapsed(p, bucket_start[inside]) // (params.decision_minutes * BAR_SECONDS)
    columns = {
        "session": np.datetime_as_string(day.astype("datetime64[D]"), unit="D"),
        "in_session": inside,
        "minutes_since_open": since,
        "minutes_to_close": to_close,
    }
    return columns, day, slot


def _intraday_session_columns(
    ts: np.ndarray,
    values: np.ndarray,
    decision: np.ndarray,
    row_day: np.ndarray,
    sessions: _Sessions,
    params: EngineParams,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Running day extremes, session VWAP and opening range from REGULAR-session 1-minute bars.

    For each row only the bars of its own trading date that had closed at the decision time are used.
    Returns the columns plus the open of the trading date's first regular bar (NaN until that bar closed).
    """
    nan = np.full(decision.shape, np.nan)
    piece, inside = sessions.locate(ts)
    if not inside.any():
        return {"day_high": nan, "day_low": nan, "vwap": nan, "or_high": nan, "or_low": nan}, nan.copy()
    reg_ts = ts[inside]
    reg = values[inside]
    reg_piece = piece[inside]
    reg_day = sessions.day[reg_piece]
    frame = pd.DataFrame(reg, columns=["open", "high", "low", "close", "volume"])
    day_high = frame["high"].groupby(reg_day).cummax().to_numpy()
    day_low = frame["low"].groupby(reg_day).cummin().to_numpy()
    vwap = ind.session_vwap(frame, pd.Series(reg_day)).to_numpy()

    last = np.searchsorted(reg_ts, decision - BAR_SECONDS, side="right") - 1
    same_day = (last >= 0) & (reg_day[np.clip(last, 0, None)] == row_day)
    known = np.where(same_day, last, -1)

    # Opening range: bars that opened within the first opening_range_min trading minutes of the date.
    window = params.opening_range_min * BAR_SECONDS
    in_range = sessions.elapsed(reg_piece, reg_ts) < window
    range_high = pd.Series(reg[in_range, 1]).groupby(reg_day[in_range]).max()
    range_low = pd.Series(reg[in_range, 2]).groupby(reg_day[in_range]).min()
    range_end = _opening_range_end(sessions, window)
    available = decision >= range_end.reindex(row_day).to_numpy(dtype=np.float64, na_value=np.nan)

    first = np.flatnonzero(np.r_[True, reg_day[1:] != reg_day[:-1]])
    first_bar = pd.DataFrame({"ts": reg_ts[first], "open": reg[first, 0]}, index=reg_day[first])
    open_ts = first_bar["ts"].reindex(row_day).to_numpy(dtype=np.float64, na_value=np.nan)
    session_open = np.where(open_ts + BAR_SECONDS <= decision, first_bar["open"].reindex(row_day).to_numpy(), np.nan)
    columns = {
        "day_high": _take(day_high, known),
        "day_low": _take(day_low, known),
        "vwap": _take(vwap, known),
        "or_high": np.where(available, range_high.reindex(row_day).to_numpy(dtype=np.float64, na_value=np.nan), np.nan),
        "or_low": np.where(available, range_low.reindex(row_day).to_numpy(dtype=np.float64, na_value=np.nan), np.nan),
    }
    return columns, session_open


def _opening_range_end(sessions: _Sessions, window: int) -> pd.Series:
    """Instant at which ``window`` trading seconds of each trading date have passed (end of the session
    when it is shorter), indexed by trading date."""
    duration = sessions.end - sessions.start
    hit = (sessions.before <= window) & (window < sessions.before + duration)
    ends = pd.Series(sessions.end, index=sessions.day).groupby(level=0).max()  # fallback: end of the date
    ends.loc[sessions.day[hit]] = sessions.start[hit] + window - sessions.before[hit]
    return ends


def _previous_sessions(row_day: np.ndarray, sessions: _Sessions) -> np.ndarray:
    """Trading date (day number) of the venue's last session strictly before each row's trading date; -1
    where the calendar window holds none."""
    session_days = np.unique(sessions.day)
    if session_days.size == 0:
        return np.full(row_day.shape, -1, dtype=np.int64)
    index = np.searchsorted(session_days, row_day, side="left") - 1
    return np.where(index >= 0, session_days[np.clip(index, 0, None)], -1)


def _daily_columns(
    daily: pd.DataFrame, row_day: np.ndarray, previous_session: np.ndarray, params: EngineParams
) -> dict[str, np.ndarray]:
    """Daily features of the newest daily bar whose date is strictly before the row's trading date.

    That bar must belong to the venue's previous session (``previous_session``, from the calendar) or a
    later date. If it is older - the previous session's daily bar has not arrived yet (24/7 venues: shortly
    after UTC midnight) or is missing - all daily columns of the row are NaN instead of silently describing
    the day before: setups that need them do not trigger. Live operation and the replay pass the daily bars
    that were available at the decision, so both see the same rows.
    """
    names = (
        "prev_day_high", "prev_day_low", "prev_day_close", "ema_daily_fast", "ema_daily_slow",
        "ema_daily_pullback", "atr_daily", "rsi_daily",
    )  # fmt: skip
    if daily.empty:
        empty = np.full(row_day.shape, np.nan)
        return {name: empty.copy() for name in names}
    days = np.asarray(daily.index.astype(str), dtype="datetime64[D]").astype(np.int64)
    if np.any(days[1:] <= days[:-1]):
        raise ValueError("daily frame must be in ascending, unique day order")
    close = daily["close"]
    previous = np.searchsorted(days, row_day, side="left") - 1
    selected_day = np.where(previous >= 0, days[np.clip(previous, 0, None)], -1)
    stale = (previous_session < 0) | (selected_day < previous_session)
    previous = np.where(stale, -1, previous)  # _take gives NaN for a negative index
    return {
        "prev_day_high": _take(daily["high"].to_numpy(), previous),
        "prev_day_low": _take(daily["low"].to_numpy(), previous),
        "prev_day_close": _take(close.to_numpy(), previous),
        "ema_daily_fast": _take(ind.ema(close, params.ema_daily_fast).to_numpy(), previous),
        "ema_daily_slow": _take(ind.ema(close, params.ema_daily_slow).to_numpy(), previous),
        "ema_daily_pullback": _take(ind.ema(close, params.ema_daily_pullback).to_numpy(), previous),
        "atr_daily": _take(ind.atr(daily, params.atr_period).to_numpy(), previous),
        "rsi_daily": _take(ind.rsi(close, params.rsi_period).to_numpy(), previous),
    }


def _relative_volume(
    volume: np.ndarray, row_day: np.ndarray, slot: np.ndarray, params: EngineParams
) -> np.ndarray:
    """Bucket volume divided by a causal baseline.

    Baseline: median volume of the same time-of-day slot over the previous ``relvol_sessions`` trading
    dates (at least ``relvol_min_sessions`` of them must have that slot); otherwise - and outside regular
    sessions - the median of the previous ``relvol_fallback_buckets`` buckets. NaN where the baseline is
    zero or unknown, which is always the case for instruments without volume data (indices, FX).
    """
    baseline = np.full(volume.shape, np.nan)
    regular = slot >= 0
    if regular.any():
        days, day_pos = np.unique(row_day[regular], return_inverse=True)
        grid = np.full((days.size, int(slot[regular].max()) + 1), np.nan)
        grid[day_pos, slot[regular]] = volume[regular]
        medians = (
            pd.DataFrame(grid)
            .rolling(params.relvol_sessions, min_periods=params.relvol_min_sessions)
            .median()
            .shift(1)
            .to_numpy()
        )
        baseline[regular] = medians[day_pos, slot[regular]]
    recent = (
        pd.Series(volume)
        .shift(1)
        .rolling(params.relvol_fallback_buckets, min_periods=params.relvol_fallback_buckets)
        .median()
        .to_numpy()
    )
    baseline = np.where(np.isnan(baseline), recent, baseline)
    return _ratio(volume, baseline)


def _benchmark_columns(
    benchmark: pd.DataFrame | None, decision: np.ndarray, params: EngineParams
) -> dict[str, np.ndarray]:
    """Trend and 60-minute return of the benchmark as of each decision time (NaN without benchmark)."""
    nan = np.full(decision.shape, np.nan)
    if benchmark is None or benchmark.empty:
        return {"bench_trend": nan, "bench_ret_60": nan.copy()}
    ts, values = bar_arrays(benchmark)
    slow = resample(benchmark, params.slow_minutes)
    close = slow["close"]
    trend = _ratio(
        (ind.ema(close, params.ema_fast) - ind.ema(close, params.ema_slow)).to_numpy(),
        ind.atr(slow, params.atr_period).to_numpy(),
    )
    slow_end = slow.index.to_numpy(dtype=np.int64) + params.slow_minutes * BAR_SECONDS
    ended = np.searchsorted(slow_end, decision, side="right") - 1
    close_1m = values[:, 3]
    window = params.return_window_min * BAR_SECONDS
    return {
        "bench_trend": _take(trend, ended),
        "bench_ret_60": _pct_change(_asof_close(ts, close_1m, decision), _asof_close(ts, close_1m, decision - window)),
    }


def _empty_features() -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    for name in FEATURE_COLUMNS:
        dtype = {"session": "str", "in_session": "bool", "data_ts": "int64"}.get(name, "float64")
        columns[name] = pd.Series([], dtype=dtype)
    return pd.DataFrame(columns, index=pd.Index([], dtype="int64", name="ts"))


# ------------------------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------------------------
def build_features(
    minutes: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    instrument: Instrument,
    calendar: MarketCalendar,
    params: EngineParams = ENGINE_PARAMS,
    benchmark_minutes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature rows (columns ``FEATURE_COLUMNS``) for every COMPLETE fast bucket of ``minutes``.

    ``minutes``: canonical CLOSED 1-minute bars (several trading days of warm-up); ``daily``: canonical
    daily bars (index 'day'); ``benchmark_minutes``: closed 1-minute bars of the benchmark or None.

    The index ``ts`` is the decision time = bucket end (int64, ascending); a bucket is complete when its
    end is <= the open time of the newest bar + 60. Recursive indicators depend on where ``minutes`` and
    ``daily`` start, and the window features see only the sessions inside the frame (``rel_volume`` uses up
    to ``relvol_sessions`` earlier trading dates, ``atr_pct_rank`` the trailing ``atr_rank_sessions``
    dates), so live operation and validation must pass the same warm-up history to get identical values -
    at least ``atr_rank_sessions`` trading dates of minutes before a row for every column to be complete.
    Float columns are NaN while warming up; the daily columns (``prev_day_*``, ``gap_pct``, daily EMAs, ATR
    and RSI) are also NaN for rows whose previous trading session has no bar in ``daily``. ``session`` is the
    trading date ('YYYY-MM-DD'), ``in_session`` is True when the bucket starts inside a regular segment,
    ``data_ts`` equals the index.

    Benchmark values are as of the row's decision time in DATA time; if the benchmark's feed is more delayed
    than the instrument's, the caller passes the benchmark truncated accordingly. Raises ``ValueError`` for
    non-canonical frames and ``KeyError`` for an exchange the calendar does not know.
    """
    ts, values = bar_arrays(minutes)
    if ts.size == 0:
        return _empty_features()
    if np.any(ts[1:] <= ts[:-1]):
        raise ValueError("minute frame must be in ascending, unique time order")
    fast = resample(minutes, params.decision_minutes, complete_before_ts=int(ts[-1]) + BAR_SECONDS)
    if fast.empty:
        return _empty_features()
    bucket_start = fast.index.to_numpy(dtype=np.int64)
    decision = bucket_start + params.decision_minutes * BAR_SECONDS
    exchange = instrument.exchange
    timezone = calendar.timezone(exchange)
    sessions = _sessions(calendar, exchange, int(ts[0]), int(decision[-1]))

    columns: dict[str, np.ndarray] = {"data_ts": decision}
    session_cols, row_day, slot = _session_columns(bucket_start, decision, sessions, timezone, params)
    columns.update(session_cols)
    columns.update(_fast_columns(fast, params))
    columns.update(_slow_columns(minutes, bucket_start, decision, sessions, timezone, params))
    intraday, session_open = _intraday_session_columns(ts, values, decision, row_day, sessions, params)
    columns.update(intraday)
    columns.update(_daily_columns(daily, row_day, _previous_sessions(row_day, sessions), params))
    columns["gap_pct"] = _pct_change(session_open, columns["prev_day_close"])
    columns["rel_volume"] = _relative_volume(fast["volume"].to_numpy(), row_day, slot, params)
    window = params.return_window_min * BAR_SECONDS
    columns["ret_60"] = _pct_change(columns["price"], _asof_close(ts, values[:, 3], decision - window))
    columns.update(_benchmark_columns(benchmark_minutes, decision, params))
    columns["rel_strength_60"] = columns["ret_60"] - columns["bench_ret_60"]
    return pd.DataFrame({name: columns[name] for name in FEATURE_COLUMNS}, index=pd.Index(decision, name="ts"))


@dataclass(frozen=True, slots=True)
class NewsContext:
    """Condensed news situation of one symbol at one instant."""

    score: float  # -1..1 time-decayed, shrunk sentiment
    count: int  # analysed items inside the news window
    news_ids: tuple[int, ...]  # newest first, at most params.news_max_ids
    fresh: SymbolNews | None  # the newest item if it qualifies for the news setup


def news_age_ref_ts(row: SymbolNews) -> int:
    """Instant from which the AGE of a news row is counted: when the platform first saw the item.

    That is the item's ``fetched_ts`` (never later than ``analyzed_ts``). ``analyzed_ts`` is only the as-of
    filter - when the analysis became known - and can lie long after the first sight: items recovered after
    a restart are analysed up to two hours after they were fetched, queued items up to 20 minutes later.
    Rows that carry no usable ``fetched_ts`` fall back to ``analyzed_ts``. The result is never later than
    ``analyzed_ts``, so the reference can only make an item older, never let it pass a freshness filter.
    """
    fetched = getattr(row, "fetched_ts", None)
    if isinstance(fetched, (int, np.integer)) and not isinstance(fetched, bool) and fetched > 0:
        return min(int(fetched), int(row.analyzed_ts))
    return int(row.analyzed_ts)


def news_context(rows: Sequence[SymbolNews], *, asof_ts: int, params: EngineParams = ENGINE_PARAMS) -> NewsContext:
    """Time-decayed news sentiment of one symbol as known at ``asof_ts``.

    Per item ``w = (relevance / 5) * 0.5 ** (age_min / news_half_life_min)``; the score is
    ``sum(w * sentiment) / (1 + sum(w))`` clipped to [-1, 1]. The '1 +' is deliberate shrinkage: a single
    weak item cannot reach +-1. Rows analysed after ``asof_ts`` or not after ``asof_ts - news_window_min``
    are ignored (defence in depth: ``repo_news.symbol_news`` already filters), as are rows without a finite
    sentiment. ``fresh`` is the newest row if relevance >= news_min_relevance, |sentiment| >=
    news_min_abs_sentiment and its age is at most news_max_age_min minutes.

    Causality: WHICH rows count is decided by ``analyzed_ts`` (the time the analysis became known). The AGE
    of a row - decay weight and freshness - is measured from ``news_age_ref_ts`` (first sight of the item),
    so an item that was analysed late does not count as just published.
    """
    oldest = asof_ts - params.news_window_min * 60
    known = sorted(
        (row for row in rows if oldest < row.analyzed_ts <= asof_ts and math.isfinite(row.sentiment)),
        key=lambda row: (row.analyzed_ts, row.news_id),
        reverse=True,
    )
    if not known:
        return NewsContext(0.0, 0, (), None)
    weight_sum = 0.0
    weighted = 0.0
    for row in known:
        age_min = (asof_ts - news_age_ref_ts(row)) / 60.0
        weight = (min(max(row.relevance, 0), 5) / 5.0) * 0.5 ** (age_min / params.news_half_life_min)
        weight_sum += weight
        weighted += weight * min(max(row.sentiment, -1.0), 1.0)
    score = min(max(weighted / (1.0 + weight_sum), -1.0), 1.0)
    newest = known[0]
    qualifies = (
        newest.relevance >= params.news_min_relevance
        and abs(newest.sentiment) >= params.news_min_abs_sentiment
        and asof_ts - news_age_ref_ts(newest) <= params.news_max_age_min * 60
    )
    return NewsContext(
        score=score,
        count=len(known),
        news_ids=tuple(row.news_id for row in known[: params.news_max_ids]),
        fresh=newest if qualifies else None,
    )
