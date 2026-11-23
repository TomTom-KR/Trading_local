"""Technical indicators.

Pure functions on pandas objects. Every function is CAUSAL: the value at row t depends only on rows <= t,
so computing an indicator on a full history and reading row t gives the same result as computing it on the
history truncated at t (apart from floating-point noise). Warm-up rows are NaN - never back-filled.

Recursive averages (EMA, Wilder's smoothing) follow the textbook definition: the first value is the simple
mean of the first ``period`` observations, every later value is the recursion. This matches the published
definitions (Wilder 1978, Appel's MACD) and the common charting packages. Because of the recursion, values
depend on where the frame starts; callers must provide enough warm-up history.

Inputs are the canonical bar frames of this code base (ascending index, float columns open/high/low/close/volume).
All periods are fixed textbook defaults chosen by the caller; nothing in here is fitted to data.
"""

import numpy as np
import pandas as pd


def _seeded_average(series: pd.Series, alpha: float, period: int) -> pd.Series:
    """Recursion ``y_t = alpha * x_t + (1 - alpha) * y_(t-1)`` seeded with a simple mean.

    The seed is the mean of the first ``period`` non-NaN observations and sits on the row of the last of
    them; all earlier rows are NaN (leading NaNs of the input are skipped). Interior NaNs after the seed
    keep the last value and are bridged like pandas ``ewm(adjust=False)`` does; the inputs used in this
    code base have none.
    """
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period}")
    values = series.to_numpy(dtype=np.float64)
    observed = np.flatnonzero(~np.isnan(values))
    if observed.size < period:
        return pd.Series(np.nan, index=series.index, dtype=np.float64)
    seed_row = observed[period - 1]
    seeded = values.copy()
    seeded[:seed_row] = np.nan
    seeded[seed_row] = values[observed[:period]].mean()
    return pd.Series(seeded, index=series.index).ewm(alpha=alpha, adjust=False).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (alpha = 2 / (span + 1)), seeded with the SMA of the first ``span`` values.

    NaN for the first span - 1 rows.
    """
    return _seeded_average(series, 2.0 / (span + 1.0), span)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over ``window`` rows."""
    return series.rolling(window, min_periods=window).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): alpha = 1 / period, seeded with the SMA of the first ``period`` observations."""
    return _seeded_average(series, 1.0 / period, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder), 0..100. 50 when there was no movement at all.

    The first value is on row ``period`` (the first change needs a previous close).
    """
    delta = close.diff()
    avg_gain = wilder(delta.clip(lower=0.0), period)
    avg_loss = wilder((-delta).clip(lower=0.0), period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    return out.mask(flat, 50.0)


def true_range(frame: pd.DataFrame) -> pd.Series:
    """True range: max(high - low, |high - previous close|, |low - previous close|).

    The first row has no previous close; its true range is high - low.
    """
    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - prev_close).abs(), (frame["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder), in price units. First value on row period - 1."""
    return wilder(true_range(frame), period)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram (columns ``macd``, ``signal``, ``hist``).

    The signal line is the EMA of the MACD line, seeded with the mean of its first ``signal`` values.
    """
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands (columns ``mid``, ``upper``, ``lower``) with the population standard deviation."""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    return pd.DataFrame({"mid": mid, "upper": mid + num_std * std, "lower": mid - num_std * std})


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder), 0..100: trend strength regardless of direction.

    Directional movement and true range need the previous bar, so they start on row 1; the first DI/DX
    value is on row ``period`` and the first ADX value on row ``2 * period - 1`` (Wilder's definition).
    """
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0).mask(up.isna())
    minus_dm = down.where((down > up) & (down > 0.0), 0.0).mask(down.isna())
    smoothed_tr = wilder(true_range(frame).mask(frame["close"].shift(1).isna()), period)
    no_range = smoothed_tr == 0.0  # completely flat market: no direction at all
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = (100.0 * wilder(plus_dm, period) / smoothed_tr).mask(no_range, 0.0)
        minus_di = (100.0 * wilder(minus_dm, period) / smoothed_tr).mask(no_range, 0.0)
        total = plus_di + minus_di
        dx = 100.0 * (plus_di - minus_di).abs() / total
    dx = dx.mask(total == 0.0, 0.0)
    return wilder(dx, period)


def donchian(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Highest high / lowest low of the PREVIOUS ``window`` bars (columns ``upper``, ``lower``).

    The current bar is excluded, so ``close > upper`` is a genuine breakout test.
    """
    upper = frame["high"].rolling(window, min_periods=window).max().shift(1)
    lower = frame["low"].rolling(window, min_periods=window).min().shift(1)
    return pd.DataFrame({"upper": upper, "lower": lower})


def session_vwap(frame: pd.DataFrame, session: pd.Series) -> pd.Series:
    """Volume-weighted average price that restarts with every session id.

    ``session`` is aligned with ``frame`` and holds one label per row (e.g. the trading date). While a session
    has no volume yet (indices and FX never have any) the running mean of the typical price is used instead.
    """
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    keys = session.to_numpy()
    volume = frame["volume"].fillna(0.0)
    cum_volume = volume.groupby(keys).cumsum()
    cum_pv = (typical * volume).groupby(keys).cumsum()
    running_mean = typical.groupby(keys).cumsum() / (typical.groupby(keys).cumcount() + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = cum_pv / cum_volume
    return vwap.where(cum_volume > 0.0, running_mean)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score over ``window`` rows (0 where the window has no variation)."""
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (series - mean) / std
    return out.mask(std == 0.0, 0.0)


def slope(series: pd.Series, window: int) -> pd.Series:
    """Average change per row over the last ``window`` rows."""
    return (series - series.shift(window)) / float(window)
