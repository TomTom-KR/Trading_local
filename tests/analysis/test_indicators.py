"""Indicators against independent, straightforward loop implementations of the textbook definitions."""

import math

import numpy as np
import pandas as pd
import pytest

from trading_local.analysis import indicators as ind

NAN = math.nan


def random_bars(n: int, seed: int = 7, *, with_volume: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(0.004 * rng.standard_normal(n)))
    open_ = np.r_[100.0, close[:-1]] * np.exp(0.001 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1.0 + 0.003 * rng.random(n))
    low = np.minimum(open_, close) * (1.0 - 0.003 * rng.random(n))
    volume = rng.uniform(100.0, 1000.0, n) if with_volume else np.zeros(n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.Index(np.arange(n, dtype=np.int64) * 60, name="ts"),
    )


def flat_bars(n: int, price: float = 50.0) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 0.0},
        index=pd.Index(np.arange(n, dtype=np.int64) * 60, name="ts"),
    )


def assert_same(actual: pd.Series, expected: list[float], rtol: float = 1e-10) -> None:
    """Equal NaN positions and values within a relative tolerance."""
    got = actual.to_numpy(dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    assert got.shape == want.shape
    assert np.array_equal(np.isnan(got), np.isnan(want)), (np.flatnonzero(np.isnan(got)), np.flatnonzero(np.isnan(want)))
    mask = ~np.isnan(want)
    np.testing.assert_allclose(got[mask], want[mask], rtol=rtol, atol=1e-12)


# ------------------------------------------------------------------------------------------------
# Reference implementations (plain loops, written from the textbook definitions)
# ------------------------------------------------------------------------------------------------
def ref_sma_seeded(values: list[float], alpha: float, period: int) -> list[float]:
    """Recursive average seeded with the mean of the first `period` values (inputs may lead with NaN)."""
    out = [NAN] * len(values)
    seen: list[float] = []
    prev = NAN
    for i, x in enumerate(values):
        if math.isnan(prev):
            if math.isnan(x):
                continue
            seen.append(x)
            if len(seen) == period:
                prev = sum(seen) / period
                out[i] = prev
        else:
            prev = alpha * x + (1.0 - alpha) * prev
            out[i] = prev
    return out


def ref_ema(values: list[float], span: int) -> list[float]:
    return ref_sma_seeded(values, 2.0 / (span + 1.0), span)


def ref_rsi(close: list[float], period: int) -> list[float]:
    out = [NAN] * len(close)
    if len(close) <= period:
        return out
    gains = [max(close[i] - close[i - 1], 0.0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0.0) for i in range(1, len(close))]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(close)):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_gain == 0.0 and avg_loss == 0.0:
            out[i] = 50.0
        elif avg_loss == 0.0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def ref_true_range(frame: pd.DataFrame) -> list[float]:
    high, low, close = frame["high"].tolist(), frame["low"].tolist(), frame["close"].tolist()
    out = []
    for i in range(len(high)):
        if i == 0:
            out.append(high[0] - low[0])
        else:
            out.append(max(high[i], close[i - 1]) - min(low[i], close[i - 1]))
    return out


def ref_atr(frame: pd.DataFrame, period: int) -> list[float]:
    tr = ref_true_range(frame)
    out = [NAN] * len(tr)
    if len(tr) < period:
        return out
    value = sum(tr[:period]) / period
    out[period - 1] = value
    for i in range(period, len(tr)):
        value = (value * (period - 1) + tr[i]) / period
        out[i] = value
    return out


def ref_adx(frame: pd.DataFrame, period: int) -> list[float]:
    """Wilder's ADX with his running SUMS (TR14 = TR14 - TR14 / 14 + TR) - an independent formulation."""
    high, low, close = frame["high"].tolist(), frame["low"].tolist(), frame["close"].tolist()
    n = len(high)
    out = [NAN] * n
    tr, pdm, mdm = [NAN], [NAN], [NAN]
    for i in range(1, n):
        tr.append(max(high[i], close[i - 1]) - min(low[i], close[i - 1]))
        up, down = high[i] - high[i - 1], low[i - 1] - low[i]
        pdm.append(up if up > down and up > 0 else 0.0)
        mdm.append(down if down > up and down > 0 else 0.0)
    if n <= period:
        return out
    tr_sum, p_sum, m_sum = sum(tr[1 : period + 1]), sum(pdm[1 : period + 1]), sum(mdm[1 : period + 1])
    dxs: list[float] = []
    adx_value = NAN
    for i in range(period, n):
        if i > period:
            tr_sum = tr_sum - tr_sum / period + tr[i]
            p_sum = p_sum - p_sum / period + pdm[i]
            m_sum = m_sum - m_sum / period + mdm[i]
        pdi = 100.0 * p_sum / tr_sum if tr_sum > 0 else 0.0
        mdi = 100.0 * m_sum / tr_sum if tr_sum > 0 else 0.0
        dx = 100.0 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi > 0 else 0.0
        dxs.append(dx)
        if len(dxs) == period:
            adx_value = sum(dxs) / period
            out[i] = adx_value
        elif len(dxs) > period:
            adx_value = (adx_value * (period - 1) + dx) / period
            out[i] = adx_value
    return out


def ref_bollinger(close: list[float], window: int, k: float) -> tuple[list[float], list[float], list[float]]:
    mid, upper, lower = [NAN] * len(close), [NAN] * len(close), [NAN] * len(close)
    for i in range(window - 1, len(close)):
        chunk = close[i - window + 1 : i + 1]
        mean = sum(chunk) / window
        std = math.sqrt(sum((x - mean) ** 2 for x in chunk) / window)
        mid[i], upper[i], lower[i] = mean, mean + k * std, mean - k * std
    return mid, upper, lower


def ref_donchian(frame: pd.DataFrame, window: int) -> tuple[list[float], list[float]]:
    high, low = frame["high"].tolist(), frame["low"].tolist()
    upper, lower = [NAN] * len(high), [NAN] * len(high)
    for i in range(window, len(high)):
        upper[i] = max(high[i - window : i])
        lower[i] = min(low[i - window : i])
    return upper, lower


def ref_vwap(frame: pd.DataFrame, session: list[int]) -> list[float]:
    out = []
    pv = vol = typ_sum = 0.0
    count = 0
    current = None
    for (high, low, close, volume), label in zip(
        frame[["high", "low", "close", "volume"]].itertuples(index=False), session
    ):
        if label != current:
            pv = vol = typ_sum = 0.0
            count = 0
            current = label
        typical = (high + low + close) / 3.0
        pv += typical * volume
        vol += volume
        typ_sum += typical
        count += 1
        out.append(pv / vol if vol > 0 else typ_sum / count)
    return out


# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("span", [1, 5, 20, 50])
def test_ema_matches_sma_seeded_reference(span: int) -> None:
    close = random_bars(300)["close"]
    assert_same(ind.ema(close, span), ref_ema(close.tolist(), span))


def test_ema_warm_up_is_span_minus_one_rows_and_seed_is_the_sma() -> None:
    close = random_bars(60)["close"]
    out = ind.ema(close, 20)
    assert out.iloc[:19].isna().all()
    assert out.iloc[19] == pytest.approx(close.iloc[:20].mean(), rel=1e-12)


def test_ema_skips_leading_nans_and_short_series_stay_nan() -> None:
    values = [NAN, NAN, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert_same(ind.ema(pd.Series(values), 3), ref_ema(values, 3))
    assert ind.ema(pd.Series([1.0, 2.0]), 3).isna().all()


def test_sma() -> None:
    close = random_bars(50)["close"]
    expected = [NAN] * 9 + [sum(close.tolist()[i - 9 : i + 1]) / 10 for i in range(9, 50)]
    assert_same(ind.sma(close, 10), expected)


@pytest.mark.parametrize("period", [2, 14])
def test_rsi_matches_wilder_reference(period: int) -> None:
    close = random_bars(400, seed=3)["close"]
    out = ind.rsi(close, period)
    assert_same(out, ref_rsi(close.tolist(), period))
    assert out.iloc[:period].isna().all() and not math.isnan(out.iloc[period])


def test_rsi_extremes_and_flat_series() -> None:
    rising = pd.Series(np.arange(40, dtype=np.float64))
    assert ind.rsi(rising, 14).iloc[14:].eq(100.0).all()
    falling = pd.Series(np.arange(40, 0, -1, dtype=np.float64))
    assert ind.rsi(falling, 14).iloc[14:].eq(0.0).all()
    flat = flat_bars(40)["close"]
    assert ind.rsi(flat, 14).iloc[14:].eq(50.0).all()


def test_true_range_and_atr_match_reference() -> None:
    frame = random_bars(300, seed=11)
    assert_same(ind.true_range(frame), ref_true_range(frame))
    assert_same(ind.atr(frame, 14), ref_atr(frame, 14))
    assert ind.atr(frame, 14).iloc[:13].isna().all()


def test_true_range_uses_gaps() -> None:
    frame = pd.DataFrame(
        {"open": [10.0, 13.0], "high": [11.0, 13.5], "low": [9.0, 12.5], "close": [10.0, 13.0], "volume": 0.0}
    )
    assert ind.true_range(frame).tolist() == [2.0, 3.5]  # second bar: high 13.5 - previous close 10


def test_atr_and_adx_on_flat_series_are_zero() -> None:
    frame = flat_bars(60)
    assert ind.atr(frame, 14).iloc[13:].eq(0.0).all()
    adx = ind.adx(frame, 14)
    assert adx.iloc[:27].isna().all()
    assert adx.iloc[27:].eq(0.0).all()


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_adx_matches_wilder_sum_reference(seed: int) -> None:
    frame = random_bars(300, seed=seed)
    out = ind.adx(frame, 14)
    assert_same(out, ref_adx(frame, 14), rtol=1e-9)
    assert out.iloc[:27].isna().all() and not math.isnan(out.iloc[27])  # first ADX on row 2 * period - 1
    assert out.dropna().between(0.0, 100.0).all()


def test_adx_of_a_steady_trend_is_high() -> None:
    n = 80
    close = np.linspace(100.0, 180.0, n)
    frame = pd.DataFrame({"open": close - 0.5, "high": close + 0.5, "low": close - 1.0, "close": close, "volume": 1.0})
    assert ind.adx(frame, 14).iloc[-1] > 90.0


def test_macd_matches_reference() -> None:
    close = random_bars(300, seed=5)["close"]
    out = ind.macd(close, 12, 26, 9)
    line = [a - b for a, b in zip(ref_ema(close.tolist(), 12), ref_ema(close.tolist(), 26))]
    signal = ref_ema(line, 9)
    assert_same(out["macd"], line)
    assert_same(out["signal"], signal)
    assert_same(out["hist"], [a - b for a, b in zip(line, signal)])
    assert out["signal"].iloc[:33].isna().all() and not math.isnan(out["signal"].iloc[33])


def test_bollinger_matches_population_std_reference() -> None:
    close = random_bars(120, seed=9)["close"]
    out = ind.bollinger(close, 20, 2.0)
    mid, upper, lower = ref_bollinger(close.tolist(), 20, 2.0)
    assert_same(out["mid"], mid)
    assert_same(out["upper"], upper, rtol=1e-9)
    assert_same(out["lower"], lower, rtol=1e-9)


def test_bollinger_flat_series_has_zero_width() -> None:
    out = ind.bollinger(flat_bars(30)["close"], 20, 2.0)
    assert (out["upper"].iloc[19:] == out["lower"].iloc[19:]).all()
    assert out["mid"].iloc[19:].eq(50.0).all()


def test_donchian_excludes_the_current_bar() -> None:
    frame = random_bars(100, seed=4)
    out = ind.donchian(frame, 20)
    upper, lower = ref_donchian(frame, 20)
    assert_same(out["upper"], upper)
    assert_same(out["lower"], lower)


def test_session_vwap_matches_reference_and_restarts() -> None:
    frame = random_bars(90, seed=8)
    session = [0] * 30 + [1] * 30 + [2] * 30
    assert_same(ind.session_vwap(frame, pd.Series(session, index=frame.index)), ref_vwap(frame, session))


def test_session_vwap_without_volume_is_the_running_typical_price() -> None:
    frame = random_bars(40, seed=8, with_volume=False)
    session = [0] * 20 + [1] * 20
    out = ind.session_vwap(frame, pd.Series(session, index=frame.index))
    typical = ((frame["high"] + frame["low"] + frame["close"]) / 3.0).tolist()
    expected = [sum(typical[:i + 1]) / (i + 1) for i in range(20)] + [
        sum(typical[20 : i + 1]) / (i - 19) for i in range(20, 40)
    ]
    assert_same(out, expected)


def test_session_vwap_volume_starting_late_switches_to_vwap() -> None:
    frame = random_bars(10, seed=2)
    frame.loc[frame.index[:3], "volume"] = 0.0
    out = ind.session_vwap(frame, pd.Series([0] * 10, index=frame.index))
    assert_same(out, ref_vwap(frame, [0] * 10))


def test_zscore_and_slope() -> None:
    series = random_bars(60, seed=6)["close"]
    window = 10
    values = series.tolist()
    expected = [NAN] * (window - 1)
    for i in range(window - 1, len(values)):
        chunk = values[i - window + 1 : i + 1]
        mean = sum(chunk) / window
        std = math.sqrt(sum((x - mean) ** 2 for x in chunk) / window)
        expected.append((values[i] - mean) / std)
    assert_same(ind.zscore(series, window), expected, rtol=1e-8)
    assert ind.zscore(flat_bars(20)["close"], 10).iloc[9:].eq(0.0).all()
    assert_same(ind.slope(series, 5), [NAN] * 5 + [(values[i] - values[i - 5]) / 5 for i in range(5, 60)])


@pytest.mark.parametrize(
    "compute",
    [
        lambda f: ind.ema(f["close"], 20),
        lambda f: ind.rsi(f["close"], 14),
        lambda f: ind.atr(f, 14),
        lambda f: ind.adx(f, 14),
        lambda f: ind.macd(f["close"])["hist"],
        lambda f: ind.bollinger(f["close"])["lower"],
        lambda f: ind.donchian(f)["upper"],
        lambda f: ind.session_vwap(f, pd.Series(f.index // 3600, index=f.index)),
    ],
)
def test_indicators_are_causal(compute) -> None:  # noqa: ANN001 - parametrised callables
    frame = random_bars(200, seed=12)
    full = compute(frame).to_numpy()
    for cut in (40, 77, 150):
        truncated = compute(frame.iloc[:cut]).to_numpy()
        np.testing.assert_allclose(truncated, full[:cut], rtol=1e-12, equal_nan=True)
