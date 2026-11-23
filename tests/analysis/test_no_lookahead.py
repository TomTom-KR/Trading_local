"""The engine never looks ahead.

For many decision times t (all with plenty of warm-up, many of them chosen where a setup fires):

1. Truncation: ``build_features`` on the data that was known at t (1-minute bars closed at t, benchmark bars
   closed at t, daily bars of completed sessions) yields exactly the row of t from the full history.
2. Perturbation: randomly changing every bar AFTER t (prices and volumes, instrument and benchmark), every
   daily row from the trading day of t onwards and every news row analysed after t (plus new ones) changes
   neither the row, nor the detected hits, nor their scores, nor the news context.
3. Calibration fitted with ``before_ts = t`` ignores every signal that closed at or after t.
"""

import math
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tests.analysis.test_features import CALENDAR, NOVEMBER, daily_from_minutes, session_minutes, utc, weekdays
from trading_local.analysis.calibration import Calibrator
from trading_local.analysis.features import FEATURE_COLUMNS, build_features, news_context
from trading_local.analysis.scoring import score_hit
from trading_local.analysis.setups import detect
from trading_local.clock import from_ts
from trading_local.db.repo_news import SymbolNews
from trading_local.models import Instrument, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
NVDA = INSTRUMENTS["NVDA"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


# ------------------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------------------
def with_volume_spikes(minutes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Scale the volume of every 5-minute bucket by a random factor so that volume-confirmed setups occur."""
    buckets = minutes.index.to_numpy() // 300
    _, position = np.unique(buckets, return_inverse=True)
    factors = np.random.default_rng(seed).lognormal(0.0, 0.6, position.max() + 1)
    return minutes.assign(volume=minutes["volume"].to_numpy() * factors[position])


def news_rows(minutes: pd.DataFrame, seed: int, count: int = 60) -> list[SymbolNews]:
    """Strong analysed news at random bar times (some trigger the news setup)."""
    rng = np.random.default_rng(seed)
    times = rng.choice(minutes.index.to_numpy(), size=count, replace=False)
    return [
        SymbolNews(i + 1, int(ts), int(rng.integers(3, 6)), float(rng.choice([-0.9, -0.6, 0.6, 0.9])), f"n{i}", "",
                   "intraday", "other")
        for i, ts in enumerate(sorted(times))
    ]  # fmt: skip


class Market:
    """Full history of one instrument plus benchmark, daily bars and news, and its full feature frame."""

    def __init__(self, instrument: Instrument, days: list[date], seed: int, *, benchmark: bool = True) -> None:
        self.instrument = instrument
        self.minutes = with_volume_spikes(session_minutes(instrument.exchange, days, seed=seed), seed)
        self.bench = (
            session_minutes(instrument.exchange, days, seed=seed + 100, start_price=6000.0) if benchmark else None
        )
        self.daily = daily_from_minutes(self.minutes, history_days=260, seed=seed)
        self.news = news_rows(self.minutes, seed)
        self.features = self.build(self.minutes, self.daily, self.bench)

    def build(self, minutes: pd.DataFrame, daily: pd.DataFrame, bench: pd.DataFrame | None) -> pd.DataFrame:
        return build_features(minutes, daily, instrument=self.instrument, calendar=CALENDAR, benchmark_minutes=bench)

    def trading_day(self, t: int) -> str:
        return CALENDAR.session_date(self.instrument.exchange, from_ts(t - 300)).isoformat()

    def known_at(self, t: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Exactly what was known at decision time t."""
        last_day = CALENDAR.last_completed_session(self.instrument.exchange, from_ts(t))
        assert last_day is not None
        minutes = self.minutes[self.minutes.index + 60 <= t]
        bench = None if self.bench is None else self.bench[self.bench.index + 60 <= t]
        return minutes, self.daily[self.daily.index <= last_day.isoformat()], bench

    def perturbed_after(self, t: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, list[SymbolNews]]:
        """Full history with everything after t randomly changed."""
        rng = np.random.default_rng(seed)
        minutes = perturb_bars(self.minutes, self.minutes.index.to_numpy() + 60 > t, rng)
        bench = None if self.bench is None else perturb_bars(self.bench, self.bench.index.to_numpy() + 60 > t, rng)
        daily = perturb_bars(self.daily, np.asarray(self.daily.index >= self.trading_day(t)), rng)
        news = [
            replace(row, sentiment=-row.sentiment, relevance=5) if row.analyzed_ts > t else row for row in self.news
        ]
        news += [
            SymbolNews(10_000 + k, t + 60 * (k + 1), 5, float(rng.choice([-1.0, 1.0])), "neu", "", "intraday", "other")
            for k in range(5)
        ]
        return minutes, daily, bench, news


def perturb_bars(frame: pd.DataFrame, mask: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    changed = frame.copy()
    rows = int(mask.sum())
    changed.loc[mask, PRICE_COLUMNS] = frame.loc[mask, PRICE_COLUMNS].to_numpy() * rng.uniform(0.7, 1.3, (rows, 4))
    changed.loc[mask, "volume"] = frame.loc[mask, "volume"].to_numpy() * rng.uniform(0.0, 5.0, rows)
    return changed


def assert_rows_equal(actual: pd.Series, expected: pd.Series, context: str) -> None:
    for column in FEATURE_COLUMNS:
        got, want = actual[column], expected[column]
        if column == "session":
            assert got == want, (context, column)
        elif column == "in_session":
            assert bool(got) == bool(want), (context, column)
        elif math.isnan(float(want)):
            assert math.isnan(float(got)), (context, column, got)
        else:
            assert float(got) == pytest.approx(float(want), rel=1e-9, abs=1e-12), (context, column)


def hits_and_scores(market: Market, row: pd.Series, news: list[SymbolNews], t: int) -> list[tuple[object, ...]]:
    context = news_context(news, asof_ts=t)
    out: list[tuple[object, ...]] = []
    for hit in detect(row, instrument=market.instrument, news=context):
        out.append((hit, score_hit(hit, row, instrument=market.instrument, news=context)))
    return out


def decision_times(market: Market, *, warm_up_from: int, with_hits: int, random: int, seed: int) -> list[int]:
    """Decision times after the warm-up: up to ``with_hits`` where a setup fires plus ``random`` others."""
    rng = np.random.default_rng(seed)
    candidates = market.features[market.features.index >= warm_up_from]
    firing = [int(t) for t, row in candidates.iterrows() if hits_and_scores(market, row, market.news, int(t))]
    chosen = set(rng.choice(firing, size=min(with_hits, len(firing)), replace=False).tolist()) if firing else set()
    chosen |= set(rng.choice(candidates.index.to_numpy(), size=random, replace=False).tolist())
    return sorted(int(t) for t in chosen)


# ------------------------------------------------------------------------------------------------
# Equity with benchmark, daily history and news
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def equity() -> Market:
    return Market(NVDA, NOVEMBER, seed=41)


@pytest.fixture(scope="module")
def equity_times(equity: Market) -> list[int]:
    return decision_times(equity, warm_up_from=utc(6, 14, 30), with_hits=14, random=10, seed=7)


def test_decision_times_include_firing_setups(equity: Market, equity_times: list[int]) -> None:
    setups = {
        hit.setup  # type: ignore[attr-defined]
        for t in equity_times
        for hit, _ in hits_and_scores(equity, equity.features.loc[t], equity.news, t)
    }
    assert len(setups) >= 2, setups  # the invariance checks below are not vacuous


def test_truncated_history_gives_the_same_row(equity: Market, equity_times: list[int]) -> None:
    for t in equity_times:
        minutes, daily, bench = equity.known_at(t)
        truncated = equity.build(minutes, daily, bench)
        assert truncated.index[-1] == t
        assert_rows_equal(truncated.iloc[-1], equity.features.loc[t], f"t={t}")


def test_future_data_changes_nothing(equity: Market, equity_times: list[int]) -> None:
    for k, t in enumerate(equity_times):
        minutes, daily, bench, news = equity.perturbed_after(t, seed=100 + k)
        perturbed = equity.build(minutes, daily, bench)
        assert_rows_equal(perturbed.loc[t], equity.features.loc[t], f"t={t}")
        assert news_context(news, asof_ts=t) == news_context(equity.news, asof_ts=t)
        assert hits_and_scores(equity, perturbed.loc[t], news, t) == hits_and_scores(
            equity, equity.features.loc[t], equity.news, t
        )
        # the perturbation really changed the future
        later = perturbed.index[perturbed.index > t]
        if later.size:
            assert not np.allclose(perturbed.loc[later, "price"], equity.features.loc[later, "price"])


def test_known_data_is_complete_at_decision_time(equity: Market, equity_times: list[int]) -> None:
    """Sanity check of the harness: what is 'known at t' contains no bar or daily row from after t."""
    for t in equity_times:
        minutes, daily, bench = equity.known_at(t)
        assert minutes.index.max() + 60 <= t and bench is not None and bench.index.max() + 60 <= t
        # the row's own date is known only once its session has ended (decision on the session end)
        assert daily.index.max() < equity.trading_day(t) or (
            daily.index.max() == equity.trading_day(t) and not CALENDAR.is_open(equity.instrument.exchange, from_ts(t))
        )


# ------------------------------------------------------------------------------------------------
# Other venue types: lunch break and 24/7
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("symbol", "days", "warm_up_from"),
    [
        ("7203.T", weekdays(date(2026, 11, 9), date(2026, 11, 20)), utc(12, 0, 0)),
        ("BTC-USD", weekdays(date(2026, 11, 13), date(2026, 11, 19)), utc(17, 0, 0)),
    ],
)
def test_other_venues_truncation_and_perturbation(symbol: str, days: list[date], warm_up_from: int) -> None:
    market = Market(INSTRUMENTS[symbol], days, seed=5, benchmark=False)
    for k, t in enumerate(decision_times(market, warm_up_from=warm_up_from, with_hits=3, random=5, seed=3)):
        minutes, daily, bench = market.known_at(t)
        assert_rows_equal(market.build(minutes, daily, bench).iloc[-1], market.features.loc[t], f"{symbol} t={t}")
        minutes, daily, bench, news = market.perturbed_after(t, seed=k)
        perturbed = market.build(minutes, daily, bench)
        assert_rows_equal(perturbed.loc[t], market.features.loc[t], f"{symbol} t={t} perturbed")
        assert hits_and_scores(market, perturbed.loc[t], news, t) == hits_and_scores(
            market, market.features.loc[t], market.news, t
        )


# ------------------------------------------------------------------------------------------------
# A session across UTC midnight whose trading date is the venue's local date
# ------------------------------------------------------------------------------------------------
class SydneyLikeCalendar(FakeCalendar):
    """XASX trades 23:00-05:00 UTC every day; the trading date is the local date at UTC+11.

    Both pieces of a session (23:00-24:00 and 00:00-05:00 UTC) belong to the SAME trading date, which is the
    UTC date of the second piece - unlike the fake calendar's UTC-date convention.
    """

    OFFSET = timedelta(hours=11)

    def __init__(self) -> None:
        super().__init__({"XASX": ((0, 300), (1380, 1440))}, weekends_closed=False)

    def timezone(self, exchange: str) -> str:
        return "Etc/GMT-11"

    def session_date(self, exchange: str, at: datetime) -> date:
        return (at.astimezone(UTC) + self.OFFSET).date()

    def last_completed_session(self, exchange: str, at: datetime) -> date | None:
        local = self.session_date(exchange, at)
        session_end = datetime(local.year, local.month, local.day, 5, tzinfo=UTC)
        return local if at >= session_end else local - timedelta(days=1)


SYDNEY_CALENDAR = SydneyLikeCalendar()
SYDNEY = Instrument("BHP.AX", "BHP", "equity", "ASIA", "XASX", "AUD", timezone="Australia/Sydney")


def local_daily(minutes: pd.DataFrame, calendar: SydneyLikeCalendar, history_days: int, seed: int) -> pd.DataFrame:
    """Daily bars per LOCAL trading date, preceded by random older history."""
    local = (minutes.index.to_numpy() + int(calendar.OFFSET.total_seconds())).astype("datetime64[s]")
    keys = np.datetime_as_string(local, unit="D")
    daily = minutes.groupby(keys).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    first = date.fromisoformat(str(daily.index[0]))
    close = float(daily["open"].iloc[0]) * np.exp(np.cumsum(0.01 * np.random.default_rng(seed).standard_normal(history_days)))
    old = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
        index=[(first - timedelta(days=history_days - i)).isoformat() for i in range(history_days)],
    )
    out = pd.concat([old, daily]).astype("float64")
    out.index = pd.Index(out.index.astype(str), dtype="str", name="day")
    return out


def test_session_across_utc_midnight_keeps_one_trading_date() -> None:
    days = [date(2026, 11, d) for d in range(16, 21)]
    minutes = session_minutes("XASX", days, seed=3, calendar=SYDNEY_CALENDAR)
    daily = local_daily(minutes, SYDNEY_CALENDAR, history_days=30, seed=3)
    features = build_features(minutes, daily, instrument=SYDNEY, calendar=SYDNEY_CALENDAR)
    assert features["in_session"].all()
    evening, morning = features.loc[utc(19, 23, 5)], features.loc[utc(20, 0, 5)]
    assert evening["session"] == morning["session"] == "2026-11-20"
    assert (evening["minutes_since_open"], evening["minutes_to_close"]) == (5.0, 355.0)
    assert (morning["minutes_since_open"], morning["minutes_to_close"]) == (65.0, 295.0)
    assert features.loc[utc(20, 5, 0), "minutes_to_close"] == 0.0
    # the session's extremes, VWAP and opening range run across midnight
    session_bars = minutes[(minutes.index >= utc(19, 23)) & (minutes.index < utc(20, 0, 5))]
    assert morning["day_high"] == session_bars["high"].max() and morning["day_low"] == session_bars["low"].min()
    opening = minutes[(minutes.index >= utc(19, 23)) & (minutes.index < utc(19, 23, 30))]
    assert morning["or_high"] == opening["high"].max()
    # daily features come from the previous LOCAL date, never from the UTC date of the row
    assert evening["prev_day_close"] == morning["prev_day_close"] == daily.loc["2026-11-19", "close"]


def test_session_across_utc_midnight_truncation_and_perturbation() -> None:
    days = [date(2026, 11, d) for d in range(1, 21)]
    minutes = with_volume_spikes(session_minutes("XASX", days, seed=17, calendar=SYDNEY_CALENDAR), 17)
    daily = local_daily(minutes, SYDNEY_CALENDAR, history_days=260, seed=17)
    news = news_rows(minutes, 17)

    def build(bars: pd.DataFrame, days_frame: pd.DataFrame) -> pd.DataFrame:
        return build_features(bars, days_frame, instrument=SYDNEY, calendar=SYDNEY_CALENDAR)

    full = build(minutes, daily)
    rng = np.random.default_rng(8)
    candidates = full.index[full.index >= utc(12, 0)].to_numpy()
    for k, t in enumerate(sorted(int(t) for t in rng.choice(candidates, size=12, replace=False))):
        last_day = SYDNEY_CALENDAR.last_completed_session("XASX", from_ts(t))
        assert last_day is not None
        known = build(minutes[minutes.index + 60 <= t], daily[daily.index <= last_day.isoformat()])
        assert known.index[-1] == t
        assert_rows_equal(known.iloc[-1], full.loc[t], f"XASX t={t}")
        trading_day = SYDNEY_CALENDAR.session_date("XASX", from_ts(t - 300)).isoformat()
        assert last_day.isoformat() <= trading_day  # equal for the last bucket: its session ended at t
        rng_k = np.random.default_rng(1_000 + k)
        changed_minutes = perturb_bars(minutes, minutes.index.to_numpy() + 60 > t, rng_k)
        changed_daily = perturb_bars(daily, np.asarray(daily.index >= trading_day), rng_k)
        perturbed = build(changed_minutes, changed_daily)
        assert_rows_equal(perturbed.loc[t], full.loc[t], f"XASX t={t} perturbed")
        context = news_context(news, asof_ts=t)
        assert detect(perturbed.loc[t], instrument=SYDNEY, news=context) == detect(
            full.loc[t], instrument=SYDNEY, news=context
        )


# ------------------------------------------------------------------------------------------------
# A feed with missing bars and a longer outage
# ------------------------------------------------------------------------------------------------
def test_gappy_feed_truncation() -> None:
    """Missing bars change which buckets exist, never what a row of the past contains."""
    complete = with_volume_spikes(session_minutes("XNAS", NOVEMBER, seed=23), 23)
    rng = np.random.default_rng(23)
    keep = rng.random(len(complete)) > 0.05
    outage = (complete.index >= utc(18, 16, 0)) & (complete.index < utc(18, 16, 47))
    minutes = complete[keep & ~outage]
    daily = daily_from_minutes(complete, history_days=260, seed=23)
    full = build_features(minutes, daily, instrument=NVDA, calendar=CALENDAR)
    assert not ((full.index > utc(18, 16, 0)) & (full.index <= utc(18, 16, 45))).any()  # no bars, no rows
    times = list(rng.choice(full.index[full.index >= utc(9, 14, 30)].to_numpy(), size=25, replace=False))
    times += [utc(18, 16, 50), utc(18, 16, 55)]  # right after the outage
    present = set(minutes.index.tolist())
    for t in sorted(int(t) for t in times):
        if t not in full.index:
            continue
        last_day = CALENDAR.last_completed_session("XNAS", from_ts(t))
        assert last_day is not None
        known = build_features(
            minutes[minutes.index + 60 <= t], daily[daily.index <= last_day.isoformat()], instrument=NVDA,
            calendar=CALENDAR,
        )
        if t - 60 in present:  # the bucket's last bar arrived: its row exists at t and equals the full row
            assert known.index[-1] == t
            assert_rows_equal(known.iloc[-1], full.loc[t], f"gappy t={t}")
        else:  # the bucket is only known to be complete once a later bar arrives
            assert known.index[-1] < t
            assert_rows_equal(known.loc[known.index[-1]], full.loc[known.index[-1]], f"gappy t={t}")


# ------------------------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------------------------
def closed_signals(seed: int, start: int, end: int, count: int = 300) -> list[Signal]:
    rng = np.random.default_rng(seed)
    signals = []
    for i in range(count):
        created = int(rng.integers(start, end))
        exit_ts = created + int(rng.integers(600, 20_000))
        win = bool(rng.random() < 0.45)
        signals.append(
            Signal(
                created_ts=created, data_ts=created, symbol="NVDA", direction="LONG", setup="breakout",
                timeframe="15m", horizon_min=180, ref_price=100.0, atr=1.0, stop_dist=1.0,
                target_dist=float(rng.choice([1.2, 2.0])), score=float(rng.uniform(0.3, 0.95)), confidence=0.4,
                engine_version="e1-x", status="target" if win else "stopped", exit_ts=exit_ts,
                outcome_r=float(rng.uniform(0.5, 2.0)) if win else -1.0, id=i + 1,
            )
        )  # fmt: skip
    return signals


def perturbed_signals(signals: list[Signal], t: int, seed: int) -> Iterator[Signal]:
    rng = np.random.default_rng(seed)
    for signal in signals:
        if signal.exit_ts is not None and signal.exit_ts >= t:
            yield replace(signal, outcome_r=-(signal.outcome_r or 0.0) * 3.0, score=float(rng.uniform(0, 1)))
        else:
            yield signal
    # a signal that was still open at t closes later with a large win
    yield replace(signals[0], id=99_999, created_ts=t - 600, exit_ts=t + 1, outcome_r=5.0, status="target")


def test_calibration_ignores_signals_closed_at_or_after_t() -> None:
    start, end = utc(2, 14, 30), utc(20, 21, 0)
    signals = closed_signals(11, start, end)
    grid = np.linspace(0.0, 1.0, 41)
    for k, t in enumerate(np.random.default_rng(4).integers(start + 86_400, end, size=8).tolist()):
        fitted = Calibrator.from_signals(signals, before_ts=t)
        again = Calibrator.from_signals(list(perturbed_signals(signals, t, seed=k)), before_ts=t)
        assert fitted.bins() == again.bins()
        assert [fitted.probability(s, 2.0) for s in grid] == [again.probability(s, 2.0) for s in grid]
    # and the fit does use the earlier signals
    late = Calibrator.from_signals(signals, before_ts=end + 100_000)
    assert sum(b.n for b in late.bins()) == len(signals)
