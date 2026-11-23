"""Validation replay: determinism, causality, no manufactured edge, costs, delayed feeds, folds, walk-forward
calibration, warnings and verdict, the random-entry tracker and the command line.

Synthetic histories on the fake calendar (XNAS 14:30-21:00 UTC on weekdays): a driftless random walk and a
planted, unmistakable pattern - a persistent up-trend with regular pullbacks (a sine wave on a drift).
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_local import validate
from trading_local.analysis import tracking
from trading_local.analysis.calibration import Calibrator
from trading_local.analysis.engine import DAILY_BARS, closed_ts, warmup_start_ts
from trading_local.analysis.features import build_features, news_context
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.analysis.scoring import score_hit
from trading_local.analysis.setups import SetupHit, detect
from trading_local.analysis.validation import (
    COOLDOWN_MIN,
    Replay,
    TradeSimulator,
    _Benchmark,
    build_report,
    decision_lag_s,
    fold_bounds,
    fold_calibrators,
    knowledge_lag_s,
    random_baseline,
    replay,
    run_validation,
    validation_period,
    verdict_de,
)
from trading_local.barstore import BarReader
from trading_local.clock import ManualClock, from_ts, to_ts
from trading_local.config import CostParams, Settings
from trading_local.db import repo_market, repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.dto import BaselineOut, FoldOut, StatBlock, ValidationResultOut
from trading_local.models import (
    SIGNAL_OUTCOME_STATUSES,
    Candidate,
    Direction,
    Instrument,
    NewsAnalysis,
    NewsItem,
    Signal,
)
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_bars, seed_bars, seed_instruments

CALENDAR = FakeCalendar()
TEMPLATE = replace(next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "NVDA"), benchmark=None)
BTC = next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "BTC-USD")
COSTS = CostParams()
FIRST_DAY = date(2026, 10, 1)
LAST_DAY = date(2026, 11, 20)  # a Friday
NOW = to_ts(datetime(2026, 11, 21, tzinfo=UTC))  # Saturday 00:00 UTC: the data ends with Friday's session
DAYS = 30
START, END = validation_period(NOW, DAYS)
SYMBOLS = 4  # trend market and the mechanical tests
NOISE_SYMBOLS = 12  # random walk: enough signals that noise cannot pass for an edge


# ------------------------------------------------------------------------------------------------
# Synthetic markets
# ------------------------------------------------------------------------------------------------
def session_bars(
    exchange: str,
    *,
    seed: int,
    drift: float,
    vol: float,
    amp: float = 0.0,
    period: float = 120.0,
    first: date = FIRST_DAY,
    last: date = LAST_DAY,
) -> pd.DataFrame:
    """1-minute bars over the regular segments: log price = random walk with ``drift`` plus a sine wave of
    amplitude ``amp`` and ``period`` minutes (regular pullbacks)."""
    stamps = []
    day = first
    while day <= last:
        midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
        for segment in CALENDAR.segments(exchange, midnight, midnight + timedelta(days=1)):
            count = int((segment.end - segment.start).total_seconds() // 60)
            stamps.append(to_ts(segment.start) + 60 * np.arange(count, dtype=np.int64))
        day += timedelta(days=1)
    ts = np.concatenate(stamps)
    rng = np.random.default_rng(seed)
    wave = amp * np.sin(2.0 * np.pi * np.arange(ts.size) / period)
    close = 100.0 * np.exp(np.cumsum(drift + vol * rng.standard_normal(ts.size)) + wave)
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) * np.exp(vol / 2 * np.abs(rng.standard_normal(ts.size)))
    low = np.minimum(open_, close) * np.exp(-vol / 2 * np.abs(rng.standard_normal(ts.size)))
    volume = 1000.0 * rng.uniform(0.5, 1.5, ts.size)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=pd.Index(ts, name="ts")
    )


def daily_bars(minutes: pd.DataFrame, history_days: int = 260, old_drift: float = 0.0) -> pd.DataFrame:
    """Daily bars per UTC date (the fake calendar's trading date) preceded by older history (EMA200) that is
    flat or, with ``old_drift``, rises by that log return per day towards the first minute bar."""
    days = np.datetime_as_string(minutes.index.to_numpy().astype("datetime64[s]"), unit="D")
    daily = minutes.groupby(days).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    first = date.fromisoformat(str(daily.index[0]))
    price = float(daily["open"].iloc[0]) * np.exp(-old_drift * np.arange(history_days, 0, -1))
    old_days = [(first - timedelta(days=history_days - i)).isoformat() for i in range(history_days)]
    old = pd.DataFrame(
        {"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": 1e6}, index=old_days
    )
    frame = pd.concat([old, daily])
    frame.index = pd.Index(frame.index.astype(str), dtype="str", name="day")
    return frame.astype("float64")


def market_bars(kind: str, seed: int, *, first: date = FIRST_DAY) -> pd.DataFrame:
    if kind == "trend":
        return session_bars("XNAS", seed=seed, drift=6e-5, vol=0.0005, amp=0.004, first=first)
    return session_bars("XNAS", seed=seed, drift=0.0, vol=0.0008, first=first)


def seed_market(db: Database, kind: str, symbols: int, *, first: date = FIRST_DAY) -> list[Instrument]:
    instruments = [replace(TEMPLATE, symbol=f"SYM{k}") for k in range(symbols)]
    seed_instruments(db, instruments)
    for k, instrument in enumerate(instruments):
        minutes = market_bars(kind, k + 1, first=first)
        seed_bars(db, instrument.symbol, minutes)
        repo_market.upsert_bars_1d(db, instrument.symbol, daily_bars(minutes))
    return instruments


@pytest.fixture(scope="module")
def trend_market(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Database, list[Instrument]]]:
    db = Database(tmp_path_factory.mktemp("trend") / "trend.sqlite")
    db.initialize()
    yield db, seed_market(db, "trend", SYMBOLS)
    db.close()


@pytest.fixture(scope="module")
def walk_market(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Database, list[Instrument]]]:
    db = Database(tmp_path_factory.mktemp("walk") / "walk.sqlite")
    db.initialize()
    yield db, seed_market(db, "walk", NOISE_SYMBOLS)
    db.close()


@pytest.fixture(scope="module")
def trend_replay(trend_market: tuple[Database, list[Instrument]]) -> Replay:
    db, instruments = trend_market
    return replay(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, start_ts=START, end_ts=END)


@pytest.fixture(scope="module")
def walk_replay(walk_market: tuple[Database, list[Instrument]]) -> Replay:
    db, instruments = walk_market
    return replay(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, start_ts=START, end_ts=END)


def signal_key(signal: Signal) -> tuple[str, int, str]:
    return signal.symbol, signal.created_ts, signal.direction


# ------------------------------------------------------------------------------------------------
# Determinism and causality
# ------------------------------------------------------------------------------------------------
def test_same_inputs_give_identical_reports(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market[0], walk_market[1][:2]
    reports = [
        run_validation(
            db, calendar=CALENDAR, instruments=instruments, costs=COSTS, days=DAYS, min_score=0.0, now_ts=NOW,
            n_random=50,
        ).model_dump()
        for _ in range(2)
    ]  # fmt: skip
    assert reports[0] == reports[1]
    # few selected signals: the live issue rule also requires a positive expected R after costs
    assert reports[0]["overall"]["n"] >= 3 and reports[0]["baseline"]["n_sims"] == 50


def test_bars_after_the_period_end_change_no_signal_that_closed_before(
    trend_market: tuple[Database, list[Instrument]], trend_replay: Replay
) -> None:
    db, instruments = trend_market[0], trend_market[1][:3]
    cut = to_ts(datetime(2026, 11, 13, tzinfo=UTC))
    early = replay(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, start_ts=START, end_ts=cut)
    full = [s for s in trend_replay.signals if s.symbol in early.instruments]  # same start, ends a week later
    later = {signal_key(signal): signal for signal in full}
    assert len(early.signals) >= 10 and len(full) > len(early.signals)
    for signal in early.signals:
        twin = later[signal_key(signal)]
        assert (twin.status, twin.entry_ts, twin.entry_price, twin.exit_ts, twin.exit_price, twin.outcome_r) == (
            signal.status, signal.entry_ts, signal.entry_price, signal.exit_ts, signal.exit_price, signal.outcome_r
        )  # fmt: skip
        assert (twin.score, twin.components, twin.setup) == (signal.score, signal.components, signal.setup)
        assert signal.checked_ts is not None and signal.checked_ts + 60 <= cut


def test_decisions_use_only_rows_inside_the_period(trend_replay: Replay) -> None:
    assert trend_replay.signals
    for signal in trend_replay.signals:
        assert START <= signal.created_ts < END
        # real-time feed: decided by the minute cycle after the bucket end, entered on the next bar at the earliest
        assert signal.issued_ts == signal.created_ts == signal.data_ts + 60
        assert signal.entry_ts is None or signal.entry_ts >= signal.data_ts + 60
    ids = [signal.id for signal in trend_replay.signals]
    assert ids == list(range(1, len(ids) + 1))


def test_one_open_signal_per_symbol_and_same_direction_cooldown(trend_replay: Replay) -> None:
    by_symbol: dict[str, list[Signal]] = {}
    for signal in trend_replay.signals:
        by_symbol.setdefault(signal.symbol, []).append(signal)
    for signals in by_symbol.values():
        for previous, current in zip(signals, signals[1:]):
            closed = closed_ts(previous)
            assert current.created_ts >= closed + 60 + decision_lag_s(TEMPLATE)  # the closing bar had arrived
            if current.direction == previous.direction:
                assert current.created_ts > closed + COOLDOWN_MIN * 60


# ------------------------------------------------------------------------------------------------
# The replay sees exactly what the live engine sees
# ------------------------------------------------------------------------------------------------
MIXED_FIRST = date(2026, 10, 26)
MIXED_DAYS = 10


@pytest.fixture(scope="module")
def mixed_market(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Database, list[Instrument]]]:
    """Real-time US share with its real-time index, delayed German share with its delayed index and crypto
    (24/7). The daily history before the minute data rises, so that the seed of the daily EMA200 matters."""
    db = Database(tmp_path_factory.mktemp("mixed") / "mixed.sqlite")
    db.initialize()
    symbols = ("NVDA", "SAP.DE", "BTC-USD", "^GSPC", "^GDAXI")
    instruments = [
        replace(i, tradable=not i.symbol.startswith("^")) for i in SAMPLE_INSTRUMENTS if i.symbol in symbols
    ]
    seed_instruments(db, instruments)
    for k, instrument in enumerate(instruments):
        minutes = session_bars(
            instrument.exchange, seed=40 + k, drift=6e-5, vol=0.0005, amp=0.004, first=MIXED_FIRST
        )
        seed_bars(db, instrument.symbol, minutes)
        repo_market.upsert_bars_1d(db, instrument.symbol, daily_bars(minutes, old_drift=0.002))
    yield db, [instrument for instrument in instruments if instrument.tradable]
    db.close()


@pytest.fixture(scope="module")
def mixed_replay(mixed_market: tuple[Database, list[Instrument]]) -> Replay:
    db, instruments = mixed_market
    start, end = validation_period(NOW, MIXED_DAYS)
    return replay(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, start_ts=start, end_ts=end)


def live_row(db: Database, instrument: Instrument, decision: int) -> dict[str, Any]:
    """The newest feature row the live engine computes at ``decision`` (``SignalEngine._snapshot``): the 1-minute
    bars from the warm-up start of that instant as they had physically arrived (delayed feeds lag behind), the
    newest ``DAILY_BARS`` daily bars of completed sessions and the benchmark bars from the same start."""
    book = InstrumentBook(db)
    arrived = BarReader(db, book, CALENDAR, decision, apply_feed_delay=True)
    start = warmup_start_ts(CALENDAR, instrument.exchange, decision)
    bench = None if instrument.benchmark is None else arrived.minutes(instrument.benchmark, start_ts=start)
    features = build_features(
        arrived.minutes(instrument.symbol, start_ts=start),
        BarReader(db, book, CALENDAR, decision).daily(instrument.symbol, limit=DAILY_BARS),
        instrument=instrument,
        calendar=CALENDAR,
        benchmark_minutes=None if bench is None or bench.empty else bench,
    )
    return features.iloc[-1].to_dict()


def test_signals_are_built_from_exactly_the_live_feature_rows(
    mixed_market: tuple[Database, list[Instrument]], mixed_replay: Replay
) -> None:
    db, instruments = mixed_market
    by_symbol = {instrument.symbol: instrument for instrument in instruments}
    assert sorted(mixed_replay.instruments) == sorted(by_symbol) and not mixed_replay.failed
    signals = mixed_replay.signals
    assert {signal.symbol for signal in signals} == set(by_symbol)
    assert len(signals) >= 10
    for signal in signals:
        instrument = by_symbol[signal.symbol]
        row = live_row(db, instrument, signal.created_ts)
        assert row["data_ts"] == signal.data_ts
        hit = SetupHit(signal.setup, signal.direction, signal.timeframe, signal.intraday, signal.horizon_min,
                       signal.atr, signal.stop_dist, signal.target_dist)  # fmt: skip
        news = news_context([], asof_ts=signal.created_ts)
        total = row["minutes_since_open"] + row["minutes_to_close"]
        hits = detect(row, instrument=instrument, news=news, session_minutes=round(total))
        assert (hit.setup, hit.direction) in {(h.setup, h.direction) for h in hits}
        candidate = tracking.candidate_from_hit(
            hit, row, instrument=instrument, decision_ts=signal.created_ts, score=0.0, components={}, news=news
        )
        assert candidate.features == signal.features, signal  # all features, incl. volatility rank and EMA200
        assert score_hit(hit, row, instrument=instrument, news=news) == (signal.score, signal.components)


def test_decision_rows_follow_the_live_rules_on_every_venue(mixed_replay: Replay) -> None:
    for symbol, market in mixed_replay.markets.items():
        instrument = mixed_replay.instruments[symbol]
        decisions = market.decision_ts.tolist()
        assert decisions == sorted(decisions) and len(set(decisions)) == len(decisions)
        assert all(CALENDAR.is_open(instrument.exchange, from_ts(at)) for at in decisions)
        lag = decision_lag_s(instrument)
        assert all((at - lag) % 300 == 0 for at in decisions)  # bucket end + feed delay + one minute cycle
    # 24/7 venue: the last bucket of a UTC day is decided one minute after midnight, on the next trading date
    crypto = mixed_replay.markets["BTC-USD"].decision_ts
    assert np.any(crypto % 86_400 == 60)
    # delayed venue (Xetra 08:00-16:30 UTC, 15 min delay): nothing after the close, nothing before 08:16
    seconds = mixed_replay.markets["SAP.DE"].decision_ts % 86_400
    assert seconds.min() >= 8 * 3600 + 16 * 60 and seconds.max() < 16 * 3600 + 30 * 60
    # every calendar day of the period is a crypto trading date (weekends included)
    assert mixed_replay.trading_days == MIXED_DAYS


def test_a_more_delayed_benchmark_is_only_seen_once_its_data_had_arrived(
    mixed_market: tuple[Database, list[Instrument]],
) -> None:
    bars = make_bars(to_ts(datetime(2026, 11, 16, 14, 30, tzinfo=UTC)), 120, seed=3)
    first, decision = int(bars.index[10]), int(bars.index[100])
    window = _Benchmark(bars, 15 * 60).window(first, decision)
    assert window is not None
    original = window.index.to_numpy() - 15 * 60
    assert original[0] == first and original[-1] + 60 + 15 * 60 == decision  # had closed 15 min before
    assert np.array_equal(window["close"].to_numpy(), bars["close"].loc[original].to_numpy())
    assert _Benchmark(bars, 0).window(decision, decision) is None

    db, instruments = mixed_market
    nvda = next(i for i in instruments if i.symbol == "NVDA")
    gspc = replace(next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "^GSPC"), delay_min=15, tradable=False)
    start, end = validation_period(NOW, 3)
    result = replay(db, calendar=CALENDAR, instruments=[nvda, gspc], costs=COSTS, start_ts=start, end_ts=end)
    assert sorted(result.instruments) == ["NVDA"] and not result.failed
    plain = replay(db, calendar=CALENDAR, instruments=[nvda], costs=COSTS, start_ts=start, end_ts=end)
    assert result.decisions > 0
    assert np.array_equal(result.markets["NVDA"].decision_ts, plain.markets["NVDA"].decision_ts)


def test_only_active_tradable_instruments_are_replayed(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market
    excluded = [replace(instruments[0], tradable=False), replace(instruments[1], active=False)]
    result = replay(db, calendar=CALENDAR, instruments=excluded, costs=COSTS, start_ts=START, end_ts=END)
    assert result.instruments == {} and result.skipped == [] and result.signals == [] and result.decisions == 0
    report = run_validation(db, calendar=CALENDAR, instruments=[], costs=COSTS, now_ts=NOW)
    assert report.symbols == 0 and report.overall.n == 0 and report.baseline is None
    assert report.verdict_de.startswith("Keine Aussage")
    assert sum(fold.stats.n for fold in report.folds) == 0 and len(report.folds) == 4


# ------------------------------------------------------------------------------------------------
# Honesty: a real pattern is found, noise is not turned into an edge
# ------------------------------------------------------------------------------------------------
def test_planted_trend_yields_trend_signals_with_positive_average_r(trend_replay: Replay) -> None:
    report = build_report(trend_replay, min_score=0.0)
    trend = next(group for group in report.by_setup if group.key == "trend_pullback")
    assert trend.label_de == "Trend-Rücksetzer"
    assert trend.stats.n >= 20 and trend.stats.avg_r is not None and trend.stats.avg_r > 0.5
    assert report.overall.avg_r_lo is not None and report.overall.avg_r_lo > 0.0
    assert report.baseline is not None and report.baseline.p_value is not None and report.baseline.p_value <= 0.05
    assert report.baseline.mean_avg_r < report.overall.avg_r


def test_random_walk_shows_no_edge(walk_replay: Replay) -> None:
    report = build_report(walk_replay, min_score=0.0, expectancy_gate=False)  # every candidate
    assert report.overall.n >= 100
    assert report.overall.avg_r is not None and report.overall.avg_r < 0.1
    assert report.overall.avg_r_lo is not None and report.overall.avg_r_hi is not None
    assert report.overall.avg_r_lo < 0.0 < report.overall.avg_r_hi
    assert report.baseline is not None and report.baseline.p_value is not None and report.baseline.p_value > 0.1
    assert any("nicht von Zufall unterscheidbar" in warning for warning in report.warnings_de)
    assert any("schließt 0 ein" in warning for warning in report.warnings_de)
    assert report.verdict_de.startswith("Kein")


def test_higher_slippage_lowers_average_r(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market[0], walk_market[1][:2]
    averages = []
    for slippage in (0.0, 10.0, 40.0):
        costs = CostParams(slippage_bps_us_equity=slippage)
        result = replay(db, calendar=CALENDAR, instruments=instruments, costs=costs, start_ts=START, end_ts=END)
        # the same candidates at every cost level (the expectancy rule would drop the expensive ones)
        averages.append(build_report(result, min_score=0.0, n_random=0, expectancy_gate=False).overall.avg_r)
    assert averages[0] > averages[1] > averages[2]


def test_delayed_feed_decides_and_enters_later(
    trend_market: tuple[Database, list[Instrument]], trend_replay: Replay
) -> None:
    db, instruments = trend_market
    realtime = trend_replay
    delayed_instruments = [replace(instrument, delay_min=15) for instrument in instruments]
    delayed = replay(db, calendar=CALENDAR, instruments=delayed_instruments, costs=COSTS, start_ts=START, end_ts=END)
    lag = decision_lag_s(delayed_instruments[0])
    assert lag == 16 * 60
    for signal in delayed.signals:
        assert signal.created_ts == signal.issued_ts == signal.data_ts + lag
        assert signal.entry_ts is None or signal.entry_ts >= signal.issued_ts
    early = {(s.symbol, s.data_ts, s.direction): s for s in realtime.signals}
    matched = [(early[key], s) for s in delayed.signals if (key := (s.symbol, s.data_ts, s.direction)) in early]
    entered = [(rt, dl) for rt, dl in matched if rt.entry_ts is not None and dl.entry_ts is not None]
    assert len(entered) >= 10
    assert all(dl.entry_ts >= rt.entry_ts + 15 * 60 for rt, dl in entered)  # type: ignore[operator]
    report_rt = build_report(realtime, min_score=0.0, n_random=0)
    report_dl = build_report(delayed, min_score=0.0, n_random=0)
    assert report_dl.overall.avg_r is not None and report_rt.overall.avg_r is not None
    assert report_dl.overall.avg_r <= report_rt.overall.avg_r


# ------------------------------------------------------------------------------------------------
# Folds and walk-forward calibration
# ------------------------------------------------------------------------------------------------
def test_folds_partition_the_period(trend_replay: Replay) -> None:
    report = build_report(trend_replay, min_score=0.0, n_random=0)
    folds = report.folds
    assert [fold.index for fold in folds] == [1, 2, 3, 4]
    assert folds[0].start_ts == report.period_start_ts == START and folds[-1].end_ts == report.period_end_ts == END
    assert all(left.end_ts == right.start_ts for left, right in zip(folds, folds[1:]))
    durations = {fold.end_ts - fold.start_ts for fold in folds}
    assert max(durations) - min(durations) <= 1
    assert sum(fold.stats.n for fold in folds) == report.overall.n
    assert sum(fold.stats.total_r for fold in folds) == pytest.approx(report.overall.total_r)
    assert fold_bounds(0, 10, 4) == [(0, 2), (2, 5), (5, 7), (7, 10)]
    with pytest.raises(ValueError):
        fold_bounds(5, 5)


def test_walk_forward_calibration_uses_only_earlier_closes(trend_replay: Replay) -> None:
    signals = trend_replay.signals
    bounds = fold_bounds(START, END)
    calibrators = fold_calibrators(signals, bounds)
    prior = Calibrator(ENGINE_PARAMS)
    grid = [(score, rr) for score in (0.3, 0.55, 0.65, 0.75, 0.9) for rr in (1.2, 2.0)]
    assert [calibrators[0].probability(s, rr) for s, rr in grid] == [prior.probability(s, rr) for s, rr in grid]
    lo = bounds[2][0]
    assert any(s.exit_ts is not None and s.exit_ts < lo for s in signals)
    # Flipping every outcome that was not known before fold 3 started changes nothing for fold 3.
    flipped = [
        replace(s, outcome_r=-(s.outcome_r or 0.0) - 1.0) if s.exit_ts is None or s.exit_ts >= lo else s
        for s in signals
    ]
    third = fold_calibrators(flipped, bounds)[2]
    assert [third.probability(s, rr) for s, rr in grid] == [calibrators[2].probability(s, rr) for s, rr in grid]
    expected = Calibrator.fit(
        [(s.score, s.rr, s.outcome_r) for s in signals if s.outcome_r is not None and s.exit_ts is not None
         and s.exit_ts < lo],
    )  # fmt: skip
    assert [expected.probability(s, rr) for s, rr in grid] == [calibrators[2].probability(s, rr) for s, rr in grid]


def test_calibration_table_pools_all_candidates_with_a_result(trend_replay: Replay) -> None:
    report = build_report(trend_replay, min_score=0.9, n_random=0)
    with_result = [s for s in trend_replay.signals if s.outcome_r is not None]
    assert sum(cell.n for cell in report.calibration) == len(with_result)
    assert [cell.lo for cell in report.calibration] == list(ENGINE_PARAMS.calibration_bins[:-1])
    assert report.calibration[-1].hi == 1.0
    for cell in report.calibration:
        assert 0.0 < cell.predicted < 1.0
        assert (cell.observed is None) == (cell.n == 0)


# ------------------------------------------------------------------------------------------------
# Random-entry baseline
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Trade:
    direction: Direction
    issued_ts: int
    ref_price: float
    stop: float
    target: float
    horizon: int
    intraday: bool


def tracked_outcome(instrument: Instrument, bars: pd.DataFrame, trade: Trade) -> tuple[float | None, str]:
    """(outcome_r or None, status) of the reference tracker for one trade."""
    candidate = Candidate(
        symbol=instrument.symbol,
        direction=trade.direction,
        setup="trend_pullback",
        timeframe="15m",
        intraday=trade.intraday,
        horizon_min=trade.horizon,
        decision_ts=trade.issued_ts,
        data_ts=trade.issued_ts,
        ref_price=trade.ref_price,
        atr=trade.stop,
        stop_dist=trade.stop,
        target_dist=trade.target,
    )
    signal = tracking.make_signal(candidate, confidence=0.5, expected_r=0.0, engine_version="test")
    tracking.issue(signal, issued_ts=trade.issued_ts, instrument=instrument)
    tracking.advance(signal, bars, instrument=instrument, calendar=CALENDAR, costs=COSTS)
    return (signal.outcome_r if signal.status in SIGNAL_OUTCOME_STATUSES else None), signal.status


def random_trade(rng: np.random.Generator, bars: pd.DataFrame) -> Trade:
    """A trade issued at (or slightly off) a random bar, sometimes with a reference price far from the entry."""
    k = int(rng.integers(1, len(bars)))
    price = float(bars["close"].iloc[k - 1])
    stop = price * float(rng.uniform(0.001, 0.006))
    return Trade(
        direction="LONG" if rng.random() < 0.5 else "SHORT",
        issued_ts=int(bars.index[k]) + int(rng.choice([0, 0, 0, 25, -40, 3600])),
        ref_price=price * (1.0 + float(rng.normal(0.0, 0.002))),
        stop=stop,
        target=stop * float(rng.choice([1.2, 2.0])),
        horizon=int(rng.choice([60, 240, 1950])),
        intraday=bool(rng.random() < 0.7),
    )


@pytest.mark.parametrize("kind", ["equity", "delayed", "crypto"])
def test_trade_simulator_matches_the_tracker(kind: str, trend_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = trend_market
    if kind == "crypto":
        instrument = BTC
        bars = make_bars(to_ts(datetime(2026, 11, 16, tzinfo=UTC)), 4 * 1440, seed=7, drift=2e-5, vol=0.0008)
    else:
        instrument = instruments[0] if kind == "equity" else replace(instruments[0], delay_min=15)
        bars = repo_market.read_bars_1m(db, instrument.symbol, start_ts=to_ts(datetime(2026, 11, 9, tzinfo=UTC)))
    simulator = TradeSimulator(instrument, bars, calendar=CALENDAR, costs=COSTS)
    rng = np.random.default_rng(11)
    statuses = set()
    for _ in range(250):
        trade = random_trade(rng, bars)
        expected, status = tracked_outcome(instrument, bars, trade)
        statuses.add(status)
        actual = simulator.outcome(
            trade.direction,
            issued_ts=trade.issued_ts,
            ref_price=trade.ref_price,
            stop_dist=trade.stop,
            target_dist=trade.target,
            horizon_min=trade.horizon,
            intraday=trade.intraday,
        )
        assert actual == expected, (trade, status)
    assert {"target", "stopped", "expired", "cancelled"} <= statuses


def test_random_baseline_is_seeded(walk_replay: Replay) -> None:
    selected = [s for s in walk_replay.signals if s.outcome_r is not None]
    engine = float(np.mean([s.outcome_r for s in selected]))
    runs = [
        random_baseline(selected, walk_replay.markets, engine_avg_r=engine, n_random=40, seed=seed)
        for seed in (1, 1, 2)
    ]
    assert runs[0] is not None and runs[2] is not None
    assert runs[0] == runs[1] and runs[0] != runs[2]
    assert runs[0].n_sims == 40 and 1 / 41 <= (runs[0].p_value or 0.0) <= 1.0
    assert runs[0].p95_avg_r >= runs[0].mean_avg_r
    assert random_baseline([], walk_replay.markets, engine_avg_r=0.0, n_random=10) is None


# ------------------------------------------------------------------------------------------------
# Warnings and verdict
# ------------------------------------------------------------------------------------------------
def test_short_period_and_missing_data_produce_warnings(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market[0], walk_market[1][:SYMBOLS]
    empty = replace(TEMPLATE, symbol="LEER")
    report = run_validation(
        db, calendar=CALENDAR, instruments=[*instruments, empty], costs=COSTS, days=5, now_ts=NOW, n_random=20
    )
    text = "\n".join(report.warnings_de)
    assert "Stichprobe zu klein" in text
    assert "Handelstage im Zeitraum" in text and report.trading_days < 10
    assert "1 Instrument mangels Daten übersprungen: LEER." in text
    assert "Keine Nachrichtenhistorie im Zeitraum" in text
    assert report.symbols == SYMBOLS


class BrokenCalendar(FakeCalendar):
    """Fake calendar that does not know the exchange 'XBAD' (like a real calendar with an unknown key)."""

    def timezone(self, exchange: str) -> str:
        if exchange == "XBAD":
            raise KeyError(exchange)
        return super().timezone(exchange)


def test_failing_instrument_is_skipped_with_a_warning(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market[0], walk_market[1][:2]
    broken = replace(instruments[0], exchange="XBAD")
    result = replay(
        db, calendar=BrokenCalendar(), instruments=[broken, *instruments[1:]], costs=COSTS, start_ts=START, end_ts=END
    )
    assert result.failed == ["SYM0"] and sorted(result.instruments) == ["SYM1"]
    report = build_report(result, n_random=0)
    assert "1 Instrument wegen eines Fehlers übersprungen (Details im Log): SYM0." in report.warnings_de


def test_signals_open_at_the_end_are_discarded_and_counted(
    trend_market: tuple[Database, list[Instrument]], trend_replay: Replay
) -> None:
    db, instruments = trend_market
    victim = next(
        s for s in trend_replay.signals
        if s.entry_ts is not None and s.exit_ts is not None and s.exit_ts - s.entry_ts >= 600
    )  # fmt: skip
    instrument = next(i for i in instruments if i.symbol == victim.symbol)
    cut = victim.entry_ts + 120  # type: ignore[operator]
    result = replay(db, calendar=CALENDAR, instruments=[instrument], costs=COSTS, start_ts=START, end_ts=cut)
    assert result.discarded_open == 1
    assert signal_key(victim) not in {signal_key(s) for s in result.signals}
    report = build_report(result, min_score=0.0, n_random=0)
    assert "1 Signal war am Ende der Daten noch offen und wurde nicht gewertet." in report.warnings_de


def test_news_in_the_period_count_as_news_history(db: Database) -> None:
    instruments = seed_market(db, "walk", 1, first=date(2026, 10, 26))
    analyzed = to_ts(datetime(2026, 11, 18, 15, 0, tzinfo=UTC))
    [item] = repo_news.insert_news(
        db, [NewsItem(uid="n1", title_key="sym0 steigt", source="test", url="https://example.invalid/1",
                      title="SYM0 steigt", fetched_ts=analyzed - 60)]
    )  # fmt: skip
    assert item.id is not None
    repo_news.save_analysis(
        db, NewsAnalysis(news_id=item.id, analyzed_ts=analyzed, model="test", relevance=5, sentiment=0.8,
                         category="product", horizon="intraday", summary_de="SYM0 steigt", symbols={"SYM0": 0.8})
    )  # fmt: skip
    start, end = validation_period(NOW, 10)
    result = replay(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, start_ts=start, end_ts=end)
    assert result.news_rows == 1
    report = build_report(result, n_random=0)
    assert not any("Nachrichtenhistorie" in warning for warning in report.warnings_de)


HANDMADE_COST_R = 0.05


def prior_expected_r(score: float, rr: float = 2.0, cost_r: float = HANDMADE_COST_R) -> float:
    """The expected R the replay stores with a signal (prior calibration, like ``validation._issue``)."""
    return Calibrator(ENGINE_PARAMS).expected_r(score, rr, cost_r)


def closed_signal(
    symbol: str, created: int, outcome: float, *, score: float = 0.7, number: int = 0, exit_after: int = 660
) -> Signal:
    """A closed replay signal decided at ``created``, entered a minute later, exiting ``exit_after`` s later."""
    entry = created + 60
    return Signal(
        created_ts=created, data_ts=created - 60, symbol=symbol, direction="LONG", setup="breakout",
        timeframe="15m", horizon_min=180, ref_price=100.0, atr=1.0, stop_dist=1.0, target_dist=2.0,
        score=score, confidence=0.5, engine_version="test", status="target" if outcome > 0 else "stopped",
        issued_ts=created, entry_ts=entry, exit_ts=created + exit_after, outcome_r=outcome, id=number or None,
        expected_r=prior_expected_r(score),
    )  # fmt: skip


def handmade_replay(outcomes: list[tuple[str, int, float]], *, score: float = 0.7) -> Replay:
    """A replay result with closed signals (symbol, fold 1..4, outcome in R), news history and no market data."""
    bounds = fold_bounds(START, END)
    result = Replay(start_ts=START, end_ts=END, trading_days=21, news_rows=3)
    for number, (symbol, fold, outcome) in enumerate(outcomes, start=1):
        created = bounds[fold - 1][0] + 3540 + number * 60
        result.signals.append(closed_signal(symbol, created, outcome, score=score, number=number))
        result.instruments.setdefault(symbol, replace(TEMPLATE, symbol=symbol))
    return result


def test_dominance_warnings_name_the_fold_or_instrument_that_carries_the_result() -> None:
    even = handmade_replay([(f"S{k % 2}", 1 + k % 4, 1.0 if k % 3 else -0.5) for k in range(24)])
    report = build_report(even, n_random=0)
    assert not any("nicht stabil" in w or "hängt an einem Instrument" in w for w in report.warnings_de)

    one_fold = handmade_replay([("S0", 2, 2.0), ("S1", 2, 2.0), ("S0", 1, 0.5), ("S1", 3, -0.5), ("S0", 4, 0.5)])
    text = "\n".join(build_report(one_fold, n_random=0).warnings_de)
    assert "Ergebnis zeitlich nicht stabil: Abschnitt 2 von 4 liefert +4,0 R des Gesamtergebnisses von +4,5 R" in text

    one_symbol = handmade_replay([("S0", 1, 2.0), ("S0", 2, 2.0), ("S0", 3, 2.0), ("S1", 4, 1.0), ("S1", 1, -1.0)])
    text = "\n".join(build_report(one_symbol, n_random=0).warnings_de)
    assert "Ergebnis hängt an einem Instrument: S0 liefert +6,0 R des Gesamtergebnisses von +6,0 R" in text

    losing = handmade_replay([("S0", 3, -1.0), ("S1", 3, -1.0), ("S0", 3, -1.0), ("S1", 1, 0.5), ("S0", 4, -0.5)])
    text = "\n".join(build_report(losing, n_random=0).warnings_de)
    assert "Abschnitt 3 von 4 liefert −3,0 R des Gesamtergebnisses von −3,0 R" in text

    single = handmade_replay([("S0", 1, 1.0), ("S0", 2, 1.0)])  # one instrument: nothing to compare with
    assert not any("hängt an einem Instrument" in w for w in build_report(single, n_random=0).warnings_de)


def test_only_selected_signals_enter_the_statistics_but_all_enter_the_calibration() -> None:
    result = handmade_replay([("S0", 1, 1.0), ("S1", 2, -1.0), ("S0", 3, 2.0)], score=0.55)
    first = result.signals[0]
    assert first.entry_ts is not None
    result.signals.append(
        replace(first, id=99, score=0.65, entry_ts=first.entry_ts + 7, expected_r=prior_expected_r(0.65))
    )
    report = build_report(result, min_score=0.6, n_random=0)
    assert report.overall.n == 1 and report.overall.total_r == 1.0
    assert sum(cell.n for cell in report.calibration) == 4
    assert [g.key for g in report.by_direction] == ["LONG"] and report.by_direction[0].label_de == "Long (Kauf)"
    assert report.by_region[0].label_de == "USA" and report.by_setup[0].label_de == "Ausbruch"
    assert report.equity_r == [(result.signals[-1].exit_ts, 1.0)]


def test_selected_signals_need_a_positive_expected_r_like_live() -> None:
    """INT-1: the live engine shadows every candidate whose expected R after costs is not positive."""
    result = handmade_replay([("S0", 1, 1.0), ("S1", 2, 2.0)])
    expensive = closed_signal("S0", fold_bounds(START, END)[2][0] + 7200, 2.0, number=50)
    expensive.expected_r = prior_expected_r(0.7, cost_r=0.5)  # costs eat the whole edge: never issued live
    assert expensive.expected_r < 0.0
    result.signals.append(expensive)
    gated = build_report(result, n_random=0)
    assert gated.overall.n == 2 and gated.overall.total_r == pytest.approx(3.0)
    assert build_report(result, n_random=0, expectancy_gate=False).overall.n == 3
    assert sum(cell.n for cell in gated.calibration) == 3  # the calibration still sees every candidate


def test_the_expectancy_rule_uses_the_walk_forward_calibrator_of_the_signals_fold() -> None:
    """INT-1, faithful version: after a run of losers the live calibrator - and so the replay's fold
    calibrator - turns the expected R of the same setup negative, although the prior said positive."""
    losers = [("S0" if k % 2 else "S1", 1 + k % 2, -1.0) for k in range(30)]  # folds 1 and 2
    winners = [("S0", 3, 2.0) for _ in range(5)]
    result = handmade_replay([*losers, *winners])
    assert all(signal.expected_r > 0.0 for signal in result.signals)  # positive under the prior
    bounds = fold_bounds(START, END)
    third = fold_calibrators(result.signals, bounds, lag_s=knowledge_lag_s(result.instruments.values()))[2]
    assert third.expected_r(0.7, 2.0, HANDMADE_COST_R) < 0.0
    gated = build_report(result, n_random=0)
    assert gated.folds[2].stats.n == 0  # the winners of fold 3 would not have been issued live
    assert gated.folds[0].stats.n == 15  # fold 1 is judged with the prior: all issued
    assert build_report(result, n_random=0, expectancy_gate=False).folds[2].stats.n == 5


def test_knowledge_lag_covers_the_largest_feed_delay() -> None:
    sap = next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "SAP.DE")
    assert knowledge_lag_s([]) == 60
    assert knowledge_lag_s([TEMPLATE]) == 60
    assert knowledge_lag_s([TEMPLATE, sap]) == (sap.delay_min + 1) * 60 == 960
    with pytest.raises(ValueError):
        fold_calibrators([], fold_bounds(START, END), lag_s=-1)


def test_walk_forward_calibration_uses_only_outcomes_known_at_the_fold_start() -> None:
    """CAUS-5: an outcome is known one minute (bar close) plus the feed delay after the exit bar opened. Losers
    whose exits were not yet KNOWN when fold 3 started must not reach its calibrator (signal decided at the
    fold start + 2 minutes)."""
    sap = replace(next(i for i in SAMPLE_INSTRUMENTS if i.symbol == "SAP.DE"), tradable=True)
    bounds = fold_bounds(START, END)
    lo = bounds[2][0]
    lag = knowledge_lag_s([sap])
    result = Replay(start_ts=START, end_ts=END, trading_days=21, news_rows=3, instruments={sap.symbol: sap})
    # 40 losers that exit within the last 15 minutes before fold 3: their data arrives after the fold start
    for k in range(40):
        exit_ts = lo - lag + 10 + k * 20
        result.signals.append(closed_signal(sap.symbol, exit_ts - 660, -1.0, number=k + 1))
    decided = closed_signal(sap.symbol, lo + 120, 2.0, number=99)
    result.signals.append(decided)
    report = build_report(result, n_random=0)
    assert report.folds[2].stats.n == 1  # issued: the fold-3 calibrator is still the prior
    unlagged = fold_calibrators(result.signals, bounds)[2]  # what the replay did before: lo without lag
    assert unlagged.expected_r(0.7, 2.0, HANDMADE_COST_R) < 0.0
    lagged = fold_calibrators(result.signals, bounds, lag_s=lag)[2]
    assert lagged.expected_r(0.7, 2.0, HANDMADE_COST_R) == pytest.approx(prior_expected_r(0.7))


def test_calibration_table_assigns_candidates_by_decision_time() -> None:
    """A candidate decided in fold 2 but entered in fold 3 is predicted by fold 2's calibrator."""
    bounds = fold_bounds(START, END)
    lo = bounds[2][0]
    result = handmade_replay([("S0", 2, -1.0) for _ in range(20)])  # losers of fold 2 (score 0.7)
    late = closed_signal("S0", lo - 30, 2.0, number=77, score=0.85)
    late.entry_ts = lo + 30  # entered after the fold boundary
    result.signals.append(late)
    report = build_report(result, n_random=0)
    calibrators = fold_calibrators(result.signals, bounds, lag_s=knowledge_lag_s(result.instruments.values()))
    cell = next(c for c in report.calibration if c.lo <= 0.85 < c.hi)
    assert cell.n == 1 and cell.predicted == pytest.approx(calibrators[1].probability(0.85, 2.0))
    assert calibrators[1].probability(0.85, 2.0) != pytest.approx(calibrators[2].probability(0.85, 2.0))


def make_report(
    *, n: int, avg: float, lo: float | None, hi: float | None, p: float | None, positive_folds: int
) -> ValidationResultOut:
    folds = [
        FoldOut(index=k + 1, start_ts=k, end_ts=k + 1, stats=StatBlock(n=5, avg_r=0.1 if k < positive_folds else -0.1))
        for k in range(4)
    ]
    return ValidationResultOut(
        period_start_ts=0, period_end_ts=4, trading_days=20, symbols=10, decisions=1000,
        overall=StatBlock(n=n, avg_r=avg, avg_r_lo=lo, avg_r_hi=hi), folds=folds,
        baseline=None if p is None else BaselineOut(n_sims=200, mean_avg_r=0.0, p95_avg_r=0.1, p_value=p),
    )  # fmt: skip


def test_verdict_claims_an_edge_only_when_every_criterion_holds() -> None:
    edge = verdict_de(make_report(n=150, avg=0.2, lo=0.05, hi=0.35, p=0.01, positive_folds=3))
    assert edge.startswith("Hinweis auf einen Vorteil")
    facts = "Erwartungswert +0,20 R (95 %-Intervall +0,05 bis +0,35) bei 150 Signalen; Zufallsvergleich p = 0,01."
    assert facts in edge
    for weaker in (
        make_report(n=150, avg=0.2, lo=0.05, hi=0.35, p=0.2, positive_folds=3),  # baseline not beaten
        make_report(n=99, avg=0.2, lo=0.05, hi=0.35, p=0.01, positive_folds=3),  # too few signals
        make_report(n=150, avg=0.2, lo=0.05, hi=0.35, p=0.01, positive_folds=2),  # unstable over time
        make_report(n=150, avg=0.1, lo=-0.02, hi=0.22, p=0.01, positive_folds=4),  # interval includes 0
        make_report(n=150, avg=0.2, lo=0.05, hi=0.35, p=None, positive_folds=4),  # no baseline
    ):
        assert verdict_de(weaker).startswith("Kein belastbarer Vorteil nachweisbar")
    losing = verdict_de(make_report(n=150, avg=-0.15, lo=-0.25, hi=-0.05, p=0.9, positive_folds=0))
    assert losing.startswith("Kein Vorteil") and "−0,15 R" in losing and "p = 0,90" in losing
    assert verdict_de(make_report(n=0, avg=0.0, lo=None, hi=None, p=None, positive_folds=0)).startswith("Keine Aussage")
    tiny = verdict_de(make_report(n=3, avg=0.5, lo=None, hi=None, p=0.004, positive_folds=1))
    assert "kein Intervall" in tiny and "p = 0,004" in tiny


def test_progress_runs_from_zero_to_one(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market[0], walk_market[1][:2]
    seen: list[float] = []
    run_validation(
        db, calendar=CALENDAR, instruments=instruments, costs=COSTS, days=10, min_score=0.0, now_ts=NOW, n_random=10,
        progress=seen.append,
    )  # fmt: skip
    assert seen and seen[-1] == 1.0 and all(0.0 <= value <= 1.0 for value in seen)
    assert seen == sorted(seen)

    def broken(_: float) -> None:
        raise RuntimeError("anzeige kaputt")

    report = run_validation(
        db, calendar=CALENDAR, instruments=instruments, costs=COSTS, days=10, now_ts=NOW, n_random=0, progress=broken
    )
    assert report.symbols == 2


def test_invalid_arguments_are_rejected(walk_market: tuple[Database, list[Instrument]]) -> None:
    db, instruments = walk_market
    invalid: list[dict[str, Any]] = [{"days": 0}, {"min_score": 1.5}, {"n_random": -1}]
    for kwargs in invalid:
        with pytest.raises(ValueError):
            run_validation(db, calendar=CALENDAR, instruments=instruments, costs=COSTS, now_ts=NOW, **kwargs)


# ------------------------------------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def cli_db(db: Database) -> Database:
    seed_market(db, "walk", 1, first=date(2026, 10, 26))
    return db


def run_cli(settings: Settings, *argv: str) -> int:
    return validate.main(list(argv), settings=settings, calendar=CALENDAR, clock=ManualClock(NOW))


def test_cli_creates_and_finishes_a_run(
    settings: Settings, cli_db: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli(settings, "--days", "10") == 0
    [run] = repo_ops.list_validation_runs(cli_db)
    assert run.status == "done" and run.params == {"days": 10} and run.engine_version == ENGINE_PARAMS.version
    assert run.ts == NOW and run.message.startswith("Abgeschlossen in")
    assert run.result is not None and run.result.symbols == 1
    assert run.result.period_end_ts == NOW and run.result.period_start_ts == NOW - 10 * 86_400
    output = capsys.readouterr().out
    assert f"Validierung #{run.id} abgeschlossen" in output and run.result.verdict_de in output
    assert "Fortschritt: 100 %" in output


def test_cli_finishes_the_run_created_by_the_api(settings: Settings, cli_db: Database) -> None:
    run_id = repo_ops.create_validation_run(
        cli_db, ts=NOW - 5, engine_version=ENGINE_PARAMS.version, params={"days": 10}
    )
    assert run_cli(settings, "--days", "10", "--run-id", str(run_id)) == 0
    [run] = repo_ops.list_validation_runs(cli_db)
    assert run.id == run_id and run.status == "done" and run.result is not None


def test_cli_marks_the_run_failed(
    settings: Settings, cli_db: Database, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(*_: object, **__: object) -> ValidationResultOut:
        raise RuntimeError("Festplatte kaputt")

    monkeypatch.setattr(validate, "run_validation", explode)
    run_id = repo_ops.create_validation_run(cli_db, ts=NOW, engine_version=ENGINE_PARAMS.version, params={"days": 10})
    assert run_cli(settings, "--run-id", str(run_id)) == 1
    run = repo_ops.get_validation_run(cli_db, run_id)
    assert run is not None and run.status == "failed" and run.result is None
    assert run.message == "Die Validierung ist fehlgeschlagen (RuntimeError): Festplatte kaputt"
    assert "fehlgeschlagen" in capsys.readouterr().err


def test_cli_marks_an_interrupted_run_failed(
    settings: Settings, cli_db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(*_: object, **__: object) -> ValidationResultOut:
        raise KeyboardInterrupt

    monkeypatch.setattr(validate, "run_validation", interrupt)
    run_id = repo_ops.create_validation_run(cli_db, ts=NOW, engine_version=ENGINE_PARAMS.version, params={"days": 10})
    assert run_cli(settings, "--run-id", str(run_id)) == 1
    run = repo_ops.get_validation_run(cli_db, run_id)
    assert run is not None and run.status == "failed" and run.message == "Die Validierung wurde abgebrochen."


def test_cli_refuses_unknown_and_finished_runs(settings: Settings, cli_db: Database) -> None:
    assert run_cli(settings, "--run-id", "999") == 1
    assert repo_ops.list_validation_runs(cli_db) == []
    run_id = repo_ops.create_validation_run(cli_db, ts=NOW, engine_version=ENGINE_PARAMS.version, params={"days": 10})
    repo_ops.finish_validation_run(cli_db, run_id, status="failed", message="alt")
    assert run_cli(settings, "--run-id", str(run_id)) == 1
    run = repo_ops.get_validation_run(cli_db, run_id)
    assert run is not None and run.status == "failed" and run.message == "alt"


@pytest.mark.parametrize("days", ["1", "366", "viele"])
def test_cli_rejects_invalid_days(settings: Settings, days: str) -> None:
    with pytest.raises(SystemExit) as raised:
        run_cli(settings, "--days", days)
    assert raised.value.code == 2


def test_cli_result_round_trips_through_the_database(settings: Settings, cli_db: Database) -> None:
    assert run_cli(settings, "--days", "10") == 0
    [run] = repo_ops.list_validation_runs(cli_db)
    assert run.result is not None
    stored = run.result.model_dump()
    numbers = [stored["overall"]["total_r"], *(cell["predicted"] for cell in stored["calibration"])]
    assert all(math.isfinite(value) for value in numbers)
    assert Path(settings.db_path).exists()
