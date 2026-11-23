"""Validation replay: the out-of-sample evidence for the deterministic part of the signal engine.

The stored minute history is replayed through EXACTLY the live code path - ``features.build_features``,
``setups.detect``, ``scoring.score_hit`` and the life cycle of ``tracking`` with the fill rules of
``trading_local.execution`` - and the outcome is reported so that nobody is fooled: sample sizes, confidence
intervals, stability over four consecutive time blocks, a walk-forward calibration and a random-entry
baseline with identical stops, targets, holding times and costs.

Scope
* Rule engine only. The LLM review cannot be replayed (hours of GPU time, and the model may remember the
  past); its contribution is measured live through the shadow-signal journal.
* No user preferences: regions, disabled setups and allowed directions are not applied - the replay judges
  the rules themselves.
* 'Selected' signals are the ones the live engine would issue on the deterministic rules: score at least
  ``min_score`` AND expected R after costs above zero (the live engine shadows every candidate with
  ``expected_r <= 0``). The expected R is re-evaluated with the walk-forward calibrator of the fold in which
  the signal was decided, i.e. with the outcomes the live engine would have known by then.
* Only stored, final REST history is replayed. The live tracker waits for SETTLED bars
  (``BarReader.settled_bars_after``) because stream/quote bars are provisional and REST fills holes later;
  on final data that rule hands out exactly the stored bars, so the replay tracks on them directly and
  needs no settle logic.

No look-ahead
* Features are causal by construction (``features``). Every bucket is decided when the minute cycle after
  the arrival of its data would have seen it: the bucket ending at T at ``T + delay_min * 60 + 60``
  (``decision_lag_s``). That time is the decision AND issue time, so the entry is the first bar opening at
  or after it - the handicap of delayed data, and for real-time feeds the one-bar wait of the live engine,
  which issues at clock time after the data arrived and therefore never fills on the bar opening at T.
* News come from ``repo_news.symbol_news`` as of the decision time (filtered on ``analyzed_ts``; ages are
  counted from the item's first sight, ``features.news_age_ref_ts``).
* A closed signal frees its symbol only once the data of its closing bar would have arrived. Live, the tracker
  closes a signal when its exit bar is SETTLED (REST data, usually within one reconciliation round; a
  provisional bar at the latest ``barstore.SETTLE_S`` later), so the live engine may keep a symbol blocked a
  few minutes longer than the replay; arrival times of REST bars are not stored, so the replay cannot
  reproduce that wait and uses its lower bound.
* The calibration used for fold k is fitted only on outcomes whose exit was KNOWN before fold k started:
  exit bar open time + 60 s + the largest feed delay of the replayed instruments before the fold start
  (``knowledge_lag_s``). Signals belong to the fold of their decision time, so every outcome a fold's
  calibrator uses was known at each of its decisions.
* Reading bars through ``repo_market`` instead of ``BarReader`` is allowed here, and only here: the
  replay loads each history once and enforces the as-of bounds itself (each feature pass only gets the
  bars that had closed at its decision times and the daily bars of sessions completed before them, the
  feature builder only uses data closed at the row's time, the tracker only bars opening at or after the
  issue time).

Same inputs and guards as live (``engine``). The live engine computes each feature row on a ROLLING window:
the 1-minute bars from ``engine.warmup_start_ts(now)`` and the newest ``engine.DAILY_BARS`` daily bars of
completed sessions. Window-based and seeded indicators (volatility rank, daily EMA200) depend on where that
window starts, so one feature pass over the whole history would NOT reproduce the live values. The replay
therefore loads each history once and runs ``features.build_features`` once per live input window - the
decision times that share the same warm-up start and last completed session, i.e. about one pass per
trading date - on exactly those inputs (``tests/analysis/test_validation.py`` compares the rows with rows
built like the live engine does). Daily features need the daily bar of the venue's previous session (NaN
otherwise, ``features``); the replay passes the daily bars of the sessions completed at each decision, as
``BarReader.daily`` does live. The data service stores a session's daily bar before the first decision that
uses it (24/7 venues: about a minute after UTC midnight, daily features are first needed 15 minutes later);
a daily bar that arrived late live left NaN daily columns there, which the replay cannot reproduce because
arrival times of daily bars are not stored. Guards: warm-up of ``params.min_bars_slow`` completed
slow-timeframe bars inside the live window, the bucket started in a regular session, the venue is open at
the decision time, the setup guards, ONE open signal per symbol (all candidates are tracked - also those
below ``min_score`` - exactly as live tracks shadow signals) and a same-direction cooldown of
``RuntimeSettings.signal_cooldown_min`` (default) after ``engine.closed_ts``.
Signals still open at the end of the data are discarded and counted. A data gap after the time stop or the
session-close flattening is closed on the next bar (``expire_stale`` needs a wall clock and is not replayed).

Determinism: the same database content and arguments always give the same report (seeded bootstrap and
Monte Carlo, chronological processing, stable tie-breaking).
"""

import bisect
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from trading_local import dto, execution
from trading_local.analysis import tracking
from trading_local.analysis.calibration import Calibrator, outcomes_before
from trading_local.analysis.engine import DAILY_BARS, closed_ts, warmup_start_ts
from trading_local.analysis.features import NewsContext, build_features, feature_float, news_context
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.analysis.scoring import score_hit
from trading_local.analysis.setups import SetupHit, detect
from trading_local.analysis.stats import cumulative_r, group_stats, stat_block
from trading_local.barstore import resample
from trading_local.clock import from_ts, to_ts
from trading_local.config import CostParams, RuntimeSettings
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_market, repo_news
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_SECONDS
from trading_local.db.repo_news import SymbolNews
from trading_local.models import SIGNAL_CLOSED_STATUSES, Direction, Instrument, Signal
from trading_local.presenters import SETUP_LABELS_DE

log = logging.getLogger(__name__)

FOLDS = 4
COOLDOWN_MIN: int = RuntimeSettings.model_fields["signal_cooldown_min"].default
DEFAULT_SEED = 20261122

MIN_SIGNALS = 100  # fewer selected signals: sample too small
MIN_TRADING_DAYS = 10
DOMINANCE_SHARE = 0.5  # one fold / one symbol with more than this share of the total R dominates
NOISE_P_VALUE = 0.1  # baseline p above this: indistinguishable from chance
EDGE_P_VALUE = 0.05
MIN_POSITIVE_FOLDS = 3

_DAY_SECONDS = 86_400
_TRACK_CHUNK_BARS = 120  # extra bars per tracker call beyond horizon + entry grace (data gaps, pre/post bars)
_REPLAY_SHARE = 0.85  # share of the progress bar used by the replay; the Monte Carlo gets the rest
_MAX_LISTED_SYMBOLS = 10
_MINUS = "−"  # typographic minus sign

REGION_LABELS_DE: dict[str, str] = {"US": "USA", "EU": "Europa", "ASIA": "Asien", "GLOBAL": "Global"}
DIRECTION_LABELS_DE: dict[str, str] = {"LONG": "Long (Kauf)", "SHORT": "Short (Leerverkauf)"}

type Progress = Callable[[float], None]


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
def decision_lag_s(instrument: Instrument) -> int:
    """Seconds from the end of a bucket (data time) to the decision on it: ``delay_min * 60 + 60``.

    The data arrives ``delay_min`` minutes after the bucket ended and is picked up by the next minute cycle;
    a real-time feed (``delay_min = 0``) is decided one minute after the bucket end, like the live engine
    that issues at clock time just after the data arrived (its entry bar is the one opening at T + 60).
    """
    return instrument.delay_min * 60 + 60


def fold_bounds(start_ts: int, end_ts: int, folds: int = FOLDS) -> list[tuple[int, int]]:
    """``folds`` consecutive half-open blocks ``[lo, hi)`` of (almost) equal duration covering [start, end)."""
    if folds < 1 or end_ts <= start_ts:
        raise ValueError(f"need folds >= 1 and end > start, got {folds}, [{start_ts}, {end_ts})")
    edges = [start_ts + (end_ts - start_ts) * k // folds for k in range(folds + 1)]
    return list(zip(edges[:-1], edges[1:]))


def knowledge_lag_s(instruments: Iterable[Instrument]) -> int:
    """Seconds from the open time of an exit bar until the outcome is known for every replayed instrument:
    the bar closes 60 s after it opened and its data arrives ``delay_min`` minutes later (largest delay)."""
    return (max((instrument.delay_min for instrument in instruments), default=0) + 1) * BAR_SECONDS


def fold_calibrators(
    signals: Sequence[Signal],
    bounds: Sequence[tuple[int, int]],
    params: EngineParams = ENGINE_PARAMS,
    *,
    lag_s: int = 0,
) -> list[Calibrator]:
    """Walk-forward calibrators: the one of fold k is fitted on the outcomes of ALL candidates whose exit was
    known before fold k started - exit time (open time of the exit bar) before ``lo - lag_s`` - or the prior
    when there is none, as for the first fold of a replay. ``build_report`` passes ``knowledge_lag_s``."""
    if lag_s < 0:
        raise ValueError(f"lag_s must not be negative, got {lag_s}")
    return [Calibrator.fit(outcomes_before(signals, before_ts=lo - lag_s), params) for lo, _ in bounds]


def fold_index(ts: int, bounds: Sequence[tuple[int, int]]) -> int:
    """Index of the fold ``[lo, hi)`` that contains ``ts`` (times outside the bounds: the nearest fold)."""
    starts = [lo for lo, _ in bounds]
    return min(max(bisect.bisect_right(starts, ts) - 1, 0), len(bounds) - 1)


class _ProgressReporter:
    """Calls a progress callback with values mapped into [lo, hi]; a failing callback never stops the run."""

    def __init__(self, callback: Progress | None, lo: float = 0.0, hi: float = 1.0) -> None:
        self._callback = callback
        self._lo = lo
        self._hi = hi

    def __call__(self, fraction: float) -> None:
        if self._callback is None:
            return
        value = self._lo + (self._hi - self._lo) * min(max(fraction, 0.0), 1.0)
        try:
            self._callback(value)
        except Exception:
            log.warning("progress callback failed; progress reporting disabled", exc_info=True)
            self._callback = None


# ------------------------------------------------------------------------------------------------
# Random-entry tracker
# ------------------------------------------------------------------------------------------------
class TradeSimulator:
    """Vectorised twin of ``tracking.advance`` for one symbol's closed 1-minute bars (Monte Carlo baseline).

    ``outcome`` returns exactly the ``outcome_r`` the tracker records for a signal issued at ``issued_ts``
    (``tests/analysis/test_validation.py`` compares both on random trades). Only the search for the exit
    bar is vectorised; the entry deadline, the drift check, the fills, the exit precedence and the result in
    R come from the same functions the tracker uses (``tracking``, ``execution``). Time stops and the
    session-close times are computed once per entry bar and horizon and cached.
    """

    def __init__(
        self,
        instrument: Instrument,
        minutes: pd.DataFrame,
        *,
        calendar: MarketCalendar,
        costs: CostParams,
        params: EngineParams = ENGINE_PARAMS,
    ) -> None:
        ts, values = repo_market.bar_arrays(minutes)
        self._instrument = instrument
        self._calendar = calendar
        self._costs = costs
        self._params = params
        self._ts = ts.copy()
        self._open = np.ascontiguousarray(values[:, 0])
        self._high = np.ascontiguousarray(values[:, 1])
        self._low = np.ascontiguousarray(values[:, 2])
        self._limits: dict[tuple[int, int, bool], tuple[int, int | None, int]] = {}

    def outcome(
        self,
        direction: Direction,
        *,
        issued_ts: int,
        ref_price: float,
        stop_dist: float,
        target_dist: float,
        horizon_min: int,
        intraday: bool,
    ) -> float | None:
        """Result in R after costs of a trade issued at ``issued_ts``; None when it never entered (entry
        deadline missed, price ran away) or is still open at the end of the bars."""
        entry = int(np.searchsorted(self._ts, issued_ts, side="left"))
        deadline = tracking.entry_deadline(issued_ts, self._instrument, self._params)
        if entry >= self._ts.size or self._ts[entry] > deadline:
            return None
        bar_open = float(self._open[entry])
        if execution.drift_exceeded(ref_price, bar_open, self._params.max_entry_drift * stop_dist):
            return None
        entry_price = execution.entry_fill_price(self._costs, self._instrument, direction, bar_open)
        sign = 1 if direction == "LONG" else -1
        stop = entry_price - sign * stop_dist
        target = entry_price + sign * target_dist
        expires, flatten, end = self._limits_for(entry, horizon_min, intraday)
        opens, highs, lows = self._open[entry:end], self._high[entry:end], self._low[entry:end]
        if direction == "LONG":
            touched = (opens <= stop) | (opens >= target) | (lows <= stop) | (highs >= target)
        else:
            touched = (opens >= stop) | (opens <= target) | (highs >= stop) | (lows <= target)
        if touched.any():
            exit_bar = entry + int(touched.argmax())
        elif end < self._ts.size:
            exit_bar = end  # first bar at or after the time stop / session flattening
        else:
            return None
        exit_ = execution.exit_on_bar(
            side=direction,
            bar_ts=int(self._ts[exit_bar]),
            bar_open=float(self._open[exit_bar]),
            bar_high=float(self._high[exit_bar]),
            bar_low=float(self._low[exit_bar]),
            stop=stop,
            target=target,
            time_stop_ts=expires,
            flatten_ts=flatten,
        )
        if exit_ is None:
            raise RuntimeError("exit search and execution.exit_on_bar disagree")
        exit_price = execution.exit_fill_price(self._costs, self._instrument, direction, exit_)
        return execution.result_r(direction, entry_price, exit_price, stop_dist)

    def _limits_for(self, entry: int, horizon_min: int, intraday: bool) -> tuple[int, int | None, int]:
        """(time stop, session flattening or None, index of the first bar at or after the earlier of both)."""
        key = (entry, horizon_min, intraday)
        cached = self._limits.get(key)
        if cached is None:
            entry_ts = int(self._ts[entry])
            exchange = self._instrument.exchange
            expires = to_ts(self._calendar.add_trading_minutes(exchange, from_ts(entry_ts), horizon_min))
            flatten = (
                execution.session_flatten_ts(self._calendar, exchange, entry_ts, self._params.flatten_before_close_min)
                if intraday
                else None
            )
            limit = expires if flatten is None else min(expires, flatten)
            end = max(entry, int(np.searchsorted(self._ts, limit, side="left")))
            cached = self._limits[key] = (expires, flatten, end)
        return cached


# ------------------------------------------------------------------------------------------------
# Replay
# ------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class SymbolMarket:
    """What the random-entry baseline needs of one replayed symbol: its bars and its decision rows.

    Decision rows are the rows inside the period that passed the common guards (warm-up, regular session,
    venue open at the decision time); ``intraday_ok`` / ``swing_ok`` additionally mark the rows that pass
    the time-of-day guards of the intraday / swing setups and have a positive slow / daily ATR.
    """

    instrument: Instrument
    simulator: TradeSimulator
    decision_ts: np.ndarray
    price: np.ndarray
    atr_slow: np.ndarray
    atr_daily: np.ndarray
    intraday_ok: np.ndarray
    swing_ok: np.ndarray


@dataclass(slots=True)
class Replay:
    """Raw outcome of a replay over ``[start_ts, end_ts)``.

    ``signals``: every candidate that closed (cancelled ones included), ordered by decision time, symbol and
    direction, with ids 1..n in that order. ``decisions``: rows on which the detectors ran.
    ``trading_days``: distinct trading dates (of any replayed venue) on which decisions were taken.
    """

    start_ts: int
    end_ts: int
    signals: list[Signal] = field(default_factory=list)
    decisions: int = 0
    trading_days: int = 0
    instruments: dict[str, Instrument] = field(default_factory=dict)  # the replayed ones
    markets: dict[str, SymbolMarket] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # not enough data
    failed: list[str] = field(default_factory=list)  # error while replaying
    discarded_open: int = 0
    news_rows: int = 0  # analysed news rows of the replayed symbols inside the period


@dataclass(slots=True)
class _SymbolResult:
    signals: list[Signal]
    decisions: int
    sessions: set[str]  # trading dates ('YYYY-MM-DD') of the decision times
    discarded_open: int
    news_rows: int
    market: SymbolMarket


class _NewsIndex:
    """Analysed news of one symbol, searchable by the time the analysis became known."""

    def __init__(self, rows: Sequence[SymbolNews]) -> None:
        self._rows = sorted(rows, key=lambda row: (row.analyzed_ts, row.news_id))
        self._times = [row.analyzed_ts for row in self._rows]

    def count_between(self, start_ts: int, end_ts: int) -> int:
        """Rows analysed in ``[start_ts, end_ts)``."""
        return bisect.bisect_left(self._times, end_ts) - bisect.bisect_left(self._times, start_ts)

    def context(self, asof_ts: int, params: EngineParams) -> NewsContext:
        """``features.news_context`` of the rows known at ``asof_ts`` inside the news window."""
        lo = bisect.bisect_right(self._times, asof_ts - params.news_window_min * 60)
        hi = bisect.bisect_right(self._times, asof_ts)
        return news_context(self._rows[lo:hi], asof_ts=asof_ts, params=params)


def replay(
    db: Database,
    *,
    calendar: MarketCalendar,
    instruments: Sequence[Instrument],
    costs: CostParams,
    start_ts: int,
    end_ts: int,
    params: EngineParams = ENGINE_PARAMS,
    progress: Progress | None = None,
) -> Replay:
    """Replay the stored history of the tradable, active ``instruments`` over the decision times
    ``[start_ts, end_ts)``; only bars closed at ``end_ts`` are read.

    Each instrument is processed on its own (signals of different symbols do not interact). An instrument
    without enough data is listed in ``skipped``, one that raises in ``failed``; neither stops the replay.
    ``progress`` receives 0..1 after each instrument.
    """
    if end_ts <= start_ts:
        raise ValueError(f"empty period [{start_ts}, {end_ts})")
    report = _ProgressReporter(progress)
    eligible = [instrument for instrument in instruments if instrument.tradable and instrument.active]
    known = {instrument.symbol: instrument for instrument in instruments}
    result = Replay(start_ts=start_ts, end_ts=end_ts)
    benchmarks: dict[tuple[str, int], pd.DataFrame] = {}
    sessions: set[str] = set()
    for done, instrument in enumerate(eligible, start=1):
        try:
            outcome = _replay_symbol(
                db,
                instrument,
                known,
                benchmarks,
                calendar=calendar,
                costs=costs,
                params=params,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except Exception:
            log.exception("validation replay of %s failed; instrument skipped", instrument.symbol)
            result.failed.append(instrument.symbol)
        else:
            if outcome is None:
                result.skipped.append(instrument.symbol)
            else:
                result.signals.extend(outcome.signals)
                result.decisions += outcome.decisions
                result.discarded_open += outcome.discarded_open
                result.news_rows += outcome.news_rows
                result.instruments[instrument.symbol] = instrument
                result.markets[instrument.symbol] = outcome.market
                sessions.update(outcome.sessions)
        report(done / len(eligible))
    result.signals.sort(key=lambda signal: (signal.created_ts, signal.symbol, signal.direction))
    for number, signal in enumerate(result.signals, start=1):
        signal.id = number
    result.trading_days = len(sessions)
    report(1.0)
    return result


def _replay_symbol(
    db: Database,
    instrument: Instrument,
    known: Mapping[str, Instrument],
    benchmarks: dict[tuple[str, int], pd.DataFrame],
    *,
    calendar: MarketCalendar,
    costs: CostParams,
    params: EngineParams,
    start_ts: int,
    end_ts: int,
) -> _SymbolResult | None:
    """Load one instrument's history once, rebuild its decision rows as live and walk them (None: no data)."""
    first_ts = warmup_start_ts(calendar, instrument.exchange, start_ts)
    minutes = _closed_minutes(db, instrument.symbol, first_ts, end_ts)
    if minutes.empty:
        return None
    candidates = _candidate_decisions(minutes, instrument, calendar, params, start_ts, end_ts)
    if candidates.size == 0:
        return None
    daily = repo_market.read_bars_1d(db, instrument.symbol, end_day=from_ts(end_ts).date().isoformat())
    bench = _benchmark(db, instrument, known, benchmarks, first_ts, end_ts)
    selected, decision_ts = _live_rows(
        minutes, daily, bench, candidates, instrument=instrument, calendar=calendar, params=params
    )
    if decision_ts.size == 0:
        return None
    news = _NewsIndex(
        repo_news.symbol_news(
            db, instrument.symbol, asof_ts=end_ts, window_s=end_ts - start_ts + params.news_window_min * 60
        )
    )
    signals, decisions, discarded = _walk(
        instrument,
        selected.to_dict("records"),
        decision_ts,
        minutes,
        news,
        calendar=calendar,
        costs=costs,
        params=params,
    )
    market = _symbol_market(instrument, minutes, selected, decision_ts, calendar=calendar, costs=costs, params=params)
    trading_dates = _constant_runs(decision_ts, lambda at: calendar.session_date(instrument.exchange, from_ts(at)))
    return _SymbolResult(
        signals=signals,
        decisions=decisions,
        sessions={day.isoformat() for _, _, day in trading_dates},
        discarded_open=discarded,
        news_rows=news.count_between(start_ts, end_ts),
        market=market,
    )


def _closed_minutes(db: Database, symbol: str, first_ts: int, end_ts: int) -> pd.DataFrame:
    """1-minute bars opening at or after ``first_ts`` that had closed at ``end_ts`` (ts + 60 <= end_ts)."""
    return repo_market.read_bars_1m(db, symbol, start_ts=first_ts, end_ts=end_ts - BAR_SECONDS + 1)


def _bars_window(bars: pd.DataFrame, first_ts: int, closed_at: int) -> pd.DataFrame:
    """The bars of a canonical 1-minute frame opening at or after ``first_ts`` that had closed at ``closed_at``."""
    ts = bars.index.to_numpy(dtype=np.int64)
    lo = int(np.searchsorted(ts, first_ts, side="left"))
    hi = int(np.searchsorted(ts, closed_at - BAR_SECONDS, side="right"))
    return bars.iloc[lo:hi]


def _daily_window(daily: pd.DataFrame, last_day: date | None) -> pd.DataFrame:
    """The newest ``DAILY_BARS`` daily bars up to and including ``last_day`` (what ``BarReader.daily`` returns
    to the live engine); none when no session has completed."""
    if last_day is None:
        return daily.iloc[:0]
    count = int(np.count_nonzero(daily.index <= last_day.isoformat()))
    return daily.iloc[max(0, count - DAILY_BARS) : count]


@dataclass(frozen=True, slots=True)
class _Benchmark:
    """Closed 1-minute bars of an instrument's benchmark and the shift that aligns them with its decisions.

    If the benchmark's data arrives later than the instrument's (larger decision lag), its bars are shifted
    forward by the difference: the feature row of data time T then sees exactly the benchmark bars that had
    arrived at the row's decision time. Slow buckets of a shifted benchmark are anchored on the shifted
    times (causal, but not bucket-for-bucket identical to live operation).
    """

    bars: pd.DataFrame
    shift_s: int

    def window(self, first_ts: int, closed_at: int) -> pd.DataFrame | None:
        """The benchmark bars the live engine passes for a window starting at ``first_ts`` and evaluated at
        ``closed_at`` (shifted; None without bars)."""
        frame = _bars_window(self.bars, first_ts, closed_at - self.shift_s)
        if frame.empty:
            return None
        if self.shift_s == 0:
            return frame
        return frame.set_axis(pd.Index(frame.index.to_numpy(dtype=np.int64) + self.shift_s, name="ts"), axis=0)


def _benchmark(
    db: Database,
    instrument: Instrument,
    known: Mapping[str, Instrument],
    cache: dict[tuple[str, int], pd.DataFrame],
    first_ts: int,
    end_ts: int,
) -> _Benchmark | None:
    """The benchmark of ``instrument`` with its closed bars from ``first_ts`` (None without benchmark or bars)."""
    symbol = instrument.benchmark
    if not symbol or symbol == instrument.symbol:
        return None
    key = (symbol, first_ts)
    if key not in cache:
        cache[key] = _closed_minutes(db, symbol, first_ts, end_ts)
    frame = cache[key]
    if frame.empty:
        return None
    reference = known.get(symbol) or repo_market.get_instrument(db, symbol)
    shift = 0 if reference is None else decision_lag_s(reference) - decision_lag_s(instrument)
    return _Benchmark(frame, max(shift, 0))


def _candidate_decisions(
    minutes: pd.DataFrame,
    instrument: Instrument,
    calendar: MarketCalendar,
    params: EngineParams,
    start_ts: int,
    end_ts: int,
) -> np.ndarray:
    """Decision times (ascending) of the complete fast buckets that are decided inside ``[start, end)`` while
    the venue is open (live operation only detects for open venues). The remaining guards need the live
    feature rows (``_live_rows``)."""
    ts = minutes.index.to_numpy(dtype=np.int64)
    fast = resample(minutes, params.decision_minutes, complete_before_ts=int(ts[-1]) + BAR_SECONDS)
    bucket_end = fast.index.to_numpy(dtype=np.int64) + params.decision_minutes * BAR_SECONDS
    decision = bucket_end + decision_lag_s(instrument)
    decision = decision[(decision >= start_ts) & (decision < end_ts)]
    exchange = instrument.exchange
    is_open = np.fromiter(
        (calendar.is_open(exchange, from_ts(int(at))) for at in decision), dtype=bool, count=decision.size
    )
    return decision[is_open]


def _live_window(calendar: MarketCalendar, exchange: str, at: int) -> tuple[int, date | None]:
    """(start of the 1-minute history, last completed session) the live engine reads at decision time ``at``."""
    return warmup_start_ts(calendar, exchange, at), calendar.last_completed_session(exchange, from_ts(at))


def _constant_runs[K](times: np.ndarray, key: Callable[[int], K]) -> list[tuple[int, int, K]]:
    """Split ascending ``times`` into consecutive runs ``[lo, hi)`` with the same ``key``, in order.

    ``key`` must never return to an earlier value as time goes on (the live window and the trading date only
    move forward), so equal keys at both ends of a range mean the same key everywhere in between: ranges are
    bisected until that holds, which needs about 2 * runs * log2(n / runs) calls of ``key`` instead of n.
    """
    cache: dict[int, K] = {}

    def key_at(index: int) -> K:
        if index not in cache:
            cache[index] = key(int(times[index]))
        return cache[index]

    runs: list[tuple[int, int, K]] = []
    pending = [(0, int(times.size))] if times.size else []
    while pending:
        lo, hi = pending.pop()
        if key_at(lo) == key_at(hi - 1):
            if runs and runs[-1][1] == lo and runs[-1][2] == key_at(lo):
                runs[-1] = (runs[-1][0], hi, key_at(lo))
            else:
                runs.append((lo, hi, key_at(lo)))
        else:
            mid = (lo + hi) // 2
            pending.extend(((mid, hi), (lo, mid)))  # the left half is processed first
    return runs


def _live_rows(
    minutes: pd.DataFrame,
    daily: pd.DataFrame,
    benchmark: _Benchmark | None,
    candidates: np.ndarray,
    *,
    instrument: Instrument,
    calendar: MarketCalendar,
    params: EngineParams,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature rows of the candidate decisions exactly as the live engine computes them, and their decision
    times, for the rows that pass the guards needing them.

    ``features.build_features`` runs once per run of decisions that share the live input window (the same
    1-minute history start and the same newest completed session - about one run per trading date) on the
    bars that had closed at the run's last decision, the newest ``DAILY_BARS`` completed daily bars and the
    benchmark from the same start; the rows of earlier decisions in the run are identical because features
    are causal. Guards: the bucket started in a regular session and at least ``params.min_bars_slow`` slow
    buckets of the live window had completed at the row's data time (``engine``'s warm-up rule).
    """
    lag = decision_lag_s(instrument)
    slow_step = params.slow_minutes * BAR_SECONDS
    frames: list[pd.DataFrame] = []
    decisions: list[np.ndarray] = []
    runs = _constant_runs(candidates, lambda at: _live_window(calendar, instrument.exchange, at))
    for lo, hi, (first_ts, last_day) in runs:
        run = candidates[lo:hi]
        window = _bars_window(minutes, first_ts, int(run[-1]))
        if window.empty:
            continue
        features = build_features(
            window,
            _daily_window(daily, last_day),
            instrument=instrument,
            calendar=calendar,
            params=params,
            benchmark_minutes=None if benchmark is None else benchmark.window(first_ts, int(run[-1])),
        )
        positions = features.index.get_indexer(run - lag)
        found = positions >= 0  # a bucket that was not complete at its decision time is not evaluated
        rows = features.iloc[positions[found]]
        slow_end = resample(window, params.slow_minutes).index.to_numpy(dtype=np.int64) + slow_step
        data_ts = rows.index.to_numpy(dtype=np.int64)
        warm = np.searchsorted(slow_end, data_ts, side="right") >= params.min_bars_slow
        keep = warm & rows["in_session"].to_numpy(dtype=bool)
        frames.append(rows[keep])
        decisions.append(run[found][keep])
    if not frames:
        return pd.DataFrame(), np.zeros(0, dtype=np.int64)
    return pd.concat(frames), np.concatenate(decisions)


def _walk(
    instrument: Instrument,
    rows: Sequence[Mapping[str, object]],
    decision_ts: np.ndarray,
    minutes: pd.DataFrame,
    news: _NewsIndex,
    *,
    calendar: MarketCalendar,
    costs: CostParams,
    params: EngineParams,
) -> tuple[list[Signal], int, int]:
    """Walk one symbol's decision rows chronologically: (closed signals, rows evaluated, discarded open).

    Each new signal is tracked to its end immediately (its outcome depends on later bars only), so the walk
    can skip the rows during which the symbol is blocked by it.
    """
    bar_ts = minutes.index.to_numpy(dtype=np.int64)
    prior = Calibrator(params)
    lag = decision_lag_s(instrument)
    blocked_until = -1  # decisions before this time see the previous signal as still open
    cooldown_until: dict[str, int] = {"LONG": -1, "SHORT": -1}  # decisions at or before this time are cooling
    signals: list[Signal] = []
    decisions = 0
    for row, decision in zip(rows, decision_ts.tolist(), strict=True):
        if decision < blocked_until:
            continue
        decisions += 1
        context = news.context(decision, params)
        session_minutes = _session_minutes(row)
        hits = detect(row, instrument=instrument, params=params, news=context, session_minutes=session_minutes)
        best = _best_hit(hits, row, instrument, params, context, decision, cooldown_until)
        if best is None:
            continue
        score, hit, components = best
        signal = _issue(hit, row, instrument, decision, score, components, context, prior, costs, params)
        if not _track(signal, minutes, bar_ts, instrument=instrument, calendar=calendar, costs=costs, params=params):
            return signals, decisions, 1  # open until the end of the data: the symbol stays blocked
        closed = closed_ts(signal)
        blocked_until = closed + BAR_SECONDS + lag  # the data of the closing bar has arrived
        cooldown_until[signal.direction] = closed + COOLDOWN_MIN * 60
        signals.append(signal)
    return signals, decisions, 0


def _session_minutes(row: Mapping[str, object]) -> int | None:
    """Regular trading minutes of the row's trading date (minutes since open + minutes to close)."""
    total = feature_float(row, "minutes_since_open") + feature_float(row, "minutes_to_close")
    return round(total) if math.isfinite(total) and total > 0 else None


def _best_hit(
    hits: Sequence[SetupHit],
    row: Mapping[str, object],
    instrument: Instrument,
    params: EngineParams,
    news: NewsContext,
    decision: int,
    cooldown_until: Mapping[str, int],
) -> tuple[float, SetupHit, dict[str, float]] | None:
    """The highest-scoring hit outside its direction's cooldown (ties: LONG first, as ``detect`` orders)."""
    best: tuple[float, SetupHit, dict[str, float]] | None = None
    for hit in hits:
        if decision <= cooldown_until[hit.direction]:
            continue
        score, components = score_hit(hit, row, instrument=instrument, params=params, news=news)
        if best is None or score > best[0]:
            best = (score, hit, components)
    return best


def _issue(
    hit: SetupHit,
    row: Mapping[str, object],
    instrument: Instrument,
    decision: int,
    score: float,
    components: dict[str, float],
    news: NewsContext,
    prior: Calibrator,
    costs: CostParams,
    params: EngineParams,
) -> Signal:
    """Signal for a hit, issued at its decision time (confidence and expected R from the prior calibration)."""
    candidate = tracking.candidate_from_hit(
        hit, row, instrument=instrument, decision_ts=decision, score=score, components=components, news=news
    )
    rr = candidate.target_dist / candidate.stop_dist
    cost_r = costs.round_trip_cost_r(instrument, candidate.ref_price, candidate.stop_dist)
    signal = tracking.make_signal(
        candidate,
        confidence=prior.probability(score, rr),
        expected_r=prior.expected_r(score, rr, cost_r),
        engine_version=params.version,
    )
    tracking.issue(signal, issued_ts=decision, instrument=instrument, params=params)
    return signal


def _track(
    signal: Signal,
    minutes: pd.DataFrame,
    bar_ts: np.ndarray,
    *,
    instrument: Instrument,
    calendar: MarketCalendar,
    costs: CostParams,
    params: EngineParams,
) -> bool:
    """Advance an issued signal over the bars from its issue time on until it closes; True if it closed.

    The bars are handed to ``tracking.advance`` in chunks a little longer than the signal can live, so a
    typical signal needs one call; the tracker skips bars it has seen, so chunking changes nothing.
    """
    start = int(np.searchsorted(bar_ts, signal.issued_ts, side="left"))
    step = signal.horizon_min + instrument.delay_min + params.entry_grace_min + _TRACK_CHUNK_BARS
    while start < bar_ts.size:
        stop = min(bar_ts.size, start + step)
        tracking.advance(
            signal, minutes.iloc[start:stop], instrument=instrument, calendar=calendar, costs=costs, params=params
        )
        if signal.status in SIGNAL_CLOSED_STATUSES:
            return True
        start = stop
    return False


def _symbol_market(
    instrument: Instrument,
    minutes: pd.DataFrame,
    rows: pd.DataFrame,
    decision_ts: np.ndarray,
    *,
    calendar: MarketCalendar,
    costs: CostParams,
    params: EngineParams,
) -> SymbolMarket:
    """Bars and decision rows of one symbol for the baseline; ``rows`` are the feature rows of the decision
    rows (aligned with ``decision_ts``)."""
    since = rows["minutes_since_open"].to_numpy(dtype=np.float64)
    to_close = rows["minutes_to_close"].to_numpy(dtype=np.float64)
    atr_slow = rows["atr_slow"].to_numpy(dtype=np.float64)
    atr_daily = rows["atr_daily"].to_numpy(dtype=np.float64)
    after_open = since >= params.no_entry_first_min
    return SymbolMarket(
        instrument=instrument,
        simulator=TradeSimulator(instrument, minutes, calendar=calendar, costs=costs, params=params),
        decision_ts=decision_ts,
        price=rows["price"].to_numpy(dtype=np.float64),
        atr_slow=atr_slow,
        atr_daily=atr_daily,
        intraday_ok=after_open & (to_close >= params.no_entry_last_min) & (atr_slow > 0.0),
        swing_ok=after_open & (atr_daily > 0.0),
    )


# ------------------------------------------------------------------------------------------------
# Report
# ------------------------------------------------------------------------------------------------
def build_report(
    result: Replay,
    *,
    params: EngineParams = ENGINE_PARAMS,
    min_score: float = 0.60,
    n_random: int = 200,
    seed: int = DEFAULT_SEED,
    progress: Progress | None = None,
    expectancy_gate: bool = True,
) -> dto.ValidationResultOut:
    """Statistics, folds, walk-forward calibration, random-entry baseline, warnings and verdict of a replay.

    'Selected' signals are the candidates with a realised result that pass the live issue rule of the
    deterministic engine: ``score >= min_score`` and - with ``expectancy_gate`` (default) - an expected R
    after costs above zero (rounded to 4 decimals like the stored live value), evaluated with the
    walk-forward calibrator of the fold that contains the signal's decision time (``_issuable``). All
    statistics except the calibration refer to them; ``expectancy_gate=False`` judges every candidate above
    ``min_score``. The calibration table pools the out-of-sample predictions for ALL candidates with a
    result (fold k predicted by ``fold_calibrators`` with ``knowledge_lag_s`` of the replayed instruments).
    ``progress`` receives 0..1 (nearly all of the work is the Monte Carlo of ``random_baseline``).
    """
    outcomes = [signal for signal in result.signals if signal.outcome_r is not None and signal.entry_ts is not None]
    bounds = fold_bounds(result.start_ts, result.end_ts)
    calibrators = fold_calibrators(
        result.signals, bounds, params, lag_s=knowledge_lag_s(result.instruments.values())
    )
    prior = Calibrator(params)
    selected = [
        signal
        for signal in outcomes
        if signal.score >= min_score
        and (not expectancy_gate or _issuable(signal, calibrators[fold_index(signal.created_ts, bounds)], prior))
    ]
    overall = stat_block([signal.outcome_r for signal in selected if signal.outcome_r is not None])
    folds = [
        dto.FoldOut(
            index=number,
            start_ts=lo,
            end_ts=hi,
            stats=stat_block([s.outcome_r for s in selected if s.outcome_r is not None and lo <= _entry(s) < hi]),
        )
        for number, (lo, hi) in enumerate(bounds, start=1)
    ]
    calibration = _calibration_table(calibrators, outcomes, bounds, params)
    baseline = None
    if overall.avg_r is not None and n_random > 0:
        baseline = random_baseline(
            selected, result.markets, engine_avg_r=overall.avg_r, n_random=n_random, seed=seed, progress=progress
        )
    regions = {symbol: instrument.region for symbol, instrument in result.instruments.items()}
    report = dto.ValidationResultOut(
        period_start_ts=result.start_ts,
        period_end_ts=result.end_ts,
        trading_days=result.trading_days,
        symbols=len(result.instruments),
        decisions=result.decisions,
        overall=overall,
        by_setup=group_stats(selected, lambda signal: signal.setup, SETUP_LABELS_DE),
        by_region=group_stats(selected, lambda signal: regions.get(signal.symbol, "GLOBAL"), REGION_LABELS_DE),
        by_direction=group_stats(selected, lambda signal: signal.direction, DIRECTION_LABELS_DE),
        folds=folds,
        calibration=calibration,
        baseline=baseline,
        equity_r=cumulative_r(selected),
    )
    report.warnings_de = _warnings(report, result, selected, min_score, expectancy_gate)
    report.verdict_de = verdict_de(report)
    _ProgressReporter(progress)(1.0)
    return report


def _entry(signal: Signal) -> int:
    return signal.entry_ts if signal.entry_ts is not None else signal.created_ts


def _issuable(signal: Signal, calibrator: Calibrator, prior: Calibrator) -> bool:
    """True if the live engine's expectancy rule would issue ``signal`` with ``calibrator``: expected R after
    costs, rounded to 4 decimals like the stored live value, above zero.

    The replay stores the PRIOR expected R (``_issue``); the round-trip cost in R it contains is recovered
    from it exactly (without outcomes ``expected_r = p * (1 + rr) - 1 - cost_r``) and the expected R is then
    evaluated with the fold's calibrator, as the live engine does with its current one.
    """
    rr = signal.rr
    if not (math.isfinite(rr) and rr > 0.0 and math.isfinite(signal.expected_r)):
        return False
    cost_r = prior.probability(signal.score, rr) * (1.0 + rr) - 1.0 - signal.expected_r
    return round(calibrator.expected_r(signal.score, rr, cost_r), 4) > 0.0


def _calibration_table(
    calibrators: Sequence[Calibrator],
    outcomes: Sequence[Signal],
    bounds: Sequence[tuple[int, int]],
    params: EngineParams,
) -> list[dto.CalibrationBin]:
    """Reliability table of the walk-forward predictions: per score bin the number of candidates with a
    result, their mean predicted win probability and the observed win rate. A candidate is predicted by the
    calibrator of the fold that contains its decision time (``created_ts``, ``fold_index``)."""
    if len(calibrators) != len(bounds):
        raise ValueError(f"{len(calibrators)} calibrators for {len(bounds)} folds")
    scores: list[float] = []
    predicted: list[float] = []
    wins: list[float] = []
    for signal in outcomes:
        if signal.outcome_r is None:
            continue
        calibrator = calibrators[fold_index(signal.created_ts, bounds)]
        scores.append(signal.score)
        predicted.append(calibrator.probability(signal.score, signal.rr))
        wins.append(1.0 if signal.outcome_r > 0.0 else 0.0)
    edges = np.asarray(params.calibration_bins, dtype=np.float64)
    lower, upper = edges[:-1], edges[1:]
    bins = np.clip(np.searchsorted(upper, np.asarray(scores, dtype=np.float64), side="right"), 0, lower.size - 1)
    probabilities = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(wins, dtype=np.float64)
    latest = calibrators[-1]
    table: list[dto.CalibrationBin] = []
    for index, (lo_edge, hi_edge) in enumerate(zip(lower.tolist(), upper.tolist(), strict=True)):
        members = bins == index
        n = int(members.sum())
        centre = (lo_edge + hi_edge) / 2.0
        mean_predicted = (
            float(probabilities[members].mean()) if n else latest.probability(centre, params.reward_risk)
        )
        table.append(
            dto.CalibrationBin(
                lo=lo_edge,
                hi=min(hi_edge, 1.0),
                n=n,
                predicted=mean_predicted,
                observed=float(observed[members].mean()) if n else None,
            )
        )
    return table


@dataclass(frozen=True, slots=True)
class _Plan:
    """Shape of one selected signal that every random trade copies."""

    market: SymbolMarket
    pool: np.ndarray  # eligible decision rows of the symbol
    atr: np.ndarray  # the ATR the signal's stop was based on, per decision row
    stop_atr: float
    reward_risk: float
    horizon_min: int
    intraday: bool


def random_baseline(
    selected: Sequence[Signal],
    markets: Mapping[str, SymbolMarket],
    *,
    engine_avg_r: float,
    n_random: int,
    seed: int = DEFAULT_SEED,
    progress: Progress | None = None,
) -> dto.BaselineOut | None:
    """Random-entry Monte Carlo (deterministic for a given seed).

    Each of the ``n_random`` runs replaces every selected signal by a trade on a random decision row of the
    SAME symbol inside the period (same guards; intraday or swing time-of-day rules as the signal) with a
    random direction. The trade keeps the signal's stop distance in units of that row's ATR (slow ATR for
    intraday setups, daily ATR for swing setups - the ATR the setups base their distances on), its
    reward-to-risk, horizon and intraday flag, and is tracked with the same rules and costs
    (``TradeSimulator``). A run's score is the mean R of its trades with a result. ``p_value`` = (1 +
    runs with mean R >= ``engine_avg_r``) / (1 + runs). None when no run produced a result. ``progress``
    receives 0..1 after each run.
    """
    report = _ProgressReporter(progress)
    plans = [plan for plan in (_plan(signal, markets) for signal in selected) if plan is not None]
    if not plans:
        return None
    rng = np.random.default_rng(seed)
    averages: list[float] = []
    for run in range(n_random):
        picks = rng.random(len(plans)).tolist()
        flips = rng.integers(0, 2, size=len(plans)).tolist()
        results: list[float] = []
        for plan, pick, flip in zip(plans, picks, flips, strict=True):
            outcome = _random_trade(plan, pick, flip)
            if outcome is not None:
                results.append(outcome)
        if results:
            averages.append(math.fsum(results) / len(results))
        report((run + 1) / n_random)
    if not averages:
        return None
    runs = np.asarray(averages, dtype=np.float64)
    beaten = int((runs >= engine_avg_r).sum())
    return dto.BaselineOut(
        n_sims=int(runs.size),
        mean_avg_r=float(runs.mean()),
        p95_avg_r=float(np.percentile(runs, 95)),
        p_value=(1 + beaten) / (1 + runs.size),
    )


def _plan(signal: Signal, markets: Mapping[str, SymbolMarket]) -> _Plan | None:
    market = markets.get(signal.symbol)
    if market is None or signal.atr <= 0.0 or signal.stop_dist <= 0.0:
        return None
    pool = np.flatnonzero(market.intraday_ok if signal.intraday else market.swing_ok)
    if pool.size == 0:
        return None
    return _Plan(
        market=market,
        pool=pool,
        atr=market.atr_slow if signal.intraday else market.atr_daily,
        stop_atr=signal.stop_dist / signal.atr,
        reward_risk=signal.rr,
        horizon_min=signal.horizon_min,
        intraday=signal.intraday,
    )


def _random_trade(plan: _Plan, pick: float, flip: int) -> float | None:
    row = int(plan.pool[min(int(pick * plan.pool.size), plan.pool.size - 1)])
    stop = plan.stop_atr * float(plan.atr[row])
    return plan.market.simulator.outcome(
        "LONG" if flip == 0 else "SHORT",
        issued_ts=int(plan.market.decision_ts[row]),
        ref_price=float(plan.market.price[row]),
        stop_dist=stop,
        target_dist=plan.reward_risk * stop,
        horizon_min=plan.horizon_min,
        intraday=plan.intraday,
    )


# ------------------------------------------------------------------------------------------------
# German texts
# ------------------------------------------------------------------------------------------------
def _number_de(value: float, digits: int = 2, *, signed: bool = False) -> str:
    """German decimal notation with a typographic minus, e.g. '+0,12' or '−0,10'."""
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", _MINUS).replace(".", ",")


def _p_de(p: float) -> str:
    return _number_de(p, 3 if p < 0.01 else 2)


def _percent_de(share: float) -> str:
    return f"{share * 100.0:.0f} %"


def _count_de(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _symbol_list(symbols: Sequence[str]) -> str:
    listed = ", ".join(symbols[:_MAX_LISTED_SYMBOLS])
    return f"{listed} …" if len(symbols) > _MAX_LISTED_SYMBOLS else listed


def _dominant(totals: Mapping[str, float], total: float) -> tuple[str, float] | None:
    """(key, R) of the group with the largest share of ``total`` if that share exceeds ``DOMINANCE_SHARE``.

    The share is signed: with a negative total the group that lost most dominates. Shares above 100 % are
    possible when the other groups partly cancel each other out.
    """
    if total == 0.0 or not totals:
        return None
    key, value = max(totals.items(), key=lambda item: (item[1] / total, item[0]))
    return (key, value) if value / total > DOMINANCE_SHARE else None


def _warnings(
    report: dto.ValidationResultOut,
    result: Replay,
    selected: Sequence[Signal],
    min_score: float,
    expectancy_gate: bool,
) -> list[str]:
    overall = report.overall
    warnings: list[str] = []
    if overall.n < MIN_SIGNALS:
        rule = f"Bewertung ab {_number_de(min_score)}"
        if expectancy_gate:
            rule += " und positiver Erwartungswert nach Slippage"
        warnings.append(
            f"Stichprobe zu klein: nur {_count_de(overall.n, 'ausgewähltes Signal', 'ausgewählte Signale')} mit "
            f"Ergebnis ({rule}); unter {MIN_SIGNALS} ist keine belastbare Aussage möglich."
        )
    if report.trading_days < MIN_TRADING_DAYS:
        warnings.append(
            f"Nur {_count_de(report.trading_days, 'Handelstag', 'Handelstage')} im Zeitraum – zu kurz, "
            "um zufällige Marktphasen auszugleichen."
        )
    if overall.avg_r_lo is not None and overall.avg_r_hi is not None and overall.avg_r_lo <= 0.0 <= overall.avg_r_hi:
        warnings.append(
            f"Das 95 %-Intervall des Erwartungswerts ({_number_de(overall.avg_r_lo, signed=True)} bis "
            f"{_number_de(overall.avg_r_hi, signed=True)} R) schließt 0 ein."
        )
    baseline = report.baseline
    if baseline is not None and baseline.p_value is not None and baseline.p_value > NOISE_P_VALUE:
        warnings.append(
            "Ergebnis nicht von Zufall unterscheidbar: zufällige Einstiege mit gleichen Stopps, Zielen und "
            f"Haltedauern waren in rund {_percent_de(baseline.p_value)} der Durchläufe mindestens so gut "
            f"(p = {_p_de(baseline.p_value)})."
        )
    total = _number_de(overall.total_r, 1, signed=True)
    fold = _dominant({str(f.index): f.stats.total_r for f in report.folds if f.stats.n}, overall.total_r)
    if fold is not None:
        warnings.append(
            f"Ergebnis zeitlich nicht stabil: Abschnitt {fold[0]} von {len(report.folds)} liefert "
            f"{_number_de(fold[1], 1, signed=True)} R des Gesamtergebnisses von {total} R (mehr als die Hälfte)."
        )
    if len(result.instruments) > 1:
        totals: dict[str, float] = {}
        for signal in selected:
            if signal.outcome_r is not None:
                totals[signal.symbol] = totals.get(signal.symbol, 0.0) + signal.outcome_r
        symbol = _dominant(totals, overall.total_r)
        if symbol is not None:
            warnings.append(
                f"Ergebnis hängt an einem Instrument: {symbol[0]} liefert {_number_de(symbol[1], 1, signed=True)} R "
                f"des Gesamtergebnisses von {total} R (mehr als die Hälfte)."
            )
    if result.news_rows == 0:
        label = SETUP_LABELS_DE.get("news_momentum", "news_momentum")
        warnings.append(
            "Keine Nachrichtenhistorie im Zeitraum: die Nachrichtenkomponente war neutral und das Setup "
            f"„{label}“ konnte nicht auslösen."
        )
    if result.discarded_open:
        verb = "wurde" if result.discarded_open == 1 else "wurden"
        warnings.append(
            f"{_count_de(result.discarded_open, 'Signal war', 'Signale waren')} am Ende der Daten noch offen "
            f"und {verb} nicht gewertet."
        )
    if result.skipped:
        warnings.append(
            f"{_count_de(len(result.skipped), 'Instrument', 'Instrumente')} mangels Daten übersprungen: "
            f"{_symbol_list(result.skipped)}."
        )
    if result.failed:
        warnings.append(
            f"{_count_de(len(result.failed), 'Instrument', 'Instrumente')} wegen eines Fehlers übersprungen "
            f"(Details im Log): {_symbol_list(result.failed)}."
        )
    return warnings


def verdict_de(report: dto.ValidationResultOut) -> str:
    """One or two German sentences; 'Hinweis auf einen Vorteil' only when the 95 % interval of the mean R
    excludes 0, the baseline p-value is at most 0.05, there are at least 100 selected signals and at least
    3 of 4 folds are positive."""
    overall = report.overall
    if overall.n == 0 or overall.avg_r is None:
        return "Keine Aussage möglich: im Zeitraum wurde kein ausgewähltes Signal mit Ergebnis abgeschlossen."
    lo, hi = overall.avg_r_lo, overall.avg_r_hi
    interval = (
        f"95 %-Intervall {_number_de(lo, signed=True)} bis {_number_de(hi, signed=True)}"
        if lo is not None and hi is not None
        else "kein Intervall bei so wenigen Signalen"
    )
    p_value = report.baseline.p_value if report.baseline is not None else None
    chance = f"Zufallsvergleich p = {_p_de(p_value)}" if p_value is not None else "kein Zufallsvergleich möglich"
    facts = (
        f"Erwartungswert {_number_de(overall.avg_r, signed=True)} R ({interval}) bei "
        f"{_count_de(overall.n, 'Signal', 'Signalen')}; {chance}."
    )
    positive = sum(1 for fold in report.folds if fold.stats.n and (fold.stats.avg_r or 0.0) > 0.0)
    edge = (
        lo is not None
        and lo > 0.0
        and p_value is not None
        and p_value <= EDGE_P_VALUE
        and overall.n >= MIN_SIGNALS
        and positive >= MIN_POSITIVE_FOLDS
    )
    if edge:
        return (
            f"Hinweis auf einen Vorteil des Regelwerks (ohne LLM): {facts} "
            f"{positive} von {len(report.folds)} Abschnitten positiv – im Live-Betrieb weiter beobachten."
        )
    if hi is not None and hi < 0.0:
        return f"Kein Vorteil – die Regeln verlieren nach Slippage: {facts}"
    return f"Kein belastbarer Vorteil nachweisbar: {facts}"


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------
def validation_period(now_ts: int, days: int) -> tuple[int, int]:
    """[start, end) of a validation over the last ``days`` calendar days before ``now_ts`` (minute-aligned)."""
    if days < 1:
        raise ValueError(f"days must be at least 1, got {days}")
    end = int(now_ts) // BAR_SECONDS * BAR_SECONDS
    return end - days * _DAY_SECONDS, end


def run_validation(
    db: Database,
    *,
    calendar: MarketCalendar,
    instruments: Sequence[Instrument],
    costs: CostParams,
    days: int = 30,
    params: EngineParams = ENGINE_PARAMS,
    min_score: float = 0.60,
    now_ts: int,
    n_random: int = 200,
    seed: int = DEFAULT_SEED,
    progress: Progress | None = None,
) -> dto.ValidationResultOut:
    """Replay the last ``days`` calendar days of stored 1-minute bars before ``now_ts`` for the tradable,
    active ``instruments`` and report whether the rule engine has an edge (see the module docstring).

    ``progress`` receives values from 0 to 1 (replay up to 0.85, Monte Carlo after that). Raises
    ``ValueError`` for invalid arguments; problems with single instruments only produce warnings.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError(f"min_score must be within [0, 1], got {min_score}")
    if n_random < 0:
        raise ValueError(f"n_random must not be negative, got {n_random}")
    started = time.perf_counter()
    start_ts, end_ts = validation_period(now_ts, days)
    result = replay(
        db,
        calendar=calendar,
        instruments=instruments,
        costs=costs,
        start_ts=start_ts,
        end_ts=end_ts,
        params=params,
        progress=_ProgressReporter(progress, 0.0, _REPLAY_SHARE),
    )
    report = build_report(
        result,
        params=params,
        min_score=min_score,
        n_random=n_random,
        seed=seed,
        progress=_ProgressReporter(progress, _REPLAY_SHARE, 1.0),
    )
    log.info(
        "validation over %d days: %d symbols, %d decisions, %d candidates, %d selected, avg R %s in %.1f s",
        days,
        report.symbols,
        report.decisions,
        len(result.signals),
        report.overall.n,
        "n/a" if report.overall.avg_r is None else f"{report.overall.avg_r:+.3f}",
        time.perf_counter() - started,
    )
    return report


__all__ = [
    "COOLDOWN_MIN",
    "DEFAULT_SEED",
    "FOLDS",
    "Replay",
    "SymbolMarket",
    "TradeSimulator",
    "build_report",
    "decision_lag_s",
    "fold_bounds",
    "fold_calibrators",
    "fold_index",
    "knowledge_lag_s",
    "random_baseline",
    "replay",
    "run_validation",
    "validation_period",
    "verdict_de",
]
