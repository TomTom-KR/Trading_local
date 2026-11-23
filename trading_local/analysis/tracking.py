"""Signal life cycle on closed 1-minute bars (pure: data in, mutated signal out).

review -> pending_entry (issued) -> active (entry filled) -> target | stopped | expired, or -> cancelled.

Every fill decision goes through ``trading_local.execution`` - the same rules as the paper broker and the
validation replay. The tracker processes each bar exactly once (``Signal.checked_ts`` is the open time of
the last bar processed), so passing the same bars twice changes nothing. Closed signals are never touched.

Because every bar is processed once, the caller must hand over FINAL bars only. The live engine reads them
through ``BarReader.settled_bars_after``: provisional stream/quote bars and bars behind a hole that the REST
reconciliation may still fill are held back, so ``checked_ts`` only advances over settled bars. The
wall-clock fallback ``expire_stale`` gets the reader's ``settle_horizon`` for the same reason and does not
close a signal while a traded minute that could still change its outcome may arrive. The validation replay
works on stored final REST history and passes all bars at once.
"""

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from trading_local import execution
from trading_local.analysis.features import FEATURE_COLUMNS, NewsContext, feature_float
from trading_local.analysis.params import ENGINE_PARAMS, EngineParams
from trading_local.analysis.setups import SetupHit
from trading_local.clock import from_ts, to_ts
from trading_local.config import CostParams
from trading_local.contracts import MarketCalendar
from trading_local.db.repo_market import BAR_SECONDS
from trading_local.models import SIGNAL_CLOSED_STATUSES, Candidate, Instrument, Signal, SignalStatus

log = logging.getLogger(__name__)

REASON_NO_ENTRY = "kein_einstieg"  # no entry bar before the entry deadline
REASON_DRIFT = "kurs_entlaufen"  # the market ran away between decision and entry
_STATUS_BY_EXIT: dict[execution.ExitReason, SignalStatus] = {
    "target": "target",
    "stop": "stopped",
    "time_stop": "expired",
    "session_close": "expired",
}
_PENDING_GRACE_S = 120  # expire_stale: extra wait for a delayed entry bar
_ACTIVE_GRACE_S = 600  # expire_stale: extra wait for a delayed exit bar


# ------------------------------------------------------------------------------------------------
# Creating signals
# ------------------------------------------------------------------------------------------------
def _reduced_features(row: Mapping[str, Any]) -> dict[str, Any]:
    """The feature row as JSON-friendly snapshot: finite numbers rounded to 6 significant digits (flags as
    0/1) plus the session string. ``data_ts`` is left out (stored in its own column; 6 digits would ruin it)."""
    snapshot: dict[str, Any] = {}
    for key in FEATURE_COLUMNS:
        if key == "data_ts":
            continue
        value = row.get(key)
        if key == "session":
            if isinstance(value, str):
                snapshot[key] = value
            continue
        number = float(bool(value)) if isinstance(value, (bool, np.bool_)) else feature_float(row, key)
        if math.isfinite(number):
            snapshot[key] = float(f"{number:.6g}")
    return snapshot


def candidate_from_hit(
    hit: SetupHit,
    row: Mapping[str, Any],
    *,
    instrument: Instrument,
    decision_ts: int,
    score: float,
    components: Mapping[str, float],
    news: NewsContext | None,
) -> Candidate:
    """Candidate for a setup hit on its feature row (``ref_price`` = the row's price, ``data_ts`` = the row's
    data time, falling back to ``decision_ts``)."""
    data_ts = feature_float(row, "data_ts")
    return Candidate(
        symbol=instrument.symbol,
        direction=hit.direction,
        setup=hit.setup,
        timeframe=hit.timeframe,
        intraday=hit.intraday,
        horizon_min=hit.horizon_min,
        decision_ts=int(decision_ts),
        data_ts=int(data_ts) if math.isfinite(data_ts) else int(decision_ts),
        ref_price=feature_float(row, "price"),
        atr=hit.atr,
        stop_dist=hit.stop_dist,
        target_dist=hit.target_dist,
        components=dict(components),
        score=float(score),
        features=_reduced_features(row),
        news_ids=list(news.news_ids) if news is not None else [],
    )


def make_signal(
    candidate: Candidate, *, confidence: float, expected_r: float, engine_version: str, status: SignalStatus = "review"
) -> Signal:
    """New (unsaved) signal with all candidate fields; ``created_ts`` is the decision time."""
    return Signal(
        created_ts=candidate.decision_ts,
        data_ts=candidate.data_ts,
        symbol=candidate.symbol,
        direction=candidate.direction,
        setup=candidate.setup,
        timeframe=candidate.timeframe,
        horizon_min=candidate.horizon_min,
        ref_price=candidate.ref_price,
        atr=candidate.atr,
        stop_dist=candidate.stop_dist,
        target_dist=candidate.target_dist,
        score=candidate.score,
        confidence=confidence,
        engine_version=engine_version,
        status=status,
        intraday=candidate.intraday,
        components=dict(candidate.components),
        features=dict(candidate.features),
        expected_r=expected_r,
        news_ids=list(candidate.news_ids),
    )


def entry_deadline(decision_ts: int, instrument: Instrument, params: EngineParams = ENGINE_PARAMS) -> int:
    """Latest open time of an entry bar: the feed delay plus ``entry_grace_min`` after ``decision_ts``."""
    return int(decision_ts) + (instrument.delay_min + params.entry_grace_min) * 60


def issue(signal: Signal, *, issued_ts: int, instrument: Instrument, params: EngineParams = ENGINE_PARAMS) -> None:
    """Finalise a reviewed signal: the entry is attempted on the first bar opening at or after ``issued_ts``.

    Raises ``ValueError`` for a signal that is already being tracked or closed.
    """
    if signal.status not in ("review", "pending_entry") or signal.entry_ts is not None:
        raise ValueError(f"signal in status {signal.status!r} cannot be issued")
    signal.issued_ts = int(issued_ts)
    signal.entry_deadline_ts = entry_deadline(issued_ts, instrument, params)
    signal.status = "pending_entry"


# ------------------------------------------------------------------------------------------------
# Tracking
# ------------------------------------------------------------------------------------------------
def _close(
    signal: Signal, entry_price: float, exit_: execution.Exit, exit_ts: int, instrument: Instrument, costs: CostParams
) -> None:
    """Record an exit: fill price with slippage, result in R, final excursions and status.

    Excursions are measured until the exit; the exit bar contributes only the quoted exit price.
    """
    signal.exit_ts = int(exit_ts)
    signal.exit_price = execution.exit_fill_price(costs, instrument, signal.direction, exit_)
    signal.outcome_r = execution.result_r(
        signal.direction, entry_price, signal.exit_price, signal.stop_dist, fee_bps=costs.fee_bps(instrument)
    )
    move = (exit_.price - entry_price) * signal.sign / signal.stop_dist if signal.stop_dist > 0 else 0.0
    signal.mfe_r = max(signal.mfe_r or 0.0, move, 0.0)
    signal.mae_r = max(signal.mae_r or 0.0, -move, 0.0)
    signal.status = _STATUS_BY_EXIT[exit_.reason]
    signal.status_reason = exit_.reason


def _enter(
    signal: Signal, bar_ts: int, bar_open: float, instrument: Instrument, calendar: MarketCalendar, costs: CostParams
) -> None:
    signal.entry_ts = bar_ts
    signal.entry_price = execution.entry_fill_price(costs, instrument, signal.direction, bar_open)
    signal.stop_price = signal.entry_price - signal.sign * signal.stop_dist
    signal.target_price = signal.entry_price + signal.sign * signal.target_dist
    signal.expires_ts = to_ts(calendar.add_trading_minutes(instrument.exchange, from_ts(bar_ts), signal.horizon_min))
    signal.status = "active"
    signal.mfe_r = 0.0
    signal.mae_r = 0.0


def _flatten_ts(signal: Signal, instrument: Instrument, calendar: MarketCalendar, params: EngineParams) -> int | None:
    if not signal.intraday or signal.entry_ts is None:
        return None
    return execution.session_flatten_ts(
        calendar, instrument.exchange, int(signal.entry_ts), params.flatten_before_close_min
    )


def advance(
    signal: Signal,
    bars: pd.DataFrame,
    *,
    instrument: Instrument,
    calendar: MarketCalendar,
    costs: CostParams,
    params: EngineParams = ENGINE_PARAMS,
) -> bool:
    """Move a signal through its life cycle with new closed 1-minute bars. Returns True if it changed.

    ``bars`` are canonical closed 1-minute bars, ascending; bars before ``issued_ts`` or at or before
    ``checked_ts`` are skipped, so the call is idempotent.

    * review: nothing happens (no entry before the review is finished).
    * pending_entry: the entry bar is the first bar with ts >= issued_ts. It cancels the signal when it
      opens after the entry deadline ('kein_einstieg') or when its open drifted more than
      ``max_entry_drift`` stop distances from ``ref_price`` ('kurs_entlaufen'); otherwise the signal is
      filled at its open plus slippage and the SAME bar is evaluated for an exit.
    * active: every bar is checked with ``execution.exit_on_bar`` (time stop, session flattening for
      intraday signals, stop, target); without an exit the running excursions are updated.
    """
    if signal.status in SIGNAL_CLOSED_STATUSES or signal.status == "review" or bars.empty:
        return False
    if signal.issued_ts is None:
        log.warning("signal %s is %s without issue time; not tracked", signal.id, signal.status)
        return False
    ts = bars.index.to_numpy(dtype=np.int64)
    # Never a bar before the issue (no entry on data older than the decision to trade), never one twice.
    lower = signal.issued_ts if signal.checked_ts is None else max(signal.issued_ts, signal.checked_ts + 1)
    keep = ts >= lower
    if not keep.any():
        return False
    ts = ts[keep]
    prices = bars[["open", "high", "low"]].to_numpy(dtype=np.float64)[keep]
    if signal.status == "pending_entry":
        bar_ts, bar_open = int(ts[0]), float(prices[0, 0])
        signal.checked_ts = bar_ts
        if signal.entry_deadline_ts is not None and bar_ts > signal.entry_deadline_ts:
            signal.status, signal.status_reason = "cancelled", REASON_NO_ENTRY
            return True
        if execution.drift_exceeded(signal.ref_price, bar_open, params.max_entry_drift * signal.stop_dist):
            signal.status, signal.status_reason = "cancelled", REASON_DRIFT
            return True
        _enter(signal, bar_ts, bar_open, instrument, calendar, costs)  # the entry bar is evaluated below
    entry_price = signal.entry_price
    if signal.status != "active" or entry_price is None:
        log.warning("signal %s in status %s cannot be tracked", signal.id, signal.status)
        return False
    flatten_ts = _flatten_ts(signal, instrument, calendar, params)
    for index in range(ts.size):
        bar_ts = int(ts[index])
        bar_open, bar_high, bar_low = (float(value) for value in prices[index])
        signal.checked_ts = bar_ts
        exit_ = execution.exit_on_bar(
            side=signal.direction,
            bar_ts=bar_ts,
            bar_open=bar_open,
            bar_high=bar_high,
            bar_low=bar_low,
            stop=signal.stop_price,
            target=signal.target_price,
            time_stop_ts=signal.expires_ts,
            flatten_ts=flatten_ts,
        )
        if exit_ is not None:
            _close(signal, entry_price, exit_, bar_ts, instrument, costs)
            return True
        favourable, adverse = execution.excursion_r(signal.direction, entry_price, bar_high, bar_low, signal.stop_dist)
        signal.mfe_r = max(signal.mfe_r or 0.0, favourable, 0.0)
        signal.mae_r = max(signal.mae_r or 0.0, adverse, 0.0)
    return True


def _ceil_minute(ts: int) -> int:
    """``ts`` rounded up to a full minute."""
    return -(-int(ts) // BAR_SECONDS) * BAR_SECONDS


def _unsettled_trading(
    calendar: MarketCalendar, instrument: Instrument, start_ts: int, end_ts: int, settle_horizon_ts: int | None
) -> bool:
    """True if a minute in ``[start_ts, end_ts)`` during which the venue traded may still receive a bar.

    A minute with open time ``ts`` is final once ``ts + 60 <= settle_horizon_ts`` (``BarReader.settle_horizon``);
    ``None`` means every minute up to now is final. Traded means inside a regular segment of the calendar; an
    exchange the calendar does not know counts as trading (wait rather than decide on missing data).
    """
    if settle_horizon_ts is None:
        return False
    first_unsettled = _ceil_minute(int(settle_horizon_ts) - BAR_SECONDS + 1)
    lo = max(_ceil_minute(start_ts), first_unsettled)
    if lo >= end_ts:
        return False
    try:
        return bool(calendar.segments(instrument.exchange, from_ts(lo), from_ts(int(end_ts))))
    except KeyError:
        return True


def expire_stale(
    signal: Signal,
    *,
    now_ts: int,
    instrument: Instrument,
    calendar: MarketCalendar,
    last_close: float | None,
    costs: CostParams,
    params: EngineParams = ENGINE_PARAMS,
    settle_horizon_ts: int | None = None,
) -> bool:
    """Close signals that no further bar will settle. Returns True if the signal changed.

    * pending_entry: more than the feed delay + 2 minutes past the entry deadline without an entry bar
      -> cancelled ('kein_einstieg').
    * active: the time stop (checked first, like ``exit_on_bar``) or the session flattening is overdue by
      more than the feed delay + 10 minutes and the venue is closed -> market exit at ``last_close`` (the
      entry price when unknown) with slippage, exit time ``now_ts`` -> expired.

    ``settle_horizon_ts`` (``BarReader.settle_horizon`` of the signal's symbol; the live engine passes an
    instant before every bar during the first settle time after a restart, when the downtime's bars are
    still being fetched) guards both rules against late data: as long as a traded minute that could still
    decide the outcome is not final - for a pending signal any minute from the issue (or the last processed
    bar) up to the entry deadline, for an active one any minute after the last processed bar - the late bar
    may still arrive (REST fills holes minutes after newer bars), so nothing is closed yet. ``None`` treats
    every minute up to ``now_ts`` as final.
    """
    if signal.status == "pending_entry":
        deadline = signal.entry_deadline_ts
        if deadline is None or now_ts <= deadline + instrument.delay_min * 60 + _PENDING_GRACE_S:
            return False
        first = deadline if signal.issued_ts is None else signal.issued_ts
        if signal.checked_ts is not None:
            first = max(first, signal.checked_ts + 1)
        if _unsettled_trading(calendar, instrument, first, deadline + BAR_SECONDS, settle_horizon_ts):
            return False
        signal.status, signal.status_reason = "cancelled", REASON_NO_ENTRY
        return True
    entry_price = signal.entry_price
    if signal.status != "active" or entry_price is None:
        return False
    grace = instrument.delay_min * 60 + _ACTIVE_GRACE_S
    due: list[tuple[int | None, execution.ExitReason]] = [
        (signal.expires_ts, "time_stop"),
        (_flatten_ts(signal, instrument, calendar, params), "session_close"),
    ]
    reason = next((name for when, name in due if when is not None and now_ts > when + grace), None)
    if reason is None or calendar.is_open(instrument.exchange, from_ts(now_ts)):
        return False
    last_seen = signal.checked_ts if signal.checked_ts is not None else signal.entry_ts
    if last_seen is not None and _unsettled_trading(
        calendar, instrument, int(last_seen) + 1, int(now_ts), settle_horizon_ts
    ):
        return False
    price = last_close if last_close is not None and math.isfinite(last_close) else entry_price
    _close(signal, entry_price, execution.Exit(price, reason, True), now_ts, instrument, costs)
    return True
