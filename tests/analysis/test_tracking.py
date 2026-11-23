"""Signal life cycle on 1-minute bars: entries, exits, cancellations, stale expiry and bookkeeping."""

import copy
import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from trading_local import execution
from trading_local.analysis.features import NewsContext
from trading_local.analysis.params import ENGINE_PARAMS
from trading_local.analysis.setups import SetupHit
from trading_local.analysis.tracking import (
    REASON_DRIFT,
    REASON_NO_ENTRY,
    advance,
    candidate_from_hit,
    entry_deadline,
    expire_stale,
    issue,
    make_signal,
)
from trading_local.clock import to_ts
from trading_local.config import CostParams
from trading_local.models import Candidate, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
NVDA = INSTRUMENTS["NVDA"]  # XNAS 14:30-21:00 UTC in the fake calendar, no feed delay, 2 bps slippage
SAP = INSTRUMENTS["SAP.DE"]  # 15 minutes feed delay
CALENDAR = FakeCalendar()
COSTS = CostParams()
SLIP = 0.0002


def utc(day: int, hour: int, minute: int = 0) -> int:
    return to_ts(datetime(2026, 11, day, hour, minute, tzinfo=UTC))


def bars(*rows: tuple[int, float, float, float]) -> pd.DataFrame:
    """Canonical 1-minute bars from (ts, open, high, low); close = open."""
    return pd.DataFrame(
        {
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[1] for r in rows],
            "volume": [100.0] * len(rows),
        },
        index=pd.Index([r[0] for r in rows], dtype="int64", name="ts"),
    )


def candidate(direction: str = "LONG", *, horizon: int = 240, intraday: bool = True, decision: int = 0) -> Candidate:
    return Candidate(
        symbol="NVDA", direction=direction, setup="trend_pullback", timeframe="15m", intraday=intraday,  # type: ignore[arg-type]
        horizon_min=horizon, decision_ts=decision or utc(20, 15, 0), data_ts=decision or utc(20, 15, 0),
        ref_price=100.0, atr=1.0, stop_dist=1.0, target_dist=2.0, components={"trend": 0.7}, score=0.61,
        features={"price": 100.0}, news_ids=[3, 1],
    )  # fmt: skip


def issued(direction: str = "LONG", *, at: int | None = None, instrument=NVDA, **kwargs) -> Signal:  # noqa: ANN001
    signal = make_signal(candidate(direction, **kwargs), confidence=0.4, expected_r=0.1, engine_version="e1-test")
    issue(signal, issued_ts=at or utc(20, 15, 0), instrument=instrument)
    return signal


def run(signal: Signal, frame: pd.DataFrame, instrument=NVDA) -> bool:  # noqa: ANN001
    return advance(signal, frame, instrument=instrument, calendar=CALENDAR, costs=COSTS)


# ------------------------------------------------------------------------------------------------
# Creation
# ------------------------------------------------------------------------------------------------
def test_make_signal_copies_the_candidate() -> None:
    cand = candidate()
    signal = make_signal(cand, confidence=0.42, expected_r=0.15, engine_version="e1-abc")
    assert (signal.created_ts, signal.data_ts, signal.symbol, signal.direction) == (utc(20, 15), utc(20, 15), "NVDA", "LONG")
    assert (signal.setup, signal.timeframe, signal.horizon_min, signal.intraday) == ("trend_pullback", "15m", 240, True)
    assert (signal.ref_price, signal.atr, signal.stop_dist, signal.target_dist, signal.score) == (100.0, 1.0, 1.0, 2.0, 0.61)
    assert (signal.confidence, signal.expected_r, signal.engine_version, signal.status) == (0.42, 0.15, "e1-abc", "review")
    assert signal.components == {"trend": 0.7} and signal.features == {"price": 100.0} and signal.news_ids == [3, 1]
    assert signal.components is not cand.components and signal.news_ids is not cand.news_ids
    assert signal.entry_deadline_ts is None and signal.issued_ts is None
    assert make_signal(cand, confidence=0.4, expected_r=0.0, engine_version="x", status="pending_entry").status == (
        "pending_entry"
    )


def test_issue_sets_the_entry_window() -> None:
    signal = make_signal(candidate(), confidence=0.4, expected_r=0.1, engine_version="e1-test")
    issue(signal, issued_ts=utc(20, 15, 2), instrument=SAP)
    assert signal.status == "pending_entry" and signal.issued_ts == utc(20, 15, 2)
    assert signal.entry_deadline_ts == utc(20, 15, 2) + (15 + 10) * 60
    assert entry_deadline(utc(20, 15), NVDA, ENGINE_PARAMS) == utc(20, 15, 10)
    signal.status = "active"
    with pytest.raises(ValueError):
        issue(signal, issued_ts=utc(20, 16), instrument=NVDA)


def test_candidate_from_hit_reduces_the_features() -> None:
    hit = SetupHit("breakout", "SHORT", "15m", True, 180, 1.25, 1.5, 3.0)
    row = {
        "price": 123.456789123, "data_ts": utc(20, 15), "session": "2026-11-20", "in_session": True,
        "rsi_fast": 33.3333333, "vwap": math.nan, "atr_slow": 1.25, "or_high": None, "unknown_extra": 5.0,
    }  # fmt: skip
    news = NewsContext(0.3, 2, (9, 4), None)
    cand = candidate_from_hit(
        hit, row, instrument=NVDA, decision_ts=utc(20, 15), score=0.6, components={"trend": 0.5}, news=news
    )
    assert (cand.symbol, cand.direction, cand.setup, cand.timeframe, cand.intraday) == ("NVDA", "SHORT", "breakout", "15m", True)
    assert (cand.horizon_min, cand.atr, cand.stop_dist, cand.target_dist) == (180, 1.25, 1.5, 3.0)
    assert (cand.decision_ts, cand.data_ts, cand.ref_price, cand.score) == (utc(20, 15), utc(20, 15), 123.456789123, 0.6)
    assert cand.features == {"price": 123.457, "session": "2026-11-20", "in_session": 1.0, "rsi_fast": 33.3333,
                             "atr_slow": 1.25}
    assert cand.news_ids == [9, 4] and cand.components == {"trend": 0.5}
    no_news = candidate_from_hit(hit, row, instrument=NVDA, decision_ts=utc(20, 15), score=0.6, components={}, news=None)
    assert no_news.news_ids == []


# ------------------------------------------------------------------------------------------------
# Full life cycles
# ------------------------------------------------------------------------------------------------
def test_long_reaches_the_target() -> None:
    signal = issued()
    frame = bars(
        (utc(20, 15, 0), 100.0, 100.5, 99.8),
        (utc(20, 15, 1), 100.4, 101.0, 100.2),
        (utc(20, 15, 2), 101.0, 102.5, 100.9),
        (utc(20, 15, 3), 102.0, 103.0, 101.0),  # after the exit: ignored
    )
    assert run(signal, frame)
    entry = 100.0 * (1 + SLIP)
    assert signal.entry_ts == utc(20, 15, 0) and signal.entry_price == pytest.approx(entry)
    assert signal.stop_price == pytest.approx(entry - 1.0) and signal.target_price == pytest.approx(entry + 2.0)
    assert signal.expires_ts == utc(20, 19, 0)  # 240 trading minutes
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("target", "target", utc(20, 15, 2))
    assert signal.exit_price == pytest.approx(entry + 2.0)  # limit fill: no slippage
    assert signal.outcome_r == pytest.approx(2.0)
    assert signal.checked_ts == utc(20, 15, 2)
    assert signal.mfe_r == pytest.approx(2.0) and signal.mae_r == pytest.approx(entry - 99.8)
    assert not run(signal, frame)  # closed signals never change


def test_long_is_stopped_with_slippage() -> None:
    signal = issued()
    run(signal, bars((utc(20, 15, 0), 100.0, 100.3, 99.9), (utc(20, 15, 1), 99.9, 100.0, 98.5)))
    entry = 100.0 * (1 + SLIP)
    assert (signal.status, signal.status_reason) == ("stopped", "stop")
    assert signal.exit_price == pytest.approx((entry - 1.0) * (1 - SLIP))
    assert signal.outcome_r == pytest.approx(execution.result_r("LONG", entry, (entry - 1.0) * (1 - SLIP), 1.0))
    assert signal.outcome_r < -1.0
    assert signal.mae_r == pytest.approx(1.0) and signal.mfe_r == pytest.approx(0.3 - 100 * SLIP)


def test_both_touched_on_one_bar_counts_as_stop() -> None:
    signal = issued()
    run(signal, bars((utc(20, 15, 0), 100.0, 100.1, 99.9), (utc(20, 15, 1), 100.0, 103.0, 98.0)))
    assert signal.status == "stopped"


def test_gap_through_the_stop_exits_at_the_open() -> None:
    signal = issued()
    run(signal, bars((utc(20, 15, 0), 100.0, 100.1, 99.9), (utc(20, 15, 1), 98.5, 99.0, 98.0)))
    assert signal.status == "stopped"
    assert signal.exit_price == pytest.approx(98.5 * (1 - SLIP))
    assert signal.mae_r == pytest.approx(100.0 * (1 + SLIP) - 98.5)


def test_exit_on_the_entry_bar() -> None:
    signal = issued()
    run(signal, bars((utc(20, 15, 0), 100.0, 100.2, 98.0)))
    assert signal.status == "stopped" and signal.exit_ts == signal.entry_ts == utc(20, 15, 0)
    assert signal.checked_ts == utc(20, 15, 0)


def test_short_life_cycles_mirror_the_long_ones() -> None:
    target = issued("SHORT")
    run(target, bars((utc(20, 15, 0), 100.0, 100.2, 99.5), (utc(20, 15, 1), 99.0, 99.2, 97.0)))
    entry = 100.0 * (1 - SLIP)  # the short sells: fills lower
    assert target.entry_price == pytest.approx(entry)
    assert target.stop_price == pytest.approx(entry + 1.0) and target.target_price == pytest.approx(entry - 2.0)
    assert (target.status, target.exit_price, target.outcome_r) == ("target", pytest.approx(entry - 2.0), pytest.approx(2.0))
    stopped = issued("SHORT")
    run(stopped, bars((utc(20, 15, 0), 100.0, 100.2, 99.9), (utc(20, 15, 1), 100.5, 101.5, 100.4)))
    assert stopped.status == "stopped"
    assert stopped.exit_price == pytest.approx((entry + 1.0) * (1 + SLIP))  # buying back: fills higher
    assert stopped.outcome_r == pytest.approx((entry - (entry + 1.0) * (1 + SLIP)) / 1.0)
    gap = issued("SHORT")
    run(gap, bars((utc(20, 15, 0), 100.0, 100.2, 99.9), (utc(20, 15, 1), 101.6, 102.0, 101.5)))
    assert gap.status == "stopped" and gap.exit_price == pytest.approx(101.6 * (1 + SLIP))


def test_time_stop_uses_trading_minutes() -> None:
    signal = issued(horizon=30)
    frame = bars(*[(utc(20, 15, 0) + 60 * i, 100.0, 100.3, 99.7) for i in range(35)])
    run(signal, frame)
    assert signal.expires_ts == utc(20, 15, 30)
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "time_stop", utc(20, 15, 30))
    assert signal.exit_price == pytest.approx(100.0 * (1 - SLIP))
    assert signal.outcome_r == pytest.approx((100.0 * (1 - SLIP) - 100.0 * (1 + SLIP)) / 1.0)


def test_time_stop_of_a_swing_signal_skips_closed_hours() -> None:
    signal = issued(horizon=120, intraday=False, at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(signal, bars((utc(20, 20, 0), 100.0, 100.3, 99.7), (utc(20, 20, 59), 100.0, 100.3, 99.7)))
    assert signal.status == "active"
    assert signal.expires_ts == utc(23, 15, 30)  # 60 minutes on Friday, 60 on Monday
    run(signal, bars((utc(23, 14, 30), 100.1, 100.4, 99.8), (utc(23, 15, 30), 100.2, 100.3, 100.0)))
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "time_stop", utc(23, 15, 30))


def test_intraday_signal_is_flattened_before_the_close() -> None:
    signal = issued(at=utc(20, 20, 0), decision=utc(20, 20, 0))
    frame = bars(*[(utc(20, 20, 0) + 60 * i, 100.0, 100.3, 99.7) for i in range(60)])
    run(signal, frame)
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "session_close", utc(20, 20, 55))
    assert signal.checked_ts == utc(20, 20, 55)


@pytest.mark.parametrize(
    ("horizon", "expires", "reason", "exit_ts"),
    [
        (150, utc(20, 4, 30), "time_stop", utc(20, 4, 30)),  # 90 minutes before the break + 60 after it
        (400, utc(23, 2, 10), "session_close", utc(20, 6, 25)),  # 270 today + 130 Monday; flattened 06:25
    ],
)
def test_lunch_break_venue(horizon: int, expires: int, reason: str, exit_ts: int) -> None:
    toyota = INSTRUMENTS["7203.T"]
    signal = make_signal(candidate(decision=utc(20, 1, 0), horizon=horizon), confidence=0.4, expected_r=0.1,
                         engine_version="e1")
    issue(signal, issued_ts=utc(20, 1, 0), instrument=toyota)
    morning = [(utc(20, 1, 0) + 60 * i, 100.0, 100.3, 99.7) for i in range(90)]
    afternoon = [(utc(20, 3, 30) + 60 * i, 100.0, 100.3, 99.7) for i in range(180)]
    advance(signal, bars(*morning, *afternoon), instrument=toyota, calendar=CALENDAR, costs=COSTS,
            params=ENGINE_PARAMS)
    assert signal.expires_ts == expires
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", reason, exit_ts)


def test_drift_cancels_the_entry() -> None:
    signal = issued()
    assert run(signal, bars((utc(20, 15, 0), 100.6, 100.8, 100.5)))
    assert (signal.status, signal.status_reason) == ("cancelled", REASON_DRIFT)
    assert signal.entry_ts is None and signal.outcome_r is None and signal.checked_ts == utc(20, 15, 0)
    within = issued()
    run(within, bars((utc(20, 15, 0), 99.5, 99.8, 99.4)))  # exactly 0.5 stop distances: still allowed
    assert within.status == "active"


def test_entry_deadline_cancels() -> None:
    signal = issued()
    assert run(signal, bars((utc(20, 15, 11), 100.0, 100.1, 99.9)))
    assert (signal.status, signal.status_reason) == ("cancelled", REASON_NO_ENTRY)
    on_time = issued()
    run(on_time, bars((utc(20, 15, 10), 100.0, 100.1, 99.9)))
    assert on_time.status == "active"


def test_review_and_unissued_signals_are_not_tracked() -> None:
    review = make_signal(candidate(), confidence=0.4, expected_r=0.1, engine_version="e1")
    frame = bars((utc(20, 15, 0), 100.0, 100.1, 99.9))
    assert not run(review, frame) and review.status == "review" and review.checked_ts is None
    broken = make_signal(candidate(), confidence=0.4, expected_r=0.1, engine_version="e1", status="pending_entry")
    assert not run(broken, frame)


def test_bars_before_the_issue_are_ignored() -> None:
    signal = issued(at=utc(20, 15, 2))
    frame = bars(
        (utc(20, 15, 0), 90.0, 90.0, 90.0),
        (utc(20, 15, 1), 90.0, 90.0, 90.0),
        (utc(20, 15, 2), 100.0, 100.1, 99.9),
    )
    run(signal, frame)
    assert signal.entry_ts == utc(20, 15, 2) and signal.status == "active"
    assert not run(issued(at=utc(20, 15, 5)), frame)


def test_checked_ts_bookkeeping_and_idempotence() -> None:
    signal = issued()
    first = bars((utc(20, 15, 0), 100.0, 100.4, 99.8), (utc(20, 15, 1), 100.1, 100.6, 99.9))
    assert run(signal, first)
    assert signal.status == "active" and signal.checked_ts == utc(20, 15, 1)
    snapshot = copy.deepcopy(signal)
    assert not run(signal, first)  # the same bars again change nothing
    assert signal == snapshot
    overlapping = bars((utc(20, 15, 1), 50.0, 50.0, 50.0), (utc(20, 15, 2), 100.2, 100.9, 100.0))
    assert run(signal, overlapping)  # only the new bar is processed (the old one would hit the stop)
    assert signal.status == "active" and signal.checked_ts == utc(20, 15, 2)
    assert signal.mfe_r == pytest.approx(100.9 - 100.0 * (1 + SLIP))
    assert not run(signal, bars())


def test_excursions_are_running_maxima_and_never_negative() -> None:
    signal = issued()
    run(signal, bars((utc(20, 15, 0), 100.0, 100.0, 100.0)))
    assert signal.mfe_r == 0.0 and signal.mae_r == pytest.approx(100.0 * SLIP)
    run(signal, bars((utc(20, 15, 1), 100.5, 100.6, 100.5)))  # never below the entry
    run(signal, bars((utc(20, 15, 2), 100.3, 100.4, 100.3)))
    assert signal.mfe_r == pytest.approx(100.6 - 100.0 * (1 + SLIP))
    assert signal.mae_r == pytest.approx(100.0 * SLIP)


def test_a_stale_checked_ts_never_allows_an_entry_before_the_issue() -> None:
    signal = issued(at=utc(20, 15, 2))
    signal.checked_ts = utc(20, 14, 0)  # e.g. set by an older code path: must not open the door to older bars
    frame = bars(
        (utc(20, 15, 0), 90.0, 90.0, 90.0),
        (utc(20, 15, 1), 90.0, 90.0, 90.0),
        (utc(20, 15, 2), 100.0, 100.1, 99.9),
    )
    assert run(signal, frame)
    assert signal.entry_ts == utc(20, 15, 2) and signal.status == "active"


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_gap_through_the_target_is_a_limit_fill_at_the_better_open(direction: str) -> None:
    signal = issued(direction)
    gap_open = 103.0 if direction == "LONG" else 97.0
    run(signal, bars((utc(20, 15, 0), 100.0, 100.1, 99.9), (utc(20, 15, 1), gap_open, gap_open + 0.5, gap_open - 0.5)))
    assert (signal.status, signal.status_reason) == ("target", "target")
    assert signal.exit_price == gap_open  # better than the target, no slippage on a resting limit order
    entry = signal.entry_price or 0.0
    assert signal.outcome_r == pytest.approx(abs(gap_open - entry))
    assert signal.outcome_r > 2.0


def test_short_time_stop_buys_back_with_slippage() -> None:
    signal = issued("SHORT", horizon=30)
    run(signal, bars(*[(utc(20, 15, 0) + 60 * i, 100.0, 100.3, 99.7) for i in range(31)]))
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "time_stop", utc(20, 15, 30))
    assert signal.exit_price == pytest.approx(100.0 * (1 + SLIP))
    assert signal.outcome_r == pytest.approx(-(100.0 * (1 + SLIP) - 100.0 * (1 - SLIP)))


def test_intraday_crypto_signal_is_flattened_before_utc_midnight() -> None:
    btc = INSTRUMENTS["BTC-USD"]
    signal = make_signal(candidate(decision=utc(20, 23, 30)), confidence=0.4, expected_r=0.1, engine_version="e1")
    issue(signal, issued_ts=utc(20, 23, 30), instrument=btc)
    frame = bars(*[(utc(20, 23, 30) + 60 * i, 100.0, 100.3, 99.7) for i in range(40)])  # runs past midnight
    advance(signal, frame, instrument=btc, calendar=CALENDAR, costs=COSTS)
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "session_close", utc(20, 23, 55))
    assert signal.exit_price == pytest.approx(100.0 * (1 - 0.0005))  # crypto: 5 bps


def test_cancelled_signals_ignore_later_bars() -> None:
    signal = issued()
    frame = bars((utc(20, 15, 0), 100.6, 100.8, 100.5), (utc(20, 15, 1), 100.0, 100.1, 99.9))
    assert run(signal, frame) and signal.status == "cancelled"
    snapshot = copy.deepcopy(signal)
    assert not run(signal, frame)
    assert signal == snapshot


def test_chunked_bars_give_the_same_result_as_one_call() -> None:
    frame = bars(*[(utc(20, 15, 0) + 60 * i, 100.0 + 0.01 * i, 100.3 + 0.01 * i, 99.7 + 0.01 * i) for i in range(200)])
    whole, chunked = issued(), issued()
    run(whole, frame)
    for start in range(0, len(frame), 7):
        run(chunked, frame.iloc[max(0, start - 3) : start + 7])  # overlapping chunks
    assert whole == chunked
    assert whole.status == "target"


# ------------------------------------------------------------------------------------------------
# expire_stale
# ------------------------------------------------------------------------------------------------
def test_expire_stale_leaves_review_and_closed_signals_alone() -> None:
    review = make_signal(candidate(), confidence=0.4, expected_r=0.1, engine_version="e1")
    assert not stale(review, utc(23, 0, 0))
    closed = issued()
    run(closed, bars((utc(20, 15, 0), 100.0, 100.2, 98.0)))
    snapshot = copy.deepcopy(closed)
    assert not stale(closed, utc(23, 0, 0))
    assert closed == snapshot


def test_expire_stale_waits_for_the_overdue_exit_even_when_the_venue_is_closed() -> None:
    swing = issued(horizon=600, intraday=False, at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(swing, bars((utc(20, 20, 0), 100.0, 100.2, 99.9)))
    assert not stale(swing, utc(21, 12, 0))  # Saturday: closed, but the time stop lies on Monday
    assert swing.status == "active"
def stale(signal: Signal, now: int, last_close: float | None = 101.0, instrument=NVDA) -> bool:  # noqa: ANN001
    return expire_stale(
        signal, now_ts=now, instrument=instrument, calendar=CALENDAR, last_close=last_close, costs=COSTS
    )


def test_expire_stale_cancels_a_pending_signal_without_entry_bar() -> None:
    signal = issued()  # deadline 15:10
    assert not stale(signal, utc(20, 15, 12))
    assert stale(signal, utc(20, 15, 12) + 1)
    assert (signal.status, signal.status_reason) == ("cancelled", REASON_NO_ENTRY)
    delayed = issued(instrument=SAP)  # deadline 15:25, plus 15 minutes feed delay
    assert not stale(delayed, utc(20, 15, 42), instrument=SAP)
    assert stale(delayed, utc(20, 15, 42) + 1, instrument=SAP)


def test_expire_stale_closes_an_overdue_active_signal_when_the_venue_is_closed() -> None:
    signal = issued(at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(signal, bars((utc(20, 20, 0), 100.0, 100.2, 99.9)))
    assert signal.status == "active"
    assert not stale(signal, utc(20, 21, 5))  # flatten 20:55 + 10 minutes grace not yet over
    assert signal.status == "active"
    assert stale(signal, utc(20, 21, 5) + 1)
    assert signal.status == "expired"


def test_expire_stale_exit_values() -> None:
    signal = issued(at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(signal, bars((utc(20, 20, 0), 100.0, 100.2, 99.9)))
    now = utc(20, 22, 0)
    assert stale(signal, now, last_close=100.8)
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "session_close", now)
    assert signal.exit_price == pytest.approx(100.8 * (1 - SLIP))
    assert signal.outcome_r == pytest.approx(100.8 * (1 - SLIP) - 100.0 * (1 + SLIP))
    assert not stale(signal, now + 3600)  # closed signals stay closed


def test_expire_stale_does_nothing_while_the_venue_is_open() -> None:
    signal = issued(horizon=30)
    run(signal, bars((utc(20, 15, 0), 100.0, 100.2, 99.9)))
    assert not stale(signal, utc(20, 16, 30))  # time stop 15:30 overdue, but the market is open


def settled_stale(signal: Signal, now: int, *, settle_s: int = 600, instrument=NVDA) -> bool:  # noqa: ANN001
    """``expire_stale`` with the settle horizon a live ``BarReader`` gives at ``now`` (real-time feed)."""
    return expire_stale(
        signal, now_ts=now, instrument=instrument, calendar=CALENDAR, last_close=101.0, costs=COSTS,
        settle_horizon_ts=now - instrument.delay_min * 60 - settle_s,
    )  # fmt: skip


def test_expire_stale_keeps_a_pending_signal_while_its_entry_window_may_still_get_bars() -> None:
    """RUN-1: after a restart the minutes of the entry window are missing and REST fills them minutes later.
    The wall clock alone would cancel the signal although its entry bar may still arrive."""
    signal = issued()  # issued 15:00, entry deadline 15:10
    assert stale(copy.deepcopy(signal), utc(20, 15, 12) + 1)  # the old wall-clock rule cancels here
    assert not settled_stale(signal, utc(20, 15, 12) + 1)  # 15:02..15:10 traded and not final yet
    assert not settled_stale(signal, utc(20, 15, 21) - 1)  # the deadline minute 15:10 is final at 15:21
    assert signal.status == "pending_entry"
    assert settled_stale(signal, utc(20, 15, 21))
    assert (signal.status, signal.status_reason) == ("cancelled", REASON_NO_ENTRY)


def test_expire_stale_does_not_wait_for_minutes_in_which_the_venue_was_closed() -> None:
    signal = issued(at=utc(20, 20, 55), decision=utc(20, 20, 55))  # deadline 21:05, the close is 21:00
    assert not settled_stale(signal, utc(20, 21, 10) - 1)  # 20:59 is not final yet
    assert settled_stale(signal, utc(20, 21, 10))  # 21:00-21:05 never trade: no need to wait for them
    assert signal.status == "cancelled"


def test_expire_stale_waits_for_late_bars_before_the_market_exit_of_an_active_signal() -> None:
    signal = issued(at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(signal, bars((utc(20, 20, 0), 100.0, 100.2, 99.9)))  # flatten 20:55, checked 20:20
    assert not settled_stale(signal, utc(20, 21, 5) + 1)  # the wall-clock rule alone would exit here
    assert not settled_stale(signal, utc(20, 21, 10) - 1)  # 20:59 may still bring the exit bar
    assert signal.status == "active"
    assert settled_stale(signal, utc(20, 21, 10))
    assert (signal.status, signal.status_reason, signal.exit_ts) == ("expired", "session_close", utc(20, 21, 10))


def test_expire_stale_prefers_the_time_stop_and_falls_back_to_the_entry_price() -> None:
    signal = issued(horizon=30, at=utc(20, 20, 0), decision=utc(20, 20, 0))
    run(signal, bars((utc(20, 20, 0), 100.0, 100.2, 99.9)))
    assert stale(signal, utc(20, 22, 0), last_close=None)
    assert signal.status_reason == "time_stop"
    assert signal.exit_price == pytest.approx(signal.entry_price * (1 - SLIP))
