"""Fills and exits on settled bars only: late gap bars after a restart, provisional stream bars that REST
replaces, wall-clock rules that must not run ahead of unsettled bars, and the entry deadline shared with the
signal journal."""

import dataclasses
from collections.abc import Callable

import pytest

from trading_local.analysis import tracking
from trading_local.barstore import BarReader
from trading_local.clock import ManualClock
from trading_local.config import CostParams
from trading_local.contracts import Broker
from trading_local.db import repo_portfolio
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.dto import OrderRequest
from trading_local.models import Signal
from trading_local.testing import FakeCalendar
from tests.sim.simkit import (
    MIN,
    SETTLE_S,
    SLIP_EU,
    SLIP_US,
    T0,
    FakeMarketData,
    add_bar,
    check_accounting,
    make_signal,
    open_position,
)


def run(broker: Broker, clock: ManualClock, ts: int) -> list:
    clock.set(ts)
    return broker.process(clock.now())


def stored_checked_ts(db: Database) -> int | None:
    (position,) = repo_portfolio.open_positions(db)
    return position.checked_ts


def order_by_id(broker: Broker, order_id: int, key: str = "me"):
    return next(o for o in broker.orders(key) if o.id == order_id)


# -- protective exits -----------------------------------------------------------------------------------
def test_a_stop_hit_inside_a_gap_filled_after_a_restart_is_executed_at_the_gap_bar(
    make_broker: Callable[[], Broker], db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    broker = make_broker()
    position = open_position(broker, db, data, clock, "NVDA", qty=10, price=100.0, stop=98.0)
    fill_ts = position.opened_ts
    add_bar(db, "NVDA", fill_ts + MIN, 100.0)
    run(broker, clock, fill_ts + 2 * MIN)
    assert stored_checked_ts(db) == fill_ts + MIN

    # The process is down from +2 to +10 minutes. After the restart the first minute cycle stores the stream
    # bar of the current minute before the backfill has written the missed minutes.
    restarted = make_broker()
    add_bar(db, "NVDA", fill_ts + 10 * MIN, 99.5, source="stream")
    assert run(restarted, clock, fill_ts + 11 * MIN) == []
    assert stored_checked_ts(db) == fill_ts + MIN  # waits at the hole instead of skipping it
    # A cycle later the backfill delivers the downtime: at +5 minutes NVDA traded through the stop.
    for minute in range(2, 10):
        add_bar(db, "NVDA", fill_ts + minute * MIN, 99.0, low=97.0 if minute == 5 else None)
    closed = run(restarted, clock, fill_ts + 12 * MIN)

    assert [(p.exit_reason, p.closed_ts) for p in closed] == [("stop", fill_ts + 5 * MIN)]
    assert closed[0].exit_price == pytest.approx(98.0 * (1 - SLIP_US))
    check_accounting(restarted, "me")


def test_a_provisional_bar_is_judged_by_the_rest_bar_that_replaces_it(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position = open_position(broker, db, data, clock, "NVDA", qty=10, price=100.0, stop=98.0)
    t = position.opened_ts
    add_bar(db, "NVDA", t + MIN, 100.0)
    run(broker, clock, t + 2 * MIN)

    # The stream bar of +2 minutes dips through the stop; the authoritative REST bar does not.
    add_bar(db, "NVDA", t + 2 * MIN, 99.0, low=97.5, source="stream")
    assert run(broker, clock, t + 3 * MIN) == []
    assert stored_checked_ts(db) == t + MIN  # not examined before it is final
    add_bar(db, "NVDA", t + 2 * MIN, 99.0, low=98.5)
    add_bar(db, "NVDA", t + 3 * MIN, 99.0, low=98.6)
    assert run(broker, clock, t + 4 * MIN) == []
    assert stored_checked_ts(db) == t + 3 * MIN

    # The other way round: a harmless stream bar whose REST replacement touches the stop.
    add_bar(db, "NVDA", t + 4 * MIN, 99.0, low=98.8, source="stream")
    assert run(broker, clock, t + 5 * MIN) == []
    add_bar(db, "NVDA", t + 4 * MIN, 98.9, low=97.9)
    closed = run(broker, clock, t + 6 * MIN)
    assert [(p.exit_reason, p.closed_ts) for p in closed] == [("stop", t + 4 * MIN)]
    assert closed[0].exit_price == pytest.approx(98.0 * (1 - SLIP_US))


# -- opening and closing fills ----------------------------------------------------------------------------
def test_an_opening_order_fills_on_the_rest_values_of_its_bar(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("NVDA", 100.0, T0)
    clock.set(T0 + 30)
    order = broker.place_order("me", OrderRequest(symbol="NVDA", side="LONG", qty=10))
    add_bar(db, "NVDA", T0 + MIN, 105.0, source="stream")  # provisional: its open is still wrong
    run(broker, clock, T0 + 2 * MIN)
    assert order_by_id(broker, order.id).status == "pending"
    add_bar(db, "NVDA", T0 + MIN, 101.0)  # REST replaces it
    run(broker, clock, T0 + 3 * MIN)
    filled = order_by_id(broker, order.id)
    assert (filled.status, filled.filled_ts) == ("filled", T0 + MIN)
    assert filled.fill_price == pytest.approx(101.0 * (1 + SLIP_US))


def test_a_provisional_bar_that_is_never_replaced_settles_by_age_and_keeps_its_time(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("NVDA", 100.0, T0)
    order = broker.place_order("me", OrderRequest(symbol="NVDA", side="LONG", qty=10))
    add_bar(db, "NVDA", T0, 100.5, source="stream")
    settled = T0 + MIN + SETTLE_S  # NVDA has no feed delay
    for ts in range(T0 + MIN, settled, MIN):
        run(broker, clock, ts)
        assert order_by_id(broker, order.id).status == "pending"
    run(broker, clock, settled)
    filled = order_by_id(broker, order.id)
    # booked at the bar's own time and open: deciding later changes nothing about the result
    assert (filled.status, filled.filled_ts) == ("filled", T0)
    assert filled.fill_price == pytest.approx(100.5 * (1 + SLIP_US))


def test_an_opening_order_waits_for_a_missing_minute_before_its_bar(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    data.set_quote("NVDA", 100.0, T0)
    clock.set(T0 + 30)
    order = broker.place_order("me", OrderRequest(symbol="NVDA", side="LONG", qty=10))
    add_bar(db, "NVDA", T0 + 3 * MIN, 102.0)  # the minutes +1 and +2 are missing while Nasdaq trades
    run(broker, clock, T0 + 4 * MIN)
    assert order_by_id(broker, order.id).status == "pending"
    add_bar(db, "NVDA", T0 + MIN, 100.7)  # REST inserts the real fill bar behind the newer one
    add_bar(db, "NVDA", T0 + 2 * MIN, 101.0)
    run(broker, clock, T0 + 5 * MIN)
    filled = order_by_id(broker, order.id)
    assert filled.filled_ts == T0 + MIN
    assert filled.fill_price == pytest.approx(100.7 * (1 + SLIP_US))


def test_a_closing_order_waits_until_the_bars_before_it_are_examined(
    broker: Broker, db: Database, data: FakeMarketData, clock: ManualClock
) -> None:
    position = open_position(broker, db, data, clock, "NVDA", qty=10, price=100.0, stop=98.0)
    t = position.opened_ts
    add_bar(db, "NVDA", t + MIN, 100.0)
    run(broker, clock, t + 2 * MIN)
    clock.set(t + 5 * MIN + 30)
    close_order = broker.close_position("me", position.id)
    add_bar(db, "NVDA", t + 6 * MIN, 99.5)  # the close's fill bar is final, but +2 .. +5 are still missing
    assert run(broker, clock, t + 7 * MIN) == []
    assert order_by_id(broker, close_order.id).status == "pending"
    for minute in range(2, 6):  # the backfill: at +4 minutes the stop was hit
        add_bar(db, "NVDA", t + minute * MIN, 99.0, low=97.0 if minute == 4 else None)
    closed = run(broker, clock, t + 8 * MIN)
    assert [(p.exit_reason, p.closed_ts) for p in closed] == [("stop", t + 4 * MIN)]
    out = order_by_id(broker, close_order.id)
    assert (out.status, out.status_reason) == ("cancelled", "Position bereits geschlossen")
    check_accounting(broker, "me")


# -- expiry and the entry deadline of the signal journal --------------------------------------------------
def issued_signal(db: Database, instruments: InstrumentBook, symbol: str, issued_ts: int, **fields: object) -> Signal:
    instrument = instruments.get(symbol)
    assert instrument is not None
    signal = make_signal(db, symbol, created_ts=issued_ts, **fields)  # type: ignore[arg-type]
    tracking.issue(signal, issued_ts=issued_ts, instrument=instrument)
    return signal


def journal_outcome(
    signal: Signal, db: Database, instruments: InstrumentBook, calendar: FakeCalendar, asof_ts: int
) -> Signal:
    """The tracked signal after all final bars (the bar-time rules of the signal journal)."""
    journal = dataclasses.replace(signal)
    instrument = instruments.get(signal.symbol)
    assert instrument is not None and journal.issued_ts is not None
    reader = BarReader(db, instruments, calendar, asof_ts)
    bars = reader.settled_bars_after(signal.symbol, None, start_ts=journal.issued_ts)
    tracking.advance(journal, bars, instrument=instrument, calendar=calendar, costs=CostParams())
    return journal


def replay_feed(
    broker: Broker, db: Database, clock: ManualClock, symbol: str, bar_times: list[int], lag_s: int, until_ts: int
) -> None:
    """Minute cycles from the clock time to ``until_ts``; each bar is stored ``lag_s`` after it closed."""
    written: set[int] = set()
    for now in range(clock.ts() + MIN, until_ts + 1, MIN):
        for ts in bar_times:
            if ts not in written and ts + MIN + lag_s <= now:
                add_bar(db, symbol, ts, 100.0, source="stream")
                written.add(ts)
        run(broker, clock, now)


def test_a_signal_order_does_not_expire_while_its_fill_bar_is_unsettled(
    broker: Broker, db: Database, clock: ManualClock, instruments: InstrumentBook, calendar: FakeCalendar
) -> None:
    signal = issued_signal(db, instruments, "SAP.DE", T0)
    out = broker.execute_signal(signal)
    deadline = T0 + 25 * MIN  # 15 minutes feed delay + 10 minutes entry grace
    assert out is not None and out.expire_ts == signal.entry_deadline_ts == deadline
    # After a halt the first bar opens exactly at the deadline. It arrives 15 minutes late as a provisional
    # stream bar and is final only SETTLE_S later - long after deadline + delay.
    replay_feed(broker, db, clock, "SAP.DE", [deadline], 15 * MIN, T0 + 60 * MIN)
    filled = order_by_id(broker, out.id, "ai")
    assert (filled.status, filled.filled_ts) == ("filled", deadline)
    assert filled.fill_price == pytest.approx(100.0 * (1 + SLIP_EU))
    journal = journal_outcome(signal, db, instruments, calendar, clock.ts())
    assert (journal.status, journal.entry_ts) == ("active", deadline)


def test_the_ai_order_expires_when_the_journal_finds_no_entry(
    broker: Broker, db: Database, clock: ManualClock, instruments: InstrumentBook, calendar: FakeCalendar
) -> None:
    # NVDA (no feed delay): the first bar after the issue opens 12 minutes later, after the entry deadline of
    # 10 minutes. The journal cancels the signal ('kein_einstieg'); the AI portfolio must not enter either.
    signal = issued_signal(db, instruments, "NVDA", T0, stop_dist=1.0, target_dist=2.0, intraday=True)
    out = broker.execute_signal(signal)
    assert out is not None and out.expire_ts == T0 + 10 * MIN
    first = T0 + 12 * MIN
    replay_feed(broker, db, clock, "NVDA", [first + k * MIN for k in range(20)], 0, T0 + 35 * MIN)
    expired = order_by_id(broker, out.id, "ai")
    assert (expired.status, expired.status_reason) == ("cancelled", "abgelaufen")
    assert broker.detail("ai").positions == [] and broker.trades("ai") == []
    journal = journal_outcome(signal, db, instruments, calendar, clock.ts())
    assert (journal.status, journal.status_reason) == ("cancelled", tracking.REASON_NO_ENTRY)


def test_the_ai_position_counts_its_holding_period_from_the_entry_bar_like_the_journal(
    broker: Broker, db: Database, clock: ManualClock, instruments: InstrumentBook, calendar: FakeCalendar
) -> None:
    # SAP (15 minutes delay), holding period 20 minutes. After a halt the first bar opens 18 minutes after the
    # issue - inside the entry deadline. The journal counts the 20 minutes from that entry bar (time stop at
    # +38 minutes); a time stop counted from the issue (+20 minutes) would end the AI trade 2 bars after its
    # entry while the journal keeps the trade open.
    signal = issued_signal(db, instruments, "SAP.DE", T0, horizon_min=20)
    out = broker.execute_signal(signal)
    assert out is not None
    first = T0 + 18 * MIN
    replay_feed(broker, db, clock, "SAP.DE", [first + k * MIN for k in range(30)], 15 * MIN, T0 + 80 * MIN)
    (trade,) = broker.trades("ai")
    journal = journal_outcome(signal, db, instruments, calendar, clock.ts())
    assert (journal.entry_ts, journal.expires_ts) == (first, first + 20 * MIN)
    assert (trade.opened_ts, trade.time_stop_ts) == (journal.entry_ts, journal.expires_ts)
    assert (journal.status_reason, journal.exit_ts) == ("time_stop", first + 20 * MIN)
    assert (trade.exit_reason, trade.closed_ts) == (journal.status_reason, journal.exit_ts)
    assert trade.exit_price == pytest.approx(journal.exit_price)


def test_the_ai_order_enters_when_the_journal_enters_on_a_delayed_feed(
    broker: Broker, db: Database, clock: ManualClock, instruments: InstrumentBook, calendar: FakeCalendar
) -> None:
    # SAP (15 minutes delay): the first bar opens 18 minutes after the issue - inside the entry deadline of
    # 25 minutes - and arrives 16 minutes after it opened. The old wall-clock expiry (created + 30 minutes) had
    # cancelled the order before the bar arrived while the journal entered.
    signal = issued_signal(db, instruments, "SAP.DE", T0, horizon_min=600)
    out = broker.execute_signal(signal)
    assert out is not None
    first = T0 + 18 * MIN
    replay_feed(broker, db, clock, "SAP.DE", [first + k * MIN for k in range(30)], 15 * MIN, T0 + 70 * MIN)
    filled = order_by_id(broker, out.id, "ai")
    assert (filled.status, filled.filled_ts) == ("filled", first)
    journal = journal_outcome(signal, db, instruments, calendar, clock.ts())
    assert (journal.status, journal.entry_ts) == ("active", first)
