"""Settlement right after a (re)start: age alone settles nothing until the live service ran a full settle period."""

from trading_local import barstore
from trading_local.barstore import NOT_SETTLED_BY_AGE, SETTLE_S, BarReader
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.testing import FakeCalendar, make_bars, seed_bars, seed_instruments

T0 = 1_795_000_020 // 60 * 60  # a minute boundary
MIN = 60


def reader(db: Database, asof: int) -> BarReader:
    return BarReader(db, InstrumentBook(db), FakeCalendar(), asof)


def test_without_a_live_start_age_settles_as_before(db: Database) -> None:
    seed_instruments(db)
    asof = T0 + 3_600
    assert reader(db, asof).settle_horizon("BTC-USD") == asof - SETTLE_S


def test_stream_bars_from_before_a_restart_wait_for_rest_or_a_full_settle_period(db: Database) -> None:
    seed_instruments(db)
    # provisional stream bars stored before the shutdown, two hours ago
    seed_bars(db, "BTC-USD", make_bars(T0, 30), source="stream")
    restart = T0 + 2 * 3_600
    barstore.set_live_since(restart)

    early = reader(db, restart + 5 * MIN)
    assert early.settle_horizon("BTC-USD") == NOT_SETTLED_BY_AGE
    assert early.settled_bars_after("BTC-USD", None, start_ts=T0).empty  # nothing is final yet

    seed_bars(db, "BTC-USD", make_bars(T0, 10), source="rest")  # the reconciliation confirms the first minutes
    confirmed = reader(db, restart + 6 * MIN).settled_bars_after("BTC-USD", None, start_ts=T0)
    assert confirmed.index.tolist() == [T0 + k * MIN for k in range(10)]  # REST bars only, contiguous

    later = reader(db, restart + SETTLE_S)
    assert later.settle_horizon("BTC-USD") == restart  # a full settle period of live operation has passed
    assert len(later.settled_bars_after("BTC-USD", None, start_ts=T0)) == 30


def test_switching_the_rule_off_restores_age_settlement(db: Database) -> None:
    seed_instruments(db)
    barstore.set_live_since(T0)
    assert reader(db, T0 + MIN).settle_horizon("BTC-USD") == NOT_SETTLED_BY_AGE
    barstore.set_live_since(None)
    assert reader(db, T0 + MIN).settle_horizon("BTC-USD") == T0 + MIN - SETTLE_S
