"""``BarReader.settled_bars_after`` / ``first_settled_bar_at_or_after``: bars for consumers that act once.

Live, a bar is not final when it first appears: stream/quote bars are provisional until the REST
reconciliation replaces them, and minutes the stream missed (or a downtime) are inserted behind newer
bars. The paper broker and the signal tracker move a high-water mark past every bar they process, so they
must only see bars that can no longer change and that no missing traded minute precedes (RUN-1, CAUS-1).

FakeCalendar (UTC): XNAS 14:30-21:00, XETR 08:00-16:30, XTKS 00:00-02:30 + 03:30-06:30 on weekdays;
CRYPTO always open. Feed delays: NVDA 0, SAP.DE 15, 7203.T 20, BTC-USD 0 minutes.
"""

import random
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from trading_local.barstore import SETTLE_S, BarReader
from trading_local.clock import to_ts
from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.db.repo_market import BAR_COLUMNS
from trading_local.models import Bar, BarSource
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, seed_instruments

INSTRUMENTS = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
MIN = 60


def utc(day: int, hour: int, minute: int = 0, second: int = 0) -> int:
    """Epoch seconds of a time in November 2026 (the 20th is a Friday)."""
    return to_ts(datetime(2026, 11, day, hour, minute, second, tzinfo=UTC))


def bar(symbol: str, ts: int, price: float, source: BarSource = "rest", *, low: float | None = None) -> Bar:
    return Bar(symbol, ts, price, price + 0.5, price - 0.5 if low is None else low, price + 0.1, 10.0, source)


def put(db: Database, *bars: Bar) -> None:
    assert repo_market.upsert_bar_rows(db, bars) == len(bars)


def reader(db: Database, asof: int, **kwargs: bool) -> BarReader:
    return BarReader(db, INSTRUMENTS, FakeCalendar(), asof, **kwargs)


def settled(db: Database, symbol: str, asof: int, ts: int | None, **kwargs: int) -> list[int]:
    return reader(db, asof).settled_bars_after(symbol, ts, **kwargs).index.tolist()


def minutes(start: int, count: int) -> list[int]:
    return [start + n * MIN for n in range(count)]


@pytest.fixture(autouse=True)
def instruments(db: Database) -> None:
    seed_instruments(db)


# ------------------------------------------------------------------------------------------------
# Provisional bars (CAUS-1)
# ------------------------------------------------------------------------------------------------
def test_a_stream_bar_is_handed_out_only_once_rest_replaced_it(db: Database) -> None:
    put(db, *(bar("NVDA", ts, 100.0) for ts in minutes(utc(20, 15), 5)))
    put(db, bar("NVDA", utc(20, 15, 5), 100.0, "stream", low=97.0))  # provisional extreme
    asof = utc(20, 15, 6, 3)

    assert reader(db, asof).bars_after("NVDA", utc(20, 15, 4)).index.tolist() == [utc(20, 15, 5)]
    assert settled(db, "NVDA", asof, utc(20, 15, 4)) == []  # may still be corrected by REST

    put(db, bar("NVDA", utc(20, 15, 5), 100.2))  # the authoritative bar has no such low
    frame = reader(db, asof + 90).settled_bars_after("NVDA", utc(20, 15, 4))

    assert frame.index.tolist() == [utc(20, 15, 5)]
    assert frame.loc[utc(20, 15, 5), "low"] == pytest.approx(99.7)  # the final values, never the provisional
    assert list(frame.columns) == list(BAR_COLUMNS) and frame.index.name == "ts"


@pytest.mark.parametrize("source", ["stream", "quote"])
def test_a_provisional_bar_settles_by_age_counted_from_its_arrival(db: Database, source: BarSource) -> None:
    put(db, bar("BTC-USD", utc(20, 10), 50.0), bar("BTC-USD", utc(20, 10, 1), 50.0, source))
    settles_at = utc(20, 10, 2) + SETTLE_S  # bar end (= arrival on a real-time feed) + 10 minutes

    assert settled(db, "BTC-USD", settles_at - 1, utc(20, 10)) == []
    assert settled(db, "BTC-USD", settles_at, utc(20, 10)) == [utc(20, 10, 1)]
    assert reader(db, settles_at - 1).settle_horizon("BTC-USD") == utc(20, 10, 2) - 1


def test_a_delayed_venue_settles_its_stream_bars_only_after_the_delay(db: Database) -> None:
    put(db, bar("SAP.DE", utc(20, 9), 80.0), bar("SAP.DE", utc(20, 9, 1), 80.0, "stream"))
    # The 15-minute delayed feed delivers the 09:01 bar at 09:17; it settles ten minutes later.
    settles_at = utc(20, 9, 2) + 15 * MIN + SETTLE_S

    assert settled(db, "SAP.DE", utc(20, 9, 17), utc(20, 9)) == []
    assert settled(db, "SAP.DE", settles_at - 1, utc(20, 9)) == []
    assert settled(db, "SAP.DE", settles_at, utc(20, 9)) == [utc(20, 9, 1)]
    assert reader(db, settles_at).settle_horizon("SAP.DE") == settles_at - 15 * MIN - SETTLE_S


def test_the_horizon_is_counted_from_arrival_also_in_a_replay(db: Database) -> None:
    asof = utc(20, 12)
    live = reader(db, asof)
    replay = reader(db, asof, apply_feed_delay=True)

    assert live.settle_horizon("SAP.DE") == live.cutoff("SAP.DE") - 15 * MIN - SETTLE_S
    assert replay.settle_horizon("SAP.DE") == replay.cutoff("SAP.DE") - SETTLE_S  # the delay counts once
    assert live.settle_horizon("SAP.DE") == replay.settle_horizon("SAP.DE")
    assert live.settle_horizon("UNKNOWN") == asof - SETTLE_S
    assert live.settle_horizon("NVDA", settle_s=0) == asof
    assert live.settle_horizon("7203.T", settle_s=120) == asof - 20 * MIN - 120


def test_settle_s_overrides_the_default_and_must_not_be_negative(db: Database) -> None:
    put(db, bar("BTC-USD", utc(20, 10), 50.0), bar("BTC-USD", utc(20, 10, 1), 50.0, "stream"))
    asof = utc(20, 10, 3)

    assert settled(db, "BTC-USD", asof, utc(20, 10)) == []
    assert settled(db, "BTC-USD", asof, utc(20, 10), settle_s=60) == [utc(20, 10, 1)]
    assert reader(db, asof).first_settled_bar_at_or_after("BTC-USD", utc(20, 10, 1), settle_s=0) is not None
    with pytest.raises(ValueError, match="settle_s"):
        reader(db, asof).settled_bars_after("BTC-USD", None, settle_s=-1)
    with pytest.raises(ValueError, match="settle_s"):
        reader(db, asof).first_settled_bar_at_or_after("BTC-USD", utc(20, 10), settle_s=-60)


def test_the_walk_stops_before_the_first_unsettled_bar_even_if_rest_bars_follow(db: Database) -> None:
    put(
        db,
        bar("NVDA", utc(20, 15), 100.0),
        bar("NVDA", utc(20, 15, 1), 100.0, "stream"),
        bar("NVDA", utc(20, 15, 2), 100.0),
        bar("NVDA", utc(20, 15, 3), 100.0),
    )

    assert settled(db, "NVDA", utc(20, 15, 5), None, start_ts=utc(20, 15)) == [utc(20, 15)]
    assert settled(db, "NVDA", utc(20, 15, 5), utc(20, 15, 1)) == [utc(20, 15, 2), utc(20, 15, 3)]


# ------------------------------------------------------------------------------------------------
# Holes (RUN-1)
# ------------------------------------------------------------------------------------------------
def restart_scenario(db: Database, symbol: str, *, use_settled: bool) -> list[tuple[int, float]]:
    """The restart scenario of RUN-1, seen by a consumer with a high-water mark (like the broker's
    protective-exit check): the first live bar after a downtime is written before the backfill fills the
    downtime, and during the downtime the price went through the stop (low 97). Returns (ts, low) of
    every bar the consumer processed, in order."""
    fill = utc(20, 15)
    processed: list[tuple[int, float]] = []
    checked: int | None = None

    def cycle(asof: int) -> None:
        nonlocal checked
        view = reader(db, asof)
        if use_settled:
            frame = view.settled_bars_after(symbol, checked, start_ts=fill)
        else:
            frame = view.bars_after(symbol, checked, start_ts=fill)
        processed.extend(zip(frame.index.tolist(), frame["low"].tolist(), strict=True))
        if not frame.empty:
            checked = int(frame.index[-1])

    put(db, bar(symbol, fill, 100.0), bar(symbol, fill + MIN, 100.0))
    cycle(fill + 2 * MIN + 3)
    # down from fill + 2 min to fill + 10 min: the first cycle after the restart has the current minute only
    put(db, bar(symbol, fill + 10 * MIN, 99.5, "stream"))
    cycle(fill + 11 * MIN + 3)
    put(db, bar(symbol, fill + 10 * MIN, 99.5))  # REST confirms it before the backfill reaches the downtime
    cycle(fill + 11 * MIN + 30)
    put(db, *(bar(symbol, fill + k * MIN, 99.0, low=97.0 if k == 5 else None) for k in range(2, 10)))
    for k in range(12, 30):
        cycle(fill + k * MIN + 3)
    return processed


def test_minutes_inserted_behind_a_newer_bar_are_not_skipped(db: Database) -> None:
    fill = utc(20, 15)

    processed = restart_scenario(db, "NVDA", use_settled=True)

    assert [ts for ts, _ in processed] == minutes(fill, 11)  # every minute exactly once, in order
    assert (fill + 5 * MIN, 97.0) in processed  # the bar that went through the stop is examined


def test_the_old_door_skips_minutes_inserted_behind_a_newer_bar(db: Database) -> None:
    """Counter-check of the scenario above with ``bars_after`` (the reason for the settled reads)."""
    processed = restart_scenario(db, "^GSPC", use_settled=False)

    assert [ts for ts, _ in processed] == [utc(20, 15), utc(20, 15, 1), utc(20, 15, 10)]
    assert all(low > 97.0 for _, low in processed)


def test_a_young_hole_blocks_until_its_minutes_are_settled_by_age(db: Database) -> None:
    """Illiquid symbol: the missing minutes never arrive, the consumer waits for the horizon only."""
    put(db, bar("BTC-USD", utc(20, 10), 50.0), bar("BTC-USD", utc(20, 10, 5), 50.0))
    accepted_at = utc(20, 10, 5) + SETTLE_S  # the newest missing minute 10:04 settles at 10:05 + 10 min

    assert settled(db, "BTC-USD", utc(20, 10, 6), utc(20, 10)) == []
    assert settled(db, "BTC-USD", accepted_at - 1, utc(20, 10)) == []
    assert settled(db, "BTC-USD", accepted_at, utc(20, 10)) == [utc(20, 10, 5)]
    # the hole is part of the interval only when the reference lies before it
    assert settled(db, "BTC-USD", utc(20, 10, 6), utc(20, 10, 4)) == [utc(20, 10, 5)]


def test_a_hole_between_returned_bars_cuts_the_result(db: Database) -> None:
    put(db, *(bar("BTC-USD", ts, 50.0) for ts in [*minutes(utc(20, 10), 3), *minutes(utc(20, 10, 4), 3)]))

    assert settled(db, "BTC-USD", utc(20, 10, 8), None, start_ts=utc(20, 10)) == minutes(utc(20, 10), 3)
    assert settled(db, "BTC-USD", utc(20, 10, 8), utc(20, 10, 2)) == []
    after = settled(db, "BTC-USD", utc(20, 10, 4) + SETTLE_S, utc(20, 10, 2))
    assert after == minutes(utc(20, 10, 4), 3)


def test_the_reference_minute_is_rounded_like_the_lower_bound(db: Database) -> None:
    put(db, bar("BTC-USD", utc(20, 10, 1), 50.0), bar("BTC-USD", utc(20, 10, 2), 50.0))
    asof = utc(20, 10, 3)

    # start 10:00:30: the first bar that can be returned opens at 10:01, so nothing is missing
    assert settled(db, "BTC-USD", asof, None, start_ts=utc(20, 10, 0, 30)) == minutes(utc(20, 10, 1), 2)
    assert settled(db, "BTC-USD", asof, utc(20, 10, 0, 30)) == minutes(utc(20, 10, 1), 2)
    # start 10:00: the 10:00 minute is missing and still young
    assert settled(db, "BTC-USD", asof, None, start_ts=utc(20, 10)) == []
    assert settled(db, "BTC-USD", asof, utc(20, 9, 59)) == []
    # the stricter of ts and start_ts is the lower bound
    assert settled(db, "BTC-USD", asof, utc(20, 9), start_ts=utc(20, 10, 1)) == minutes(utc(20, 10, 1), 2)
    # without any bound the first bar has no reference
    assert settled(db, "BTC-USD", asof, None) == minutes(utc(20, 10, 1), 2)


def test_nights_weekends_and_lunch_breaks_are_no_holes(db: Database) -> None:
    put(db, bar("NVDA", utc(20, 20, 59), 100.0), bar("NVDA", utc(23, 14, 30), 101.0))
    put(db, bar("7203.T", utc(20, 2, 29), 2_900.0), bar("7203.T", utc(20, 3, 30), 2_910.0))

    assert settled(db, "NVDA", utc(23, 14, 31), utc(20, 20, 59)) == [utc(23, 14, 30)]  # weekend
    assert settled(db, "7203.T", utc(20, 3, 31), utc(20, 2, 29)) == [utc(20, 3, 30)]  # Tokyo lunch break
    first = reader(db, utc(23, 14, 31)).first_settled_bar_at_or_after("NVDA", utc(20, 21))
    assert first is not None and first.ts == utc(23, 14, 30)


def test_a_missing_first_minute_of_the_session_is_a_hole(db: Database) -> None:
    put(db, bar("NVDA", utc(20, 20, 59), 100.0), bar("NVDA", utc(23, 14, 31), 101.0))  # 14:30 is missing

    assert settled(db, "NVDA", utc(23, 14, 32), utc(20, 20, 59)) == []
    assert settled(db, "NVDA", utc(23, 14, 31) + SETTLE_S - 1, utc(20, 20, 59)) == []
    assert settled(db, "NVDA", utc(23, 14, 31) + SETTLE_S, utc(20, 20, 59)) == [utc(23, 14, 31)]


def test_without_calendar_or_instrument_every_missing_minute_counts_as_traded(db: Database) -> None:
    put(db, bar("NVDA", utc(20, 20, 59), 100.0), bar("NVDA", utc(23, 14, 30), 101.0))
    put(db, bar("ORPHAN", utc(20, 20, 59), 10.0), bar("ORPHAN", utc(23, 14, 30), 11.0))
    asof = utc(23, 14, 31)

    no_calendar = BarReader(db, INSTRUMENTS, None, asof)
    assert no_calendar.settled_bars_after("NVDA", utc(20, 20, 59)).empty  # the weekend could be a hole
    assert reader(db, asof).settled_bars_after("ORPHAN", utc(20, 20, 59)).empty
    later = BarReader(db, INSTRUMENTS, None, utc(23, 14, 31) + SETTLE_S)
    assert later.settled_bars_after("NVDA", utc(20, 20, 59)).index.tolist() == [utc(23, 14, 30)]


def test_a_venue_unknown_to_the_calendar_is_treated_like_a_24_7_venue(db: Database) -> None:
    class StrictCalendar(FakeCalendar):
        def segments(self, exchange, start, end, *, extended=False):  # type: ignore[no-untyped-def]
            raise KeyError(exchange)

    put(db, bar("NVDA", utc(20, 20, 59), 100.0), bar("NVDA", utc(23, 14, 30), 101.0))
    view = BarReader(db, INSTRUMENTS, StrictCalendar(), utc(23, 14, 31))

    assert view.settled_bars_after("NVDA", utc(20, 20, 59)).empty
    assert view.first_settled_bar_at_or_after("NVDA", utc(20, 21)) is None


# ------------------------------------------------------------------------------------------------
# Fill bar
# ------------------------------------------------------------------------------------------------
def test_the_fill_bar_is_returned_only_once_it_is_settled(db: Database) -> None:
    placed = utc(20, 15, 0, 20)
    put(db, bar("NVDA", utc(20, 15), 100.0), bar("NVDA", utc(20, 15, 1), 100.4, "stream"))

    def fill(asof: int) -> Bar | None:
        return reader(db, asof).first_settled_bar_at_or_after("NVDA", placed)

    assert reader(db, utc(20, 15, 2, 3)).first_bar_at_or_after("NVDA", placed) is not None
    assert fill(utc(20, 15, 2, 3)) is None  # provisional: the order stays pending
    put(db, bar("NVDA", utc(20, 15, 1), 100.2))
    filled = fill(utc(20, 15, 3, 3))
    assert filled == bar("NVDA", utc(20, 15, 1), 100.2)  # authoritative open price
    assert fill(utc(20, 16)) == filled  # later bars never change the fill bar


def test_the_fill_bar_waits_for_a_missing_minute_that_rest_may_still_insert(db: Database) -> None:
    placed = utc(20, 15, 0, 20)
    put(db, bar("NVDA", utc(20, 15), 100.0), bar("NVDA", utc(20, 15, 3), 100.6))  # 15:01, 15:02 missing

    def fill(asof: int) -> int | None:
        found = reader(db, asof).first_settled_bar_at_or_after("NVDA", placed)
        return None if found is None else found.ts

    assert fill(utc(20, 15, 4, 3)) is None
    put(db, bar("NVDA", utc(20, 15, 1), 100.2))  # the REST reconciliation inserts 15:01
    assert fill(utc(20, 15, 5, 3)) == utc(20, 15, 1)


def test_the_fill_bar_accepts_an_old_hole_and_an_old_provisional_bar(db: Database) -> None:
    placed = utc(20, 15, 0, 20)
    put(db, bar("NVDA", utc(20, 15, 3), 100.6, "stream"))  # 15:01 and 15:02 never traded

    def fill(asof: int) -> int | None:
        found = reader(db, asof).first_settled_bar_at_or_after("NVDA", placed)
        return None if found is None else found.ts

    assert fill(utc(20, 15, 4) + SETTLE_S - 1) is None
    assert fill(utc(20, 15, 4) + SETTLE_S) == utc(20, 15, 3)


def test_the_fill_bar_is_never_a_bar_that_is_still_open(db: Database) -> None:
    put(db, bar("NVDA", utc(20, 15, 5), 100.0))

    assert reader(db, utc(20, 15, 5, 59)).first_settled_bar_at_or_after("NVDA", utc(20, 15, 5)) is None
    assert reader(db, utc(20, 15, 6)).first_settled_bar_at_or_after("NVDA", utc(20, 15, 5)) is not None
    assert reader(db, utc(20, 15, 6)).first_settled_bar_at_or_after("NVDA", utc(20, 15, 6)) is None
    assert reader(db, utc(20, 15, 6)).first_settled_bar_at_or_after("UNKNOWN", 0) is None


# ------------------------------------------------------------------------------------------------
# The consumer promise
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("symbol", ["BTC-USD", "SAP.DE", "NVDA"])
def test_a_consumer_processes_every_final_bar_exactly_once_with_its_final_values(
    db: Database, symbol: str
) -> None:
    """Simulated live feed: provisional stream bars arrive when their minute ends (+ feed delay), some
    are missed (including a 20-minute stream outage), the authoritative REST bars follow within the
    reconciliation round, and some minutes never trade. A consumer that wakes every minute and moves a
    high-water mark over ``settled_bars_after`` must process exactly the final bars, each once, with the
    final values. The same consumer on ``bars_after`` does not."""
    instrument = INSTRUMENTS[symbol]
    delay = instrument.delay_min * MIN
    calendar = FakeCalendar()
    start = utc(20, 15)  # all three venues trade 15:00-16:30
    end = utc(20, 16, 30)
    rng = random.Random(f"settled-{symbol}")
    arrivals: list[tuple[int, int, Bar]] = []  # (arrival time, order, bar)
    final: dict[int, Bar] = {}
    for minute in range(start, end, MIN):
        if rng.random() < 0.12:
            continue  # nothing traded in this minute
        price = 100.0 + rng.uniform(-1, 1)
        rest = Bar(symbol, minute, price, price + 0.4, price - 0.4, price + 0.1, 5.0, "rest")
        final[minute] = rest
        arrive = minute + MIN + delay
        stream_down = utc(20, 15, 30) <= minute < utc(20, 15, 50)
        if not stream_down and rng.random() < 0.85:
            source: BarSource = rng.choice(["stream", "stream", "quote"])
            provisional = Bar(symbol, minute, price + 0.2, price + 0.9, price - 0.9, price - 0.1, 3.0, source)
            arrivals.append((arrive + rng.randrange(0, 5), len(arrivals), provisional))
        arrivals.append((arrive + rng.randrange(60, 540), len(arrivals), rest))  # within the REST round
    arrivals.sort()

    def consume(*, use_settled: bool) -> list[tuple[int, tuple[float, ...]]]:
        repo_market.delete_bars_before(db, end + 86_400)
        pending = list(arrivals)
        seen: list[tuple[int, tuple[float, ...]]] = []
        checked: int | None = None
        for wake in range(start + 3, end + delay + SETTLE_S + 30 * MIN, MIN):
            while pending and pending[0][0] <= wake:
                repo_market.upsert_bar_rows(db, [pending.pop(0)[2]])
            view = BarReader(db, INSTRUMENTS, calendar, wake)
            frame = (
                view.settled_bars_after(symbol, checked, start_ts=start)
                if use_settled
                else view.bars_after(symbol, checked, start_ts=start)
            )
            assert all(ts + MIN <= wake for ts in frame.index)
            for ts, row in zip(frame.index.tolist(), frame.itertuples(index=False, name=None), strict=True):
                seen.append((ts, tuple(row)))
            if not frame.empty:
                checked = int(frame.index[-1])
        return seen

    expected = [
        (ts, (b.open, b.high, b.low, b.close, b.volume)) for ts, b in sorted(final.items())
    ]
    assert len(expected) > 60
    assert consume(use_settled=True) == expected
    assert consume(use_settled=False) != expected  # the old door acts on provisional bars and skips minutes


def test_settled_results_are_prefixes_that_only_grow(db: Database) -> None:
    rng = np.random.default_rng(7)
    start = utc(20, 15)
    for minute in minutes(start, 60):
        if rng.random() < 0.15:
            continue
        put(db, bar("BTC-USD", minute, 50.0, "stream" if rng.random() < 0.3 else "rest"))

    previous: pd.DataFrame | None = None
    for asof in range(start, start + 90 * MIN, 37):
        view = BarReader(db, INSTRUMENTS, FakeCalendar(), asof)
        frame = view.settled_bars_after("BTC-USD", None, start_ts=start)
        everything = view.bars_after("BTC-USD", None, start_ts=start)
        pd.testing.assert_frame_equal(frame, everything.iloc[: len(frame)], check_exact=True)
        if previous is not None:
            assert len(frame) >= len(previous)
            pd.testing.assert_frame_equal(frame.iloc[: len(previous)], previous, check_exact=True)
        first = view.first_settled_bar_at_or_after("BTC-USD", start)
        assert (None if first is None else first.ts) == (None if frame.empty else int(frame.index[0]))
        previous = frame
    assert previous is not None and len(previous) > 40
