import threading

import pytest

from trading_local.data.provider import Tick
from trading_local.data.quotebook import QuoteBook, merge_quotes, quote_from_tick
from trading_local.models import Quote


def q(ts: int, price: float, **fields: object) -> Quote:
    return Quote(symbol="SAP.DE", ts=ts, received_ts=ts + 1, price=price, **fields)  # type: ignore[arg-type]


def test_newer_update_wins_and_missing_fields_keep_their_value() -> None:
    stored = q(100, 180.0, bid=179.9, ask=180.1, prev_close=178.0, day_volume=5_000.0, market_state="REGULAR")
    merged = merge_quotes(stored, q(110, 181.0, ask=181.2, source="stream"))
    assert merged.price == 181.0 and merged.ts == 110 and merged.source == "stream"
    assert merged.ask == 181.2 and merged.bid == 179.9 and merged.day_volume == 5_000.0
    assert merged.market_state == "REGULAR"


def test_older_update_only_fills_unknown_fields() -> None:
    stored = q(110, 181.0, source="stream")
    merged = merge_quotes(stored, q(100, 180.0, prev_close=178.0, bid=179.9))
    assert merged.price == 181.0 and merged.ts == 110 and merged.source == "stream"
    assert merged.prev_close == 178.0 and merged.bid == 179.9
    assert merged.change_pct == pytest.approx((181.0 / 178.0 - 1) * 100)


def test_change_pct_is_computed_from_prev_close_when_missing() -> None:
    merged = merge_quotes(None, q(100, 110.0, prev_close=100.0))
    assert merged.change_pct == pytest.approx(10.0)
    newer = merge_quotes(merged, q(101, 99.0))  # new price without change: recomputed
    assert newer.change_pct == pytest.approx(-1.0)
    delivered = merge_quotes(newer, q(102, 99.5, change_pct=0.25))  # delivered value is used
    assert delivered.change_pct == 0.25


def test_change_pct_is_dropped_when_it_cannot_be_recomputed_for_a_new_price() -> None:
    merged = merge_quotes(q(100, 10.0, change_pct=1.5), q(101, 11.0))
    assert merged.change_pct is None
    same_price = merge_quotes(q(100, 10.0, change_pct=1.5), q(101, 10.0))
    assert same_price.change_pct == 1.5


def test_change_tracking_per_consumer() -> None:
    book = QuoteBook()
    assert book.update(q(100, 180.0))
    assert not book.update(q(100, 180.0))  # nothing changed
    assert [quote.symbol for quote in book.take_changed("publish")] == ["SAP.DE"]
    assert book.take_changed("publish") == []
    assert [quote.price for quote in book.take_changed("persist")] == [180.0]
    with pytest.raises(KeyError):
        book.take_changed("unknown")


def test_load_does_not_mark_changes_and_reads_return_copies() -> None:
    book = QuoteBook()
    book.load([q(100, 180.0)])
    assert book.take_changed("publish") == []
    copy = book.get("SAP.DE")
    assert copy is not None
    copy.price = 1.0
    assert book.get("SAP.DE").price == 180.0  # type: ignore[union-attr]
    assert book.snapshot()["SAP.DE"].price == 180.0
    book.remove("SAP.DE")
    assert book.get("SAP.DE") is None


def test_quote_from_tick() -> None:
    tick = Tick("BTC-USD", 1000.7, 61_000.5, 1002.2, day_volume=5e9, change_pct=-0.5, market_state="REGULAR")
    quote = quote_from_tick(tick)
    assert (quote.ts, quote.received_ts, quote.price, quote.source) == (1000, 1002, 61_000.5, "stream")
    assert quote.day_volume == 5e9 and quote.change_pct == -0.5 and quote.market_state == "REGULAR"


def test_concurrent_updates_are_consistent() -> None:
    book = QuoteBook()

    def writer(offset: int) -> None:
        for ts in range(offset, offset + 2_000, 2):
            book.update(Quote("X", ts, ts, float(ts)))

    threads = [threading.Thread(target=writer, args=(start,)) for start in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final = book.get("X")
    assert final is not None and final.ts == 1_999 and final.price == 1_999.0
