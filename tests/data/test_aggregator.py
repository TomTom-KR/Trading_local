import random

from trading_local.data.aggregator import MinuteAggregator
from trading_local.data.provider import Tick

T0 = 1_795_186_800  # 2026-11-20 15:00:00 UTC (a full minute)


def tick(ts: float, price: float, *, dv: float | None = None, lag: float = 1.0, symbol: str = "NVDA") -> Tick:
    return Tick(symbol=symbol, ts=ts, price=price, received_ts=ts + lag, day_volume=dv)


def test_ohlc_of_a_minute_and_emission_on_the_next_minute() -> None:
    agg = MinuteAggregator("stream")
    assert agg.add(tick(T0 + 1, 10.0)) == []
    assert agg.add(tick(T0 + 20, 12.0)) == []
    assert agg.add(tick(T0 + 40, 9.0)) == []
    assert agg.add(tick(T0 + 59, 11.0)) == []
    bars = agg.add(tick(T0 + 61, 11.5))
    assert len(bars) == 1
    bar = bars[0]
    assert (bar.symbol, bar.ts, bar.open, bar.high, bar.low, bar.close) == ("NVDA", T0, 10.0, 12.0, 9.0, 11.0)
    assert bar.source == "stream"


def test_volume_from_day_volume_deltas_and_session_reset() -> None:
    agg = MinuteAggregator("quote")
    agg.add(tick(T0 + 1, 10.0, dv=1_000))  # first value: baseline only
    agg.add(tick(T0 + 30, 10.0, dv=1_300))
    agg.add(tick(T0 + 50, 10.0, dv=1_350))
    (first,) = agg.add(tick(T0 + 65, 10.0, dv=1_400))
    assert first.volume == 350 and first.source == "quote"
    # the counter goes DOWN: a new session started, the new value counts completely
    (second,) = agg.add(tick(T0 + 125, 10.0, dv=200))
    assert second.ts == T0 + 60 and second.volume == 50
    (third,) = agg.flush(T0 + 200)
    assert third.ts == T0 + 120 and third.volume == 200


def test_volume_never_negative_and_zero_without_day_volume() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 1, 10.0))
    agg.add(tick(T0 + 2, 10.0, dv=None))
    (bar,) = agg.add(tick(T0 + 61, 10.0))
    assert bar.volume == 0.0


def test_flush_waits_for_the_end_of_the_minute_plus_grace() -> None:
    agg = MinuteAggregator("stream", grace_s=1.0)
    agg.add(tick(T0 + 10, 10.0, lag=2.0))
    assert agg.flush(T0 + 59) == []
    assert agg.flush(T0 + 62) == []  # lag 2 s + grace 1 s: not yet
    (bar,) = agg.flush(T0 + 63)
    assert bar.ts == T0
    assert agg.flush(T0 + 120) == []  # emitted only once


def test_flush_respects_the_delay_of_a_delayed_feed() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0, lag=900.0))  # a 15-minute delayed feed
    assert agg.flush(T0 + 120) == []
    assert agg.flush(T0 + 959) == []
    (bar,) = agg.flush(T0 + 960)  # minute end + 900 s delay (no grace by default)
    assert bar.ts == T0


def test_duplicate_ticks_are_ignored() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 5, 10.0, dv=100))
    agg.add(tick(T0 + 6, 11.0, dv=150))
    agg.add(tick(T0 + 6, 11.0, dv=150))  # same tick from a second connection
    (bar,) = agg.flush(T0 + 100)
    assert bar.volume == 50 and bar.high == 11.0


def test_late_tick_corrects_an_emitted_bar() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0))
    agg.add(tick(T0 + 30, 10.5))
    agg.add(tick(T0 + 70, 11.0))  # emits minute T0
    (corrected,) = agg.add(tick(T0 + 45, 12.0))  # late, newest of minute T0
    assert (corrected.ts, corrected.high, corrected.close) == (T0, 12.0, 12.0)
    (again,) = agg.add(tick(T0 + 20, 8.0))  # late but older than the close tick: close stays
    assert (again.low, again.close) == (8.0, 12.0)


def test_late_tick_for_the_open_minute_is_merged_without_emission() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0))
    agg.add(tick(T0 + 50, 10.5))
    assert agg.add(tick(T0 + 30, 13.0)) == []
    (bar,) = agg.flush(T0 + 100)
    assert (bar.high, bar.close) == (13.0, 10.5)


def test_late_tick_older_than_the_retention_is_dropped() -> None:
    agg = MinuteAggregator("stream", retention_s=120)
    agg.add(tick(T0 + 10, 10.0))
    agg.add(tick(T0 + 600, 11.0))
    assert agg.add(tick(T0 + 20, 9.0)) == []


def test_gaps_produce_no_bars() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0))
    bars = agg.add(tick(T0 + 5 * 60 + 3, 11.0))
    assert [bar.ts for bar in bars] == [T0]
    assert [bar.ts for bar in agg.flush(T0 + 1000)] == [T0 + 300]


def test_late_tick_for_a_minute_without_bar_creates_one() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0))
    agg.add(tick(T0 + 130, 11.0))  # emits T0, opens T0 + 120
    (bar,) = agg.add(tick(T0 + 70, 10.7))  # minute T0 + 60 had no tick so far
    assert (bar.ts, bar.open, bar.close) == (T0 + 60, 10.7, 10.7)


def test_in_order_tick_into_a_flushed_minute_corrects_it() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 10, 10.0, lag=1.0))
    (flushed,) = agg.flush(T0 + 62)
    assert flushed.close == 10.0
    (corrected,) = agg.add(tick(T0 + 50, 10.4, lag=20.0))  # arrives late but in source order
    assert (corrected.ts, corrected.close, corrected.high) == (T0, 10.4, 10.4)


def test_invalid_prices_are_ignored_and_symbols_are_independent() -> None:
    agg = MinuteAggregator("stream")
    assert agg.add(tick(T0 + 1, 0.0)) == []
    assert agg.add(tick(T0 + 2, float("nan"))) == []
    agg.add(tick(T0 + 3, 10.0, symbol="A"))
    agg.add(tick(T0 + 4, 20.0, symbol="B"))
    bars = agg.flush(T0 + 100)
    assert [(bar.symbol, bar.close) for bar in bars] == [("A", 10.0), ("B", 20.0)]


def test_random_tick_streams_keep_the_bar_invariants() -> None:
    rng = random.Random(42)
    for _ in range(50):
        agg = MinuteAggregator("stream", retention_s=600)
        ticks = [tick(T0 + rng.uniform(0, 900), rng.uniform(90, 110), dv=float(rng.randint(0, 5_000))) for _ in range(200)]
        ticks += rng.sample(ticks, 20)  # duplicates
        ticks.sort(key=lambda t: t.ts + rng.uniform(-30, 0))  # mostly ordered, some late arrivals
        bars = [bar for t in ticks for bar in agg.add(t)] + agg.flush(T0 + 10_000)
        latest = {bar.ts: bar for bar in bars}  # corrections replace earlier versions
        for bar in latest.values():
            assert bar.ts % 60 == 0 and T0 <= bar.ts < T0 + 900
            assert bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high
            assert bar.volume >= 0
        assert agg.flush(T0 + 20_000) == []  # nothing left open


def test_forget_discards_the_state_of_a_symbol() -> None:
    agg = MinuteAggregator("stream")
    agg.add(tick(T0 + 1, 10.0))
    agg.forget("NVDA")
    assert agg.flush(T0 + 1000) == []
