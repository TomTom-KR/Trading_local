"""Live smoke test against Kraken's public API (deselected by default; run with ``-m network``).

Four REST requests (AssetPairs, Ticker, OHLC 1-minute, OHLC daily) and one WebSocket session of a few seconds
(until live quotes and closed candles - those of the subscription snapshot - arrived). No API key.
"""

import asyncio

import numpy as np
import pytest

from trading_local.clock import SystemClock
from trading_local.data.kraken import KrakenProvider
from trading_local.data.provider import Tick
from trading_local.models import Bar

pytestmark = pytest.mark.network


async def test_btc_eur_ticker_ohlc_and_websocket() -> None:
    clock = SystemClock()
    provider = KrakenProvider(clock=clock)
    try:
        now = clock.ts()
        (quote,) = await provider.fetch_quotes(["BTC-EUR"])  # AssetPairs (validation) + Ticker
        assert quote.symbol == "BTC-EUR" and quote.price > 0 and quote.bid and quote.ask and quote.bid <= quote.ask
        assert quote.prev_close and quote.change_pct is not None and quote.day_volume
        assert provider.pair("BTC-EUR").status == "online"  # type: ignore[union-attr]

        start = now // 60 * 60 - 60 * 60
        btc = (await provider.fetch_minutes(["BTC-EUR"], start_ts=start))["BTC-EUR"]
        assert len(btc) > 30 and btc.index.dtype == np.int64 and btc.index.is_unique
        assert btc.index.is_monotonic_increasing and (btc.index % 60 == 0).all()
        assert btc.index.min() >= start and btc.index.max() + 60 <= clock.ts()  # no forming candle
        assert (btc["volume"] > 0).all()  # minutes without trades have no bar
        assert btc["close"].iloc[-1] == pytest.approx(quote.price, rel=0.02)

        daily = (await provider.fetch_daily(["BTC-EUR"], start_ts=now - 10 * 86_400))["BTC-EUR"]
        assert 9 <= len(daily) <= 11 and daily.index[-1] == clock.now().strftime("%Y-%m-%d")
        assert provider.requests == 4

        ticks: list[Tick] = []
        bars: list[Bar] = []
        states: list[bool] = []
        stream = provider.open_candle_stream(ticks.append, bars.append, states.append)
        await stream.subscribe(["BTC-EUR", "ETH-EUR"])
        await stream.connect()
        try:
            async with asyncio.timeout(90):
                while not (stream.connected and ticks and bars and clock.ts() % 60 > 5):
                    await asyncio.sleep(0.5)
        finally:
            await stream.close()
        assert states and states[0] is True
        assert {tick.symbol for tick in ticks} <= {"BTC-EUR", "ETH-EUR"} and ticks[0].price > 0
        assert all(bar.ts + 60 <= clock.ts() and bar.source == "stream" for bar in bars)  # only closed candles
        assert abs(ticks[-1].ts - clock.ts()) < 3_600
    finally:
        await provider.close()
