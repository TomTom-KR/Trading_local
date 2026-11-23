"""Live smoke tests against Yahoo Finance (deselected by default; run with ``-m network``).

About 6 HTTP requests (cookie + crumb, one chart, one daily chart, two quote batches) and one WebSocket.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from trading_local.clock import SystemClock
from trading_local.data.provider import Tick
from trading_local.data.stream import YahooStream
from trading_local.data.yahoo import YahooProvider

pytestmark = pytest.mark.network


async def test_btc_minutes_daily_quote_and_lookup(tmp_path: Path) -> None:
    clock = SystemClock()
    provider = YahooProvider(clock=clock, cache_dir=tmp_path / "yfinance")
    try:
        now = clock.ts()
        frames = await provider.fetch_minutes(["BTC-USD"], start_ts=now - 86_400 + 120)
        btc = frames["BTC-USD"]
        assert len(btc) > 500 and btc.index.dtype == np.int64
        assert btc.index.is_monotonic_increasing and btc.index.is_unique
        assert (btc.index % 60 == 0).all() and (btc[["open", "high", "low", "close"]] > 0).all().all()
        assert now - 900 <= btc.index[-1] <= now

        daily = (await provider.fetch_daily(["BTC-USD"], start_ts=now - 10 * 86_400))["BTC-USD"]
        assert 8 <= len(daily) <= 11 and daily.index.name == "day"

        (quote,) = await provider.fetch_quotes(["BTC-USD"])
        assert quote.symbol == "BTC-USD" and quote.price > 0 and abs(quote.ts - now) < 600
        assert quote.price == pytest.approx(btc["close"].iloc[-1], rel=0.05)

        shell = await provider.lookup("SHEL.L")
        assert shell is not None and (shell.currency, shell.price_scale, shell.exchange) == ("GBP", 0.01, "XLON")
        assert provider.paused_until() is None
    finally:
        await provider.close()


async def test_stream_delivers_crypto_ticks() -> None:
    ticks: list[Tick] = []
    states: list[bool] = []
    stream = YahooStream(ticks.append, states.append, clock=SystemClock())
    await stream.subscribe(["BTC-USD", "ETH-USD"])
    await stream.connect()
    try:
        for _ in range(300):
            if len({tick.symbol for tick in ticks}) == 2:
                break
            await asyncio.sleep(0.1)
        assert {tick.symbol for tick in ticks} == {"BTC-USD", "ETH-USD"}
        assert states == [True] and stream.connected
        assert all(tick.price > 0 and tick.received_ts - tick.ts < 120 for tick in ticks)
    finally:
        await stream.close()
