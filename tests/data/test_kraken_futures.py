"""Kraken Futures client: symbol mapping, answer parsing, paging and politeness (offline, MockTransport)."""

import json

import httpx
import pytest

from trading_local.clock import ManualClock
from trading_local.data.kraken import AsyncRateLimiter
from trading_local.data.kraken_futures import (
    KrakenFuturesClient,
    parse_funding_history,
    parse_series,
    parse_tickers,
    perpetual_symbol,
)
from trading_local.data.provider import ProviderError, RateLimitError
from trading_local.models import Instrument

# Shapes as answered by the live API on 2026-11-24 (values shortened).
TICKERS = {
    "result": "success",
    "serverTime": "2026-11-24T21:56:07.733Z",
    "tickers": [
        {"symbol": "PF_XBTUSD", "tag": "perpetual", "pair": "XBT:USD", "markPrice": 59804.887, "indexPrice": 59801.7,
         "openInterest": 2080.3556, "fundingRate": 0.5537, "fundingRatePrediction": 0.6263,
         "relativeFundingRate": 9.2749625e-06, "relativeFundingRatePrediction": 1.0473e-05, "suspended": False},
        {"symbol": "PF_ETHUSD", "tag": "perpetual", "markPrice": "1678.29", "indexPrice": 1678.3,
         "openInterest": 31260.076, "relativeFundingRate": "bad", "suspended": True},
        {"symbol": "FI_XBTUSD_261225", "tag": "quarter", "markPrice": 60500.0},
        "garbage",
    ],
}  # fmt: skip


def crypto(symbol: str) -> Instrument:
    return Instrument(symbol, symbol, "crypto", "GLOBAL", "KRAKEN", "EUR")


def test_perpetual_of_each_crypto_instrument() -> None:
    assert perpetual_symbol(crypto("BTC-EUR")) == "PF_XBTUSD"  # bitcoin is XBT on Kraken
    assert perpetual_symbol(crypto("ETH-EUR")) == "PF_ETHUSD"
    assert perpetual_symbol(crypto("DOGE-EUR")) == "PF_DOGEUSD"
    assert perpetual_symbol(crypto("BTC-USD")) == "PF_XBTUSD"
    assert perpetual_symbol(Instrument("NVDA", "NVIDIA", "equity", "US", "XNAS", "USD")) is None


def test_parse_tickers_keeps_perpetuals_and_checks_every_value() -> None:
    found = parse_tickers(TICKERS)
    assert set(found) == {"PF_XBTUSD", "PF_ETHUSD"}
    btc = found["PF_XBTUSD"]
    assert (btc.mark_price, btc.index_price, btc.open_interest) == (59804.887, 59801.7, 2080.3556)
    assert (btc.funding_rate, btc.funding_pred, btc.suspended) == (9.2749625e-06, 1.0473e-05, False)
    eth = found["PF_ETHUSD"]
    assert eth.mark_price == 1678.29 and eth.funding_rate is None and eth.suspended is True
    with pytest.raises(ProviderError):
        parse_tickers({"result": "error", "error": "apiLimitExceeded"})


def test_parse_series_of_every_shape() -> None:
    liquidations = {"timestamp": [1795554000, 1795554060, 1795554120], "data": ["0", "0.03", "x"], "more": False}
    assert parse_series(liquidations) == ([(1795554000, 0.0), (1795554060, 0.03)], False)
    interest = {"timestamp": [1795554000], "data": [["2068.7", "2068.8", "2067.4", "2067.41"]], "more": True}
    assert parse_series(interest) == ([(1795554000, 2067.41)], True)
    funding = {"timestamp": [1795554000000], "data": {"rate": [["-0.15", "0.55", "-0.15", "0.55"]]}, "more": False}
    assert parse_series(funding) == ([(1795554000, 0.55)], False)  # milliseconds
    with pytest.raises(ProviderError):
        parse_series({"timestamp": [1, 2], "data": ["0"]})
    with pytest.raises(ProviderError):
        parse_series(None)


def test_parse_funding_history() -> None:
    body = {"rates": [
        {"timestamp": "2026-11-24T21:00:00Z", "fundingRate": 0.55, "relativeFundingRate": 9.27e-06},
        {"timestamp": "2026-11-24T20:00:00Z", "fundingRate": -0.15, "relativeFundingRate": -2.59e-06},
        {"timestamp": "not a date", "relativeFundingRate": 1.0},
        {"timestamp": "2026-11-24T19:00:00Z"},
    ]}  # fmt: skip
    assert parse_funding_history(body) == [(1795546800 + 3600, -2.59e-06), (1795546800 + 7200, 9.27e-06)]
    with pytest.raises(ProviderError):
        parse_funding_history({"result": "success"})


def make_client(handler, clock: ManualClock) -> KrakenFuturesClient:
    return KrakenFuturesClient(
        clock, transport=httpx.MockTransport(handler), limiter=AsyncRateLimiter(rate=1000.0, capacity=1000.0)
    )


async def test_client_follows_analytics_pages(clock: ManualClock) -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen.append(params)
        assert request.url.path == "/api/charts/v1/analytics/PF_XBTUSD/liquidation-volume"
        if params["since"] == "1000":
            body = {"result": {"timestamp": [1000, 1060], "data": ["1", "2"], "more": True}, "errors": []}
        else:
            body = {"result": {"timestamp": [1060, 1120], "data": ["2", "3"], "more": False}, "errors": []}
        return httpx.Response(200, json=body)

    client = make_client(handler, clock)
    points = await client.series("PF_XBTUSD", "liquidation-volume", interval_s=60, since_ts=1000)
    assert points == [(1000, 1.0), (1060, 2.0), (1120, 3.0)]
    assert [params["since"] for params in seen] == ["1000", "1120"] and seen[0]["interval"] == "60"
    await client.close()


async def test_client_errors_and_rate_limit(clock: ManualClock) -> None:
    answers = [
        httpx.Response(200, json={"result": "error", "error": "unavailable"}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(503),
        httpx.Response(429),
        httpx.Response(200, json=TICKERS),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return answers.pop(0)

    client = make_client(handler, clock)
    for _ in range(3):
        with pytest.raises(ProviderError):
            await client.tickers()
    with pytest.raises(RateLimitError):
        await client.tickers()
    assert client.paused_until() == clock.ts() + 60
    with pytest.raises(RateLimitError):  # paused: no request is sent
        await client.tickers()
    clock.advance(61)
    assert set(await client.tickers()) == {"PF_XBTUSD", "PF_ETHUSD"}
    await client.close()
    with pytest.raises(ProviderError):
        await client.tickers()


async def test_client_reports_analytics_errors(clock: ManualClock) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({"result": None, "errors": ["unknown symbol"]}))

    client = make_client(handler, clock)
    with pytest.raises(ProviderError, match="unknown symbol"):
        await client.series("PF_NOPEUSD", "open-interest", interval_s=3600, since_ts=0)
    await client.close()
