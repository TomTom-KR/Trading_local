"""YahooProvider plumbing with yfinance replaced by fakes: request budget, circuit breaker, hidden errors."""

import logging
import threading
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from trading_local.clock import ManualClock
from trading_local.data.provider import ProviderError, RateLimitError
from trading_local.data.yahoo import CircuitBreaker, TokenBucket, YahooProvider

T0 = 1_795_186_800  # 2026-11-20 15:00:00 UTC


class FakeMonotonic:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_token_bucket_rate_and_capacity() -> None:
    clock = FakeMonotonic()
    bucket = TokenBucket(rate=2.0, capacity=4, monotonic=clock)
    stop = threading.Event()
    assert bucket.wait_time(4) == 0.0
    assert bucket.acquire(4, stop)
    assert bucket.wait_time(1) == pytest.approx(0.5)
    clock.now = 1.0  # two tokens refilled
    assert bucket.acquire(2, stop)
    assert bucket.wait_time(10) == pytest.approx(2.0)  # needs min(10, capacity) = 4 tokens
    clock.now = 100.0
    assert bucket.wait_time(4) == 0.0  # never more than the capacity is stored
    assert bucket.acquire(10, stop)  # a big call is allowed and paid off later
    assert bucket.wait_time(1) == pytest.approx(3.5)


def test_token_bucket_acquire_aborts_when_stopped() -> None:
    bucket = TokenBucket(rate=0.001, capacity=1, monotonic=FakeMonotonic())
    stop = threading.Event()
    assert bucket.acquire(1, stop)
    stop.set()
    assert not bucket.acquire(1, stop)


def test_circuit_breaker_escalates_and_resets() -> None:
    clock = ManualClock(T0)
    breaker = CircuitBreaker(clock, initial_s=600, max_s=2_000)
    assert breaker.paused_until() is None
    breaker.check()
    assert breaker.trip("429") == T0 + 600
    assert breaker.trip("again") == T0 + 600  # no escalation while paused
    with pytest.raises(RateLimitError):
        breaker.check()
    clock.advance(600)
    assert breaker.paused_until() is None
    assert breaker.trip("429") == T0 + 600 + 1_200  # doubled
    clock.advance(1_200)
    assert breaker.trip("429") == clock.ts() + 2_000  # capped
    clock.advance(2_000)
    breaker.record_success()
    assert breaker.trip("429") == clock.ts() + 600  # back to the initial pause


# ------------------------------------------------------------------------------------------------
# Provider with yfinance replaced
# ------------------------------------------------------------------------------------------------
def _yf_frame(symbols: list[str], start: int, minutes: int) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(start + 60 * np.arange(minutes), unit="s", utc=True)).tz_convert(
        "America/New_York"
    )
    blocks = {
        symbol: pd.DataFrame(
            {"Open": 10.0, "High": 11.0, "Low": 9.0, "Close": 10.5, "Adj Close": 10.5, "Volume": 100.0}, index=index
        )
        for symbol in symbols
    }
    return pd.concat(blocks.values(), axis=1, keys=blocks.keys(), names=["Ticker", "Price"])


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(T0)


@pytest.fixture
def provider(clock: ManualClock, monkeypatch: pytest.MonkeyPatch) -> Iterator[YahooProvider]:
    def fake_open_session(self: YahooProvider) -> None:
        self._capture.install()
        self._session = object()

    monkeypatch.setattr(YahooProvider, "_open_session", fake_open_session)
    monkeypatch.setattr(YahooProvider, "_seed_timezones", lambda self, results: None)
    instance = YahooProvider(clock=clock, bucket=TokenBucket(rate=1_000, capacity=1_000))
    yield instance
    instance._stop.set()
    instance._executor.shutdown(wait=True)


async def test_fetch_minutes_converts_the_download(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_download(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        seen.update(kwargs, tickers=tickers, thread=threading.current_thread().name)
        return _yf_frame(tickers, T0 - 600, 10)

    monkeypatch.setattr("yfinance.download", fake_download)
    frames = await provider.fetch_minutes(["NVDA", "AAPL"], start_ts=T0 - 1_200)
    assert set(frames) == {"NVDA", "AAPL"}
    assert frames["NVDA"].index[0] == T0 - 600 and len(frames["NVDA"]) == 10
    assert seen["start"] == T0 - 1_200 and isinstance(seen["start"], int)
    assert seen["interval"] == "1m" and seen["auto_adjust"] is False and seen["progress"] is False
    assert seen["threads"] == 2 and seen["session"] is provider._session
    assert "repair" not in seen and "ignore_tz" not in seen and "period" not in seen
    assert seen["thread"].startswith("yahoo-rest")  # the dedicated worker thread


async def test_hidden_rate_limit_in_the_yfinance_log_opens_the_breaker(
    provider: YahooProvider, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def limited_download(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
        calls.append(tickers)
        logging.getLogger("yfinance").error(
            "%s: YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')", tickers
        )
        return pd.DataFrame()

    monkeypatch.setattr("yfinance.download", limited_download)
    with pytest.raises(RateLimitError):
        await provider.fetch_minutes(["NVDA"], start_ts=T0 - 1_200)
    assert provider.paused_until() == T0 + 600
    with pytest.raises(RateLimitError):  # refused without any request while paused
        await provider.fetch_daily(["NVDA"], start_ts=T0 - 86_400 * 10)
    assert len(calls) == 1
    clock.advance(601)
    assert provider.paused_until() is None


async def test_raised_rate_limit_and_transport_errors(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    from yfinance.exceptions import YFRateLimitError

    def raising(exc: Exception) -> Any:
        def download(tickers: list[str], **kwargs: Any) -> pd.DataFrame:
            raise exc

        return download

    monkeypatch.setattr("yfinance.download", raising(OSError("connection reset")))
    with pytest.raises(ProviderError):
        await provider.fetch_minutes(["NVDA"], start_ts=T0 - 1_200)
    assert provider.paused_until() is None
    monkeypatch.setattr("yfinance.download", raising(YFRateLimitError()))
    with pytest.raises(RateLimitError):
        await provider.fetch_minutes(["NVDA"], start_ts=T0 - 1_200)
    assert provider.paused_until() is not None


async def test_minute_window_limits(provider: YahooProvider) -> None:
    with pytest.raises(ValueError, match="8 days"):
        await provider.fetch_minutes(["NVDA"], start_ts=T0 - 9 * 86_400)
    with pytest.raises(ValueError, match="30 days"):
        await provider.fetch_minutes(["NVDA"], start_ts=T0 - 31 * 86_400, end_ts=T0 - 25 * 86_400)


class FakeResponse:
    def __init__(self, status: int, body: dict[str, Any] | None = None) -> None:
        self.status_code = status
        self._body = body or {}
        self.text = str(body)

    def json(self) -> dict[str, Any]:
        return self._body


def _fake_yfdata(responses: list[FakeResponse], seen: list[dict[str, Any]]) -> type:
    class FakeYfData:
        def __init__(self, session: object = None) -> None:
            pass

        def get(self, url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> FakeResponse:
            seen.append(params or {})
            return responses.pop(0)

    return FakeYfData


async def test_snapshot_quotes_in_one_request(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "quoteResponse": {
            "result": [
                {"symbol": "NVDA", "regularMarketPrice": 180.5, "regularMarketTime": T0 - 3, "marketState": "REGULAR"},
                {"symbol": "BRK.B", "quoteType": "EQUITY"},  # stub without price
            ],
            "error": None,
        }
    }
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr("yfinance.data.YfData", _fake_yfdata([FakeResponse(200, body)], seen))
    quotes = await provider.fetch_quotes(["NVDA", "BRK.B", "NVDA", "XXXX"])
    assert [(quote.symbol, quote.price, quote.received_ts) for quote in quotes] == [("NVDA", 180.5, T0)]
    assert len(seen) == 1 and seen[0]["symbols"] == "NVDA,BRK.B,XXXX"
    assert "regularMarketPrice" in seen[0]["fields"]


async def test_quote_http_429_opens_the_breaker(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yfinance.data.YfData", _fake_yfdata([FakeResponse(429)], []))
    with pytest.raises(RateLimitError):
        await provider.fetch_quotes(["NVDA"])
    assert provider.paused_until() == T0 + 600


async def test_quote_server_error_is_a_provider_error(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("yfinance.data.YfData", _fake_yfdata([FakeResponse(500)], []))
    with pytest.raises(ProviderError):
        await provider.fetch_quotes(["NVDA"])
    assert provider.paused_until() is None


async def test_lookup_derives_the_instrument(provider: YahooProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {
        "symbol": "AZN.L", "quoteType": "EQUITY", "currency": "GBp", "exchange": "LSE", "regularMarketPrice": 11_000.0,
        "regularMarketTime": T0, "exchangeDataDelayedBy": 15, "longName": "AstraZeneca PLC",
    }  # fmt: skip
    responses = [FakeResponse(200, {"quoteResponse": {"result": [raw]}}), FakeResponse(200, {"quoteResponse": {"result": []}})]
    monkeypatch.setattr("yfinance.data.YfData", _fake_yfdata(responses, []))
    instrument = await provider.lookup("AZN.L")
    assert instrument is not None and instrument.currency == "GBP" and instrument.price_scale == 0.01
    assert instrument.aliases[:2] == ("AstraZeneca PLC", "AstraZeneca")
    assert await provider.lookup("NOPE.XX") is None


async def test_close_stops_the_worker(clock: ManualClock) -> None:
    provider = YahooProvider(clock=clock)
    await provider.close()
    with pytest.raises(ProviderError):
        await provider.fetch_quotes(["NVDA"])
