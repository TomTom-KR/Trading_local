"""KrakenProvider: parsing of recorded Kraken answers, trades aggregation and pagination, politeness, errors."""

import asyncio
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

from trading_local.clock import ManualClock
from trading_local.data import kraken
from trading_local.data.kraken import (
    KRAKEN_PAIRS,
    AsyncRateLimiter,
    KrakenProvider,
    api_errors,
    is_rate_limit,
    ohlc_daily_frame,
    ohlc_minute_frame,
    parse_trades,
    quote_from_ticker,
    result_entry,
    trades_to_minutes,
    update_pairs,
)
from trading_local.data.provider import CandleProvider, Provider, ProviderError, RateLimitError
from trading_local.db.repo_market import BAR_COLUMNS

FIXTURES = Path(__file__).parent / "fixtures" / "kraken"
LTC_FORMING = 1_795_389_660  # open time of the forming candle in ohlc_ltc_eur_1.json
LTC_LAST = 1_795_389_600  # 'last' of that answer = newest committed candle


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def ok(result: Any) -> httpx.Response:
    return httpx.Response(200, json={"error": [], "result": result})


class FakeKraken:
    """httpx handler serving recorded answers by endpoint; records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[str, Callable[[httpx.Request], httpx.Response]] = {
            "AssetPairs": lambda r: httpx.Response(200, json=fixture("asset_pairs.json")),
            "Ticker": lambda r: httpx.Response(200, json=fixture("ticker.json")),
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        endpoint = request.url.path.rsplit("/", 1)[-1]
        route = self.routes.get(endpoint)
        if route is None:
            return httpx.Response(404, json={"error": ["EGeneral:Unknown method"]})
        return route(request)

    def of(self, endpoint: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/" + endpoint)]


@pytest.fixture
def fake() -> FakeKraken:
    return FakeKraken()


def no_wait_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(rate=1000.0, capacity=1000.0)


@pytest.fixture
async def provider(fake: FakeKraken, clock: ManualClock):
    clock.set(LTC_FORMING + 30)
    instance = KrakenProvider(clock=clock, transport=httpx.MockTransport(fake), limiter=no_wait_limiter())
    yield instance
    await instance.close()


# ------------------------------------------------------------------------------------------------
# Pair metadata
# ------------------------------------------------------------------------------------------------
def test_the_static_pair_table_matches_the_recorded_asset_pairs() -> None:
    updated, missing = update_pairs(KRAKEN_PAIRS, fixture("asset_pairs.json")["result"])
    assert missing == []
    assert updated == dict(KRAKEN_PAIRS)  # nothing changed: the static defaults are Kraken's values
    btc = updated["BTC-EUR"]
    assert (btc.ws_symbol, btc.altname, btc.rest_pair, btc.base, btc.quote) == ("BTC/EUR", "XBTEUR", "XXBTZEUR", "XXBT", "ZEUR")
    assert (btc.price_decimals, btc.lot_decimals, btc.tick_size, btc.ordermin) == (1, 8, Decimal("0.1"), Decimal("0.00006"))
    assert btc.lot_step == Decimal("0.00000001")
    assert updated["DOGE-EUR"].altname == "XDGEUR" and updated["DOGE-EUR"].tick_size == Decimal("0.0000001")


def test_changed_and_missing_pairs_are_applied() -> None:
    result = fixture("asset_pairs.json")["result"]
    result["LTC/EUR"]["ordermin"] = "0.1"
    result["LTC/EUR"]["status"] = "cancel_only"
    result["DOT/EUR"]["tick_size"] = "not a number"
    del result["ADA/EUR"]
    updated, missing = update_pairs(KRAKEN_PAIRS, result)
    assert missing == ["ADA-EUR"] and updated["ADA-EUR"].status == "unknown"
    assert (updated["LTC-EUR"].ordermin, updated["LTC-EUR"].status) == (Decimal("0.1"), "cancel_only")
    assert updated["DOT-EUR"].tick_size == KRAKEN_PAIRS["DOT-EUR"].tick_size  # malformed value ignored


def test_result_entries_are_found_under_any_kraken_name() -> None:
    pair = KRAKEN_PAIRS["BTC-EUR"]
    assert result_entry({"BTC/EUR": 1}, pair) == 1
    assert result_entry({"XXBTZEUR": 2, "last": 5}, pair) == 2
    assert result_entry({"XBTEUR": 3}, pair) == 3
    assert result_entry({"SOMETHING": 4, "last": 5}, pair) == 4  # the single entry of a one-pair answer
    assert result_entry({"A": 1, "B": 2}, pair) is None


# ------------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------------
def test_ticker_quotes_refer_to_the_utc_day_open() -> None:
    raw = fixture("ticker.json")["result"]["BTC/EUR"]
    quote = quote_from_ticker("BTC-EUR", raw, 1_795_389_700)
    assert quote is not None
    assert (quote.price, quote.bid, quote.ask) == (55643.0, 55648.2, 55655.2)
    assert quote.day_open == quote.prev_close == 54890.1
    assert quote.change_pct == pytest.approx((55643.0 / 54890.1 - 1) * 100)
    assert (quote.day_high, quote.day_low, quote.day_volume) == (56227.2, 53577.8, float(raw["v"][0]))
    assert (quote.ts, quote.received_ts, quote.source, quote.market_state) == (1_795_389_700, 1_795_389_700, "quote", None)
    assert quote_from_ticker("BTC-EUR", {"c": ["0", "1"]}, 1) is None
    assert quote_from_ticker("BTC-EUR", None, 1) is None


def test_ohlc_minutes_drop_the_forming_candle_and_the_fillers_without_trades() -> None:
    body = fixture("ohlc_ltc_eur_1.json")["result"]
    rows = body["LTC/EUR"]
    frame = ohlc_minute_frame(rows, body["last"])
    assert list(frame.columns) == list(BAR_COLUMNS) and frame.index.name == "ts"
    assert frame.index.dtype == np.int64 and all(dtype == np.float64 for dtype in frame.dtypes)
    traded = {row[0] for row in rows if row[7] > 0 and row[0] <= LTC_LAST}
    assert set(frame.index) == traded and LTC_FORMING not in frame.index
    assert len(traded) < len(rows) - 1  # the fixture has filler minutes (count 0)
    first = next(row for row in rows if row[0] in traded)
    assert frame.loc[first[0]].tolist() == [float(v) for v in first[1:5]] + [float(first[6])]
    window = ohlc_minute_frame(rows, body["last"], start_ts=rows[10][0], end_ts=rows[20][0])
    assert window.index.min() >= rows[10][0] and window.index.max() < rows[20][0]
    assert ohlc_minute_frame([], 5).empty and ohlc_minute_frame([[1, "x"]], None).empty


def test_ohlc_daily_bars_are_dated_by_the_utc_day() -> None:
    rows = fixture("ohlc_xbteur_1440.json")["result"]["XXBTZEUR"]
    frame = ohlc_daily_frame(rows)
    assert frame.index.name == "day" and frame.index[-1] == "2026-11-22"  # the running day stays (caller drops it)
    assert frame.index[-2] == "2026-11-21" and frame.loc["2026-11-21", "close"] == 54887.2
    assert len(ohlc_daily_frame(rows, start_ts=1_795_219_200)) == 2  # from 2026-11-21 00:00 UTC


def test_trades_become_minute_bars_without_bars_for_quiet_minutes() -> None:
    page = parse_trades(fixture("trades_ltc_eur_page1.json")["result"]["LTC/EUR"], "1795384756247305842")
    assert page.raw_count == 50 and page.last == "1795384756247305842" and page.ids.size == 50
    frame = trades_to_minutes(page.ids, page.times, page.prices, page.volumes)
    rows = fixture("trades_ltc_eur_page1.json")["result"]["LTC/EUR"]
    expected: dict[int, list[float]] = {}
    for price, volume, at, *_ in rows:
        minute = int(at // 60 * 60)
        p, v = float(price), float(volume)
        if minute not in expected:
            expected[minute] = [p, p, p, p, v]
        else:
            bar = expected[minute]
            bar[1], bar[2], bar[3], bar[4] = max(bar[1], p), min(bar[2], p), p, bar[4] + v
    assert list(frame.index) == sorted(expected)
    for minute, values in expected.items():
        assert frame.loc[minute].tolist() == pytest.approx(values)
    span = (frame.index[-1] - frame.index[0]) // 60 + 1
    assert len(frame) < span  # minutes without trades have no bar


def test_duplicate_trades_count_once_and_order_follows_time_then_id() -> None:
    ids = np.array([3, 1, 2, 2], dtype=np.int64)
    times = np.array([120.5, 120.1, 120.1, 120.1])
    prices = np.array([12.0, 10.0, 11.0, 11.0])
    volumes = np.array([1.0, 1.0, 1.0, 1.0])
    frame = trades_to_minutes(ids, times, prices, volumes)
    assert frame.loc[120].tolist() == [10.0, 12.0, 10.0, 12.0, 3.0]
    assert trades_to_minutes(ids, times, prices, volumes, start_ts=180).empty


def test_api_error_classification() -> None:
    assert api_errors({"error": ["EGeneral:Too many requests", "WGeneral:Note"]}) == ["EGeneral:Too many requests"]
    assert is_rate_limit(["EAPI:Rate limit exceeded"]) and is_rate_limit(["EService:Throttled: 1795389000"])
    assert not is_rate_limit(["EQuery:Unknown asset pair"])
    assert api_errors(None) == [] and api_errors({"error": "x"}) == []


# ------------------------------------------------------------------------------------------------
# Provider with recorded answers
# ------------------------------------------------------------------------------------------------
def test_the_provider_is_a_candle_provider(provider: KrakenProvider) -> None:
    assert isinstance(provider, Provider) and isinstance(provider, CandleProvider)


async def test_one_ticker_request_serves_all_pairs(provider: KrakenProvider, fake: FakeKraken) -> None:
    quotes = await provider.fetch_quotes(list(KRAKEN_PAIRS) + ["NVDA"])
    assert {q.symbol for q in quotes} == set(KRAKEN_PAIRS)
    assert len(fake.of("Ticker")) == 1
    assert fake.of("Ticker")[0].url.params["pair"] == ",".join(p.ws_symbol for p in KRAKEN_PAIRS.values())
    assert len(fake.of("AssetPairs")) == 1  # metadata validated once ...
    await provider.fetch_quotes(["BTC-EUR"])
    assert len(fake.of("AssetPairs")) == 1  # ... and not again within a day


async def test_recent_minutes_come_from_ohlc_without_the_forming_candle(
    provider: KrakenProvider, fake: FakeKraken
) -> None:
    fake.routes["OHLC"] = lambda r: httpx.Response(200, json=fixture("ohlc_ltc_eur_1.json"))
    start = LTC_FORMING - 30 * 60
    frames = await provider.fetch_minutes(["LTC-EUR", "UNKNOWN"], start_ts=start)
    assert set(frames) == {"LTC-EUR"}
    frame = frames["LTC-EUR"]
    assert frame.index.min() >= start and frame.index.max() <= LTC_LAST
    request = fake.of("OHLC")[0]
    assert (request.url.params["pair"], request.url.params["interval"]) == ("LTC/EUR", "1")
    assert int(request.url.params["since"]) == start - 60
    assert not fake.of("Trades")


async def test_daily_bars_come_from_ohlc_1440(provider: KrakenProvider, fake: FakeKraken) -> None:
    fake.routes["OHLC"] = lambda r: httpx.Response(200, json=fixture("ohlc_xbteur_1440.json"))
    frames = await provider.fetch_daily(["BTC-EUR"], start_ts=1_795_132_800)
    assert list(frames["BTC-EUR"].index) == ["2026-11-20", "2026-11-21", "2026-11-22"]
    assert fake.of("OHLC")[0].url.params["interval"] == "1440"


async def test_older_minutes_are_built_from_paginated_trades(
    provider: KrakenProvider, fake: FakeKraken, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kraken, "TRADES_PAGE", 50)  # the recorded pages hold 50 trades each
    page1, page2 = fixture("trades_ltc_eur_page1.json"), fixture("trades_ltc_eur_page2.json")
    start, end = 1_795_382_760, 1_795_384_860
    served: list[str] = []

    def trades(request: httpx.Request) -> httpx.Response:
        since = request.url.params["since"]
        served.append(since)
        assert request.url.params["count"] == "50" and request.url.params["pair"] == "LTC/EUR"
        return httpx.Response(200, json={str(start): page1, page1["result"]["last"]: page2}[since])

    fake.routes["Trades"] = trades
    clock.set(start + 2 * 86_400)  # the window is far older than the 720 OHLC minutes
    frames = await provider.fetch_minutes(["LTC-EUR"], start_ts=start, end_ts=end)
    assert served == [str(start), page1["result"]["last"]]  # 'since' = previous 'last' (inclusive cursor)
    rows = {r[6]: r for r in [*page1["result"]["LTC/EUR"], *page2["result"]["LTC/EUR"]]}  # overlap counted once
    volume = sum(float(r[1]) for r in rows.values() if start <= r[2] // 60 * 60 < end)
    frame = frames["LTC-EUR"]
    assert frame["volume"].sum() == pytest.approx(volume)
    assert frame.index.min() >= start and frame.index.max() < end
    assert not fake.of("OHLC")


async def test_trades_pagination_stops_when_kraken_has_caught_up(
    provider: KrakenProvider, fake: FakeKraken, clock: ManualClock
) -> None:
    page = fixture("trades_ltc_eur_page1.json")  # 50 trades < 1000: the newest trades were reached
    fake.routes["Trades"] = lambda r: httpx.Response(200, json=page)
    clock.set(1_795_382_760 + 2 * 86_400)
    frames = await provider.fetch_minutes(["LTC-EUR"], start_ts=1_795_382_760, end_ts=1_795_390_000)
    assert len(fake.of("Trades")) == 1 and not frames["LTC-EUR"].empty


async def test_a_page_that_does_not_advance_skips_one_nanosecond(
    provider: KrakenProvider, fake: FakeKraken, clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kraken, "TRADES_PAGE", 2)
    stuck = {"error": [], "result": {"LTC/EUR": [["36.5", "1", 100.5, "b", "m", "", 7], ["36.6", "1", 100.5, "b", "m", "", 8]],
                                     "last": "100500000000"}}  # fmt: skip
    done = {"error": [], "result": {"LTC/EUR": [["36.7", "1", 200.0, "s", "l", "", 9]], "last": "200000000000"}}
    fake.routes["Trades"] = lambda r: httpx.Response(200, json=stuck if r.url.params["since"] in ("60", "100500000000") else done)
    clock.set(10 * 86_400)
    await provider.fetch_minutes(["LTC-EUR"], start_ts=60, end_ts=300)
    assert [r.url.params["since"] for r in fake.of("Trades")] == ["60", "100500000000", "100500000001"]


async def test_rate_limit_trips_the_breaker_and_later_calls_make_no_request(
    provider: KrakenProvider, fake: FakeKraken, clock: ManualClock
) -> None:
    fake.routes["Ticker"] = lambda r: httpx.Response(200, json={"error": ["EGeneral:Too many requests"]})
    with pytest.raises(RateLimitError):
        await provider.fetch_quotes(["BTC-EUR"])
    paused = provider.paused_until()
    assert paused is not None and paused == clock.ts() + 60
    calls = len(fake.requests)
    with pytest.raises(RateLimitError):
        await provider.fetch_quotes(["BTC-EUR"])
    assert len(fake.requests) == calls
    clock.set(paused + 1)
    fake.routes["Ticker"] = lambda r: httpx.Response(429, text="slow down")
    with pytest.raises(RateLimitError):
        await provider.fetch_quotes(["BTC-EUR"])
    assert provider.paused_until() == clock.ts() + 120  # doubled after a repeated limit
    clock.set(clock.ts() + 121)
    fake.routes["Ticker"] = lambda r: httpx.Response(200, json=fixture("ticker.json"))
    assert len(await provider.fetch_quotes(["BTC-EUR"])) == 1
    assert provider.paused_until() is None


async def test_one_failing_pair_does_not_fail_the_others(provider: KrakenProvider, fake: FakeKraken) -> None:
    def ohlc(request: httpx.Request) -> httpx.Response:
        if request.url.params["pair"] == "DOT/EUR":
            return httpx.Response(200, json={"error": ["EQuery:Unknown asset pair"]})
        return httpx.Response(200, json=fixture("ohlc_ltc_eur_1.json"))

    fake.routes["OHLC"] = ohlc
    frames = await provider.fetch_minutes(["DOT-EUR", "LTC-EUR"], start_ts=LTC_FORMING - 600)
    assert set(frames) == {"LTC-EUR"}
    with pytest.raises(ProviderError, match="DOT-EUR"):
        await provider.fetch_minutes(["DOT-EUR"], start_ts=LTC_FORMING - 600)


async def test_network_errors_and_bad_answers_are_provider_errors(provider: KrakenProvider, fake: FakeKraken) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    fake.routes["Ticker"] = broken
    with pytest.raises(ProviderError, match="ConnectError"):
        await provider.fetch_quotes(["BTC-EUR"])
    fake.routes["Ticker"] = lambda r: httpx.Response(502, text="<html>bad gateway</html>")
    with pytest.raises(ProviderError, match="no JSON"):
        await provider.fetch_quotes(["BTC-EUR"])
    fake.routes["Ticker"] = lambda r: httpx.Response(200, json={"error": []})
    with pytest.raises(ProviderError, match="without result"):
        await provider.fetch_quotes(["BTC-EUR"])
    assert provider.paused_until() is None  # none of these is a rate limit


async def test_lookup_returns_the_default_instrument_of_a_listed_pair(provider: KrakenProvider, fake: FakeKraken) -> None:
    instrument = await provider.lookup("ltc-eur")
    assert instrument is not None and (instrument.symbol, instrument.exchange, instrument.name) == (
        "LTC-EUR", "KRAKEN", "Litecoin"
    )
    assert await provider.lookup("BTC-USD") is None
    fake.routes["AssetPairs"] = lambda r: ok({})  # Kraken stopped listing every pair
    fresh = KrakenProvider(clock=provider._clock, transport=httpx.MockTransport(fake), limiter=no_wait_limiter())  # noqa: SLF001
    assert await fresh.lookup("LTC-EUR") is None
    await fresh.close()


async def test_failed_pair_validation_keeps_the_static_table(provider: KrakenProvider, fake: FakeKraken) -> None:
    fake.routes["AssetPairs"] = lambda r: httpx.Response(200, json={"error": ["EService:Unavailable"]})
    assert len(await provider.fetch_quotes(["BTC-EUR"])) == 1
    assert provider.pairs() == dict(KRAKEN_PAIRS)


async def test_a_pair_kraken_no_longer_lists_is_left_out_of_the_ticker_batch(
    provider: KrakenProvider, fake: FakeKraken
) -> None:
    """Kraken refuses a multi-pair request as a whole when ONE pair is unknown (checked live with 'NOPE/EUR'):
    a delisted pair must not cost the quotes of the other nine."""
    assert len(await provider.fetch_quotes(list(KRAKEN_PAIRS))) == len(KRAKEN_PAIRS)  # validated, all listed
    unknown = {"error": ["EQuery:Unknown asset pair"]}
    pairs = fixture("asset_pairs.json")["result"]
    listed = {key: info for key, info in pairs.items() if info["altname"] != "DOTEUR"}  # full list without DOT

    def asset_pairs(request: httpx.Request) -> httpx.Response:
        if "pair" not in request.url.params:
            return ok(listed)
        return httpx.Response(200, json=unknown if "DOT/EUR" in request.url.params["pair"] else ok(listed).json())

    def ticker(request: httpx.Request) -> httpx.Response:
        if "DOT/EUR" in request.url.params["pair"]:
            return httpx.Response(200, json=unknown)
        return httpx.Response(200, json=fixture("ticker.json"))

    fake.routes["AssetPairs"], fake.routes["Ticker"] = asset_pairs, ticker
    with pytest.raises(ProviderError, match="Unknown asset pair"):  # the delisting shows up in the batch ...
        await provider.fetch_quotes(list(KRAKEN_PAIRS))
    quotes = await provider.fetch_quotes(list(KRAKEN_PAIRS))  # ... and the next call finds the pair that is gone
    assert {q.symbol for q in quotes} == set(KRAKEN_PAIRS) - {"DOT-EUR"}
    assert provider.pair("DOT-EUR").status == "unknown"  # type: ignore[union-attr]
    assert "DOT/EUR" not in fake.of("Ticker")[-1].url.params["pair"]
    assert [("pair" in r.url.params) for r in fake.of("AssetPairs")] == [True, True, False]  # one full list read
    assert await provider.lookup("DOT-EUR") is None
    assert len(await provider.fetch_quotes(list(KRAKEN_PAIRS))) == len(KRAKEN_PAIRS) - 1
    assert len(fake.of("AssetPairs")) == 3  # validated again only after a day


async def test_closed_provider_refuses_requests(provider: KrakenProvider) -> None:
    await provider.close()
    with pytest.raises(ProviderError, match="closed"):
        await provider.fetch_quotes(["BTC-EUR"])


# ------------------------------------------------------------------------------------------------
# Rate limiter
# ------------------------------------------------------------------------------------------------
async def test_rate_limiter_paces_requests_and_lets_live_work_overtake_the_backfill() -> None:
    now = [0.0]
    order: list[str] = []

    async def sleep(seconds: float) -> None:
        now[0] += seconds
        await asyncio.sleep(0)

    limiter = AsyncRateLimiter(rate=1.0, capacity=1.0, monotonic=lambda: now[0], sleep=sleep)

    async def take(name: str, background: bool) -> None:
        await limiter.acquire(background=background)
        order.append(f"{name}@{now[0]:.0f}")

    await asyncio.gather(take("history1", True), take("history2", True), take("quotes", False))
    # the first history page had the token; the waiting quote request overtakes the second history page
    assert [entry.split("@")[0] for entry in order] == ["history1", "quotes", "history2"]
    times = [float(entry.split("@")[1]) for entry in order]
    assert times == sorted(times) and times[1] - times[0] >= 1 and times[2] - times[1] >= 1  # 1 request per second


def test_rate_limiter_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        AsyncRateLimiter(rate=0)
    with pytest.raises(ValueError):
        AsyncRateLimiter(rate=1, capacity=0.5)


def test_frames_are_canonical_for_the_service() -> None:
    frame = ohlc_minute_frame(fixture("ohlc_ltc_eur_1.json")["result"]["LTC/EUR"], LTC_LAST)
    assert isinstance(frame, pd.DataFrame) and frame.index.is_monotonic_increasing and frame.index.is_unique
