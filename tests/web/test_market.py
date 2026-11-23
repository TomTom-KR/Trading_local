"""Quotes, sparklines and chart bars."""

import math
from typing import Any

import pytest
from fastapi.testclient import TestClient

from trading_local.clock import ManualClock
from trading_local.services import Services

from .conftest import FRIDAY, NOW_TS, THURSDAY, Fakes, at, quote, seed_daily, seed_market, seed_minutes


@pytest.fixture
def market(services: Services) -> Services:
    seed_market(services.db)
    return services


def test_quotes_come_from_memory(client: TestClient, fakes: Fakes) -> None:
    fakes.data.quote_book.update({"SAP.DE": quote("SAP.DE", 241.0, NOW_TS - 60), "NVDA": quote("NVDA", 181.0, NOW_TS)})
    body = client.get("/api/quotes").json()
    assert [(item["symbol"], item["price"]) for item in body] == [("NVDA", 181.0), ("SAP.DE", 241.0)]
    assert body[0]["change_pct"] == 1.0


# ------------------------------------------------------------------------------------------------
# Sparklines
# ------------------------------------------------------------------------------------------------
def test_sparklines_cover_the_current_or_last_session_in_5_minute_buckets(
    client: TestClient, market: Services
) -> None:
    series = client.get("/api/sparklines").json()["series"]
    assert set(series) == set(market.instruments.symbols())
    nvda = series["NVDA"]  # current session 14:30-15:00, the 15:00 bar is not closed yet
    assert [point[0] for point in nvda] == [at(FRIDAY, 14, 30) + 300 * i for i in range(6)]
    assert series["SAP.DE"][-1][0] == at(FRIDAY, 14, 55) and len(series["SAP.DE"]) == 80  # 84 buckets, newest 80
    assert series["^GSPC"][0][0] == at(THURSDAY, 14, 30) and len(series["^GSPC"]) == 78  # Friday has no bar yet
    tokyo = series["7203.T"]  # closed venue: its last completed session with the lunch break
    assert (len(tokyo), tokyo[0][0], tokyo[-1][0]) == (66, at(FRIDAY, 0), at(FRIDAY, 6, 25))
    assert (len(series["BTC-USD"]), series["BTC-USD"][0][0]) == (36, at(FRIDAY, 12))  # UTC day of a 24/7 venue
    assert series["EURUSD=X"] == []
    assert all(ts % 300 == 0 for points in series.values() for ts, _ in points)


def test_sparkline_points_are_the_last_close_of_each_bucket(client: TestClient, market: Services) -> None:
    frame = market.db.query("SELECT ts, close FROM bars_1m WHERE symbol = 'NVDA' AND ts = ?", (at(FRIDAY, 14, 59),))
    last_point = client.get("/api/sparklines").json()["series"]["NVDA"][-1]
    assert last_point == [at(FRIDAY, 14, 55), pytest.approx(frame[0]["close"])]


def test_sparklines_are_cached_for_60_seconds(client: TestClient, market: Services, clock: ManualClock) -> None:
    assert client.get("/api/sparklines").json()["series"]["EURUSD=X"] == []
    seed_minutes(market.db, "EURUSD=X", at(FRIDAY, 13), 60, price=1.08)
    clock.advance(59)
    assert client.get("/api/sparklines").json()["series"]["EURUSD=X"] == []
    clock.advance(1)
    assert len(client.get("/api/sparklines").json()["series"]["EURUSD=X"]) == 12


# ------------------------------------------------------------------------------------------------
# Bars
# ------------------------------------------------------------------------------------------------
def bars(client: TestClient, symbol: str = "NVDA", **params: Any) -> dict[str, Any]:
    response = client.get(f"/api/bars/{symbol}", params=params)
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


def assert_clean_overlays(body: dict[str, Any], names: set[str], *, filled: set[str] | None = None) -> None:
    """All overlays present, finite and on visible bars; the ``filled`` ones (default: all) not empty."""
    assert set(body["overlays"]) == names
    visible = {bar[0] for bar in body["bars"]}
    for name, line in body["overlays"].items():
        assert line or name not in (names if filled is None else filled), name
        assert all(math.isfinite(value) for _, value in line), name
        assert {ts for ts, _ in line} <= visible, name


def test_1m_bars_are_closed_bars_only_with_overlays(client: TestClient, market: Services) -> None:
    body = bars(client, tf="1m", limit=10)
    assert (body["symbol"], body["timeframe"], body["currency"], body["delay_min"]) == ("NVDA", "1m", "USD", 0)
    assert [bar[0] for bar in body["bars"]] == [at(FRIDAY, 14, 50) + 60 * i for i in range(10)]  # no 15:00 bar
    assert all(len(bar) == 6 for bar in body["bars"])
    assert_clean_overlays(body, {"ema20", "ema50", "vwap"})
    assert len(body["overlays"]["ema50"]) == 10  # warmed up on older bars: no gap in the visible part
    assert len(body["overlays"]["vwap"]) == 10


@pytest.mark.parametrize(
    ("tf", "step", "last", "filled"),
    [
        ("5m", 300, at(FRIDAY, 14, 55), {"ema20", "ema50", "vwap"}),
        ("15m", 900, at(FRIDAY, 14, 45), {"ema20", "vwap"}),  # 28 buckets of history: no EMA50 yet
        ("1h", 3600, at(FRIDAY, 14), {"vwap"}),  # 9 buckets of history: no EMA at all yet
    ],
)
def test_intraday_buckets_are_complete_and_aligned(
    client: TestClient, market: Services, tf: str, step: int, last: int, filled: set[str]
) -> None:
    body = bars(client, tf=tf, limit=20)
    stamps = [bar[0] for bar in body["bars"]]
    assert stamps[-1] == last and all(ts % step == 0 for ts in stamps)
    assert stamps == sorted(set(stamps))
    assert_clean_overlays(body, {"ema20", "ema50", "vwap"}, filled=filled)


def test_daily_bars_use_completed_sessions_and_the_date_at_midnight(client: TestClient, market: Services) -> None:
    seed_daily(market.db, "NVDA", FRIDAY.date(), 400)  # Friday's stored row must not be used: the session runs
    body = bars(client, tf="1d", limit=50)
    assert len(body["bars"]) == 50
    assert body["bars"][-2][0] == at(THURSDAY, 0)  # the newest completed session
    assert body["bars"][-1][0] == at(FRIDAY, 0)  # today's session, from its closed minutes (not the stored row)
    assert all(bar[0] % 86_400 == 0 for bar in body["bars"])
    assert_clean_overlays(body, {"ema20", "ema50", "ema200"})
    assert len(body["overlays"]["ema200"]) == 50


def minute_rows(market: Services, symbol: str, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
    return market.db.query(
        "SELECT ts, open, high, low, close, volume FROM bars_1m WHERE symbol = ? AND ts >= ? AND ts < ? ORDER BY ts",
        (symbol, start_ts, end_ts),
    )


def test_daily_series_ends_with_the_running_session_from_its_closed_minutes(client: TestClient, market: Services) -> None:
    seed_daily(market.db, "NVDA", THURSDAY.date(), 30)
    rows = minute_rows(market, "NVDA", at(FRIDAY, 0), NOW_TS - 59)  # the 15:00 bar is still open
    assert len(rows) == 30
    body = bars(client, tf="1d", limit=5)
    assert [bar[0] for bar in body["bars"]][-2:] == [at(THURSDAY, 0), at(FRIDAY, 0)]
    ts, o, h, low, c, v = body["bars"][-1]
    assert o == pytest.approx(rows[0]["open"]) and c == pytest.approx(rows[-1]["close"])
    assert h == pytest.approx(max(r["high"] for r in rows)) and low == pytest.approx(min(r["low"] for r in rows))
    assert v == pytest.approx(sum(r["volume"] for r in rows))
    assert body["overlays"]["ema20"][-1][0] == at(FRIDAY, 0)


def test_the_running_session_of_a_24h_market_starts_at_utc_midnight(client: TestClient, market: Services) -> None:
    seed_daily(market.db, "BTC-USD", THURSDAY.date(), 10)
    seed_minutes(market.db, "BTC-USD", at(THURSDAY, 23), 60, price=89_000.0)  # Thursday's UTC day: not today
    rows = minute_rows(market, "BTC-USD", at(FRIDAY, 0), NOW_TS - 59)
    body = bars(client, "BTC-USD", tf="1d", limit=3)
    assert body["bars"][-1][0] == at(FRIDAY, 0)
    assert body["bars"][-1][1] == pytest.approx(rows[0]["open"])  # opens with Friday's first bar (12:00 UTC)


def test_no_running_session_row_before_the_open_or_when_paging(
    client: TestClient, market: Services, clock: ManualClock
) -> None:
    seed_daily(market.db, "^GSPC", THURSDAY.date(), 10)  # ^GSPC has no Friday minute yet
    assert bars(client, "^GSPC", tf="1d", limit=3)["bars"][-1][0] == at(THURSDAY, 0)
    seed_daily(market.db, "NVDA", THURSDAY.date(), 10)
    page = bars(client, tf="1d", limit=3, before=NOW_TS)  # history pages carry completed sessions only
    assert page["bars"][-1][0] == at(THURSDAY, 0)
    clock.advance(7 * 3600)  # Friday 22:00 UTC: the session has ended, its daily row is not stored yet
    closed = bars(client, tf="1d", limit=3)
    assert closed["bars"][-1][0] == at(FRIDAY, 0)


def test_before_pages_back_without_overlap(client: TestClient, market: Services) -> None:
    newest = bars(client, tf="1m", limit=5)["bars"]
    older = bars(client, tf="1m", limit=5, before=newest[0][0])
    assert [bar[0] for bar in older["bars"]] == [newest[0][0] - 60 * (5 - i) for i in range(5)]
    seed_daily(market.db, "NVDA", THURSDAY.date(), 30)
    page = bars(client, tf="1d", limit=3, before=at(THURSDAY, 0))  # Thursday's session is not complete then
    assert [bar[0] for bar in page["bars"]] == [at(THURSDAY, 0) - 86_400 * i for i in (3, 2, 1)]


def test_the_oldest_partial_session_gets_no_vwap(client: TestClient, market: Services) -> None:
    body = bars(client, tf="1m", limit=60)  # 30 Thursday bars + 30 Friday bars, read cut off on Thursday
    vwap_stamps = [ts for ts, _ in body["overlays"]["vwap"]]
    assert vwap_stamps == [bar[0] for bar in body["bars"] if bar[0] >= at(FRIDAY, 14, 30)]


def test_symbols_without_bars_return_an_empty_series(client: TestClient, market: Services) -> None:
    body = bars(client, "EURUSD=X", tf="5m")
    assert body["bars"] == [] and all(line == [] for line in body["overlays"].values())


@pytest.mark.parametrize(
    ("params", "status", "fragment"),
    [
        ({"tf": "2m"}, 422, "Abfrageparameter 'tf' muss einer der Werte '1m', '5m', '15m', '1h' oder '1d' sein"),
        ({"limit": 0}, 422, "Abfrageparameter 'limit' muss mindestens 1 sein"),
        ({"limit": 2001}, 422, "Abfrageparameter 'limit' darf höchstens 2000 sein"),
        ({"before": -1}, 422, "Abfrageparameter 'before' muss mindestens 0 sein"),
    ],
)
def test_bar_parameters_are_validated(client: TestClient, params: dict[str, Any], status: int, fragment: str) -> None:
    response = client.get("/api/bars/NVDA", params=params)
    assert response.status_code == status
    assert fragment in response.json()["detail"]


def test_unknown_symbols_are_404(client: TestClient) -> None:
    response = client.get("/api/bars/NOPE")
    assert response.status_code == 404
    assert response.json()["detail"] == "Unbekanntes Symbol: NOPE"


def test_lower_case_symbols_are_accepted(client: TestClient, market: Services) -> None:
    assert bars(client, "sap.de", tf="15m", limit=3)["symbol"] == "SAP.DE"
