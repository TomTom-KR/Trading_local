import logging

import pytest

from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.models import Instrument
from trading_local.sim import build_fx
from trading_local.sim.fx import FxTable
from trading_local.testing import seed_instruments
from tests.sim.simkit import MIN, T0, FakeMarketData, add_bar, add_daily

USDCHF = Instrument("USDCHF=X", "USD/CHF", "fx", "GLOBAL", "FX", "CHF", tradable=False)


class FakeTimer:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def table(db: Database, instruments: InstrumentBook, data: FakeMarketData) -> FxTable:
    return FxTable(db=db, instruments=instruments, data=data, base_currency="EUR", cache_ttl_s=0.0)


def test_factory_builds_an_fx_table(db: Database, instruments: InstrumentBook, data: FakeMarketData) -> None:
    fx = build_fx(db=db, instruments=instruments, data=data, base_currency="EUR")
    assert fx.rate("USD", "EUR") == pytest.approx(0.8)


def test_same_currency_is_one(table: FxTable) -> None:
    assert table.rate("EUR", "EUR") == 1.0
    assert table.rate("XYZ", "XYZ", 123) == 1.0


def test_pair_in_both_orientations(table: FxTable) -> None:
    assert table.rate("USD", "EUR") == pytest.approx(1 / 1.25)  # EURUSD=X inverted
    assert table.rate("EUR", "USD") == pytest.approx(1.25)  # EURUSD=X direct
    assert table.rate("GBP", "EUR") == pytest.approx(1 / 0.8)
    assert table.rate("JPY", "EUR") == pytest.approx(1 / 160)


def test_cross_via_the_base_currency(table: FxTable) -> None:
    # JPY -> EUR -> USD (no JPY/USD pair exists)
    assert table.rate("JPY", "USD") == pytest.approx(1 / 160 * 1.25)


def test_cross_via_usd(db: Database, instruments: InstrumentBook, data: FakeMarketData, table: FxTable) -> None:
    seed_instruments(db, [USDCHF])
    instruments.reload()
    data.set_quote("USDCHF=X", 0.9, T0)
    # CHF -> USD -> EUR
    assert table.rate("CHF", "EUR") == pytest.approx(1 / 0.9 * 0.8)


def test_unknown_currency_and_pair_without_instrument(table: FxTable, data: FakeMarketData) -> None:
    assert table.rate("AUD", "EUR") is None
    data.set_quote("EURCAD=X", 1.5, T0)  # a quote alone does not make a pair: it must be an instrument
    assert table.rate("CAD", "EUR") is None


def test_latest_falls_back_to_bars_then_daily_closes(db: Database, table: FxTable, data: FakeMarketData) -> None:
    data.drop_quote("EURUSD=X")
    assert table.rate("USD", "EUR") is None
    add_daily(db, "EURUSD=X", "2026-11-19", 1.6)
    assert table.rate("USD", "EUR") == pytest.approx(1 / 1.6)
    add_bar(db, "EURUSD=X", T0, 1.0, close=2.0)
    assert table.rate("USD", "EUR") == pytest.approx(0.5)
    data.set_quote("EURUSD=X", 0.0, T0)  # unusable price: ignored
    assert table.rate("USD", "EUR") == pytest.approx(0.5)


def test_historical_rate_uses_only_bars_closed_at_that_time(db: Database, table: FxTable) -> None:
    add_bar(db, "EURUSD=X", T0 - 2 * MIN, 1.20, close=1.20)
    add_bar(db, "EURUSD=X", T0 - MIN, 1.25, close=1.25)
    add_bar(db, "EURUSD=X", T0, 1.30, close=1.30)  # closes at T0 + 60: unknown at T0
    assert table.rate("EUR", "USD", T0) == pytest.approx(1.25)
    assert table.rate("EUR", "USD", T0 + MIN) == pytest.approx(1.30)
    assert table.rate("EUR", "USD", T0 - MIN) == pytest.approx(1.20)


def test_historical_rate_falls_back_to_completed_daily_closes(db: Database, table: FxTable) -> None:
    add_daily(db, "EURUSD=X", "2026-11-18", 1.10)
    add_daily(db, "EURUSD=X", "2026-11-20", 1.40)  # the UTC day of T0 has not ended yet
    assert table.rate("EUR", "USD", T0) == pytest.approx(1.10)
    assert table.rate("EUR", "USD", T0 - 5 * 86_400) is None


def test_latest_rates_are_cached_briefly(db: Database, instruments: InstrumentBook, data: FakeMarketData) -> None:
    timer = FakeTimer()
    cached = FxTable(db=db, instruments=instruments, data=data, base_currency="EUR", cache_ttl_s=5.0, timer=timer)
    assert cached.rate("EUR", "USD") == pytest.approx(1.25)
    data.set_quote("EURUSD=X", 1.30, T0)
    timer.now += 4.0
    assert cached.rate("EUR", "USD") == pytest.approx(1.25)
    timer.now += 1.5
    assert cached.rate("EUR", "USD") == pytest.approx(1.30)


def test_errors_yield_none(
    table: FxTable, data: FakeMarketData, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(symbol: str) -> None:
        raise RuntimeError("quote book unavailable")

    monkeypatch.setattr(data, "quote", broken)
    with caplog.at_level(logging.ERROR, logger="trading_local.sim.fx"):
        assert table.rate("USD", "EUR") is None
    assert "could not be determined" in caplog.text
