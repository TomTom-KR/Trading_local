from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.contracts import MarketData
from trading_local.data import MarketDataService, RoutedMarketData, build_market_data
from trading_local.data.kraken import KrakenProvider
from trading_local.data.yahoo import YahooProvider
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus
from trading_local.models import Instrument
from trading_local.testing import FakeCalendar


async def test_factory_builds_the_service_without_network_or_disk(
    settings: Settings, db: Database, clock: ManualClock, bus: EventBus
) -> None:
    data = build_market_data(
        settings=settings, db=db, clock=clock, bus=bus, calendar=FakeCalendar(), instruments=InstrumentBook(db)
    )
    assert isinstance(data, MarketData) and isinstance(data, RoutedMarketData)
    assert not (settings.data_dir / "yfinance").exists()
    assert data.status().symbols_active == 0 and data.quotes() == {}
    await data.stop()


async def test_factory_routes_kraken_instruments_to_the_kraken_lane(
    settings: Settings, db: Database, clock: ManualClock, bus: EventBus
) -> None:
    data = build_market_data(
        settings=settings, db=db, clock=clock, bus=bus, calendar=FakeCalendar(), instruments=InstrumentBook(db)
    )
    assert isinstance(data, RoutedMarketData)
    yahoo, kraken = data.lanes
    assert (yahoo.name, kraken.name) == ("Yahoo", "Kraken")
    assert isinstance(yahoo.service, MarketDataService) and isinstance(kraken.service, MarketDataService)
    assert isinstance(yahoo.service._provider, YahooProvider)  # noqa: SLF001 - wiring check
    assert isinstance(kraken.service._provider, KrakenProvider)  # noqa: SLF001
    btc = Instrument("BTC-EUR", "Bitcoin", "crypto", "GLOBAL", "KRAKEN", "EUR")
    nvda = Instrument("NVDA", "NVIDIA", "equity", "US", "XNAS", "USD")
    assert kraken.owns(btc) and not kraken.owns(nvda) and yahoo.owns(nvda) and not yahoo.owns(btc)
    assert kraken.claims("btc-eur") and kraken.claims("LTC-EUR") and not kraken.claims("BTC-USD")
    await data.stop()
