"""Market data: quotes and 1-minute / daily bars into the database.

Two sources: Kraken's public real-time data for the instruments of the pseudo venue KRAKEN, Yahoo Finance for
everything else. Public entry point for the runtime: ``build_market_data`` (returns a ``contracts.MarketData``).
"""

from trading_local.clock import Clock
from trading_local.config import Settings
from trading_local.contracts import MarketCalendar, MarketData
from trading_local.data.kraken import KRAKEN_PAIRS, KRAKEN_VENUES
from trading_local.data.provider import CandleProvider, Provider, ProviderError, RateLimitError, Tick, TickStream
from trading_local.data.service import (
    KRAKEN_OPTIONS,
    DataOptions,
    InstrumentView,
    Lane,
    MarketDataService,
    RoutedMarketData,
)
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus
from trading_local.models import Instrument


def is_kraken(instrument: Instrument) -> bool:
    """True for the instruments served by the Kraken provider (pseudo venue KRAKEN)."""
    return instrument.exchange in KRAKEN_VENUES


def is_yahoo(instrument: Instrument) -> bool:
    return not is_kraken(instrument)


def claims_kraken_symbol(symbol: str) -> bool:
    """A symbol that is not in the instruments table yet goes to Kraken if it names a known Kraken pair."""
    return symbol.strip().upper() in KRAKEN_PAIRS


def build_market_data(
    *,
    settings: Settings,
    db: Database,
    clock: Clock,
    bus: EventBus,
    calendar: MarketCalendar,
    instruments: InstrumentBook,
) -> MarketData:
    """Create the routed market-data service (Yahoo lane + Kraken lane). Touches neither network nor disk:
    connections and the yfinance cache directory (``<data_dir>/yfinance``) are opened on first use."""
    from trading_local.data.kraken import KrakenProvider
    from trading_local.data.yahoo import YahooProvider  # imports yfinance: only needed by the live platform

    common = {"settings": settings, "db": db, "clock": clock, "bus": bus, "calendar": calendar}
    yahoo = MarketDataService(
        **common,
        instruments=InstrumentView(instruments, is_yahoo),
        provider=YahooProvider(clock=clock, cache_dir=settings.data_dir / "yfinance"),
    )
    kraken = MarketDataService(
        **common,
        instruments=InstrumentView(instruments, is_kraken),
        provider=KrakenProvider(clock=clock),
        options=KRAKEN_OPTIONS,
    )
    return RoutedMarketData(
        [Lane("Yahoo", yahoo, is_yahoo), Lane("Kraken", kraken, is_kraken, claims_kraken_symbol)], instruments
    )


__all__ = [
    "CandleProvider",
    "DataOptions",
    "InstrumentView",
    "KRAKEN_OPTIONS",
    "Lane",
    "MarketDataService",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "RoutedMarketData",
    "Tick",
    "TickStream",
    "build_market_data",
    "claims_kraken_symbol",
    "is_kraken",
    "is_yahoo",
]
