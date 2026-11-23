"""Fixtures for the paper-trading tests (on top of settings, db, clock and bus from tests/conftest.py)."""

from collections.abc import Callable

import pytest

from trading_local.clock import ManualClock
from trading_local.config import CostParams, Settings
from trading_local.contracts import Broker
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.sim import build_broker
from trading_local.sim.fx import FxTable
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, seed_instruments
from tests.sim.simkit import FUTURE, FX_QUOTES, T0, FakeMarketData, RecordingBus


@pytest.fixture
def bus() -> RecordingBus:
    return RecordingBus()


@pytest.fixture
def instruments(db: Database) -> InstrumentBook:
    seed_instruments(db, [*SAMPLE_INSTRUMENTS, FUTURE])
    return InstrumentBook(db)


@pytest.fixture
def data() -> FakeMarketData:
    market = FakeMarketData()
    for symbol, price in FX_QUOTES.items():
        market.set_quote(symbol, price, T0)
    return market


@pytest.fixture
def calendar() -> FakeCalendar:
    return FakeCalendar()


@pytest.fixture
def runtime_settings(db: Database) -> SettingsStore:
    return SettingsStore(db)


@pytest.fixture
def costs() -> CostParams:
    return CostParams()


@pytest.fixture
def fx(db: Database, instruments: InstrumentBook, data: FakeMarketData) -> FxTable:
    # No cache: the tests change FX quotes between two calls.
    return FxTable(db=db, instruments=instruments, data=data, base_currency="EUR", cache_ttl_s=0.0)


@pytest.fixture
def make_broker(
    settings: Settings,
    db: Database,
    clock: ManualClock,
    bus: RecordingBus,
    calendar: FakeCalendar,
    instruments: InstrumentBook,
    runtime_settings: SettingsStore,
    costs: CostParams,
    data: FakeMarketData,
    fx: FxTable,
) -> Callable[[], Broker]:
    """Builds a broker on the shared database (a second call simulates a restart)."""

    def make() -> Broker:
        return build_broker(
            settings=settings,
            db=db,
            clock=clock,
            bus=bus,
            calendar=calendar,
            instruments=instruments,
            runtime_settings=runtime_settings,
            costs=costs,
            data=data,
            fx=fx,
        )

    return make


@pytest.fixture
def broker(make_broker: Callable[[], Broker]) -> Broker:
    return make_broker()
