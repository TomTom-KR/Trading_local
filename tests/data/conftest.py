"""Fixtures of the market-data tests: a service wired to a FakeProvider and the FakeCalendar."""

import logging
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.data.fakes import FakeProvider
from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.data.service import DataOptions, MarketDataService
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus
from trading_local.models import Instrument
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar, make_settings, seed_instruments

TEST_OPTIONS = DataOptions(rest_offset_s=0, backfill_pause_s=0)


@dataclass
class Env:
    service: MarketDataService
    provider: FakeProvider
    db: Database
    clock: ManualClock
    bus: EventBus
    calendar: FakeCalendar
    instruments: InstrumentBook


type EnvFactory = Callable[..., Env]


@pytest.fixture(autouse=True)
def restore_yfinance_logger() -> Iterator[None]:
    """A YahooProvider installs its capture handler on the process-wide 'yfinance' logger and stops its
    propagation (by design: once per process). Restore the logger after every test, so that the tests of
    other packages (logging set-up) find it untouched whatever the test order."""
    logger = logging.getLogger("yfinance")
    handlers, propagate, level = list(logger.handlers), logger.propagate, logger.level
    yield
    logger.handlers[:] = handlers
    logger.propagate = propagate
    logger.setLevel(level)


@pytest.fixture
def net_settings(tmp_path: Path) -> Settings:
    """Settings with network features on (the service talks only to the FakeProvider) but no stream."""
    return make_settings(tmp_path, offline=False, enable_stream=False)


@pytest.fixture
async def make_env(net_settings: Settings, db: Database, clock: ManualClock, bus: EventBus) -> AsyncIterator[EnvFactory]:
    services: list[MarketDataService] = []

    def build(
        instruments: Iterable[Instrument] = SAMPLE_INSTRUMENTS,
        *,
        settings: Settings | None = None,
        options: DataOptions | None = None,
    ) -> Env:
        seed_instruments(db, instruments)
        book = InstrumentBook(db)
        provider = FakeProvider(clock)
        calendar = FakeCalendar()
        service = MarketDataService(
            settings=settings or net_settings,
            db=db,
            clock=clock,
            bus=bus,
            calendar=calendar,
            instruments=book,
            provider=provider,
            options=options or TEST_OPTIONS,
        )
        services.append(service)
        return Env(service, provider, db, clock, bus, calendar, book)

    yield build
    for service in services:
        await service.stop()


@pytest.fixture
async def env(make_env: EnvFactory) -> Env:
    return make_env()
