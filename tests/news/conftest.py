"""Fixtures of the news tests."""

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import pytest

from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.news.feeds import Feed
from trading_local.news.service import NewsService
from trading_local.testing import seed_instruments

from tests.news.helpers import NOW, FakeLlm, FeedServer

type ServiceFactory = Callable[..., NewsService]


@pytest.fixture
def now_clock(clock: ManualClock) -> ManualClock:
    clock.set(NOW)
    return clock


@pytest.fixture
def book(db: Database) -> InstrumentBook:
    seed_instruments(db)
    return InstrumentBook(db)


@pytest.fixture
def server() -> FeedServer:
    return FeedServer()


@pytest.fixture
def llm(now_clock: ManualClock) -> FakeLlm:
    return FakeLlm(now_clock)


@pytest.fixture
def runtime_settings(db: Database) -> SettingsStore:
    return SettingsStore(db)


@pytest.fixture
async def make_service(
    settings: Settings,
    db: Database,
    now_clock: ManualClock,
    bus: EventBus,
    book: InstrumentBook,
    server: FeedServer,
    runtime_settings: SettingsStore,
) -> AsyncIterator[ServiceFactory]:
    """Builds ``NewsService`` objects on the test database whose HTTP client talks to ``server``; every service
    is stopped at teardown (before the database closes)."""
    created: list[NewsService] = []

    def factory(feeds: Sequence[Feed], llm: Any, *, service_settings: Settings | None = None) -> NewsService:
        service = NewsService(
            settings=service_settings or settings,
            db=db,
            clock=now_clock,
            bus=bus,
            instruments=book,
            runtime_settings=runtime_settings,
            llm=llm,
            feeds=feeds,
            client_factory=server.client,
            host_gap_s=0.0,
        )
        created.append(service)
        return service

    yield factory
    for service in created:
        if not service._closed:  # noqa: SLF001 - teardown of services a test did not stop itself
            await service.stop()
