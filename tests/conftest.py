"""Fixtures shared by all test packages."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.db.database import Database
from trading_local.events import EventBus
from trading_local.testing import make_settings

# Friday 2026-11-20 15:00 UTC: New York (10:00) and Xetra (16:00) are in their regular sessions.
FRIDAY_SESSION = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def db(settings: Settings) -> Iterator[Database]:
    database = Database(settings.db_path)
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(FRIDAY_SESSION)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture(autouse=True)
def _reset_live_since() -> Iterator[None]:
    """The runtime records the start of live data in a process-wide setting; never let it leak between tests."""
    from trading_local import barstore

    yield
    barstore.set_live_since(None)
