"""Fixtures of the trading-calendar tests."""

import pytest

from trading_local.markets import ExchangeCalendar


@pytest.fixture(scope="session")
def calendar() -> ExchangeCalendar:
    """One calendar for the whole session: every venue is built only once (XKRX takes about 2 s)."""
    return ExchangeCalendar()
