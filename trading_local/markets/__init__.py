"""Trading calendar: sessions, lunch breaks, holidays and early closes of all venues, and the session overview."""

from trading_local.contracts import MarketCalendar
from trading_local.markets.calendar import ExchangeCalendar
from trading_local.markets.overview import sessions_overview


def build_calendar() -> MarketCalendar:
    """The platform's trading calendar. Construction is cheap: each venue is built on its first query."""
    return ExchangeCalendar()


__all__ = ["ExchangeCalendar", "build_calendar", "sessions_overview"]
