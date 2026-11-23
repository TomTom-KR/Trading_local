"""Server-side candlestick charts (PNG) for the vision LLM and the alert mails.

Entry point for the runtime: ``build_chart_renderer``. Drawing and the renderer service live in ``render``.
"""

from trading_local.charts.render import ChartRenderer
from trading_local.clock import Clock
from trading_local.contracts import ChartRenderer as ChartRendererProtocol
from trading_local.contracts import MarketCalendar
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook


def build_chart_renderer(
    *,
    db: Database,
    clock: Clock,
    calendar: MarketCalendar,
    instruments: InstrumentBook,
    timezone: str = "Europe/Berlin",
) -> ChartRendererProtocol:
    """The platform's chart renderer; ``timezone`` is the user's zone for axis labels and the title time.

    Construction does no I/O; the worker thread starts with the first render.
    """
    return ChartRenderer(db=db, clock=clock, calendar=calendar, instruments=instruments, timezone=timezone)


__all__ = ["ChartRenderer", "build_chart_renderer"]
