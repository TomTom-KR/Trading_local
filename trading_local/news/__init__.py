"""News scraping and hand-off to the LLM.

Modules
* ``feeds``    registry of the polled RSS/Atom feeds (tier 1 and tier 2 of the news probe)
* ``fetcher``  polite conditional GET, size cap, backoff, robust parsing to ``NewsItem``
* ``matcher``  deterministic headline -> instrument / macro-index matching and near-duplicate detection
* ``service``  ``NewsService``: the poll round, storage, selection policy and background LLM analyses

The runtime wires the platform through ``build_news``.
"""

from trading_local.clock import Clock
from trading_local.config import Settings
from trading_local.contracts import Llm, News
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.news.service import NewsService


def build_news(
    *,
    settings: Settings,
    db: Database,
    clock: Clock,
    bus: EventBus,
    instruments: InstrumentBook,
    runtime_settings: SettingsStore,
    llm: Llm,
) -> News:
    """The news service of the platform (not started; the runtime calls ``start`` / ``poll`` / ``stop``).

    Construction does not touch the network or the database.
    """
    return NewsService(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        instruments=instruments,
        runtime_settings=runtime_settings,
        llm=llm,
    )


__all__ = ["NewsService", "build_news"]
