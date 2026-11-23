"""Builds the service container: creates every component through its module factory and wires them."""

import logging

from trading_local.clock import Clock, SystemClock
from trading_local.config import DEFAULT_COSTS, Settings
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.runtime.scheduler import Runtime
from trading_local.services import Services

log = logging.getLogger(__name__)


def build_services(settings: Settings, *, clock: Clock | None = None) -> Services:
    """Open the database, migrate the instrument universe (seeds it on first start) and construct all components.

    Nothing is started here: ``Services.runtime.start()`` (called from the web app's lifespan) starts the
    background work. Construction never touches the network.
    """
    # Imported here so that importing trading_local.runtime stays cheap (pandas, matplotlib, yfinance ...).
    from trading_local.alerts import build_mailer
    from trading_local.analysis import build_engine
    from trading_local.charts import build_chart_renderer
    from trading_local.data import build_market_data
    from trading_local.llm import build_llm
    from trading_local.markets import build_calendar
    from trading_local.news import build_news
    from trading_local.sim import build_broker, build_fx
    from trading_local.universe import migrate_universe, seed_default_universe

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = Database(settings.db_path)
    db.initialize()
    clock = clock or SystemClock()
    bus = EventBus()

    # Bring the instruments table to the current default universe (versioned, a no-op once applied).
    changes = migrate_universe(db, clock=clock)
    if changes:
        message = " ".join(changes)
        log.info("instrument universe: %s", message)
        repo_ops.log_activity(db, ts=clock.ts(), level="info", category="system", message=message)

    instruments = InstrumentBook(db)
    if not instruments.all(active_only=False):
        written = seed_default_universe(db)
        instruments.reload()
        log.info("seeded the default universe (%d instruments)", written)
    runtime_settings = SettingsStore(db)
    costs = DEFAULT_COSTS

    calendar = build_calendar()
    data = build_market_data(
        settings=settings, db=db, clock=clock, bus=bus, calendar=calendar, instruments=instruments
    )
    fx = build_fx(db=db, instruments=instruments, data=data, base_currency=settings.base_currency)
    llm = build_llm(settings=settings, db=db, clock=clock, runtime_settings=runtime_settings)
    news = build_news(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        instruments=instruments,
        runtime_settings=runtime_settings,
        llm=llm,
    )
    charts = build_chart_renderer(
        db=db, clock=clock, calendar=calendar, instruments=instruments, timezone=settings.timezone
    )
    mailer = build_mailer(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        runtime_settings=runtime_settings,
        instruments=instruments,
    )
    broker = build_broker(
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
    engine = build_engine(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        calendar=calendar,
        instruments=instruments,
        runtime_settings=runtime_settings,
        costs=costs,
        llm=llm,
        charts=charts,
        mailer=mailer,
        broker=broker,
    )
    runtime = Runtime(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        data=data,
        broker=broker,
        engine=engine,
        news=news,
        llm=llm,
        calendar=calendar,
    )
    return Services(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        instruments=instruments,
        runtime_settings=runtime_settings,
        costs=costs,
        calendar=calendar,
        data=data,
        fx=fx,
        llm=llm,
        news=news,
        charts=charts,
        mailer=mailer,
        broker=broker,
        engine=engine,
        runtime=runtime,
    )
