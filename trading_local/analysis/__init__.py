"""Signal engine: indicators, features, setups, scoring, calibration, tracking, readings and validation.

Entry point for the runtime: ``build_engine``.
"""

from trading_local.analysis.engine import SignalEngine
from trading_local.analysis.movers import MoveWatch
from trading_local.clock import Clock
from trading_local.config import CostParams, Settings
from trading_local.contracts import Broker, ChartRenderer, Engine, Llm, Mailer, MarketCalendar, News, PriceTrigger
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus


def build_engine(
    *,
    settings: Settings,
    db: Database,
    clock: Clock,
    bus: EventBus,
    calendar: MarketCalendar,
    instruments: InstrumentBook,
    runtime_settings: SettingsStore,
    costs: CostParams,
    llm: Llm,
    charts: ChartRenderer,
    mailer: Mailer,
    broker: Broker,
) -> Engine:
    """The platform's signal engine (``SignalEngine``, version ``ENGINE_PARAMS.version``).

    Construction does no I/O; the worker thread starts with the first minute cycle.
    """
    return SignalEngine(
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


def build_price_trigger(
    *,
    db: Database,
    clock: Clock,
    bus: EventBus,
    calendar: MarketCalendar,
    instruments: InstrumentBook,
    runtime_settings: SettingsStore,
    news: News,
) -> PriceTrigger:
    """The price trigger (``movers.MoveWatch``): unusual moves with volume pull the instrument's news forward.

    Construction does no I/O; the worker thread starts with the first minute cycle.
    """
    return MoveWatch(
        db=db,
        clock=clock,
        bus=bus,
        calendar=calendar,
        instruments=instruments,
        runtime_settings=runtime_settings,
        news=news,
    )


__all__ = ["MoveWatch", "SignalEngine", "build_engine", "build_price_trigger"]
