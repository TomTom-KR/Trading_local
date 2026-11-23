"""Paper trading: the AI portfolio ('ai', KI) versus the user's portfolio ('me', Ich).

Public entry points used by the runtime wiring:

* ``build_fx``     -> ``contracts.FxRates`` from the FX pairs of the universe
* ``build_broker`` -> ``contracts.Broker`` for both portfolios (identical rules, costs and fills)
"""

from trading_local.clock import Clock
from trading_local.config import CostParams, Settings
from trading_local.contracts import Broker, FxRates, MarketCalendar, MarketData
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.sim.broker import PaperBroker
from trading_local.sim.fx import FxTable


def build_fx(*, db: Database, instruments: InstrumentBook, data: MarketData, base_currency: str) -> FxRates:
    """Currency converter backed by the quote book of ``data`` and the stored FX bars."""
    return FxTable(db=db, instruments=instruments, data=data, base_currency=base_currency)


def build_broker(
    *,
    settings: Settings,
    db: Database,
    clock: Clock,
    bus: EventBus,
    calendar: MarketCalendar,
    instruments: InstrumentBook,
    runtime_settings: SettingsStore,
    costs: CostParams,
    data: MarketData,
    fx: FxRates,
) -> Broker:
    """Paper broker for both portfolios; creates them (with ``settings.start_capital``) if they are missing."""
    return PaperBroker(
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


__all__ = ["FxTable", "PaperBroker", "build_broker", "build_fx"]
