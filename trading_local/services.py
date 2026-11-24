"""The service container: one object that holds every wired component of the running platform.

The runtime (``trading_local.runtime``) builds it once at start-up; the web layer and the background
jobs receive it and talk to the components only through the protocols in ``trading_local.contracts``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from trading_local.clock import Clock
from trading_local.config import CostParams, Settings
from trading_local.contracts import (
    Broker,
    ChartRenderer,
    Derivatives,
    Engine,
    FxRates,
    Llm,
    Mailer,
    MarketCalendar,
    MarketData,
    News,
)
from trading_local.db.database import Database
from trading_local.dto import JobStatusOut
from trading_local.events import EventBus

if TYPE_CHECKING:
    from trading_local.db.repo_market import InstrumentBook
    from trading_local.db.repo_ops import SettingsStore


@runtime_checkable
class RuntimeInfo(Protocol):
    """What the web layer may know about the background runtime."""

    started_ts: int
    offline: bool  # True: background jobs are disabled, the platform only serves the existing database

    def jobs(self) -> list[JobStatusOut]:
        """Latest state of every background job (for the system page)."""
        ...


@dataclass(slots=True)
class Services:
    settings: Settings
    db: Database
    clock: Clock
    bus: EventBus
    instruments: "InstrumentBook"
    runtime_settings: "SettingsStore"
    costs: CostParams
    calendar: MarketCalendar
    data: MarketData
    fx: FxRates
    llm: Llm
    news: News
    charts: ChartRenderer
    mailer: Mailer
    broker: Broker
    engine: Engine
    runtime: RuntimeInfo
    derivatives: Derivatives | None = None
