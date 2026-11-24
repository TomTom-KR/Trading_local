"""Agenda: macro data, central-bank decisions and earnings dates with expectation, actual value and surprise.

Modules
* ``rules``    pure rules: published values, surprise, matching, which instruments an event concerns, entry block
* ``sources``  parsing of the ForexFactory calendar, the FinancialJuice actual-value headlines and Yahoo earnings
* ``service``  ``AgendaService``: the minute refresh that keeps the table ``agenda_events`` current

The runtime wires the platform through ``build_agenda``.
"""

from trading_local.clock import Clock
from trading_local.contracts import Agenda, MarketData
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus


def build_agenda(
    *, db: Database, clock: Clock, bus: EventBus, instruments: InstrumentBook, data: MarketData
) -> Agenda:
    """The agenda service (not started; the runtime calls ``start`` / ``refresh`` / ``stop``). Construction does
    not touch the network."""
    from trading_local.agenda.service import AgendaService  # imported here: the engine imports agenda.rules

    return AgendaService(db=db, clock=clock, bus=bus, instruments=instruments, data=data)


__all__ = ["build_agenda"]
