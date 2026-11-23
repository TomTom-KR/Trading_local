"""E-mail alerts: entry and exit recommendations, test mails (modes smtp | file | off).

Entry point for the runtime: ``build_mailer``. The content lives in ``templates`` (pure functions) and
``formatting`` (German number and date helpers); delivery, policy and pacing in ``mailer``.
"""

from trading_local.alerts.mailer import AlertMailer
from trading_local.clock import Clock
from trading_local.config import Settings
from trading_local.contracts import Mailer
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus


def build_mailer(
    *,
    settings: Settings,
    db: Database,
    clock: Clock,
    bus: EventBus,
    runtime_settings: SettingsStore,
    instruments: InstrumentBook,
) -> Mailer:
    """The platform's ``Mailer`` (see ``AlertMailer``). Construction does no I/O.

    ``instruments`` is part of the uniform wiring signature; the mailer itself receives the instrument with
    every ``send_*`` call and does not need the book.
    """
    del instruments
    return AlertMailer(settings=settings, db=db, clock=clock, bus=bus, runtime_settings=runtime_settings)


__all__ = ["AlertMailer", "build_mailer"]
