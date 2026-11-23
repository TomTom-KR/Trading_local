"""A mail whose SMTP transaction has started is recorded even if the caller is cancelled (RUN-2)."""

import asyncio
from collections.abc import Callable

import pytest

from trading_local.alerts.mailer import AlertMailer
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.models import Signal

from .conftest import INSTRUMENTS, SMTP_SETTINGS, SmtpRecorder


async def _until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.01)


async def test_a_mail_cancelled_during_the_smtp_transaction_is_still_recorded(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database
) -> None:
    smtp.send_delay_s = 0.3  # the SMTP DATA phase takes a while
    mailer = make_mailer(SMTP_SETTINGS)
    signal = make_signal()

    caller = asyncio.create_task(mailer.send_signal(signal, INSTRUMENTS["NVDA"]))
    await _until(lambda: any("send_message" in c.names() for c in smtp.connections))
    caller.cancel()  # e.g. the engine stops while the transaction runs
    with pytest.raises(asyncio.CancelledError):
        await caller

    await _until(lambda: repo_ops.count_mails(seeded_db, since_ts=0) == 1)
    assert len(smtp.sent) == 1  # delivered once ...
    stored = repo_signals.get_signal(seeded_db, signal.id)
    assert stored is not None and stored.mailed_ts is not None  # ... and recorded, so its exit mail will follow
