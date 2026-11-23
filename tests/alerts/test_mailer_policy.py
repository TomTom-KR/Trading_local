"""Sending policy of send_signal / send_exit (file mode: no pacing, fast)."""

from collections.abc import Callable
from typing import Any

import pytest

from trading_local.alerts.mailer import AlertMailer
from trading_local.clock import ManualClock
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.models import Signal

from .conftest import INSTRUMENTS


def statuses(db: Database) -> list[tuple[str, str]]:
    """(kind, status) of all mail-log entries, oldest first."""
    return [(entry.kind, entry.status) for entry in reversed(repo_ops.list_mail_log(db))]


def last_detail(db: Database) -> str:
    return repo_ops.list_mail_log(db, limit=1)[0].detail


def log_delivered(db: Database, ts: int, count: int, **fields: Any) -> None:
    values: dict[str, Any] = {"kind": "signal", "recipient": "x@example.invalid", "subject": "alt", "status": "file"}
    values.update(fields)
    for _ in range(count):
        repo_ops.log_mail(db, ts=ts, **values)


def closed_copy(signal: Signal) -> Signal:
    """The signal after the tracker closed it at its target."""
    signal.status = "target"
    signal.entry_ts, signal.entry_price = signal.created_ts + 60, signal.ref_price
    signal.exit_ts, signal.exit_price, signal.outcome_r = signal.created_ts + 3600, signal.ref_price + 2.9, 1.9
    return signal


async def test_mails_disabled(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_enabled": False}, clock.ts())
    signal = make_signal()
    assert await make_mailer().send_signal(signal, INSTRUMENTS["NVDA"]) is False
    assert statuses(seeded_db) == [("signal", "skipped")]
    assert last_detail(seeded_db) == "E-Mail-Benachrichtigungen sind in den Einstellungen ausgeschaltet."
    stored = repo_signals.get_signal(seeded_db, signal.id or 0)
    assert stored is not None and stored.mailed_ts is None and signal.mailed_ts is None


async def test_region_not_selected(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_regions": ["EU", "ASIA"]}, clock.ts())
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert last_detail(seeded_db) == "E-Mails für die Region USA sind in den Einstellungen nicht ausgewählt."
    assert await mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is True


async def test_hourly_cap(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_max_per_hour": 3}, clock.ts())
    log_delivered(seeded_db, clock.ts() - 3601, 5)  # older than one hour: not counted
    log_delivered(seeded_db, clock.ts() - 600, 2, status="sent", kind="test")
    log_delivered(seeded_db, clock.ts() - 300, 4, status="skipped")  # not delivered: not counted
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True  # third mail this hour
    assert await mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is False
    assert last_detail(seeded_db) == "Stündliches E-Mail-Limit erreicht: 3 E-Mails in der letzten Stunde (erlaubt: 3)."


async def test_hourly_cap_zero_disables_all_mails(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    signal = make_signal(mailed_ts=clock.ts() - 600)
    store.update({"mail_max_per_hour": 0}, clock.ts())
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert await mailer.send_exit(closed_copy(signal), INSTRUMENTS["NVDA"]) is False
    assert statuses(seeded_db) == [("signal", "skipped"), ("exit", "skipped")]
    assert last_detail(seeded_db).startswith("Das stündliche E-Mail-Limit steht auf 0")


async def test_symbol_cooldown(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    clock: ManualClock,
) -> None:
    mailer = make_mailer()  # default cooldown: 30 minutes
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    clock.advance(10 * 60)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert last_detail(seeded_db) == (
        "Für NVDA wurde vor 10 Minuten bereits eine E-Mail verschickt (Sperrfrist 30 Minuten)."
    )
    assert await mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is True
    clock.advance(20 * 60)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    assert statuses(seeded_db) == [("signal", "file"), ("signal", "skipped"), ("signal", "file"), ("signal", "file")]


async def test_cooldown_zero_allows_repeated_mails(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], store: SettingsStore,
    clock: ManualClock,
) -> None:
    store.update({"mail_symbol_cooldown_min": 0}, clock.ts())
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True


async def test_exit_needs_a_mailed_entry(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal]
) -> None:
    assert await make_mailer().send_exit(closed_copy(make_signal()), INSTRUMENTS["NVDA"]) is False
    assert statuses(seeded_db) == []  # nothing to follow up: no attempt


async def test_exit_respects_mail_on_exit(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_on_exit": False}, clock.ts())
    signal = make_signal(mailed_ts=clock.ts() - 600)
    assert await make_mailer().send_exit(closed_copy(signal), INSTRUMENTS["NVDA"]) is False
    assert last_detail(seeded_db) == "Ausstiegs-E-Mails sind in den Einstellungen ausgeschaltet."


async def test_exit_is_exempt_from_cooldown_and_region_filter(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    mailer = make_mailer()
    signal = make_signal()
    assert await mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is True
    store.update({"mail_regions": ["EU"]}, clock.ts())
    clock.advance(60)
    assert await mailer.send_exit(closed_copy(signal), INSTRUMENTS["NVDA"]) is True
    assert statuses(seeded_db) == [("signal", "file"), ("exit", "file")]
    exit_entry = repo_ops.list_mail_log(seeded_db, limit=1)[0]
    assert exit_entry.subject == "SCHLIESSEN NVDA (Long): Ziel erreicht, +1,9 R"
    assert exit_entry.signal_id == signal.id
    stored = repo_signals.get_signal(seeded_db, signal.id or 0)
    assert stored is not None and stored.mailed_ts == clock.ts() - 60  # the exit does not touch mailed_ts


@pytest.mark.parametrize(("limit", "exit_allowance"), [(4, 6), (1, 1), (12, 18)])
async def test_exits_may_exceed_the_hourly_cap_by_half(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock, limit: int, exit_allowance: int,
) -> None:
    store.update({"mail_max_per_hour": limit}, clock.ts())
    log_delivered(seeded_db, clock.ts() - 60, limit)
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is False
    results = []
    for _ in range(exit_allowance - limit + 1):
        signal = closed_copy(make_signal(mailed_ts=clock.ts() - 3000))
        results.append(await mailer.send_exit(signal, INSTRUMENTS["NVDA"]))
    assert results == [True] * (exit_allowance - limit) + [False]
    assert last_detail(seeded_db) == (
        f"Stündliches E-Mail-Limit erreicht: {exit_allowance} E-Mails in der letzten Stunde "
        f"(erlaubt: {exit_allowance})."
    )
