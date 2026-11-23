"""Failure injection, boundaries and restart behaviour of the mailer (review tests)."""

import asyncio
import logging
import smtplib
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from trading_local.alerts import mailer as mailer_module
from trading_local.alerts import templates
from trading_local.alerts.mailer import AUTH_FAILED_DE, AlertMailer
from trading_local.clock import ManualClock
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.models import Signal

from .conftest import INSTRUMENTS, SMTP_SETTINGS, SMTP_USER, UMLAUT_PASSWORD, FakeSleep, SmtpRecorder


def statuses(db: Database) -> list[str]:
    return [entry.status for entry in reversed(repo_ops.list_mail_log(db))]


def log_mail(db: Database, ts: int, count: int = 1, **fields: Any) -> None:
    values: dict[str, Any] = {"kind": "signal", "recipient": "x@example.invalid", "subject": "alt", "status": "sent"}
    values.update(fields)
    for _ in range(count):
        repo_ops.log_mail(db, ts=ts, **values)


# ------------------------------------------------------------------------------------------------
# A delivered mail stays delivered
# ------------------------------------------------------------------------------------------------
async def test_database_error_after_delivery_still_reports_the_mail_as_sent(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    mailer = make_mailer(SMTP_SETTINGS)
    signal = make_signal()

    def locked(*args: Any, **kwargs: Any) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repo_ops, "log_mail", locked)
    assert await mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is True  # the mail did go out
    assert signal.mailed_ts == clock.ts()  # so the engine's next update_signal persists it
    assert "database is locked" in caplog.text
    assert await mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is False  # and it is never mailed twice
    assert len(smtp.sent) == 1


async def test_concurrent_calls_for_the_same_signal_send_one_mail(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock, seeded_db: Database,
) -> None:
    store.update({"mail_symbol_cooldown_min": 0}, clock.ts())
    mailer = make_mailer(SMTP_SETTINGS)
    signal = make_signal()
    results = await asyncio.gather(*(mailer.send_signal(signal, INSTRUMENTS["NVDA"]) for _ in range(3)))
    assert sorted(results) == [False, False, True]
    assert len(smtp.sent) == 1
    assert statuses(seeded_db) == ["sent"]


async def test_template_error_is_recorded_as_a_failed_attempt(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], seeded_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("FLAT")

    monkeypatch.setattr(mailer_module, "entry_mail", broken)
    signal = make_signal()
    assert await make_mailer().send_signal(signal, INSTRUMENTS["NVDA"]) is False
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert (entry.status, entry.kind, entry.signal_id) == ("failed", "signal", signal.id)
    assert entry.subject == "Einstiegssignal NVDA"
    assert entry.detail.startswith("Unerwarteter Fehler")
    [activity] = repo_ops.list_activity(seeded_db, category="mail")
    assert activity.level == "error"


async def test_header_injection_through_the_symbol_is_refused(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], seeded_db: Database,
    smtp: SmtpRecorder,
) -> None:
    signal = make_signal()
    signal.symbol = "NVDA\r\nBcc: evil@example.invalid"
    assert await make_mailer(SMTP_SETTINGS).send_signal(signal, INSTRUMENTS["NVDA"]) is False
    assert smtp.connections == []
    assert repo_ops.list_mail_log(seeded_db)[0].status == "failed"


# ------------------------------------------------------------------------------------------------
# Secrets
# ------------------------------------------------------------------------------------------------
async def test_escaped_forms_of_a_non_ascii_password_never_leak(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    seeded_db: Database, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    escaped = (
        repr(UMLAUT_PASSWORD.encode())[2:-1],  # b'G\xc3\xa4hnende...' in the str() of an SMTP exception
        UMLAUT_PASSWORD.encode("unicode_escape").decode(),
    )
    mailer = make_mailer({**SMTP_SETTINGS, "smtp_password": UMLAUT_PASSWORD})
    smtp.plan = [("connect", smtplib.SMTPConnectError(554, f"echo {UMLAUT_PASSWORD}".encode()))]
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert "SMTPConnectError" in caplog.text  # the failure was reported (and its server text redacted)
    texts = [record.getMessage() for record in caplog.records]
    texts += [entry.detail for entry in repo_ops.list_mail_log(seeded_db)]
    texts += [entry.message for entry in repo_ops.list_activity(seeded_db)]
    texts.append(mailer.status().last_error)
    for text in texts:
        assert UMLAUT_PASSWORD not in text
        for form in escaped:
            assert form not in text


# ------------------------------------------------------------------------------------------------
# Restart behaviour
# ------------------------------------------------------------------------------------------------
async def test_the_smtp_pause_survives_a_restart(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    seeded_db: Database, clock: ManualClock,
) -> None:
    smtp.plan = [("login", smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid"))]
    assert await make_mailer(SMTP_SETTINGS).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    clock.advance(10 * 60)

    restarted = make_mailer(SMTP_SETTINGS)  # a new process: nothing in memory
    assert restarted.status().last_error.startswith(AUTH_FAILED_DE)
    assert await restarted.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is False
    assert len(smtp.connections) == 1  # no second login attempt
    skipped = repo_ops.list_mail_log(seeded_db, limit=1)[0]
    assert skipped.status == "skipped"
    assert skipped.detail == f"SMTP-Versand pausiert bis 16:30 Uhr nach dem Fehler: {AUTH_FAILED_DE}"

    clock.advance(20 * 60 + 1)  # 30 minutes after the rejection
    assert await restarted.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is True
    assert statuses(seeded_db) == ["failed", "skipped", "sent"]


async def test_an_old_or_lifted_pause_is_not_restored(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    seeded_db: Database, clock: ManualClock,
) -> None:
    detail = f"{AUTH_FAILED_DE} SMTP-Versand pausiert bis 16:00 Uhr."
    log_mail(seeded_db, clock.ts() - 31 * 60, status="failed", detail=detail)
    assert await make_mailer(SMTP_SETTINGS).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True

    log_mail(seeded_db, clock.ts() - 60, status="failed", detail=detail)
    log_mail(seeded_db, clock.ts() - 30, kind="test", status="sent")  # the user's test mail lifted it
    assert await make_mailer(SMTP_SETTINGS).send_signal(
        make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]
    ) is True
    assert len(smtp.sent) == 2


# ------------------------------------------------------------------------------------------------
# Boundaries
# ------------------------------------------------------------------------------------------------
async def test_hourly_window_includes_exactly_one_hour(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], store: SettingsStore,
    seeded_db: Database, clock: ManualClock,
) -> None:
    store.update({"mail_max_per_hour": 1}, clock.ts())
    log_mail(seeded_db, clock.ts() - 3600, status="file")
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    clock.advance(1)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True


async def test_symbol_cooldown_ends_after_exactly_the_configured_minutes(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], clock: ManualClock,
) -> None:
    mailer = make_mailer()
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    clock.advance(30 * 60 - 1)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    clock.advance(1)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True


async def test_pacing_boundary_and_daily_cap_boundary(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, fake_sleep: FakeSleep,
    make_signal: Callable[..., Signal], store: SettingsStore, seeded_db: Database, clock: ManualClock,
) -> None:
    store.update({"mail_max_per_hour": 120, "mail_symbol_cooldown_min": 0}, clock.ts())
    log_mail(seeded_db, clock.ts() - 7200, 38)  # 38 SMTP mails earlier today
    mailer = make_mailer(SMTP_SETTINGS)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True  # 39th
    clock.advance(20)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True  # 40th, exactly 20 s later
    assert fake_sleep.calls == []
    clock.advance(60)
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert repo_ops.list_mail_log(seeded_db, limit=1)[0].detail.startswith("Tageslimit erreicht: 40 E-Mails")
    assert len(smtp.sent) == 2


async def test_transient_recipient_refusal_is_retried_once(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal],
    seeded_db: Database,
) -> None:
    refused = smtplib.SMTPRecipientsRefused({"empfaenger@example.invalid": (450, b"Mailbox busy")})
    smtp.plan = [("send", refused), None]
    assert await make_mailer(SMTP_SETTINGS).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    assert len(smtp.connections) == 2
    assert statuses(seeded_db) == ["sent"]
    assert smtp.connections[1].calls[1] == ("login", (SMTP_USER, SMTP_SETTINGS["smtp_password"]))


# ------------------------------------------------------------------------------------------------
# Exit texts
# ------------------------------------------------------------------------------------------------
def test_cancellation_reasons_are_told_apart() -> None:
    base: dict[str, Any] = {
        "id": 7, "created_ts": 1_763_650_800, "data_ts": 1_763_650_800, "symbol": "NVDA", "direction": "LONG",
        "setup": "trend_pullback", "timeframe": "15m", "horizon_min": 390, "ref_price": 142.35, "atr": 1.2,
        "stop_dist": 1.45, "target_dist": 2.9, "score": 0.6, "confidence": 0.55, "engine_version": "t",
        "status": "cancelled", "mailed_ts": 1_763_650_800,
    }
    drift = templates.exit_mail(
        Signal(**base, status_reason="kurs_entlaufen"), INSTRUMENTS["NVDA"], timezone="Europe/Berlin",
        gui_base="http://127.0.0.1:8000",
    )
    assert drift.subject == "ENTFÄLLT NVDA (Long): Kurs vor dem Einstieg davongelaufen"
    assert "Einstiegsfrist" not in drift.text
    no_entry = templates.exit_mail(
        Signal(**base, status_reason="kein_einstieg"), INSTRUMENTS["NVDA"], timezone="Europe/Berlin",
        gui_base="http://127.0.0.1:8000",
    )
    assert no_entry.subject == "ENTFÄLLT NVDA (Long): kein Einstieg zustande gekommen"
    assert "Einstiegsfrist" in no_entry.text


def test_exit_note_matches_how_the_result_is_computed() -> None:
    """Entry and exit prices of a signal already include the simulated slippage (execution.py), so the
    percentage shown next to the R result is not 'before costs'."""
    assert "vor Kosten" not in templates.R_NOTE_DE
    assert "Slippage" in templates.R_NOTE_DE


async def test_mark_mailed_is_not_written_for_skipped_or_failed_mails(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], smtp: SmtpRecorder,
    seeded_db: Database,
) -> None:
    smtp.plan = [("send", smtplib.SMTPDataError(552, b"too big"))]
    signal = make_signal()
    assert await make_mailer(SMTP_SETTINGS).send_signal(signal, INSTRUMENTS["NVDA"]) is False
    stored = repo_signals.get_signal(seeded_db, signal.id or 0)
    assert stored is not None and stored.mailed_ts is None and signal.mailed_ts is None
