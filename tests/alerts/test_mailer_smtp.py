"""SMTP mode against a fake SMTP class (and once against a local aiosmtpd server on 127.0.0.1)."""

import asyncio
import base64
import email
import email.policy
import logging
import smtplib
import socket
import ssl
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from trading_local.alerts import mailer as mailer_module
from trading_local.alerts.mailer import AUTH_FAILED_DE, AlertMailer, classify_smtp_error
from trading_local.clock import ManualClock
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.models import Signal

from .conftest import INSTRUMENTS, PASSWORD, SMTP_SETTINGS, SMTP_USER, UMLAUT_PASSWORD, FakeSleep, SmtpRecorder

STARTTLS_SETTINGS = {**SMTP_SETTINGS, "smtp_security": "starttls", "smtp_port": 587}
SECRET_FORMS = (
    PASSWORD,
    base64.b64encode(PASSWORD.encode()).decode(),
    base64.b64encode(f"\0{SMTP_USER}\0{PASSWORD}".encode()).decode(),
)


def statuses(db: Database) -> list[str]:
    return [entry.status for entry in reversed(repo_ops.list_mail_log(db))]


@pytest.fixture
def smtp_mailer(make_mailer: Callable[..., AlertMailer]) -> AlertMailer:
    return make_mailer(SMTP_SETTINGS)


# ------------------------------------------------------------------------------------------------
# Connection paths
# ------------------------------------------------------------------------------------------------
async def test_ssl_path(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    clock: ManualClock,
) -> None:
    signal = make_signal()
    assert await smtp_mailer.send_signal(signal, INSTRUMENTS["NVDA"], b"\x89PNG\r\n\x1a\nx") is True

    [connection] = smtp.connections
    assert (connection.kind, connection.host, connection.port) == ("ssl", "mail.example.invalid", 465)
    assert connection.kwargs["timeout"] == 20.0
    assert isinstance(connection.kwargs["context"], ssl.SSLContext)
    assert connection.kwargs["context"].verify_mode == ssl.CERT_REQUIRED
    assert connection.kwargs["local_hostname"] == "trading.local"
    assert connection.names() == ["ehlo", "login", "send_message", "quit"]
    assert connection.calls[1] == ("login", (SMTP_USER, PASSWORD))
    assert connection.calls[2] == ("send_message", (SMTP_USER, ["empfaenger@example.invalid"]))
    [message] = smtp.sent
    assert message["Subject"].startswith("LONG NVDA 142,35 USD")

    [entry] = repo_ops.list_mail_log(seeded_db)
    assert (entry.status, entry.kind, entry.signal_id) == ("sent", "signal", signal.id)
    stored = repo_signals.get_signal(seeded_db, signal.id or 0)
    assert stored is not None and stored.mailed_ts == clock.ts()
    [activity] = repo_ops.list_activity(seeded_db, category="mail")
    assert activity.message.startswith("E-Mail verschickt: LONG NVDA")


async def test_starttls_path(
    make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder, make_signal: Callable[..., Signal]
) -> None:
    assert await make_mailer(STARTTLS_SETTINGS).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    [connection] = smtp.connections
    assert (connection.kind, connection.port) == ("plain", 587)
    assert "context" not in connection.kwargs and connection.kwargs["timeout"] == 20.0
    assert connection.names() == ["ehlo", "starttls", "ehlo", "login", "send_message", "quit"]
    assert isinstance(connection.calls[1][1][0], ssl.SSLContext)


# ------------------------------------------------------------------------------------------------
# Retries and failures
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "step",
    [
        ("connect", ConnectionRefusedError(10061, "refused")),
        ("connect", TimeoutError("timed out")),
        ("send", smtplib.SMTPDataError(451, b"4.3.0 Temporary failure")),
        ("send", smtplib.SMTPServerDisconnected("Connection unexpectedly closed")),
    ],
)
async def test_one_retry_after_a_transient_failure(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, fake_sleep: FakeSleep, make_signal: Callable[..., Signal],
    seeded_db: Database, step: tuple[str, BaseException],
) -> None:
    smtp.plan = [step, None]
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    assert len(smtp.connections) == 2
    assert fake_sleep.calls == [5.0]  # the retry pause; no pacing before the first mail
    assert statuses(seeded_db) == ["sent"]


async def test_transient_failure_twice(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database
) -> None:
    smtp.plan = [("connect", TimeoutError()), ("connect", TimeoutError())]
    signal = make_signal()
    assert await smtp_mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is False
    assert len(smtp.connections) == 2
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert entry.status == "failed"
    assert entry.detail == (
        "Keine Verbindung zum Mailserver mail.example.invalid:465 (Zeitüberschreitung). "
        "Auch der zweite Versuch ist fehlgeschlagen."
    )
    assert signal.mailed_ts is None
    assert smtp_mailer.status().last_error == entry.detail


async def test_authentication_failure_is_not_retried_and_pauses_smtp(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, fake_sleep: FakeSleep, make_signal: Callable[..., Signal],
    seeded_db: Database, clock: ManualClock,
) -> None:
    smtp.plan = [("login", smtplib.SMTPAuthenticationError(535, b"Authentication credentials invalid"))]
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert len(smtp.connections) == 1
    assert smtp.connections[0].names() == ["ehlo", "login", "quit"]
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert entry.detail == f"{AUTH_FAILED_DE} SMTP-Versand pausiert bis 16:30 Uhr."
    [activity] = repo_ops.list_activity(seeded_db, category="mail")
    assert activity.level == "error" and AUTH_FAILED_DE in activity.message
    assert smtp_mailer.status().last_error.startswith("SMTP-Versand pausiert bis 16:30 Uhr nach dem Fehler: Anmeldung")

    # while paused: no connection at all
    clock.advance(29 * 60)
    assert await smtp_mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is False
    assert len(smtp.connections) == 1
    assert repo_ops.list_mail_log(seeded_db, limit=1)[0].detail.startswith("SMTP-Versand pausiert bis 16:30 Uhr")

    # after the pause sending resumes and the error disappears from the status
    clock.advance(61)
    assert await smtp_mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is True
    assert statuses(seeded_db) == ["failed", "skipped", "sent"]
    assert smtp_mailer.status().last_error == ""
    assert fake_sleep.calls == []  # the 30 minutes passed on their own: no pacing wait needed


@pytest.mark.parametrize(
    ("error", "pauses", "detail_start"),
    [
        (smtplib.SMTPSenderRefused(550, b"Sender address is not allowed", SMTP_USER), True,
         "Der Mailserver hat die Absenderadresse abgelehnt (550: Sender address is not allowed)."),
        (smtplib.SMTPDataError(554, b"Transaction failed / Reject due to policy restrictions"), True,
         "Der Mailserver hat die Nachricht abgelehnt (554: Transaction failed / Reject due to policy restrictions)"),
        (smtplib.SMTPRecipientsRefused({"empfaenger@example.invalid": (550, b"Mailbox unavailable")}), True,
         "Der Mailserver hat den Empfänger abgelehnt (550)."),
        (smtplib.SMTPDataError(552, b"Message size exceeds limit"), False,
         "Der Mailserver hat die Nachricht als zu groß abgelehnt (552: Message size exceeds limit)."),
    ],
)
async def test_permanent_rejections_are_not_retried(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    error: Exception, pauses: bool, detail_start: str,
) -> None:
    smtp.plan = [("send", error)]
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert len(smtp.connections) == 1
    detail = repo_ops.list_mail_log(seeded_db)[0].detail
    assert detail.startswith(detail_start)
    assert ("SMTP-Versand pausiert bis" in detail) is pauses
    assert bool(smtp_mailer.status().last_error.startswith("SMTP-Versand pausiert")) is pauses


def test_error_classification() -> None:
    endpoint = "mail.gmx.net:465"
    assert classify_smtp_error(socket.gaierror(11001, "getaddrinfo failed"), endpoint).detail_de == (
        "Keine Verbindung zum Mailserver mail.gmx.net:465 (Servername nicht auflösbar)."
    )
    tls = classify_smtp_error(ssl.SSLCertVerificationError("certificate verify failed"), endpoint)
    assert not tls.transient and "TLS/Zertifikat" in tls.detail_de
    unsupported = classify_smtp_error(smtplib.SMTPNotSupportedError("STARTTLS extension not supported"), endpoint)
    assert not unsupported.transient and "ssl = Port 465, starttls = Port 587" in unsupported.detail_de
    greeting = classify_smtp_error(smtplib.SMTPConnectError(421, b"Service not available"), endpoint)
    assert greeting.transient and not greeting.pauses
    app_password = smtplib.SMTPAuthenticationError(534, b"Application-specific password required")
    auth = classify_smtp_error(app_password, endpoint)
    assert (auth.detail_de, auth.transient, auth.pauses) == (AUTH_FAILED_DE, False, True)
    odd = classify_smtp_error(ValueError("boom"), endpoint)
    assert not odd.transient and "ValueError" in odd.detail_de


# ------------------------------------------------------------------------------------------------
# Provider protection: pacing, daily cap
# ------------------------------------------------------------------------------------------------
async def test_pacing_keeps_twenty_seconds_between_smtp_mails(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, fake_sleep: FakeSleep, make_signal: Callable[..., Signal],
    clock: ManualClock,
) -> None:
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    clock.advance(5)
    assert await smtp_mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"]) is True
    assert fake_sleep.calls == [pytest.approx(15.0)]
    clock.advance(60)
    assert await smtp_mailer.send_test() is not None
    assert fake_sleep.calls == [pytest.approx(15.0)]  # a minute later: no wait
    assert len(smtp.sent) == 3


async def test_file_mode_is_not_paced(
    make_mailer: Callable[..., AlertMailer], fake_sleep: FakeSleep, make_signal: Callable[..., Signal],
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_symbol_cooldown_min": 0}, clock.ts())
    mailer = make_mailer()
    for _ in range(3):
        assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    assert fake_sleep.calls == []


async def test_daily_smtp_cap(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    store: SettingsStore, clock: ManualClock,
) -> None:
    store.update({"mail_max_per_hour": 120}, clock.ts())
    for hour in range(40):  # 40 SMTP mails within the last 24 hours, but none in the last hour
        repo_ops.log_mail(seeded_db, ts=clock.ts() - 3700 - hour * 1000, kind="signal",
                          recipient="x", subject="alt", status="sent")
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert smtp.connections == []
    assert repo_ops.list_mail_log(seeded_db, limit=1)[0].detail.startswith(
        "Tageslimit erreicht: 40 E-Mails über SMTP in den letzten 24 Stunden (höchstens 40"
    )
    test_result = await smtp_mailer.send_test()  # the test mail ignores the limits
    assert (test_result.ok, test_result.status) == (True, "sent")


async def test_concurrent_sends_are_serialised_in_call_order(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, fake_sleep: FakeSleep, make_signal: Callable[..., Signal],
    seeded_db: Database,
) -> None:
    smtp.send_delay_s = 0.03
    symbols = ["NVDA", "SAP.DE", "SHEL.L", "7203.T"]
    signals = [make_signal(symbol=symbol, ref_price=100.0) for symbol in symbols]
    results = await asyncio.gather(*(smtp_mailer.send_signal(s, INSTRUMENTS[s.symbol]) for s in signals))
    assert results == [True] * 4
    assert smtp.max_active == 1  # one SMTP connection at a time
    assert [message["Subject"].split()[1] for message in smtp.sent] == symbols
    assert fake_sleep.calls == [pytest.approx(20.0)] * 3
    assert statuses(seeded_db) == ["sent"] * 4


async def test_mails_do_not_block_the_event_loop(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal]
) -> None:
    smtp.send_delay_s = 0.15
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True
    finally:
        task.cancel()
    assert ticks >= 5


# ------------------------------------------------------------------------------------------------
# send_test in SMTP mode
# ------------------------------------------------------------------------------------------------
async def test_send_test_via_smtp(smtp_mailer: AlertMailer, smtp: SmtpRecorder, seeded_db: Database) -> None:
    result = await smtp_mailer.send_test()
    assert (result.ok, result.status) == (True, "sent")
    assert result.detail_de == (
        "Testmail an e***@example.invalid über mail.example.invalid:465 (SSL) verschickt. "
        "Bitte Posteingang und Spam-Ordner prüfen."
    )
    assert smtp.sent[0]["Subject"] == "Testmail von Trading Local"
    assert "SMTP über mail.example.invalid:465 (SSL)" in smtp.sent[0].get_body(preferencelist=("plain",)).get_content()
    assert repo_ops.list_mail_log(seeded_db)[0].kind == "test"


async def test_send_test_failure_and_pause_lift(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database
) -> None:
    smtp.plan = [("login", smtplib.SMTPAuthenticationError(535, b"invalid")), None]
    failed = await smtp_mailer.send_test()
    assert (failed.ok, failed.status) == (False, "failed")
    assert failed.detail_de.startswith(f"Die Testmail konnte nicht verschickt werden: {AUTH_FAILED_DE}")
    assert smtp_mailer.status().last_error.startswith("SMTP-Versand pausiert")
    # the user fixed the mailbox setting and tests again: the test bypasses the pause and lifts it
    ok = await smtp_mailer.send_test()
    assert ok.ok
    assert smtp_mailer.status().last_error == ""
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is True


async def test_unconfigured_smtp(make_mailer: Callable[..., AlertMailer], smtp: SmtpRecorder) -> None:
    mailer = make_mailer({**SMTP_SETTINGS, "smtp_password": "", "smtp_user": ""})
    result = await mailer.send_test()
    assert (result.ok, result.status) == (False, "skipped")
    assert result.detail_de == (
        "Der E-Mail-Versand ist nicht vollständig eingerichtet: TL_SMTP_USER, TL_SMTP_PASSWORD fehlen. "
        "Es wurde keine Testmail erzeugt."
    )
    assert smtp.connections == []
    assert mailer.status().configured is False


# ------------------------------------------------------------------------------------------------
# Secrets and robustness
# ------------------------------------------------------------------------------------------------
async def test_password_never_leaks(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    smtp.plan = [
        ("login", smtplib.SMTPAuthenticationError(535, f"bad credentials {SECRET_FORMS[2]} {PASSWORD}".encode())),
        ("send", smtplib.SMTPDataError(451, f"echo {PASSWORD}".encode())),
        ("send", smtplib.SMTPDataError(451, f"echo {SECRET_FORMS[1]}".encode())),
    ]
    results = [await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]), await smtp_mailer.send_test()]
    assert results[0] is False
    texts = [record.getMessage() + (record.exc_text or "") for record in caplog.records]
    texts += [entry.detail for entry in repo_ops.list_mail_log(seeded_db)]
    texts += [entry.message for entry in repo_ops.list_activity(seeded_db)]
    texts += [smtp_mailer.status().model_dump_json(), results[1].model_dump_json(), repr(smtp_mailer.status())]
    assert texts
    for text in texts:
        for secret in SECRET_FORMS:
            assert secret not in text


async def test_mail_addresses_never_leave_the_mailer_in_full(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SMTP errors repeat the sender / recipient (exception texts, refused-recipient dicts, server replies such as
    '553 5.7.1 <addr>: Sender address rejected'); log file, mail_log, activity, status and test reply mask them."""
    caplog.set_level(logging.DEBUG)
    recipient = SMTP_SETTINGS["mail_to"]
    smtp.plan = [
        ("send", smtplib.SMTPSenderRefused(
            553, f"5.7.1 <{SMTP_USER.upper()}>: Sender address rejected: not owned by user".encode(), SMTP_USER
        )),
        ("send", smtplib.SMTPRecipientsRefused({recipient: (550, f"5.1.1 <{recipient}>: mailbox unavailable".encode())})),
    ]
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    test_reply = await smtp_mailer.send_test()
    assert test_reply.ok is False
    refused = {recipient: (550, f"<{recipient}> unknown".encode())}
    monkeypatch.setattr(smtp_mailer, "_smtp_transaction", lambda msg: refused)  # accepted, one recipient refused
    assert (await smtp_mailer.send_test()).ok is True

    texts = [record.getMessage() for record in caplog.records]
    texts += [entry.detail for entry in repo_ops.list_mail_log(seeded_db)]
    texts += [entry.message for entry in repo_ops.list_activity(seeded_db)]
    texts += [smtp_mailer.status().model_dump_json(), test_reply.model_dump_json()]
    for text in texts:
        assert SMTP_USER not in text.lower() and recipient not in text.lower(), text
    joined = "\n".join(texts)
    assert "<A***@EXAMPLE.INVALID>: Sender address rejected" in joined  # the reply stays readable, the address not
    assert "refused some recipients: {'e***@example.invalid'" in joined


async def test_unexpected_errors_never_raise(
    smtp_mailer: AlertMailer, make_signal: Callable[..., Signal], store: SettingsStore,
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    def broken() -> Any:
        raise RuntimeError(f"settings exploded {PASSWORD}")

    monkeypatch.setattr(store, "get", broken)
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert await smtp_mailer.send_exit(make_signal(mailed_ts=1), INSTRUMENTS["NVDA"]) is False
    result = await smtp_mailer.send_test()
    assert (result.ok, result.status) == (False, "failed")
    assert caplog.text.count("unexpected error") == 3 and "RuntimeError" in caplog.text
    assert PASSWORD not in caplog.text


async def test_unexpected_error_while_composing_is_recorded(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("header injection rejected")

    monkeypatch.setattr(mailer_module, "compose_message", broken)
    assert await smtp_mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert smtp.connections == []
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert entry.status == "failed" and entry.detail.startswith("Unerwarteter Fehler beim Erstellen oder Versenden")


async def test_an_already_mailed_signal_is_not_mailed_again(
    smtp_mailer: AlertMailer, smtp: SmtpRecorder, make_signal: Callable[..., Signal], seeded_db: Database
) -> None:
    signal = make_signal()
    assert await smtp_mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is True
    assert await smtp_mailer.send_signal(signal, INSTRUMENTS["NVDA"]) is False
    assert len(smtp.sent) == 1
    assert statuses(seeded_db) == ["sent"]


# ------------------------------------------------------------------------------------------------
# End to end against a local SMTP server (aiosmtpd on 127.0.0.1, real smtplib, real AUTH exchange)
# ------------------------------------------------------------------------------------------------
class _Sink:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[str], bytes]] = []

    async def handle_DATA(self, server: Any, session: Any, envelope: Any) -> str:  # noqa: N802 - aiosmtpd hook
        self.messages.append((envelope.mail_from, list(envelope.rcpt_tos), envelope.content))
        return "250 Message accepted"


@pytest.fixture
def smtp_server() -> Iterator[tuple[_Sink, int, list[tuple[bytes, bytes]]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # aiosmtpd 1.4.6: typing.ByteString on Python 3.15
        from aiosmtpd.controller import Controller
        from aiosmtpd.smtp import AuthResult

    logins: list[tuple[bytes, bytes]] = []

    def authenticate(server: Any, session: Any, envelope: Any, mechanism: str, auth_data: Any) -> Any:
        logins.append((auth_data.login, auth_data.password))
        accepted = auth_data.password in (PASSWORD.encode(), UMLAUT_PASSWORD.encode())
        return AuthResult(success=accepted, handled=False)  # handled=False: aiosmtpd answers 535 itself

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    sink = _Sink()
    controller = Controller(
        sink, hostname="127.0.0.1", port=port, server_hostname="sink.local", authenticator=authenticate,
        auth_require_tls=False,
    )
    controller.start()
    try:
        yield sink, port, logins
    finally:
        controller.stop()


class _LoopbackSMTP(smtplib.SMTP):
    """Real smtplib client whose STARTTLS is a no-op (the loopback test server has no certificate)."""

    def starttls(self, *args: Any, **kwargs: Any) -> tuple[int, bytes]:
        return 220, b"loopback: no TLS"


async def test_end_to_end_with_a_local_smtp_server(
    make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], seeded_db: Database,
    smtp_server: tuple[_Sink, int, list[tuple[bytes, bytes]]],
) -> None:
    sink, port, logins = smtp_server
    mailer = make_mailer(
        {**STARTTLS_SETTINGS, "smtp_host": "127.0.0.1", "smtp_port": port}, smtp_factory=_LoopbackSMTP
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"], png) is True

    assert logins == [(SMTP_USER.encode(), PASSWORD.encode())]
    [(mail_from, recipients, raw)] = sink.messages
    assert (mail_from, recipients) == (SMTP_USER, ["empfaenger@example.invalid"])
    message = email.message_from_bytes(raw, policy=email.policy.default)
    assert message["Subject"] == "LONG NVDA 142,35 USD | Stopp 140,90 | Ziel 145,25 | Konfidenz 58 %"
    images = [part.get_content() for part in message.walk() if part.get_content_type() == "image/png"]
    assert images == [png, png]
    assert statuses(seeded_db) == ["sent"]



@pytest.mark.parametrize(
    ("password", "accepted"), [(UMLAUT_PASSWORD, True), ("falsch", False), ("fälsch", False)]
)
async def test_local_smtp_server_login_variants(
    make_mailer: Callable[..., AlertMailer], smtp_server: tuple[_Sink, int, list[tuple[bytes, bytes]]],
    password: str, accepted: bool,
) -> None:
    """Non-ASCII passwords go through the UTF-8 AUTH PLAIN fallback; rejected logins are reported in German."""
    sink, port, logins = smtp_server
    mailer = make_mailer(
        {**STARTTLS_SETTINGS, "smtp_host": "127.0.0.1", "smtp_port": port, "smtp_password": password},
        smtp_factory=_LoopbackSMTP,
    )
    result = await mailer.send_test()
    # after a rejection smtplib tries its next mechanism (PLAIN, then LOGIN) within the same connection
    assert set(logins) == {(SMTP_USER.encode(), password.encode())}
    assert result.ok is accepted
    assert len(sink.messages) == (1 if accepted else 0)
    if not accepted:
        assert result.status == "failed" and AUTH_FAILED_DE in result.detail_de
        assert password not in result.detail_de
