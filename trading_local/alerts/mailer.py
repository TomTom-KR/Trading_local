"""Alert mail delivery: sending policy, provider-friendly pacing, the modes smtp / file / off and the mail log.

One mail runs through four steps (``AlertMailer._run``). One asyncio lock serialises them, so that policy
decisions, pacing and SMTP connections of concurrent calls never interleave and mails leave in call order:

1. gate     mode and configuration; the user's preferences (mails enabled, regions, hourly cap, per-symbol
            cooldown, exit rules); in SMTP mode the provider protection: a 30-minute pause after a rejected
            login (any authentication failure) or a 550/554 reply - restored from mail_log after a restart -
            and at most 40 SMTP mails in 24 hours.
            Test mails skip the preferences and the provider protection (the user asked for them).
2. content  subject, plain text and HTML (``templates``) assembled into one MIME message: multipart/alternative,
            the chart embedded by Content-ID and attached as a file.
3. deliver  'file': the complete RFC 5322 message as .eml in the outbox; 'smtp': wait until 20 s have passed
            since the previous SMTP connection, then send; one retry after a short pause for transient
            errors (connection problems, 4xx replies), none for authentication failures and 5xx replies.
4. record   a mail_log row for every attempt (skipped mails and unexpected errors included), ``mark_mailed``
            for delivered entry mails, an activity-log entry (category 'mail') for every delivery and failure.
            A delivered mail counts as delivered even if recording it fails.

The pacing and the GMX-oriented limits follow the provider research: GMX publishes no limits, bursts get
accounts blocked, so the platform stays near one mail per minute and well below the reported thresholds.

Secrets: the SMTP password is used for the SMTP login and nowhere else. Every text that leaves this
module (log records, mail_log details, activity entries, the status, the test-mail reply) passes through the
redactor, which removes the password and masks the configured addresses.
"""

import asyncio
import base64
import logging
import re
import smtplib
import socket
import sqlite3
import ssl
import traceback
import uuid
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.message import EmailMessage
from email.policy import SMTP as SMTP_POLICY
from email.utils import format_datetime, formataddr, getaddresses
from pathlib import Path
from typing import Final, Literal
from zoneinfo import ZoneInfo

from trading_local.alerts.formatting import REGION_LABELS_DE, format_time
from trading_local.alerts.templates import MailContent, check_mail, entry_mail, exit_mail, gui_base_url
from trading_local.clock import Clock
from trading_local.config import RuntimeSettings, Settings
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.dto import MailStatusOut, MailTestOut
from trading_local.events import EventBus
from trading_local.models import Instrument, Signal
from trading_local.presenters import address_masker, mask_email

log = logging.getLogger(__name__)

SMTP_TIMEOUT_S: Final = 20.0
EHLO_NAME: Final = "trading.local"  # fixed EHLO name: no slow getfqdn(), reveals neither PC name nor LAN IP
MIN_SMTP_INTERVAL_S: Final = 20.0  # between two SMTP connections
DAILY_SMTP_CAP: Final = 40  # SMTP mails per rolling 24 hours
SMTP_PAUSE_S: Final = 30 * 60  # after a rejected login or message
RETRY_PAUSE_S: Final = 5.0  # before the single retry of a transient failure
PAUSING_REPLY_CODES: Final = frozenset({535, 550, 554})
HOUR_S: Final = 3600
DAY_S: Final = 86_400

AUTH_FAILED_DE: Final = (
    "Anmeldung am Mailserver fehlgeschlagen. Zugangsdaten prüfen und im Postfach den Versand über externe "
    "Programme (POP3/IMAP/SMTP) freischalten."
)
UNEXPECTED_DE: Final = "Unerwarteter Fehler beim Erstellen oder Versenden - Einzelheiten im Protokoll."
PAUSE_MARKER_DE: Final = "SMTP-Versand pausiert bis"  # starts the pause note; also found in mail_log details
KIND_DE: Final[dict[str, str]] = {"signal": "Einstiegssignal", "exit": "Ausstieg", "test": "Testmail"}

_FALLBACK_SENDER = "trading-local@localhost"  # file mode without any configured sender
_SENDER_NAME = "Trading Local"
_UNSAFE_FILE_CHARS = re.compile(r"[^A-Za-z0-9.-]+")
_MAX_REPLY_CHARS = 160
_MAX_FILE_ATTEMPTS = 100
_REDACTED = "***"
_PAUSE_LOOKBACK_ROWS = 500  # mail_log rows searched for a pause after a restart (skipped mails included)

type MailKind = Literal["signal", "exit", "test"]
type DeliveryStatus = Literal["sent", "file", "skipped", "failed"]
type SmtpFactory = Callable[..., smtplib.SMTP]
type Sleep = Callable[[float], Awaitable[None]]


# ------------------------------------------------------------------------------------------------
# Value types
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Delivery:
    """Outcome of one mail attempt; ``detail`` is German and stored in mail_log."""

    status: DeliveryStatus
    detail: str = ""
    path: Path | None = None  # the .eml file in file mode

    @property
    def delivered(self) -> bool:
        return self.status in repo_ops.DELIVERED_MAIL_STATUSES


@dataclass(frozen=True, slots=True)
class SmtpFailure:
    """German summary of an SMTP error and how to react to it."""

    detail_de: str
    transient: bool  # worth one retry
    pauses: bool  # stop SMTP sending for a while (repeated rejections can get the account blocked)


@dataclass(frozen=True, slots=True)
class ChartImage:
    png: bytes
    cid: str  # bare Content-ID, without angle brackets
    filename: str


@dataclass(frozen=True, slots=True)
class _Job:
    kind: MailKind
    render: Callable[[str | None], MailContent]  # chart Content-ID (None: no chart) -> mail content
    signal: Signal | None = None
    instrument: Instrument | None = None
    chart_png: bytes | None = None

    @property
    def symbol(self) -> str:
        return "" if self.signal is None else self.signal.symbol


@dataclass(frozen=True, slots=True)
class _Gate:
    reason: str | None  # German reason why the mail is not sent; None: send it
    attach_chart: bool


# ------------------------------------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------------------------------------
def parse_recipients(value: str) -> list[str]:
    """Bare addresses from a recipient setting (display names dropped; ',' or ';' separate addresses)."""
    return [address for _, address in getaddresses([value.replace(";", ",")]) if "@" in address]


def safe_file_part(text: str) -> str:
    """``text`` reduced to letters, digits, '.' and '-' (other runs become '_'), for use in file names."""
    return _UNSAFE_FILE_CHARS.sub("_", text).strip("._-")[:40]


def compose_message(
    content: MailContent,
    *,
    sender: str,
    recipients: Sequence[str],
    date: datetime,
    message_id: str,
    chart: ChartImage | None = None,
) -> EmailMessage:
    """MIME message with CRLF line endings (``email.policy.SMTP``).

    Without a chart: multipart/alternative (text/plain, text/html). With a chart: multipart/mixed of
    multipart/alternative (text/plain, multipart/related (text/html, inline image/png)) and the same PNG as
    attachment. Non-ASCII headers are RFC 2047 encoded automatically.
    """
    msg = EmailMessage(policy=SMTP_POLICY)
    msg["Subject"] = content.subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Date"] = format_datetime(date)
    msg["Message-ID"] = message_id
    msg["Auto-Submitted"] = "auto-generated"  # RFC 3834: suppresses auto-replies
    msg.set_content(content.text)
    msg.add_alternative(content.html, subtype="html")
    if chart is not None:
        html_part = msg.get_payload()[1]
        html_part.add_related(
            chart.png, maintype="image", subtype="png", cid=f"<{chart.cid}>", disposition="inline",
            filename=chart.filename,
        )
        msg.add_attachment(chart.png, maintype="image", subtype="png", filename=chart.filename)
    return msg


def write_eml(outbox: Path, stem: str, msg: EmailMessage) -> Path:
    """Write ``msg`` to ``outbox/<stem>.eml`` (``<stem>-2.eml`` ... if taken; never overwrites) and return the path."""
    outbox.mkdir(parents=True, exist_ok=True)
    data = msg.as_bytes()
    for attempt in range(1, _MAX_FILE_ATTEMPTS + 1):
        path = outbox / (f"{stem}.eml" if attempt == 1 else f"{stem}-{attempt}.eml")
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"no free file name for {stem} in {outbox}")


def classify_smtp_error(exc: BaseException, endpoint: str) -> SmtpFailure:
    """German summary of an exception raised while talking to the SMTP server ``endpoint`` ('host:port')."""
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return SmtpFailure(AUTH_FAILED_DE, transient=False, pauses=True)
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        codes = [code for code, _ in exc.recipients.values()] or [550]
        if all(400 <= code < 500 for code in codes):
            return SmtpFailure(
                f"Der Mailserver hat den Empfänger vorübergehend abgelehnt ({codes[0]}).", transient=True, pauses=False
            )
        code = max(codes)
        return SmtpFailure(
            f"Der Mailserver hat den Empfänger abgelehnt ({code}). Adresse in TL_MAIL_TO prüfen.",
            transient=False,
            pauses=code in PAUSING_REPLY_CODES,
        )
    if isinstance(exc, smtplib.SMTPNotSupportedError):
        return SmtpFailure(
            "Der Mailserver bietet STARTTLS bzw. die Anmeldung nicht an. TL_SMTP_SECURITY und TL_SMTP_PORT prüfen "
            "(ssl = Port 465, starttls = Port 587).",
            transient=False,
            pauses=False,
        )
    if isinstance(exc, smtplib.SMTPResponseException):
        return _reply_failure(exc)
    if isinstance(exc, smtplib.SMTPServerDisconnected):
        return SmtpFailure(
            f"Die Verbindung zum Mailserver {endpoint} wurde unterbrochen.", transient=True, pauses=False
        )
    if isinstance(exc, smtplib.SMTPException):
        return SmtpFailure(
            f"Der Mailserver {endpoint} hat die Anfrage nicht angenommen ({_short(str(exc))}).",
            transient=False,
            pauses=False,
        )
    if isinstance(exc, ssl.SSLError):
        return SmtpFailure(
            f"Die verschlüsselte Verbindung zum Mailserver {endpoint} ist fehlgeschlagen (TLS/Zertifikat). "
            "TL_SMTP_SECURITY und TL_SMTP_PORT prüfen.",
            transient=False,
            pauses=False,
        )
    if isinstance(exc, OSError):
        return SmtpFailure(
            f"Keine Verbindung zum Mailserver {endpoint} ({_network_reason(exc)}).", transient=True, pauses=False
        )
    return SmtpFailure(f"Unerwarteter Fehler beim Mailversand ({type(exc).__name__}).", transient=False, pauses=False)


def _reply_failure(exc: smtplib.SMTPResponseException) -> SmtpFailure:
    code = exc.smtp_code
    raw = exc.smtp_error
    reply = _short(raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw))
    if 400 <= code < 500:
        return SmtpFailure(
            f"Der Mailserver hat den Versand vorübergehend abgelehnt ({code}: {reply}).", transient=True, pauses=False
        )
    if isinstance(exc, smtplib.SMTPSenderRefused):
        detail = (
            f"Der Mailserver hat die Absenderadresse abgelehnt ({code}: {reply}). Als Absender muss die Adresse "
            "des Postfachs eingetragen sein (TL_MAIL_FROM bzw. TL_SMTP_USER)."
        )
    elif code == 554:
        detail = (
            f"Der Mailserver hat die Nachricht abgelehnt ({code}: {reply}), z. B. wegen Spamverdacht oder einer "
            "Richtlinie des Anbieters."
        )
    elif code == 552:
        detail = f"Der Mailserver hat die Nachricht als zu groß abgelehnt ({code}: {reply})."
    else:
        detail = f"Der Mailserver hat den Versand abgelehnt ({code}: {reply})."
    return SmtpFailure(detail, transient=False, pauses=code in PAUSING_REPLY_CODES)


def _network_reason(exc: OSError) -> str:
    if isinstance(exc, TimeoutError):
        return "Zeitüberschreitung"
    if isinstance(exc, socket.gaierror):
        return "Servername nicht auflösbar"
    if isinstance(exc, ConnectionRefusedError):
        return "Verbindung abgelehnt"
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
        return "Verbindung unterbrochen"
    return "Netzwerkfehler"


def _short(text: str) -> str:
    """One line of server text, at most ``_MAX_REPLY_CHARS`` characters."""
    line = " ".join(text.split())
    return line if len(line) <= _MAX_REPLY_CHARS else line[: _MAX_REPLY_CHARS - 1] + "…"


def _ago_de(seconds: int) -> str:
    minutes = seconds // 60
    if minutes < 1:
        return "vor weniger als einer Minute"
    return "vor 1 Minute" if minutes == 1 else f"vor {minutes} Minuten"


class _Redactor:
    """Replaces the SMTP password by '***': plain, in the escaped forms Python prints when an exception text
    shows the server's reply bytes (``b'G\\xc3\\xa4...'``) or a repr, and in the base64 forms of AUTH PLAIN / LOGIN.

    Also masks the configured mail addresses ('t***@example.org', any letter case): SMTP errors such as
    ``SMTPSenderRefused`` / ``SMTPRecipientsRefused`` and many server replies ('553 5.7.1 <addr>: Sender address
    rejected') repeat them, and those texts reach the log file, mail_log, the status and the activity log.
    """

    def __init__(self, user: str, password: str, addresses: Collection[str] = ()) -> None:
        variants: set[str] = set()
        if password:
            for raw in (password, f"\0{user}\0{password}", f"{user}\0{password}"):
                variants.add(base64.b64encode(raw.encode("utf-8")).decode("ascii"))
            variants.update(
                (password, repr(password.encode("utf-8"))[2:-1], password.encode("unicode_escape").decode("ascii"))
            )
        self._secrets = tuple(sorted(variants, key=len, reverse=True))
        self._mask_addresses = address_masker(addresses)

    def __call__(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _REDACTED)
        return self._mask_addresses(text)


# ------------------------------------------------------------------------------------------------
# The mailer
# ------------------------------------------------------------------------------------------------
class AlertMailer:
    """``contracts.Mailer``: sends entry, exit and test mails; never raises from ``send_*``.

    ``smtp_factory`` / ``smtp_ssl_factory`` create the client for 'starttls' / 'ssl' (default
    ``smtplib.SMTP`` / ``smtplib.SMTP_SSL``); ``sleep`` performs the pacing and retry waits. Both are
    replaceable for tests; the limits are keyword arguments for the same reason.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        bus: EventBus,
        runtime_settings: SettingsStore,
        smtp_factory: SmtpFactory = smtplib.SMTP,
        smtp_ssl_factory: SmtpFactory = smtplib.SMTP_SSL,
        sleep: Sleep = asyncio.sleep,
        retry_pause_s: float = RETRY_PAUSE_S,
        min_interval_s: float = MIN_SMTP_INTERVAL_S,
        daily_cap: int = DAILY_SMTP_CAP,
        pause_s: float = SMTP_PAUSE_S,
    ) -> None:
        ZoneInfo(settings.timezone)  # fail fast on a misconfigured time zone
        self._settings = settings
        self._db = db
        self._clock = clock
        self._bus = bus
        self._runtime_settings = runtime_settings
        self._smtp_factory = smtp_factory
        self._smtp_ssl_factory = smtp_ssl_factory
        self._sleep = sleep
        self._retry_pause_s = retry_pause_s
        self._min_interval_s = min_interval_s
        self._daily_cap = daily_cap
        self._pause_s = pause_s

        self._tz = settings.timezone
        self._recipients = parse_recipients(settings.mail_to)
        self._sender_address = settings.sender or _FALLBACK_SENDER
        self._gui_base = gui_base_url(settings.host, settings.port)
        self._endpoint = f"{settings.smtp_host}:{settings.smtp_port}"
        self._redact = _Redactor(
            settings.smtp_user,
            settings.smtp_password.get_secret_value(),
            addresses=(settings.mail_to, settings.sender, settings.smtp_user),
        )

        self._lock = asyncio.Lock()  # one mail at a time: gate, pacing, connection, record
        self._inflight: set[asyncio.Future[Delivery]] = set()  # shielded deliveries (see _send_guarded)
        self._last_smtp_ts: float | None = None  # end of the previous SMTP connection (epoch seconds)
        self._paused_until: float | None = None
        self._pause_reason = ""
        self._pause_restored = False  # a pause of a previous process was looked up in mail_log

    # -- contracts.Mailer ---------------------------------------------------------------------------
    async def send_signal(self, signal: Signal, instrument: Instrument, chart_png: bytes | None = None) -> bool:
        """Entry recommendation. True if the mail was sent or written to the outbox.

        On delivery the signal is marked as mailed in the database AND on the given object (so that a later
        ``update_signal`` of that object keeps ``mailed_ts``). A signal that is already marked as mailed is not
        mailed again (False, no attempt).
        """
        if signal.mailed_ts is not None:
            log.debug("signal %s was already mailed at %s", signal.id, signal.mailed_ts)
            return False
        job = _Job(
            "signal",
            lambda cid: entry_mail(signal, instrument, timezone=self._tz, gui_base=self._gui_base, chart_cid=cid),
            signal=signal,
            instrument=instrument,
            chart_png=chart_png,
        )
        return await self._send_guarded(job)

    async def send_exit(self, signal: Signal, instrument: Instrument) -> bool:
        """Exit recommendation; only for signals whose entry mail was delivered (``mailed_ts`` set).

        Exempt from the region filter and the per-symbol cooldown; may exceed the hourly cap by up to 50 %.
        """
        if signal.mailed_ts is None:
            log.debug("no exit mail for signal %s: its entry was never mailed", signal.id)
            return False
        job = _Job(
            "exit",
            lambda _cid: exit_mail(signal, instrument, timezone=self._tz, gui_base=self._gui_base),
            signal=signal,
            instrument=instrument,
        )
        return await self._send_guarded(job)

    async def send_test(self) -> MailTestOut:
        """Test mail that ignores all limits (pacing still applies); a successful SMTP test lifts a pause."""
        job = _Job("test", lambda _cid: check_mail(
            transport_de=self._transport_de(), now_ts=self._clock.ts(), timezone=self._tz, gui_base=self._gui_base
        ))
        try:
            delivery = await self._send_serialised(job)
        except Exception as exc:  # noqa: BLE001 - never raise to the API
            self._log_unexpected(job.kind, exc)
            return MailTestOut(
                ok=False, status="failed", detail_de="Unerwarteter Fehler beim Testversand - Einzelheiten im Protokoll."
            )
        return MailTestOut(ok=delivery.delivered, status=delivery.status, detail_de=self._test_detail(delivery))

    def status(self) -> MailStatusOut:
        now_ts = self._clock.ts()
        try:
            sent = repo_ops.count_mails(self._db, since_ts=now_ts - HOUR_S)
            last_error = self._pause_note() or repo_ops.last_mail_error(self._db)
        except sqlite3.Error:
            log.exception("mail status: database query failed")
            sent, last_error = 0, "Der Mail-Status konnte nicht aus der Datenbank gelesen werden."
        return MailStatusOut(
            mode=self._settings.mail_mode,
            configured=self._settings.mail_configured,
            recipient_masked=mask_email(self._settings.mail_to),
            sent_last_hour=sent,
            last_error=self._redact(last_error),
        )

    # -- flow ---------------------------------------------------------------------------------------
    async def _send_guarded(self, job: _Job) -> bool:
        """Send one mail. The delivery and its record run in their own task, shielded from the caller's
        cancellation: a mail whose SMTP transaction has started (it cannot be taken back) is always recorded
        in mail_log and mailed_ts, even if the caller gives up meanwhile (e.g. at shutdown). Without that
        record the later exit mail of the signal would be suppressed."""
        task = asyncio.ensure_future(self._send_serialised(job))
        self._inflight.add(task)
        task.add_done_callback(self._finish_inflight)
        try:
            return (await asyncio.shield(task)).delivered
        except asyncio.CancelledError:
            log.info("%s mail for %s: caller cancelled, delivery continues in the background", job.kind, job.symbol)
            raise
        except Exception as exc:  # noqa: BLE001 - one broken mail must never stop the caller
            self._log_unexpected(job.kind, exc)
            return False

    def _finish_inflight(self, task: "asyncio.Future[Delivery]") -> None:
        self._inflight.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("background mail delivery failed: %s", type(task.exception()).__name__)

    async def _send_serialised(self, job: _Job) -> Delivery:
        async with self._lock:
            if job.kind == "signal" and job.signal is not None and job.signal.mailed_ts is not None:
                # a concurrent call for the same signal held the lock first and delivered it: no attempt
                log.debug("signal %s was mailed while this call waited", job.signal.id)
                return Delivery("skipped", "Das Signal wurde bereits verschickt.")
            return await self._run(job)

    async def _run(self, job: _Job) -> Delivery:
        """Gate, content and delivery of one mail, then the record. Every outcome leaves a mail_log entry."""
        now_ts = self._clock.ts()
        subject = f"{KIND_DE[job.kind]} {job.symbol}".strip()  # stands in if the content cannot be built
        try:
            gate = await asyncio.to_thread(self._gate, job, now_ts)
            chart = self._chart_image(job) if gate.attach_chart else None
            content = job.render(None if chart is None else chart.cid)
            subject = content.subject
            if gate.reason is not None:
                delivery = Delivery("skipped", gate.reason)
            else:
                delivery = await self._deliver(job, self._compose(content, chart, now_ts), now_ts)
        except Exception as exc:  # noqa: BLE001 - a bug here must still leave a mail_log entry
            self._log_unexpected(job.kind, exc)
            delivery = Delivery("failed", UNEXPECTED_DE)
        record_ts = self._clock.ts()
        if job.kind == "signal" and delivery.delivered and job.signal is not None:
            # set before the database write: if that write fails, the caller's object still says 'mailed' (and
            # its next update_signal stores it), so the same signal is never mailed twice
            job.signal.mailed_ts = record_ts
        try:
            await asyncio.to_thread(self._record, job, subject, delivery, record_ts)
        except Exception as exc:  # noqa: BLE001 - the outcome stands even if it cannot be recorded
            self._log_unexpected(job.kind, exc)
        return delivery

    # -- step 1: gate (runs in a worker thread: database reads) --------------------------------------
    def _gate(self, job: _Job, now_ts: int) -> _Gate:
        prefs = self._runtime_settings.get()
        reason = self._setup_reason() or self._preference_reason(job, prefs, now_ts)
        if reason is None and self._settings.mail_mode == "smtp" and job.kind != "test":
            reason = self._provider_reason(now_ts)
        return _Gate(reason, attach_chart=job.chart_png is not None and prefs.mail_attach_chart)

    def _setup_reason(self) -> str | None:
        """Why the configured mode cannot deliver at all (None if it can)."""
        settings = self._settings
        if settings.mail_mode == "off":
            return "Der E-Mail-Versand ist ausgeschaltet (TL_MAIL_MODE=off)."
        missing: list[str] = []
        if settings.mail_mode == "smtp":
            if not settings.smtp_host:
                missing.append("TL_SMTP_HOST")
            if not settings.smtp_user:
                missing.append("TL_SMTP_USER")
            if not settings.smtp_password.get_secret_value():
                missing.append("TL_SMTP_PASSWORD")
        if not settings.mail_to:
            missing.append("TL_MAIL_TO")
        elif not self._recipients:
            missing.append("eine gültige Empfängeradresse in TL_MAIL_TO")
        if not missing:
            return None
        verb = "fehlt" if len(missing) == 1 else "fehlen"
        return f"Der E-Mail-Versand ist nicht vollständig eingerichtet: {', '.join(missing)} {verb}."

    def _preference_reason(self, job: _Job, prefs: RuntimeSettings, now_ts: int) -> str | None:
        """The user's sending policy (not applied to test mails)."""
        if job.kind == "test":
            return None
        if not prefs.mail_enabled:
            return "E-Mail-Benachrichtigungen sind in den Einstellungen ausgeschaltet."
        if job.kind == "exit" and not prefs.mail_on_exit:
            return "Ausstiegs-E-Mails sind in den Einstellungen ausgeschaltet."
        if job.kind == "signal" and job.instrument is not None and job.instrument.region not in prefs.mail_regions:
            region = REGION_LABELS_DE.get(job.instrument.region, job.instrument.region)
            return f"E-Mails für die Region {region} sind in den Einstellungen nicht ausgewählt."
        limit = prefs.mail_max_per_hour
        if limit == 0:
            return "Das stündliche E-Mail-Limit steht auf 0 - E-Mails sind damit ausgeschaltet."
        allowed = limit + limit // 2 if job.kind == "exit" else limit  # an exit must not get lost behind entries
        sent = repo_ops.count_mails(self._db, since_ts=now_ts - HOUR_S)
        if sent >= allowed:
            return f"Stündliches E-Mail-Limit erreicht: {sent} E-Mails in der letzten Stunde (erlaubt: {allowed})."
        cooldown_min = prefs.mail_symbol_cooldown_min
        if job.kind == "signal" and cooldown_min > 0:
            last_ts = repo_ops.last_mail_ts_for_symbol(self._db, job.symbol)
            if last_ts is not None and now_ts - last_ts < cooldown_min * 60:
                return (
                    f"Für {job.symbol} wurde {_ago_de(now_ts - last_ts)} bereits eine E-Mail verschickt "
                    f"(Sperrfrist {cooldown_min} Minuten)."
                )
        return None

    def _provider_reason(self, now_ts: int) -> str | None:
        """Protection of the SMTP account: pause after rejections, daily cap."""
        if not self._pause_restored:
            self._restore_pause(now_ts)
        note = self._pause_note()
        if note:
            return note
        sent = repo_ops.count_mails(self._db, since_ts=now_ts - DAY_S, statuses=("sent",))
        if sent >= self._daily_cap:
            return (
                f"Tageslimit erreicht: {sent} E-Mails über SMTP in den letzten 24 Stunden "
                f"(höchstens {self._daily_cap}, Schutz vor einer Sperre durch den Mailanbieter)."
            )
        return None

    def _restore_pause(self, now_ts: int) -> None:
        """Resume a pause that a previous process started (the pause itself lives in memory): without this a
        restart would allow another login attempt right after a rejected one.

        Looks at the mail_log entries of the last ``pause_s`` seconds, newest first: a 'sent' entry means
        delivery worked after any rejection; a 'failed' entry whose detail announced a pause restarts it
        from that entry's time.
        """
        self._pause_restored = True
        since = now_ts - self._pause_s
        for entry in repo_ops.list_mail_log(self._db, limit=_PAUSE_LOOKBACK_ROWS):
            if entry.ts < since or entry.status == "sent":
                return
            if entry.status == "failed" and PAUSE_MARKER_DE in entry.detail:
                self._paused_until = entry.ts + self._pause_s
                self._pause_reason = entry.detail.split(f" {PAUSE_MARKER_DE}", 1)[0]
                log.info("SMTP: pause after a rejection at %d restored", entry.ts)
                return

    # -- step 2: content ------------------------------------------------------------------------------
    def _chart_image(self, job: _Job) -> ChartImage | None:
        if job.chart_png is None:
            return None
        name = safe_file_part(job.symbol) or "chart"
        return ChartImage(job.chart_png, cid=f"chart.{uuid.uuid4().hex}@{EHLO_NAME}", filename=f"chart_{name}.png")

    def _compose(self, content: MailContent, chart: ChartImage | None, now_ts: int) -> EmailMessage:
        return compose_message(
            content,
            sender=formataddr((_SENDER_NAME, self._sender_address)),
            recipients=self._recipients,
            date=self._clock.now().astimezone(ZoneInfo(self._tz)),
            message_id=f"<{now_ts}.{uuid.uuid4().hex}@{EHLO_NAME}>",
            chart=chart,
        )

    # -- step 3: deliver ------------------------------------------------------------------------------
    async def _deliver(self, job: _Job, msg: EmailMessage, now_ts: int) -> Delivery:
        if self._settings.mail_mode == "file":
            return await asyncio.to_thread(self._write_file, job, msg, now_ts)
        return await self._send_smtp(job, msg)

    def _write_file(self, job: _Job, msg: EmailMessage, now_ts: int) -> Delivery:
        outbox = self._settings.outbox_dir
        stamp = datetime.fromtimestamp(now_ts, UTC).strftime("%Y%m%dT%H%M%SZ")
        stem = "_".join(part for part in (stamp, job.kind, safe_file_part(job.symbol)) if part)
        try:
            path = write_eml(outbox, stem, msg)
        except OSError as exc:
            log.warning("could not write mail file to %s: %s", outbox, exc)
            reason = exc.strerror or type(exc).__name__
            return Delivery("failed", f"Die Mail-Datei konnte nicht im Ordner {outbox} gespeichert werden ({reason}).")
        log.info("%s mail written to %s", job.kind, path)
        return Delivery("file", f"Als Datei gespeichert: {path}", path)

    async def _send_smtp(self, job: _Job, msg: EmailMessage) -> Delivery:
        await self._wait_for_slot()
        retried = False
        try:
            failure = await self._attempt(msg)
            if failure is not None and failure.transient:
                log.warning("SMTP: transient failure, one retry in %.0f s", self._retry_pause_s)
                await self._sleep(self._retry_pause_s)
                retried = True
                failure = await self._attempt(msg)
        finally:
            self._last_smtp_ts = self._clock.now().timestamp()
        if failure is None:
            if self._paused_until is not None:
                log.info("SMTP: delivery works again, pause lifted")
            self._paused_until, self._pause_reason = None, ""
            log.info("%s mail sent via %s to %s", job.kind, self._endpoint, mask_email(self._settings.mail_to))
            return Delivery("sent", f"Verschickt über {self._endpoint} an {mask_email(self._settings.mail_to)}.")
        detail = failure.detail_de
        if retried:
            detail += " Auch der zweite Versuch ist fehlgeschlagen."
        if failure.pauses:
            self._paused_until = self._clock.now().timestamp() + self._pause_s
            self._pause_reason = failure.detail_de
            detail += f" {PAUSE_MARKER_DE} {self._pause_end_de()} Uhr."
        return Delivery("failed", detail)

    async def _wait_for_slot(self) -> None:
        """Keep at least ``min_interval_s`` between two SMTP connections (later mails wait in call order)."""
        if self._last_smtp_ts is None:
            return
        wait_s = self._last_smtp_ts + self._min_interval_s - self._clock.now().timestamp()
        if wait_s > 0:
            log.debug("SMTP pacing: waiting %.1f s", wait_s)
            await self._sleep(wait_s)

    async def _attempt(self, msg: EmailMessage) -> SmtpFailure | None:
        """One SMTP transaction in a worker thread; the classified failure, or None on success."""
        try:
            refused = await asyncio.to_thread(self._smtp_transaction, msg)
        except Exception as exc:  # noqa: BLE001 - classified and reported below
            failure = classify_smtp_error(exc, self._endpoint)
            log.warning(
                "SMTP delivery via %s failed (%s): %s",
                self._endpoint,
                type(exc).__name__,
                self._redact(_short(str(exc))),
            )
            return failure
        if refused:
            log.warning("SMTP: the server refused some recipients: %s", self._redact(_short(str(refused))))
        return None

    def _smtp_transaction(self, msg: EmailMessage) -> dict[str, tuple[int, bytes]]:
        """Blocking: connect, (STARTTLS), login, send, quit. Returns the refused recipients ({} = all accepted)."""
        settings = self._settings
        context = ssl.create_default_context()
        if settings.smtp_security == "ssl":
            client = self._smtp_ssl_factory(
                settings.smtp_host, settings.smtp_port, local_hostname=EHLO_NAME, timeout=SMTP_TIMEOUT_S,
                context=context,
            )
        else:
            client = self._smtp_factory(
                settings.smtp_host, settings.smtp_port, local_hostname=EHLO_NAME, timeout=SMTP_TIMEOUT_S
            )
        try:
            client.ehlo()
            if settings.smtp_security == "starttls":
                client.starttls(context=context)
                client.ehlo()
            _login(client, settings.smtp_user, settings.smtp_password.get_secret_value())
            return client.send_message(msg, from_addr=self._sender_address, to_addrs=self._recipients)
        finally:
            _close_quietly(client)

    # -- step 4: record (runs in a worker thread) -----------------------------------------------------
    def _record(self, job: _Job, subject: str, delivery: Delivery, ts: int) -> None:
        signal_id = None if job.signal is None else job.signal.id
        detail = self._redact(delivery.detail)
        with self._db.write():
            repo_ops.log_mail(
                self._db,
                ts=ts,
                kind=job.kind,
                recipient=self._settings.mail_to,
                subject=subject,
                status=delivery.status,
                detail=detail,
                signal_id=signal_id,
            )
            if job.kind == "signal" and delivery.delivered and signal_id is not None:
                repo_signals.mark_mailed(self._db, signal_id, ts)
        activity = self._activity(job, subject, delivery, detail)
        if activity is not None:
            level, message = activity
            repo_ops.log_activity(
                self._db,
                ts=ts,
                level=level,
                category="mail",
                message=message,
                ref_type=None if signal_id is None else "signal",
                ref_id=signal_id,
                bus=self._bus,
            )

    @staticmethod
    def _activity(job: _Job, subject: str, delivery: Delivery, detail: str) -> tuple[str, str] | None:
        """(level, German message) for the activity log; skipped mails are only in the mail log."""
        match delivery.status:
            case "sent":
                return "info", f"E-Mail verschickt: {subject}"
            case "file":
                name = "" if delivery.path is None else f" ({delivery.path.name})"
                return "info", f"E-Mail als Datei gespeichert{name}: {subject}"
            case "failed":
                return "error", f"E-Mail-Versand fehlgeschlagen ({KIND_DE[job.kind]}): {detail}"
            case _:
                return None

    # -- texts ----------------------------------------------------------------------------------------
    def _transport_de(self) -> str:
        match self._settings.mail_mode:
            case "smtp":
                return f"SMTP über {self._endpoint} ({self._settings.smtp_security.upper()})"
            case "file":
                return f"Datei im Ordner {self._settings.outbox_dir}"
            case _:
                return "ausgeschaltet"

    def _test_detail(self, delivery: Delivery) -> str:
        match delivery.status:
            case "sent":
                return (
                    f"Testmail an {mask_email(self._settings.mail_to)} über {self._endpoint} "
                    f"({self._settings.smtp_security.upper()}) verschickt. Bitte Posteingang und Spam-Ordner prüfen."
                )
            case "file":
                return (
                    "Versandart 'file': Die Testmail wurde nicht verschickt, sondern als Datei gespeichert: "
                    f"{delivery.path}"
                )
            case "skipped":
                return f"{self._redact(delivery.detail)} Es wurde keine Testmail erzeugt."
            case _:
                return f"Die Testmail konnte nicht verschickt werden: {self._redact(delivery.detail)}"

    def _pause_note(self) -> str:
        """German note while SMTP sending is paused after a rejection; '' otherwise."""
        if self._paused_until is None or self._clock.now().timestamp() >= self._paused_until:
            return ""
        return f"{PAUSE_MARKER_DE} {self._pause_end_de()} Uhr nach dem Fehler: {self._pause_reason}"

    def _pause_end_de(self) -> str:
        return "" if self._paused_until is None else format_time(int(self._paused_until), self._tz)

    def _log_unexpected(self, kind: str, exc: BaseException) -> None:
        trace = "".join(traceback.format_exception(exc))
        log.error("unexpected error while handling a %s mail:\n%s", kind, self._redact(trace))


def _login(client: smtplib.SMTP, user: str, password: str) -> None:
    """Authenticate. ``smtplib`` encodes credentials as ASCII and fails on umlauts with a ``UnicodeEncodeError``;
    such credentials are sent with AUTH PLAIN in UTF-8 instead (RFC 4616)."""
    if (user + password).isascii():
        client.login(user, password)
        return
    client.ehlo_or_helo_if_needed()
    mechanisms = client.esmtp_features.get("auth", "").upper().split()
    if "PLAIN" not in mechanisms:
        raise smtplib.SMTPNotSupportedError("AUTH PLAIN (needed for non-ASCII credentials) is not offered")
    token = base64.b64encode(f"\0{user}\0{password}".encode("utf-8")).decode("ascii")
    code, reply = client.docmd("AUTH", f"PLAIN {token}")
    if code != 235:
        raise smtplib.SMTPAuthenticationError(code, reply)


def _close_quietly(client: smtplib.SMTP) -> None:
    """QUIT and close; errors are irrelevant here (the outcome of the transaction is already known)."""
    try:
        client.quit()
    except (smtplib.SMTPException, OSError):
        client.close()
