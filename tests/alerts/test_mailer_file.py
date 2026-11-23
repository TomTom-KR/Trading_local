"""File and off modes, the MIME structure of the messages, send_test and status."""

import email
import email.policy
from collections.abc import Callable
from email.message import EmailMessage
from pathlib import Path

from trading_local.alerts import build_mailer
from trading_local.alerts.mailer import AlertMailer, safe_file_part
from trading_local.alerts.templates import FOOTER_DE
from trading_local.clock import ManualClock
from trading_local.contracts import Mailer
from trading_local.db import repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.models import Signal
from trading_local.testing import make_settings

from .conftest import INSTRUMENTS

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


def parse(path: Path) -> EmailMessage:
    message = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    assert isinstance(message, EmailMessage)
    return message


def outbox_files(tmp_path: Path) -> list[Path]:
    outbox = tmp_path / "outbox"
    return sorted(outbox.iterdir()) if outbox.exists() else []


def test_factory_builds_a_mailer(
    tmp_path: Path, seeded_db: Database, clock: ManualClock, bus: EventBus, store: SettingsStore
) -> None:
    mailer = build_mailer(
        settings=make_settings(tmp_path), db=seeded_db, clock=clock, bus=bus, runtime_settings=store,
        instruments=InstrumentBook(seeded_db),
    )
    assert isinstance(mailer, Mailer)
    assert isinstance(mailer, AlertMailer)


async def test_file_mode_writes_a_complete_parseable_message(
    tmp_path: Path, seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal],
    clock: ManualClock,
) -> None:
    signal = make_signal()
    assert await make_mailer().send_signal(signal, INSTRUMENTS["NVDA"], PNG) is True

    [path] = outbox_files(tmp_path)
    assert path.name == "20261120T150000Z_signal_NVDA.eml"
    raw = path.read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")  # RFC 5322 line endings
    message = parse(path)
    assert message["Subject"] == "LONG NVDA 142,35 USD | Stopp 140,90 | Ziel 145,25 | Konfidenz 58 %"
    assert message["From"] == "Trading Local <absender@example.invalid>"
    assert message["To"] == "empfaenger@example.invalid"
    assert message["Auto-Submitted"] == "auto-generated"
    assert message["Message-ID"].endswith("@trading.local>")
    assert message["Date"].datetime.utcoffset().total_seconds() == 3600  # Berlin winter time
    assert int(message["Date"].datetime.timestamp()) == clock.ts()

    structure = [(part.get_content_type(), part.get_content_disposition()) for part in message.walk()]
    assert structure == [
        ("multipart/mixed", None),
        ("multipart/alternative", None),
        ("text/plain", None),
        ("multipart/related", None),
        ("text/html", None),
        ("image/png", "inline"),
        ("image/png", "attachment"),
    ]
    plain = message.get_body(preferencelist=("plain",))
    html = message.get_body(preferencelist=("html",))
    assert plain is not None and html is not None
    assert "Kaufen (Long): NVIDIA (NVDA) – Nasdaq" in plain.get_content()
    assert FOOTER_DE in html.get_content()
    images = [part for part in message.walk() if part.get_content_type() == "image/png"]
    inline, attachment = images
    cid = inline["Content-ID"].strip("<>")
    assert f'src="cid:{cid}"' in html.get_content()
    assert inline.get_content() == PNG and attachment.get_content() == PNG
    assert attachment.get_filename() == "chart_NVDA.png"


async def test_file_mode_logs_marks_and_announces(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], clock: ManualClock
) -> None:
    signal = make_signal()
    await make_mailer().send_signal(signal, INSTRUMENTS["NVDA"])

    [entry] = repo_ops.list_mail_log(seeded_db)
    assert (entry.kind, entry.status, entry.signal_id, entry.ts) == ("signal", "file", signal.id, clock.ts())
    assert entry.subject.startswith("LONG NVDA")
    assert "Als Datei gespeichert" in entry.detail
    stored = repo_signals.get_signal(seeded_db, signal.id or 0)
    assert stored is not None and stored.mailed_ts == clock.ts()
    assert signal.mailed_ts == clock.ts()  # the caller's object as well
    [activity] = repo_ops.list_activity(seeded_db, category="mail")
    assert activity.level == "info" and activity.ref_id == signal.id
    assert activity.message.startswith("E-Mail als Datei gespeichert")


async def test_message_without_chart_is_plain_alternative(
    tmp_path: Path, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], store: SettingsStore,
    clock: ManualClock,
) -> None:
    mailer = make_mailer()
    await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"])
    store.update({"mail_attach_chart": False}, clock.ts())
    await mailer.send_signal(make_signal(symbol="SAP.DE", ref_price=182.0), INSTRUMENTS["SAP.DE"], PNG)

    for path in outbox_files(tmp_path):
        message = parse(path)
        types = [part.get_content_type() for part in message.walk()]
        assert types == ["multipart/alternative", "text/plain", "text/html"], path.name
        html = message.get_body(preferencelist=("html",))
        assert html is not None and "<img" not in html.get_content()


async def test_file_names_are_safe_and_never_overwritten(
    tmp_path: Path, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], store: SettingsStore,
    clock: ManualClock,
) -> None:
    store.update({"mail_symbol_cooldown_min": 0}, clock.ts())
    mailer = make_mailer()
    for _ in range(2):
        await mailer.send_signal(make_signal(symbol="^GDAXI", ref_price=23456.0, stop_dist=40, target_dist=80),
                                 INSTRUMENTS["^GDAXI"])
    names = [path.name for path in outbox_files(tmp_path)]
    assert names == ["20261120T150000Z_signal_GDAXI-2.eml", "20261120T150000Z_signal_GDAXI.eml"]
    assert safe_file_part("EURUSD=X") == "EURUSD_X"
    assert safe_file_part("../..\\evil name") == "evil_name"
    assert safe_file_part("7203.T") == "7203.T"


async def test_unwritable_outbox_fails_softly(
    tmp_path: Path, seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal]
) -> None:
    (tmp_path / "outbox").write_text("a file where the folder should be", encoding="utf-8")
    signal = make_signal()
    assert await make_mailer().send_signal(signal, INSTRUMENTS["NVDA"]) is False
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert entry.status == "failed"
    assert "Die Mail-Datei konnte nicht im Ordner" in entry.detail
    assert signal.mailed_ts is None
    [activity] = repo_ops.list_activity(seeded_db, category="mail")
    assert activity.level == "error" and "fehlgeschlagen" in activity.message


async def test_off_mode_only_logs_a_skipped_attempt(
    tmp_path: Path, seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal]
) -> None:
    mailer = make_mailer({"mail_mode": "off"})
    assert await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"], PNG) is False
    assert outbox_files(tmp_path) == []
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert (entry.status, entry.detail) == ("skipped", "Der E-Mail-Versand ist ausgeschaltet (TL_MAIL_MODE=off).")
    assert repo_ops.list_activity(seeded_db, category="mail") == []


async def test_missing_recipient_is_reported(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal]
) -> None:
    assert await make_mailer({"mail_to": ""}).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    assert await make_mailer({"mail_to": "kein-adresse"}).send_signal(make_signal(), INSTRUMENTS["NVDA"]) is False
    entries = repo_ops.list_mail_log(seeded_db)
    assert [entry.detail for entry in entries] == [
        "Der E-Mail-Versand ist nicht vollständig eingerichtet: eine gültige Empfängeradresse in TL_MAIL_TO fehlt.",
        "Der E-Mail-Versand ist nicht vollständig eingerichtet: TL_MAIL_TO fehlt.",
    ]


async def test_several_recipients(
    tmp_path: Path, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal]
) -> None:
    mailer = make_mailer({"mail_to": "Max <max@example.invalid>; erika@example.invalid"})
    await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"])
    [path] = outbox_files(tmp_path)
    assert parse(path)["To"] == "max@example.invalid, erika@example.invalid"
    assert mailer.status().recipient_masked == "m***@example.invalid, e***@example.invalid"


async def test_send_test_in_file_mode_names_the_file(
    tmp_path: Path, seeded_db: Database, make_mailer: Callable[..., AlertMailer], store: SettingsStore,
    clock: ManualClock,
) -> None:
    store.update({"mail_enabled": False, "mail_max_per_hour": 0}, clock.ts())  # limits do not apply to tests
    result = await make_mailer().send_test()
    [path] = outbox_files(tmp_path)
    assert (result.ok, result.status) == (True, "file")
    assert str(path) in result.detail_de
    assert result.detail_de.startswith("Versandart 'file': Die Testmail wurde nicht verschickt")
    message = parse(path)
    assert message["Subject"] == "Testmail von Trading Local"
    assert path.name == "20261120T150000Z_test.eml"
    [entry] = repo_ops.list_mail_log(seeded_db)
    assert (entry.kind, entry.status, entry.signal_id) == ("test", "file", None)


async def test_send_test_in_off_mode(seeded_db: Database, make_mailer: Callable[..., AlertMailer]) -> None:
    result = await make_mailer({"mail_mode": "off"}).send_test()
    assert (result.ok, result.status) == (False, "skipped")
    assert result.detail_de == (
        "Der E-Mail-Versand ist ausgeschaltet (TL_MAIL_MODE=off). Es wurde keine Testmail erzeugt."
    )
    assert repo_ops.list_mail_log(seeded_db)[0].status == "skipped"


async def test_status(
    seeded_db: Database, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], clock: ManualClock
) -> None:
    mailer = make_mailer()
    empty = mailer.status()
    assert (empty.mode, empty.configured, empty.recipient_masked) == ("file", True, "e***@example.invalid")
    assert (empty.sent_last_hour, empty.last_error) == (0, "")
    await mailer.send_signal(make_signal(), INSTRUMENTS["NVDA"])
    repo_ops.log_mail(seeded_db, ts=clock.ts() - 4000, kind="signal", recipient="x", subject="alt", status="file")
    assert mailer.status().sent_last_hour == 1
    off = make_mailer({"mail_mode": "off"}).status()
    assert (off.mode, off.configured) == ("off", False)


async def test_non_ascii_subject_is_rfc2047_encoded(
    tmp_path: Path, make_mailer: Callable[..., AlertMailer], make_signal: Callable[..., Signal], clock: ManualClock
) -> None:
    signal = make_signal(
        status="stopped", mailed_ts=clock.ts() - 600, entry_ts=clock.ts() - 540, entry_price=142.4,
        exit_ts=clock.ts() - 60, exit_price=140.9, outcome_r=-1.04,
    )
    assert await make_mailer().send_exit(signal, INSTRUMENTS["NVDA"]) is True
    [path] = outbox_files(tmp_path)
    assert path.name == "20261120T150000Z_exit_NVDA.eml"
    raw = path.read_bytes()
    subject_header = raw.split(b"\r\nFrom:", 1)[0]
    assert raw.isascii()  # 7-bit transport: headers RFC 2047, bodies quoted-printable/base64
    assert b"=?utf-8?" in subject_header
    assert parse(path)["Subject"] == "SCHLIESSEN NVDA (Long): Stopp ausgelöst, -1,0 R"
    plain = parse(path).get_body(preferencelist=("plain",))
    assert plain is not None and "Long-Position schließen: NVIDIA (NVDA)" in plain.get_content()
