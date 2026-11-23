"""Fixtures for the alert tests: a fake SMTP server, a fake sleep that advances the manual clock, signals."""

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from trading_local.alerts.mailer import AlertMailer
from trading_local.clock import ManualClock
from trading_local.db import repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.models import Instrument, Signal
from trading_local.testing import SAMPLE_INSTRUMENTS, make_settings, seed_instruments

PASSWORD = "s3cr3t-Passwort!42"
UMLAUT_PASSWORD = "Gähnende-Lücke-ß-42"  # smtplib alone cannot log in with this one
SMTP_USER = "absender@example.invalid"
INSTRUMENTS: dict[str, Instrument] = {instrument.symbol: instrument for instrument in SAMPLE_INSTRUMENTS}
SMTP_SETTINGS: dict[str, Any] = {
    "mail_mode": "smtp",
    "smtp_host": "mail.example.invalid",
    "smtp_port": 465,
    "smtp_security": "ssl",
    "smtp_user": SMTP_USER,
    "smtp_password": PASSWORD,
    "mail_from": SMTP_USER,
    "mail_to": "empfaenger@example.invalid",
}


# ------------------------------------------------------------------------------------------------
# Fake SMTP
# ------------------------------------------------------------------------------------------------
@dataclass
class Connection:
    kind: str  # 'ssl' | 'plain'
    host: str
    port: int
    kwargs: dict[str, Any]
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


class SmtpRecorder:
    """Shared state of all fake SMTP connections. ``plan`` holds one entry per connection attempt:
    ``None`` (success) or ``(stage, exception)`` with stage 'connect', 'login' or 'send'."""

    def __init__(self) -> None:
        self.plan: list[tuple[str, BaseException] | None] = []
        self.connections: list[Connection] = []
        self.sent: list[EmailMessage] = []
        self.send_delay_s = 0.0
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def factory(self, kind: str) -> Callable[..., "FakeSmtp"]:
        def create(host: str, port: int, **kwargs: Any) -> FakeSmtp:
            step = self.plan.pop(0) if self.plan else None
            if step is not None and step[0] == "connect":
                self.connections.append(Connection(kind, host, port, kwargs))
                raise step[1]
            return FakeSmtp(self, Connection(kind, host, port, kwargs), step)

        return create

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active -= 1


class FakeSmtp:
    def __init__(self, recorder: SmtpRecorder, connection: Connection, step: tuple[str, BaseException] | None) -> None:
        self._recorder = recorder
        self._connection = connection
        self._step = step
        recorder.connections.append(connection)
        recorder.enter()

    def _maybe_fail(self, stage: str) -> None:
        if self._step is not None and self._step[0] == stage:
            raise self._step[1]

    def ehlo(self, name: str = "") -> tuple[int, bytes]:
        self._connection.calls.append(("ehlo", ()))
        return 250, b"ok"

    def starttls(self, *, context: Any = None) -> tuple[int, bytes]:
        self._connection.calls.append(("starttls", (context,)))
        return 220, b"ready"

    def login(self, user: str, password: str) -> tuple[int, bytes]:
        self._connection.calls.append(("login", (user, password)))
        self._maybe_fail("login")
        return 235, b"ok"

    def send_message(self, msg: EmailMessage, from_addr: str | None = None, to_addrs: Any = None) -> dict[str, Any]:
        self._connection.calls.append(("send_message", (from_addr, to_addrs)))
        if self._recorder.send_delay_s:
            time.sleep(self._recorder.send_delay_s)
        self._maybe_fail("send")
        self._recorder.sent.append(msg)
        return {}

    def quit(self) -> tuple[int, bytes]:
        self._connection.calls.append(("quit", ()))
        self._recorder.leave()
        return 221, b"bye"

    def close(self) -> None:
        self._connection.calls.append(("close", ()))


class FakeSleep:
    """Replacement for ``asyncio.sleep`` that records the durations and advances the manual clock."""

    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)
        await asyncio.sleep(0)


# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------
@pytest.fixture
def seeded_db(db: Database) -> Database:
    seed_instruments(db)
    return db


@pytest.fixture
def store(seeded_db: Database) -> SettingsStore:
    return SettingsStore(seeded_db)


@pytest.fixture
def smtp() -> SmtpRecorder:
    return SmtpRecorder()


@pytest.fixture
def fake_sleep(clock: ManualClock) -> FakeSleep:
    return FakeSleep(clock)


@pytest.fixture
def make_signal(seeded_db: Database, clock: ManualClock) -> Callable[..., Signal]:
    """Store a signal (defaults: actionable LONG NVDA, 15m trend pullback) and return it with its id."""

    def make(**overrides: Any) -> Signal:
        values: dict[str, Any] = {
            "created_ts": clock.ts(),
            "issued_ts": clock.ts(),
            "data_ts": clock.ts() - 60,
            "symbol": "NVDA",
            "direction": "LONG",
            "setup": "trend_pullback",
            "timeframe": "15m",
            "horizon_min": 390,
            "ref_price": 142.35,
            "atr": 1.2,
            "stop_dist": 1.45,
            "target_dist": 2.9,
            "score": 0.64,
            "confidence": 0.58,
            "engine_version": "test",
            "expected_r": 0.32,
            "llm_verdict": "confirm",
            "llm_conviction": 0.7,
            "rationale_de": "Aufwärtstrend intakt, Rücksetzer an die EMA 20.",
            "risks_de": "Quartalszahlen in zwei Tagen.",
            "status": "pending_entry",
            "actionable": True,
        }
        values.update(overrides)
        return repo_signals.insert_signal(seeded_db, Signal(**values))

    return make


@pytest.fixture
def make_mailer(
    tmp_path: Path,
    seeded_db: Database,
    clock: ManualClock,
    bus: EventBus,
    store: SettingsStore,
    smtp: SmtpRecorder,
    fake_sleep: FakeSleep,
) -> Iterator[Callable[..., AlertMailer]]:
    """Build an ``AlertMailer`` on the test database: ``settings`` overrides, keyword arguments for the mailer."""

    def make(settings: dict[str, Any] | None = None, **kwargs: Any) -> AlertMailer:
        options: dict[str, Any] = {
            "smtp_factory": smtp.factory("plain"),
            "smtp_ssl_factory": smtp.factory("ssl"),
            "sleep": fake_sleep,
        }
        options.update(kwargs)
        return AlertMailer(
            settings=make_settings(tmp_path, **(settings or {})),
            db=seeded_db,
            clock=clock,
            bus=bus,
            runtime_settings=store,
            **options,
        )

    yield make
