"""Fixtures of the web tests: a Services container of fakes plus the real repositories on the test database."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from trading_local.clock import ManualClock, to_ts
from trading_local.config import DEFAULT_COSTS, Settings
from trading_local.db import repo_market, repo_news, repo_signals
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.models import NewsAnalysis, NewsItem, Quote, Signal
from trading_local.services import Services
from trading_local.testing import FakeCalendar, make_bars, make_settings, seed_bars, seed_instruments
from trading_local.web import create_app
from trading_local.web.ws import HubTiming

from .fakes import (
    FakeBroker,
    FakeCharts,
    FakeEngine,
    FakeFx,
    FakeLlm,
    FakeMailer,
    FakeMarketData,
    FakeNews,
    FakeRuntime,
)

BASE_URL = "http://127.0.0.1:8000"
ORIGIN = BASE_URL
WS_URL = "ws://127.0.0.1:8000/ws"
SMTP_PASSWORD = "Geheimes-Passwort-4711"
MAIL_TO = "empfaenger@example.invalid"
MAIL_FROM = "absender@example.invalid"

FRIDAY = datetime(2026, 11, 20, tzinfo=UTC)
THURSDAY = FRIDAY - timedelta(days=1)
NOW = FRIDAY + timedelta(hours=15)  # the conftest clock: Friday 15:00 UTC
NOW_TS = to_ts(NOW)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Test settings with an SMTP password and full addresses, so leak checks have something to find."""
    return make_settings(tmp_path, smtp_password=SMTP_PASSWORD, smtp_user=MAIL_FROM, mail_to=MAIL_TO,
                         mail_from=MAIL_FROM)


@dataclass
class Fakes:
    data: FakeMarketData
    llm: FakeLlm
    news: FakeNews
    mailer: FakeMailer
    broker: FakeBroker
    engine: FakeEngine
    runtime: FakeRuntime


@pytest.fixture
def fakes(clock: ManualClock) -> Fakes:
    return Fakes(
        data=FakeMarketData(),
        llm=FakeLlm(),
        news=FakeNews(),
        mailer=FakeMailer(MAIL_TO),
        broker=FakeBroker(now_ts=clock.ts()),
        engine=FakeEngine(),
        runtime=FakeRuntime(started_ts=clock.ts() - 600),
    )


@pytest.fixture
def services(settings: Settings, db: Database, clock: ManualClock, bus: EventBus, fakes: Fakes) -> Services:
    seed_instruments(db)
    return Services(
        settings=settings,
        db=db,
        clock=clock,
        bus=bus,
        instruments=InstrumentBook(db),
        runtime_settings=SettingsStore(db),
        costs=DEFAULT_COSTS,
        calendar=FakeCalendar(),
        data=fakes.data,
        fx=FakeFx(),
        llm=fakes.llm,
        news=fakes.news,
        charts=FakeCharts(),
        mailer=fakes.mailer,
        broker=fakes.broker,
        engine=fakes.engine,
        runtime=fakes.runtime,
    )


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "static"
    (directory / "assets").mkdir(parents=True)
    (directory / "index.html").write_text("<!doctype html><title>Trading Local</title>", encoding="utf-8")
    files = {
        "app.mjs": "export const x = 1;",
        "assets/main.js": "console.log(1);",
        "assets/style.css": "body { color: black; }",
        "assets/icon.svg": "<svg xmlns='http://www.w3.org/2000/svg'/>",
        "assets/data.json": "{}",
        "assets/app.mjs.map": "{}",
    }
    for name, text in files.items():
        (directory / name).write_text(text, encoding="utf-8")
    for name, payload in {"assets/font.woff2": b"wOF2", "assets/logo.png": b"\x89PNG", "favicon.ico": b"\0\0"}.items():
        (directory / name).write_bytes(payload)
    return directory


@pytest.fixture
def app(services: Services, static_dir: Path) -> FastAPI:
    application = create_app(services, static_dir=static_dir)
    application.state.hub.timing = HubTiming(status_interval_s=3600, ping_interval_s=3600, quote_flush_s=0.05)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as test_client:
        yield test_client


# ------------------------------------------------------------------------------------------------
# Seeding helpers
# ------------------------------------------------------------------------------------------------
def at(day: datetime, hour: int, minute: int = 0) -> int:
    return to_ts(day + timedelta(hours=hour, minutes=minute))


def seed_minutes(
    db: Database, symbol: str, start_ts: int, minutes: int, *, seed: int = 1, price: float = 100.0
) -> None:
    seed_bars(db, symbol, make_bars(start_ts, minutes, start_price=price, seed=seed))


def seed_market(db: Database) -> None:
    """Bars around the test clock (Friday 15:00 UTC, FakeCalendar sessions)."""
    seed_minutes(db, "NVDA", at(THURSDAY, 14, 30), 390, seed=1, price=180.0)
    seed_minutes(db, "NVDA", at(FRIDAY, 14, 30), 31, seed=2, price=182.0)  # the 15:00 bar is still open
    seed_minutes(db, "SAP.DE", at(FRIDAY, 8), 421, seed=3, price=240.0)
    seed_minutes(db, "7203.T", at(FRIDAY, 0), 150, seed=4, price=2900.0)
    seed_minutes(db, "7203.T", at(FRIDAY, 3, 30), 180, seed=5, price=2910.0)
    seed_minutes(db, "^GSPC", at(THURSDAY, 14, 30), 390, seed=6, price=6000.0)  # nothing yet on Friday
    seed_minutes(db, "BTC-USD", at(FRIDAY, 12), 181, seed=7, price=90_000.0)


def seed_daily(db: Database, symbol: str, last_day: date, days: int) -> list[str]:
    keys = [(last_day - timedelta(days=offset)).isoformat() for offset in range(days)][::-1]
    close = 100.0 + np.arange(days, dtype=np.float64)
    frame = pd.DataFrame(
        {"open": close - 0.5, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0},
        index=pd.Index(keys, dtype="str", name="day"),
    )
    repo_market.upsert_bars_1d(db, symbol, frame)
    return keys


def quote(symbol: str, price: float, ts: int) -> Quote:
    return Quote(symbol=symbol, ts=ts, received_ts=ts, price=price, prev_close=price * 0.99, change_pct=1.0)


def insert_signal(
    db: Database,
    symbol: str,
    *,
    status: str,
    actionable: bool,
    created_ts: int,
    news_ids: list[int] | None = None,
    entry_price: float | None = None,
) -> Signal:
    signal = Signal(
        created_ts=created_ts, data_ts=created_ts, symbol=symbol, direction="LONG", setup="breakout",
        timeframe="15m", horizon_min=120, ref_price=180.0, atr=2.0, stop_dist=3.0, target_dist=6.0, score=0.7,
        confidence=0.6, engine_version="engine-test-1", status=status, actionable=actionable,
        news_ids=news_ids or [], entry_price=entry_price,
    )
    return repo_signals.insert_signal(db, signal)


def insert_news(db: Database, uid: str, title: str, fetched_ts: int, *, source: str = "reuters_business",
                symbols: dict[str, float] | None = None, relevance: int = 4) -> int:
    news_item = NewsItem(
        uid=uid, title_key=uid, source=source, url=f"https://example.invalid/{uid}", title=title, fetched_ts=fetched_ts
    )
    item = repo_news.insert_news(db, [news_item])[0]
    assert item.id is not None
    if symbols is not None:
        repo_news.save_analysis(db, NewsAnalysis(news_id=item.id, analyzed_ts=fetched_ts + 30, model="test",
                                                 relevance=relevance, sentiment=0.4, category="earnings",
                                                 horizon="days", summary_de="Zusammenfassung", symbols=symbols))
    return item.id
