"""Crypto derivatives: storage, as-of metrics, the minute poll, the backfill and the liquidation trigger."""

from collections.abc import Collection
from datetime import datetime

import pytest

from trading_local.clock import ManualClock
from trading_local.contracts import Derivatives
from trading_local.data import build_derivatives
from trading_local.data.derivatives import KrakenDerivatives, burst_message_de, is_burst
from trading_local.data.kraken_futures import PerpetualTicker
from trading_local.data.provider import ProviderError
from trading_local.db import repo_derivs, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_derivs import DerivSnapshot
from trading_local.db.repo_market import InstrumentBook
from trading_local.db.repo_ops import SettingsStore
from trading_local.events import EventBus
from trading_local.testing import seed_instruments

HOUR = 3600
DAY = 86_400
FUNDING = 9.2749625e-06  # relative funding per hour of PF_XBTUSD on 2026-11-24


class FakeClient:
    """``KrakenFuturesClient`` double with canned answers."""

    def __init__(self) -> None:
        self.tickers_answer: dict[str, PerpetualTicker] | Exception = {
            "PF_XBTUSD": PerpetualTicker("PF_XBTUSD", 60_000.0, 59_970.0, 2080.0, FUNDING, 1.0e-05)
        }
        self.series_data: dict[tuple[str, ...], list[tuple[int, float]]] = {}  # (perpetual, kind[, interval])
        self.funding: list[tuple[int, float]] = []
        self.calls: list[tuple[str, ...]] = []
        self.closed = False

    async def tickers(self) -> dict[str, PerpetualTicker]:
        self.calls.append(("tickers",))
        if isinstance(self.tickers_answer, Exception):
            raise self.tickers_answer
        return self.tickers_answer

    async def series(
        self, perpetual: str, kind: str, *, interval_s: int, since_ts: int, background: bool = False
    ) -> list[tuple[int, float]]:
        self.calls.append(("series", perpetual, kind, str(interval_s), str(since_ts)))
        data = self.series_data.get((perpetual, kind, str(interval_s)), self.series_data.get((perpetual, kind), []))
        return [point for point in data if point[0] >= since_ts]

    async def funding_history(self, perpetual: str) -> list[tuple[int, float]]:
        self.calls.append(("funding", perpetual))
        return list(self.funding)

    async def close(self) -> None:
        self.closed = True


class RecordingNews:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def prioritize(self, symbols: Collection[str], now: datetime) -> int:
        self.calls.append(list(symbols))
        return 0


@pytest.fixture
def parts(db: Database, clock: ManualClock, bus: EventBus) -> tuple[KrakenDerivatives, FakeClient, RecordingNews]:
    seed_instruments(db)  # BTC-USD is the only crypto instrument -> PF_XBTUSD
    client, news = FakeClient(), RecordingNews()
    service = KrakenDerivatives(
        db=db,
        clock=clock,
        bus=bus,
        instruments=InstrumentBook(db),
        runtime_settings=SettingsStore(db),
        news=news,  # type: ignore[arg-type]
        client_factory=lambda: client,  # type: ignore[arg-type,return-value]
    )
    return service, client, news


def liquidation_history(end_ts: int, *, days: float, per_minute: float) -> list[tuple[int, float]]:
    start = end_ts - int(days * DAY)
    return [(ts, per_minute) for ts in range(start, end_ts, 60)]


# ------------------------------------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------------------------------------
def test_metrics_rank_funding_and_measure_changes(db: Database, clock: ManualClock) -> None:
    now = clock.ts()
    hourly = [DerivSnapshot("BTC-USD", now - h * HOUR, now - h * HOUR, funding_rate=1e-05 + (h % 2) * 2e-06)
              for h in range(1, 30 * 24)]  # fmt: skip
    repo_derivs.upsert_snapshots(db, hourly)
    repo_derivs.upsert_snapshots(db, [
        DerivSnapshot("BTC-USD", now - DAY, now - DAY, open_interest=2000.0),
        DerivSnapshot("BTC-USD", now - HOUR - 300, now - HOUR, open_interest=2060.0),
        DerivSnapshot("BTC-USD", now, now, FUNDING * 3, 1e-05, 2080.0, 60_000.0, 59_970.0),
    ])  # fmt: skip
    repo_derivs.upsert_liquidations(db, "BTC-USD", liquidation_history(now, days=2, per_minute=0.01), received_ts=now)

    metrics = repo_derivs.metrics(db, "BTC-USD", asof_ts=now + 30)
    assert metrics is not None and not metrics.stale
    assert metrics.funding_annual_pct == pytest.approx(FUNDING * 3 * 24 * 365 * 100)
    assert metrics.funding_z is not None and metrics.funding_z > 10  # far above the 30-day funding
    assert metrics.oi_change_1h_pct == pytest.approx((2080 / 2060 - 1) * 100)
    assert metrics.oi_change_24h_pct == pytest.approx(4.0)
    assert metrics.basis_pct == pytest.approx((60_000 / 59_970 - 1) * 100)
    assert metrics.liq_15m == pytest.approx(0.15) and metrics.liq_60m == pytest.approx(0.6)
    assert metrics.liq_60m_usd == pytest.approx(0.6 * 60_000) and metrics.liq_5m_ratio == pytest.approx(1.0)
    assert repo_derivs.metrics(db, "ETH-USD", asof_ts=now) is None


def test_metrics_never_see_rows_learned_later(db: Database, clock: ManualClock) -> None:
    now = clock.ts()
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now - 600, now - 600, FUNDING, None, 2000.0, 60_000.0)])
    # a backfill stored now: values for an hour ago, learned only now
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now - HOUR - 600, now, open_interest=1000.0)])
    repo_derivs.upsert_liquidations(db, "BTC-USD", [(now - 900, 5.0)], received_ts=now)
    early = repo_derivs.metrics(db, "BTC-USD", asof_ts=now - 300)
    assert early is not None and early.oi_change_1h_pct is None and early.liq_60m is None
    late = repo_derivs.metrics(db, "BTC-USD", asof_ts=now + 60)
    assert late is not None and late.oi_change_1h_pct == pytest.approx(100.0) and late.liq_60m == 5.0
    assert late.stale is False and repo_derivs.metrics(db, "BTC-USD", asof_ts=now + HOUR).stale is True  # type: ignore[union-attr]


def test_backfill_rows_never_replace_live_rows_and_prune_keeps_hours(db: Database, clock: ManualClock) -> None:
    now = clock.ts() // HOUR * HOUR
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now, now, FUNDING, None, 2080.0, 60_000.0)])
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now, now + 5, funding_rate=1.0)], replace=False)
    assert db.scalar("SELECT funding_rate FROM deriv_snapshots WHERE ts = ?", (now,)) == FUNDING
    assert db.scalar("SELECT received_ts FROM deriv_snapshots WHERE ts = ?", (now,)) == now  # nothing filled
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now, now + 9, index_price=59_000.0)], replace=False)
    row = db.query_one("SELECT index_price, funding_rate, received_ts FROM deriv_snapshots WHERE ts = ?", (now,))
    assert row is not None and tuple(row) == (59_000.0, FUNDING, now + 9)  # filled: learned only now
    repo_derivs.upsert_snapshots(db, [DerivSnapshot("BTC-USD", now - 60 * DAY + 60, 0, open_interest=1.0),
                                      DerivSnapshot("BTC-USD", now - 60 * DAY, 0, funding_rate=2e-05)])
    repo_derivs.upsert_liquidations(db, "BTC-USD", [(now - 60 * DAY, 1.0), (now - 60, 2.0)], received_ts=now)
    assert repo_derivs.prune(db, before_ts=now - DAY) == 2
    assert db.scalar("SELECT COUNT(*) FROM deriv_snapshots") == 2  # the full hour survives
    assert repo_derivs.newest_ts(db, "deriv_liquidations", "BTC-USD") == now - 60
    with pytest.raises(ValueError):
        repo_derivs.newest_ts(db, "news", "BTC-USD")


# ------------------------------------------------------------------------------------------------
# Rules and texts
# ------------------------------------------------------------------------------------------------
def test_burst_rule() -> None:
    quiet = [0.05] * 2000
    assert is_burst(5.0, quiet, usd=300_000.0)
    assert not is_burst(0.4, quiet, usd=300_000.0)  # less than 10 x usual
    assert not is_burst(5.0, quiet, usd=20_000.0)  # below the USD floor
    assert not is_burst(5.0, [*quiet[:-30], *([6.0] * 30)], usd=300_000.0)  # bursts this big are not rare
    assert not is_burst(0.0, quiet, usd=0.0) and not is_burst(5.0, [], usd=300_000.0)


def test_burst_message() -> None:
    assert burst_message_de("BTC-EUR", 2.4, 143_600.0, 18.2, 0) == (
        "Liquidationswelle: BTC-EUR 2,40 BTC (≈ 143.600 USD) in 5 Min, 18-fach üblich – Nachrichten vorgezogen"
    )
    assert burst_message_de("DOGE-EUR", 1_250_000.0, 70_000.0, None, 2).endswith(
        "DOGE (≈ 70.000 USD) in 5 Min – Nachrichten vorgezogen, 2 Meldungen erneut eingereiht"
    )


# ------------------------------------------------------------------------------------------------
# Service
# ------------------------------------------------------------------------------------------------
async def test_poll_stores_the_snapshot_and_complete_minutes(
    parts: tuple[KrakenDerivatives, FakeClient, RecordingNews], db: Database, clock: ManualClock
) -> None:
    service, client, _ = parts
    assert isinstance(service, Derivatives) and service.targets() == {"BTC-USD": "PF_XBTUSD"}
    minute = clock.ts()
    client.series_data[("PF_XBTUSD", "liquidation-volume")] = [(minute - 120, 0.5), (minute - 60, 0.25), (minute, 9.0)]
    await service.start()
    clock.advance(20)
    assert await service.poll(clock.now()) == 3  # one snapshot, two complete minutes (the current one is not)
    row = db.query_one("SELECT * FROM deriv_snapshots WHERE ts = ?", (minute,))
    assert row is not None and row["open_interest"] == 2080.0 and row["received_ts"] == minute + 20
    assert repo_derivs.newest_ts(db, "deriv_liquidations", "BTC-USD") == minute - 60
    series = [call for call in client.calls if call[0] == "series" and call[2] == "liquidation-volume"]
    assert series[-1][3:] == ("60", str(minute - 3600))  # nothing stored yet: the last hour

    metrics = service.metrics("BTC-USD")
    assert metrics is not None and metrics.liq_15m == pytest.approx(0.75)
    assert service.metrics("NVDA") is None

    client.tickers_answer = ProviderError("down")  # a failing source never breaks the poll
    clock.advance(60)
    assert await service.poll(clock.now()) >= 0
    await service.stop()
    assert client.closed and await service.poll(clock.now()) == 0


async def test_backfill_loads_history_once(
    parts: tuple[KrakenDerivatives, FakeClient, RecordingNews], db: Database, clock: ManualClock
) -> None:
    service, client, _ = parts
    now = clock.ts()
    client.funding = [(now - h * HOUR, 1e-05) for h in range(1, 70 * 24)][::-1]
    client.series_data[("PF_XBTUSD", "open-interest", "3600")] = [
        (now - h * HOUR, 2000.0 + h) for h in range(2, 30 * 24)
    ][::-1]
    client.series_data[("PF_XBTUSD", "open-interest", "300")] = [(now - k * 300, 2100.0) for k in range(2, 26 * 12)][::-1]
    client.series_data[("PF_XBTUSD", "liquidation-volume")] = liquidation_history(now, days=8, per_minute=0.01)
    await service.start()
    await service.wait_backfill()
    funding_rows = db.scalar("SELECT COUNT(*) FROM deriv_snapshots WHERE funding_rate IS NOT NULL")
    assert funding_rows == 60 * 24  # the hours of the last 60 days
    hourly = db.scalar("SELECT COUNT(*) FROM deriv_snapshots WHERE open_interest IS NOT NULL AND ts % 3600 = 0")
    assert hourly == 30 * 24 - 2
    fine = db.scalar("SELECT COUNT(*) FROM deriv_snapshots WHERE open_interest IS NOT NULL AND ts % 3600 != 0")
    assert fine == 310 - 25  # 5-minute closes of the last 26 hours (the full hours are hourly rows already)
    assert db.scalar("SELECT COUNT(*) FROM deriv_liquidations") == 7 * 1440  # 7 days
    assert db.scalar("SELECT MIN(received_ts) FROM deriv_liquidations") == now  # learned at the backfill
    clock.advance(30)
    await service.poll(clock.now())
    metrics = service.metrics("BTC-USD")
    # an hour before the poll is a full hour: its value is the hourly close (2002), stored first
    assert metrics is not None and metrics.oi_change_1h_pct == pytest.approx((2080 / 2002 - 1) * 100)
    await service.stop()

    again = KrakenDerivatives(
        db=db, clock=clock, bus=EventBus(), instruments=InstrumentBook(db), runtime_settings=SettingsStore(db),
        news=RecordingNews(), client_factory=lambda: client,  # type: ignore[arg-type,return-value]
    )  # fmt: skip
    client.calls.clear()
    await again.start()
    await again.wait_backfill()
    assert ("funding", "PF_XBTUSD") not in client.calls  # the funding history is complete already
    await again.stop()


async def test_liquidation_burst_pulls_the_news_forward_once(
    parts: tuple[KrakenDerivatives, FakeClient, RecordingNews], db: Database, clock: ManualClock
) -> None:
    service, client, news = parts
    now = clock.ts()
    history = liquidation_history(now - 300, days=2, per_minute=0.01)
    burst = [(ts, 1.0) for ts in range(now - 300, now, 60)]  # 5 BTC in 5 minutes (~300.000 USD)
    client.series_data[("PF_XBTUSD", "liquidation-volume")] = [*history, *burst]
    repo_derivs.upsert_liquidations(db, "BTC-USD", history, received_ts=now - 300)
    await service.start()
    await service.wait_backfill()
    clock.advance(10)
    await service.poll(clock.now())
    assert news.calls == [["BTC-USD"]]
    messages = [entry.message for entry in repo_ops.list_activity(db, category="data")]
    assert len(messages) == 1 and messages[0].startswith("Liquidationswelle: BTC-USD 5,00 BTC (≈ 300.000 USD) in 5 Min")

    clock.advance(60)
    await service.poll(clock.now())
    assert news.calls == [["BTC-USD"]]  # cooldown
    await service.stop()


async def test_liquidation_trigger_respects_the_setting(
    parts: tuple[KrakenDerivatives, FakeClient, RecordingNews], db: Database, clock: ManualClock
) -> None:
    service, client, news = parts
    SettingsStore(db).update({"price_trigger_enabled": False}, clock.ts())
    now = clock.ts()
    history = liquidation_history(now - 300, days=2, per_minute=0.01)
    repo_derivs.upsert_liquidations(db, "BTC-USD", history, received_ts=now - 300)
    client.series_data[("PF_XBTUSD", "liquidation-volume")] = [*history, *[(ts, 1.0) for ts in range(now - 300, now, 60)]]
    await service.start()
    await service.wait_backfill()
    await service.poll(clock.now())
    assert news.calls == []
    await service.stop()


def test_factory_builds_the_service(db: Database, clock: ManualClock, bus: EventBus) -> None:
    built = build_derivatives(
        db=db, clock=clock, bus=bus, instruments=InstrumentBook(db), runtime_settings=SettingsStore(db),
        news=RecordingNews(),  # type: ignore[arg-type]
    )  # fmt: skip
    assert isinstance(built, KrakenDerivatives) and isinstance(built, Derivatives)
