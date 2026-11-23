"""build_services: the universe migration runs right after the database is opened."""

from pathlib import Path

from trading_local.clock import ManualClock
from trading_local.data import RoutedMarketData
from trading_local.db import repo_market, repo_ops
from trading_local.runtime.wiring import build_services
from trading_local.testing import make_settings
from trading_local.universe import DEFAULT_INSTRUMENTS, UNIVERSE_VERSION, UNIVERSE_VERSION_KEY

T0 = 1_795_300_000


def test_build_services_migrates_the_universe_once(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, offline=True)
    services = build_services(settings, clock=ManualClock(T0))
    try:
        rows = repo_market.list_instruments(services.db, active_only=False)
        assert {i.symbol for i in rows} == {i.symbol for i in DEFAULT_INSTRUMENTS}
        assert repo_ops.get_setting(services.db, UNIVERSE_VERSION_KEY) == UNIVERSE_VERSION
        entries = [e for e in repo_ops.list_activity(services.db, category="system") if "Instrumentenliste" in e.message]
        assert len(entries) == 1 and entries[0].ts == T0
        assert isinstance(services.data, RoutedMarketData)  # Kraken pairs go to their own lane
        assert services.instruments.get("BTC-EUR") is not None and services.instruments.get("BTC-USD") is None
    finally:
        services.db.close()

    again = build_services(settings, clock=ManualClock(T0 + 60))
    try:
        entries = [e for e in repo_ops.list_activity(again.db, category="system") if "Instrumentenliste" in e.message]
        assert len(entries) == 1  # nothing to migrate the second time
    finally:
        again.db.close()
