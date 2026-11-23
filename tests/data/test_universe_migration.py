"""Kraken crypto instruments of the default universe and the versioned universe migration."""

from dataclasses import replace

import pytest

from trading_local.clock import ManualClock
from trading_local.data.kraken import KRAKEN_PAIRS
from trading_local.db import repo_market, repo_ops, repo_signals
from trading_local.db.database import Database
from trading_local.models import Instrument, Signal
from trading_local.testing import make_bars, seed_bars
from trading_local.universe import (
    DEFAULT_INSTRUMENTS,
    REGION_SORT_BASE,
    UNIVERSE_VERSION,
    UNIVERSE_VERSION_KEY,
    migrate_universe,
)

KRAKEN = [i for i in DEFAULT_INSTRUMENTS if i.exchange == "KRAKEN"]
BY_SYMBOL = {i.symbol: i for i in DEFAULT_INSTRUMENTS}
RETIRED = ("BTC-USD", "ETH-USD", "SOL-USD")
T0 = 1_795_000_000 // 60 * 60


def old_universe() -> list[Instrument]:
    """The default universe as version 1 stored it: Yahoo crypto in USD, the 10-minute COMEX/NYMEX delays."""
    rows = [i for i in DEFAULT_INSTRUMENTS if i.exchange != "KRAKEN"]
    rows = [replace(i, delay_min=10) if i.symbol in ("GC=F", "CL=F", "BZ=F") else i for i in rows]
    crypto = [
        Instrument(symbol, name, "crypto", "GLOBAL", "CRYPTO", "USD", timezone="UTC", sort_order=4010 + 10 * n,
                   aliases=(name,))
        for n, (symbol, name) in enumerate(zip(RETIRED, ("Bitcoin", "Ethereum", "Solana"), strict=True))
    ]  # fmt: skip
    return [*crypto, *rows]


def stored(db: Database) -> dict[str, Instrument]:
    return {i.symbol: i for i in repo_market.list_instruments(db, active_only=False)}


def make_signal(symbol: str) -> Signal:
    return Signal(
        created_ts=T0, data_ts=T0, symbol=symbol, direction="LONG", setup="breakout", timeframe="15m",
        horizon_min=60, ref_price=50_000.0, atr=100.0, stop_dist=100.0, target_dist=200.0, score=0.6,
        confidence=0.5, engine_version="test",
    )  # fmt: skip


# ------------------------------------------------------------------------------------------------
# The Kraken instruments
# ------------------------------------------------------------------------------------------------
def test_ten_kraken_coins_against_eur_with_german_names() -> None:
    assert [(i.symbol, i.name) for i in KRAKEN] == [
        ("BTC-EUR", "Bitcoin"), ("ETH-EUR", "Ethereum"), ("SOL-EUR", "Solana"), ("XRP-EUR", "XRP"),
        ("ADA-EUR", "Cardano"), ("DOGE-EUR", "Dogecoin"), ("AVAX-EUR", "Avalanche"), ("LINK-EUR", "Chainlink"),
        ("DOT-EUR", "Polkadot"), ("LTC-EUR", "Litecoin"),
    ]  # fmt: skip
    for instrument in KRAKEN:
        assert (instrument.asset_class, instrument.region, instrument.currency, instrument.timezone) == (
            "crypto", "GLOBAL", "EUR", "UTC"
        )
        assert instrument.delay_min == 0 and instrument.tradable and instrument.active
        assert instrument.price_scale == 1.0
        assert instrument.benchmark == (None if instrument.symbol == "BTC-EUR" else "BTC-EUR")
        assert instrument.symbol in KRAKEN_PAIRS  # every coin has its pair metadata
        assert KRAKEN_PAIRS[instrument.symbol].ws_symbol == instrument.symbol.replace("-", "/")
        assert len(instrument.aliases) >= 2


def test_kraken_coins_lead_the_global_group() -> None:
    glob = [i for i in DEFAULT_INSTRUMENTS if i.region == "GLOBAL"]
    assert [i.symbol for i in glob[:10]] == [i.symbol for i in KRAKEN]
    assert all(REGION_SORT_BASE["GLOBAL"] < i.sort_order < REGION_SORT_BASE["GLOBAL"] + 900 for i in glob)
    assert set(KRAKEN_PAIRS) == {i.symbol for i in KRAKEN}


# ------------------------------------------------------------------------------------------------
# Migration
# ------------------------------------------------------------------------------------------------
def test_fresh_database_gets_exactly_the_default_universe(db: Database, clock: ManualClock) -> None:
    changes = migrate_universe(db, clock=clock)
    assert stored(db) == {i.symbol: i for i in DEFAULT_INSTRUMENTS}
    assert len(changes) == 1 and "angelegt" in changes[0] and str(len(DEFAULT_INSTRUMENTS)) in changes[0]
    assert repo_ops.get_setting(db, UNIVERSE_VERSION_KEY) == UNIVERSE_VERSION
    assert migrate_universe(db, clock=clock) == []  # idempotent


def test_existing_database_is_migrated_once_keeping_history_and_user_flags(db: Database, clock: ManualClock) -> None:
    repo_market.upsert_instruments(db, old_universe(), keep_user_flags=False)
    user_added = Instrument("RIO.L", "Rio Tinto", "equity", "EU", "XLON", "GBP", price_scale=0.01,
                            timezone="Europe/London", sort_order=2900)  # fmt: skip
    repo_market.upsert_instruments(db, [user_added], keep_user_flags=False)
    repo_market.set_instrument_flags(db, "NVDA", active=False, tradable=False)  # the user's choice
    seed_bars(db, "BTC-USD", make_bars(T0, 30))
    signal = repo_signals.insert_signal(db, make_signal("BTC-USD"))

    changes = migrate_universe(db, clock=clock)

    rows = stored(db)
    for symbol in RETIRED:  # deactivated, not deleted
        assert symbol in rows and not rows[symbol].active and not rows[symbol].tradable
    assert repo_market.bar_range(db, "BTC-USD") == (T0, T0 + 29 * 60)  # bars stay
    assert signal.id is not None and repo_signals.get_signal(db, signal.id) is not None  # signals stay
    for instrument in KRAKEN:
        assert rows[instrument.symbol] == instrument
    for symbol in ("GC=F", "CL=F", "BZ=F"):  # corrected descriptive data reaches the existing rows
        assert rows[symbol].delay_min == 30
    assert not rows["NVDA"].active and not rows["NVDA"].tradable  # user flags kept
    assert rows["NVDA"] == replace(BY_SYMBOL["NVDA"], active=False, tradable=False)
    assert rows["RIO.L"] == user_added  # instruments the user added are left alone
    assert set(rows) == {*BY_SYMBOL, *RETIRED, "RIO.L"}

    text = " ".join(changes)
    assert "Version 2" in text and "BTC-EUR (Bitcoin)" in text and "GC=F" in text
    assert "Deaktiviert" in text and all(symbol in text for symbol in RETIRED)
    assert "Achtung" not in text
    assert migrate_universe(db, clock=clock) == []


def test_a_retired_symbol_the_user_reactivates_stays_active(db: Database, clock: ManualClock) -> None:
    repo_market.upsert_instruments(db, old_universe(), keep_user_flags=False)
    migrate_universe(db, clock=clock)
    repo_market.set_instrument_flags(db, "BTC-USD", active=True)
    assert migrate_universe(db, clock=clock) == []
    assert stored(db)["BTC-USD"].active


def test_open_positions_in_retired_symbols_are_reported(db: Database, clock: ManualClock) -> None:
    repo_market.upsert_instruments(db, old_universe(), keep_user_flags=False)
    with db.write() as conn:
        conn.execute(
            "INSERT INTO portfolios (key, name, base_currency, start_cash, cash, created_ts, reset_ts) "
            "VALUES ('me', 'Ich', 'EUR', 100000, 100000, ?, ?)",
            (T0, T0),
        )
        pid = conn.execute("SELECT id FROM portfolios WHERE key = 'me'").fetchone()[0]
        conn.execute(
            "INSERT INTO positions (portfolio_id, symbol, side, qty, entry_price, entry_fx, opened_ts) "
            "VALUES (?, 'ETH-USD', 'LONG', 1.0, 1700.0, 0.86, ?)",
            (pid, T0),
        )
    changes = migrate_universe(db, clock=clock)
    warning = [line for line in changes if line.startswith("Achtung")]
    assert len(warning) == 1 and "ETH-USD" in warning[0] and "BTC-USD" not in warning[0]


def test_running_signals_in_retired_symbols_are_reported(db: Database, clock: ManualClock) -> None:
    """The real database had an active SOL-USD signal at the switch: its tracking needs bars that no longer come."""
    repo_market.upsert_instruments(db, old_universe(), keep_user_flags=False)
    repo_signals.insert_signal(db, replace(make_signal("SOL-USD"), status="active"))
    repo_signals.insert_signal(db, replace(make_signal("BTC-USD"), status="stopped"))  # closed: nothing to say
    changes = migrate_universe(db, clock=clock)
    warning = [line for line in changes if line.startswith("Achtung")]
    assert len(warning) == 1 and "SOL-USD" in warning[0] and "BTC-USD" not in warning[0]


@pytest.mark.parametrize("version", [UNIVERSE_VERSION, UNIVERSE_VERSION + 1])
def test_an_up_to_date_database_is_not_touched(db: Database, clock: ManualClock, version: int) -> None:
    repo_market.upsert_instruments(db, old_universe(), keep_user_flags=False)
    repo_ops.set_setting(db, UNIVERSE_VERSION_KEY, version, T0)
    assert migrate_universe(db, clock=clock) == []
    assert "BTC-EUR" not in stored(db) and stored(db)["BTC-USD"].active
