from zoneinfo import ZoneInfo

from trading_local.db import repo_market
from trading_local.db.database import Database
from trading_local.universe import (
    DEFAULT_INSTRUMENTS,
    REGION_SORT_BASE,
    TRADABLE_INDICES,
    VENUE_BENCHMARK,
    VENUE_KEYS,
    VENUE_REGION,
    VENUE_TIMEZONE,
    YAHOO_EXCHANGE_TO_VENUE,
    normalize_currency,
    seed_default_universe,
)

# The venue keys of the calendar brief.
CALENDAR_KEYS = set(
    "XNYS XNAS XETR XFRA XLON XPAR XAMS XSWX XCSE XTKS XHKG XSHG XSHE XKRX XTAI XNSE XASX CRYPTO KRAKEN FX CME".split()
)
BY_SYMBOL = {instrument.symbol: instrument for instrument in DEFAULT_INSTRUMENTS}


def test_symbols_are_unique_and_the_retired_roche_line_is_replaced() -> None:
    assert len(BY_SYMBOL) == len(DEFAULT_INSTRUMENTS) == 115
    assert "ROG.SW" not in BY_SYMBOL and "ROP.SW" in BY_SYMBOL
    assert not {"BTC-USD", "ETH-USD", "SOL-USD"} & set(BY_SYMBOL)  # replaced by the Kraken EUR pairs


def test_every_exchange_is_a_known_venue_key_with_region_and_timezone() -> None:
    assert set(VENUE_KEYS) == CALENDAR_KEYS
    assert set(YAHOO_EXCHANGE_TO_VENUE.values()) <= CALENDAR_KEYS
    assert set(VENUE_REGION) == set(VENUE_TIMEZONE) == CALENDAR_KEYS
    for instrument in DEFAULT_INSTRUMENTS:
        assert instrument.exchange in CALENDAR_KEYS, instrument.symbol
        assert instrument.timezone == VENUE_TIMEZONE[instrument.exchange]
        ZoneInfo(instrument.timezone)


def test_every_non_eur_currency_has_an_fx_pair_to_eur() -> None:
    fx = {i.currency: i for i in DEFAULT_INSTRUMENTS if i.asset_class == "fx" and i.symbol.startswith("EUR")}
    for instrument in DEFAULT_INSTRUMENTS:
        if instrument.currency != "EUR":
            pair = fx.get(instrument.currency)
            assert pair is not None, f"no EUR pair for {instrument.currency} ({instrument.symbol})"
            assert pair.symbol == f"EUR{instrument.currency}=X"


def test_benchmarks_exist_and_follow_the_rules() -> None:
    for instrument in DEFAULT_INSTRUMENTS:
        if instrument.asset_class == "equity":
            assert instrument.benchmark == VENUE_BENCHMARK[instrument.exchange]
            assert BY_SYMBOL[instrument.benchmark].asset_class == "index"
        elif instrument.exchange == "KRAKEN":  # the altcoins take Bitcoin as their market context
            assert instrument.benchmark == (None if instrument.symbol == "BTC-EUR" else "BTC-EUR"), instrument.symbol
        else:
            assert instrument.benchmark is None, instrument.symbol
    assert BY_SYMBOL["SAP.DE"].benchmark == "^GDAXI"
    assert BY_SYMBOL["ASML.AS"].benchmark == "^STOXX50E"
    assert BY_SYMBOL["NVDA"].benchmark == "^GSPC"
    assert BY_SYMBOL["7203.T"].benchmark == "^N225"
    assert BY_SYMBOL["600519.SS"].benchmark == "^HSI"
    assert BY_SYMBOL["RELIANCE.NS"].benchmark == "^NSEI"


def test_tradable_flags() -> None:
    assert TRADABLE_INDICES == {"^GSPC", "^NDX", "^DJI", "^GDAXI", "^STOXX50E", "^FTSE", "^N225", "^HSI"}
    for instrument in DEFAULT_INSTRUMENTS:
        if instrument.asset_class in ("equity", "crypto"):
            assert instrument.tradable, instrument.symbol
        elif instrument.asset_class == "index":
            assert instrument.tradable == (instrument.symbol in TRADABLE_INDICES), instrument.symbol
        else:
            assert not instrument.tradable, instrument.symbol
        assert instrument.active


def test_pence_quotes_and_currency_normalisation() -> None:
    for symbol in ("SHEL.L", "AZN.L", "HSBA.L"):
        assert (BY_SYMBOL[symbol].currency, BY_SYMBOL[symbol].price_scale) == ("GBP", 0.01)
    assert (BY_SYMBOL["^FTSE"].currency, BY_SYMBOL["^FTSE"].price_scale) == ("GBP", 1.0)
    assert normalize_currency("GBp") == ("GBP", 0.01)
    assert normalize_currency("ZAc") == ("ZAR", 0.01)
    assert normalize_currency("usd") == ("USD", 1.0)


def test_sort_order_groups_by_region() -> None:
    orders = [instrument.sort_order for instrument in DEFAULT_INSTRUMENTS]
    assert orders == sorted(orders) and len(set(orders)) == len(orders)
    for instrument in DEFAULT_INSTRUMENTS:
        base = REGION_SORT_BASE[instrument.region]
        assert base < instrument.sort_order < base + 900
    regions = [instrument.region for instrument in DEFAULT_INSTRUMENTS]
    assert regions == sorted(regions, key=["ASIA", "EU", "US", "GLOBAL"].index)
    assert BY_SYMBOL["ES=F"].region == "US" and BY_SYMBOL["GC=F"].region == "GLOBAL"


def test_yahoo_exchange_mapping_covers_the_required_codes() -> None:
    required = {
        "NMS": "XNAS", "NGM": "XNAS", "NCM": "XNAS", "NYQ": "XNYS", "PCX": "XNYS", "ASE": "XNYS", "BTS": "XNYS",
        "GER": "XETR", "FRA": "XFRA", "LSE": "XLON", "PAR": "XPAR", "AMS": "XAMS", "EBS": "XSWX", "CPH": "XCSE",
        "JPX": "XTKS", "HKG": "XHKG", "SHH": "XSHG", "SHZ": "XSHE", "KSC": "XKRX", "KOE": "XKRX", "TAI": "XTAI",
        "TWO": "XTAI", "NSI": "XNSE", "BSE": "XNSE", "ASX": "XASX", "CCC": "CRYPTO", "CCY": "FX", "CME": "CME",
        "CMX": "CME", "NYM": "CME", "CBT": "CME",
    }  # fmt: skip
    for code, venue in required.items():
        assert YAHOO_EXCHANGE_TO_VENUE[code] == venue, code


def test_conservative_feed_delays() -> None:
    assert BY_SYMBOL["BTC-EUR"].delay_min == 0 and BY_SYMBOL["EURUSD=X"].delay_min == 0
    assert BY_SYMBOL["SAP.DE"].delay_min == 15
    assert BY_SYMBOL["NESN.SW"].delay_min == 30  # Yahoo help page states 30 minutes for Swiss data
    assert BY_SYMBOL["ES=F"].delay_min == 10
    for symbol in ("GC=F", "CL=F", "BZ=F"):  # COMEX/NYMEX: API field 10, help page 30 -> the larger value
        assert BY_SYMBOL[symbol].delay_min == 30, symbol


def test_seed_keeps_the_user_flags(db: Database) -> None:
    assert seed_default_universe(db) == len(DEFAULT_INSTRUMENTS)
    repo_market.set_instrument_flags(db, "NVDA", active=False, tradable=False)
    seed_default_universe(db)
    stored = repo_market.get_instrument(db, "NVDA")
    assert stored is not None and not stored.active and not stored.tradable
    assert stored.aliases == ("Nvidia",)
    assert len(repo_market.list_instruments(db, active_only=False)) == len(DEFAULT_INSTRUMENTS)
