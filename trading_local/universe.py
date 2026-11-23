"""Default instrument universe and the venue tables used to classify instruments.

The universe was validated against Yahoo Finance on 2026-11-22 (one batched quote for every symbol, see the
market-data probe): names, currencies, venues, time zones and declared feed delays come from that run, the
news aliases from the alias table of the news probe. ROG.SW no longer exists at Yahoo and is replaced by
ROP.SW (the liquid Roche participation certificate).

Crypto (universe version 2, 2026-11-23): the ten largest coins against EUR come from Kraken's public real-time
data (pseudo venue KRAKEN, symbols 'BTC-EUR' ...; pair metadata in ``trading_local.data.kraken``). They replace
Yahoo's BTC-USD, ETH-USD and SOL-USD, which ``migrate_universe`` deactivates in existing databases (their bars
and signals stay). The altcoins use Bitcoin as their market context (benchmark 'BTC-EUR').

Rules
* Equities and crypto are tradable. The major indices ^GSPC, ^NDX, ^DJI, ^GDAXI, ^STOXX50E, ^FTSE, ^N225 and
  ^HSI are tradable as well (the simulation trades them like an index CFD); every other index, ^VIX, the FX
  pairs and the futures are context only (``tradable=False``).
* ``delay_min`` is the LARGER of the delay Yahoo declares in its quote (``exchangeDataDelayedBy``) and the
  delay its help page states for the venue: understating a delay would let the validation replay use data
  earlier than it really arrives (look-ahead), overstating it only makes the replay more pessimistic.
* London shares are quoted in pence ('GBp'): currency 'GBP' with ``price_scale`` 0.01.
* Every currency of the universe can be converted into EUR through one of the FX pairs.
* ``sort_order`` groups the instruments by region (ASIA, EU, US, GLOBAL - the way the trading day moves
  round the globe) and orders them by importance inside the region.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType

from trading_local.clock import Clock, SystemClock
from trading_local.db import repo_market, repo_ops
from trading_local.db.database import Database
from trading_local.models import AssetClass, Instrument, Region

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------
# Venues
# ------------------------------------------------------------------------------------------------
# Calendar keys every component agrees on (MICs plus the pseudo venues of 24h markets).
VENUE_KEYS: tuple[str, ...] = (
    "XNYS", "XNAS", "XETR", "XFRA", "XLON", "XPAR", "XAMS", "XSWX", "XCSE",
    "XTKS", "XHKG", "XSHG", "XSHE", "XKRX", "XTAI", "XNSE", "XASX",
    "CRYPTO", "KRAKEN", "FX", "CME",
)  # fmt: skip
KRAKEN_BENCHMARK = "BTC-EUR"  # market context of the Kraken altcoins

# Yahoo exchange code (quote field ``exchange``) -> calendar key. Codes verified in the universe validation
# run: NMS NYQ GER AMS PAR EBS CPH LSE JPX HKG KSC TAI SHH SHZ NSI ASX CCC CCY CME CMX NYM and the index
# codes SNP NIM DJI WCB CXI ZRH FGI OSA. The remaining codes are Yahoo's documented codes of the same venues.
YAHOO_EXCHANGE_TO_VENUE: Mapping[str, str] = MappingProxyType(
    {
        # USA
        "NMS": "XNAS", "NGM": "XNAS", "NCM": "XNAS", "NIM": "XNAS",
        "NYQ": "XNYS", "PCX": "XNYS", "ASE": "XNYS", "BTS": "XNYS",
        "SNP": "XNYS", "DJI": "XNYS", "WCB": "XNYS", "CXI": "XNYS",
        # Europe
        "GER": "XETR", "FRA": "XFRA",
        "LSE": "XLON", "IOB": "XLON", "FGI": "XLON",
        "PAR": "XPAR", "AMS": "XAMS",
        "EBS": "XSWX", "VTX": "XSWX", "ZRH": "XSWX",
        "CPH": "XCSE",
        # Asia / Pacific
        "JPX": "XTKS", "OSA": "XTKS",
        "HKG": "XHKG",
        "SHH": "XSHG", "SHZ": "XSHE",
        "KSC": "XKRX", "KOE": "XKRX",
        "TAI": "XTAI", "TWO": "XTAI",
        "NSI": "XNSE", "BSE": "XNSE",
        "ASX": "XASX",
        # 24h markets
        "CCC": "CRYPTO", "CCY": "FX",
        "CME": "CME", "CMX": "CME", "NYM": "CME", "CBT": "CME",
    }
)  # fmt: skip

VENUE_REGION: Mapping[str, Region] = MappingProxyType(
    {
        "XNYS": "US", "XNAS": "US", "CME": "US",
        "XETR": "EU", "XFRA": "EU", "XLON": "EU", "XPAR": "EU", "XAMS": "EU", "XSWX": "EU", "XCSE": "EU",
        "XTKS": "ASIA", "XHKG": "ASIA", "XSHG": "ASIA", "XSHE": "ASIA", "XKRX": "ASIA", "XTAI": "ASIA",
        "XNSE": "ASIA", "XASX": "ASIA",
        "CRYPTO": "GLOBAL", "KRAKEN": "GLOBAL", "FX": "GLOBAL",
    }
)  # fmt: skip

# IANA time zone of every venue (24h pseudo venues date their sessions in UTC).
VENUE_TIMEZONE: Mapping[str, str] = MappingProxyType(
    {
        "XNYS": "America/New_York", "XNAS": "America/New_York", "CME": "America/Chicago",
        "XETR": "Europe/Berlin", "XFRA": "Europe/Berlin", "XLON": "Europe/London", "XPAR": "Europe/Paris",
        "XAMS": "Europe/Amsterdam", "XSWX": "Europe/Zurich", "XCSE": "Europe/Copenhagen",
        "XTKS": "Asia/Tokyo", "XHKG": "Asia/Hong_Kong", "XSHG": "Asia/Shanghai", "XSHE": "Asia/Shanghai",
        "XKRX": "Asia/Seoul", "XTAI": "Asia/Taipei", "XNSE": "Asia/Kolkata", "XASX": "Australia/Sydney",
        "CRYPTO": "UTC", "KRAKEN": "UTC", "FX": "UTC",
    }
)  # fmt: skip

# Market context of an EQUITY by venue (indices, crypto, FX and futures have no benchmark).
VENUE_BENCHMARK: Mapping[str, str] = MappingProxyType(
    {
        "XNYS": "^GSPC", "XNAS": "^GSPC",
        "XETR": "^GDAXI", "XFRA": "^GDAXI",
        "XLON": "^STOXX50E", "XPAR": "^STOXX50E", "XAMS": "^STOXX50E", "XSWX": "^STOXX50E", "XCSE": "^STOXX50E",
        "XTKS": "^N225", "XHKG": "^HSI", "XSHG": "^HSI", "XSHE": "^HSI",
        "XKRX": "^KS11", "XTAI": "^TWII", "XNSE": "^NSEI", "XASX": "^AXJO",
    }
)  # fmt: skip

# Indices the simulation may trade (like an index CFD); all other indices are context only.
TRADABLE_INDICES: frozenset[str] = frozenset({"^GSPC", "^NDX", "^DJI", "^GDAXI", "^STOXX50E", "^FTSE", "^N225", "^HSI"})

# sort_order = region base + position inside the region; instruments added later get base + USER_SORT_OFFSET.
REGION_SORT_BASE: Mapping[Region, int] = MappingProxyType({"ASIA": 1000, "EU": 2000, "US": 3000, "GLOBAL": 4000})
USER_SORT_OFFSET = 900

# Quote currencies that are a sub-unit of an ISO currency: code -> (ISO code, price_scale).
_SUBUNIT_CURRENCIES: Mapping[str, tuple[str, float]] = MappingProxyType(
    {"GBp": ("GBP", 0.01), "GBX": ("GBP", 0.01), "ZAc": ("ZAR", 0.01), "ILA": ("ILS", 0.01)}
)


def normalize_currency(code: str) -> tuple[str, float]:
    """Yahoo currency code -> (ISO 4217 code, price_scale).

    Sub-unit quotes are converted: 'GBp' (pence) -> ('GBP', 0.01). Codes are case-sensitive at Yahoo
    ('GBp' is pence, 'GBP' pounds); every other code is returned upper-case with scale 1.0.
    """
    if code in _SUBUNIT_CURRENCIES:
        return _SUBUNIT_CURRENCIES[code]
    return code.upper(), 1.0


def is_tradable(symbol: str, asset_class: AssetClass) -> bool:
    """Equities and crypto are tradable, indices only if they are one of ``TRADABLE_INDICES``."""
    if asset_class in ("equity", "crypto"):
        return True
    return asset_class == "index" and symbol in TRADABLE_INDICES


def default_benchmark(venue: str, asset_class: AssetClass, symbol: str | None = None) -> str | None:
    """Benchmark of an instrument: the venue's index for equities, Bitcoin ('BTC-EUR') for the other Kraken
    coins, None for everything else."""
    if asset_class == "crypto" and venue == "KRAKEN":
        return None if symbol == KRAKEN_BENCHMARK else KRAKEN_BENCHMARK
    return VENUE_BENCHMARK.get(venue) if asset_class == "equity" else None


# ------------------------------------------------------------------------------------------------
# Default universe
# ------------------------------------------------------------------------------------------------
type _Row = tuple[str, str, AssetClass, str, str, int, tuple[str, ...]]  # symbol, name, class, venue, ccy, delay, aliases


def _region_rows(region: Region, rows: Sequence[_Row]) -> list[Instrument]:
    """Instruments of one region group in importance order (``sort_order`` follows the row order)."""
    instruments = []
    for position, (symbol, name, asset_class, venue, currency_code, delay, aliases) in enumerate(rows, start=1):
        currency, price_scale = normalize_currency(currency_code)
        instruments.append(
            Instrument(
                symbol=symbol,
                name=name,
                asset_class=asset_class,
                region=region,
                exchange=venue,
                currency=currency,
                price_scale=price_scale,
                timezone=VENUE_TIMEZONE[venue],
                delay_min=delay,
                tradable=is_tradable(symbol, asset_class),
                active=True,
                benchmark=default_benchmark(venue, asset_class, symbol),
                aliases=aliases,
                sort_order=REGION_SORT_BASE[region] + 10 * position,
            )
        )
    return instruments


_ASIA: list[_Row] = [
    ("^N225", "Nikkei 225", "index", "XTKS", "JPY", 30, (
        "Nikkei 225", "Nikkei225", "Nikkei index", "Nikkei-Index", "Nikkei average", "Nikkei Stock Average",
        "Nikkei share average", "Japan 225", "Tokyo stocks", "Tokyo shares", "Japanese stocks", "Japanese shares",
        "Japan stocks")),
    ("^HSI", "Hang Seng Index", "index", "XHKG", "HKD", 15, (
        "Hang Seng", "Hang-Seng-Index", "Hong Kong stocks", "Hong Kong shares")),
    ("7203.T", "Toyota Motor Corp.", "equity", "XTKS", "JPY", 20, ("Toyota", "Toyota Motor")),
    ("6758.T", "Sony Group Corp.", "equity", "XTKS", "JPY", 20, ("Sony",)),
    ("9984.T", "SoftBank Group Corp.", "equity", "XTKS", "JPY", 20, ("SoftBank", "SoftBank Group")),
    ("8306.T", "Mitsubishi UFJ Financial Group", "equity", "XTKS", "JPY", 20, ("Mitsubishi UFJ", "MUFG")),
    ("6861.T", "Keyence Corp.", "equity", "XTKS", "JPY", 20, ("Keyence",)),
    ("8035.T", "Tokyo Electron Ltd.", "equity", "XTKS", "JPY", 20, ("Tokyo Electron",)),
    ("0700.HK", "Tencent Holdings Ltd.", "equity", "XHKG", "HKD", 15, ("Tencent",)),
    ("9988.HK", "Alibaba Group Holding Ltd.", "equity", "XHKG", "HKD", 15, ("Alibaba",)),
    ("3690.HK", "Meituan", "equity", "XHKG", "HKD", 15, ("Meituan",)),
    ("1299.HK", "AIA Group Ltd.", "equity", "XHKG", "HKD", 15, ("AIA Group",)),
    ("1810.HK", "Xiaomi Corp.", "equity", "XHKG", "HKD", 15, ("Xiaomi",)),
    ("005930.KS", "Samsung Electronics Co. Ltd.", "equity", "XKRX", "KRW", 20, (
        "Samsung", "Samsung Electronics", "Samsung Elec")),
    ("000660.KS", "SK hynix Inc.", "equity", "XKRX", "KRW", 20, ("SK Hynix", "Hynix")),
    ("2330.TW", "Taiwan Semiconductor Manufacturing Co.", "equity", "XTAI", "TWD", 20, (
        "TSMC", "Taiwan Semiconductor")),
    ("600519.SS", "Kweichow Moutai Co. Ltd.", "equity", "XSHG", "CNY", 30, ("Kweichow Moutai", "Moutai", "Maotai")),
    ("300750.SZ", "Contemporary Amperex Technology (CATL)", "equity", "XSHE", "CNY", 30, (
        "CATL", "Contemporary Amperex")),
    ("RELIANCE.NS", "Reliance Industries Ltd.", "equity", "XNSE", "INR", 15, (
        "RIL", "Reliance Industries", "Reliance Jio", "Reliance Retail", "Jio Platforms", "Reliance Brands")),
    ("BHP.AX", "BHP Group Ltd.", "equity", "XASX", "AUD", 20, ("BHP", "BHP Group", "BHP Billiton")),
    ("000001.SS", "SSE Composite Index", "index", "XSHG", "CNY", 30, (
        "Shanghai Composite", "SSE Composite", "Shanghai stocks", "Shanghai index", "China stocks", "Chinese stocks",
        "Chinese shares", "China shares", "CSI 300", "chinesische Aktien", "Chinas Börsen")),
    ("^KS11", "KOSPI", "index", "XKRX", "KRW", 20, (
        "KOSPI", "Korea Composite", "South Korean stocks", "Seoul stocks", "Seoul shares")),
    ("^TWII", "TAIEX", "index", "XTAI", "TWD", 20, (
        "Taiex", "Taiwan Weighted", "Taiwan stocks", "Taiwan shares", "Taipei stocks")),
    ("^NSEI", "Nifty 50", "index", "XNSE", "INR", 15, ("Nifty", "Nifty 50", "Nifty50", "NSE Nifty", "Sensex")),
    ("^AXJO", "S&P/ASX 200", "index", "XASX", "AUD", 20, (
        "ASX 200", "ASX200", "S&P/ASX 200", "Australian shares", "Australian stocks", "Aussie shares")),
]  # fmt: skip

_EU: list[_Row] = [
    ("^GDAXI", "DAX 40", "index", "XETR", "EUR", 15, (
        "DAX", "Dax", "Deutscher Aktienindex", "DAX 40", "Leitindex Dax", "German DAX", "Germany 40")),
    # Calculated during the Xetra/Eurex session; Yahoo lists it under 'ZRH', the Xetra calendar fits its days.
    ("^STOXX50E", "EURO STOXX 50", "index", "XETR", "EUR", 0, (
        "Euro Stoxx 50", "EuroStoxx 50", "EuroStoxx50", "Euro Stoxx", "EuroStoxx", "Eurozone 50", "ESTX50",
        "European shares", "European stocks", "STOXX 600", "Stoxx Europe 600", "Europas Börsen",
        "europäische Aktien", "European equities")),
    ("^FTSE", "FTSE 100", "index", "XLON", "GBP", 15, ("FTSE 100", "FTSE100", "Footsie")),
    ("SAP.DE", "SAP SE", "equity", "XETR", "EUR", 15, ("SAP", "SAP SE", "SAP-Aktie")),
    ("SIE.DE", "Siemens AG", "equity", "XETR", "EUR", 15, ("Siemens", "Siemens AG", "Siemens-Aktie")),
    ("ALV.DE", "Allianz SE", "equity", "XETR", "EUR", 15, (
        "Allianz", "Allianz SE", "Allianz-Aktie", "Allianz Global Investors", "Allianz Trade", "Allianz-Chef",
        "Allianz-Konzern", "Allianz-Tochter", "Versicherer Allianz", "Allianz Versicherung")),
    ("DTE.DE", "Deutsche Telekom AG", "equity", "XETR", "EUR", 15, (
        "Telekom", "Deutsche Telekom", "Dt. Telekom", "Telekom-Aktie", "T-Aktie", "Telekom-Chef")),
    ("AIR.DE", "Airbus SE", "equity", "XETR", "EUR", 15, ("Airbus",)),
    ("MBG.DE", "Mercedes-Benz Group AG", "equity", "XETR", "EUR", 15, (
        "Daimler", "Mercedes-Benz", "Mercedes", "Mercedes-Aktie")),
    ("BMW.DE", "BMW AG", "equity", "XETR", "EUR", 15, ("BMW", "Bayerische Motoren Werke", "BMW-Aktie")),
    ("VOW3.DE", "Volkswagen AG (Vz.)", "equity", "XETR", "EUR", 15, (
        "VW", "Volkswagen", "VW-Konzern", "VW-Aktie", "VW-Chef")),
    ("BAS.DE", "BASF SE", "equity", "XETR", "EUR", 15, ("BASF",)),
    ("IFX.DE", "Infineon Technologies AG", "equity", "XETR", "EUR", 15, ("Infineon",)),
    ("RHM.DE", "Rheinmetall AG", "equity", "XETR", "EUR", 15, ("Rheinmetall",)),
    ("MUV2.DE", "Münchener Rück AG", "equity", "XETR", "EUR", 15, (
        "Munich Re", "Münchener Rück", "Münchner Rück", "Münchener Rückversicherung", "Munich Reinsurance",
        "MunichRe")),
    ("DBK.DE", "Deutsche Bank AG", "equity", "XETR", "EUR", 15, ("Deutsche Bank", "Deutsche-Bank-Aktie", "Dt. Bank")),
    ("ADS.DE", "adidas AG", "equity", "XETR", "EUR", 15, ("Adidas",)),
    ("BAYN.DE", "Bayer AG", "equity", "XETR", "EUR", 15, ("Bayer", "Bayer AG", "Bayer-Aktie", "Bayer-Konzern", "Bayer-Chef")),
    ("ENR.DE", "Siemens Energy AG", "equity", "XETR", "EUR", 15, ("Siemens Energy", "Siemens-Energy")),
    ("ASML.AS", "ASML Holding NV", "equity", "XAMS", "EUR", 15, ("ASML", "ASML Holding", "ASML-Aktie")),
    ("MC.PA", "LVMH Moët Hennessy Louis Vuitton SE", "equity", "XPAR", "EUR", 15, (
        "LVMH", "Louis Vuitton", "Moët Hennessy")),
    ("OR.PA", "L'Oréal SA", "equity", "XPAR", "EUR", 15, ("L'Oréal", "LOreal")),
    ("TTE.PA", "TotalEnergies SE", "equity", "XPAR", "EUR", 15, ("TotalEnergies", "Total Energies")),
    ("SAN.PA", "Sanofi SA", "equity", "XPAR", "EUR", 15, ("Sanofi",)),
    ("NESN.SW", "Nestlé SA", "equity", "XSWX", "CHF", 30, ("Nestlé",)),
    ("NOVN.SW", "Novartis AG", "equity", "XSWX", "CHF", 30, ("Novartis",)),
    ("ROP.SW", "Roche Holding AG (Genussschein)", "equity", "XSWX", "CHF", 30, (
        "Roche", "Roche Holding", "Hoffmann-La Roche", "Roche-Aktie")),
    ("NOVO-B.CO", "Novo Nordisk A/S (Class B)", "equity", "XCSE", "DKK", 0, ("Novo Nordisk", "Wegovy", "Ozempic")),
    ("SHEL.L", "Shell plc", "equity", "XLON", "GBp", 20, (
        "Shell", "Shell plc", "Royal Dutch Shell", "Shell-Aktie", "Shell-Konzern", "Ölkonzern Shell",
        "oil major Shell")),
    ("AZN.L", "AstraZeneca PLC", "equity", "XLON", "GBp", 20, ("AstraZeneca", "Astra Zeneca")),
    ("HSBA.L", "HSBC Holdings plc", "equity", "XLON", "GBp", 20, ("HSBC", "HSBC Holdings")),
    ("^FCHI", "CAC 40", "index", "XPAR", "EUR", 15, ("CAC 40", "CAC40")),
    ("^SSMI", "Swiss Market Index", "index", "XSWX", "CHF", 30, (
        "SMI", "Swiss Market Index", "Schweizer Leitindex", "Leitindex SMI")),
]  # fmt: skip

_US: list[_Row] = [
    ("^GSPC", "S&P 500", "index", "XNYS", "USD", 0, (
        "SPX", "S&P 500", "S&P500", "Standard & Poor's 500", "S&P-500-Index")),
    ("^NDX", "Nasdaq 100", "index", "XNAS", "USD", 0, (
        "NDX", "Nasdaq 100", "Nasdaq100", "Nasdaq Composite", "Nasdaq index", "tech-heavy Nasdaq",
        "Technologiebörse Nasdaq", "Nasdaq-Index", "Technologieindex Nasdaq")),
    ("^DJI", "Dow Jones Industrial Average", "index", "XNYS", "USD", 0, (
        "Dow Jones", "Dow Jones Industrial Average", "DJIA", "Dow Industrials", "Dow-Jones-Index", "Dow futures")),
    ("AAPL", "Apple Inc.", "equity", "XNAS", "USD", 15, (
        "Apple", "Apple Inc", "iPhone maker", "iPhone-Hersteller", "iPhone-Konzern", "Apple-Aktie")),
    ("MSFT", "Microsoft Corp.", "equity", "XNAS", "USD", 15, ("Microsoft",)),
    ("NVDA", "NVIDIA Corp.", "equity", "XNAS", "USD", 15, ("Nvidia",)),
    ("AMZN", "Amazon.com Inc.", "equity", "XNAS", "USD", 15, (
        "Amazon", "AWS", "Amazon.com", "Amazon Web Services", "Amazon-Aktie")),
    ("GOOGL", "Alphabet Inc. (Class A)", "equity", "XNAS", "USD", 15, (
        "Alphabet", "Alphabet Inc", "Google", "Google-Mutter", "Google parent", "Alphabet-Aktie", "Waymo", "DeepMind")),
    ("META", "Meta Platforms Inc.", "equity", "XNAS", "USD", 15, (
        "Meta", "Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Meta-Aktie", "Facebook-Mutter",
        "Facebook parent", "Facebook-Konzern")),
    ("TSLA", "Tesla Inc.", "equity", "XNAS", "USD", 15, ("Tesla",)),
    ("AVGO", "Broadcom Inc.", "equity", "XNAS", "USD", 15, ("Broadcom",)),
    ("BRK-B", "Berkshire Hathaway Inc. (Class B)", "equity", "XNYS", "USD", 20, (
        "Berkshire", "Berkshire Hathaway", "Buffetts Berkshire", "Buffett's Berkshire")),
    ("JPM", "JPMorgan Chase & Co.", "equity", "XNYS", "USD", 20, ("JPMorgan", "JP Morgan", "J.P. Morgan", "JPMorgan Chase")),
    ("LLY", "Eli Lilly and Co.", "equity", "XNYS", "USD", 20, ("Eli Lilly", "Lilly and Company", "Zepbound", "Mounjaro")),
    ("V", "Visa Inc.", "equity", "XNYS", "USD", 20, ("Visa Inc",)),
    ("MA", "Mastercard Inc.", "equity", "XNYS", "USD", 20, ("Mastercard",)),
    ("XOM", "ExxonMobil Holdings Corp.", "equity", "XNYS", "USD", 20, ("Exxon Mobil", "ExxonMobil", "Exxon")),
    ("WMT", "Walmart Inc.", "equity", "XNAS", "USD", 15, ("Walmart", "Wal-Mart")),
    ("UNH", "UnitedHealth Group Inc.", "equity", "XNYS", "USD", 20, (
        "UnitedHealth", "UnitedHealth Group", "UnitedHealthcare", "United Health Group")),
    ("COST", "Costco Wholesale Corp.", "equity", "XNAS", "USD", 15, ("Costco",)),
    ("NFLX", "Netflix Inc.", "equity", "XNAS", "USD", 15, ("Netflix",)),
    ("AMD", "Advanced Micro Devices Inc.", "equity", "XNAS", "USD", 15, ("AMD", "Advanced Micro Devices")),
    ("ORCL", "Oracle Corp.", "equity", "XNYS", "USD", 20, ("Oracle", "Oracle Corp", "Oracle-Aktie")),
    ("JNJ", "Johnson & Johnson", "equity", "XNYS", "USD", 20, ("Johnson & Johnson", "J&J")),
    ("PG", "Procter & Gamble Co.", "equity", "XNYS", "USD", 20, ("Procter & Gamble", "P&G")),
    ("HD", "Home Depot Inc.", "equity", "XNYS", "USD", 20, ("Home Depot",)),
    ("BAC", "Bank of America Corp.", "equity", "XNYS", "USD", 20, ("Bank of America", "BofA")),
    ("PLTR", "Palantir Technologies Inc.", "equity", "XNAS", "USD", 15, ("Palantir",)),
    ("^RUT", "Russell 2000", "index", "XNYS", "USD", 15, ("Russell 2000",)),
    ("^VIX", "CBOE Volatility Index", "index", "XNYS", "USD", 15, (
        "VIX", "CBOE Volatility Index", "Cboe Volatility", "fear gauge", "Angstbarometer")),
    ("ES=F", "E-mini S&P 500 futures", "future", "CME", "USD", 10, (
        "S&P 500 futures", "S&P 500 future", "S&P futures", "E-mini S&P", "S&P 500 E-mini", "S&P-500-Future",
        "US stock futures", "U.S. stock futures", "Wall Street futures", "US equity futures", "U.S. equity futures",
        "US-Futures", "US-Aktienfutures")),
    ("NQ=F", "E-mini Nasdaq 100 futures", "future", "CME", "USD", 10, (
        "Nasdaq futures", "Nasdaq 100 futures", "Nasdaq-Future", "Nasdaq-100-Future", "E-mini Nasdaq")),
]  # fmt: skip

_GLOBAL: list[_Row] = [
    # Kraken spot pairs in EUR, real time (pseudo venue KRAKEN). Short tickers that are also ordinary words or
    # abbreviations (LINK, DOT, ADA, SOL, DOGE, LTC ...) only count in crypto context: see news.matcher.
    ("BTC-EUR", "Bitcoin", "crypto", "KRAKEN", "EUR", 0, (
        "Bitcoin", "BTC", "XBT", "Bitcoin-Kurs", "Bitcoinkurs", "Bitcoin-Preis")),
    ("ETH-EUR", "Ethereum", "crypto", "KRAKEN", "EUR", 0, ("Ethereum", "Ether", "ETH")),
    ("SOL-EUR", "Solana", "crypto", "KRAKEN", "EUR", 0, ("Solana", "SOL")),
    ("XRP-EUR", "XRP", "crypto", "KRAKEN", "EUR", 0, ("XRP", "Ripple")),
    ("ADA-EUR", "Cardano", "crypto", "KRAKEN", "EUR", 0, ("Cardano", "ADA")),
    ("DOGE-EUR", "Dogecoin", "crypto", "KRAKEN", "EUR", 0, ("Dogecoin", "DOGE")),
    ("AVAX-EUR", "Avalanche", "crypto", "KRAKEN", "EUR", 0, ("Avalanche", "AVAX")),
    ("LINK-EUR", "Chainlink", "crypto", "KRAKEN", "EUR", 0, ("Chainlink", "LINK")),
    ("DOT-EUR", "Polkadot", "crypto", "KRAKEN", "EUR", 0, ("Polkadot", "DOT")),
    ("LTC-EUR", "Litecoin", "crypto", "KRAKEN", "EUR", 0, ("Litecoin", "LTC")),
    # COMEX/NYMEX: Yahoo declares 10 minutes, its help page states 30 (the larger value applies, see above).
    ("GC=F", "Gold futures", "future", "CME", "USD", 30, (
        "gold price", "gold prices", "gold futures", "spot gold", "Comex gold", "XAU/USD", "XAUUSD", "gold bullion",
        "Feinunze Gold", "Goldpreis", "Goldkurs")),
    ("CL=F", "WTI crude oil futures", "future", "CME", "USD", 30, (
        "WTI", "West Texas Intermediate", "crude oil", "oil price", "oil prices", "crude futures", "crude prices",
        "US crude", "U.S. crude", "oil futures", "Ölpreis", "Rohöl")),
    ("BZ=F", "Brent crude oil futures", "future", "CME", "USD", 30, (
        "Brent crude", "Brent oil", "Brent futures", "Nordseeöl", "Nordseesorte Brent", "Brent-Öl", "Brent-Rohöl",
        "Sorte Brent")),
    # FX: price = units of the quote currency per ONE unit of the base currency (EURUSD=X: USD per EUR).
    ("EURUSD=X", "EUR/USD", "fx", "FX", "USD", 0, (
        "EUR/USD", "EURUSD", "Euro-Dollar", "Euro/Dollar", "Euro-Kurs", "euro against the dollar", "euro vs dollar",
        "Euro zum Dollar", "Euro gegenüber dem Dollar", "euro versus the dollar", "Eurokurs")),
    ("EURGBP=X", "EUR/GBP", "fx", "FX", "GBP", 0, (
        "EUR/GBP", "EURGBP", "Euro/Pfund", "Euro-Pfund", "euro-sterling", "pound against the euro",
        "sterling against the euro", "britische Pfund", "Britische Pfund", "British pound", "GBP/USD",
        "pound sterling")),
    ("EURCHF=X", "EUR/CHF", "fx", "FX", "CHF", 0, (
        "EUR/CHF", "EURCHF", "Euro/Franken", "Euro-Franken", "Swiss franc", "Schweizer Franken", "Frankenkurs",
        "USD/CHF")),
    ("EURJPY=X", "EUR/JPY", "fx", "FX", "JPY", 0, ("EUR/JPY", "EURJPY", "Euro/Yen", "Euro-Yen", "yen against the euro")),
    ("USDJPY=X", "USD/JPY", "fx", "FX", "JPY", 0, (
        "USD/JPY", "USDJPY", "Dollar/Yen", "Dollar-Yen", "yen against the dollar", "yen per dollar", "Yen zum Dollar",
        "Japanese yen", "japanische Yen", "japanischen Yen")),
    ("EURHKD=X", "EUR/HKD", "fx", "FX", "HKD", 0, (
        "EUR/HKD", "EURHKD", "Hong Kong dollar", "Hongkong-Dollar", "USD/HKD", "HKD peg")),
    ("EURCNY=X", "EUR/CNY", "fx", "FX", "CNY", 0, (
        "EUR/CNY", "EURCNY", "USD/CNY", "USD/CNH", "onshore yuan", "offshore yuan", "yuan fixing", "yuan midpoint",
        "Renminbi")),
    ("EURKRW=X", "EUR/KRW", "fx", "FX", "KRW", 0, (
        "EUR/KRW", "EURKRW", "Korean won", "South Korean won", "südkoreanische Won", "südkoreanischen Won", "USD/KRW")),
    ("EURTWD=X", "EUR/TWD", "fx", "FX", "TWD", 0, (
        "EUR/TWD", "EURTWD", "Taiwan dollar", "New Taiwan dollar", "Taiwan-Dollar", "USD/TWD")),
    ("EURINR=X", "EUR/INR", "fx", "FX", "INR", 0, ("EUR/INR", "EURINR", "USD/INR", "Indian rupee", "indische Rupie")),
    ("EURAUD=X", "EUR/AUD", "fx", "FX", "AUD", 0, (
        "EUR/AUD", "EURAUD", "Australian dollar", "Aussie dollar", "australische Dollar", "australischen Dollar",
        "AUD/USD")),
    ("EURDKK=X", "EUR/DKK", "fx", "FX", "DKK", 0, (
        "EUR/DKK", "EURDKK", "Danish krone", "Danish crown", "dänische Krone", "dänischen Krone")),
]  # fmt: skip

# The US index futures belong to the US group; commodity futures, crypto and FX are global.
DEFAULT_INSTRUMENTS: list[Instrument] = [
    *_region_rows("ASIA", _ASIA),
    *_region_rows("EU", _EU),
    *_region_rows("US", _US),
    *_region_rows("GLOBAL", _GLOBAL),
]


def seed_default_universe(db: Database) -> int:
    """Upsert ``DEFAULT_INSTRUMENTS`` into the instruments table, keeping the user's active/tradable flags
    on rows that already exist. Returns the number of rows written."""
    return repo_market.upsert_instruments(db, DEFAULT_INSTRUMENTS, keep_user_flags=True)


# ------------------------------------------------------------------------------------------------
# Migration of existing databases
# ------------------------------------------------------------------------------------------------
UNIVERSE_VERSION = 2  # raise whenever DEFAULT_INSTRUMENTS changes in a way existing databases must receive
UNIVERSE_VERSION_KEY = "universe_version"  # settings table
# Instruments retired by a version: deactivated once (active = tradable = 0), never deleted - their bars,
# signals and portfolio history stay readable.
RETIRED_SYMBOLS: Mapping[int, tuple[str, ...]] = MappingProxyType(
    {2: ("BTC-USD", "ETH-USD", "SOL-USD")}  # replaced by the Kraken EUR pairs
)
_LIST_LIMIT = 12  # symbols named per line of the migration report


def _symbols_text(symbols: Sequence[str]) -> str:
    shown = ", ".join(symbols[:_LIST_LIMIT])
    rest = len(symbols) - _LIST_LIMIT
    return f"{shown} und {rest} weitere" if rest > 0 else shown


def _descriptive(instrument: Instrument) -> Instrument:
    """The instrument without the user's flags (what an upsert with ``keep_user_flags`` may change)."""
    return replace(instrument, active=True, tradable=True)


def migrate_universe(db: Database, *, clock: Clock | None = None) -> list[str]:
    """Bring the instruments table to ``UNIVERSE_VERSION`` (idempotent; the version lives in the settings table).

    In one transaction: every default instrument is upserted - missing ones are inserted, the descriptive columns
    of existing rows (name, venue, currency, delay, time zone, benchmark, aliases, sort order ...) are refreshed,
    the user's ``active`` / ``tradable`` flags of existing rows are kept - and the instruments retired by the
    versions passed over are deactivated once. Instruments the user added are left alone. A fresh database ends
    up with exactly ``DEFAULT_INSTRUMENTS``.

    Returns German sentences describing the changes (empty when the table was already up to date).
    """
    now_ts = (clock or SystemClock()).ts()
    stored = repo_ops.get_setting(db, UNIVERSE_VERSION_KEY, 0)
    version = stored if isinstance(stored, int) and not isinstance(stored, bool) else 0
    if version >= UNIVERSE_VERSION:
        return []
    changes: list[str] = []
    with db.write():
        existing = {i.symbol: i for i in repo_market.list_instruments(db, active_only=False)}
        added = [i for i in DEFAULT_INSTRUMENTS if i.symbol not in existing]
        refreshed = [
            i.symbol
            for i in DEFAULT_INSTRUMENTS
            if i.symbol in existing and _descriptive(existing[i.symbol]) != _descriptive(i)
        ]
        repo_market.upsert_instruments(db, DEFAULT_INSTRUMENTS, keep_user_flags=True)
        retired = [
            symbol
            for passed in range(version + 1, UNIVERSE_VERSION + 1)
            for symbol in RETIRED_SYMBOLS.get(passed, ())
            if symbol in existing and (existing[symbol].active or existing[symbol].tradable)
        ]
        for symbol in retired:
            repo_market.set_instrument_flags(db, symbol, active=False, tradable=False)
        open_retired = _with_open_business(db, retired)
        repo_ops.set_setting(db, UNIVERSE_VERSION_KEY, UNIVERSE_VERSION, now_ts)
    if not existing:
        changes.append(f"Standard-Instrumentenliste angelegt ({len(added)} Instrumente).")
    else:
        changes.append(f"Instrumentenliste auf Version {UNIVERSE_VERSION} aktualisiert.")
        if added:
            changes.append(f"Neu aufgenommen: {_symbols_text([f'{i.symbol} ({i.name})' for i in added])}.")
        if refreshed:
            changes.append(f"Stammdaten aktualisiert: {_symbols_text(refreshed)}.")
        if retired:
            changes.append(
                f"Deaktiviert (ersetzt, Kursdaten und Signale bleiben erhalten): {_symbols_text(retired)}."
            )
        if open_retired:
            changes.append(
                f"Achtung: offene Positionen, Orders oder laufende Signale in {_symbols_text(open_retired)} – diese "
                "Werte erhalten keine neuen Kurse mehr; Signale laufen mit dem letzten bekannten Kurs aus. Positionen "
                "bitte manuell schließen oder den Wert wieder aktivieren."
            )
    log.info("universe migrated to version %d: %s", UNIVERSE_VERSION, " ".join(changes))
    return changes


def _with_open_business(db: Database, symbols: Sequence[str]) -> list[str]:
    """The given symbols that still have an open position or a pending order in a portfolio, or a signal that
    waits for its entry or exit (its tracking needs new bars)."""
    if not symbols:
        return []
    marks = ", ".join("?" * len(symbols))
    rows = db.query(
        f"SELECT symbol FROM positions WHERE status = 'open' AND symbol IN ({marks}) "
        f"UNION SELECT symbol FROM orders WHERE status = 'pending' AND symbol IN ({marks}) "
        f"UNION SELECT symbol FROM signals WHERE status IN ('pending_entry', 'active') AND symbol IN ({marks})",
        (*symbols, *symbols, *symbols),
    )
    found = {row[0] for row in rows}
    return [symbol for symbol in symbols if symbol in found]


__all__ = [
    "DEFAULT_INSTRUMENTS",
    "KRAKEN_BENCHMARK",
    "REGION_SORT_BASE",
    "RETIRED_SYMBOLS",
    "TRADABLE_INDICES",
    "UNIVERSE_VERSION",
    "UNIVERSE_VERSION_KEY",
    "USER_SORT_OFFSET",
    "VENUE_BENCHMARK",
    "VENUE_KEYS",
    "VENUE_REGION",
    "VENUE_TIMEZONE",
    "YAHOO_EXCHANGE_TO_VENUE",
    "default_benchmark",
    "is_tradable",
    "migrate_universe",
    "normalize_currency",
    "seed_default_universe",
]
