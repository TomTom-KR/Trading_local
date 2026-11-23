"""Deterministic matching of news to instruments (no LLM) and near-duplicate detection across sources.

``AliasMatcher`` maps a headline (and its summary) to the active instruments of the ``InstrumentBook``.
Every instrument contributes its symbol, its name and its ``aliases``; instruments of the default universe
additionally carry a curated rule set (``CURATED_RULES``, ported from the news probe: 99.1 % precision on 452
hand-labelled real headline/symbol pairs) with German and English name variants, ISINs, veto phrases and
context requirements for ambiguous names and tickers.

Precision rules (in this order of importance)
* Word boundaries everywhere; flexible space/hyphen, '&'/'and'/'und', optional dots and apostrophes,
  accent variants (Nestle/Nestlé, Muenchener/Münchener) and an optional genitive (Nvidias, Apple's).
* Single-word names that are also ordinary words (Apple, Shell, Meta, Oracle, Allianz ...) match
  case-sensitively; aliases of up to three characters (SAP, BMW, AMD) only in exactly the given spelling.
* Ambiguous tickers (V, MA, HD, PG, AIR, SAN, BAS, ADS, ALV, COST, META ...) match only in ticker context:
  '$V', 'NYSE: V', '(NASDAQ: COST)'.
* Context aliases ('Visa', 'Nikkei', 'gold', 'yen' ...) need a required word somewhere in title or summary
  and none of the forbidden phrases; currency words additionally reject amounts ('5 billion yuan').
* Veto phrases overlapping a match kill it ('Allianz Arena', 'Meta-Analyse', 'Oracle of Omaha').
* Between DIFFERENT symbols the longest overlapping match wins (Siemens Energy beats Siemens).
* Banks named as analyst or broker ('JPMorgan hebt SAP auf Overweight', 'Analysts at Bank of America ...')
  are context, not the subject of the story: such mentions get role 'broker' and are not returned by
  ``match``.
* Exact hints delivered by the feed (ISIN, ticker) beat every alias.

A second, separate mapping (``MACRO_TERMS``) turns macro and market headlines (central banks, economic data,
trade policy, index names) into the index symbols they move. Only symbols that are active in the book are
returned. All alias and term data is plain text: it is escaped before it becomes part of a pattern.

``NearDupIndex`` clusters headlines of the last 48 hours by weighted token similarity, grouped by the matched
symbols, so that the same story from a second source is recognised as a duplicate.
"""

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

from trading_local.models import Instrument

MAX_ALIAS_SYMBOLS: Final = 8  # instruments per item handed to the LLM as candidates (strongest first)

MentionKind = Literal["name", "name_cs", "name_ctx", "ticker", "ticker_ctx", "isin"]
MentionRole = Literal["subject", "broker"]


# ------------------------------------------------------------------------------------------------
# Rule types
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ContextAlias:
    """An alias that only counts with one of ``require`` and none of ``forbid`` in title + summary.

    ``require`` / ``forbid`` entries are phrases with prefix semantics ('devalu' matches 'devaluation').
    ``amount_guard`` rejects a match directly preceded by an amount ('5 billion yuan').
    """

    alias: str
    require: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    case_sensitive: bool = False
    amount_guard: bool = False


@dataclass(frozen=True, slots=True)
class SymbolRules:
    """Alias rules of one instrument.

    names         case-insensitive names
    names_cs      case-sensitive names (ordinary words: Apple, Shell, Meta)
    names_prefix  case-insensitive word prefixes for German compounds ('Goldpreis' hits 'Goldpreisverfall')
    names_ctx     context-dependent aliases
    tickers       tickers matched as bare upper-case tokens
    tickers_ctx   ambiguous tickers, matched only as '$X' or 'EXCHANGE: X'
    vetoes        phrases that cancel an overlapping match (prefix semantics, a leading '-' is literal)
    broker        a bank: mentions followed by rating verbs, 'Research', 'analysts' ... get role 'broker'
    """

    isin: str | None = None
    names: tuple[str, ...] = ()
    names_cs: tuple[str, ...] = ()
    names_prefix: tuple[str, ...] = ()
    names_ctx: tuple[ContextAlias, ...] = ()
    tickers: tuple[str, ...] = ()
    tickers_ctx: tuple[str, ...] = ()
    vetoes: tuple[str, ...] = ()
    broker: bool = False


@dataclass(frozen=True, slots=True)
class MacroTerm:
    """A macro or market topic and the symbols it moves (``targets``, most relevant first)."""

    topic: str
    targets: tuple[str, ...]
    names: tuple[str, ...] = ()
    names_cs: tuple[str, ...] = ()
    names_ctx: tuple[ContextAlias, ...] = ()
    vetoes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Mention:
    """One instrument mention found in a text (``start``/``end`` index the normalised title or summary)."""

    symbol: str
    alias: str
    kind: MentionKind
    where: Literal["title", "summary"]
    start: int
    end: int
    role: MentionRole = "subject"


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Everything the news service needs to know about one item.

    ``symbols``  subject instruments, strongest first, at most ``MAX_ALIAS_SYMBOLS``
    ``macro``    index symbols mapped from macro terms that are not in ``symbols``
    ``topics``   the macro topics found (e.g. 'FED', 'ECB')
    """

    symbols: tuple[str, ...]
    macro: tuple[str, ...]
    topics: tuple[str, ...]

    @property
    def hinted(self) -> list[str]:
        """``symbols`` followed by ``macro``: the value stored in ``NewsItem.hinted_symbols``."""
        return [*self.symbols, *self.macro]


# ------------------------------------------------------------------------------------------------
# Curated rules of the default universe (ported from the news probe's aliases.json, version 3)
# ------------------------------------------------------------------------------------------------
# Words that make 'Nasdaq', 'Dow', 'FTSE', 'Nikkei' an index (and not the exchange operator or the newspaper).
_MARKET_MOVES: Final[tuple[str, ...]] = (
    "falls", "fell", "rises", "rose", "gains", "gained", "drops", "dropped", "plunge", "slides", "slid", "slips",
    "sinks", "sank", "tumbles", "jumps", "rallies", "rally", "surges", "climbs", "closes", "closed", "ends", "ended",
    "opens", "points", "record", "index", "stocks", "shares", "equities", "Wall Street", "selloff", "sell-off",
    "correction", "bear market", "bull market", "futures", "rout", "rebound", "slump", "worst week", "best week",
    "fällt", "faellt", "steigt", "schließt", "schliesst", "Punkte", "verliert", "gewinnt", "stürzt", "sackt",
    "rutscht", "klettert", "Minus", "Plus", "Rekord", "Börse", "Boerse", "Index", "Aktien", "Kursrutsch", "Ausverkauf",
    "Crash", "Erholung", "tiefrot", "Handelsschluss", "Wochenverlust", "Tagesminus",
)
# Words that make 'yen', 'yuan', 'rupee', 'sterling' a currency move.
_FX_MOVES: Final[tuple[str, ...]] = (
    "dollar", "Dollar", "euro", "Euro", "currency", "currencies", "forex", "FX", "exchange rate", "weakens", "weaker",
    "strengthens", "stronger", "intervention", "devalu", "Währung", "Waehrung", "Devisen", "Wechselkurs",
    "central bank", "Notenbank", "Zentralbank", "record low", "Rekordtief", "carry trade", "appreciat", "depreciat",
    "Aufwertung", "Abwertung", "wertet", "USD", "EUR", "climbs", "slides", "falls", "rises", "gains", "drops", "jumps",
    "slumps", "surges", "steigt", "fällt", "sinkt",
)
_NIKKEI_MOVES: Final[tuple[str, ...]] = (*_MARKET_MOVES, "Tokyo", "Tokio", "yen", "Yen")
# 'Nasdaq' as exchange operator or listing venue, not as index.
_NASDAQ_VENUE: Final[tuple[str, ...]] = (
    "Nasdaq:", "Nasdaq-listed", "Nasdaq listed", "Nasdaq Inc", "Nasdaq debut", "Nasdaq listing", "Nasdaq IPO",
    "Nasdaq Stockholm", "Nasdaq Helsinki", "Nasdaq Copenhagen", "Nasdaq Baltic", "Nasdaq First North",
    "Nasdaq Nordic", "Nasdaq Dubai", "(Nasdaq", "NasdaqGS", "NasdaqGM", "NasdaqCM", "Nasdaq Trader", "Nasdaq Tallinn",
    "Nasdaq Riga", "Nasdaq Vilnius", "Nasdaq Iceland", "Nasdaq Capital Market", "Nasdaq Global", "Nasdaq-notiert",
    "Nasdaq GS", "on Nasdaq", "on the Nasdaq", "an der Nasdaq",
)
# 'Nikkei' as the newspaper, not as index.
_NIKKEI_PAPER: Final[tuple[str, ...]] = (
    "Nikkei Asia", "Nikkei report", "Nikkei says", "Nikkei newspaper", "Nikkei business daily", "according to Nikkei",
    "Nikkei reported", "Nikkei-Bericht", "laut Nikkei", "- Nikkei", "Nikkei Inc", ": Nikkei", "Nikkei reports",
    "Nikkei survey", "Nikkei poll",
)
_DAX_VETOES: Final[tuple[str, ...]] = ("Dax Shepard", "DAX SUKHRAJ", "Dax Sukhraj")
# Crypto context for short coin tickers that are also ordinary words or common abbreviations (LINK, DOT, ADA,
# SOL, AVAX, DOGE, LTC ...): distinctive words only, because require phrases match case-insensitively with
# prefix semantics ('token' also hits 'tokens', so 'coin' or 'DeFi' would hit 'coincide' / 'deficit').
_CRYPTO_STRICT: Final[tuple[str, ...]] = (
    "crypto", "Krypto", "token", "altcoin", "memecoin", "meme coin", "stablecoin", "blockchain", "Bitcoin",
    "Ethereum", "Solana", "Cardano", "Dogecoin", "Litecoin", "Polkadot", "Chainlink", "XRP", "Binance", "Coinbase",
    "Kraken", "Bitpanda", "Bitstamp", "Bybit", "staking", "on-chain", "onchain", "mainnet", "web3", "satoshi",
    "halving", "smart contract", "Digitalwährung", "Digitalwaehrung", "DeFi ", "DeFi-", "NFT ", "NFTs",
)  # fmt: skip
# BTC / ETH / Ether are hardly ever anything else: market words and prices count as context as well.
_CRYPTO_BROAD: Final[tuple[str, ...]] = (
    *_CRYPTO_STRICT, *_MARKET_MOVES, "$", "USD", "EUR", "price", "Kurs", "ETF", "derivative", "liquidat", "whale",
    "trader", "funding rate", "open interest", "retail", "options",
)  # fmt: skip
_COIN_PAIR_TICKERS: Final = ("{b}-EUR", "{b}EUR", "{b}/EUR", "{b}-USD", "{b}USD", "{b}/USD", "{b}-USDT", "{b}/USDT")


def _coin_tickers(*bases: str) -> tuple[str, ...]:
    """Pair spellings of a coin ('BTC-EUR', 'BTC/USD', 'XBTEUR' ...) for every base code."""
    return tuple(form.format(b=base) for base in bases for form in _COIN_PAIR_TICKERS)


def _coin_ticker(ticker: str, require: tuple[str, ...] = _CRYPTO_STRICT, forbid: tuple[str, ...] = ()) -> ContextAlias:
    """A coin ticker as an upper-case word that only counts with crypto context."""
    return ContextAlias(ticker, require=require, forbid=forbid, case_sensitive=True)


# Kraken EUR pairs (pseudo venue KRAKEN). Names that are proper nouns of the coin match plainly; tickers and
# names that are ordinary words (Ripple, Avalanche) need crypto context; well-known false friends are vetoed.
_KRAKEN_COINS: Final[dict[str, SymbolRules]] = {
    "BTC-EUR": SymbolRules(
        names=("Bitcoin",),
        names_prefix=("Bitcoin-Kurs", "Bitcoinkurs", "Bitcoin-Preis"),
        tickers=_coin_tickers("BTC", "XBT"),
        names_ctx=(_coin_ticker("BTC", _CRYPTO_BROAD), _coin_ticker("XBT")),
        # 'BTC-ECHO' is a news site: every item of its feed ends with 'Source: BTC-ECHO BTC-ECHO'.
        vetoes=("Bitcoin Cash", "Bitcoin SV", "Bitcoin Depot", "Bitcoin Magazine", "Bitcoin Group SE", "BTC-ECHO"),
    ),
    "ETH-EUR": SymbolRules(
        names=("Ethereum",),
        tickers=_coin_tickers("ETH"),
        names_ctx=(_coin_ticker("ETH", _CRYPTO_BROAD), _coin_ticker("Ether", _CRYPTO_BROAD)),
        # The university (ETH Zürich): with the broad context ('Rekord', 'Index', 'Plus' ...) its research news
        # would count as ether news.
        vetoes=(
            "ETH Zürich", "ETH Zurich", "ETH Lausanne", "Ethereum Classic", "Ether.fi", "ETH-Bibliothek", "ETH-Rat",
            "ETH-Bereich", "ETH Domain", "an der ETH", "ETH-Forsch", "ETH-Stud", "ETH-Profess", "ETH-Ökonom",
            "ETH-Wissenschaft", "ETH-Präsident", "ETH-Konjunktur", "ETH-Spin-off", "ETH researcher", "ETH scientist",
        ),
    ),
    "SOL-EUR": SymbolRules(
        names=("Solana",),
        tickers=_coin_tickers("SOL"),
        tickers_ctx=("SOL",),
        names_ctx=(_coin_ticker("SOL"),),
        vetoes=("Solana Beach",),
    ),
    "XRP-EUR": SymbolRules(
        names=("XRP Ledger", "XRPL", "Ripple Labs"),
        names_cs=("XRP",),
        tickers=_coin_tickers("XRP"),
        names_ctx=(_coin_ticker("Ripple", forbid=("ripple effect", "ripple through", "rippling")),),
    ),
    "ADA-EUR": SymbolRules(
        names_cs=("Cardano", "CARDANO"),
        tickers=_coin_tickers("ADA"),
        names_ctx=(
            _coin_ticker("ADA", forbid=("Disabilities", "Dental Association", "Diabetes Association", "Lovelace")),
        ),
        vetoes=("Gerolamo Cardano", "Girolamo Cardano"),
    ),
    "DOGE-EUR": SymbolRules(
        names=("Dogecoin",),
        tickers=_coin_tickers("DOGE", "XDG"),
        names_ctx=(
            _coin_ticker(
                "DOGE",
                forbid=(
                    "Government Efficiency", "Regierungseffizienz", "Effizienzbehörde", "Effizienzbehoerde",
                    "federal workers", "federal employees", "federal agencies", "Bundesbedienstete",
                ),
            ),
        ),
    ),
    "AVAX-EUR": SymbolRules(
        names=("Avalanche network", "Avalanche blockchain", "Avalanche-Blockchain", "Avalanche Foundation", "Ava Labs"),
        tickers=_coin_tickers("AVAX"),
        names_ctx=(
            _coin_ticker("AVAX"),
            _coin_ticker(
                "Avalanche", (*_CRYPTO_STRICT, "AVAX"),
                forbid=("Colorado Avalanche", "Lawine", "avalanche warning", "avalanche risk", "avalanche danger",
                        "skier", "snow", "Schnee", "Seattle Kraken", "NHL", "hockey", "Eishockey"),
            ),
        ),  # fmt: skip
    ),
    "LINK-EUR": SymbolRules(
        names_cs=("Chainlink", "CHAINLINK"),
        tickers=_coin_tickers("LINK"),
        names_ctx=(_coin_ticker("LINK"),),
        vetoes=("Chainlink fenc",),  # chain-link fence
    ),
    "DOT-EUR": SymbolRules(
        names_cs=("Polkadot", "POLKADOT"),
        tickers=_coin_tickers("DOT"),
        names_ctx=(
            _coin_ticker(
                "DOT", forbid=("Department of Transportation", "Transportation Department", "U.S. DOT", "US DOT",
                               "USDOT", "FAA", "Federal Aviation", "highway", "pipeline safety"),
            ),
        ),  # fmt: skip
    ),
    "LTC-EUR": SymbolRules(
        names=("Litecoin",),
        tickers=_coin_tickers("LTC"),
        names_ctx=(_coin_ticker("LTC", forbid=("long-term care", "Long Term Care", "Pflege")),),
    ),
}

_CURATED: Final[dict[str, SymbolRules]] = {
    "AAPL": SymbolRules(
        isin="US0378331005",
        names=("Apple Inc", "iPhone maker", "iPhone-Hersteller", "iPhone-Konzern", "Apple-Aktie"),
        names_cs=("Apple",),
        tickers=("AAPL",),
        vetoes=(
            "Big Apple", "Apple Valley", "Apple Hospitality", "Apple Bank", "Applebee", "Fiona Apple", "Apple Tree",
            "Apple Records", "Apple Corps", "Golden Apple", "Apple Leisure", "Apple Rush",
        ),
    ),
    "MSFT": SymbolRules(isin="US5949181045", names=("Microsoft",), tickers=("MSFT",)),
    "NVDA": SymbolRules(isin="US67066G1040", names=("Nvidia",), tickers=("NVDA",)),
    "AMZN": SymbolRules(
        isin="US0231351067",
        names=("Amazon.com", "Amazon Web Services", "Amazon-Aktie"),
        names_cs=("Amazon", "AWS"),
        tickers=("AMZN",),
        vetoes=(
            "Amazon rainforest", "Amazon basin", "Amazon river", "Amazon forest", "Amazon region", "Amazonas",
            "Brazilian Amazon", "Amazon jungle", "Amazon deforestation", "Amazon tribe", "Amazon fires",
            "the Amazon's",
        ),
    ),
    "GOOGL": SymbolRules(
        isin="US02079K3059",
        names=("Alphabet Inc", "Google", "Google-Mutter", "Google parent", "Alphabet-Aktie", "Waymo", "DeepMind"),
        names_cs=("Alphabet",),
        tickers=("GOOGL", "GOOG"),
        vetoes=("alphabet soup",),
    ),
    "META": SymbolRules(
        isin="US30303M1027",
        names=(
            "Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Meta-Aktie", "Facebook-Mutter", "Facebook parent",
            "Facebook-Konzern",
        ),
        names_cs=("Meta",),
        tickers_ctx=("META",),
        vetoes=(
            "Meta-Analyse", "Meta-Studie", "Meta-Ebene", "Meta-Daten", "Meta-Suche", "Meta analysis", "Meta-analysis",
        ),
    ),
    "TSLA": SymbolRules(isin="US88160R1014", names=("Tesla",), tickers=("TSLA",), vetoes=("Nikola Tesla",)),
    "AVGO": SymbolRules(isin="US11135F1012", names=("Broadcom",), tickers=("AVGO",)),
    "BRK-B": SymbolRules(
        isin="US0846707026",
        names=("Berkshire Hathaway", "Buffetts Berkshire", "Buffett's Berkshire"),
        names_cs=("Berkshire",),
        tickers=("BRK.B", "BRK-B", "BRK.A", "BRK-A", "BRK/B"),
        vetoes=(
            "Berkshire County", "Royal Berkshire", "West Berkshire", "Berkshire Hills", "Berkshire Grey",
            "Berkshire, England",
        ),
    ),
    "JPM": SymbolRules(
        isin="US46625H1005",
        names=("JPMorgan", "JP Morgan", "J.P. Morgan", "JPMorgan Chase"),
        tickers=("JPM",),
        broker=True,
    ),
    "LLY": SymbolRules(
        isin="US5324571083",
        names=("Eli Lilly", "Lilly and Company", "Zepbound", "Mounjaro"),
        tickers=("LLY",),
        names_ctx=(
            ContextAlias(
                "Lilly",
                require=(
                    "Zepbound", "Mounjaro", "obesity", "drug", "pharma", "Pharma", "Novo", "GLP-1", "FDA", "Abnehm",
                    "weight-loss", "weight loss", "orforglipron", "retatrutide", "Alzheimer", "insulin", "Insulin",
                    "Medikament",
                ),
                case_sensitive=True,
            ),
        ),
    ),
    "V": SymbolRules(
        isin="US92826C8394",
        names=("Visa Inc",),
        tickers_ctx=("V",),
        names_ctx=(
            ContextAlias(
                "Visa",
                require=(
                    "Mastercard", "card", "cards", "payment", "payments", "credit", "debit", "network", "swipe",
                    "interchange", "stock", "shares", "earnings", "Aktie", "Kreditkart", "Zahlungs", "Quartal", "NYSE",
                    "fintech", "stablecoin", "merchant", "Dow", "antitrust", "Bezahl",
                ),
                forbid=(
                    "H-1B", "immigration", "visa ban", "visa rules", "visa policy", "tourist", "student visa",
                    "work visa", "visa-free", "Visum", "visa waiver", "visa restrictions", "consulate", "embassy",
                    "golden visa", "visa application", "visas", "Visa-Vergabe", "Einreise", "visa fee", "visa program",
                    "visa holders", "travel ban", "asylum", "border",
                ),
                case_sensitive=True,
            ),
        ),
    ),
    "MA": SymbolRules(isin="US57636Q1040", names=("Mastercard",), tickers_ctx=("MA",)),
    "XOM": SymbolRules(isin="US30231G1022", names=("Exxon Mobil", "ExxonMobil", "Exxon"), tickers=("XOM",)),
    "WMT": SymbolRules(isin="US9311421039", names=("Walmart", "Wal-Mart"), tickers=("WMT",)),
    "UNH": SymbolRules(
        isin="US91324P1021",
        names=("UnitedHealth", "UnitedHealth Group", "UnitedHealthcare", "United Health Group"),
        tickers=("UNH",),
    ),
    "COST": SymbolRules(isin="US22160K1051", names=("Costco",), tickers_ctx=("COST",)),
    "NFLX": SymbolRules(isin="US64110L1061", names=("Netflix",), tickers=("NFLX",)),
    "AMD": SymbolRules(
        isin="US0079031078",
        names=("Advanced Micro Devices",),
        names_cs=("AMD",),
        vetoes=("macular degeneration", "Makuladegeneration"),
    ),
    "ORCL": SymbolRules(
        isin="US68389X1054",
        names=("Oracle Corp", "Oracle-Aktie"),
        names_cs=("Oracle",),
        tickers=("ORCL",),
        vetoes=(
            "Oracle of Omaha", "Oracle Red Bull", "Oracle Park", "Oracle Arena", "price oracle", "Oracle network",
            "Oracle Bone", "Oracle of Delphi",
        ),
    ),
    "JNJ": SymbolRules(isin="US4781601046", names=("Johnson & Johnson", "J&J"), tickers=("JNJ",)),
    "PG": SymbolRules(isin="US7427181091", names=("Procter & Gamble", "P&G"), tickers_ctx=("PG",)),
    "HD": SymbolRules(isin="US4370761029", names=("Home Depot",), tickers_ctx=("HD",)),
    "BAC": SymbolRules(isin="US0605051046", names=("Bank of America", "BofA"), tickers_ctx=("BAC",), broker=True),
    "PLTR": SymbolRules(isin="US69608A1088", names=("Palantir",), tickers=("PLTR",)),
    "SAP.DE": SymbolRules(
        isin="DE0007164600",
        names=("SAP SE", "SAP-Aktie"),
        names_cs=("SAP",),
        tickers=("SAP.DE",),
        vetoes=("SAP Arena", "SAP Garden", "SAP Center"),
    ),
    "SIE.DE": SymbolRules(
        isin="DE0007236101",
        names=("Siemens AG", "Siemens-Aktie"),
        names_cs=("Siemens",),
        tickers=("SIE.DE",),
        tickers_ctx=("SIE",),
        vetoes=(
            "Siemens Energy", "Siemens Healthineers", "Siemens Gamesa", "Siemens-Energy", "Siemens-Healthineers",
            "Siemens-Gamesa", "Siemens Healthcare",
        ),
    ),
    "ALV.DE": SymbolRules(
        isin="DE0008404005",
        names=(
            "Allianz SE", "Allianz-Aktie", "Allianz Global Investors", "Allianz Trade", "Allianz-Chef",
            "Allianz-Konzern", "Allianz-Tochter", "Versicherer Allianz", "Allianz Versicherung",
        ),
        names_cs=("Allianz",),
        tickers=("ALV.DE",),
        tickers_ctx=("ALV",),
        vetoes=(
            "Allianz Arena", "Allianz gegen", "Allianz für", "Allianz fuer", "Allianz aus", "Allianz zwischen",
            "Allianz von", "Allianz der", "Allianz mit", "eine Allianz", "einer Allianz", "neue Allianz",
            "neuen Allianz", "strategische Allianz", "globale Allianz", "breite Allianz", "-Allianz", "Allianz pro",
            "Allianz Deutscher", "Allianz Stadion", "Allianz Field", "Allianz Riviera", "Allianz Parque",
            "Evangelische Allianz", "Allianz Stadium", "Allianz zur", "Allianz zum",
        ),
    ),
    "DTE.DE": SymbolRules(
        isin="DE0005557508",
        names=("Deutsche Telekom", "Dt. Telekom", "Telekom-Aktie", "T-Aktie", "Telekom-Chef"),
        names_cs=("Telekom",),
        tickers=("DTE.DE",),
        tickers_ctx=("DTE",),
        vetoes=(
            "Telekom Austria", "Telekom Srbija", "Telekom Slovenije", "Telekom Malaysia", "Magyar Telekom",
            "Hrvatski Telekom", "Türk Telekom", "Turk Telekom", "Telekom-Branche", "Telekom-Sektor", "Telekomsektor",
            "Telekom-Konzerne", "Telekom-Unternehmen", "Telekomwerte", "Telekom-Werte", "Telekomanbieter",
            "Telekom-Anbieter", "Telekom Baskets",
        ),
    ),
    "AIR.DE": SymbolRules(isin="NL0000235190", names=("Airbus",), tickers=("AIR.DE", "AIR.PA"), tickers_ctx=("AIR",)),
    "MBG.DE": SymbolRules(
        isin="DE0007100000",
        names=("Mercedes-Benz", "Mercedes", "Mercedes-Aktie"),
        names_cs=("Daimler",),
        tickers=("MBG.DE",),
        tickers_ctx=("MBG",),
        vetoes=(
            "Daimler Truck", "Daimler-Truck", "Daimler Buses", "Mercedes-AMG Petronas", "Mercedes-Pilot",
            "Mercedes driver", "Mercedes-Fahrer", "Mercedes-Teamchef", "Mercedes-Benz Arena", "Mercedes-Benz Stadium",
            "Mercedes-Benz Superdome",
        ),
    ),
    "BMW.DE": SymbolRules(
        isin="DE0005190003",
        names=("Bayerische Motoren Werke", "BMW-Aktie"),
        names_cs=("BMW",),
        tickers=("BMW.DE",),
    ),
    "VOW3.DE": SymbolRules(
        isin="DE0007664039",
        names=("Volkswagen", "VW-Konzern", "VW-Aktie", "VW-Chef"),
        names_cs=("VW",),
        tickers=("VOW3.DE", "VOW3", "VOW.DE"),
        vetoes=("Volkswagen Arena",),
    ),
    "BAS.DE": SymbolRules(isin="DE000BASF111", names=("BASF",), tickers=("BAS.DE",), tickers_ctx=("BAS",)),
    "IFX.DE": SymbolRules(isin="DE0006231004", names=("Infineon",), tickers=("IFX.DE",), tickers_ctx=("IFX",)),
    "RHM.DE": SymbolRules(isin="DE0007030009", names=("Rheinmetall",), tickers=("RHM.DE",), tickers_ctx=("RHM",)),
    "MUV2.DE": SymbolRules(
        isin="DE0008430026",
        names=(
            "Munich Re", "Münchener Rück", "Münchner Rück", "Münchener Rückversicherung", "Munich Reinsurance",
            "MunichRe",
        ),
        tickers=("MUV2.DE", "MUV2"),
    ),
    "DBK.DE": SymbolRules(
        isin="DE0005140008",
        names=("Deutsche Bank", "Deutsche-Bank-Aktie", "Dt. Bank"),
        tickers=("DBK.DE",),
        tickers_ctx=("DBK",),
        vetoes=("Deutsche Bank Park", "Deutsche Bank Center", "Deutsche Bank Championship"),
        broker=True,
    ),
    "ADS.DE": SymbolRules(isin="DE000A1EWWW0", names=("Adidas",), tickers=("ADS.DE",), tickers_ctx=("ADS",)),
    "BAYN.DE": SymbolRules(
        isin="DE000BAY0017",
        names=("Bayer AG", "Bayer-Aktie", "Bayer-Konzern", "Bayer-Chef"),
        names_cs=("Bayer",),
        tickers=("BAYN.DE", "BAYN"),
        vetoes=(
            "Bayer Leverkusen", "Bayer 04", "Bayer-Leverkusen", "Bayer-Trainer", "Bayer-Coach", "Bayer-Profi",
            "Bayer-Stürmer", "Bayer Uerdingen",
        ),
    ),
    "ENR.DE": SymbolRules(
        isin="DE000ENER6Y0",
        names=("Siemens Energy", "Siemens-Energy"),
        tickers=("ENR.DE",),
        tickers_ctx=("ENR",),
    ),
    "ASML.AS": SymbolRules(
        isin="NL0010273215",
        names=("ASML Holding", "ASML-Aktie"),
        names_cs=("ASML",),
        tickers=("ASML.AS",),
    ),
    "MC.PA": SymbolRules(
        isin="FR0000121014",
        names=("LVMH", "Louis Vuitton", "Moët Hennessy"),
        tickers=("MC.PA",),
        tickers_ctx=("MC",),
    ),
    "OR.PA": SymbolRules(isin="FR0000120321", names=("L'Oréal", "LOreal"), tickers=("OR.PA",), tickers_ctx=("OR",)),
    "TTE.PA": SymbolRules(
        isin="FR0000120271",
        names=("TotalEnergies", "Total Energies"),
        tickers=("TTE.PA",),
        tickers_ctx=("TTE",),
    ),
    "SAN.PA": SymbolRules(isin="FR0000120578", names=("Sanofi",), tickers=("SAN.PA",), tickers_ctx=("SAN",)),
    "NESN.SW": SymbolRules(
        isin="CH0038863350",
        names=("Nestlé",),
        tickers=("NESN.SW", "NESN"),
        vetoes=(
            "Nestlé India", "Nestlé Nigeria", "Nestlé Malaysia", "Nestlé Pakistan", "Nestlé Lanka",
            "Nestlé Bangladesh",
        ),
    ),
    "NOVN.SW": SymbolRules(isin="CH0012005267", names=("Novartis",), tickers=("NOVN.SW", "NOVN")),
    "ROG.SW": SymbolRules(
        isin="CH0012032048",
        names=("Roche Holding", "Hoffmann-La Roche", "Roche-Aktie"),
        names_cs=("Roche",),
        tickers=("ROG.SW",),
        tickers_ctx=("ROG",),
        vetoes=("La Roche-Posay", "La Roche Posay", "Roche Bobois"),
    ),
    "NOVO-B.CO": SymbolRules(
        isin="DK0062498333",
        names=("Novo Nordisk", "Wegovy", "Ozempic"),
        tickers=("NOVO-B.CO", "NOVO-B", "NVO"),
        names_ctx=(
            ContextAlias(
                "Novo",
                require=(
                    "Wegovy", "Ozempic", "obesity", "Nordisk", "Lilly", "Abnehm", "weight-loss", "weight loss",
                    "GLP-1", "Danish", "dänisch", "semaglutide", "CagriSema", "drugmaker", "Pharma",
                ),
                forbid=("Novo Banco", "Novo Mesto", "Novo Holdings", "Novo Nordisk Foundation"),
                case_sensitive=True,
            ),
        ),
    ),
    "SHEL.L": SymbolRules(
        isin="GB00BP6MXD84",
        names=("Shell plc", "Royal Dutch Shell", "Shell-Aktie", "Shell-Konzern", "Ölkonzern Shell", "oil major Shell"),
        names_cs=("Shell",),
        tickers=("SHEL.L", "SHEL"),
        vetoes=(
            "shell compan", "shell game", "shell out", "shell shock", "shell-shock", "shell script",
            "shell corporation", "shell firm", "shell entit", "nuclear shell", "artillery shell", "Shell-Schock",
            "egg shell", "shell casing", "shell bank", "tank shell", "mortar shell", "shell deal",
        ),
    ),
    "AZN.L": SymbolRules(isin="GB0009895292", names=("AstraZeneca", "Astra Zeneca"), tickers=("AZN.L", "AZN")),
    "HSBA.L": SymbolRules(
        isin="GB0005405286",
        names=("HSBC Holdings",),
        names_cs=("HSBC",),
        tickers=("HSBA.L", "HSBA"),
        broker=True,
    ),
    "7203.T": SymbolRules(
        isin="JP3633400001",
        names=("Toyota", "Toyota Motor"),
        tickers=("7203.T",),
        vetoes=("Toyota Industries", "Toyota Tsusho", "Toyota Boshoku", "Toyota Center", "Toyota Stadium"),
    ),
    "6758.T": SymbolRules(isin="JP3435000009", names=("Sony",), tickers=("6758.T",), vetoes=("Sony Financial",)),
    "9984.T": SymbolRules(isin="JP3436100006", names=("SoftBank", "SoftBank Group"), tickers=("9984.T",)),
    "8306.T": SymbolRules(isin="JP3902900004", names=("Mitsubishi UFJ", "MUFG"), tickers=("8306.T",), broker=True),
    "6861.T": SymbolRules(isin="JP3236200006", names=("Keyence",), tickers=("6861.T",)),
    "8035.T": SymbolRules(isin="JP3571400005", names=("Tokyo Electron",), tickers=("8035.T",)),
    "0700.HK": SymbolRules(
        isin="KYG875721634",
        names=("Tencent",),
        tickers=("0700.HK", "700.HK"),
        vetoes=("Tencent Music",),
    ),
    "9988.HK": SymbolRules(isin="KYG017191142", names=("Alibaba",), tickers=("9988.HK", "BABA")),
    "3690.HK": SymbolRules(isin="KYG596691041", names=("Meituan",), tickers=("3690.HK",)),
    "1299.HK": SymbolRules(
        isin="HK0000069689",
        names=("AIA Group",),
        tickers=("1299.HK",),
        names_ctx=(
            ContextAlias(
                "AIA",
                require=(
                    "insur", "Versicher", "Hong Kong", "Hongkong", "HKEX", "Hang Seng", "shares", "premium",
                    "Prudential",
                ),
                case_sensitive=True,
            ),
        ),
    ),
    "1810.HK": SymbolRules(isin="KYG9830T1067", names=("Xiaomi",), tickers=("1810.HK",)),
    "005930.KS": SymbolRules(
        isin="KR7005930003",
        names=("Samsung Electronics", "Samsung Elec"),
        names_cs=("Samsung",),
        tickers=("005930.KS", "005930"),
        vetoes=(
            "Samsung SDI", "Samsung Biologics", "Samsung Heavy", "Samsung C&T", "Samsung Life", "Samsung Securities",
            "Samsung Fire", "Samsung Electro-Mechanics", "Samsung SDS", "Samsung Engineering", "Samsung Bioepis",
            "Samsung Card", "Samsung Lions", "Samsung Asset", "Samsung E&A",
        ),
    ),
    "000660.KS": SymbolRules(isin="KR7000660001", names=("SK Hynix", "Hynix"), tickers=("000660.KS", "000660")),
    "2330.TW": SymbolRules(
        isin="TW0002330008",
        names=("TSMC", "Taiwan Semiconductor"),
        tickers=("2330.TW",),
        tickers_ctx=("TSM",),
    ),
    "600519.SS": SymbolRules(
        isin="CNE0000018R8",
        names=("Kweichow Moutai", "Moutai", "Maotai"),
        tickers=("600519.SS", "600519"),
    ),
    "300750.SZ": SymbolRules(
        isin="CNE100003662",
        names=("Contemporary Amperex",),
        names_cs=("CATL",),
        tickers=("300750.SZ", "300750", "3750.HK"),
    ),
    "RELIANCE.NS": SymbolRules(
        isin="INE002A01018",
        names=("Reliance Industries", "Reliance Jio", "Reliance Retail", "Jio Platforms", "Reliance Brands"),
        names_cs=("RIL",),
        tickers=("RELIANCE.NS", "RELIANCE.BO"),
        vetoes=(
            "Reliance Power", "Reliance Infrastructure", "Reliance Infra", "Reliance Capital",
            "Reliance Communications", "Reliance Home Finance", "Reliance Steel", "Reliance Global",
            "Reliance Worldwide",
        ),
        names_ctx=(
            ContextAlias(
                "Reliance",
                require=("Ambani", "Jio", "RIL", "Sensex", "Nifty", "crore", "Mumbai", "refin", "Rs "),
                forbid=(
                    "Reliance on", "Reliance Power", "Reliance Infra", "Reliance Capital", "Reliance Communications",
                    "Reliance Home", "Reliance Steel", "Reliance Global", "Reliance Worldwide",
                ),
                case_sensitive=True,
            ),
        ),
    ),
    "BHP.AX": SymbolRules(
        isin="AU000000BHP4",
        names=("BHP Group", "BHP Billiton"),
        names_cs=("BHP",),
        tickers=("BHP.AX",),
    ),
    "^GSPC": SymbolRules(names=("S&P 500", "S&P500", "Standard & Poor's 500", "S&P-500-Index"), names_cs=("SPX",)),
    "^NDX": SymbolRules(
        names=(
            "Nasdaq 100", "Nasdaq100", "Nasdaq Composite", "Nasdaq index", "tech-heavy Nasdaq",
            "Technologiebörse Nasdaq", "Nasdaq-Index", "Technologieindex Nasdaq",
        ),
        names_cs=("NDX",),
        names_ctx=(ContextAlias("Nasdaq", require=_MARKET_MOVES, forbid=_NASDAQ_VENUE),),
    ),
    "^DJI": SymbolRules(
        names=(
            "Dow Jones", "Dow Jones Industrial Average", "DJIA", "Dow Industrials", "Dow-Jones-Index", "Dow futures",
        ),
        vetoes=("Dow Jones Newswires", "Dow Jones & Company", "Dow Jones Sustainability", "Dow Jones & Co"),
        names_ctx=(
            ContextAlias(
                "Dow",
                require=(*_MARKET_MOVES, "S&P", "Nasdaq"),
                forbid=("Dow Inc", "Dow Chemical", "Dow Corning", "Dow Jones Newswires", "Dow Jones & Co"),
                case_sensitive=True,
            ),
        ),
    ),
    "^RUT": SymbolRules(names=("Russell 2000",)),
    "^VIX": SymbolRules(
        names=("CBOE Volatility Index", "Cboe Volatility", "fear gauge", "Angstbarometer"),
        names_cs=("VIX",),
        vetoes=("India VIX", "VIX India"),
    ),
    "^GDAXI": SymbolRules(
        names=("Deutscher Aktienindex", "DAX 40", "Leitindex Dax", "German DAX", "Germany 40"),
        names_cs=("DAX", "Dax"),
        vetoes=_DAX_VETOES,
    ),
    "^STOXX50E": SymbolRules(
        names=(
            "Euro Stoxx 50", "EuroStoxx 50", "EuroStoxx50", "Euro Stoxx", "EuroStoxx", "Eurozone 50", "ESTX50",
            "European shares", "European stocks", "STOXX 600", "Stoxx Europe 600", "Europas Börsen",
            "europäische Aktien", "European equities",
        ),
    ),
    "^FTSE": SymbolRules(
        names=("FTSE 100", "FTSE100", "Footsie"),
        names_ctx=(
            ContextAlias(
                "FTSE",
                require=(*_MARKET_MOVES, "London", "UK", "blue-chip"),
                forbid=(
                    "FTSE 250", "FTSE Russell", "FTSE China", "FTSE MIB", "FTSE All", "FTSE4Good", "FTSE Bursa",
                    "FTSE EPRA", "FTSE Developed", "FTSE Emerging", "FTSE World",
                ),
                case_sensitive=True,
            ),
        ),
    ),
    "^FCHI": SymbolRules(names=("CAC 40", "CAC40")),
    "^SSMI": SymbolRules(
        names=("Swiss Market Index", "Schweizer Leitindex", "Leitindex SMI"),
        names_cs=("SMI",),
        vetoes=("SMI Mid",),
    ),
    "^N225": SymbolRules(
        names=(
            "Nikkei 225", "Nikkei225", "Nikkei index", "Nikkei-Index", "Nikkei average", "Nikkei Stock Average",
            "Nikkei share average", "Japan 225", "Tokyo stocks", "Tokyo shares", "Japanese stocks", "Japanese shares",
            "Japan stocks",
        ),
        names_ctx=(ContextAlias("Nikkei", require=_NIKKEI_MOVES, forbid=_NIKKEI_PAPER, case_sensitive=True),),
    ),
    "^HSI": SymbolRules(
        names=("Hang Seng", "Hang-Seng-Index", "Hong Kong stocks", "Hong Kong shares"),
        tickers_ctx=("HSI",),
        vetoes=("Hang Seng Bank", "Hang Seng Tech"),
    ),
    "000001.SS": SymbolRules(
        names=(
            "Shanghai Composite", "SSE Composite", "Shanghai stocks", "Shanghai index", "China stocks",
            "Chinese stocks", "Chinese shares", "China shares", "CSI 300", "chinesische Aktien", "Chinas Börsen",
        ),
        tickers=("000001.SS",),
    ),
    "^KS11": SymbolRules(names=("KOSPI", "Korea Composite", "South Korean stocks", "Seoul stocks", "Seoul shares")),
    "^TWII": SymbolRules(names=("Taiex", "Taiwan Weighted", "Taiwan stocks", "Taiwan shares", "Taipei stocks")),
    "^NSEI": SymbolRules(
        names=("Nifty 50", "Nifty50", "NSE Nifty", "Sensex"),
        names_cs=("Nifty",),
        vetoes=(
            "Nifty Bank", "Bank Nifty", "Nifty IT", "Nifty Midcap", "Nifty Next", "Nifty Auto", "Nifty Pharma",
            "Nifty FMCG", "Nifty Metal", "Nifty Realty", "Nifty Smallcap", "Nifty PSU",
        ),
    ),
    "^AXJO": SymbolRules(
        names=("ASX 200", "ASX200", "S&P/ASX 200", "Australian shares", "Australian stocks", "Aussie shares"),
    ),
    "BTC-USD": SymbolRules(
        names=("Bitcoin",),
        names_cs=("BTC",),
        names_prefix=("Bitcoin-Kurs", "Bitcoinkurs", "Bitcoin-Preis"),
        tickers=("BTC-USD", "BTCUSD", "BTC/USD", "XBT"),
        vetoes=("Bitcoin Cash", "Bitcoin SV", "Bitcoin Depot", "Bitcoin Magazine"),
    ),
    "ETH-USD": SymbolRules(
        names=("Ethereum",),
        names_cs=("Ether", "ETH"),
        tickers=("ETH-USD", "ETHUSD", "ETH/USD"),
        vetoes=(
            "ETH Zürich", "ETH Zurich", "ETH Lausanne", "Ethereum Classic", "Ether.fi", "ETH-Bibliothek", "ETH-Rat",
        ),
    ),
    "SOL-USD": SymbolRules(names=("Solana",), tickers=("SOL-USD", "SOLUSD", "SOL/USD"), tickers_ctx=("SOL",)),
    "EURUSD=X": SymbolRules(
        names=(
            "EUR/USD", "EURUSD", "Euro-Dollar", "Euro/Dollar", "Euro-Kurs", "euro against the dollar",
            "euro vs dollar", "Euro zum Dollar", "Euro gegenüber dem Dollar", "euro versus the dollar",
        ),
        names_prefix=("Eurokurs",),
    ),
    "EURJPY=X": SymbolRules(names=("EUR/JPY", "EURJPY", "Euro/Yen", "Euro-Yen", "yen against the euro")),
    "EURGBP=X": SymbolRules(
        names=(
            "EUR/GBP", "EURGBP", "Euro/Pfund", "Euro-Pfund", "euro-sterling", "pound against the euro",
            "sterling against the euro", "britische Pfund", "Britische Pfund", "British pound", "GBP/USD",
            "pound sterling",
        ),
        names_ctx=(
            ContextAlias(
                "sterling",
                require=(*_FX_MOVES, "BoE", "Bank of England", "gilt"),
                amount_guard=True,
            ),
            ContextAlias(
                "Pfund",
                require=("Euro", "Dollar", "Währung", "Bank of England", "Devisen", "Wechselkurs"),
                case_sensitive=True,
                amount_guard=True,
            ),
        ),
    ),
    "EURCHF=X": SymbolRules(
        names=(
            "EUR/CHF", "EURCHF", "Euro/Franken", "Euro-Franken", "Swiss franc", "Schweizer Franken", "Frankenkurs",
            "USD/CHF",
        ),
        names_ctx=(
            ContextAlias(
                "Franken",
                require=(
                    "Euro", "SNB", "Nationalbank", "Dollar", "Währung", "Devisen", "Kurs", "Aufwertung", "Abwertung",
                ),
                case_sensitive=True,
                amount_guard=True,
            ),
        ),
    ),
    "EURHKD=X": SymbolRules(names=("EUR/HKD", "EURHKD", "Hong Kong dollar", "Hongkong-Dollar", "USD/HKD", "HKD peg")),
    "EURKRW=X": SymbolRules(
        names=(
            "EUR/KRW", "EURKRW", "Korean won", "South Korean won", "südkoreanische Won", "südkoreanischen Won",
            "USD/KRW",
        ),
    ),
    "EURTWD=X": SymbolRules(
        names=("EUR/TWD", "EURTWD", "Taiwan dollar", "New Taiwan dollar", "Taiwan-Dollar", "USD/TWD"),
    ),
    "EURCNY=X": SymbolRules(
        names=(
            "EUR/CNY", "EURCNY", "USD/CNY", "USD/CNH", "onshore yuan", "offshore yuan", "yuan fixing", "yuan midpoint",
            "Renminbi",
        ),
        names_ctx=(
            ContextAlias(
                "yuan",
                require=(*_FX_MOVES, "PBOC", "PBoC", "People's Bank", "fixing", "midpoint"),
                amount_guard=True,
            ),
        ),
    ),
    "EURDKK=X": SymbolRules(
        names=("EUR/DKK", "EURDKK", "Danish krone", "Danish crown", "dänische Krone", "dänischen Krone"),
    ),
    "EURINR=X": SymbolRules(
        names=("EUR/INR", "EURINR", "USD/INR", "Indian rupee", "indische Rupie"),
        names_ctx=(
            ContextAlias(
                "rupee",
                require=(*_FX_MOVES, "RBI", "Reserve Bank", "paise"),
                amount_guard=True,
            ),
        ),
    ),
    "EURAUD=X": SymbolRules(
        names=(
            "EUR/AUD", "EURAUD", "Australian dollar", "Aussie dollar", "australische Dollar", "australischen Dollar",
            "AUD/USD",
        ),
    ),
    "USDJPY=X": SymbolRules(
        names=(
            "USD/JPY", "USDJPY", "Dollar/Yen", "Dollar-Yen", "yen against the dollar", "yen per dollar",
            "Yen zum Dollar", "Japanese yen", "japanische Yen", "japanischen Yen",
        ),
        names_ctx=(
            ContextAlias(
                "yen",
                require=(
                    *_FX_MOVES, "BOJ", "BoJ", "Bank of Japan", "slides", "slumps", "surges", "jumps", "falls", "rises",
                    "weak", "strong",
                ),
                amount_guard=True,
            ),
        ),
    ),
    "ES=F": SymbolRules(
        names=(
            "S&P 500 futures", "S&P 500 future", "S&P futures", "E-mini S&P", "S&P 500 E-mini", "S&P-500-Future",
            "US stock futures", "U.S. stock futures", "Wall Street futures", "US equity futures",
            "U.S. equity futures", "US-Futures", "US-Aktienfutures",
        ),
        tickers=("ES=F",),
    ),
    "NQ=F": SymbolRules(
        names=("Nasdaq futures", "Nasdaq 100 futures", "Nasdaq-Future", "Nasdaq-100-Future", "E-mini Nasdaq"),
        tickers=("NQ=F",),
    ),
    "GC=F": SymbolRules(
        names=(
            "gold price", "gold prices", "gold futures", "spot gold", "Comex gold", "XAU/USD", "XAUUSD",
            "gold bullion", "Feinunze Gold",
        ),
        names_prefix=("Goldpreis", "Goldkurs"),
        tickers=("GC=F",),
        names_ctx=(
            ContextAlias(
                "gold",
                require=(
                    "ounce", "bullion", "silver", "Silber", "precious", "Edelmetall", "safe haven", "safe-haven",
                    "record high", "Rekordhoch", "rally", "Fed", "dollar", "yields", "inflation", "Unze", "prices",
                    "Preis", "metals", "Comex", "central bank", "Zentralbank", "ETF", "spot", "$", "Bitcoin",
                    "commodit", "Rohstoff", "crash", "low", "falls", "fällt", "steigt", "rises", "bull", "bear",
                    "outlook", "forecast", "Prognose", "Ausblick", "weakness", "demand", "Nachfrage",
                ),
                forbid=(
                    "gold medal", "Goldmedaille", "gold standard", "Gold Coast", "golden", "Gold Cup", "gold card",
                    "gold-plated", "Gold Fields", "Gold Royalty", "Gold Corp", "Gold Mining", "Gold Mines", "Gold Inc",
                    "Gold Ltd", "Gold Resources", "Gold Reserve", "Olympic", "copper-gold", "gold-copper",
                    "gold project", "gold mine", "gold deposit", "gold explor", "gold discovery", "Goldprojekt",
                    "Goldmine", "drill",
                ),
            ),
        ),
    ),
    "CL=F": SymbolRules(
        names=(
            "West Texas Intermediate", "crude oil", "oil price", "oil prices", "crude futures", "crude prices",
            "US crude", "U.S. crude", "oil futures",
        ),
        names_cs=("WTI",),
        names_prefix=("Ölpreis", "Rohöl"),
        tickers=("CL=F",),
        vetoes=(
            "olive oil", "palm oil", "cooking oil", "edible oil", "fish oil", "essential oil", "Olivenöl", "Palmöl",
            "Speiseöl", "sunflower oil", "vegetable oil", "heating oil", "Heizöl",
        ),
        names_ctx=(
            ContextAlias(
                "oil",
                require=(
                    "OPEC", "barrel", "Brent", "WTI", "crude", "prices", "futures", "output", "supply", "Barrel",
                    "Fass", "inventories", "stockpiles", "refiner", "tanker", "embargo", "sanction", "glut",
                ),
                forbid=(
                    "olive oil", "palm oil", "cooking oil", "oil painting", "essential oil", "snake oil", "Olivenöl",
                    "Palmöl", "Speiseöl", "edible oil", "fish oil", "oil major", "oil giant", "oil company",
                    "oil companies",
                ),
            ),
        ),
    ),
    "BZ=F": SymbolRules(
        names=(
            "Brent crude", "Brent oil", "Brent futures", "Nordseeöl", "Nordseesorte Brent", "Brent-Öl", "Brent-Rohöl",
            "Sorte Brent",
        ),
        tickers=("BZ=F",),
        names_ctx=(
            ContextAlias(
                "Brent",
                require=("crude", "oil", "barrel", "OPEC", "WTI", "Öl", "Barrel", "Fass", "$"),
                case_sensitive=True,
            ),
        ),
    ),
}

# Roche: the default universe trades the bearer share ROP.SW instead of the Genussschein ROG.SW.
# Crypto: the Kraken EUR pairs replace BTC-USD / ETH-USD / SOL-USD (whose rules stay for old databases).
CURATED_RULES: Final[Mapping[str, SymbolRules]] = MappingProxyType(
    {
        **_CURATED,
        "ROP.SW": replace(_CURATED["ROG.SW"], isin=None, tickers=("ROP.SW",), tickers_ctx=("ROP", "ROG")),
        **_KRAKEN_COINS,
    }
)

# ------------------------------------------------------------------------------------------------
# Macro terms: topic -> the index symbols it moves
# ------------------------------------------------------------------------------------------------
MACRO_TERMS: Final[tuple[MacroTerm, ...]] = (
    # --- market-wide phrases and index names -----------------------------------------------------
    MacroTerm(
        "WALL_STREET",
        ("^GSPC",),
        names=("Wall Street", "S&P 500", "S&P500", "US stocks", "U.S. stocks", "US-Börsen", "US-Aktien", "Wall-Street"),
    ),
    MacroTerm(
        "DAX",
        ("^GDAXI",),
        names=("Deutscher Aktienindex", "Leitindex Dax", "German stocks", "German shares", "deutsche Aktien"),
        names_cs=("DAX", "Dax"),
        vetoes=_DAX_VETOES,
    ),
    MacroTerm(
        "NASDAQ",
        ("^NDX",),
        names=("Nasdaq 100", "Nasdaq Composite", "tech-heavy Nasdaq", "Technologiebörse Nasdaq", "Nasdaq-Index"),
        names_ctx=(ContextAlias("Nasdaq", require=_MARKET_MOVES, forbid=_NASDAQ_VENUE),),
    ),
    MacroTerm(
        "NIKKEI",
        ("^N225",),
        names=("Nikkei 225", "Nikkei-Index", "Nikkei average", "Tokyo stocks", "Tokyo shares", "Japanese stocks"),
        names_ctx=(ContextAlias("Nikkei", require=_NIKKEI_MOVES, forbid=_NIKKEI_PAPER, case_sensitive=True),),
    ),
    MacroTerm(
        "HANG_SENG",
        ("^HSI",),
        names=("Hang Seng", "Hang-Seng-Index", "Hong Kong stocks", "Hong Kong shares"),
        vetoes=("Hang Seng Bank", "Hang Seng Tech"),
    ),
    MacroTerm(
        "EUROPE",
        ("^STOXX50E",),
        names=(
            "Euro Stoxx 50", "Euro Stoxx", "European shares", "European stocks", "STOXX 600", "Europas Börsen",
            "europäische Aktien", "European equities",
        ),
    ),
    # --- central banks -------------------------------------------------------------------------
    MacroTerm(
        "FED",
        ("^GSPC", "^NDX"),
        names=(
            "Federal Reserve", "FOMC", "US-Notenbank", "Fed-Chef", "Fed chair", "Fed-Sitzung", "Fed meeting",
            "Fed minutes", "Fed-Protokoll", "Fed-Zins", "Fed governor", "Fed officials", "Fed rate", "US-Leitzins",
            "Notenbank Fed",
        ),
        names_cs=("Fed",),
        vetoes=("Fed up", "Fed Up", "Fed Cup", "Fed-Ex"),
    ),
    MacroTerm(
        "ECB",
        ("^GDAXI", "^STOXX50E"),
        names=(
            "European Central Bank", "Europäische Zentralbank", "Europäischen Zentralbank", "EZB-Rat",
            "ECB Governing Council", "Lagarde", "EZB-Chefin", "EZB-Direktorium",
        ),
        names_cs=("ECB", "EZB"),
        vetoes=("Cricket",),
    ),
    MacroTerm(
        "BOJ",
        ("^N225",),
        names=("Bank of Japan", "japanische Notenbank", "Japans Notenbank"),
        names_cs=("BOJ", "BoJ"),
    ),
    MacroTerm("BOE", ("^FTSE",), names=("Bank of England", "britische Notenbank"), names_cs=("BoE", "BOE")),
    MacroTerm(
        "SNB",
        ("^SSMI",),
        names=("Swiss National Bank", "Schweizerische Nationalbank", "Schweizer Nationalbank"),
        names_cs=("SNB",),
    ),
    MacroTerm(
        "PBOC",
        ("000001.SS", "^HSI"),
        names=("People's Bank of China", "chinesische Notenbank", "chinesische Zentralbank", "loan prime rate"),
        names_cs=("PBOC", "PBoC"),
    ),
    MacroTerm("BOK", ("^KS11",), names=("Bank of Korea",), names_cs=("BOK", "BoK")),
    MacroTerm("RBI", ("^NSEI",), names=("Reserve Bank of India",), names_cs=("RBI",)),
    MacroTerm("RBA", ("^AXJO",), names=("Reserve Bank of Australia",), names_cs=("RBA",)),
    # --- economic data -------------------------------------------------------------------------
    MacroTerm(
        "US_DATA",
        ("^GSPC", "^NDX"),
        names=(
            "nonfarm payrolls", "non-farm payrolls", "jobs report", "US-Arbeitsmarktbericht", "consumer price index",
            "US-Inflation", "U.S. inflation", "US inflation", "jobless claims", "retail sales",
            "US-Einzelhandelsumsätze", "payrolls", "US-Verbraucherpreise", "consumer sentiment", "US GDP", "U.S. GDP",
            "US-BIP", "Erstanträge auf Arbeitslosenhilfe",
        ),
        names_cs=("CPI", "PCE", "ISM"),
    ),
    MacroTerm(
        "EZ_DATA",
        ("^GDAXI", "^STOXX50E"),
        names=(
            "Eurozone inflation", "euro zone inflation", "euro area inflation", "Inflation im Euroraum",
            "Inflation in der Eurozone", "Eurozone GDP", "ifo-Geschäftsklima", "ifo-Index", "ifo business climate",
            "ZEW-Konjunkturerwartungen", "ZEW-Index", "Einkaufsmanagerindex", "deutsche Inflation",
            "German inflation", "Verbraucherpreise", "Erzeugerpreise", "Bruttoinlandsprodukt",
        ),
        names_cs=("HICP", "HVPI", "BIP"),
    ),
    # --- commodities and trade -----------------------------------------------------------------
    MacroTerm("OPEC", ("CL=F", "BZ=F"), names=("OPEC+", "OPEC"), names_cs=("Opec",)),
    MacroTerm(
        "TRADE",
        ("^GSPC", "^GDAXI", "^HSI"),
        names=(
            "tariffs", "tariff", "Strafzölle", "Zölle", "trade war", "Handelskrieg", "export controls",
            "Exportkontrollen", "export curbs", "Handelsstreit", "trade deal", "Handelsabkommen",
        ),
    ),
)

# Tickers that are ordinary words or abbreviations: they only count in ticker context ('$V', 'NYSE: V').
AMBIGUOUS_TICKERS: Final[frozenset[str]] = frozenset(
    {
        "V", "MA", "HD", "PG", "COST", "META", "BAC", "AIR", "SAN", "BAS", "ADS", "ALV", "SIE", "DTE", "MBG", "IFX",
        "RHM", "DBK", "ENR", "MC", "OR", "TTE", "ROG", "ROP", "TSM", "HSI", "SOL", "ALL", "ARE", "BIG", "CAN", "CAT",
        "CEO", "EAT", "ETF", "FUN", "GDP", "HAS", "IPO", "KEY", "LOW", "NOW", "ONE", "OUT", "TRUE", "USA", "WELL",
        "ADA", "DOT", "LINK", "DOGE", "LTC", "AVAX", "ETH", "BTC", "XBT",  # coin tickers: see _KRAKEN_COINS
    }
)


# ------------------------------------------------------------------------------------------------
# Pattern building
# ------------------------------------------------------------------------------------------------
_ACCENTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "é": "[eé]", "è": "[eè]", "ë": "[eë]", "ö": "(?:ö|oe)", "ü": "(?:ü|ue)", "ä": "(?:ä|ae)", "ß": "(?:ß|ss)",
        "Ö": "(?:Ö|Oe|OE)", "Ü": "(?:Ü|Ue|UE)", "Ä": "(?:Ä|Ae|AE)",
    }
)
_HYPHEN: Final = r"[\s\-‐-―]"
_EXCHANGES: Final = (
    r"(?:NYSE|NASDAQ|Nasdaq|NasdaqGS|NasdaqGM|NasdaqCM|XETRA|Xetra|ETR|FRA|LSE|LON|EPA|SWX|VTX|TYO|TSE|HKG|HKEX|"
    r"SEHK|KRX|ASX|NSE|BSE|AMEX|NYSEARCA|OTC[A-Z]*|CPH|AMS)"
)
_AMOUNT_BEFORE: Final = re.compile(
    r"(?:\d[\d.,]*\s*(?:billion|million|trillion|bn|mn|m|k|Milliarden|Millionen|Billionen|Mrd\.?|Mio\.?)?"
    r"|(?:billion|million|trillion|Milliarden|Millionen|Mrd\.?|Mio\.?))\s*-?$",
    re.I,
)
_RATING_VERB: Final = (
    r"\b(?:hebt|senkt|stuft|belässt|belaesst|bestätigt|startet|erhöht|kürzt|raises|raised|cuts|cut|lifts|lowers|"
    r"upgrades|upgraded|downgrades|downgraded|initiates|reiterates|maintains|starts|resumes|hikes|trims|boosts|"
    r"slashes)\b"
)
_RATING_WORD: Final = (
    r"(?:price target|\btarget\b|\bPT\b|rating|Kursziel|\bZiel\b|Overweight|Underweight|Outperform|Underperform|"
    r"\bNeutral\b|\bBuy\b|\bSell\b|\bHold\b|Kaufen|Verkaufen|Halten|Equal[- ]weight|coverage|Einstufung)"
)
_RATING_LABEL: Final = (
    r"""['"‘’“„](?:Buy|Sell|Hold|Neutral|Overweight|Underweight|Outperform|Underperform|Kaufen|Verkaufen|"""
    r"""Halten|Equal[- ]weight|Market Perform|Sector Perform|Add|Reduce)['"‘’“”]"""
)
# A bank is a BROKER (context, not the subject) when it is followed by research/analyst words, a rating verb
# with a rating word ('hebt SAP auf Overweight'), a price target or a quoted rating label.
_BROKER_AFTER: Final = re.compile(
    r"^[\s'’s]{0,3}(?:(?:global\s+)?research|securities|analyst\w*|strategist\w*|economist\w*|survey|poll|study|"
    r"Umfrage|Studie)\b"
    r"|^.{0,25}?" + _RATING_VERB + r".{0,70}?" + _RATING_WORD
    + r"|^.{3,80}?(?:price targets?|Kursziel\w*|\bPT\b)"
    r"|^.{0,80}?" + _RATING_LABEL,
    re.I | re.S,
)
_BROKER_BEFORE: Final = re.compile(
    r"(?:analysts?|strategists?|economists?)\s+(?:at|from|bei|von|der)\s+(?:the\s+)?$|(?:according to|laut)\s+$", re.I
)
_BROKER_TRAIL_BEFORE: Final = re.compile(r"(?:\s[–—-]|:)\s*(?:says\s+|laut\s+)?$")  # '... – MUFG', '...: HSBC'
_RATED_OBJECT_BEFORE: Final = re.compile(_RATING_VERB + r"(?:\s+\S+){0,3}\s+$", re.I)  # 'UBS stuft Deutsche Bank'
_COMMENT_AFTER: Final = re.compile(
    r"^\s{1,2}(?:says|sees|expects|warns|predicts|forecasts|favou?rs|recommends|flags|rät|erwartet|sieht|warnt|"
    r"empfiehlt|prognostiziert)\b",
    re.I,
)
_ISIN_IN_TEXT: Final = re.compile(r"(?<![A-Z0-9])[A-Z]{2}[A-Z0-9]{9}\d(?![A-Z0-9])")
_ALNUM_RUN: Final = re.compile(r"[A-Za-z0-9]+")
_SINGLE_WORD: Final = re.compile(r"[^\W\d_]+")
_LEGAL_SUFFIX: Final = re.compile(
    r"(?:[\s,]+(?:inc|incorporated|corp|corporation|co|company|ag|se|kgaa|plc|n\.?v|s\.?a|s\.?p\.?a|a/s|ltd|"
    r"limited|holdings?|group|aktiengesellschaft|vz|vzo|& co|class [a-c]|adr)\.?)+$",
    re.I,
)
_SHORT_ALIAS: Final = 3  # aliases up to this length match only in exactly the given spelling


def _alias_rx(alias: str) -> str:
    """A plain alias as tolerant regex fragment (flexible spaces/hyphens, '&'/'and'/'und', accents, dots)."""
    alias = re.sub(r"\s*&\s*", "&", alias.strip())
    out: list[str] = []
    for i, ch in enumerate(alias):
        if ch == " ":
            out.append(_HYPHEN + "+")
        elif ch == "-" and i in (0, len(alias) - 1):
            out.append(r"[\-‐-―]")  # leading/trailing hyphen is literal ('-Allianz' = compound suffix)
        elif ch == "-":
            out.append(_HYPHEN + "?")
        elif ch == "&":
            out.append(r"\s*(?:&|\+|and|und)\s*")
        elif ch in "'’":
            out.append(r"['’`´]?")
        elif ch == ".":
            out.append(r"\.?\s?")
        elif ch == "/":
            out.append(r"\s?/\s?")
        else:
            out.append(_ACCENTS.get(ch) or re.escape(ch))
    return "".join(out)


def _bounded(aliases: Iterable[str], flags: int) -> re.Pattern[str] | None:
    """Word-bounded alternation of the aliases with an optional genitive ('Nvidias', "Apple's")."""
    alternatives = sorted({_alias_rx(alias) for alias in aliases}, key=len, reverse=True)
    if not alternatives:
        return None
    return re.compile(r"(?<![\w&])(?:" + "|".join(alternatives) + r")(?:['’]s|s)?(?![\w&])", flags)


def _phrases(phrases: Iterable[str]) -> re.Pattern[str] | None:
    """Veto / require / forbid lists: case-insensitive, left word boundary, PREFIX semantics on the right."""
    parts = [
        (r"(?<!\w)" if phrase[:1].isalnum() else "") + _alias_rx(phrase)
        for phrase in sorted(set(phrases), key=len, reverse=True)
    ]
    return re.compile("|".join(parts), re.I) if parts else None


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True, slots=True)
class _Context:
    pattern: re.Pattern[str]
    require: re.Pattern[str] | None
    forbid: re.Pattern[str] | None
    amount_guard: bool


@dataclass(frozen=True, slots=True)
class _Compiled:
    """The compiled rules of one symbol (or of one macro topic)."""

    key: str
    patterns: tuple[tuple[re.Pattern[str], MentionKind], ...]
    contexts: tuple[_Context, ...]
    vetoes: re.Pattern[str] | None
    broker: bool
    # Every match contains the longest ASCII-alphanumeric run of one of its aliases (those characters are
    # matched literally), so a substring test on the lower-cased text skips about 95 % of the symbols.
    triggers: tuple[str, ...]


def _compile(key: str, rules: SymbolRules) -> _Compiled:
    patterns: list[tuple[re.Pattern[str], MentionKind]] = []
    for pattern, kind in (
        (_bounded(rules.names, re.I), "name"),
        (_bounded(rules.names_cs, 0), "name_cs"),
    ):
        if pattern is not None:
            patterns.append((pattern, kind))
    if rules.names_prefix:
        alternatives = sorted({_alias_rx(alias) for alias in rules.names_prefix}, key=len, reverse=True)
        patterns.append((re.compile(r"(?<![\w&])(?:" + "|".join(alternatives) + r")\w*", re.I), "name"))
    if rules.tickers:
        alternatives = sorted({re.escape(ticker) for ticker in rules.tickers}, key=len, reverse=True)
        patterns.append(
            (re.compile(r"(?<![\w.$/^-])(?:" + "|".join(alternatives) + r")(?!\w|[.\-/][A-Za-z0-9])"), "ticker")
        )
    in_context = _unique((*rules.tickers_ctx, *rules.tickers))
    if in_context:
        alternatives = "(?:" + "|".join(sorted({re.escape(t) for t in in_context}, key=len, reverse=True)) + ")"
        pattern = r"(?:\$" + alternatives + "|" + _EXCHANGES + r"\s?:\s?" + alternatives + r")(?!\w)"
        patterns.append((re.compile(pattern), "ticker_ctx"))
    contexts = tuple(
        _Context(
            pattern=re.compile(
                r"(?<![\w&])(?:" + _alias_rx(c.alias) + r")(?:['’]s|s)?(?![\w&])", 0 if c.case_sensitive else re.I
            ),
            require=_phrases(c.require),
            forbid=_phrases(c.forbid),
            amount_guard=c.amount_guard,
        )
        for c in rules.names_ctx
    )
    aliases = (
        *rules.names, *rules.names_cs, *rules.names_prefix, *rules.tickers, *rules.tickers_ctx,
        *(c.alias for c in rules.names_ctx),
    )
    triggers: set[str] = set()
    for alias in aliases:
        runs = _ALNUM_RUN.findall(alias)
        triggers.add(max(runs, key=len).lower() if runs else "")
    if "" in triggers:
        triggers = {""}  # an alias without an ASCII run: never skip this symbol
    return _Compiled(key, tuple(patterns), contexts, _phrases(rules.vetoes), rules.broker, tuple(sorted(triggers)))


def _scan(rule: _Compiled, text: str, low: str, full: str) -> list[tuple[int, int, MentionKind, MentionRole]]:
    """All non-vetoed matches of one symbol in ``text`` as (start, end, kind, role).

    ``low`` is ``text.lower()`` (for the trigger pre-filter), ``full`` is title + summary (for the context
    requirements of ``names_ctx``).
    """
    if not any(trigger in low for trigger in rule.triggers):
        return []
    found: list[tuple[int, int, MentionKind]] = [
        (m.start(), m.end(), kind) for pattern, kind in rule.patterns for m in pattern.finditer(text)
    ]
    for context in rule.contexts:
        spans = [
            m.span()
            for m in context.pattern.finditer(text)
            if not (context.amount_guard and _AMOUNT_BEFORE.search(text[max(0, m.start() - 28) : m.start()]))
        ]
        # The alias is checked first: the require/forbid alternations are the expensive part.
        if not spans:
            continue
        if context.require is not None and not context.require.search(full):
            continue
        if context.forbid is not None and context.forbid.search(full):
            continue
        found += [(start, end, "name_ctx") for start, end in spans]
    if not found:
        return []
    vetoes = [(m.start(), m.end()) for m in rule.vetoes.finditer(text)] if rule.vetoes else []
    hits: list[tuple[int, int, MentionKind, MentionRole]] = []
    for start, end, kind in found:
        if any(start < veto_end and veto_start < end for veto_start, veto_end in vetoes):
            continue
        hits.append((start, end, kind, _role(rule, text, start, end)))
    return hits


def _role(rule: _Compiled, text: str, start: int, end: int) -> MentionRole:
    """'broker' for a bank named as analyst/broker (see module docstring), else 'subject'."""
    if not rule.broker or _RATED_OBJECT_BEFORE.search(text[max(0, start - 45) : start]):
        return "subject"
    if (
        _BROKER_AFTER.search(text[end : end + 100])
        or _BROKER_BEFORE.search(text[max(0, start - 40) : start])
        or (not text[end:].strip(" .)\"'”’") and _BROKER_TRAIL_BEFORE.search(text[:start]))
    ):
        return "broker"
    return "subject"


def _normalise(text: str) -> str:
    """NFKC, straight apostrophes and no soft hyphens: the form all patterns are written for."""
    return unicodedata.normalize("NFKC", text or "").replace("’", "'").replace("‘", "'").replace("­", "")


# ------------------------------------------------------------------------------------------------
# Rules of one instrument
# ------------------------------------------------------------------------------------------------
def _core_name(name: str) -> str:
    """The name without legal-form suffixes: 'Alibaba Group Holding Limited' -> 'Alibaba'."""
    return _LEGAL_SUFFIX.sub("", name).strip(" ,")


def _generic_tickers(instrument: Instrument) -> tuple[list[str], list[str]]:
    """Tickers derived from the symbol as (plain, context-only).

    Index symbols ('^GDAXI') never appear in text. Every other symbol counts as written ('SAP.DE', 'ES=F');
    for equities the exchange-local base ('SAP' of 'SAP.DE') counts in ticker context, and a US ticker that is
    short or an ordinary word ('V', 'COST') only in ticker context.
    """
    symbol = instrument.symbol
    if symbol.startswith("^"):
        return [], []
    if instrument.asset_class != "equity":
        return [symbol], []
    base, dot, _ = symbol.partition(".")
    if dot:
        return [symbol], [base]
    if len(symbol) <= 2 or symbol in AMBIGUOUS_TICKERS:
        return [], [symbol]
    if "-" in symbol:  # share classes: 'BRK-B' is written 'BRK.B' or 'BRK/B' as well
        return [symbol, symbol.replace("-", "."), symbol.replace("-", "/")], []
    return [symbol], []


def _case_variants(word: str) -> list[str]:
    """Case-sensitive spellings of a single-word alias: as given, upper case, and 'Nvidia' for 'NVIDIA'."""
    variants = [word, word.upper()]
    if word.isupper() and len(word) > 4:
        variants.append(word.capitalize())
    return variants


def merge_rules(instrument: Instrument, curated: SymbolRules | None) -> SymbolRules:
    """The rules of one instrument: its curated rules extended by its symbol, name and ``aliases``.

    Curated rules win for every alias they mention (an alias that the curated rules only accept in context,
    such as 'Visa', stays context-only even if the instrument lists it plainly). Other aliases are classified
    by shape: up to three characters -> exact spelling; one alphabetic word -> case-sensitive (as given and in
    upper case); anything else (several words, digits, hyphens, '&', '/') -> case-insensitive.
    """
    base = curated or SymbolRules()
    plain, in_context = _generic_tickers(instrument)
    tickers_ctx = _unique((*base.tickers_ctx, *in_context))
    tickers = _unique(t for t in (*base.tickers, *plain) if t not in tickers_ctx)
    taken = {
        alias.casefold()
        for alias in (*base.names, *base.names_cs, *base.names_prefix, *(c.alias for c in base.names_ctx))
    }
    names, names_cs = list(base.names), list(base.names_cs)
    for alias in _unique(
        a.strip() for a in (instrument.name, _core_name(instrument.name), *instrument.aliases) if a != instrument.symbol
    ):
        if len(alias) < 2 or alias.casefold() in taken or alias in tickers_ctx or alias in AMBIGUOUS_TICKERS:
            continue
        taken.add(alias.casefold())
        if len(alias) <= _SHORT_ALIAS:
            names_cs.append(alias)
        elif _SINGLE_WORD.fullmatch(alias):
            names_cs.extend(_case_variants(alias))
        else:
            names.append(alias)
    return replace(base, names=tuple(names), names_cs=_unique(names_cs), tickers=tickers, tickers_ctx=tickers_ctx)


# ------------------------------------------------------------------------------------------------
# Matcher
# ------------------------------------------------------------------------------------------------
class AliasMatcher:
    """Maps news text to the active instruments of a book. Immutable after construction and thread-safe.

    Build a new matcher when the instruments change (construction compiles all patterns: some 10-400 ms for the
    default universe, depending on the regex cache); ``analyse`` costs about 0.25 ms for a headline and 0.7 ms
    for a headline with a 600-character summary (measured on 4,069 real items).
    """

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        curated: Mapping[str, SymbolRules] = CURATED_RULES,
        macro_terms: Sequence[MacroTerm] = MACRO_TERMS,
    ) -> None:
        active = [instrument for instrument in instruments if instrument.active]
        merged = {instrument.symbol: merge_rules(instrument, curated.get(instrument.symbol)) for instrument in active}
        self._rules = [_compile(symbol, rules) for symbol, rules in merged.items()]
        self._brokers = frozenset(symbol for symbol, rules in merged.items() if rules.broker)
        self._isins = {rules.isin: symbol for symbol, rules in merged.items() if rules.isin}
        self._hint_tickers: dict[str, str] = {}
        for instrument in active:
            if instrument.asset_class == "equity":
                self._hint_tickers.setdefault(instrument.symbol.upper(), instrument.symbol)
                if "." not in instrument.symbol:  # US share classes: hint 'BRK.B' for symbol 'BRK-B'
                    self._hint_tickers.setdefault(instrument.symbol.replace("-", ".").upper(), instrument.symbol)
        self._macro: list[tuple[_Compiled, tuple[str, ...]]] = []
        for term in macro_terms:
            targets = tuple(target for target in term.targets if target in merged)
            if targets:
                rules = SymbolRules(names=term.names, names_cs=term.names_cs, names_ctx=term.names_ctx,
                                    vetoes=term.vetoes)
                self._macro.append((_compile(term.topic, rules), targets))

    # -- instruments ----------------------------------------------------------------------------
    def mentions(self, title: str, summary: str = "") -> list[Mention]:
        """All instrument mentions in title and summary (title first, then by position), with their role."""
        title, summary = _normalise(title), _normalise(summary)
        full = f"{title} \n {summary}"
        found: list[Mention] = []
        for where, text in (("title", title), ("summary", summary)):
            if text:
                found += self._mentions_in(text, full, where)
        found.sort(key=lambda m: (m.where != "title", m.start))
        return found

    def match(
        self, text: str, summary: str = "", *, hint_tickers: Iterable[str] = (), hint_isins: Iterable[str] = ()
    ) -> list[str]:
        """Subject instruments of a news item, strongest first.

        Order of strength: ISIN hints and ticker hints delivered by the feed, then mentions in ``text`` (the
        title) and finally in ``summary``, each by position. Instruments that are only named as broker or
        analyst are left out.
        """
        ordered: dict[str, None] = {}
        for isin in hint_isins:
            symbol = self._isins.get(isin.strip().upper())
            if symbol is not None:
                ordered.setdefault(symbol)
        for ticker in hint_tickers:
            symbol = self._hint_tickers.get(ticker.strip().upper())
            if symbol is not None:
                ordered.setdefault(symbol)
        for mention in self.mentions(text, summary):
            if mention.role == "subject":
                ordered.setdefault(mention.symbol)
        return list(ordered)

    # -- macro ----------------------------------------------------------------------------------
    def topics(self, text: str) -> list[str]:
        """Macro topics found in ``text`` (e.g. 'FED', 'ECB', 'DAX'), in order of their first occurrence."""
        text = _normalise(text)
        low = text.lower()
        hits: list[tuple[int, str]] = []
        for rule, _ in self._macro:
            starts = [start for start, _, _, _ in _scan(rule, text, low, text)]
            if starts:
                hits.append((min(starts), rule.key))
        return [topic for _, topic in sorted(hits)]

    def macro(self, text: str) -> list[str]:
        """Active index symbols moved by the macro topics in ``text``, most relevant first."""
        targets = {rule.key: symbols for rule, symbols in self._macro}
        return list(_unique(symbol for topic in self.topics(text) for symbol in targets[topic]))

    def analyse(
        self, title: str, summary: str = "", *, hint_tickers: Iterable[str] = (), hint_isins: Iterable[str] = ()
    ) -> MatchResult:
        """Instruments (title + summary + feed hints) and macro topics (title only) of one news item.

        Macro terms are matched in the title only: a summary that mentions tariffs or the Fed in passing does
        not make a story macro news.
        """
        symbols = tuple(self.match(title, summary, hint_tickers=hint_tickers, hint_isins=hint_isins))
        symbols = symbols[:MAX_ALIAS_SYMBOLS]
        topics = tuple(self.topics(title))
        targets = {rule.key: symbols_ for rule, symbols_ in self._macro}
        macro = _unique(s for topic in topics for s in targets[topic] if s not in symbols)
        return MatchResult(symbols=symbols, macro=macro, topics=topics)

    # -- internals ------------------------------------------------------------------------------
    def _mentions_in(self, text: str, full: str, where: Literal["title", "summary"]) -> list[Mention]:
        low = text.lower()
        found = [
            Mention(rule.key, text[start:end], kind, where, start, end, role)
            for rule in self._rules
            for start, end, kind, role in _scan(rule, text, low, full)
        ]
        found += [
            Mention(self._isins[m.group(0)], m.group(0), "isin", where, m.start(), m.end())
            for m in _ISIN_IN_TEXT.finditer(text)
            if m.group(0) in self._isins
        ]
        # The longest match wins when the spans of DIFFERENT symbols overlap (Siemens Energy > Siemens).
        found.sort(key=lambda m: (m.start - m.end, m.start))
        kept: list[Mention] = []
        for mention in found:
            if not any(
                other.symbol != mention.symbol
                and other.start <= mention.start
                and mention.end <= other.end
                and other.end - other.start > mention.end - mention.start
                for other in kept
            ):
                kept.append(mention)
        # 'JPMorgan says bitcoin ...': a bank commenting on ANOTHER instrument is a broker mention.
        if len({m.symbol for m in kept}) > 1:
            kept = [
                replace(m, role="broker")
                if m.role == "subject" and m.symbol in self._brokers and _COMMENT_AFTER.search(text[m.end : m.end + 20])
                else m
                for m in kept
            ]
        return kept


# ------------------------------------------------------------------------------------------------
# Near-duplicate headlines across sources
# ------------------------------------------------------------------------------------------------
_STOP_WORDS: Final[frozenset[str]] = frozenset(
    """a an the and or of to in on for at by with from as is are was were be been it its this that these those
    after before over under into about amid says say said will would could should may might has have had not no new
    more than up down out off vs via per how why what when who which his her their our your you we they he she i
    der die das den dem des ein eine einen einem einer eines und oder von zu zum zur im in an am auf aus bei mit nach
    vor für fuer über ueber unter um durch gegen ohne bis als auch nicht kein keine ist sind war waren wird werden
    wurde hat haben sich es er sie wir ihr so wie was wer wo dass noch nur schon mehr sehr nun jetzt laut doch
    aber""".split()
)
_WIRE_PREFIX: Final = re.compile(
    r"^(?:(?:roundup|update|wdh|korr|eil|breaking(?: news)?|exclusive|exklusiv|analysis|analyse(?:-flash)?|factbox|"
    r"explainer|opinion|kommentar|aktie im fokus|aktien im fokus|original-research|eqs-[a-z]+|dgap-[a-z]+|ots|"
    r"pta-[a-z]+|irw-news|dpa-afx[a-z -]*|financialjuice|presse|press release|\(?\d?(?:st|nd|rd|th)? ?ld\)?|"
    r"\(lead\))\s*\d*\s*[:/|-]+\s*)+",
    re.I,
)
_TOKEN: Final = re.compile(r"[^\W_]+(?:[.,]\d+)?")


def title_tokens(title: str) -> list[str]:
    """Significant tokens of a headline in order: case-folded, wire prefixes ('ROUNDUP:', 'EQS-Adhoc:') and
    English/German stop words removed, numbers normalised ('60,000' -> '60000')."""
    text = _WIRE_PREFIX.sub("", unicodedata.normalize("NFKC", title or "").casefold())
    tokens: list[str] = []
    for word in _TOKEN.findall(text):
        if word in _STOP_WORDS or (len(word) < 2 and not word.isdigit()):
            continue
        tokens.append(word.replace(",", "").replace(".", "") if word[0].isdigit() else word)
    return tokens


@dataclass(frozen=True, slots=True)
class _Headline:
    tokens: frozenset[str]
    ordered: tuple[str, ...]
    lead: str  # first non-numeric token
    numbers: frozenset[str]
    ts: int
    ref: int  # news id of the cluster's original item
    group: frozenset[str]
    source: str


def _headline(title: str, ts: int, ref: int, group: frozenset[str], source: str) -> _Headline:
    ordered = title_tokens(title)
    tokens = frozenset(ordered)
    return _Headline(
        tokens=tokens,
        ordered=tuple(ordered),
        lead=next((w for w in ordered if not w[0].isdigit()), ""),
        numbers=frozenset(w for w in tokens if w[0].isdigit()),
        ts=ts,
        ref=ref,
        group=group,
        source=source,
    )


def _same_template(a: Sequence[str], b: Sequence[str], max_slot: int = 4) -> bool:
    """True if a and b share a prefix/suffix frame but differ in a short, completely different filler
    ("DZ BANK stuft VINCI auf 'Kaufen'" vs "DZ BANK stuft DEUTZ AG auf 'Kaufen'")."""
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    j = 0
    while j < len(a) - i and j < len(b) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    rest_a, rest_b = a[i : len(a) - j], b[i : len(b) - j]
    return (
        bool(rest_a) and bool(rest_b) and i + j >= 2 and len(rest_a) <= max_slot and len(rest_b) <= max_slot
        and not set(rest_a) & set(rest_b)
    )


class NearDupIndex:
    """Online clustering of headlines: a new headline is a duplicate of the best matching earlier one.

    Similarity = max(weighted Jaccard, 0.9 x weighted containment) of the token sets, every token weighted by
    its rarity in the index (1 / (1 + ln(1 + document frequency))): sharing 'Nvidia' on a day with hundreds of
    Nvidia headlines means little, sharing 'Mailchimp' means a lot. Guards keep template headlines apart:
    different number sets, a different lead word, the same frame with a different filler, disjoint symbol
    groups, and the same source more than ``same_source_gap_s`` apart (a recurring column).

    Memory is bounded: entries older than ``window_s`` are dropped by ``prune`` and the oldest entries are
    evicted beyond ``max_entries``. Not thread-safe; one owner thread.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.62,
        window_s: int = 48 * 3600,
        min_common: int = 3,
        same_source_gap_s: int = 6 * 3600,
        max_entries: int = 50_000,
    ) -> None:
        self._threshold = threshold
        self._window_s = window_s
        self._min_common = min_common
        self._same_source_gap_s = same_source_gap_s
        self._max_entries = max_entries
        self._entries: dict[int, _Headline] = {}  # insertion order = age order
        self._postings: dict[str, set[int]] = {}
        self._next_key = 0

    def __len__(self) -> int:
        return len(self._entries)

    def find(self, title: str, ts: int, *, group: frozenset[str] = frozenset(), source: str = "") -> int | None:
        """The ``ref`` of the earlier headline this one duplicates, or None. Does not change the index."""
        probe = _headline(title, ts, 0, group, source)
        if len(probe.tokens) < self._min_common:
            return None
        common: dict[int, int] = {}
        for token in probe.tokens:
            for key in self._postings.get(token, ()):
                common[key] = common.get(key, 0) + 1
        weights = {token: self._weight(token) for token in probe.tokens}
        weight_a = sum(weights.values())
        best_ref, best_sim = None, 0.0
        for key, shared in common.items():
            if shared < self._min_common:
                continue
            other = self._entries[key]
            gap = abs(ts - other.ts)
            if gap > self._window_s or (source and source == other.source and gap > self._same_source_gap_s):
                continue
            weight_common = sum(weights[token] for token in probe.tokens & other.tokens)
            weight_b = weight_common + sum(self._weight(token) for token in other.tokens - probe.tokens)
            jaccard = weight_common / (weight_a + weight_b - weight_common)
            containment = weight_common / min(weight_a, weight_b)
            similarity = max(jaccard, 0.9 * containment if min(len(probe.tokens), len(other.tokens)) >= 5 else 0.0)
            if similarity <= best_sim or similarity < self._threshold or self._guarded(probe, other, similarity):
                continue
            best_ref, best_sim = other.ref, similarity
        return best_ref

    def add(self, title: str, ts: int, ref: int, *, group: frozenset[str] = frozenset(), source: str = "") -> int:
        """Index a stored headline; ``ref`` is the news id of its cluster's original. Returns the entry key."""
        key = self._next_key
        self._next_key += 1
        entry = _headline(title, ts, ref, group, source)
        self._entries[key] = entry
        for token in entry.tokens:
            self._postings.setdefault(token, set()).add(key)
        while len(self._entries) > self._max_entries:
            self.discard(next(iter(self._entries)))
        return key

    def discard(self, key: int) -> None:
        """Remove one entry (no-op for an unknown key)."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        for token in entry.tokens:
            keys = self._postings.get(token)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._postings[token]

    def prune(self, now_ts: int) -> int:
        """Drop the entries older than the window; returns how many were dropped."""
        cutoff = now_ts - self._window_s
        expired = [key for key, entry in self._entries.items() if entry.ts < cutoff]
        for key in expired:
            self.discard(key)
        return len(expired)

    def _weight(self, token: str) -> float:
        return 1.0 / (1.0 + math.log1p(len(self._postings.get(token, ()))))

    @staticmethod
    def _guarded(a: _Headline, b: _Headline, similarity: float) -> bool:
        """True if one of the template guards says the two headlines are different events."""
        if a.numbers and b.numbers and not (a.numbers <= b.numbers or b.numbers <= a.numbers):
            return True
        if a.lead and b.lead and a.lead not in b.tokens and b.lead not in a.tokens:
            return True
        if a.group and b.group and not a.group & b.group:
            return True
        return similarity < 0.999 and _same_template(a.ordered, b.ordered)
