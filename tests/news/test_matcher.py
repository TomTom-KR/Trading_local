"""AliasMatcher: precision rules, German/English names, feed hints, macro mapping, generic instruments."""

import json
from dataclasses import replace
from functools import cache

import pytest

from trading_local.models import Instrument
from trading_local.news.matcher import MAX_ALIAS_SYMBOLS, AliasMatcher, merge_rules
from trading_local.testing import SAMPLE_INSTRUMENTS

from tests.news.helpers import FIXTURES, universe


@cache
def full() -> AliasMatcher:
    """Matcher over the whole default universe (109 instruments)."""
    return AliasMatcher(universe())


def subjects(text: str, summary: str = "") -> set[str]:
    return set(full().match(text, summary))


def brokers(text: str) -> set[str]:
    mentions = full().mentions(text)
    return {m.symbol for m in mentions if m.role == "broker"} - {m.symbol for m in mentions if m.role == "subject"}


PROBE_TOPICS = {"FED", "ECB", "BOJ", "BOE", "SNB", "PBOC", "BOK", "RBI", "RBA", "US_DATA", "EZ_DATA", "OPEC", "TRADE"}

# The news probe's constructed edge cases (text, subjects, brokers, macro topics or None = not checked).
PROBE_CASES = [
    # ambiguous tickers must not fire as plain words
    ("Trump tightens H-1B visa rules for tech workers", set(), set(), None),
    ("Visa-Vergabe für Fachkräfte soll schneller werden", set(), set(), None),
    ("Visa and Mastercard settle swipe-fee lawsuit", {"V", "MA"}, set(), None),
    ("Visa Q4 earnings beat as payments volume grows", {"V"}, set(), None),
    ("V for Vendetta sequel announced", set(), set(), None),
    ("Stocks to watch: $V, $MA and $COST", {"V", "MA", "COST"}, set(), None),
    ("Costco (NASDAQ: COST) raises membership fee", {"COST"}, set(), None),
    ("COST OF LIVING: Energy bills soar again", set(), set(), None),
    ("Ma Huateng's Tencent posts record profit", {"0700.HK"}, set(), None),
    ("MA lawmakers pass budget", set(), set(), None),
    ("HD video streaming boom lifts Netflix", {"NFLX"}, set(), None),
    ("Home Depot (NYSE: HD) beats estimates", {"HD"}, set(), None),
    ("PG&E shares slump after wildfire ruling", set(), set(), None),
    ("Procter & Gamble, le lessivier qui adoucit ses heurts", {"PG"}, set(), None),
    ("Air France-KLM cuts outlook as fuel costs rise", set(), set(), None),
    ("Airbus delivers 70 jets in October", {"AIR.DE"}, set(), None),
    ("San Francisco Fed's Daly sees two more hikes", set(), set(), {"FED"}),
    ("Santander (SAN) lifts profit target", set(), set(), None),
    ("Sanofi buys biotech for $2 bln", {"SAN.PA"}, set(), None),
    ("BASF streicht 2000 Stellen in Ludwigshafen", {"BAS.DE"}, set(), None),
    ("Google ads revenue jumps 12%", {"GOOGL"}, set(), None),
    ("Adidas hebt Prognose nach starkem Quartal", {"ADS.DE"}, set(), None),
    ("Arbeitslosenversicherung (ALV) wird 2027 teurer", set(), set(), None),
    ("Allianz hebt Gewinnprognose an", {"ALV.DE"}, set(), None),
    ("Nato-Allianz berät in Brüssel über Ukraine", set(), set(), None),
    ("Breite Allianz gegen Rechts ruft zu Demo auf", set(), set(), None),
    ("FC Bayern gewinnt in der Allianz Arena", set(), set(), None),
    ("Meta to cut 10% of staff in AI pivot", {"META"}, set(), None),
    ("Metas KI-Chef verlässt den Konzern", {"META"}, set(), None),
    ("Meta-Analyse zeigt: Kaffee schützt das Herz", set(), set(), None),
    ("A meta-analysis of 40 studies finds no effect", set(), set(), None),
    ("ORCL jumps 8% after earnings beat", {"ORCL"}, set(), None),
    ("The Oracle of Omaha sells Apple stake", {"AAPL"}, set(), None),
    ("An oracle problem haunts DeFi lending", set(), set(), None),
    # similar company names
    ("Siemens Energy hebt Prognose erneut an", {"ENR.DE"}, set(), None),
    ("Siemens Healthineers mit Rekordauftrag", set(), set(), None),
    ("Siemens streicht 6000 Stellen", {"SIE.DE"}, set(), None),
    ("Samsung SDI wins battery order from BMW", {"BMW.DE"}, set(), None),
    ("Samsung unveils Galaxy S27", {"005930.KS"}, set(), None),
    ("Telekom Austria wächst im Mobilfunk", set(), set(), None),
    ("Deutsche Telekom erhöht Dividende", {"DTE.DE"}, set(), None),
    ("Daimler Truck senkt Absatzprognose", set(), set(), None),
    ("Mercedes-Benz und BMW senken Prognose", {"MBG.DE", "BMW.DE"}, set(), None),
    ("VW-Chef Blume kündigt Sparprogramm an", {"VOW3.DE"}, set(), None),
    ("Bayer Leverkusen gewinnt 3:0 gegen Mainz", set(), set(), None),
    ("Roundup-Vergleich durch: Bayer fällt ein milliardenschwerer Stein vom Herzen", {"BAYN.DE"}, set(), None),
    ("La Roche-Posay launches new sunscreen", set(), set(), None),
    ("Roche erhält EU-Zulassung für Krebsmittel", {"ROG.SW", "ROP.SW"}, set(), None),
    ("LVMH: Schmuckunternehmer Paolo Bulgari gestorben", {"MC.PA"}, set(), None),
    ("Münchener Rück und Munich Re: Rekordgewinn", {"MUV2.DE"}, set(), None),
    ("Reliance on Chinese rare earths worries EU", set(), set(), None),
    ("Reliance Industries Q2 profit up 9%", {"RELIANCE.NS"}, set(), None),
    ("Novo Nordisk shares plunge after Wegovy trial", {"NOVO-B.CO"}, set(), None),
    ("TSMC sales jump 40% on AI demand", {"2330.TW"}, set(), None),
    # common words that are also company names
    ("Shell companies used to launder billions, report says", set(), set(), None),
    ("Shell raises dividend and buyback", {"SHEL.L"}, set(), None),
    ("Apple's iPhone sales beat estimates", {"AAPL"}, set(), None),
    ("Big Apple rents hit record high", set(), set(), None),
    ("Late frost hits apple harvest in South Tyrol", set(), set(), None),
    ("Amazon rainforest deforestation hits record", set(), set(), None),
    ("Amazon to cut 14,000 corporate jobs", {"AMZN"}, set(), None),
    ("Amazons Cloud-Sparte AWS wächst um 20 Prozent", {"AMZN"}, set(), None),
    ("Microsoft (NasdaqGS:MSFT) and Danaher announce AI partnership", {"MSFT"}, set(), None),
    ("AMD stock soars 140% in 2026, Nvidia lags", {"AMD", "NVDA"}, set(), None),
    ("Rheinmetall-Aktie -65%: Jetzt müssen 4,8 Milliarden € her!", {"RHM.DE"}, set(), None),
    ("ISIN DE0007164600: Kursziel angehoben", {"SAP.DE"}, set(), None),
    # broker role
    ("ANALYSE-FLASH: JPMorgan hebt SAP auf 'Overweight' - Ziel auf 275 Euro", {"SAP.DE"}, {"JPM"}, None),
    ("JPMorgan profit beats estimates as trading surges", {"JPM"}, set(), None),
    ("Analysts at Bank of America see S&P 500 at 7,000", {"^GSPC"}, {"BAC"}, None),
    ("Deutsche Bank Research senkt Kursziel für BASF", {"BAS.DE"}, {"DBK.DE"}, None),
    ("Deutsche Bank streicht 3500 Stellen", {"DBK.DE"}, set(), None),
    ("HSBC raises Tesla price target to $400", {"TSLA"}, {"HSBA.L"}, None),
    # indices / macro
    ("Dax fällt unter 25.000 Punkte", {"^GDAXI"}, set(), None),
    ("MDax legt gegen den Trend zu", set(), set(), None),
    ("DAX-Konzern SAP übertrifft Erwartungen", {"^GDAXI", "SAP.DE"}, set(), None),
    ("Nasdaq falls 5% as AI trade unwinds", {"^NDX"}, set(), None),
    ("Acme Corp (NASDAQ: ACME) announces public offering", set(), set(), None),
    ("Nasdaq-listed biotech prices IPO", set(), set(), None),
    ("S&P 500 futures slip ahead of Fed decision", {"ES=F"}, set(), {"FED"}),
    ("Wall Street: S&P 500 closes at record high", {"^GSPC"}, set(), None),
    ("Nikkei falls 3% as yen surges on BOJ hike bets", {"^N225", "USDJPY=X"}, set(), {"BOJ"}),
    ("Toyota to build battery plant in Kyushu, Nikkei reports", {"7203.T"}, set(), None),
    ("Hang Seng slides; Tencent, Alibaba lead losses", {"^HSI", "0700.HK", "9988.HK"}, set(), None),
    ("EZB lässt Leitzins unverändert", set(), set(), {"ECB"}),
    ("Fed up with fees: customers leave banks", set(), set(), set()),
    ("FedEx cuts forecast", set(), set(), set()),
    # commodities / fx / crypto
    ("Gold falls below $3,800 as Fed hike bets rise", {"GC=F"}, set(), {"FED"}),
    ("She won the Olympic gold medal in Milan", set(), set(), None),
    ("Brent crude jumps 4% after OPEC+ cut", {"BZ=F"}, set(), {"OPEC"}),
    ("Analyst Brent Thill raises Microsoft target", {"MSFT"}, set(), None),
    ("Oil prices slump on glut fears", {"CL=F"}, set(), None),
    ("Olive oil prices hit record in Spain", set(), set(), None),
    ("Bitcoin rebounds above $60,000", {"BTC-USD"}, set(), None),
    ("ETH Zürich entwickelt neuen Quantenchip", set(), set(), None),
    ("Ether falls 5% as crypto rout deepens", {"ETH-USD"}, set(), None),
    ("China's exports rise by 5 billion yuan", set(), set(), None),
    ("Yuan weakens past 7.3 per dollar as PBOC sets weak fixing", {"EURCNY=X"}, set(), {"PBOC"}),
    ("Der Eurokurs steigt über 1,10 Dollar", {"EURUSD=X"}, set(), None),
]


@pytest.mark.parametrize(("text", "want_subjects", "want_brokers", "want_topics"), PROBE_CASES,
                         ids=[case[0][:50] for case in PROBE_CASES])
def test_probe_constructed_cases(
    text: str, want_subjects: set[str], want_brokers: set[str], want_topics: set[str] | None
) -> None:
    assert subjects(text) == want_subjects
    assert brokers(text) == want_brokers
    if want_topics is not None:
        assert set(full().topics(text)) & PROBE_TOPICS == want_topics


# Positive and negative examples for every ambiguous ticker / name of the brief.
AMBIGUOUS = [
    ("V", "Visa raises dividend as payments volume grows", "Student visa applications fall 20%"),
    ("V", "Payment stocks: $V and $PYPL rally", "Plan V: the government's last resort"),
    ("MA", "Mastercard beats on cross-border payments", "MA candidates face a tougher job market"),
    ("HD", "Home Depot (NYSE: HD) raises outlook", "Netflix adds HD tier for emerging markets"),
    ("PG", "P&G raises prices on Pampers", "PG&E files new wildfire safety plan"),
    ("AIR.DE", "Airbus hebt Auslieferungsziel an", "AIR quality alerts issued for Delhi"),
    ("SAN.PA", "Sanofi kauft Biotech-Firma für 9 Milliarden", "SAN Diego port strike ends"),
    ("BAS.DE", "BASF senkt Prognose für 2026", "BAS-Spitzenverband warnt vor Kassendefizit"),
    ("ADS.DE", "Adidas übertrifft Erwartungen im Quartal", "ADS-B transponder data shows detour"),
    ("ALV.DE", "Allianz-Chef Bäte erwartet Rekordgewinn", "Allianz für Weltoffenheit startet Kampagne"),
    ("COST", "Costco (NASDAQ: COST) reports November sales", "COST of capital rises for small firms"),
    ("META", "Meta Platforms erhöht KI-Investitionen", "Meta-Studie: Homeoffice senkt Produktivität kaum"),
    ("ORCL", "Oracle-Aktie springt nach Cloud-Zahlen an", "Oracle of Omaha's annual letter due Saturday"),
]


@pytest.mark.parametrize(
    ("symbol", "positive", "negative"), AMBIGUOUS, ids=[f"{case[0]}-{i}" for i, case in enumerate(AMBIGUOUS)]
)
def test_ambiguous_tickers_and_names(symbol: str, positive: str, negative: str) -> None:
    assert symbol in subjects(positive)
    assert symbol not in subjects(negative)


def test_ambiguous_tickers_count_in_ticker_context() -> None:
    assert subjects("Movers: XETRA: BAS, ETR: ADS, $COST") >= {"BAS.DE", "ADS.DE", "COST"}
    assert subjects("Allianz (XETRA: ALV) and Airbus (EPA: AIR) gain") >= {"ALV.DE", "AIR.DE"}


def test_german_and_english_name_variants() -> None:
    assert subjects("Muenchener Rueck erhöht Dividende") == {"MUV2.DE"}
    assert subjects("Munich Re raises dividend") == {"MUV2.DE"}
    assert subjects("Nestle cuts outlook") == subjects("Nestlé senkt Ausblick") == {"NESN.SW"}
    assert subjects("Deutsche-Bank-Aktie legt zu") == {"DBK.DE"}
    assert subjects("Nvidias Umsatz verdoppelt sich") == {"NVDA"}
    assert subjects("iPhone-Hersteller meldet Rekordquartal") == {"AAPL"}


def test_labelled_real_headlines_regression() -> None:
    """452 hand-labelled real (headline, symbol) pairs of the news probe: 99.1 % precision, all errors fixed."""
    data = json.loads((FIXTURES / "matcher_labelled.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for pair in data["pairs"]:
        roles = {m.role for m in full().mentions(pair["title"]) if m.symbol == pair["symbol"]}
        role = "subject" if "subject" in roles else ("broker" if roles else None)
        expected = pair["role"] if pair["correct"] else None
        if role != expected:
            failures.append(f"{pair['symbol']}: {role} != {expected}: {pair['title']}")
    for missed in data["missed"]:
        if not set(missed["symbols"]) <= subjects(missed["title"]):
            failures.append(f"missed {missed['symbols']}: {missed['title']}")
    assert failures == []
    assert len(data["pairs"]) == 452


def test_feed_hints_beat_aliases_and_come_first() -> None:
    matcher = full()
    # an ISIN hint names SAP although the headline does not
    assert matcher.match("EQS-News: Quartalszahlen veröffentlicht", hint_isins=["DE0007164600"]) == ["SAP.DE"]
    assert matcher.match("Delta flights delayed; Nvidia unaffected", hint_tickers=["DAL", "AAPL"]) == ["AAPL", "NVDA"]
    assert matcher.match("Berkshire update", hint_tickers=["BRK.B"])[0] == "BRK-B"
    assert matcher.match("No company here", hint_tickers=["XYZ"], hint_isins=["US0000000000"]) == []


def test_order_is_title_by_position_then_summary() -> None:
    assert full().match("Siemens and SAP sign deal; BASF watches", "Analysts also mention Adidas.") == [
        "SIE.DE", "SAP.DE", "BAS.DE", "ADS.DE"
    ]


def test_analyse_caps_symbols_and_reads_macro_from_the_title_only() -> None:
    title = "Apple, Microsoft, Nvidia, Amazon, Alphabet, Meta, Tesla, Broadcom and Netflix lead Fed-day rally"
    result = full().analyse(title, "Tariffs were not discussed.")
    assert len(result.symbols) == MAX_ALIAS_SYMBOLS
    assert result.symbols[:3] == ("AAPL", "MSFT", "NVDA")
    assert "FED" in result.topics and "TRADE" not in result.topics
    assert result.hinted == [*result.symbols, *result.macro]
    assert set(result.macro).isdisjoint(result.symbols)


# ------------------------------------------------------------------------------------------------
# Macro terms
# ------------------------------------------------------------------------------------------------
MACRO = [
    ("Dax schließt mit Gewinnen", ["^GDAXI"]),
    ("S&P 500 erreicht Rekordhoch", ["^GSPC"]),
    ("Wall Street ends mixed ahead of holiday", ["^GSPC"]),
    ("Nasdaq tumbles as chip stocks slide", ["^NDX"]),
    ("Nikkei closes at record as yen weakens", ["^N225"]),
    ("Hang Seng rebounds on stimulus hopes", ["^HSI"]),
    ("Fed holds rates steady", ["^GSPC", "^NDX"]),
    ("FOMC minutes show split on cuts", ["^GSPC", "^NDX"]),
    ("US inflation cools to 2.6%", ["^GSPC", "^NDX"]),
    ("EZB senkt den Leitzins", ["^GDAXI", "^STOXX50E"]),
    ("ECB's Lagarde warns on growth", ["^GDAXI", "^STOXX50E"]),
    ("Bank of Japan raises rates to 1%", ["^N225"]),
    ("OPEC+ agrees surprise output cut", ["CL=F", "BZ=F"]),
]


@pytest.mark.parametrize(("text", "expected"), MACRO, ids=[m[0][:40] for m in MACRO])
def test_macro_terms_map_to_index_symbols(text: str, expected: list[str]) -> None:
    assert full().macro(text) == expected


def test_macro_negatives() -> None:
    for text in ("Nikkei Asia reports Toyota plans", "Nasdaq-listed startup files for IPO", "FedEx beats estimates",
                 "Hang Seng Bank raises mortgage rates", "MDax-Wert Hensoldt springt an"):
        assert full().macro(text) == [], text


def test_macro_returns_only_active_instruments_of_the_book() -> None:
    sample = AliasMatcher(SAMPLE_INSTRUMENTS)  # has ^GDAXI and ^GSPC, but no ^NDX / ^STOXX50E / ^N225
    assert sample.macro("EZB senkt den Leitzins") == ["^GDAXI"]
    assert sample.macro("Fed holds rates steady") == ["^GSPC"]
    assert sample.macro("Bank of Japan raises rates") == []
    assert sample.topics("Bank of Japan raises rates") == []  # topics without any active target are not compiled
    inactive = [replace(i, active=False) if i.symbol == "^GDAXI" else i for i in SAMPLE_INSTRUMENTS]
    assert AliasMatcher(inactive).macro("EZB senkt den Leitzins") == []
    assert AliasMatcher(inactive).match("Dax fällt deutlich") == []


# ------------------------------------------------------------------------------------------------
# Instruments without curated rules (added by the user)
# ------------------------------------------------------------------------------------------------
def test_generic_rules_from_name_aliases_and_symbol() -> None:
    acme = Instrument("ACME", "Acme Robotics Inc.", "equity", "US", "XNAS", "USD", aliases=("Acme Robotics",))
    ge = Instrument("GE", "GE Aerospace", "equity", "US", "XNYS", "USD")
    target = Instrument("TGT", "Target Corporation", "equity", "US", "XNYS", "USD")
    henkel = Instrument("HEN3.DE", "Henkel AG & Co. KGaA Vz", "equity", "EU", "XETR", "EUR")
    matcher = AliasMatcher([acme, ge, target, henkel])
    assert matcher.match("Acme Robotics wins Navy contract") == ["ACME"]
    assert matcher.match("ACME jumps 12% premarket") == ["ACME"]
    assert matcher.match("The acme of engineering") == []
    assert matcher.match("GE Aerospace raises guidance") == ["GE"]
    assert matcher.match("GE: what to expect") == []  # two-letter ticker only in context
    assert matcher.match("Engines: NYSE: GE gains") == ["GE"]
    assert matcher.match("Target cuts holiday forecast") == ["TGT"]
    assert matcher.match("Price target raised for chip stocks") == []  # single-word name is case-sensitive
    assert matcher.match("Henkel senkt Prognose; HEN3.DE fällt") == ["HEN3.DE"]


def test_curated_rules_win_over_plain_instrument_aliases() -> None:
    visa = Instrument("V", "Visa Inc.", "equity", "US", "XNYS", "USD", aliases=("Visa", "V"))
    rules = merge_rules(visa, None)
    assert "V" in rules.tickers_ctx and "V" not in rules.tickers
    matcher = AliasMatcher([visa])
    assert matcher.match("Visa earnings beat as payments grow") == ["V"]
    assert matcher.match("Visa rules for students tightened") == []  # 'Visa' stays context-only
    assert matcher.match("V shaped recovery expected") == []


def test_symbol_aliases_of_the_sample_universe() -> None:
    sample = AliasMatcher(SAMPLE_INSTRUMENTS)
    assert sample.match("SAP und Shell im Fokus; Bitcoin steigt") == ["SAP.DE", "SHEL.L", "BTC-USD"]
    assert sample.match("sap flows in maple trees") == []
    assert sample.match("Toyota recalls 1 million cars") == ["7203.T"]
