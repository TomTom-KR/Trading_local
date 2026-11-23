"""AliasMatcher precision for the Kraken coins of the default universe (names, tickers, crypto context)."""

import json
from functools import cache

import pytest

from tests.news.helpers import FIXTURES
from trading_local.news.matcher import CURATED_RULES, AliasMatcher, merge_rules
from trading_local.universe import DEFAULT_INSTRUMENTS

KRAKEN = {i.symbol for i in DEFAULT_INSTRUMENTS if i.exchange == "KRAKEN"}


@cache
def matcher() -> AliasMatcher:
    """The real default universe, with the aliases stored in the instruments table."""
    return AliasMatcher(DEFAULT_INSTRUMENTS)


def coins(text: str, summary: str = "") -> set[str]:
    return set(matcher().match(text, summary)) & KRAKEN


POSITIVE = [
    ("Bitcoin rebounds above $60,000", {"BTC-EUR"}),
    ("Bitcoin-Kurs fällt unter 56.000 Dollar", {"BTC-EUR"}),
    ("BTC falls below $90,000 as ETF outflows mount", {"BTC-EUR"}),
    ("BTC/EUR auf Kraken: 55.600 Euro", {"BTC-EUR"}),
    ("Ethereum upgrade goes live", {"ETH-EUR"}),
    ("Ether falls 5% as crypto rout deepens", {"ETH-EUR"}),
    ("ETH derivatives reset and the next retail trade", {"ETH-EUR"}),
    ("Solana ETF approved", {"SOL-EUR"}),
    ("$SOL rallies as memecoin season returns", {"SOL-EUR"}),
    ("XRP surges after court ruling", {"XRP-EUR"}),
    ("Ripple wins case, XRP token jumps", {"XRP-EUR"}),
    ("Cardano founder plans new upgrade", {"ADA-EUR"}),
    ("Cardano's ADA token rallies 8%", {"ADA-EUR"}),
    ("Dogecoin rallies on Musk post", {"DOGE-EUR"}),
    ("DOGE token jumps 12% in crypto rally", {"DOGE-EUR"}),
    ("Avalanche (AVAX) jumps as network upgrade goes live", {"AVAX-EUR"}),
    ("AVAX staking yields climb", {"AVAX-EUR"}),
    ("Chainlink partners with SWIFT on tokenization", {"LINK-EUR"}),
    ("LINK token hits 3-month high", {"LINK-EUR"}),
    ("Polkadot parachain auction raises funds", {"DOT-EUR"}),
    ("DOT price surges as Polkadot staking grows", {"DOT-EUR"}),
    ("Litecoin halving approaches", {"LTC-EUR"}),
    ("LTC climbs as crypto market recovers", {"LTC-EUR"}),
    ("Bitcoin, Ether & Co: So schlagen sich die Kryptowährungen", {"BTC-EUR", "ETH-EUR"}),
    ("ETH profit-taking weighs on price as traders cut longs", {"ETH-EUR"}),  # not 'ETH professor'
    ("BTC-ECHO: Bitcoin fällt unter 60.000 Dollar", {"BTC-EUR"}),  # the coin itself still counts
]

NEGATIVE = [
    "ETH Zürich entwickelt neuen Quantenchip",
    "ETH Zurich researchers publish record study on stocks",
    "Ripple effect: tariffs hit retailers",
    "ADA compliance lawsuits rise across US retailers",
    "Ada Lovelace Day celebrates women in tech",
    "DOGE cuts federal workforce by 10%",
    "Musk's DOGE targets federal agencies",
    "Avalanche kills three skiers in the Alps",
    "Colorado Avalanche beat Oilers 4-2",
    "An avalanche of sell orders hit stocks",
    "Click the LINK below to read more",
    "click the link in bio",
    "Supply chain link breaks as port strike widens",
    "Polka dot dresses are back in fashion",
    "US DOT fines airline over refunds",
    "DOT proposes new highway safety rules",
    "LTC insurance costs rise for retirees",
    "SOL Meliá reports record hotel bookings",
    "AVAX",  # a bare ticker without any context
    # the university ETH Zürich in research and economy news (broad context words such as 'Rekord', 'Index')
    "ETH-Forscher stellen Rekord bei Solarzellen auf",
    "ETH-Studie: Schweizer Aktien schlagen den Index",
    "ETH-Ökonomen erwarten ein Plus beim BIP",
    "ETH professor wins award for record battery",
    "Forscher an der ETH entwickeln Rekord-Akku",
    "Solana Beach home prices hit record",
    "Chainlink fence prices rise",
    "Seattle Kraken beat the Avalanche 3-2 in NHL thriller",
    "EQS-Adhoc: Bitcoin Group SE: Abberufung des Vorstandsmitgliedes",
]


@pytest.mark.parametrize(("text", "expected"), POSITIVE, ids=[case[0][:40] for case in POSITIVE])
def test_coin_mentions_are_found(text: str, expected: set[str]) -> None:
    assert coins(text) == expected


@pytest.mark.parametrize("text", NEGATIVE, ids=[text[:40] for text in NEGATIVE])
def test_ordinary_words_and_other_abbreviations_are_not_coins(text: str) -> None:
    assert coins(text) == set()


def test_context_may_come_from_the_summary() -> None:
    assert coins("LINK jumps 15%", "The Chainlink token led the altcoin rally.") == {"LINK-EUR"}
    assert coins("LINK jumps 15%", "The rail link between the cities reopened.") == set()


def test_the_btc_echo_source_line_is_no_bitcoin_mention() -> None:
    """Every item of the BTC-ECHO feed ends with 'Source: BTC-ECHO BTC-ECHO' (recorded from the real feed): that
    line must not make each of its stories a Bitcoin story."""
    summary = (
        "Wer Geld verschiebt, könnte Banken überfordern. Im Krypto-Bereich läuft das längst. Source: BTC-ECHO BTC-ECHO"
    )
    assert coins("Fed-Gouverneur warnt vor KI-Agenten: Bankansturm ist in Minuten möglich", summary) == set()
    assert coins("Gold-Prognose bis Ende 2028: Klettert das Edelmetall auf 5.600 Dollar?", summary) == set()
    assert coins("Binance-Hacker bewegen Ether für 600 Millionen US-Dollar", summary) == {"ETH-EUR"}


def test_short_tickers_are_never_plain_names() -> None:
    """Instrument aliases like 'DOT' or 'LINK' must not become plain (context-free) names via merge_rules."""
    short = {"BTC", "XBT", "ETH", "Ether", "SOL", "ADA", "DOGE", "AVAX", "LINK", "DOT", "LTC", "Ripple", "Avalanche"}
    for instrument in DEFAULT_INSTRUMENTS:
        if instrument.symbol not in KRAKEN:
            continue
        rules = merge_rules(instrument, CURATED_RULES.get(instrument.symbol))
        assert not short & {*rules.names, *rules.names_cs, *rules.tickers}, instrument.symbol
        assert {alias.casefold() for alias in instrument.aliases} <= {
            a.casefold() for a in (*rules.names, *rules.names_cs, *rules.names_prefix, *(c.alias for c in rules.names_ctx))
        } | {"xrp"}, instrument.symbol


def test_labelled_real_headlines_without_crypto_get_no_coin() -> None:
    """None of the 439 hand-labelled real headlines that do not mention crypto matches a Kraken coin."""
    data = json.loads((FIXTURES / "matcher_labelled.json").read_text(encoding="utf-8"))
    crypto_words = ("bitcoin", "ether", "eth ", "solana", "xrp", "krypto", "crypto", "btc")
    titles = {pair["title"] for pair in data["pairs"]} | {missed["title"] for missed in data["missed"]}
    wrong = [t for t in titles if coins(t) and not any(word in t.lower() for word in crypto_words)]
    assert wrong == []
    assert len(titles) > 400
