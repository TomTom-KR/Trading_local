"""NearDupIndex: cross-source duplicates, template guards, window, pruning and bounded memory."""

from trading_local.news.matcher import NearDupIndex, title_tokens

T0 = 1_795_300_000
HOUR = 3600


def index_with(*titles: tuple[str, str], group: frozenset[str] = frozenset()) -> NearDupIndex:
    """An index holding the given (source, title) pairs as news ids 1, 2, ..."""
    index = NearDupIndex()
    for ref, (source, title) in enumerate(titles, start=1):
        index.add(title, T0, ref, group=group, source=source)
    return index


def test_title_tokens() -> None:
    assert title_tokens("ROUNDUP 2: Nvidia übertrifft die Erwartungen") == ["nvidia", "übertrifft", "erwartungen"]
    assert title_tokens("EQS-Adhoc: TAKKT AG: Prognose") == ["takkt", "ag", "prognose"]
    assert title_tokens("Bitcoin above 60,000 as ETF inflows rise") == ["bitcoin", "above", "60000", "etf", "inflows",
                                                                        "rise"]


def test_reworded_story_from_another_source_is_a_duplicate() -> None:
    index = index_with(("cnbc_top", "UN climate talks collapse as EU and others reject deal"))
    assert index.find("UN climate talks in Turkey collapse as nations reject deal", T0 + 600, source="taipei") == 1


def test_identical_headline_with_different_quotes_is_a_duplicate() -> None:
    index = index_with(("wo_konjunktur", "Dröge: Scheitern der Weltklimakonferenz 'fatales Ergebnis'"))
    assert index.find("Dröge: Scheitern der Weltklimakonferenz ’fatales Ergebnis’", T0 + 60, source="fn_aktien") == 1


def test_duplicates_point_to_the_cluster_original() -> None:
    index = NearDupIndex()
    index.add("Intuit to sell Mailchimp to Bending Spoons for $3.68 billion", T0, 7, source="cnbc")
    index.add("Bending Spoons buys Mailchimp for $3.68 billion from Intuit", T0 + 60, 7, source="techcrunch")
    assert index.find("Intuit agrees to sell Mailchimp to Bending Spoons for $3.68 billion", T0 + 120, source="x") == 7


def test_unrelated_headlines_are_not_duplicates() -> None:
    index = index_with(("a", "Nvidia beats estimates as data center sales soar"))
    assert index.find("Apple unveils new iPhone with satellite messaging", T0, source="b") is None
    assert index.find("Too short", T0, source="b") is None


def test_number_guard_keeps_template_headlines_apart() -> None:
    index = index_with(("a", "Quartalsberichte heute, 19.11.2026: Siemens, Infineon, Bayer"))
    assert index.find("Quartalsberichte heute, 20.11.2026: Siemens, Infineon, Bayer", T0, source="b") is None


def test_lead_word_guard() -> None:
    index = index_with(("et", "Tata Steel Share Price Highlights today live updates"))
    assert index.find("Power Grid Share Price Highlights today live updates", T0, source="et2") is None


def test_same_frame_with_different_filler_is_not_a_duplicate() -> None:
    index = index_with(("fn", "DZ BANK stuft VINCI auf 'Kaufen' nach Zahlen"))
    assert index.find("DZ BANK stuft DEUTZ AG auf 'Kaufen' nach Zahlen", T0, source="fn2") is None


def test_disjoint_symbol_groups_are_not_duplicates() -> None:
    index = NearDupIndex()
    index.add("Company raises full-year outlook after record orders", T0, 1, group=frozenset({"SIE.DE"}), source="a")
    title = "Company raises full-year outlook after record orders"
    assert index.find(title, T0, group=frozenset({"ENR.DE"}), source="b") is None
    assert index.find(title, T0, group=frozenset({"SIE.DE", "ENR.DE"}), source="b") == 1
    assert index.find(title, T0, source="b") == 1  # no symbols: the guard does not apply


def test_same_source_more_than_six_hours_apart_is_a_new_edition() -> None:
    index = index_with(("investing", "What are the main events for today in the markets"))
    assert index.find("What are the main events for today in the markets", T0 + 5 * HOUR, source="investing") == 1
    assert index.find("What are the main events for today in the markets", T0 + 7 * HOUR, source="investing") is None
    assert index.find("What are the main events for today in the markets", T0 + 7 * HOUR, source="other") == 1


def test_window_prune_and_bounded_memory() -> None:
    index = NearDupIndex(window_s=48 * HOUR, max_entries=3)
    title = "Delta flights delayed due to automated system issue"
    index.add(title, T0, 1, source="a")
    assert index.find(title, T0 + 49 * HOUR, source="b") is None  # outside the window
    assert index.prune(T0 + 49 * HOUR) == 1 and len(index) == 0
    for ref in range(1, 6):
        index.add(f"Story number {ref} about something else entirely", T0, ref, source="a")
    assert len(index) == 3  # the two oldest entries were evicted
    assert index.find("Story number 1 about something else entirely", T0, source="b") is None
    assert index.find("Story number 5 about something else entirely", T0, source="b") == 5


def test_discard_removes_an_entry() -> None:
    index = NearDupIndex()
    key = index.add("Siemens Energy raises outlook again after strong orders", T0, 9, source="a")
    index.discard(key)
    index.discard(key)  # unknown keys are ignored
    assert len(index) == 0
    assert index.find("Siemens Energy raises outlook again after strong orders", T0, source="b") is None
