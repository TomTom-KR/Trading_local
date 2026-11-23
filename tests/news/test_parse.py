"""Parsing of real RSS/Atom samples (recorded 2026-11-22) and of constructed edge cases."""

from datetime import UTC, datetime

import pytest

from trading_local.news.feeds import FEEDS_BY_ID
from trading_local.news.fetcher import (
    SUMMARY_MAX_CHARS,
    FeedParseError,
    canonical_url,
    clean_text,
    item_uid,
    parse_feed,
    title_key,
)

from tests.news.helpers import NOW_TS, Entry, fixture_bytes, make_feed, rss


def parse(feed_id: str, *, final_url: str = "", content_type: str = "") -> tuple[list, dict]:
    return parse_feed(
        fixture_bytes(f"{feed_id}.xml"), FEEDS_BY_ID[feed_id], NOW_TS, content_type=content_type, final_url=final_url
    )


def utc(ts: int | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------------------------
# Real samples
# ------------------------------------------------------------------------------------------------
def test_german_rss_with_isin_hints_finanznachrichten() -> None:
    items, hints = parse("fn_adhoc", content_type="application/rss+xml; charset=utf-8")
    assert len(items) == 6
    first = items[0]
    assert first.title.startswith("EQS-Adhoc: CEOTRONICS AG: Negative Umsatzentwicklung")
    assert first.lang == "de"
    assert first.source == "fn_adhoc"
    assert first.fetched_ts == NOW_TS
    assert utc(first.published_ts) == "2026-11-20T10:03:00"
    assert first.url.startswith("https://www.finanznachrichten.de/nachrichten-2026-11/70187783-")
    assert first.summary.startswith("EQS-Ad-hoc: CEOTRONICS AG / Schlagwort(e): Prognose")
    assert [item.published_ts for item in items] == sorted(item.published_ts for item in items)  # oldest first
    assert hints[items[-1].uid].isins == ("LI1147158318",)
    assert len(hints) == 6 and all(not h.tickers for h in hints.values())


def test_german_rss_with_dc_date_and_advert_wallstreet_online() -> None:
    items, _ = parse("wo_adhoc")
    assert len(items) == 4  # the 'Der neue Online Broker - Smartbroker' advert is dropped
    assert not any("Smartbroker" in item.title for item in items)
    assert utc(items[0].published_ts) == "2026-11-20T18:26:03"  # <dc:date> with +01:00
    assert "Schlagwort(e)" in items[0].summary


def test_english_rss_without_zone_guid_and_summary_investing() -> None:
    items, _ = parse("investing_news")
    assert len(items) == 5
    nvidia = next(item for item in items if "Nvidia" in item.title)
    assert utc(nvidia.published_ts) == "2026-11-22T17:54:25"  # '2026-11-22 17:54:25' without zone = UTC
    assert nvidia.summary == ""
    assert nvidia.uid == item_uid("www.investing.com", "", nvidia.url, nvidia.title)


def test_atom_feed_bls() -> None:
    items, _ = parse("bls_cpi")
    assert [utc(item.published_ts) for item in items] == [
        "2026-09-11T11:50:40",
        "2026-10-14T11:50:31",
        "2026-11-10T12:55:47",
    ]
    assert items[-1].title == "CPI for all items increases 0.4% in October; shelter rises"
    assert items[-1].url == "https://www.bls.gov/news.release/archives/cpi_11102026.htm"
    assert items[-1].summary.startswith("In October, the Consumer Price Index")


def test_html_summaries_are_plain_text() -> None:
    for feed_id in ("tagesschau_wirtschaft", "cointelegraph", "ifo", "rbi_press"):
        items, _ = parse(feed_id)
        assert items, feed_id
        for item in items:
            assert "<" not in item.summary and "&lt;" not in item.summary and "&amp;" not in item.summary, feed_id
            assert len(item.summary) <= SUMMARY_MAX_CHARS
            assert "  " not in item.summary


def test_long_summary_is_truncated_at_a_word_boundary() -> None:
    (item,), _ = parse("rbi_press")
    assert len(item.summary) <= SUMMARY_MAX_CHARS
    assert item.summary.endswith("…")
    # the leading HTML table is removed as a block, the text after it remains
    assert item.summary.startswith("The market participants are required to place their bids")


def test_feed_default_zone_and_html_inside_pubdate() -> None:
    (rbi,), _ = parse("rbi_press")
    assert utc(rbi.published_ts) == "2026-11-20T14:15:00"  # 'Fri, 20 Nov 2026 19:45:00' in Asia/Kolkata
    ifo, _ = parse("ifo")
    assert utc(ifo[0].published_ts) == "2026-11-19T08:00:00"  # <time datetime="..."> inside <pubDate>


def test_title_decorations_are_stripped() -> None:
    juice, _ = parse("financialjuice")
    assert not any(item.title.startswith("FinancialJuice") for item in juice)
    titan, hints = parse("stocktitan")
    assert not any("Stock News" in item.title for item in titan)
    assert {h.tickers for h in hints.values()} >= {("WRD",), ("NEGG",)}  # ticker from the link /news/WRD/


def test_symbol_hints_globenewswire_seeking_alpha_nasdaq_trader() -> None:
    items, hints = parse("gnw_public")
    schneider = next(item for item in items if item.title.startswith("Schneider Electric"))
    assert hints[schneider.uid].tickers == ("SBGSY",)
    assert hints[schneider.uid].isins == ("US80687P1066",)
    items, hints = parse("sa_market_currents")
    delta = next(item for item in items if item.title.startswith("Delta flights"))
    assert hints[delta.uid].tickers == ("DAL",)
    assert "utm_source" in delta.url  # the link is kept as published ...
    assert delta.uid == item_uid("seekingalpha.com", "", canonical_url(delta.url), delta.title)  # ... the uid is not
    halts, hints = parse("nasdaq_halts")
    assert [item.title for item in halts] == ["IMCC", "AEI", "AUUD", "CYAB"]
    assert all(item.url == "" for item in halts)  # no link in the feed
    assert hints[halts[0].uid].tickers == ("IMCC",)


def test_numeric_guids_and_byte_order_mark() -> None:
    cnbc, _ = parse("cnbc_top")
    assert len(cnbc) == 4 and len({item.uid for item in cnbc}) == 4
    rbi, _ = parse("rbi_press")  # the document starts with a UTF-8 byte order mark
    assert rbi[0].title == "Conversion/Switch of State Government Securities"


def test_uid_and_title_key_are_stable_across_parses() -> None:
    first, _ = parse("fn_adhoc")
    again, _ = parse_feed(fixture_bytes("fn_adhoc.xml"), FEEDS_BY_ID["fn_adhoc"], NOW_TS + 3600)
    assert [(i.uid, i.title_key) for i in first] == [(i.uid, i.title_key) for i in again]
    assert all(len(item.uid) == 40 and len(item.title_key) == 40 for item in first)


def test_same_headline_from_two_sources_shares_the_title_key() -> None:
    fn, _ = parse("fn_adhoc")
    wo, _ = parse("wo_adhoc")
    takkt_fn = next(item for item in fn if item.title.startswith("EQS-Adhoc: TAKKT AG: TAKKT passt"))
    takkt_wo = next(item for item in wo if item.title.startswith("EQS-Adhoc: TAKKT AG: TAKKT passt"))
    assert takkt_fn.title_key == takkt_wo.title_key
    assert takkt_fn.uid != takkt_wo.uid


# ------------------------------------------------------------------------------------------------
# Constructed cases
# ------------------------------------------------------------------------------------------------
FEED = make_feed("t_parse", host="feeds.example.test")


def test_future_dates_are_clamped_and_missing_dates_are_none() -> None:
    body = rss([
        Entry("Tomorrow's edition", NOW_TS + 86_400, "https://example.test/a"),
        Entry("No date at all", None, "https://example.test/b"),
    ])
    items, _ = parse_feed(body, FEED, NOW_TS)
    by_title = {item.title: item for item in items}
    assert by_title["Tomorrow's edition"].published_ts == NOW_TS
    assert by_title["No date at all"].published_ts is None


def test_only_http_links_and_relative_links_are_resolved() -> None:
    body = rss([
        Entry("Script link", NOW_TS, "javascript:alert(1)"),
        Entry("Relative link", NOW_TS, "/news/relative"),
        Entry("Mail link", NOW_TS, "mailto:someone@example.test"),
    ])
    items, _ = parse_feed(body, FEED, NOW_TS, final_url="https://feeds.example.test/rss/t_parse.xml")
    urls = {item.title: item.url for item in items}
    assert urls == {"Script link": "", "Relative link": "https://feeds.example.test/news/relative", "Mail link": ""}


def test_entries_without_title_and_repeated_entries_are_skipped() -> None:
    body = rss([
        Entry("", NOW_TS, "https://example.test/empty"),
        Entry("Twice", NOW_TS, "https://example.test/twice", guid="g-1"),
        Entry("Twice", NOW_TS, "https://example.test/twice", guid="g-1"),
    ])
    items, _ = parse_feed(body, FEED, NOW_TS)
    assert [item.title for item in items] == ["Twice"]


def test_shared_link_falls_back_to_the_guid() -> None:
    body = rss([
        Entry("First", NOW_TS, "https://example.test/ticker/XYZ", guid="item-1"),
        Entry("Second", NOW_TS, "https://example.test/ticker/XYZ", guid="item-2"),
    ])
    items, _ = parse_feed(body, FEED, NOW_TS)
    assert len({item.uid for item in items}) == 2
    # feedparser resolves a relative guid against the document URL; the identity is that guid, not the link
    assert items[0].uid == item_uid(
        "feeds.example.test", "https://feeds.example.test/item-1", "https://example.test/ticker/XYZ", "First",
        link_unique=False,
    )


def test_tracking_parameters_do_not_change_the_uid() -> None:
    plain = item_uid("h", "", "https://www.example.test/story/1/", "T")
    tracked = item_uid("h", "", "http://example.test/story/1?utm_source=rss&utm_medium=feed&.tsrc=rss#top", "T")
    assert plain == tracked
    assert canonical_url("https://WWW.Example.test/a/?id=7&utm_campaign=x&ref=home") == "https://example.test/a?id=7"


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"<!DOCTYPE html><html><body>Please accept cookies</body></html>",
        b"\x00\x01\x02 not xml",
        b'{"items": []}',
    ],
    ids=["empty", "html", "binary", "json"],
)
def test_non_feed_documents_raise(body: bytes) -> None:
    with pytest.raises(FeedParseError):
        parse_feed(body, FEED, NOW_TS)


def test_truncated_xml_is_parsed_leniently() -> None:
    body = fixture_bytes("fn_adhoc.xml")
    items, _ = parse_feed(body[: len(body) // 2], FEEDS_BY_ID["fn_adhoc"], NOW_TS)
    assert 1 <= len(items) < 6


def test_title_key_normalisation() -> None:
    assert title_key("Nvidia beats estimates - Reuters") == title_key("NVIDIA beats estimates!")
    assert title_key("Nvidia beats estimates | The Wall Street Journal") == title_key("Nvidia beats estimates")
    assert title_key("Delta outage - flights grounded") != title_key("Delta outage - CEO apologizes")
    assert title_key("JPMorgan hebt SAP auf 'Overweight' - Ziel 275 Euro") != title_key(
        "JPMorgan hebt SAP auf 'Overweight' - Ziel 300 Euro"
    )
    assert title_key("Tesla - Q3") != title_key("Tesla")


def test_clean_text() -> None:
    assert clean_text("<p>Hello&nbsp;<b>world</b></p><script>x()</script> Read more") == "Hello world"
    assert clean_text("Price &amp;amp; volume &amp;#8217;s") == "Price & volume ’s"  # double-escaped
    assert clean_text("&lt;img src='x.png'&gt; Text") == "Text"
    assert clean_text("Ctrl\x00chars\x07 here") == "Ctrlchars here"
    long = clean_text("word " * 300)
    assert len(long) <= SUMMARY_MAX_CHARS and long.endswith("…")
    assert clean_text(None) == ""
