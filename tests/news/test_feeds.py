import re
from collections import Counter, defaultdict

from trading_local.news.feeds import FEEDS, FEEDS_BY_ID
from trading_local.news.service import HOST_BUDGET_PER_MINUTE


def test_registry_has_the_vetted_tier_1_and_tier_2_feeds() -> None:
    assert len(FEEDS) == 147
    assert len(FEEDS_BY_ID) == len(FEEDS)
    tiers = Counter(feed.tier for feed in FEEDS)
    assert tiers == {1: 50, 2: 97}
    assert all(feed.enabled for feed in FEEDS)


def test_feed_fields_are_valid() -> None:
    for feed in FEEDS:
        assert re.fullmatch(r"[a-z0-9_]+", feed.id), feed.id
        assert feed.url.startswith("https://"), feed.id
        assert feed.region in ("US", "EU", "ASIA", "GLOBAL")
        assert feed.category in ("markets", "macro", "companies", "press_release", "crypto")
        assert 1 <= feed.quality <= 5
        assert feed.language in ("en", "de", "fr", "ja")
        for pattern in (feed.title_strip, feed.drop_title_re):
            if pattern:
                re.compile(pattern)


def test_polling_is_minute_level_for_tier_1_and_slower_for_tier_2() -> None:
    for feed in FEEDS:
        if feed.tier == 1:
            assert 60 <= feed.poll_seconds <= 300, feed.id
        else:
            assert 120 <= feed.poll_seconds <= 1800, feed.id


def test_every_host_stays_under_five_requests_per_minute() -> None:
    per_host: defaultdict[str, float] = defaultdict(float)
    for feed in FEEDS:
        per_host[feed.host] += 60 / feed.poll_seconds
    assert max(per_host.values()) < 5
    assert HOST_BUDGET_PER_MINUTE < 5


def test_german_mandatory_disclosures_are_polled_fast() -> None:
    assert FEEDS_BY_ID["fn_adhoc"].poll_seconds == 60
    assert FEEDS_BY_ID["wo_adhoc"].poll_seconds == 60
    assert FEEDS_BY_ID["wo_dd"].poll_seconds == 600


def test_regions_and_topics_are_covered() -> None:
    regions = Counter(feed.region for feed in FEEDS)
    assert all(regions[region] >= 20 for region in ("US", "EU", "ASIA", "GLOBAL"))
    assert sum(feed.language == "de" for feed in FEEDS) >= 40
    categories = Counter(feed.category for feed in FEEDS)
    assert all(categories[category] >= 5 for category in ("markets", "macro", "companies", "press_release", "crypto"))


def test_tier_3_sources_are_not_registered() -> None:
    hosts = {feed.host for feed in FEEDS}
    assert not any(host.endswith(("news.google.com", "asia.nikkei.com")) for host in hosts)
    assert not any("cgi-bin/browse-edgar" in feed.url for feed in FEEDS)


def test_junk_filters_are_attached() -> None:
    assert re.search(FEEDS_BY_ID["wo_adhoc"].drop_title_re or "", "Der neue Online Broker - Smartbroker")
    assert re.search(FEEDS_BY_ID["gnw_public"].drop_title_re or "", "Rosen Law Firm Encourages ACME Inc. Investors")
    assert re.sub(FEEDS_BY_ID["financialjuice"].title_strip or "", "", "FinancialJuice: Oil jumps") == "Oil jumps"
