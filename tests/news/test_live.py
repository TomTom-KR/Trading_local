"""Live smoke test against one real feed (needs the internet; deselected by default, run with -m network)."""

import time

import pytest

from trading_local.db.repo_news import FeedState
from trading_local.news.feeds import FEEDS_BY_ID
from trading_local.news.fetcher import fetch_feed, make_client


@pytest.mark.network
async def test_ecb_press_releases_parse() -> None:
    feed = FEEDS_BY_ID["ecb_press"]
    async with make_client() as client:
        result = await fetch_feed(client, feed, FeedState(feed.id), int(time.time()))
    assert result.error == "", result.error
    assert result.items, "the ECB feed always lists its recent press releases"
    assert all(item.url.startswith("https://") and item.title for item in result.items)
