"""News items with their LLM analyses."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Query

from trading_local import presenters
from trading_local.db import repo_news
from trading_local.dto import NewsOut
from trading_local.services import Services
from trading_local.web.deps import MAX_ID, MAX_SYMBOL_LENGTH, Ctx, require_instrument

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/news")
async def list_news(
    ctx: Ctx,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    before_id: Annotated[int | None, Query(ge=1, le=MAX_ID)] = None,
    symbol: Annotated[str | None, Query(min_length=1, max_length=MAX_SYMBOL_LENGTH)] = None,
    min_relevance: Annotated[int | None, Query(ge=1, le=5)] = None,
    analyzed_only: bool = False,
) -> list[NewsOut]:
    filter_symbol = None if symbol is None else require_instrument(ctx.services, symbol).symbol
    return await ctx.run(
        _news_page,
        ctx.services,
        limit=limit,
        before_id=before_id,
        symbol=filter_symbol,
        min_relevance=min_relevance,
        analyzed_only=analyzed_only,
    )


def _news_page(services: Services, **filters: Any) -> list[NewsOut]:
    names = feed_source_names()
    return [
        presenters.news_out(item, analysis, source_names=names)
        for item, analysis in repo_news.list_news(services.db, **filters)
    ]


def feed_source_names() -> dict[str, str]:
    """Feed id -> display name from the feed registry (``trading_local.news.feeds.FEEDS``, entries with
    ``id`` and ``name``). Optional: without a readable registry items are shown by their feed id."""
    try:
        from trading_local.news.feeds import FEEDS
    except ImportError:
        return {}
    except Exception:  # the registry module is built separately: a failing import must not break the news
        log.warning("the feed registry cannot be imported; showing feed ids", exc_info=True)
        return {}
    try:
        return {str(feed.id): str(feed.name) for feed in FEEDS}
    except (TypeError, AttributeError):
        log.warning("the feed registry has an unexpected shape; showing feed ids")
        return {}
