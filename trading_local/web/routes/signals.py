"""The signal journal (actionable and shadow signals) and its outcome statistics."""

from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query

from trading_local import presenters
from trading_local.db import repo_news, repo_signals
from trading_local.dto import SignalDetailOut, SignalOut, SignalStatsOut
from trading_local.models import SIGNAL_CLOSED_STATUSES, SIGNAL_OPEN_STATUSES, Quote, Signal
from trading_local.services import Services
from trading_local.web.deps import MAX_ID, MAX_SYMBOL_LENGTH, MAX_TS, Ctx, not_found, require_instrument
from trading_local.web.routes.news import feed_source_names

router = APIRouter(prefix="/api")

_STATUSES: dict[str, tuple[str, ...] | None] = {
    "open": SIGNAL_OPEN_STATUSES,
    "closed": SIGNAL_CLOSED_STATUSES,
    "all": None,
}
_ACTIONABLE: dict[str, bool | None] = {"true": True, "false": False, "all": None}


@router.get("/signals")
async def list_signals(
    ctx: Ctx,
    scope: Literal["open", "closed", "all"] = "all",
    actionable: Literal["true", "false", "all"] = "all",
    symbol: Annotated[str | None, Query(min_length=1, max_length=MAX_SYMBOL_LENGTH)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    before_id: Annotated[int | None, Query(ge=1, le=MAX_ID)] = None,
) -> list[SignalOut]:
    filter_symbol = None if symbol is None else require_instrument(ctx.services, symbol).symbol
    return await ctx.run(
        _signal_page, ctx.services, _STATUSES[scope], _ACTIONABLE[actionable], filter_symbol, limit, before_id
    )


@router.get("/signals/stats")
async def get_signal_stats(
    ctx: Ctx, since_ts: Annotated[int | None, Query(ge=0, le=MAX_TS)] = None
) -> SignalStatsOut:
    return await ctx.run(ctx.services.engine.stats, since_ts)


@router.get("/signals/{signal_id}")
async def get_signal(ctx: Ctx, signal_id: Annotated[int, Path(ge=1, le=MAX_ID)]) -> SignalDetailOut:
    detail = await ctx.run(_signal_detail, ctx.services, signal_id)
    if detail is None:
        raise not_found(f"Signal {signal_id} nicht gefunden.")
    return detail


def _signal_page(
    services: Services,
    statuses: tuple[str, ...] | None,
    actionable: bool | None,
    symbol: str | None,
    limit: int,
    before_id: int | None,
) -> list[SignalOut]:
    signals = repo_signals.list_signals(
        services.db, statuses=statuses, actionable=actionable, symbol=symbol, before_id=before_id, limit=limit
    )
    return [
        presenters.signal_out(signal, services.instruments.get(signal.symbol), quote=_live_quote(services, signal))
        for signal in signals
    ]


def _signal_detail(services: Services, signal_id: int) -> SignalDetailOut | None:
    signal = repo_signals.get_signal(services.db, signal_id)
    if signal is None:
        return None
    names = feed_source_names()
    news = [
        presenters.news_out(item, analysis, source_names=names)
        for item, analysis in filter(None, (repo_news.get_news(services.db, news_id) for news_id in signal.news_ids))
    ]
    return presenters.signal_detail_out(
        signal, services.instruments.get(signal.symbol), news=news, quote=_live_quote(services, signal)
    )


def _live_quote(services: Services, signal: Signal) -> Quote | None:
    """The latest quote for signals that are still open (closed signals show their realised result)."""
    return services.data.quote(signal.symbol) if signal.status in SIGNAL_OPEN_STATUSES else None
