"""The two paper portfolios: 'ai' (traded by the signal engine only) and 'me' (traded by the user)."""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query

from trading_local.contracts import OrderRejected
from trading_local.dto import (
    EquityCurvesOut,
    OrderOut,
    OrderRequest,
    PortfolioDetailOut,
    PortfolioSummaryOut,
    PositionOut,
    PositionUpdateRequest,
    ResetRequest,
)
from trading_local.services import Services
from trading_local.web.deps import (
    MAX_ID,
    MAX_TS,
    ApiContext,
    Ctx,
    normalize_symbol,
    not_found,
    require_storable_numbers,
)

router = APIRouter(prefix="/api")

PORTFOLIO_KEYS = ("ai", "me")  # listing order: the AI portfolio first
MANUAL_PORTFOLIO = "me"

ItemId = Annotated[int, Path(ge=1, le=MAX_ID)]


def _require_key(key: str) -> str:
    if key not in PORTFOLIO_KEYS:
        raise not_found(f"Unbekanntes Portfolio: {key[:16]}")
    return key


def _require_manual(key: str) -> str:
    """Only the user's own portfolio accepts orders through the API."""
    if _require_key(key) != MANUAL_PORTFOLIO:
        raise HTTPException(
            status_code=403,
            detail="Das KI-Portfolio handelt nur die Signal-Engine; manuelle Orders sind nicht möglich.",
        )
    return key


async def _trade[T](ctx: ApiContext, action: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run a broker action; a rule violation becomes a 400 with the broker's German explanation."""
    try:
        return await ctx.run(action, *args, **kwargs)
    except OrderRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc) or "Order abgelehnt.") from None


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------
@router.get("/portfolios")
async def list_portfolios(ctx: Ctx) -> list[PortfolioSummaryOut]:
    return await ctx.run(_summaries, ctx.services)


@router.get("/portfolios/equity")
async def get_equity_curves(
    ctx: Ctx,
    since_ts: Annotated[int | None, Query(ge=0, le=MAX_TS)] = None,
    max_points: Annotated[int, Query(ge=2, le=5000)] = 600,
) -> EquityCurvesOut:
    return await ctx.run(ctx.services.broker.equity_curves, since_ts, max_points)


@router.get("/portfolios/{key}")
async def get_portfolio(ctx: Ctx, key: str) -> PortfolioDetailOut:
    return await ctx.run(ctx.services.broker.detail, _require_key(key))


@router.get("/portfolios/{key}/trades")
async def list_trades(ctx: Ctx, key: str, limit: Annotated[int, Query(ge=1, le=1000)] = 200) -> list[PositionOut]:
    return await ctx.run(ctx.services.broker.trades, _require_key(key), limit)


@router.get("/portfolios/{key}/orders")
async def list_orders(ctx: Ctx, key: str, limit: Annotated[int, Query(ge=1, le=1000)] = 200) -> list[OrderOut]:
    return await ctx.run(ctx.services.broker.orders, _require_key(key), limit)


# ------------------------------------------------------------------------------------------------
# Trading (user portfolio only)
# ------------------------------------------------------------------------------------------------
@router.post("/portfolios/reset")
async def reset_portfolios(ctx: Ctx, body: ResetRequest) -> list[PortfolioSummaryOut]:
    """Close everything and restart BOTH portfolios with the same capital (requires ``confirm``)."""
    require_storable_numbers(body)
    if not body.confirm:
        raise HTTPException(status_code=400, detail="Bitte das Zurücksetzen ausdrücklich bestätigen (confirm).")
    await _trade(ctx, ctx.services.broker.reset, body.start_capital)
    return await ctx.run(_summaries, ctx.services)


@router.post("/portfolios/{key}/orders", status_code=201)
async def place_order(ctx: Ctx, key: str, body: OrderRequest) -> OrderOut:
    portfolio_key = _require_manual(key)
    require_storable_numbers(body)
    request = body.model_copy(update={"symbol": normalize_symbol(body.symbol)})
    return await _trade(ctx, ctx.services.broker.place_order, portfolio_key, request)


@router.delete("/portfolios/{key}/orders/{order_id}")
async def cancel_order(ctx: Ctx, key: str, order_id: ItemId) -> OrderOut:
    return await _trade(ctx, ctx.services.broker.cancel_order, _require_manual(key), order_id)


@router.post("/portfolios/{key}/positions/{position_id}/close")
async def close_position(ctx: Ctx, key: str, position_id: ItemId) -> OrderOut:
    return await _trade(ctx, ctx.services.broker.close_position, _require_manual(key), position_id)


@router.patch("/portfolios/{key}/positions/{position_id}")
async def update_position(ctx: Ctx, key: str, position_id: ItemId, body: PositionUpdateRequest) -> PositionOut:
    portfolio_key = _require_manual(key)
    require_storable_numbers(body)
    return await _trade(
        ctx,
        ctx.services.broker.update_position,
        portfolio_key,
        position_id,
        stop_price=body.stop_price,
        target_price=body.target_price,
    )


def _summaries(services: Services) -> list[PortfolioSummaryOut]:
    return [services.broker.summary(key) for key in PORTFOLIO_KEYS]
