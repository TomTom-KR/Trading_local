"""System pages: status, trading sessions, activity log, LLM calls, feeds and background jobs."""

from typing import Annotated, Literal

from fastapi import APIRouter, Query

from trading_local.db import repo_ops
from trading_local.dto import ActivityOut, FeedStateOut, JobStatusOut, LlmCallOut, SessionsOut, SystemStatusOut
from trading_local.services import Services
from trading_local.web.deps import MAX_ID, Ctx
from trading_local.web.status import compose_status

router = APIRouter(prefix="/api")


@router.get("/status")
async def get_status(ctx: Ctx) -> SystemStatusOut:
    return await ctx.run(compose_status, ctx.services)


@router.get("/sessions")
async def get_sessions(ctx: Ctx) -> SessionsOut:
    return await ctx.run(_sessions_overview, ctx.services)


def _sessions_overview(services: Services) -> SessionsOut:
    # Imported lazily: the markets module loads the exchange calendars (slow) and is built separately.
    from trading_local.markets.overview import sessions_overview

    return sessions_overview(services.calendar, services.instruments.all(), services.clock.now())


@router.get("/activity")
async def list_activity(
    ctx: Ctx,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    level: Literal["info", "warning", "error"] | None = None,
    category: Annotated[str | None, Query(min_length=1, max_length=32)] = None,
    before_id: Annotated[int | None, Query(ge=1, le=MAX_ID)] = None,
) -> list[ActivityOut]:
    return await ctx.run(
        repo_ops.list_activity, ctx.services.db, limit=limit, level=level, category=category, before_id=before_id
    )


@router.get("/system/llm-calls")
async def list_llm_calls(ctx: Ctx, limit: Annotated[int, Query(ge=1, le=1000)] = 100) -> list[LlmCallOut]:
    return await ctx.run(repo_ops.list_llm_calls, ctx.services.db, limit=limit)


@router.get("/system/feeds")
async def list_feeds(ctx: Ctx) -> list[FeedStateOut]:
    return await ctx.run(ctx.services.news.feeds)


@router.get("/system/jobs")
async def list_jobs(ctx: Ctx) -> list[JobStatusOut]:
    return await ctx.run(ctx.services.runtime.jobs)
