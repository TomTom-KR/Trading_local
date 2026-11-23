"""The user's runtime settings (editable in the GUI)."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import ValidationError

from trading_local.config import RuntimeSettings
from trading_local.web.deps import Ctx, validation_detail

router = APIRouter(prefix="/api")


@router.get("/settings")
async def get_settings(ctx: Ctx) -> RuntimeSettings:
    return await ctx.run(ctx.services.runtime_settings.get)


@router.put("/settings")
async def update_settings(ctx: Ctx, patch: Annotated[dict[str, Any], Body()]) -> RuntimeSettings:
    """Merge any subset of the fields over the stored settings; nothing is stored if one value is invalid."""
    unknown = sorted(set(patch) - set(RuntimeSettings.model_fields))
    if unknown:
        names = ", ".join(f"'{name[:40]}'" for name in unknown[:5])
        raise HTTPException(status_code=422, detail=f"Unbekannte Einstellung: {names}.")
    try:
        return await ctx.run(ctx.services.runtime_settings.update, patch, ctx.services.clock.ts())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=validation_detail(exc.errors())) from None
