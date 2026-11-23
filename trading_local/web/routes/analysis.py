"""Technical analysis of one instrument and the LLM chart analyses."""

from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel

from trading_local import presenters
from trading_local.db import repo_signals
from trading_local.dto import AnalysisOut, ChartAnalysisOut
from trading_local.services import Services
from trading_local.web.deps import Ctx, require_instrument

router = APIRouter(prefix="/api")


QUEUED_DE = "Chartanalyse angefordert – das Ergebnis erscheint hier, sobald die KI geantwortet hat."
LLM_DISABLED_DE = "Die KI ist ausgeschaltet (TL_ENABLE_LLM=false): Chartanalysen sind nicht möglich."
NOT_QUEUED_DE = (
    "Chartanalyse derzeit nicht möglich: Für diesen Wert liegen zu wenige Kursdaten vor oder es läuft bereits "
    "eine Analyse in einem anderen Zeitrahmen."
)


class ChartQueuedOut(BaseModel):
    queued: bool  # False: the engine did not accept the job (LLM disabled, too little data, another one running)
    detail_de: str = ""  # German explanation for the GUI


@router.get("/analysis/{symbol}")
async def get_analysis(ctx: Ctx, symbol: str) -> AnalysisOut:
    instrument = require_instrument(ctx.services, symbol)
    return await ctx.run(ctx.services.engine.analysis, instrument.symbol)


@router.post("/analysis/{symbol}/chart", status_code=202)
async def request_chart_analysis(ctx: Ctx, symbol: str) -> ChartQueuedOut:
    """Queue an LLM chart analysis. The result arrives later as a 'chart_analysis' WebSocket message - or never,
    when the model gives no usable answer in time (the GUI stops waiting after the engine's timeout)."""
    instrument = require_instrument(ctx.services, symbol)
    queued = await ctx.services.engine.request_chart_analysis(instrument.symbol)
    if queued:
        detail = QUEUED_DE
    elif not ctx.services.settings.enable_llm:
        detail = LLM_DISABLED_DE
    else:
        detail = NOT_QUEUED_DE
    return ChartQueuedOut(queued=queued, detail_de=detail)


@router.get("/chart-analyses/{symbol}")
async def list_chart_analyses(
    ctx: Ctx, symbol: str, limit: Annotated[int, Query(ge=1, le=200)] = 20
) -> list[ChartAnalysisOut]:
    instrument = require_instrument(ctx.services, symbol)
    return await ctx.run(_chart_analyses, ctx.services, instrument.symbol, limit)


def _chart_analyses(services: Services, symbol: str, limit: int) -> list[ChartAnalysisOut]:
    return [
        presenters.chart_analysis_out(analysis)
        for analysis in repo_signals.list_chart_analyses(services.db, symbol, limit=limit)
    ]
