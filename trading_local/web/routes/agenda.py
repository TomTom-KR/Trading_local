"""The agenda: macro data, central-bank decisions and earnings dates with expectation, actual and surprise."""

from typing import Annotated, Literal

from fastapi import APIRouter, Query

from trading_local.agenda import rules, texts
from trading_local.db import repo_agenda
from trading_local.db.repo_agenda import AgendaEvent
from trading_local.dto import AgendaEventOut
from trading_local.models import Instrument
from trading_local.services import Services
from trading_local.web.deps import MAX_SYMBOL_LENGTH, MAX_TS, Ctx, require_instrument

router = APIRouter(prefix="/api")

DEFAULT_BACK_S = 86_400
DEFAULT_AHEAD_S = 7 * 86_400
MAX_SPAN_S = 120 * 86_400


@router.get("/agenda")
async def list_agenda(
    ctx: Ctx,
    from_ts: Annotated[int | None, Query(ge=0, le=MAX_TS)] = None,
    to_ts: Annotated[int | None, Query(ge=0, le=MAX_TS)] = None,
    min_importance: Annotated[int, Query(ge=1, le=3)] = 1,
    kind: Literal["macro", "central_bank", "earnings"] | None = None,
    symbol: Annotated[str | None, Query(min_length=1, max_length=MAX_SYMBOL_LENGTH)] = None,
) -> list[AgendaEventOut]:
    """Events between ``from_ts`` (default: a day ago) and ``to_ts`` (default: in a week; at most 120 days),
    oldest first. ``symbol`` keeps the events that concern that instrument (its earnings, macro data of its
    currency)."""
    instrument = None if symbol is None else require_instrument(ctx.services, symbol)
    now_ts = ctx.services.clock.ts()
    start = now_ts - DEFAULT_BACK_S if from_ts is None else from_ts
    end = now_ts + DEFAULT_AHEAD_S if to_ts is None else to_ts
    end = min(max(end, start), start + MAX_SPAN_S)
    return await ctx.run(_agenda, ctx.services, start, end, min_importance, kind, instrument, now_ts)


def _agenda(
    services: Services,
    start_ts: int,
    end_ts: int,
    min_importance: int,
    kind: str | None,
    instrument: Instrument | None,
    now_ts: int,
) -> list[AgendaEventOut]:
    events = repo_agenda.events_between(
        services.db,
        start_ts=start_ts,
        end_ts=end_ts,
        min_importance=min_importance,
        kinds=None if kind is None else (kind,),
        symbol=None if instrument is None else instrument.symbol,
    )
    if instrument is not None:
        events = [event for event in events if rules.affects(event, instrument)]
    prefs = services.runtime_settings.get()
    tradable = services.instruments.tradable()
    names = {item.symbol: item.name for item in services.instruments.all(active_only=False)}
    return [
        _event_out(event, now_ts, prefs.event_block_enabled, prefs.event_block_before_min * 60, tradable, names)
        for event in events
        if event.id is not None
    ]


def _event_out(
    event: AgendaEvent,
    now_ts: int,
    block_enabled: bool,
    before_s: int,
    tradable: list[Instrument],
    names: dict[str, str],
) -> AgendaEventOut:
    blocking = (
        event.kind != "earnings" and event.time_known and event.importance >= rules.BLOCK_MIN_IMPORTANCE
    )
    block_from = event.event_ts - before_s if blocking else None
    return AgendaEventOut(
        id=event.id or 0,
        kind=event.kind,  # type: ignore[arg-type]
        kind_de=texts.KIND_LABELS_DE.get(event.kind, event.kind),
        source=event.source,
        currency=event.currency,
        symbol=event.symbol,
        name=names.get(event.symbol or "", ""),
        title=event.title,
        event_ts=event.event_ts,
        time_known=event.time_known,
        importance=event.importance,
        forecast_text=event.forecast_text,
        previous_text=event.previous_text,
        actual_text=event.actual_text,
        surprise=event.surprise,
        surprise_z=event.surprise_z,
        surprise_label=texts.surprise_label(event),
        result_de=texts.result_de(event),
        affected=sum(1 for instrument in tradable if rules.affects(event, instrument)),
        block_from_ts=block_from,
        blocks_entries=block_enabled and block_from is not None and block_from <= now_ts <= event.event_ts,
    )
