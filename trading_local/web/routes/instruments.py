"""Watchlist: list, add and switch instruments."""

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime

from fastapi import APIRouter, HTTPException

from trading_local import presenters
from trading_local.contracts import MarketCalendar
from trading_local.db import repo_market, repo_signals
from trading_local.dto import InstrumentCreate, InstrumentOut, InstrumentUpdate
from trading_local.models import SIGNAL_OPEN_STATUSES, Instrument, SessionState, Signal
from trading_local.services import Services
from trading_local.sim import sizing
from trading_local.web.deps import MAX_SYMBOL_LENGTH, Ctx, normalize_symbol, not_found

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_OPEN_SIGNAL_SCAN = 1000  # upper bound of open actionable signals read for the watchlist (one query)


@router.get("/instruments")
async def list_instruments(ctx: Ctx, include_inactive: bool = False) -> list[InstrumentOut]:
    instruments = ctx.services.instruments.all(active_only=not include_inactive)
    return await ctx.run(instrument_outs, ctx.services, instruments)


@router.post("/instruments", status_code=201)
async def add_instrument(ctx: Ctx, body: InstrumentCreate) -> InstrumentOut:
    """Validate the symbol at the data source, store it and start its history download in the background."""
    services = ctx.services
    symbol = normalize_symbol(body.symbol)
    if not symbol:
        raise HTTPException(status_code=422, detail="Ungültige Eingabe: Feld 'symbol' ist leer.")
    await ctx.run(_require_new, services, symbol)
    try:
        instrument = await services.data.lookup(symbol)
    except Exception:
        log.exception("lookup of %s at the data source failed", symbol)
        raise HTTPException(
            status_code=502, detail="Die Datenquelle ist nicht erreichbar. Bitte später erneut versuchen."
        ) from None
    if instrument is None:
        raise not_found("Symbol bei der Datenquelle nicht gefunden")
    if instrument.symbol != symbol:  # the data source may know the symbol under another spelling
        await ctx.run(_require_new, services, instrument.symbol)
    out = await ctx.run(_store_new_instrument, services, instrument)
    ctx.spawn(services.data.ensure_history(instrument.symbol), name=f"ensure-history-{instrument.symbol}")
    return out


@router.patch("/instruments/{symbol}")
async def update_instrument(ctx: Ctx, symbol: str, body: InstrumentUpdate) -> InstrumentOut:
    """Switch ``active`` / ``tradable``. 400 for ``tradable: true`` on an asset class the simulation cannot
    trade (FX conversion pairs, futures): the engine would otherwise mail recommendations nobody can follow."""
    out = await ctx.run(_update_flags, ctx.services, normalize_symbol(symbol), body)
    if out is None:
        raise not_found(f"Unbekanntes Symbol: {normalize_symbol(symbol)[:MAX_SYMBOL_LENGTH]}")
    return out


# ------------------------------------------------------------------------------------------------
# Blocking parts (API executor)
# ------------------------------------------------------------------------------------------------
def instrument_outs(services: Services, instruments: Sequence[Instrument]) -> list[InstrumentOut]:
    """Watchlist entries with quote, session phase and the newest open actionable signal (one signal query)."""
    quotes = services.data.quotes()
    signals = _newest_open_signals(repo_signals.list_signals(
        services.db, statuses=SIGNAL_OPEN_STATUSES, actionable=True, limit=_OPEN_SIGNAL_SCAN
    ))
    sessions = _SessionLookup(services.calendar, services.clock.now())
    return [
        presenters.instrument_out(
            instrument,
            quote=quotes.get(instrument.symbol),
            state=sessions.state(instrument.exchange),
            open_signal=signals.get(instrument.symbol),
            exchange_name=sessions.name(instrument.exchange),
        )
        for instrument in instruments
    ]


def _require_new(services: Services, symbol: str) -> None:
    """409 if the symbol is already stored (active or not)."""
    if repo_market.get_instrument(services.db, symbol) is not None:
        raise HTTPException(status_code=409, detail=f"Das Symbol {symbol} ist bereits in der Liste.")


def _store_new_instrument(services: Services, instrument: Instrument) -> InstrumentOut:
    repo_market.upsert_instruments(services.db, [instrument], keep_user_flags=True)
    services.instruments.reload()
    return instrument_outs(services, [services.instruments.require(instrument.symbol)])[0]


def _update_flags(services: Services, symbol: str, body: InstrumentUpdate) -> InstrumentOut | None:
    current = repo_market.get_instrument(services.db, symbol)
    if current is None:
        return None
    if body.tradable and not sizing.is_tradable_class(current):
        raise HTTPException(status_code=400, detail=not_tradable_detail(current))
    instrument = repo_market.set_instrument_flags(services.db, symbol, active=body.active, tradable=body.tradable)
    if instrument is None:
        return None
    services.instruments.reload()
    return instrument_outs(services, [instrument])[0]


def not_tradable_detail(instrument: Instrument) -> str:
    """German reason why an instrument of a non-tradable asset class cannot be switched to tradable."""
    role = "ein Umrechnungskurs" if instrument.asset_class == "fx" else "ein Vergleichswert"
    return (
        f"{instrument.symbol} ({instrument.name}) ist {role} und kann im Papierdepot nicht gehandelt werden; "
        "„Handelbar“ lässt sich deshalb nicht einschalten."
    )


def _newest_open_signals(signals: Iterable[Signal]) -> dict[str, Signal]:
    """The first signal per symbol of a newest-first list."""
    newest: dict[str, Signal] = {}
    for signal in signals:
        newest.setdefault(signal.symbol, signal)
    return newest


class _SessionLookup:
    """Session state and display name per venue, computed once per request."""

    def __init__(self, calendar: MarketCalendar, now: datetime) -> None:
        self._calendar = calendar
        self._now = now
        self._states: dict[str, SessionState | None] = {}

    def state(self, exchange: str) -> SessionState | None:
        if exchange not in self._states:
            try:
                self._states[exchange] = self._calendar.state(exchange, self._now)
            except KeyError:
                log.warning("instrument venue %r is unknown to the trading calendar", exchange)
                self._states[exchange] = None
        return self._states[exchange]

    def name(self, exchange: str) -> str:
        try:
            return self._calendar.display_name(exchange)
        except KeyError:
            return exchange
