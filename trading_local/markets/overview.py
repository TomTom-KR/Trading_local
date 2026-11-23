"""Session overview: payload of GET /api/sessions and of the GUI's 24-hour session ribbon."""

from collections.abc import Collection, Sequence
from datetime import datetime, timedelta
from typing import Final

from trading_local.clock import to_ts
from trading_local.contracts import MarketCalendar
from trading_local.dto import ExchangeSessionOut, RegionSessionOut, SessionsOut, TimelineSegmentOut
from trading_local.markets.venues import OVERVIEW_REGIONS, REGION_LABELS_DE, venues_of_region
from trading_local.models import Instrument, Region
from trading_local.presenters import to_ts as optional_ts

TIMELINE_PAST: Final = timedelta(hours=6)
TIMELINE_FUTURE: Final = timedelta(hours=18)


def sessions_overview(calendar: MarketCalendar, instruments: Sequence[Instrument], now: datetime) -> SessionsOut:
    """Session state of the venues that carry active instruments, grouped ASIA, EU, US, plus their timeline.

    Each region lists its venues (in ``venues.VENUES`` order) with at least one ACTIVE instrument; a region
    without such a venue shows its first venue anyway. The timeline covers [now - 6 h, now + 18 h): all
    regular segments and pre/post windows of the listed venues, clipped to that window and ordered by
    region, then start. Pseudo venues (CRYPTO, KRAKEN, FX, CME - region GLOBAL) are not part of the overview.
    """
    active = {instrument.exchange for instrument in instruments if instrument.active}
    timeline_start, timeline_end = now - TIMELINE_PAST, now + TIMELINE_FUTURE
    regions: list[RegionSessionOut] = []
    timeline: list[TimelineSegmentOut] = []
    for region in OVERVIEW_REGIONS:
        keys = _listed_venues(region, active)
        exchanges = [_exchange_out(calendar, key, region, now) for key in keys]
        regions.append(
            RegionSessionOut(
                region=region,
                label=REGION_LABELS_DE[region],
                is_open=any(exchange.is_open for exchange in exchanges),
                exchanges=exchanges,
            )
        )
        segments = [
            segment for key in keys for segment in calendar.segments(key, timeline_start, timeline_end, extended=True)
        ]
        segments.sort(key=lambda segment: segment.start)  # stable: the venue order breaks ties
        timeline.extend(
            TimelineSegmentOut(
                exchange=segment.exchange,
                region=region,
                start=to_ts(segment.start),
                end=to_ts(segment.end),
                kind=segment.kind,
            )
            for segment in segments
        )
    return SessionsOut(
        now=to_ts(now),
        regions=regions,
        timeline_start=to_ts(timeline_start),
        timeline_end=to_ts(timeline_end),
        timeline=timeline,
    )


def _listed_venues(region: Region, active_exchanges: Collection[str]) -> list[str]:
    candidates = venues_of_region(region)
    return [key for key in candidates if key in active_exchanges] or [candidates[0]]


def _exchange_out(calendar: MarketCalendar, exchange: str, region: Region, now: datetime) -> ExchangeSessionOut:
    state = calendar.state(exchange, now)
    return ExchangeSessionOut(
        exchange=exchange,
        name=calendar.display_name(exchange),
        region=region,
        timezone=calendar.timezone(exchange),
        phase=state.phase,
        is_open=state.is_open,
        since=optional_ts(state.since),
        until=optional_ts(state.until),
        next_open=optional_ts(state.next_open),
        next_close=optional_ts(state.next_close),
        note=state.note,
    )
