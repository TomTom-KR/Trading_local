"""sessions_overview: regions, listed venues, timeline window and clipping."""

from dataclasses import replace
from datetime import timedelta

from tests.markets.helpers import utc
from trading_local.clock import to_ts
from trading_local.dto import SessionsOut
from trading_local.markets import ExchangeCalendar, sessions_overview
from trading_local.models import Instrument
from trading_local.testing import SAMPLE_INSTRUMENTS, FakeCalendar

# Thursday 2026-11-19 22:00 UTC: timeline 16:00 UTC Thursday .. 16:00 UTC Friday.
NOW = utc(2026, 11, 19, 22)


def overview(calendar: ExchangeCalendar, instruments: list[Instrument] = SAMPLE_INSTRUMENTS) -> SessionsOut:
    return sessions_overview(calendar, instruments, NOW)


def test_regions_list_the_venues_of_active_instruments(calendar: ExchangeCalendar) -> None:
    result = overview(calendar)
    assert [(region.region, region.label) for region in result.regions] == [
        ("ASIA", "Asien"), ("EU", "Europa"), ("US", "USA")
    ]
    assert [[exchange.exchange for exchange in region.exchanges] for region in result.regions] == [
        ["XTKS"], ["XETR", "XLON"], ["XNYS", "XNAS"]
    ]
    xetra = result.regions[1].exchanges[0]
    assert (xetra.name, xetra.timezone, xetra.region) == ("Xetra", "Europe/Berlin", "EU")


def test_inactive_instruments_are_ignored_and_empty_regions_show_their_first_venue(calendar: ExchangeCalendar) -> None:
    instruments = [
        replace(instrument, active=instrument.exchange not in ("XTKS", "XLON")) for instrument in SAMPLE_INSTRUMENTS
    ] + [Instrument("0700.HK", "Tencent", "equity", "ASIA", "XHKG", "HKD", active=False)]
    result = overview(calendar, instruments)
    assert [[exchange.exchange for exchange in region.exchanges] for region in result.regions] == [
        ["XTKS"], ["XETR"], ["XNYS", "XNAS"]
    ]
    nothing = overview(calendar, [])
    assert [[exchange.exchange for exchange in region.exchanges] for region in nothing.regions] == [
        ["XTKS"], ["XETR"], ["XNYS"]
    ]


def test_region_open_flags_and_session_fields(calendar: ExchangeCalendar) -> None:
    friday = sessions_overview(calendar, SAMPLE_INSTRUMENTS, utc(2026, 11, 20, 15))
    assert [region.is_open for region in friday.regions] == [False, True, True]
    xnys = friday.regions[2].exchanges[0]
    assert (xnys.phase, xnys.is_open, xnys.note) == ("open", True, "Handel")
    assert (xnys.since, xnys.until, xnys.next_close) == (
        to_ts(utc(2026, 11, 20, 14, 30)), to_ts(utc(2026, 11, 20, 21)), to_ts(utc(2026, 11, 20, 21))
    )
    assert xnys.next_open == to_ts(utc(2026, 11, 23, 14, 30))
    late = overview(calendar)  # Thursday 22:00 UTC: New York in after-hours, Asia and Europe closed
    assert [region.is_open for region in late.regions] == [False, False, False]
    assert late.regions[2].exchanges[0].phase == "post"


def test_timeline_window_clipping_and_order(calendar: ExchangeCalendar) -> None:
    result = overview(calendar)
    assert result.now == to_ts(NOW)
    window = (to_ts(NOW - timedelta(hours=6)), to_ts(NOW + timedelta(hours=18)))
    assert (result.timeline_start, result.timeline_end) == window
    assert all(window[0] <= segment.start < segment.end <= window[1] for segment in result.timeline)
    regions = [segment.region for segment in result.timeline]
    assert regions == sorted(regions, key=["ASIA", "EU", "US"].index)
    for region in ("ASIA", "EU", "US"):
        starts = [segment.start for segment in result.timeline if segment.region == region]
        assert starts == sorted(starts)
    as_tuples = {(s.exchange, s.kind, s.start, s.end) for s in result.timeline}
    # New York's Thursday session is clipped at the window start, Xetra's Friday session at its end
    assert ("XNYS", "regular", to_ts(utc(2026, 11, 19, 16)), to_ts(utc(2026, 11, 19, 21))) in as_tuples
    assert ("XETR", "regular", to_ts(utc(2026, 11, 20, 8)), to_ts(utc(2026, 11, 20, 16))) in as_tuples
    assert ("XNYS", "post", to_ts(utc(2026, 11, 19, 21)), to_ts(utc(2026, 11, 20, 1))) in as_tuples
    assert ("XNAS", "pre", to_ts(utc(2026, 11, 20, 9)), to_ts(utc(2026, 11, 20, 14, 30))) in as_tuples
    assert ("XTKS", "regular", to_ts(utc(2026, 11, 20, 3, 30)), to_ts(utc(2026, 11, 20, 6, 30))) in as_tuples
    assert {s.exchange for s in result.timeline} == {"XTKS", "XETR", "XLON", "XNYS", "XNAS"}  # no pseudo venues


def test_overview_works_with_any_market_calendar() -> None:
    result = sessions_overview(FakeCalendar(), SAMPLE_INSTRUMENTS, NOW)
    assert [[exchange.exchange for exchange in region.exchanges] for region in result.regions] == [
        ["XTKS"], ["XETR", "XLON"], ["XNYS", "XNAS"]
    ]
    assert result.timeline


def test_kraken_instruments_never_appear_in_the_session_lanes(calendar: ExchangeCalendar) -> None:
    kraken = [
        Instrument(symbol, name, "crypto", "GLOBAL", "KRAKEN", "EUR", timezone="UTC")
        for symbol, name in (("BTC-EUR", "Bitcoin"), ("ETH-EUR", "Ethereum"))
    ]
    result = overview(calendar, [*SAMPLE_INSTRUMENTS, *kraken])
    listed = {exchange.exchange for region in result.regions for exchange in region.exchanges}
    assert "KRAKEN" not in listed and "CRYPTO" not in listed
    assert {segment.exchange for segment in result.timeline}.isdisjoint({"KRAKEN", "CRYPTO"})
    only_crypto = overview(calendar, kraken)  # still the three regions with their first venue
    assert [[exchange.exchange for exchange in region.exchanges] for region in only_crypto.regions] == [
        ["XTKS"], ["XETR"], ["XNYS"]
    ]
