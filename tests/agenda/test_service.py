"""AgendaService: the weekly calendar, actual values from headlines, earnings dates and the activity log."""

import json
from datetime import UTC, datetime

import httpx
import pytest

from trading_local.agenda import build_agenda, sources
from trading_local.agenda.service import AgendaService, release_message_de
from trading_local.clock import ManualClock, to_ts
from trading_local.contracts import Agenda
from trading_local.db import repo_agenda, repo_news, repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_market import InstrumentBook
from trading_local.events import EventBus
from trading_local.models import EarningsDate, NewsItem
from trading_local.testing import seed_instruments

NOW = datetime(2026, 11, 24, 15, 0, 20, tzinfo=UTC)  # Tuesday, 10:00:20 in New York
NOW_TS = to_ts(NOW)

WEEK = [
    {"title": "CB Consumer Confidence", "country": "USD", "date": "2026-11-24T10:00:00-05:00", "impact": "High",
     "forecast": "92.4", "previous": "91.7"},
    {"title": "Richmond Manufacturing Index", "country": "USD", "date": "2026-11-24T10:00:00-05:00",
     "impact": "Low", "forecast": "-2", "previous": "-2"},
    {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-11-25T08:30:00-05:00", "impact": "High",
     "forecast": "0.3%", "previous": "0.3%"},
]  # fmt: skip


class FakeData:
    def __init__(self) -> None:
        self.answers: dict[str, list[EarningsDate] | None] = {}
        self.calls: list[str] = []

    async def earnings(self, symbol: str) -> list[EarningsDate] | None:
        self.calls.append(symbol)
        return self.answers.get(symbol, [])


class Calendar:
    """Serves the ForexFactory week through ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.body: object = WEEK
        self.status = 200
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        assert str(request.url) == sources.FOREXFACTORY_URL
        return httpx.Response(self.status, content=json.dumps(self.body))


def headline(
    db: Database, title: str, fetched_ts: int, source: str = "financialjuice", published_ts: int | None = None
) -> None:
    item = NewsItem(uid=f"uid-{title}-{fetched_ts}", title_key=f"key-{title}-{fetched_ts}", source=source,
                    url="https://news.test/x", title=title, fetched_ts=fetched_ts, published_ts=published_ts)  # fmt: skip
    repo_news.insert_news(db, [item])


@pytest.fixture
def setup(db: Database, clock: ManualClock, bus: EventBus) -> tuple[AgendaService, FakeData, Calendar]:
    seed_instruments(db)
    clock.set(NOW)
    data, calendar = FakeData(), Calendar()
    service = AgendaService(
        db=db,
        clock=clock,
        bus=bus,
        instruments=InstrumentBook(db),
        data=data,  # type: ignore[arg-type]
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(calendar)),
    )
    return service, data, calendar


def by_title(db: Database) -> dict[str, repo_agenda.AgendaEvent]:
    events = repo_agenda.events_between(db, start_ts=0, end_ts=2**40)
    return {event.title if event.kind != "earnings" else f"{event.symbol} {event.event_ts}": event for event in events}


async def test_weekly_calendar_is_loaded_hourly_and_follows_changes(
    setup: tuple[AgendaService, FakeData, Calendar], db: Database, clock: ManualClock
) -> None:
    service, _, calendar = setup
    await service.start()
    assert await service.refresh(clock.now()) >= 3
    assert set(by_title(db)) >= {"CB Consumer Confidence", "Richmond Manufacturing Index", "Core PCE Price Index m/m"}
    clock.advance(600)
    await service.refresh(clock.now())
    assert calendar.requests == 1  # hourly

    calendar.body = [WEEK[0], {**WEEK[2], "date": "2026-11-26T08:30:00-05:00"}]  # moved by a day
    clock.advance(3600)
    await service.refresh(clock.now())
    titles = {event.uid for event in repo_agenda.events_between(db, start_ts=0, end_ts=2**40)}
    assert titles == {"ff:USD:CB Consumer Confidence:2026-11-24", "ff:USD:Core PCE Price Index m/m:2026-11-26"}

    calendar.status = 503
    clock.advance(3600)
    await service.refresh(clock.now())  # a failing calendar keeps what is known
    assert len(titles) == 2 and calendar.requests == 3
    await service.stop()


async def test_actual_values_are_matched_and_judged(
    setup: tuple[AgendaService, FakeData, Calendar], db: Database, clock: ManualClock
) -> None:
    service, _, _ = setup
    await service.start()
    await service.refresh(clock.now())
    release = to_ts(datetime(2026, 11, 24, 15, 0, 30, tzinfo=UTC))
    headline(db, "US CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)", release)
    headline(db, "Richmond Fed Comp. Index Actual 3 (Forecast -2, Previous -2)", release)
    headline(db, "US Redbook YoY Actual 11.3% (Forecast -, Previous 9.5%)", release)
    headline(db, "US CB Consumer Confidence Actual 103.3 (Forecast 92.4, Previous 91.7)", release + 60)  # again
    headline(db, "Bitcoin rebounds above 60,000", release)
    headline(db, "US CB Consumer Confidence Actual 1 (Forecast 2, Previous 3)", release, source="forexlive")
    clock.set(release + 90)
    await service.refresh(clock.now())

    found = by_title(db)
    confidence = found["CB Consumer Confidence"]
    assert (confidence.actual, confidence.actual_text, confidence.actual_ts) == (103.3, "103.3", release)
    assert confidence.surprise == pytest.approx(10.9) and confidence.surprise_z is None  # no history yet
    richmond = found["Richmond Manufacturing Index"]  # matched by its forecast and previous value
    assert richmond.actual == 3.0 and richmond.surprise == pytest.approx(5.0)
    redbook = found["Redbook YoY"]  # no calendar entry: a minor release of its own
    assert (redbook.source, redbook.importance, redbook.actual, redbook.surprise) == ("financialjuice", 1, 11.3, None)
    assert len([event for event in found.values() if event.title.startswith("CB Consumer")]) == 1
    assert [entry.message for entry in repo_ops.list_activity(db, category="news")] == [
        "Konjunkturdaten USD: CB Consumer Confidence 103.3 (Prognose 92.4) – über Erwartung"
    ]  # only the high-importance release
    # an engine deciding before the release cannot see the actual value
    early = repo_agenda.events_between(db, start_ts=0, end_ts=2**40, asof_ts=release - 1)
    assert all(event.actual is None for event in early if event.title == "CB Consumer Confidence")
    await service.stop()


async def test_headlines_fetched_in_a_burst_are_matched_by_their_publication_time(
    setup: tuple[AgendaService, FakeData, Calendar], db: Database, clock: ManualClock
) -> None:
    """Live 2026-11-24: after the start at 09:59 UTC the Ifo headline of 09:00 arrived with the burst."""
    service, _, calendar = setup
    calendar.body = [{"title": "German ifo Business Climate", "country": "EUR", "date": "2026-11-24T04:00:00-05:00",
                      "impact": "Medium", "forecast": "90.5", "previous": "90.2"}]  # fmt: skip
    await service.start()
    await service.refresh(clock.now())
    release = to_ts(datetime(2026, 11, 24, 9, 0, 20, tzinfo=UTC))
    burst = to_ts(datetime(2026, 11, 24, 9, 59, 30, tzinfo=UTC))
    headline(db, "German IFO Business Climate Actual 90.6 (Forecast 90.5, Previous 90.2)", burst, published_ts=release)
    clock.set(burst + 60)
    await service.refresh(clock.now())
    ifo = by_title(db)["German ifo Business Climate"]
    assert (ifo.actual, ifo.actual_ts) == (90.6, burst)  # known since it was seen, not since it was published
    assert "IFO Business Climate" not in by_title(db)
    await service.stop()


def test_release_time_of_a_headline() -> None:
    from trading_local.agenda.service import release_ts

    assert release_ts(1000, 1060) == 1000
    assert release_ts(None, 1060) == 1060
    assert release_ts(1060 - 13 * 3600, 1060) == 1060  # implausibly old
    assert release_ts(1060 + 3600, 1060) == 1060  # from the future


async def test_standardised_surprise_from_earlier_releases(db: Database) -> None:
    for month in range(1, 8):
        event = repo_agenda.AgendaEvent(
            uid=f"ff:USD:CPI m/m:2026-0{month}-10", kind="macro", source="forexfactory", title="CPI m/m",
            event_ts=1_767_000_000 + month * 2_600_000, first_seen_ts=0, updated_ts=0, currency="USD",
            importance=3, forecast_text="0.3%", actual_text="0.4%" if month % 2 else "0.2%", actual=0.4,
            surprise=0.1 if month % 2 else -0.1, actual_ts=0,
        )  # fmt: skip
        repo_agenda.upsert_events(db, [event])
    earlier = repo_agenda.earlier_surprises(db, currency="USD", title="CPI m/m", before_ts=2**40)
    assert len(earlier) == 7 and earlier[0] == pytest.approx(0.1)


async def test_earnings_are_refreshed_a_few_companies_at_a_time(
    setup: tuple[AgendaService, FakeData, Calendar], db: Database, clock: ManualClock
) -> None:
    service, data, _ = setup
    upcoming = int(datetime(2026, 11, 25, 21, 0, tzinfo=UTC).timestamp())
    data.answers["NVDA"] = [EarningsDate("NVDA", upcoming, True, 1.25)]
    data.answers["SAP.DE"] = None  # failed
    await service.start()
    await service.refresh(clock.now())
    assert data.calls == ["NVDA", "SAP.DE"]  # two per minute, by sort order
    assert f"NVDA {upcoming}" in by_title(db)
    clock.advance(60)
    await service.refresh(clock.now())
    assert data.calls[2:] == ["SHEL.L", "7203.T"]  # every share once ...
    clock.advance(60)
    await service.refresh(clock.now())
    assert data.calls[4:] == []  # ... then not before the next interval

    # within two days before the release: every 3 hours; the date moved meanwhile
    moved = upcoming + 86_400
    data.answers["NVDA"] = [EarningsDate("NVDA", moved, True, 1.25)]
    clock.advance(3 * 3600)
    await service.refresh(clock.now())
    assert data.calls[4:5] == ["NVDA"]
    assert {key for key in by_title(db) if key.startswith("NVDA")} == {f"NVDA {moved}"}  # the old date is gone
    await service.stop()


def test_release_message() -> None:
    event = repo_agenda.AgendaEvent(uid="u", kind="macro", source="forexfactory", title="CPI m/m", event_ts=0,
                                    first_seen_ts=0, updated_ts=0, currency="USD", forecast_text="0.3%")  # fmt: skip
    value = sources.parse_actual_headline("US CPI MoM Actual 0.4% (Forecast 0.3%, Previous 0.2%)")
    assert value is not None
    assert release_message_de(event, value, 1.84) == (
        "Konjunkturdaten USD: CPI m/m 0.4% (Prognose 0.3%) – über Erwartung (+1,8σ)"
    )


def test_factory(db: Database, clock: ManualClock, bus: EventBus) -> None:
    agenda = build_agenda(db=db, clock=clock, bus=bus, instruments=InstrumentBook(db), data=FakeData())  # type: ignore[arg-type]
    assert isinstance(agenda, AgendaService) and isinstance(agenda, Agenda)
