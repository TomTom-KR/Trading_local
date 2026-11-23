"""OllamaLlm on top of the fake client: protocol behaviour, priorities, timeouts, retries, logging, status."""

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.llm.fakes import (
    NVDA,
    SAP,
    FakeOllamaClient,
    Gate,
    chart_answer,
    make_news,
    make_review_context,
    news_answer,
    review_answer,
    timeout_error,
    unreachable_error,
)
from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.contracts import Llm
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.llm import OllamaLlm, build_llm
from trading_local.llm.service import WARMUP_ATTEMPTS
from trading_local.llm.client import LlmError
from trading_local.testing import make_settings

type Factory = Callable[..., OllamaLlm]


@pytest.fixture
def fake() -> FakeOllamaClient:
    return FakeOllamaClient()


@pytest.fixture
async def make_llm(
    settings: Settings, db: Database, clock: ManualClock, fake: FakeOllamaClient
) -> AsyncIterator[Factory]:
    created: list[OllamaLlm] = []

    def factory(**kwargs: Any) -> OllamaLlm:
        llm = OllamaLlm(settings=kwargs.pop("settings", settings), db=db, clock=clock, client=fake, **kwargs)
        created.append(llm)
        return llm

    yield factory
    for llm in created:
        await llm.stop()


@pytest.fixture
async def llm(make_llm: Factory) -> OllamaLlm:
    service = make_llm()
    await service.start()
    return service


def _rows(db: Database) -> list[Any]:
    return list(reversed(repo_ops.list_llm_calls(db)))  # oldest first


# -- the three job types --------------------------------------------------------------------------
async def test_analyze_news_returns_the_mapped_analysis(
    llm: OllamaLlm, fake: FakeOllamaClient, clock: ManualClock
) -> None:
    impacts = [{"symbol": "NVDA", "sentiment": 0.7}, {"symbol": "TSLA", "sentiment": 0.9}]
    fake.replies.append(news_answer(impacts=impacts))
    analysis = await llm.analyze_news(make_news(5), [NVDA, SAP, NVDA])

    assert analysis is not None
    assert analysis.news_id == 5 and analysis.analyzed_ts == clock.ts() and analysis.model == "qwen3.5:9b"
    assert analysis.symbols == {"NVDA": 0.7}  # TSLA is no candidate
    call = fake.calls[0]
    assert call.schema["properties"]["impacts"]["items"]["properties"]["symbol"]["enum"] == ["NVDA", "SAP.DE"]
    assert call.think is False and call.timeout_s == 20.0
    assert call.options == {"num_predict": 400}
    assert "<news>" in call.user and call.images == ()


async def test_review_trade_returns_the_verdict_with_end_to_end_latency(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    fake.replies.append(review_answer(verdict="reduce", conviction=0.55, risks_de=["RSI überkauft"]))
    review = await llm.review_trade(make_review_context(chart_png=b"\x89PNG-review"), timeout_s=5)

    assert review.verdict == "reduce" and review.conviction == 0.55
    assert review.risks_de == "RSI überkauft"
    assert review.latency_ms is not None and review.latency_ms >= 0
    assert fake.calls[0].images == (b"\x89PNG-review",)
    assert fake.calls[0].timeout_s <= 5


async def test_analyze_chart_passes_the_image_and_checks_levels_against_the_last_price(
    llm: OllamaLlm, fake: FakeOllamaClient, clock: ManualClock
) -> None:
    fake.replies.append(chart_answer(support=[70.0, 98.5], resistance=[103.0]))
    analysis = await llm.analyze_chart(NVDA, "15m", b"\x89PNG-chart", {"Kurs": 100.0, "EMA20 (60 Min)": 99.1})

    assert analysis is not None
    assert (analysis.symbol, analysis.timeframe, analysis.ts) == ("NVDA", "15m", clock.ts())
    assert (analysis.support, analysis.resistance) == (98.5, 103.0)
    assert fake.calls[0].images == (b"\x89PNG-chart",)
    assert "Last price: 100" in fake.calls[0].user


async def test_chart_levels_are_dropped_when_the_facts_have_no_last_price(
    llm: OllamaLlm, fake: FakeOllamaClient
) -> None:
    analysis = await llm.analyze_chart(NVDA, "15m", b"png", {"EMA20 (60 Min)": 99.1})
    assert analysis is not None and analysis.support is None and analysis.resistance is None


# -- priorities and deadlines ---------------------------------------------------------------------
async def test_reviews_jump_ahead_of_news_and_charts(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    gate = Gate(fake)
    first = asyncio.create_task(llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}))
    await gate.wait_busy()
    chart = asyncio.create_task(llm.analyze_chart(SAP, "15m", b"png", {"Kurs": 100.0}))
    old_news = asyncio.create_task(llm.analyze_news(make_news(1, title="old news"), [NVDA]))
    await asyncio.sleep(0.01)
    new_news = asyncio.create_task(llm.analyze_news(make_news(2, title="fresh news"), [NVDA]))
    review = asyncio.create_task(llm.review_trade(make_review_context(), timeout_s=5))
    await asyncio.sleep(0.01)
    assert llm.status().queue_depth == 4 and llm.status().busy is True
    gate.open()
    await asyncio.gather(first, chart, old_news, new_news, review)

    order = [(call.kind, call.user) for call in fake.calls]
    assert [kind for kind, _ in order] == ["chart", "trade_review", "news", "news", "chart"]
    assert "fresh news" in order[2][1] and "old news" in order[3][1]
    assert fake.max_active == 1


async def test_stale_news_is_dropped_without_calling_the_model(make_llm: Factory, fake: FakeOllamaClient) -> None:
    llm = make_llm(news_max_wait_s=0.05)
    await llm.start()
    gate = Gate(fake)
    busy = asyncio.create_task(llm.review_trade(make_review_context(), timeout_s=5))
    await gate.wait_busy()

    assert await llm.analyze_news(make_news(3), [NVDA]) is None

    gate.open()
    await busy
    assert [call.kind for call in fake.calls] == ["trade_review"]


async def test_review_timeout_while_waiting_never_reaches_the_model(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    gate = Gate(fake)
    busy = asyncio.create_task(llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}))
    await gate.wait_busy()

    review = await llm.review_trade(make_review_context(), timeout_s=0.05)

    assert review.verdict == "unavailable"
    assert review.latency_ms is not None and 40 <= review.latency_ms < 1000
    assert llm.status().queue_depth == 0
    gate.open()
    await busy
    assert [call.kind for call in fake.calls] == ["chart"]


async def test_review_timeout_during_generation_cancels_the_call(
    llm: OllamaLlm, fake: FakeOllamaClient, db: Database
) -> None:
    Gate(fake)  # the model never answers
    review = await llm.review_trade(make_review_context(), timeout_s=0.05)
    await asyncio.sleep(0.01)

    assert review.verdict == "unavailable"
    assert fake.active == 0 and len(fake.cancelled) == 1  # no zombie request left on the GPU
    assert llm.status().busy is False
    await llm.stop()
    rows = [(row.kind, row.ok, row.error) for row in _rows(db)]
    assert rows == [("trade_review", False, "Abgebrochen vor der Antwort")]


async def test_review_never_raises(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    fake.replies.extend([unreachable_error(), review_answer(verdict="maybe")])
    assert (await llm.review_trade(make_review_context(), timeout_s=5)).verdict == "unavailable"
    assert (await llm.review_trade(make_review_context(), timeout_s=5)).verdict == "unavailable"
    assert (await llm.review_trade(make_review_context(), timeout_s=0)).verdict == "unavailable"
    assert len(fake.calls) == 2  # no retries for reviews


# -- retries and validation -----------------------------------------------------------------------
async def test_invalid_news_answer_is_retried_once_with_changed_sampling(
    llm: OllamaLlm, fake: FakeOllamaClient, db: Database
) -> None:
    fake.replies.extend([news_answer(summary_de="Die Aktie pr채gt"), news_answer(summary_de="Nvidia steigt.")])
    analysis = await llm.analyze_news(make_news(4), [NVDA])

    assert analysis is not None and analysis.summary_de == "Nvidia steigt."
    assert [call.options for call in fake.calls] == [
        {"num_predict": 400},
        {"num_predict": 400, "temperature": 0.2},  # the configured 0.1 plus 0.1
    ]
    await llm.stop()
    rows = _rows(db)
    assert [(row.ok, row.error) for row in rows] == [
        (False, "Feld summary_de enthält verstümmelte Zeichen"),
        (True, ""),
    ]
    assert rows[1].queue_wait_ms == 0


async def test_news_gives_up_after_one_retry(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    invalid = LlmError(
        "schema violation: $: missing 'horizon'",
        reason_de="Antwort entspricht nicht dem Schema",
        output_invalid=True,
        latency_ms=900,
    )
    fake.replies.extend([invalid, invalid, news_answer()])
    assert await llm.analyze_news(make_news(4), [NVDA]) is None
    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        unreachable_error(),
        timeout_error(20),
        LlmError("prompt truncated by Ollama (too long)", reason_de="Eingabe zu lang (von Ollama gekürzt)"),
    ],
)
async def test_news_is_not_retried_when_the_model_could_not_answer(
    llm: OllamaLlm, fake: FakeOllamaClient, error: LlmError
) -> None:
    fake.replies.append(error)
    assert await llm.analyze_news(make_news(4), [NVDA]) is None
    assert len(fake.calls) == 1


async def test_news_without_database_id_is_not_analysed(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    assert await llm.analyze_news(make_news(None), [NVDA]) is None  # type: ignore[arg-type]
    assert fake.calls == []


async def test_chart_failure_returns_none(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    fake.replies.extend([chart_answer(bias="sideways"), LlmError("server error 500: boom", reason_de="Serverfehler")])
    assert await llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}) is None
    assert await llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}) is None
    assert len(fake.calls) == 2  # chart analyses are not retried


# -- logging and status ---------------------------------------------------------------------------
async def test_every_model_call_is_logged(
    llm: OllamaLlm, fake: FakeOllamaClient, db: Database, clock: ManualClock
) -> None:
    fake.replies.extend([news_answer(), review_answer(), timeout_error(30, latency_ms=30_000)])
    await llm.analyze_news(make_news(1), [NVDA])
    await llm.review_trade(make_review_context(), timeout_s=5)
    await llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0})
    await llm.stop()

    rows = _rows(db)
    assert [(row.kind, row.ok, row.error) for row in rows] == [
        ("news", True, ""),
        ("trade_review", True, ""),
        ("chart", False, "Zeitüberschreitung nach 30 s"),  # German: the GUI shows it
    ]
    assert all(row.ts == clock.ts() for row in rows)
    assert (rows[0].prompt_tokens, rows[0].output_tokens) == (800, 90)
    assert rows[2].latency_ms == 30_000 and rows[2].prompt_tokens is None


async def test_queue_wait_is_logged(llm: OllamaLlm, fake: FakeOllamaClient, db: Database) -> None:
    gate = Gate(fake)
    first = asyncio.create_task(llm.analyze_news(make_news(1), [NVDA]))
    await gate.wait_busy()
    second = asyncio.create_task(llm.analyze_news(make_news(2), [NVDA]))
    await asyncio.sleep(0.05)
    gate.open()
    await asyncio.gather(first, second)
    await llm.stop()
    waits = [row.queue_wait_ms for row in _rows(db)]
    assert waits[0] < 40 and waits[1] >= 40


async def test_status_counts_calls_and_errors_of_the_last_hour(
    llm: OllamaLlm, fake: FakeOllamaClient, clock: ManualClock
) -> None:
    fake.replies.extend([news_answer(), timeout_error(20, latency_ms=20_000), review_answer()])
    await llm.analyze_news(make_news(1), [NVDA])
    await llm.analyze_news(make_news(2), [NVDA])
    await llm.review_trade(make_review_context(), timeout_s=5)

    status = llm.status()
    assert (status.model, status.enabled, status.available) == ("qwen3.5:9b", True, True)
    assert (status.calls_last_hour, status.errors_last_hour) == (3, 1)
    assert status.last_error == "Zeitüberschreitung nach 20 s"
    assert status.last_latency_ms is not None and status.last_latency_ms < 20_000  # successful calls only
    assert (status.busy, status.queue_depth) == (False, 0)

    clock.advance(3601)
    status = llm.status()
    assert (status.calls_last_hour, status.errors_last_hour, status.last_error) == (0, 0, "")


async def test_availability_follows_health_checks_and_call_outcomes(
    make_llm: Factory, fake: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_local.llm.service.WARMUP_RETRY_S", 0.005)
    fake.healthy = False
    fake.warmup_error = unreachable_error()
    llm = make_llm(health_interval_s=0.02)
    await llm.start()
    await asyncio.sleep(0.08)
    assert llm.status().available is False
    assert fake.warmups == WARMUP_ATTEMPTS and fake.health_calls >= 1

    fake.healthy = True
    await asyncio.sleep(0.05)
    assert llm.status().available is True

    fake.replies.append(unreachable_error())
    fake.healthy = False
    await llm.analyze_news(make_news(1), [NVDA])
    assert llm.status().available is False


async def test_unreachable_warm_up_is_retried_and_succeeds(
    make_llm: Factory, fake: FakeOllamaClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_local.llm.service.WARMUP_RETRY_S", 0.005)
    original = fake.warm_up
    failures = [unreachable_error()]

    async def flaky_warm_up() -> int:
        if failures:
            fake.warmups += 1
            raise failures.pop()
        return await original()

    monkeypatch.setattr(fake, "warm_up", flaky_warm_up)
    llm = make_llm(health_interval_s=30.0)
    await llm.start()
    await asyncio.sleep(0.05)
    assert fake.warmups == 2
    assert llm.status().available is True


async def test_health_is_not_polled_more_often_than_the_interval(make_llm: Factory, fake: FakeOllamaClient) -> None:
    llm = make_llm(health_interval_s=30.0)
    fake.warmup_error = LlmError(
        "model load took longer than 120 s", reason_de="Laden des Modells dauerte länger als 120 s"
    )
    await llm.start()
    await asyncio.sleep(0.05)
    assert fake.health_calls == 1


# -- disabled mode and lifecycle ------------------------------------------------------------------
async def test_disabled_llm_answers_at_once_without_calling_the_model(
    make_llm: Factory, fake: FakeOllamaClient, tmp_path: Path
) -> None:
    llm = make_llm(settings=make_settings(tmp_path, enable_llm=False))
    await llm.start()

    assert await llm.analyze_news(make_news(1), [NVDA]) is None
    review = await llm.review_trade(make_review_context(), timeout_s=5)
    assert (review.verdict, review.latency_ms) == ("unavailable", None)
    assert await llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}) is None
    status = llm.status()
    assert (status.enabled, status.available, status.busy, status.queue_depth) == (False, False, False, 0)
    await llm.stop()
    assert fake.calls == [] and fake.warmups == 0 and fake.health_calls == 0 and not fake.closed


async def test_start_warms_up_in_the_background(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    await asyncio.sleep(0.01)
    assert fake.warmups == 1
    assert llm.status().available is True


async def test_stop_cancels_waiting_and_running_jobs(llm: OllamaLlm, fake: FakeOllamaClient, db: Database) -> None:
    gate = Gate(fake)
    running = asyncio.create_task(llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}))
    await gate.wait_busy()
    news = asyncio.create_task(llm.analyze_news(make_news(1), [NVDA]))
    review = asyncio.create_task(llm.review_trade(make_review_context(), timeout_s=5))
    await asyncio.sleep(0.01)

    await llm.stop()

    assert await running is None
    assert await news is None
    assert (await review).verdict == "unavailable"
    assert len(fake.calls) == 1 and len(fake.cancelled) == 1 and fake.closed
    assert await llm.analyze_news(make_news(2), [NVDA]) is None  # stopped for good
    assert [(row.kind, row.ok) for row in _rows(db)] == [("chart", False)]


async def test_build_llm_implements_the_protocol(settings: Settings, db: Database, clock: ManualClock) -> None:
    llm = build_llm(settings=settings, db=db, clock=clock, runtime_settings=SettingsStore(db))
    assert isinstance(llm, Llm) and isinstance(llm, OllamaLlm)
    assert llm.status().model == settings.ollama_model
    await llm.stop()
