"""Attack tests: races around stop(), invalid timeouts, failure injection, restart and thread-safety."""

import asyncio
import math
import threading
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from tests.llm.fakes import NVDA, SAP, T0, FakeOllamaClient, Gate, make_news, make_review_context
from trading_local.clock import ManualClock
from trading_local.config import Settings
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.llm import OllamaLlm, schemas
from trading_local.llm.jobs import Priority, PriorityJobQueue, QueueStopped

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
        llm = OllamaLlm(settings=settings, db=db, clock=clock, client=fake, **kwargs)
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


# -- queue races ----------------------------------------------------------------------------------
async def test_stop_in_the_iteration_in_which_a_job_finishes_does_not_strand_its_caller() -> None:
    queue = PriorityJobQueue()
    queue.start()

    async def work(_: int) -> str:
        assert queue._worker is not None
        queue._worker.cancel()  # what stop() does when it runs right as the job ends
        return "done"

    caller = asyncio.create_task(queue.run(Priority.NEWS, work))
    await asyncio.sleep(0.01)
    await queue.stop()
    async with asyncio.timeout(1):  # before the fix the caller waited forever
        assert await caller == "done"


async def test_a_caller_that_gave_up_before_the_worker_took_the_job_never_starts_it() -> None:
    queue = PriorityJobQueue()
    queue.start()
    started: list[int] = []

    async def work(wait_ms: int) -> None:
        started.append(wait_ms)

    try:
        caller = asyncio.create_task(queue.run(Priority.TRADE_REVIEW, work))
        await asyncio.sleep(0)  # the job is queued and the worker is woken ...
        caller.cancel()  # ... but the caller gives up before the worker runs
        with pytest.raises(asyncio.CancelledError):
            await caller
        for _ in range(5):
            await asyncio.sleep(0)
        assert started == []
        assert queue.metrics().last_wait_ms is None  # the worker skipped the job instead of executing it
        assert queue.metrics().depth == 0
    finally:
        await queue.stop()


async def test_nan_waiting_time_is_rejected_instead_of_crashing_the_event_loop() -> None:
    queue = PriorityJobQueue()
    queue.start()

    async def work(_: int) -> str:
        return "ok"

    try:
        with pytest.raises(ValueError):
            await queue.run(Priority.NEWS, work, max_wait_s=math.nan)
        assert await queue.run(Priority.NEWS, work, max_wait_s=math.inf) == "ok"
        await asyncio.sleep(0.01)  # the loop is still alive
    finally:
        await queue.stop()


async def test_stop_while_a_job_runs_settles_every_caller_exactly_once() -> None:
    queue = PriorityJobQueue()
    queue.start()
    gate = asyncio.Event()

    async def held(_: int) -> str:
        await gate.wait()
        return "never"

    callers = [asyncio.create_task(queue.run(Priority.CHART, held)) for _ in range(4)]
    await asyncio.sleep(0.01)
    await queue.stop()
    results = await asyncio.gather(*callers, return_exceptions=True)
    assert all(isinstance(result, QueueStopped) for result in results)
    metrics = queue.metrics()
    assert (metrics.depth, metrics.busy, metrics.expired) == (0, False, 0)


# -- review timeouts ------------------------------------------------------------------------------
@pytest.mark.parametrize("timeout_s", [math.nan, -1.0, 0.0])
async def test_review_with_an_invalid_timeout_is_unavailable_at_once(
    llm: OllamaLlm, fake: FakeOllamaClient, timeout_s: float
) -> None:
    review = await llm.review_trade(make_review_context(), timeout_s=timeout_s)
    assert review.verdict == "unavailable"
    await asyncio.sleep(0.01)  # the event loop survived (a NaN timer used to crash it)
    assert fake.calls == []
    assert (await llm.review_trade(make_review_context(), timeout_s=5)).verdict == "confirm"


async def test_review_timeout_covers_queue_wait_plus_generation(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    fake.delay_s = 0.06  # every call takes 60 ms
    busy = asyncio.create_task(llm.analyze_chart(NVDA, "15m", b"png", {"Kurs": 100.0}))
    await asyncio.sleep(0.005)
    loop = asyncio.get_running_loop()
    started = loop.time()
    review = await llm.review_trade(make_review_context(), timeout_s=0.09)  # 55 ms queue wait + 60 ms > 90 ms
    elapsed = loop.time() - started
    await busy
    await asyncio.sleep(0.01)  # the cancelled model call unwinds one loop iteration after the caller returned

    assert review.verdict == "unavailable"
    assert 0.08 <= elapsed < 0.3
    assert fake.calls[1].kind == "trade_review" and fake.calls[1].timeout_s < 0.09  # remaining budget only
    assert fake.active == 0


# -- failure injection ----------------------------------------------------------------------------
async def test_a_conversion_defect_is_logged_not_retried_and_not_raised(
    llm: OllamaLlm, fake: FakeOllamaClient, db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any, **__: Any) -> Any:
        raise KeyError("bug")

    monkeypatch.setattr(schemas, "to_news_analysis", broken)
    assert await llm.analyze_news(make_news(1), [NVDA]) is None
    assert len(fake.calls) == 1  # a defect is not an invalid answer: no retry
    await llm.stop()
    rows = repo_ops.list_llm_calls(db)
    assert [(row.kind, row.ok, row.error) for row in rows] == [
        ("news", False, "Antwort konnte nicht verarbeitet werden")
    ]


async def test_a_news_caller_that_stops_waiting_leaves_no_zombie_job(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    gate = Gate(fake)
    running = asyncio.create_task(llm.analyze_news(make_news(1), [NVDA]))
    await gate.wait_busy()
    waiting = asyncio.create_task(llm.analyze_news(make_news(2, title="second item"), [NVDA]))
    await asyncio.sleep(0.01)
    assert llm.status().queue_depth == 1

    waiting.cancel()
    running.cancel()
    await asyncio.gather(waiting, running, return_exceptions=True)
    await asyncio.sleep(0.01)

    assert llm.status().queue_depth == 0 and llm.status().busy is False
    assert len(fake.calls) == 1 and len(fake.cancelled) == 1  # the running call was cancelled, the other never ran


async def test_stop_during_the_warm_up_releases_every_caller(make_llm: Factory, fake: FakeOllamaClient) -> None:
    warm_up_started = asyncio.Event()

    async def slow_warm_up() -> int:
        warm_up_started.set()
        await asyncio.sleep(10)
        return 0

    fake.warm_up = slow_warm_up  # type: ignore[method-assign]
    llm = make_llm()
    await llm.start()
    await warm_up_started.wait()
    news = asyncio.create_task(llm.analyze_news(make_news(1), [NVDA]))
    review = asyncio.create_task(llm.review_trade(make_review_context(), timeout_s=30))
    chart = asyncio.create_task(llm.analyze_chart(SAP, "15m", b"png", {"Kurs": 100.0}))
    await asyncio.sleep(0.01)
    assert llm.status().busy is True and llm.status().queue_depth == 3

    async with asyncio.timeout(1):
        await llm.stop()
        assert await news is None
        assert (await review).verdict == "unavailable"
        assert await chart is None
    assert fake.calls == []


# -- status ---------------------------------------------------------------------------------------
async def test_status_of_the_last_hour_survives_a_restart(
    make_llm: Factory, db: Database, clock: ManualClock
) -> None:
    now = clock.ts()
    repo_ops.log_llm_call(db, ts=now - 4000, kind="news", ok=False, latency_ms=1, error="zu alt")
    repo_ops.log_llm_call(db, ts=now - 1800, kind="news", ok=True, latency_ms=1700)
    repo_ops.log_llm_call(db, ts=now - 900, kind="chart", ok=False, latency_ms=30_000, error="Zeitüberschreitung")
    repo_ops.log_llm_call(db, ts=now - 60, kind="trade_review", ok=True, latency_ms=3100)

    llm = make_llm()
    await llm.start()
    status = llm.status()
    assert (status.calls_last_hour, status.errors_last_hour) == (3, 1)
    assert status.last_error == "Zeitüberschreitung"
    assert status.last_latency_ms == 3100

    clock.advance(2800)  # the calls at -1800 s and -900 s leave the window
    status = llm.status()
    assert (status.calls_last_hour, status.errors_last_hour, status.last_error) == (1, 0, "")


async def test_status_can_be_read_from_other_threads_while_jobs_run(llm: OllamaLlm, fake: FakeOllamaClient) -> None:
    fake.delay_s = 0.002
    failures: list[BaseException] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.wait(0.0005):  # API threads poll; a pause keeps them from monopolising the GIL
            try:
                status = llm.status()
                assert status.queue_depth >= 0
            except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
                failures.append(exc)
                return

    readers = [threading.Thread(target=poll) for _ in range(2)]
    for reader in readers:
        reader.start()
    try:
        jobs = [llm.analyze_news(make_news(i), [NVDA]) for i in range(1, 21)]
        jobs.append(llm.review_trade(make_review_context(), timeout_s=5))
        results = await asyncio.gather(*jobs)
    finally:
        stop.set()
        for reader in readers:
            reader.join()
    assert failures == []
    assert all(result is not None for result in results)
    assert fake.max_active == 1
    assert llm.status().calls_last_hour == 21


async def test_analyses_are_stamped_with_the_time_they_became_known(
    llm: OllamaLlm, fake: FakeOllamaClient, clock: ManualClock
) -> None:
    gate = Gate(fake)
    task = asyncio.create_task(llm.analyze_news(make_news(1), [NVDA]))
    await gate.wait_busy()
    clock.advance(90)  # the model needs time; the analysis must not claim to be older than it is
    gate.open()
    analysis = await task
    assert analysis is not None and analysis.analyzed_ts == T0 + 90
