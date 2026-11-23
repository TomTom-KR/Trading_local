"""PriorityJobQueue: ordering, deadlines, cancellation without zombie work, stop semantics, metrics."""

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

import pytest

from trading_local.llm.jobs import JobExpired, Priority, PriorityJobQueue, QueueStopped


class Recorder:
    """Jobs that log when they start; the first job can be held open with a gate."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.waits: dict[str, int] = {}
        self.gate = asyncio.Event()

    def job(self, name: str, *, hold: bool = False, seconds: float = 0.0) -> Callable[[int], Coroutine[Any, Any, str]]:
        async def work(queue_wait_ms: int) -> str:
            self.started.append(name)
            self.waits[name] = queue_wait_ms
            try:
                if hold:
                    await self.gate.wait()
                if seconds:
                    await asyncio.sleep(seconds)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise
            return name

        return work

    async def until_started(self, name: str) -> None:
        async with asyncio.timeout(2):
            while name not in self.started:
                await asyncio.sleep(0.001)


@pytest.fixture
async def queue() -> AsyncIterator[PriorityJobQueue]:
    q = PriorityJobQueue()
    q.start()
    yield q
    await q.stop()


async def _settle() -> None:
    """Let freshly created tasks reach their first await."""
    for _ in range(5):
        await asyncio.sleep(0)


async def test_strict_priority_newest_news_first_and_fifo_reviews_and_charts(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    blocker = asyncio.create_task(queue.run(Priority.CHART, rec.job("blocker", hold=True)))
    await rec.until_started("blocker")
    submissions = [
        (Priority.CHART, "chart1"),
        (Priority.NEWS, "news1"),
        (Priority.TRADE_REVIEW, "review1"),
        (Priority.NEWS, "news2"),
        (Priority.CHART, "chart2"),
        (Priority.TRADE_REVIEW, "review2"),
        (Priority.NEWS, "news3"),
    ]
    tasks = []
    for priority, name in submissions:
        tasks.append(asyncio.create_task(queue.run(priority, rec.job(name))))
        await _settle()
    assert queue.metrics().depth == len(submissions)
    rec.gate.set()
    results = await asyncio.gather(blocker, *tasks)

    assert results == ["blocker", *(name for _, name in submissions)]
    assert rec.started == ["blocker", "review1", "review2", "news3", "news2", "news1", "chart1", "chart2"]
    assert queue.metrics().depth == 0


async def test_one_job_at_a_time(queue: PriorityJobQueue) -> None:
    running = 0
    peak = 0

    async def work(_: int) -> None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.005)
        running -= 1

    await asyncio.gather(*(queue.run(Priority.NEWS, work) for _ in range(6)))
    assert peak == 1


async def test_waiting_job_is_dropped_at_its_deadline_without_running(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    blocker = asyncio.create_task(queue.run(Priority.TRADE_REVIEW, rec.job("blocker", hold=True)))
    await rec.until_started("blocker")
    loop = asyncio.get_running_loop()
    started = loop.time()

    with pytest.raises(JobExpired):
        await queue.run(Priority.NEWS, rec.job("stale"), max_wait_s=0.05)

    assert loop.time() - started < 0.5  # the caller learns it at the deadline, not when the worker is free
    assert queue.metrics().expired == 1
    assert queue.metrics().depth == 0
    rec.gate.set()
    await blocker
    await queue.run(Priority.NEWS, rec.job("after"))
    assert rec.started == ["blocker", "after"]


async def test_deadline_does_not_cut_a_job_that_already_runs(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    assert await queue.run(Priority.NEWS, rec.job("slow", seconds=0.08), max_wait_s=0.02) == "slow"
    assert queue.metrics().expired == 0


async def test_caller_giving_up_while_waiting_removes_the_job(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    blocker = asyncio.create_task(queue.run(Priority.TRADE_REVIEW, rec.job("blocker", hold=True)))
    await rec.until_started("blocker")

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await queue.run(Priority.CHART, rec.job("impatient"))

    assert queue.metrics().depth == 0
    rec.gate.set()
    await blocker
    await queue.run(Priority.CHART, rec.job("next"))
    assert rec.started == ["blocker", "next"]  # the abandoned job never ran


async def test_caller_giving_up_while_running_cancels_the_work(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.03):
            await queue.run(Priority.TRADE_REVIEW, rec.job("review", hold=True))
    await _settle()

    assert rec.cancelled == ["review"]
    assert queue.metrics().busy is False
    assert await queue.run(Priority.NEWS, rec.job("next")) == "next"


async def test_work_exception_reaches_the_caller_and_the_worker_survives(queue: PriorityJobQueue) -> None:
    async def broken(_: int) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await queue.run(Priority.NEWS, broken)
    rec = Recorder()
    assert await queue.run(Priority.NEWS, rec.job("fine")) == "fine"


async def test_stop_cancels_waiting_and_running_jobs() -> None:
    queue = PriorityJobQueue()
    queue.start()
    rec = Recorder()
    running = asyncio.create_task(queue.run(Priority.TRADE_REVIEW, rec.job("running", hold=True)))
    await rec.until_started("running")
    waiting = [asyncio.create_task(queue.run(Priority.NEWS, rec.job(f"news{i}"))) for i in range(3)]
    await _settle()

    await queue.stop()

    for task in [running, *waiting]:
        with pytest.raises(QueueStopped):
            await task
    assert rec.started == ["running"]
    assert rec.cancelled == ["running"]
    assert queue.metrics().depth == 0
    with pytest.raises(QueueStopped):
        await queue.run(Priority.NEWS, rec.job("late"))


async def test_metrics_report_busy_depth_and_waits(queue: PriorityJobQueue) -> None:
    rec = Recorder()
    first = asyncio.create_task(queue.run(Priority.NEWS, rec.job("first", hold=True)))
    await rec.until_started("first")
    second = asyncio.create_task(queue.run(Priority.NEWS, rec.job("second")))
    await asyncio.sleep(0.03)

    metrics = queue.metrics()
    assert metrics.busy is True
    assert metrics.depth == 1
    rec.gate.set()
    await asyncio.gather(first, second)
    metrics = queue.metrics()
    assert metrics.busy is False
    assert metrics.depth == 0
    assert rec.waits["second"] >= 25  # it waited behind 'first'
    assert metrics.last_wait_ms == rec.waits["second"]
    assert metrics.last_run_ms is not None


async def test_warmup_runs_before_the_first_job_and_its_failure_is_not_fatal() -> None:
    events: list[str] = []

    async def warmup() -> None:
        events.append("warmup")
        raise RuntimeError("model missing")

    async def work(_: int) -> str:
        events.append("job")
        return "ok"

    queue = PriorityJobQueue()
    pending = asyncio.create_task(queue.run(Priority.NEWS, work))  # submitted before the worker exists
    await _settle()
    queue.start(warmup=warmup)
    try:
        assert await pending == "ok"
        assert events == ["warmup", "job"]
    finally:
        await queue.stop()
