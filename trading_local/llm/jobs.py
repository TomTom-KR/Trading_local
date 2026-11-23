"""Priority job queue in front of the GPU: exactly one worker task runs one job at a time.

Why the application schedules itself (LLM probe): the Ollama server runs with ``OLLAMA_NUM_PARALLEL=1`` and
serves requests strictly first-in-first-out, so a time-critical trade review would otherwise wait behind
every queued news item, invisibly and without priority. Hence one request in flight, ordered here:

* priority TRADE_REVIEW (0) before NEWS (1) before CHART (2);
* first-in-first-out within TRADE_REVIEW and CHART, newest first within NEWS (fresh news matters most);
* a job that is still waiting when its ``max_wait_s`` has passed is dropped without running
  (``JobExpired``); its caller learns it at that moment, not when the worker would have reached it;
* a caller that stops waiting (timeout or cancellation of its task) cancels the job: a waiting job is
  removed, a running job is cancelled. With the Ollama client, cancelling closes the HTTP connection and
  the server stops generating within ~0.1 s (verified), so no zombie job keeps burning GPU time.

Pre-emption of a running low-priority job is not needed: a running news or chart call delays an urgent
review by at most ~2-5 s (measured p90), far below the review timeout.

All methods must be called from the event loop thread, except ``metrics`` (thread-safe snapshot).
"""

import asyncio
import heapq
import logging
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

log = logging.getLogger(__name__)


class Priority(IntEnum):
    TRADE_REVIEW = 0
    NEWS = 1
    CHART = 2


class JobNotRun(Exception):
    """The job did not run to completion inside the queue (see the subclasses)."""


class JobExpired(JobNotRun):
    """The job was still waiting when its maximum waiting time had passed."""


class QueueStopped(JobNotRun):
    """The queue was stopped before the job could finish."""


@dataclass(frozen=True, slots=True)
class QueueMetrics:
    depth: int  # jobs waiting (the running job is not counted)
    busy: bool  # a job (or the warm-up) is running
    last_run_ms: int | None  # duration of the most recently finished job
    last_wait_ms: int | None  # queue wait of the most recently started job
    expired: int  # jobs dropped because their waiting time ran out (since the queue was created)


type JobWork[T] = Callable[[int], Awaitable[T]]  # receives the job's queue wait in milliseconds


@dataclass(eq=False, slots=True)
class _Job:
    priority: Priority
    work: JobWork[Any]
    future: asyncio.Future[Any]
    enqueued: float  # time.monotonic()
    state: Literal["waiting", "running", "done"] = "waiting"
    expiry: asyncio.TimerHandle | None = None
    task: asyncio.Task[Any] | None = None


@dataclass(slots=True)
class _Stats:
    busy: bool = False
    last_run_ms: int | None = None
    last_wait_ms: int | None = None
    expired: int = 0


class PriorityJobQueue:
    """Runs submitted jobs one at a time in priority order (see the module docstring)."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, _Job]] = []  # (priority, order, sequence, job)
        self._waiting: set[_Job] = set()
        self._sequence = 0
        self._wakeup = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._current: _Job | None = None
        self._stopped = False
        self._stats = _Stats()

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self, warmup: Callable[[], Awaitable[None]] | None = None) -> None:
        """Start the worker. ``warmup`` (e.g. loading the model) runs once before the first job; jobs that
        arrive meanwhile wait in priority order. Exceptions of ``warmup`` are logged, never fatal."""
        if self._worker is not None:
            return
        self._stopped = False
        self._worker = asyncio.create_task(self._work(warmup), name="llm-queue-worker")

    async def stop(self) -> None:
        """Stop without draining: waiting and running jobs end with ``QueueStopped`` for their callers."""
        self._stopped = True
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        current, self._current = self._current, None
        if current is not None and current.task is not None:
            current.task.cancel()
            await asyncio.gather(current.task, return_exceptions=True)
            self._settle(current, current.task)
        for job in [*self._waiting, *([current] if current is not None else [])]:
            self._finish(job, QueueStopped("the LLM queue was stopped"))  # no-op for a settled job
        self._waiting.clear()
        self._heap.clear()

    # -- submitting -----------------------------------------------------------------------------
    async def run[T](self, priority: Priority, work: JobWork[T], *, max_wait_s: float | None = None) -> T:
        """Queue ``work`` and return its result once it has run.

        ``work`` is called with the job's queue wait in milliseconds. Raises what ``work`` raises,
        ``JobExpired`` if the job was still waiting after ``max_wait_s`` seconds (None or infinity: no limit;
        zero or less: expires at the next loop iteration unless the worker takes it first), ``QueueStopped``
        if the queue is or gets stopped and ``ValueError`` for a NaN ``max_wait_s`` (a NaN timer would crash
        the event loop). Cancelling the awaiting task removes or cancels the job.
        """
        if max_wait_s is not None and math.isnan(max_wait_s):
            raise ValueError("max_wait_s must not be NaN")
        if self._stopped:
            raise QueueStopped("the LLM queue is stopped")
        loop = asyncio.get_running_loop()
        job = _Job(priority=priority, work=work, future=loop.create_future(), enqueued=time.monotonic())
        self._sequence += 1
        order = -self._sequence if priority is Priority.NEWS else self._sequence
        heapq.heappush(self._heap, (int(priority), order, self._sequence, job))
        self._waiting.add(job)
        if max_wait_s is not None:
            job.expiry = loop.call_later(max_wait_s, self._expire, job)
        job.future.add_done_callback(lambda _: self._on_future_done(job))
        self._wakeup.set()
        result: T = await job.future
        return result

    def metrics(self) -> QueueMetrics:
        stats = self._stats
        return QueueMetrics(
            depth=len(self._waiting),
            busy=stats.busy,
            last_run_ms=stats.last_run_ms,
            last_wait_ms=stats.last_wait_ms,
            expired=stats.expired,
        )

    # -- worker ---------------------------------------------------------------------------------
    async def _work(self, warmup: Callable[[], Awaitable[None]] | None) -> None:
        if warmup is not None:
            self._stats.busy = True
            try:
                await warmup()
            except Exception:
                log.exception("LLM warm-up failed")
            finally:
                self._stats.busy = False
        while True:
            job = await self._next_job()
            await self._execute(job)

    async def _next_job(self) -> _Job:
        while True:
            while self._heap:
                job = heapq.heappop(self._heap)[3]
                # A cancelled caller's future is done at once; its done-callback (which marks the job) may not
                # have run yet.
                if job.state == "waiting" and not job.future.done():
                    return job
            self._wakeup.clear()
            await self._wakeup.wait()

    async def _execute(self, job: _Job) -> None:
        """Run one job; the caller's future gets its result or exception (unless the caller gave up)."""
        self._waiting.discard(job)
        if job.expiry is not None:
            job.expiry.cancel()
        wait_ms = round((time.monotonic() - job.enqueued) * 1000)
        job.state = "running"
        self._current = job
        self._stats.busy = True
        self._stats.last_wait_ms = wait_ms
        started = time.monotonic()
        task = asyncio.create_task(job.work(wait_ms))
        job.task = task
        try:
            await asyncio.wait({task})  # never raises for the task's own outcome
        finally:
            # Also runs when the worker itself is cancelled (stop), possibly in the very loop iteration in
            # which the job finished: a finished job is settled here, an unfinished one is cancelled and left
            # as ``_current`` for stop(), which awaits it and settles the future.
            self._stats.busy = False
            if task.done():
                self._current = None
                self._stats.last_run_ms = round((time.monotonic() - started) * 1000)
                self._settle(job, task)
            else:
                task.cancel()

    def _settle(self, job: _Job, task: asyncio.Task[Any]) -> None:
        """Hand the outcome of a finished job task to the caller's future."""
        if task.cancelled():
            self._finish(job, QueueStopped("the job was cancelled"))
        elif (error := task.exception()) is not None:
            self._finish(job, error)
        else:
            self._finish(job, None, task.result())

    # -- job state ------------------------------------------------------------------------------
    def _expire(self, job: _Job) -> None:
        if job.state != "waiting" or job.future.done():  # running, finished, or its caller already gave up
            return
        self._waiting.discard(job)
        self._stats.expired += 1
        log.info("LLM job (priority %s) dropped after waiting too long", job.priority.name)
        self._finish(job, JobExpired("the job waited longer than its deadline"))

    def _on_future_done(self, job: _Job) -> None:
        """The caller stopped waiting: drop a waiting job, cancel a running one."""
        if not job.future.cancelled():
            return
        if job.state == "waiting":
            self._waiting.discard(job)
            if job.expiry is not None:
                job.expiry.cancel()
            job.state = "done"
        elif job.state == "running" and job.task is not None:
            job.task.cancel()

    @staticmethod
    def _finish(job: _Job, error: BaseException | None, result: Any = None) -> None:
        job.state = "done"
        if job.expiry is not None:
            job.expiry.cancel()
        if job.future.done():
            return
        if error is not None:
            job.future.set_exception(error)
        else:
            job.future.set_result(result)
