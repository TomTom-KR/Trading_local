"""``OllamaLlm``: the platform's gateway to the local model (implements ``contracts.Llm``).

Every request becomes one job of the priority queue (``jobs.PriorityJobQueue``), so the GPU serves one
request at a time: trade reviews before news before chart analyses. Every model call is recorded in the
``llm_calls`` table (written by one dedicated thread, so a busy database never stalls the queue) and in an
in-memory window of the last hour for ``status()`` (restored from the table at start). Error texts in the
table and in ``status()`` are German (the GUI shows them); log messages carry the English reason.

Decisions taken from the LLM probe (numbers measured on this machine):

* ``think`` is off for all three job types: thinking cost 30-110 s per call and 25-50 % of the answers
  never finished within the token budget; without it news take ~1.7 s, reviews ~3 s (~4 s with chart
  image), chart analyses ~3.5-4 s.
* News are classified ONE item per call. A batch of 5 took ~6.0 s (p50), i.e. 1.3 s instead of ~1.7 s per
  item, but 8 of 60 batched items had a wrong field versus 2 of 36 single ones, mostly symbols leaking
  from one item into another (TSLA on a Toyota recall, DAX members on an ECB decision), and a batch
  blocks an urgent review for 5-10 s. Not reliable enough for a 1.3x gain.
* An invalid news answer is asked once more with a temperature 0.1 higher than configured (the same
  near-deterministic sampling would mostly repeat the same answer). Reviews and chart analyses are not
  retried: a review is time-critical, a chart analysis is routine and simply comes again in the next scan.
"""

import asyncio
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Real
from typing import Any

from trading_local.clock import Clock
from trading_local.config import Settings
from trading_local.contracts import TradeReview, TradeReviewContext
from trading_local.db import repo_ops
from trading_local.db.database import Database
from trading_local.dto import LlmStatusOut
from trading_local.llm import prompts, schemas
from trading_local.llm.client import ChatClient, ChatResult, LlmError
from trading_local.llm.jobs import JobExpired, JobNotRun, Priority, PriorityJobQueue
from trading_local.llm.prompts import Prompt
from trading_local.llm.schemas import OutputError
from trading_local.models import ChartAnalysis, Instrument, NewsAnalysis, NewsItem

log = logging.getLogger(__name__)

NEWS_MAX_WAIT_S = 20 * 60.0  # a news item still waiting after 20 minutes is stale
CHART_MAX_WAIT_S = 30 * 60.0
HEALTH_INTERVAL_S = 30.0
STATS_WINDOW_S = 3600
# Rows read from llm_calls at start to restore the status window (one hour holds at most ~2,500 calls).
RESTORE_CALLS_LIMIT = 5000
CANCELLED_DE = "Abgebrochen vor der Antwort"  # llm_calls.error of a call whose caller stopped waiting
CONVERSION_FAILED_DE = "Antwort konnte nicht verarbeitet werden"
RETRY_TEMPERATURE_STEP = 0.1  # added to settings.ollama_temperature for the retry of a rejected answer
WARMUP_ATTEMPTS = 3  # start-up model load: attempts while Ollama is not reachable
WARMUP_RETRY_S = 5.0
# Key of the last price in the chart facts ('Kurs' is what analysis.readings.chart_facts writes).
LAST_PRICE_KEYS: tuple[str, ...] = ("Kurs", "last_price")


@dataclass(frozen=True, slots=True)
class JobSpec:
    """Model settings of one job type. ``call_timeout_s`` bounds one model call (~5-10x the measured p90)."""

    kind: str  # llm_calls.kind
    priority: Priority
    think: bool
    num_predict: int  # output token cap (measured: news 62-111, review 72-207, chart 120-215)
    call_timeout_s: float


NEWS_JOB = JobSpec("news", Priority.NEWS, think=False, num_predict=400, call_timeout_s=20.0)
REVIEW_JOB = JobSpec("trade_review", Priority.TRADE_REVIEW, think=False, num_predict=450, call_timeout_s=60.0)
CHART_JOB = JobSpec("chart", Priority.CHART, think=False, num_predict=450, call_timeout_s=30.0)


@dataclass(frozen=True, slots=True)
class _CallRecord:
    ts: int
    ok: bool
    error: str


class OllamaLlm:
    """``contracts.Llm`` on top of one priority queue and one model client.

    With ``settings.enable_llm`` False every method returns at once (None / verdict 'unavailable') and
    nothing is started. One ``start`` / ``stop`` cycle per instance.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        clock: Clock,
        client: ChatClient,
        news_max_wait_s: float = NEWS_MAX_WAIT_S,
        chart_max_wait_s: float = CHART_MAX_WAIT_S,
        health_interval_s: float = HEALTH_INTERVAL_S,
    ) -> None:
        self._db = db
        self._clock = clock
        self._client = client
        self._model = settings.ollama_model
        self._enabled = settings.enable_llm
        self._retry_options = {"temperature": round(settings.ollama_temperature + RETRY_TEMPERATURE_STEP, 3)}
        self._news_max_wait_s = news_max_wait_s
        self._chart_max_wait_s = chart_max_wait_s
        self._health_interval_s = health_interval_s
        self._queue = PriorityJobQueue()
        self._log_writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-calls-log")
        self._lock = threading.Lock()  # guards the call window and the last latency (status() runs in API threads)
        self._calls: deque[_CallRecord] = deque()
        self._last_latency_ms: int | None = None
        self._available = False
        self._last_success = -math.inf  # time.monotonic() of the newest successful model call
        self._health_task: asyncio.Task[None] | None = None
        self._started = False

    # -- lifecycle ------------------------------------------------------------------------------
    async def start(self) -> None:
        """Restore the status window from ``llm_calls`` (one short query off the event loop), then start the
        queue worker (which first loads the model) and the periodic health check without waiting for them."""
        if not self._enabled or self._started:
            return
        self._started = True
        await asyncio.to_thread(self._restore_recent_calls)
        self._queue.start(warmup=self._warm_up)
        self._health_task = asyncio.create_task(self._health_loop(), name="llm-health")

    async def stop(self) -> None:
        """Cancel waiting and running jobs (their callers get None / 'unavailable') and stop all background work."""
        if not self._enabled:
            return
        health, self._health_task = self._health_task, None
        if health is not None:
            health.cancel()
            await asyncio.gather(health, return_exceptions=True)
        await self._queue.stop()
        await self._client.close()
        await asyncio.to_thread(self._log_writer.shutdown, wait=True)

    def status(self) -> LlmStatusOut:
        metrics = self._queue.metrics()
        now_ts = self._clock.ts()
        with self._lock:
            self._prune(now_ts)
            calls = list(self._calls)
            last_latency = self._last_latency_ms
        errors = [call for call in calls if not call.ok]
        return LlmStatusOut(
            model=self._model,
            enabled=self._enabled,
            available=self._enabled and self._available,
            busy=metrics.busy,
            queue_depth=metrics.depth,
            last_latency_ms=last_latency,
            calls_last_hour=len(calls),
            errors_last_hour=len(errors),
            last_error=errors[-1].error if errors else "",
        )

    # -- the three job types --------------------------------------------------------------------
    async def analyze_news(self, item: NewsItem, candidates: Sequence[Instrument]) -> NewsAnalysis | None:
        """Classify one news item (see ``contracts.Llm``). None when disabled, stale, failed or stopped."""
        if not self._enabled:
            return None
        news_id = item.id
        if news_id is None:
            log.warning("news item %s has no database id; not analysed", item.uid)
            return None
        try:
            allowed = prompts.select_candidates(candidates)
            symbols = [instrument.symbol for instrument in allowed]
            prompt = prompts.news_prompt(item, allowed)
            schema = schemas.news_schema(symbols)

            def convert(result: ChatResult) -> NewsAnalysis:
                return schemas.to_news_analysis(
                    result.data,
                    candidates=symbols,
                    news_id=news_id,
                    analyzed_ts=self._clock.ts(),
                    model=self._model,
                    latency_ms=result.latency_ms,
                )

            async def work(queue_wait_ms: int) -> NewsAnalysis | None:
                return await self._news_attempts(news_id, prompt, schema, convert, queue_wait_ms)

            return await self._queue.run(NEWS_JOB.priority, work, max_wait_s=self._news_max_wait_s)
        except JobNotRun as exc:
            log.info("news item %d not analysed: %s", news_id, exc)
        except Exception:
            log.exception("news item %d: analysis failed", news_id)
        return None

    async def review_trade(self, ctx: TradeReviewContext, *, timeout_s: float) -> TradeReview:
        """Review a trade candidate within ``timeout_s`` seconds (queue wait plus generation).

        ``latency_ms`` of the result is the time from the request to the verdict. Never raises (except
        cancellation of the caller); verdict 'unavailable' on timeout, error, stop or when disabled, and at
        once for a ``timeout_s`` that is not positive (or NaN, which as a timer would crash the event loop).
        """
        if not self._enabled:
            return TradeReview(verdict="unavailable")
        if not timeout_s > 0:
            log.warning("trade review of %s not started: invalid timeout %r", ctx.instrument.symbol, timeout_s)
            return TradeReview(verdict="unavailable", latency_ms=0)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + timeout_s
        symbol = ctx.instrument.symbol
        review: TradeReview | None = None
        try:
            prompt = prompts.trade_review_prompt(ctx)
            images = (ctx.chart_png,) if ctx.chart_png else ()

            def convert(result: ChatResult) -> TradeReview:
                return schemas.to_trade_review(result.data, latency_ms=result.latency_ms)

            async def work(queue_wait_ms: int) -> TradeReview:
                budget = min(REVIEW_JOB.call_timeout_s, deadline - loop.time())
                if budget <= 0:  # taken in the loop iteration of its deadline, before the expiry fired
                    raise JobExpired("the review deadline passed before the model was free")
                return await self._attempt(
                    REVIEW_JOB, prompt, schemas.trade_review_schema(), convert,
                    images=images, timeout_s=budget, queue_wait_ms=queue_wait_ms,
                )

            async with asyncio.timeout_at(deadline):
                review = await self._queue.run(REVIEW_JOB.priority, work, max_wait_s=timeout_s)
        except TimeoutError:
            log.info("trade review of %s timed out after %.0f s", symbol, timeout_s)
        except (JobNotRun, LlmError) as exc:
            log.info("trade review of %s unavailable: %s", symbol, exc)
        except Exception:
            log.exception("trade review of %s failed", symbol)
        elapsed_ms = round((loop.time() - started) * 1000)
        if review is None:
            return TradeReview(verdict="unavailable", latency_ms=elapsed_ms)
        review.latency_ms = elapsed_ms
        return review

    async def analyze_chart(
        self, instrument: Instrument, timeframe: str, chart_png: bytes, facts: Mapping[str, Any]
    ) -> ChartAnalysis | None:
        """Vision analysis of a chart (lowest priority). Support / resistance are checked against the last
        price in ``facts`` (key 'Kurs'). None when disabled, stale, failed or stopped."""
        if not self._enabled:
            return None
        symbol = instrument.symbol
        try:
            last_price = _last_price(facts)
            prompt = prompts.chart_prompt(
                instrument, timeframe, facts, last_price=last_price, has_image=bool(chart_png)
            )
            images = (chart_png,) if chart_png else ()

            def convert(result: ChatResult) -> ChartAnalysis:
                return schemas.to_chart_analysis(
                    result.data,
                    symbol=symbol,
                    timeframe=timeframe,
                    ts=self._clock.ts(),
                    model=self._model,
                    last_price=last_price,
                    latency_ms=result.latency_ms,
                )

            async def work(queue_wait_ms: int) -> ChartAnalysis:
                return await self._attempt(
                    CHART_JOB, prompt, schemas.chart_schema(), convert,
                    images=images, timeout_s=CHART_JOB.call_timeout_s, queue_wait_ms=queue_wait_ms,
                )

            return await self._queue.run(CHART_JOB.priority, work, max_wait_s=self._chart_max_wait_s)
        except (JobNotRun, LlmError) as exc:
            log.info("chart analysis of %s %s unavailable: %s", symbol, timeframe, exc)
        except Exception:
            log.exception("chart analysis of %s %s failed", symbol, timeframe)
        return None

    # -- model calls ----------------------------------------------------------------------------
    async def _news_attempts(
        self,
        news_id: int,
        prompt: Prompt,
        schema: dict[str, Any],
        convert: Callable[[ChatResult], NewsAnalysis],
        queue_wait_ms: int,
    ) -> NewsAnalysis | None:
        """One call, plus one retry with changed sampling if the answer was invalid."""
        timeout_s = NEWS_JOB.call_timeout_s
        try:
            return await self._attempt(
                NEWS_JOB, prompt, schema, convert, timeout_s=timeout_s, queue_wait_ms=queue_wait_ms
            )
        except LlmError as exc:
            if not exc.output_invalid:
                log.info("news item %d not analysed: %s", news_id, exc.reason)
                return None
            log.info("news item %d: invalid answer (%s), asking once more", news_id, exc.reason)
        try:
            return await self._attempt(
                NEWS_JOB, prompt, schema, convert, timeout_s=timeout_s, queue_wait_ms=0, retry=True
            )
        except LlmError as exc:
            log.info("news item %d not analysed after a retry: %s", news_id, exc.reason)
            return None

    async def _attempt[T](
        self,
        spec: JobSpec,
        prompt: Prompt,
        schema: dict[str, Any],
        convert: Callable[[ChatResult], T],
        *,
        timeout_s: float,
        queue_wait_ms: int,
        images: Sequence[bytes] = (),
        retry: bool = False,
    ) -> T:
        """One model call, converted with ``convert`` and recorded exactly once (success, error or
        cancellation). Raises ``LlmError`` (an unusable answer has ``output_invalid=True``)."""
        options = {"num_predict": spec.num_predict, **(self._retry_options if retry else {})}
        started = time.perf_counter()
        try:
            result = await self._client.chat_json(
                system=prompt.system,
                user=prompt.user,
                schema=schema,
                images=images,
                think=spec.think,
                timeout_s=timeout_s,
                options=options,
            )
        except LlmError as exc:
            if exc.unreachable:
                self._available = False
            self._record(
                spec.kind, False, exc.latency_ms, queue_wait_ms, exc.prompt_tokens, exc.output_tokens, exc.reason_de
            )
            raise
        except asyncio.CancelledError:
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            self._record(spec.kind, False, elapsed_ms, queue_wait_ms, None, None, CANCELLED_DE)
            raise
        self._available = True
        self._last_success = time.monotonic()
        try:
            value = convert(result)
        except Exception as exc:
            if isinstance(exc, OutputError):
                reason, reason_de, retryable = str(exc), exc.reason_de, True
            else:  # a defect in the conversion: asking again would not help
                log.exception("%s answer could not be converted", spec.kind)
                reason, reason_de, retryable = f"conversion failed: {type(exc).__name__}", CONVERSION_FAILED_DE, False
            self._record(
                spec.kind, False, result.latency_ms, queue_wait_ms, result.prompt_tokens, result.output_tokens,
                reason_de,
            )
            raise LlmError(
                reason,
                reason_de=reason_de,
                output_invalid=retryable,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,
            ) from None
        self._record(spec.kind, True, result.latency_ms, queue_wait_ms, result.prompt_tokens, result.output_tokens, "")
        return value

    # -- bookkeeping ----------------------------------------------------------------------------
    def _record(
        self,
        kind: str,
        ok: bool,
        latency_ms: int,
        queue_wait_ms: int,
        prompt_tokens: int | None,
        output_tokens: int | None,
        error: str,
    ) -> None:
        """Count the call for ``status()`` and hand the ``llm_calls`` row to the writer thread."""
        ts = self._clock.ts()
        with self._lock:
            self._calls.append(_CallRecord(ts, ok, error))
            self._prune(ts)
            if ok:
                self._last_latency_ms = latency_ms
        row = {
            "ts": ts,
            "kind": kind,
            "ok": ok,
            "latency_ms": latency_ms,
            "queue_wait_ms": queue_wait_ms,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "error": error,
        }
        try:
            self._log_writer.submit(self._write_call, row)
        except RuntimeError:  # the writer was shut down (stop)
            log.debug("llm call not logged, the service is stopped: %s", row)

    def _write_call(self, row: dict[str, Any]) -> None:
        try:
            repo_ops.log_llm_call(self._db, **row)
        except Exception:
            log.exception("could not log an llm call")

    def _restore_recent_calls(self) -> None:
        """Fill the status window and the last latency from the ``llm_calls`` rows of the last hour, so that
        ``status()`` survives a restart. Runs in a worker thread before any call is made; errors are logged."""
        since_ts = self._clock.ts() - STATS_WINDOW_S
        try:
            rows = repo_ops.list_llm_calls(self._db, limit=RESTORE_CALLS_LIMIT)  # newest first
        except Exception:
            log.exception("could not read the recent llm calls")
            return
        recent = [row for row in reversed(rows) if row.ts >= since_ts]
        with self._lock:
            self._calls.extend(_CallRecord(row.ts, row.ok, row.error) for row in recent)
            self._last_latency_ms = next((row.latency_ms for row in reversed(recent) if row.ok), None)

    def _prune(self, now_ts: int) -> None:
        """Drop calls older than the status window (caller holds the lock)."""
        cutoff = now_ts - STATS_WINDOW_S
        while self._calls and self._calls[0].ts < cutoff:
            self._calls.popleft()

    async def _warm_up(self) -> None:
        """Load the model at start-up. A connection failure is retried a few times: right after start-up the
        process builds its trading calendars (CPU-heavy threads), which can starve the connect of its timeout."""
        for attempt in range(1, WARMUP_ATTEMPTS + 1):
            try:
                duration_ms = await self._client.warm_up()
            except LlmError as exc:
                log.warning(
                    "LLM warm-up attempt %d/%d failed after %s ms: %s",
                    attempt,
                    WARMUP_ATTEMPTS,
                    exc.latency_ms,
                    exc.reason,
                )
                if exc.unreachable and attempt < WARMUP_ATTEMPTS:
                    await asyncio.sleep(WARMUP_RETRY_S)
                    continue
                if exc.unreachable:
                    self._available = False
                return
            log.info("LLM model %s loaded in %d ms", self._model, duration_ms)
            self._available = True
            self._last_success = time.monotonic()
            return

    async def _health_loop(self) -> None:
        """Refresh ``available`` at most every ``health_interval_s`` (skipped while calls succeed anyway)."""
        while True:
            if time.monotonic() - self._last_success >= self._health_interval_s:
                try:
                    self._available = await self._client.health()
                except Exception:
                    log.exception("LLM health check failed")
                    self._available = False
            await asyncio.sleep(self._health_interval_s)


def _last_price(facts: Mapping[str, Any]) -> float | None:
    """The last price from the chart facts (positive and finite), None if the facts do not contain it."""
    for key in LAST_PRICE_KEYS:
        value = facts.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            return float(value)
    return None
