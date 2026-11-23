"""Thin async wrapper around ``ollama.AsyncClient`` for schema-constrained JSON chat calls.

Production settings (measured by the LLM probe on this machine: qwen3.5:9b, RTX 2080 SUPER 8 GB, Ollama
with ``OLLAMA_NUM_PARALLEL=1``):

* the load-affecting options ``settings.ollama_num_ctx`` (and ``settings.ollama_num_gpu`` if set) are sent
  IDENTICALLY with every request, warm-up included; callers cannot override them. Any change makes Ollama
  reload the model (~5 s). The configured context should equal the server's ``OLLAMA_CONTEXT_LENGTH`` so
  that other programs using the model do not force reloads either. At ``num_ctx`` 4096 Ollama keeps all 33
  layers on the 8 GB GPU by itself; from 8192 on it moves layers to the CPU (45 instead of 60 tokens/s)
  unless ``num_gpu`` is 99.
* near-deterministic decoding for structured output: ``settings.ollama_temperature`` (low) and
  ``presence_penalty`` 0. The model's defaults (temperature 1, presence penalty 1.5) tripled the
  classification errors and produced wrong symbols and empty strings.
* an answer is accepted only if ``done_reason == 'stop'`` (anything else is truncated JSON), the prompt was
  not truncated by Ollama (it silently keeps only the END of an over-long prompt, i.e. drops the system
  prompt; the tell-tale sign is ``prompt_eval_count == num_ctx // 2 + 2``), the content is a JSON object and
  it matches the request schema.
* cancelling an awaiting call (``asyncio`` timeout or task cancellation) closes the HTTP connection and
  Ollama stops generating within ~0.1 s (verified for non-streaming requests and for requests that were
  still queued inside Ollama), so a timed-out call does not keep the GPU busy.
"""

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import ollama

from trading_local.config import Settings

log = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 3.0
WRITE_TIMEOUT_S = 10.0
HEALTH_TIMEOUT_S = 3.0
LOAD_TIMEOUT_S = 120.0  # model load: 4-8 s with a warm file cache, ~35 s after a reboot
RELOAD_WARNING_S = 1.0  # load_duration of a call is ~0.005 s while the model stays loaded
_ERROR_TEXT_CHARS = 160


class LlmError(Exception):
    """A model call failed. ``reason`` is a short English description for the logs, ``reason_de`` the same
    in German for the user (it is stored in ``llm_calls.error`` and shown as the last error in the GUI).

    ``output_invalid`` is True when the server answered but the answer is unusable (truncated, not JSON,
    schema violation): asking again with different sampling may help. It is False when the model could
    not be asked at all or did not finish in time (connection refused, model missing, timeout, server error).
    ``unreachable`` marks errors that mean the server or the model is not available.
    """

    def __init__(
        self,
        reason: str,
        *,
        reason_de: str,
        output_invalid: bool = False,
        unreachable: bool = False,
        latency_ms: int = 0,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.reason_de = reason_de
        self.output_invalid = output_invalid
        self.unreachable = unreachable
        self.latency_ms = latency_ms
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens


@dataclass(frozen=True, slots=True)
class ChatResult:
    """A schema-valid JSON answer. ``latency_ms`` is the wall time of the HTTP call."""

    data: dict[str, Any]
    prompt_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class ChatClient(Protocol):
    """What the LLM service needs from a model client (implemented by ``OllamaClient`` and test fakes)."""

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: Sequence[bytes] = (),
        think: bool,
        timeout_s: float,
        options: Mapping[str, Any] | None = None,
    ) -> ChatResult: ...

    async def health(self) -> bool: ...

    async def warm_up(self) -> int: ...

    async def close(self) -> None: ...


class OllamaClient:
    """One ``ollama.AsyncClient`` with the production settings. Every public coroutine maps all failures to
    ``LlmError`` (``health`` returns False instead); ``asyncio.CancelledError`` always propagates."""

    def __init__(self, settings: Settings) -> None:
        self._model = settings.ollama_model
        self._keep_alive = settings.ollama_keep_alive
        self._num_ctx = settings.ollama_num_ctx
        self._load_options: dict[str, Any] = {"num_ctx": settings.ollama_num_ctx}
        if settings.ollama_num_gpu is not None:
            self._load_options["num_gpu"] = settings.ollama_num_gpu
        self._sampling: dict[str, Any] = {"temperature": settings.ollama_temperature, "presence_penalty": 0.0}
        # No httpx read timeout: every call runs inside an asyncio timeout chosen per call.
        timeout = httpx.Timeout(connect=CONNECT_TIMEOUT_S, read=None, write=WRITE_TIMEOUT_S, pool=CONNECT_TIMEOUT_S)
        self._client = ollama.AsyncClient(host=settings.ollama_host or None, timeout=timeout)

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        images: Sequence[bytes] = (),
        think: bool,
        timeout_s: float,
        options: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        """One non-streaming chat call whose answer is constrained by the JSON ``schema``.

        ``options`` (e.g. ``num_predict``, a retry temperature) are merged over the sampling defaults; the
        load-affecting options are always the shared configured values. ``timeout_s`` bounds the whole
        call (also time spent waiting inside Ollama). Raises ``LlmError`` on every failure.
        """
        user_message: dict[str, Any] = {"role": "user", "content": user}
        if images:
            user_message["images"] = list(images)
        messages = [{"role": "system", "content": system}, user_message]
        merged = {**self._sampling, **(options or {}), **self._load_options}
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout_s):
                response = await self._client.chat(
                    model=self._model,
                    messages=messages,
                    think=think,
                    format=schema,
                    options=merged,
                    keep_alive=self._keep_alive,
                )
        except TimeoutError:
            raise LlmError(
                f"timeout after {timeout_s:g} s",
                reason_de=f"Zeitüberschreitung nach {timeout_s:g} s",
                latency_ms=_elapsed_ms(started),
            ) from None
        except Exception as exc:
            raise self._call_error(exc, started) from exc
        return self._result(response, schema, _elapsed_ms(started))

    async def health(self) -> bool:
        """True if the server answers within a few seconds and the configured model is installed."""
        wanted = _with_tag(self._model)
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT_S):
                listing = await self._client.list()
        except Exception as exc:
            log.debug("Ollama health check failed: %s", _describe(exc))
            return False
        return any(_with_tag(entry.model or "") == wanted for entry in listing.models)

    async def warm_up(self) -> int:
        """Load the model with the production options (no prompt) and return the duration in milliseconds."""
        started = time.perf_counter()
        try:
            async with asyncio.timeout(LOAD_TIMEOUT_S):
                await self._client.generate(model=self._model, options=self._load_options, keep_alive=self._keep_alive)
        except TimeoutError:
            raise LlmError(
                f"model load took longer than {LOAD_TIMEOUT_S:.0f} s",
                reason_de=f"Laden des Modells dauerte länger als {LOAD_TIMEOUT_S:.0f} s",
                latency_ms=_elapsed_ms(started),
            ) from None
        except Exception as exc:
            raise self._call_error(exc, started) from exc
        return _elapsed_ms(started)

    async def close(self) -> None:
        """Close the HTTP connection pool (errors while closing are only logged)."""
        try:
            await self._client.close()
        except Exception as exc:
            log.debug("closing the Ollama client failed: %s", _describe(exc))

    # -- internals ------------------------------------------------------------------------------
    def _call_error(self, exc: Exception, started: float) -> LlmError:
        """Translate a library / transport exception into an ``LlmError``."""
        latency_ms = _elapsed_ms(started)
        match exc:
            case ConnectionError() | httpx.ConnectError() | httpx.ConnectTimeout():
                return LlmError(
                    "Ollama not reachable", reason_de="Ollama nicht erreichbar", unreachable=True, latency_ms=latency_ms
                )
            case ollama.ResponseError(status_code=404):
                return LlmError(
                    f"model {self._model} not installed",
                    reason_de=f"Modell {self._model} ist nicht installiert",
                    unreachable=True,
                    latency_ms=latency_ms,
                )
            case ollama.ResponseError():
                detail = _clip(exc.error)
                return LlmError(
                    f"server error {exc.status_code}: {detail}",
                    reason_de=f"Serverfehler {exc.status_code}: {detail}",
                    latency_ms=latency_ms,
                )
            case httpx.TransportError():
                name = type(exc).__name__
                return LlmError(
                    f"transport error: {name}", reason_de=f"Verbindungsfehler ({name})", latency_ms=latency_ms
                )
            case _:
                log.warning("unexpected error from the Ollama client", exc_info=exc)
                return LlmError(
                    f"unexpected error: {_describe(exc)}",
                    reason_de=f"Unerwarteter Fehler ({type(exc).__name__})",
                    latency_ms=latency_ms,
                )

    def _result(self, response: ollama.ChatResponse, schema: Mapping[str, Any], latency_ms: int) -> ChatResult:
        """Check a finished response (done reason, prompt truncation, JSON, schema) and wrap its data."""
        prompt_tokens, output_tokens = response.prompt_eval_count, response.eval_count
        load_s = (response.load_duration or 0) / 1e9
        if load_s > RELOAD_WARNING_S:
            log.warning(
                "Ollama (re)loaded the model during a call (%.1f s); another client may use other load options", load_s
            )

        def invalid(reason: str, reason_de: str) -> LlmError:
            return LlmError(
                reason,
                reason_de=reason_de,
                output_invalid=True,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )

        if prompt_tokens == self._num_ctx // 2 + 2:
            # Not marked output_invalid: asking again would truncate the same prompt again.
            raise LlmError(
                "prompt truncated by Ollama (too long)",
                reason_de="Eingabe zu lang (von Ollama gekürzt)",
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
            )
        if response.done_reason != "stop":
            reason = f"done_reason={response.done_reason}"
            raise invalid(f"output incomplete ({reason})", f"Antwort unvollständig ({reason})")
        try:
            data = json.loads(response.message.content or "")
        except ValueError:
            raise invalid("answer is not valid JSON", "Antwort ist kein gültiges JSON") from None
        problem = schema_violation(data, schema)
        if problem:
            raise invalid(f"schema violation: {problem}", "Antwort entspricht nicht dem Schema")
        return ChatResult(data=data, prompt_tokens=prompt_tokens, output_tokens=output_tokens, latency_ms=latency_ms)


def schema_violation(value: Any, schema: Mapping[str, Any], path: str = "$") -> str:
    """First violation of ``value`` against ``schema`` as '<path>: <problem>', or '' if it conforms.

    Supports the keywords the request schemas of this package use: ``type`` (one type or a list), ``enum``,
    ``properties`` / ``required``, ``items``, ``minItems`` / ``maxItems`` and ``minimum`` / ``maximum``.
    """
    expected = schema.get("type")
    if expected is not None:
        types = [expected] if isinstance(expected, str) else list(expected)
        if not any(_is_type(value, name) for name in types):
            return f"{path}: expected {'/'.join(types)}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: value not allowed"
    if isinstance(value, dict):
        for key in schema.get("required", ()):
            if key not in value:
                return f"{path}: missing '{key}'"
        for key, sub_schema in schema.get("properties", {}).items():
            if key in value:
                problem = schema_violation(value[key], sub_schema, f"{path}.{key}")
                if problem:
                    return problem
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            return f"{path}: too few items"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return f"{path}: too many items"
        if "items" in schema:
            for index, item in enumerate(value):
                problem = schema_violation(item, schema["items"], f"{path}[{index}]")
                if problem:
                    return problem
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: below minimum"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: above maximum"
    return ""


def _is_type(value: Any, name: str) -> bool:
    match name:
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "string":
            return isinstance(value, str)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        case "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case _:
            return False


def _with_tag(model: str) -> str:
    """Ollama model names without a tag mean ':latest'."""
    return model if ":" in model else f"{model}:latest"


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _clip(text: object) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= _ERROR_TEXT_CHARS else flat[: _ERROR_TEXT_CHARS - 1] + "…"


def _describe(exc: BaseException) -> str:
    return _clip(f"{type(exc).__name__}: {exc}")
