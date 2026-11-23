"""OllamaClient: production options, response checks and the mapping of every failure to LlmError."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import ollama
import pytest

from trading_local.config import Settings
from trading_local.llm.client import LlmError, OllamaClient, schema_violation
from trading_local.testing import make_settings

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["a", "b"]}, "score": {"type": "number"}},
    "required": ["label", "score"],
}


class StubAsyncClient:
    """Replaces ``ollama.AsyncClient`` inside OllamaClient: records arguments, returns or raises as told."""

    def __init__(self, *, response: Any = None, error: BaseException | None = None, delay_s: float = 0.0) -> None:
        self.response = response
        self.error = error
        self.delay_s = delay_s
        self.chat_kwargs: dict[str, Any] = {}
        self.generate_kwargs: dict[str, Any] = {}
        self.cancelled = False
        self.closed = False

    async def _answer(self) -> Any:
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.error is not None:
            raise self.error
        return self.response

    async def chat(self, **kwargs: Any) -> Any:
        self.chat_kwargs = kwargs
        return await self._answer()

    async def generate(self, **kwargs: Any) -> Any:
        self.generate_kwargs = kwargs
        return await self._answer()

    async def list(self) -> Any:
        return await self._answer()

    async def close(self) -> None:
        self.closed = True


def _response(
    content: str = '{"label": "a", "score": 0.5}', *, done_reason: str = "stop", prompt_tokens: int = 812
) -> ollama.ChatResponse:
    return ollama.ChatResponse(
        message=ollama.Message(role="assistant", content=content),
        done_reason=done_reason,
        prompt_eval_count=prompt_tokens,
        eval_count=42,
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(
        tmp_path, ollama_num_ctx=4096, ollama_num_gpu=None, ollama_temperature=0.1, ollama_keep_alive="30m",
        ollama_model="qwen3.5:9b",
    )


def _client(settings: Settings, stub: StubAsyncClient) -> OllamaClient:
    client = OllamaClient(settings)
    client._client = stub  # type: ignore[assignment]
    return client


async def _chat(client: OllamaClient, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {"system": "sys", "user": "usr", "schema": SCHEMA, "think": False, "timeout_s": 5.0}
    kwargs.update(overrides)
    return await client.chat_json(**kwargs)


# -- successful calls -----------------------------------------------------------------------------
async def test_chat_json_sends_the_production_settings_and_returns_the_data(settings: Settings) -> None:
    stub = StubAsyncClient(response=_response())
    result = await _chat(_client(settings, stub), images=[b"png-bytes"], options={"num_predict": 300})

    assert result.data == {"label": "a", "score": 0.5}
    assert (result.prompt_tokens, result.output_tokens) == (812, 42)
    assert result.latency_ms >= 0
    sent = stub.chat_kwargs
    assert sent["model"] == "qwen3.5:9b" and sent["think"] is False and sent["keep_alive"] == "30m"
    assert sent["format"] == SCHEMA
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr", "images": [b"png-bytes"]},
    ]
    assert sent["options"] == {"temperature": 0.1, "presence_penalty": 0.0, "num_predict": 300, "num_ctx": 4096}


async def test_num_gpu_is_sent_only_when_configured(tmp_path: Path) -> None:
    stub = StubAsyncClient(response=_response())
    await _chat(_client(make_settings(tmp_path, ollama_num_ctx=8192, ollama_num_gpu=99), stub))
    assert (stub.chat_kwargs["options"]["num_ctx"], stub.chat_kwargs["options"]["num_gpu"]) == (8192, 99)


async def test_callers_cannot_change_load_affecting_options(tmp_path: Path) -> None:
    stub = StubAsyncClient(response=_response())
    client = _client(make_settings(tmp_path, ollama_num_ctx=4096, ollama_num_gpu=99), stub)
    await _chat(client, options={"num_ctx": 2048, "num_gpu": 10, "temperature": 0.2})
    options = stub.chat_kwargs["options"]
    assert (options["num_ctx"], options["num_gpu"]) == (4096, 99)  # a different value would reload the model
    assert options["temperature"] == 0.2


async def test_a_model_reload_during_a_call_is_logged(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    response = _response()
    response.load_duration = 4_800_000_000  # nanoseconds
    await _chat(_client(settings, StubAsyncClient(response=response)))
    assert "loaded the model during a call (4.8 s)" in caplog.text


async def test_messages_without_images_have_no_images_key(settings: Settings) -> None:
    stub = StubAsyncClient(response=_response())
    await _chat(_client(settings, stub))
    assert "images" not in stub.chat_kwargs["messages"][1]


# -- failures -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "reason", "reason_de", "unreachable"),
    [
        (ConnectionError("Failed to connect to Ollama"), "Ollama not reachable", "Ollama nicht erreichbar", True),
        (httpx.ConnectTimeout("connect timed out"), "Ollama not reachable", "Ollama nicht erreichbar", True),
        (
            ollama.ResponseError("model 'qwen3.5:9b' not found", 404),
            "model qwen3.5:9b not installed",
            "Modell qwen3.5:9b ist nicht installiert",
            True,
        ),
        (
            ollama.ResponseError("repeat limit reached", 500),
            "server error 500: repeat limit reached",
            "Serverfehler 500: repeat limit reached",
            False,
        ),
        (httpx.ReadError("connection reset"), "transport error: ReadError", "Verbindungsfehler (ReadError)", False),
        (
            RuntimeError("surprise"),
            "unexpected error: RuntimeError: surprise",
            "Unerwarteter Fehler (RuntimeError)",
            False,
        ),
    ],
)
async def test_transport_and_server_failures_become_llm_errors(
    settings: Settings, error: BaseException, reason: str, reason_de: str, unreachable: bool
) -> None:
    with pytest.raises(LlmError) as caught:
        await _chat(_client(settings, StubAsyncClient(error=error)))
    assert caught.value.reason == reason
    assert caught.value.reason_de == reason_de
    assert caught.value.unreachable is unreachable
    assert caught.value.output_invalid is False


async def test_timeout_cancels_the_request(settings: Settings) -> None:
    stub = StubAsyncClient(response=_response(), delay_s=1.0)
    with pytest.raises(LlmError) as caught:
        await _chat(_client(settings, stub), timeout_s=0.05)
    assert caught.value.reason == "timeout after 0.05 s"
    assert caught.value.reason_de == "Zeitüberschreitung nach 0.05 s"
    assert caught.value.output_invalid is False
    assert stub.cancelled  # the HTTP request was abandoned, which makes Ollama stop generating


@pytest.mark.parametrize(
    ("response", "reason", "reason_de"),
    [
        (
            _response(done_reason="length"),
            "output incomplete (done_reason=length)",
            "Antwort unvollständig (done_reason=length)",
        ),
        (_response('{"label": "a", "score": '), "answer is not valid JSON", "Antwort ist kein gültiges JSON"),
        (_response(""), "answer is not valid JSON", "Antwort ist kein gültiges JSON"),
        (
            _response('{"label": "c", "score": 1}'),
            "schema violation: $.label: value not allowed",
            "Antwort entspricht nicht dem Schema",
        ),
        (_response('{"label": "a"}'), "schema violation: $: missing 'score'", "Antwort entspricht nicht dem Schema"),
        (_response('["a"]'), "schema violation: $: expected object", "Antwort entspricht nicht dem Schema"),
    ],
)
async def test_unusable_answers_are_output_invalid(
    settings: Settings, response: ollama.ChatResponse, reason: str, reason_de: str
) -> None:
    with pytest.raises(LlmError) as caught:
        await _chat(_client(settings, StubAsyncClient(response=response)))
    assert caught.value.reason == reason
    assert caught.value.reason_de == reason_de
    assert caught.value.output_invalid is True
    assert (caught.value.prompt_tokens, caught.value.output_tokens) == (812, 42)


async def test_a_prompt_truncated_by_ollama_is_rejected_and_not_retryable(settings: Settings) -> None:
    stub = StubAsyncClient(response=_response(prompt_tokens=4096 // 2 + 2))
    with pytest.raises(LlmError) as caught:
        await _chat(_client(settings, stub))
    assert caught.value.reason == "prompt truncated by Ollama (too long)"
    assert caught.value.reason_de == "Eingabe zu lang (von Ollama gekürzt)"
    assert caught.value.output_invalid is False


async def test_warm_up_timeout_is_an_llm_error(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trading_local.llm.client.LOAD_TIMEOUT_S", 0.05)
    stub = StubAsyncClient(response=ollama.GenerateResponse(done_reason="load"), delay_s=1.0)
    with pytest.raises(LlmError) as caught:
        await _client(settings, stub).warm_up()
    assert caught.value.reason_de.startswith("Laden des Modells dauerte länger als")
    assert stub.cancelled


async def test_cancellation_propagates(settings: Settings) -> None:
    stub = StubAsyncClient(response=_response(), delay_s=1.0)
    task = asyncio.create_task(_chat(_client(settings, stub)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stub.cancelled


# -- health, warm-up, close -----------------------------------------------------------------------
def _listing(*names: str) -> ollama.ListResponse:
    return ollama.ListResponse(models=[ollama.ListResponse.Model(model=name) for name in names])


@pytest.mark.parametrize(
    ("stub", "expected"),
    [
        (StubAsyncClient(response=_listing("llama3:8b", "qwen3.5:9b")), True),
        (StubAsyncClient(response=_listing("llama3:8b")), False),
        (StubAsyncClient(error=ConnectionError("down")), False),
        (StubAsyncClient(response=_listing("qwen3.5:9b"), delay_s=5.0), False),  # no answer within 3 s
    ],
)
async def test_health(
    settings: Settings, stub: StubAsyncClient, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("trading_local.llm.client.HEALTH_TIMEOUT_S", 0.05)
    assert await _client(settings, stub).health() is expected


async def test_health_resolves_the_latest_tag(tmp_path: Path) -> None:
    stub = StubAsyncClient(response=_listing("mistral:latest"))
    client = _client(make_settings(tmp_path, ollama_model="mistral"), stub)
    assert await client.health() is True


async def test_warm_up_loads_the_model_with_the_same_load_options(settings: Settings) -> None:
    stub = StubAsyncClient(response=ollama.GenerateResponse(done_reason="load"))
    duration_ms = await _client(settings, stub).warm_up()
    assert duration_ms >= 0
    assert stub.generate_kwargs == {"model": "qwen3.5:9b", "options": {"num_ctx": 4096}, "keep_alive": "30m"}


async def test_warm_up_failure_is_an_llm_error(settings: Settings) -> None:
    with pytest.raises(LlmError) as caught:
        await _client(settings, StubAsyncClient(error=ConnectionError("down"))).warm_up()
    assert caught.value.unreachable


async def test_close(settings: Settings) -> None:
    stub = StubAsyncClient()
    await _client(settings, stub).close()
    assert stub.closed


def test_the_real_library_client_is_configured_from_settings(tmp_path: Path) -> None:
    client = OllamaClient(make_settings(tmp_path, ollama_host="http://127.0.0.1:12345"))
    inner = client._client._client  # the httpx.AsyncClient
    assert str(inner.base_url).startswith("http://127.0.0.1:12345")
    assert inner.timeout.connect == 3.0 and inner.timeout.read is None


# -- schema check ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "schema", "problem"),
    [
        ({"a": 1}, {"type": "object", "required": ["a"]}, ""),
        ({"a": True}, {"type": "object", "properties": {"a": {"type": "integer"}}}, "$.a: expected integer"),
        (2.5, {"type": "integer"}, "$: expected integer"),
        (3, {"type": "number"}, ""),
        (None, {"type": ["number", "null"]}, ""),
        ([1, 2, 3], {"type": "array", "maxItems": 2}, "$: too many items"),
        ([], {"type": "array", "minItems": 1}, "$: too few items"),
        (["x", 5], {"type": "array", "items": {"type": "string"}}, "$[1]: expected string"),
        (7, {"type": "integer", "minimum": 1, "maximum": 5}, "$: above maximum"),
        (0, {"type": "integer", "minimum": 1}, "$: below minimum"),
        ({"s": "NVDA"}, {"properties": {"s": {"enum": ["AMD"]}}}, "$.s: value not allowed"),
    ],
)
def test_schema_violation(value: Any, schema: dict[str, Any], problem: str) -> None:
    assert schema_violation(value, schema) == problem


def test_schema_violation_on_a_full_news_answer() -> None:
    from tests.llm.fakes import news_answer
    from trading_local.llm.schemas import news_schema

    answer = news_answer(impacts=[{"symbol": "TSLA", "sentiment": 0.5}])
    problem = schema_violation(json.loads(json.dumps(answer)), news_schema(["NVDA"]))
    assert problem == "$.impacts[0].symbol: value not allowed"
