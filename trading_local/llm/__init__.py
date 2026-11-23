"""LLM gateway: the local Ollama model behind one priority queue (implements ``contracts.Llm``).

    llm = build_llm(settings=settings, db=db, clock=clock, runtime_settings=runtime_settings)
    await llm.start()                  # returns at once; the model is loaded in the background
    analysis = await llm.analyze_news(item, candidates)

Modules: ``client`` (Ollama wrapper and production settings), ``jobs`` (priority queue, one worker),
``prompts`` / ``schemas`` (prompts, request schemas, answer validation), ``service`` (``OllamaLlm``).
"""

from trading_local.clock import Clock
from trading_local.config import Settings
from trading_local.contracts import Llm
from trading_local.db.database import Database
from trading_local.db.repo_ops import SettingsStore
from trading_local.llm.client import ChatClient, ChatResult, LlmError, OllamaClient
from trading_local.llm.jobs import Priority, PriorityJobQueue
from trading_local.llm.service import OllamaLlm


def build_llm(*, settings: Settings, db: Database, clock: Clock, runtime_settings: SettingsStore) -> Llm:
    """The platform's LLM gateway.

    ``runtime_settings`` is accepted for uniform wiring; the user's switches for LLM work
    (``news_llm_enabled``, ``llm_review``, ``chart_scan_enabled``) are applied by the callers, which decide
    whether to ask at all. ``settings.enable_llm`` False yields a gateway that answers every request at once
    with None / 'unavailable'.
    """
    return OllamaLlm(settings=settings, db=db, clock=clock, client=OllamaClient(settings))


__all__ = [
    "ChatClient",
    "ChatResult",
    "LlmError",
    "OllamaClient",
    "OllamaLlm",
    "Priority",
    "PriorityJobQueue",
    "build_llm",
]
