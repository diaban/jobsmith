"""Composition root of the global agent.

`build_app()` assembles the whole product with zero configuration: provider
auto-selection (or injected clients), the default LLM-only capability pack,
neutral prompts/profile, in-memory persistence. Every piece can be overridden
by argument — a domain agent is just a different composition (see
agent_oo/examples/).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ..chat import ChatSession
from ..core.builder import AgentBuilder
from ..core.deps import Deps
from ..core.registry import CapabilityRegistry
from ..jobs.manager import JobManager
from .capabilities import default_capabilities
from .providers import make_chat_model, make_llm, pick_provider


@dataclass
class AgentApp:
    """A ready-to-serve agent: its job engine + a factory for chat sessions."""

    manager: JobManager
    session_factory: Callable[[], ChatSession]

    def new_session(self) -> ChatSession:
        return self.session_factory()


def build_app(
    *,
    llm: Any = None,
    chat_model: Any = None,
    checkpointer: Any = None,
    store: Any = None,
    reports_dir: str = "artifacts",
) -> AgentApp:
    if llm is None or chat_model is None:
        choice = pick_provider()
        llm = llm if llm is not None else make_llm(choice)
        chat_model = chat_model if chat_model is not None else make_chat_model(choice)

    registry = CapabilityRegistry(default_capabilities(llm))
    graph = AgentBuilder(
        Deps(llm=llm), registry, checkpointer=checkpointer or MemorySaver()
    ).build()
    manager = JobManager(graph, store or InMemoryStore(), reports_dir=reports_dir)

    def session_factory() -> ChatSession:
        return ChatSession(manager, chat_model, checkpointer=MemorySaver())

    return AgentApp(manager, session_factory)
