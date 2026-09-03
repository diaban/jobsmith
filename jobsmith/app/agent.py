"""Composition root of the global agent.

`build_app()` assembles the whole product: provider auto-selection (or
injected clients), the default LLM-only capability pack, neutral
prompts/profile, and the persistence backend. Every piece can be overridden
by argument; which agent is composed comes from `jobsmith/agents/`.

It is a coroutine because real backends (SQLite/Postgres connections and
pools) must be opened inside the event loop that will use them; `AgentApp`
owns their teardown via `aclose()`.
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from ..agents import get_agent
from ..chat import ChatSession
from ..core.builder import AgentBuilder
from ..core.deps import Deps
from ..core.registry import CapabilityRegistry
from ..jobs.manager import JobManager
from ..jobs.report import MarkdownReport
from .persistence import open_persistence, pick_db
from .providers import make_chat_model, make_llm, pick_provider


@dataclass
class AgentApp:
    """A ready-to-serve agent: its job engine + a factory for chat sessions."""

    manager: JobManager
    session_factory: Callable[..., ChatSession]   # optional session_id argument
    agent_name: str = "default"
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    def new_session(self, session_id: str | None = None) -> ChatSession:
        return self.session_factory(session_id) if session_id else self.session_factory()

    async def aclose(self) -> None:
        """Release persistence resources (connections, pools)."""
        await self._stack.aclose()


async def build_app(
    *,
    agent: str | None = None,
    llm: Any = None,
    chat_model: Any = None,
    db: str | None = None,
    reports_dir: str = "artifacts",
) -> AgentApp:
    # An agent is a capability pack + a profile (+ a chat persona): everything
    # else below is shared, whichever one is asked for.
    definition = get_agent(agent)
    if llm is None or chat_model is None:
        choice = pick_provider()
        llm = llm if llm is not None else make_llm(choice)
        chat_model = chat_model if chat_model is not None else make_chat_model(choice)

    stack = AsyncExitStack()
    try:
        checkpointer, store = await open_persistence(pick_db(db), stack)

        registry = CapabilityRegistry(definition.capabilities(llm))
        graph = AgentBuilder(
            Deps(llm=llm), registry,
            profile=definition.profile, checkpointer=checkpointer,
        ).build()
        manager = JobManager(
            graph, store,
            reporter=MarkdownReport(registry),   # capabilities present their own results
            reports_dir=reports_dir,
        )
        # A previous process may have died mid-run: settle those jobs first.
        await manager.recover_interrupted()
    except BaseException:
        await stack.aclose()
        raise

    def session_factory(session_id: str | None = None) -> ChatSession:
        # Same checkpointer as the job graph: thread_id namespaces conversations
        # (session_id) apart from job runs (job_id), so both survive a restart.
        prompt = {"system_prompt": definition.chat_prompt} if definition.chat_prompt else {}
        return ChatSession(manager, chat_model, session_id=session_id,
                           checkpointer=checkpointer, **prompt)

    return AgentApp(manager, session_factory, definition.name, stack)
