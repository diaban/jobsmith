"""Composition root of the global agent.

`build_app()` assembles the whole product: provider auto-selection (or
injected clients), the default LLM-only capability pack, neutral
prompts/profile, and the persistence backend. Every piece can be overridden
by argument; which agent is composed comes from `jobsmith/agents/`.

It is a coroutine because real backends must be opened inside the event loop
that will use them — the persistence layer, and whatever the chosen agent
opens for its own capabilities (a vector-store pool, an HTTP session, an MCP
connection). Everything lands on one `AsyncExitStack`, so `AgentApp.aclose()`
tears it all down in reverse order, including when startup itself failed.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from ..agents import get_agent
from ..agents.base import AgentContext, open_agent_resources
from ..chat import ChatSession
from ..core.builder import AgentBuilder
from ..core.deps import Deps
from ..core.registry import CapabilityRegistry
from ..jobs.manager import JobManager
from ..jobs.report import compose_reporters, parse_report_formats
from .persistence import open_persistence, pick_db
from .providers import make_chat_model, make_llm, pick_provider


@dataclass
class AgentApp:
    """A ready-to-serve agent: its job engine + a factory for chat sessions."""

    manager: JobManager
    session_factory: Callable[..., ChatSession]   # optional session_id argument
    agent_name: str = "default"
    resources: Any = None                         # whatever the agent opened
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    def new_session(self, session_id: str | None = None) -> ChatSession:
        return self.session_factory(session_id) if session_id else self.session_factory()

    def service(self) -> Any:
        """The inbound port over this app — what every entrypoint talks to."""
        from ..service import LocalAgentService

        return LocalAgentService(self.manager, self.session_factory, on_close=self.aclose)

    async def aclose(self) -> None:
        """Release persistence resources (connections, pools)."""
        await self._stack.aclose()


def pick_report_formats(flag: str | None = None) -> list[str]:
    """Which formats a finished job hands back: argument > env > markdown.

    Same precedence shape as `pick_db`, and the value is a comma-separated
    list: `JOBSMITH_REPORT_FORMAT=markdown,html` makes one run write both.
    A single name is the ordinary case and behaves exactly as it always did.
    **The first name is the main deliverable** — the one `report_path` and
    `/report` point at — so the order is a decision, not a formality.

    There is no CLI flag yet: the entrypoints belong to another seam, so the
    environment variable is how an operator switches the deliverable today.
    """
    spec = flag or os.environ.get("JOBSMITH_REPORT_FORMAT") or "markdown"
    return parse_report_formats(spec) or ["markdown"]


async def build_app(
    *,
    agent: str | None = None,
    llm: Any = None,
    chat_model: Any = None,
    resources: Any = None,
    db: str | None = None,
    reports_dir: str = "artifacts",
    report_format: str | None = None,
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
        # The agent opens what it needs on OUR stack: it knows what its
        # backends are, we own their lifetime and the loop they live in.
        # An injected `resources` belongs to the caller — we do not close it.
        if resources is None:
            resources = await open_agent_resources(definition, stack)

        registry = CapabilityRegistry(definition.capabilities(AgentContext(llm, resources)))
        graph = AgentBuilder(
            Deps(llm=llm), registry,
            profile=definition.profile, checkpointer=checkpointer,
        ).build()
        manager = JobManager(
            graph, store,
            # The registry is passed so capabilities present their own
            # results; the formats asked for are composed into one reporter,
            # whose first name is the main deliverable.
            reporter=compose_reporters(pick_report_formats(report_format), registry),
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

    return AgentApp(manager, session_factory, definition.name, resources, stack)
