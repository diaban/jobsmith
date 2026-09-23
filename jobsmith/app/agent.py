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
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from ..agents import get_agent
from ..agents.base import AgentContext, open_agent_resources
from ..chat import DEFAULT_CHAT_SYSTEM_PROMPT, ChatSession
from ..core.artifacts import LocalArtifactStore
from ..core.builder import AgentBuilder
from ..core.deps import Deps
from ..core.registry import CapabilityRegistry
from ..jobs.manager import JobManager
from ..jobs.prior import RepositoryPriorJobs
from ..jobs.report import (
    available_formats,
    compose_reporters,
    ensure_formats_available,
    parse_report_formats,
)
from ..jobs.repository import StoreJobRepository
from .persistence import open_persistence, pick_db, pick_reports_dir
from .providers import make_chat_model, make_llm, pick_provider


@dataclass
class AgentApp:
    """A ready-to-serve agent: its job engine + a factory for chat sessions."""

    manager: JobManager
    session_factory: Callable[..., ChatSession]   # optional session_id argument
    agent_name: str = "default"
    resources: Any = None                         # whatever the agent opened
    registry: Any = None                          # what it can actually do
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
    """The format(s) of a document ASKED FOR without naming one: arg > env > markdown.

    Same precedence shape as `pick_db`, and the value is a comma-separated
    list: `JOBSMITH_REPORT_FORMAT=markdown,html` makes such a document come in
    both. **The first name is the main deliverable** — the one `report_path`
    and `/report` point at — so the order is a decision, not a formality.

    What it answers narrowed with #96, and the narrowing is the point. It
    used to answer "which formats, when a file is written and nobody named
    one" — and a file was written for every run that planned, so it was in
    effect the format of *every* silent request. A silent request now gets no
    file at all, so it no longer asks this question. What still asks it is a
    request that wants a document and names no format: "write me a report"
    read by the graph's document step, or a caller passing
    `DEFAULT_FORMATS_ALIAS` ("default") — the chat model, `POST /jobs`. It
    is therefore never a reason to write a file, only the answer to *which*
    file once one was asked for.

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
    reports_dir: str | None = None,
    report_format: str | None = None,
) -> AgentApp:
    # An agent is a capability pack + a profile (+ a chat persona): everything
    # else below is shared, whichever one is asked for.
    definition = get_agent(agent)
    if llm is None or chat_model is None:
        choice = pick_provider()
        llm = llm if llm is not None else make_llm(choice)
        chat_model = chat_model if chat_model is not None else make_chat_model(choice)

    # Absolute, and next to the jobs unless somebody said otherwise (#63):
    # every path a job records is built from this, and those records now
    # outlive the process — a relative `artifacts/` would be a path to
    # nothing from any other working directory.
    reports_root = pick_reports_dir(reports_dir)

    stack = AsyncExitStack()
    try:
        checkpointer, store = await open_persistence(pick_db(db), stack)
        # One repository over that store, shared by the manager that writes
        # job records and by the port a capability reads an earlier run's
        # material through (#74). Built here rather than left to `JobManager`
        # because it is needed BEFORE the manager exists — the registry is
        # composed first — and two repositories over one store would be two
        # answers to the question "what did that job produce".
        repository = StoreJobRepository(store)
        # The agent opens what it needs on OUR stack: it knows what its
        # backends are, we own their lifetime and the loop they live in.
        # An injected `resources` belongs to the caller — we do not close it.
        if resources is None:
            resources = await open_agent_resources(definition, stack)

        # A capability that produces a file writes through this port; it is
        # rooted where the manager keeps deliverables, so a job's annexes sit
        # next to its report and no capability has to know that layout.
        artifacts = LocalArtifactStore(reports_root)
        # ...and reads a named one from under the same root. That is the whole
        # of what this deployment declares readable: the tree the product's own
        # paths point into, so the report a job just wrote is a file the next
        # request can name. An agent may add what it already exposes by other
        # means (`--docs`); it may not add anything else. See `core/paths.py`.
        registry = CapabilityRegistry(
            definition.capabilities(
                AgentContext(llm, resources, artifacts,
                             readable_roots=(str(reports_root),),
                             # ...and reads what an EARLIER RUN produced by its
                             # id (#74). The other referent, and the one that
                             # needs no file: a follow-up asking for "a
                             # one-pager out of that job" gets the run's own
                             # material rather than the prose of a document
                             # that may not even have been written (#84).
                             prior_jobs=RepositoryPriorJobs(repository))
            )
        )
        # What "a document" is here when a request wants one and names no
        # format (#96). Composed now, so a format nothing can render — `pdf`
        # without pango — fails at startup rather than at the end of the
        # first job that asked for a report; the same rule `PdfReport`
        # applies to its engine.
        default_formats = pick_report_formats(report_format)
        ensure_formats_available(default_formats, registry=registry)
        graph = AgentBuilder(
            Deps(llm=llm), registry,
            profile=definition.profile, checkpointer=checkpointer,
            # What a request may ask its document to be, here (#90). The
            # engine reads the request for a format when the caller named
            # none, and it may only choose among what this deployment can
            # actually render — `.[pdf]` needs pango where the daemon runs,
            # so the list is composed here and nowhere in `core/`.
            document_formats=available_formats(registry),
            # ...and what it resolves "a report, no format named" to: the
            # SAME list the manager resolves the "default" argument to, so the
            # sentence and the argument cannot disagree about one request.
            default_document_formats=default_formats,
        ).build()
        # The registry is passed so capabilities present their own results;
        # a job's formats are composed into one reporter, whose first name is
        # the main deliverable. A factory, because every document is now one
        # somebody asked for (#55, #96) — the deployment's knowledge has to
        # reach a reporter built for that job, or a requested PDF would come
        # back without the registry.
        def reporter_for(formats: Sequence[str]) -> Any:
            return compose_reporters(formats, registry)

        manager = JobManager(
            graph, store, repository=repository,
            reporter_factory=reporter_for,
            default_formats=default_formats,
            reports_dir=reports_root,
        )
        # A previous process may have died mid-run: settle those jobs first.
        await manager.recover_interrupted()
    except BaseException:
        await stack.aclose()
        raise

    def session_factory(session_id: str | None = None) -> ChatSession:
        # Same checkpointer as the job graph: thread_id namespaces conversations
        # (session_id) apart from job runs (job_id), so both survive a restart.
        # Named rather than splatted from a dict: a `**kwargs` of unrelated
        # values types as one lowest common denominator, so every keyword
        # after it stops being checked — which is exactly what the type gate
        # is here to catch (see CLAUDE.md on #31).
        return ChatSession(
            manager, chat_model, session_id=session_id, checkpointer=checkpointer,
            system_prompt=definition.chat_prompt or DEFAULT_CHAT_SYSTEM_PROMPT,
        )

    return AgentApp(manager, session_factory, definition.name, resources, registry, stack)
