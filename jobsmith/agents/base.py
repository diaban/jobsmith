"""What an agent *is*, in this project.

An agent is not a class hierarchy and not a separate application: it is a
**pack of capabilities plus a profile**. The runtime (`core/`), the job
engine (`jobs/`), the chat layer, the CLI and the HTTP API are shared by all
of them; a new agent supplies only what is genuinely domain-specific.

    AgentDefinition
      open_resources(stack)  what it needs OPEN   -> pools, sessions, clients
      capabilities(ctx)      what it can do       -> the registry, hence the planner
      profile                how it speaks        -> prompts, validation rules
      chat_prompt            the conversational persona (optional)

That is the whole contract. `build_app(agent=...)` composes any of them, so
adding an agent never means writing another composition root — which is
exactly what a domain agent used to have to duplicate, once per entrypoint.

**Why resources are opened by the agent but owned by the composition root**:
only the agent knows what a "document index" means for it (which backend,
which collection, which embedder); only the composition root knows the
process lifetime and the event loop the connections must be created in. So
the agent gets handed an `AsyncExitStack` and returns whatever it built —
teardown then happens in reverse order when the app closes, whether it closed
cleanly or because something raised.

Two capabilities that need the same backend differently should share the
*connection* and get **one adapter each**, one per port they declared —
never a single fat client exposing both sets of methods.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from ..core.artifacts import ArtifactStore
from ..core.capability import Capability
from ..core.deps import LLMClient
from ..core.profile import AgentProfile


@dataclass(frozen=True)
class AgentContext:
    """What a capability pack is built from.

    A single parameter on purpose: what an agent may be handed will grow
    (settings, tracing), and growing this dataclass does not break every
    agent's signature.

    `artifacts` is the store a capability that produces a FILE writes to
    (`core/artifacts.py`). It is here rather than in `resources` because it is
    the composition root's to provide, not the agent's to open: where a job's
    files live is decided next to `reports_dir`, by the same reasoning that
    keeps a capability from knowing that layout. `build_app` always supplies
    one; it is `None` only for a context assembled by hand, and a capability
    that cannot write should then stay out of the registry — the same rule
    `documents` and `vision` already follow.

    `readable_roots` is the other half of that, and the same kind of fact: the
    local directories this deployment lets a capability READ a file out of
    when the request names one (`core/paths.py` says what "inside a root"
    means, and refuses everything else). It is the composition root's to
    decide for exactly the reason `artifacts` is — where a job's files live is
    decided next to `reports_dir`, and a capability that could widen its own
    readable set would be a capability that could read anything. An agent may
    add to it what it already exposes by other means (the `--docs` directory
    is searchable, so refusing to open a file *in* it by name would be
    incoherent), never more. Empty means no root: a capability that needs one
    then stays out of the registry, the rule `documents` and `vision` follow.
    """

    llm: LLMClient
    resources: Any = None
    artifacts: ArtifactStore | None = None
    readable_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    capabilities: Callable[[AgentContext], list[Capability]]
    profile: AgentProfile
    chat_prompt: str | None = None
    open_resources: Callable[[AsyncExitStack], Awaitable[Any]] | None = None


async def open_agent_resources(definition: AgentDefinition, stack: AsyncExitStack) -> Any:
    """Open an agent's external dependencies on the caller's stack (or nothing)."""
    if definition.open_resources is None:
        return None
    return await definition.open_resources(stack)
