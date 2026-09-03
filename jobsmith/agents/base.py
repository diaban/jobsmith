"""What an agent *is*, in this project.

An agent is not a class hierarchy and not a separate application: it is a
**pack of capabilities plus a profile**. The runtime (`core/`), the job
engine (`jobs/`), the chat layer, the CLI and the HTTP API are shared by all
of them; a new agent supplies only what is genuinely domain-specific.

    AgentDefinition
      capabilities(llm)   what it can do   -> the registry, hence the planner
      profile             how it speaks    -> prompts, validation rules
      chat_prompt         the conversational persona (optional)

That is the whole contract. `build_app(agent=...)` composes any of them, so
adding an agent never means writing another composition root — which is
exactly what a domain agent used to have to duplicate, once per entrypoint.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.capability import Capability
from ..core.deps import LLMClient
from ..core.profile import AgentProfile


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    capabilities: Callable[[LLMClient], list[Capability]]
    profile: AgentProfile
    chat_prompt: str | None = None
