"""The banking agent: a domain agent built on the shared runtime.

It supplies only what is domain-specific — three capabilities, their ports
and adapters, and a French-speaking profile. Everything else (job engine,
chat, CLI, API, persistence) is the same code the default agent runs on.

    jobsmith --agent banking chat
    jobsmith --agent banking serve
"""
from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

from ...core.capability import Capability
from ..base import AgentContext, AgentDefinition
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .fakes import FakeS3, KeywordSearch
from .profile import BANKING_CHAT_PROMPT, BANKING_PROFILE


@dataclass(frozen=True)
class BankingResources:
    """One adapter per port this agent declared in `deps.py`.

    Real backends belong here too: two capabilities needing the same store
    differently share the *connection* opened below, and get one adapter each
    — never a single client exposing both sets of methods.
    """

    search: KeywordSearch
    objects: FakeS3


async def open_banking_resources(stack: AsyncExitStack) -> BankingResources:
    # `stack` is the app's: anything entered here is closed in reverse order
    # when the app closes, cleanly or on error. A real pool would be opened
    # with `await stack.enter_async_context(...)` right here.
    return BankingResources(search=KeywordSearch(), objects=FakeS3())


def banking_capabilities(ctx: AgentContext) -> list[Capability]:
    # Each capability receives exactly the ports it declared, nothing more.
    res: BankingResources = ctx.resources
    return [
        SearchCapability(ctx.llm, res.search),
        VisionCapability(ctx.llm, res.objects),
        RefsCapability(res.search),
    ]


BANKING_AGENT = AgentDefinition(
    name="banking",
    description="Banking assistant: document search, slide vision, past references (French).",
    capabilities=banking_capabilities,
    profile=BANKING_PROFILE,
    chat_prompt=BANKING_CHAT_PROMPT,
    open_resources=open_banking_resources,
)

__all__ = ["BANKING_AGENT", "BankingResources", "banking_capabilities", "open_banking_resources"]
