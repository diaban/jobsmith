"""The banking agent: a domain agent built on the shared runtime.

It supplies only what is domain-specific — three capabilities, their ports
and adapters, and a French-speaking profile. Everything else (job engine,
chat, CLI, API, persistence) is the same code the default agent runs on.

    jobsmith --agent banking chat
    jobsmith --agent banking serve
"""
from __future__ import annotations

from ...core.capability import Capability
from ...core.deps import LLMClient
from ..base import AgentDefinition
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .fakes import FakeS3, KeywordSearch
from .profile import BANKING_CHAT_PROMPT, BANKING_PROFILE


def banking_capabilities(llm: LLMClient) -> list[Capability]:
    # The adapters are chosen here, in the agent's own composition: each
    # capability receives exactly the ports it declared.
    search = KeywordSearch()
    return [
        SearchCapability(llm, search),
        VisionCapability(llm, FakeS3()),
        RefsCapability(search),
    ]


BANKING_AGENT = AgentDefinition(
    name="banking",
    description="Banking assistant: document search, slide vision, past references (French).",
    capabilities=banking_capabilities,
    profile=BANKING_PROFILE,
    chat_prompt=BANKING_CHAT_PROMPT,
)

__all__ = ["BANKING_AGENT", "banking_capabilities"]
