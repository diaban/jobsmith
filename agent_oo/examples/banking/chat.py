"""Banking REPL: the generic chat shell wired to the banking domain.

Run with:  python -m agent_oo.examples.banking.chat  [--llm=anthropic|openai|fake]

Everything generic (provider selection, fakes, REPL loop, HITL rendering)
lives in agent_oo.app — this module only supplies the banking capabilities,
profile, and domain fakes (search/S3).
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ...app.providers import load_dotenv, make_chat_model, make_llm, pick_provider
from ...app.repl import run_repl
from ...chat import ChatSession
from ...core.builder import AgentBuilder
from ...core.deps import Deps
from ...core.registry import CapabilityRegistry
from ...jobs.manager import JobManager
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .profile import BANKING_CHAT_PROMPT, BANKING_PROFILE

# ---------------------------------------------------------------- domain fakes

class KeywordSearch:
    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        if "past_slides" in query:
            return [{"id": "ref_0", "summary": f"deck related to '{query[:40]}'"}]
        return [
            {"id": "doc_0", "text": f"KB article matching '{query[:40]}'."},
            {"id": "doc_1", "text": "Second supporting document."},
        ]

    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        return await self.search(query, top_k=top_k)


class FakeS3:
    async def get_object(self, key: str) -> bytes:
        return f"fake-image-bytes:{key}".encode()


# ---------------------------------------------------------------- wiring

def build_chat(llm: Any | None = None) -> JobManager:
    llm = llm if llm is not None else make_llm(pick_provider())
    search = KeywordSearch()
    registry = CapabilityRegistry([
        SearchCapability(llm, search),
        VisionCapability(llm, FakeS3()),
        RefsCapability(search),
    ])
    builder = AgentBuilder(
        Deps(llm=llm), registry,
        profile=BANKING_PROFILE, checkpointer=MemorySaver(),
    )
    return JobManager(builder.build(), InMemoryStore())


async def repl() -> None:
    load_dotenv()
    choice = pick_provider()
    manager = build_chat(make_llm(choice))
    session = ChatSession(
        manager,
        make_chat_model(choice),
        system_prompt=BANKING_CHAT_PROMPT,
        checkpointer=MemorySaver(),
    )
    await run_repl(manager, session)


if __name__ == "__main__":
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        sys.exit(0)
