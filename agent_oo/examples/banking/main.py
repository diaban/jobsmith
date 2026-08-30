"""Banking demo: wire the framework with fake clients and run a job lifecycle.

Run with:  python -m agent_oo.examples.banking.main
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ...core.builder import AgentBuilder
from ...core.deps import Deps
from ...core.registry import CapabilityRegistry
from ...jobs.manager import JobManager
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .profile import BANKING_PROFILE

# --- Fake clients (replace with real ones in production wiring) ---

class DemoLLM:
    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        system = messages[0]["content"]
        if "planner" in system:
            return json.dumps({
                "steps": [
                    {"capability": "search", "depends_on": []},
                    {"capability": "refs", "depends_on": ["search"]},
                ],
                "rationale": "search the KB, then pull related decks",
            })
        if "Rewrite" in system:
            return json.dumps({"query": "credit exposure acme corp"})
        return (
            "Acme Corp's current credit exposure is EUR 12M [doc_1], "
            "trending down since Q2 [doc_2]. See also the 2025 review deck (ref_0)."
        )

    async def vision(self, image_bytes: bytes, prompt: str, **kwargs: Any) -> str:
        return "a chart of quarterly credit exposure"


class DemoSearch:
    async def search(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        if "past_slides" in query:
            return [{"id": "ref_0", "summary": "2025 credit review deck"}]
        return [
            {"id": "doc_1", "text": "Acme Corp exposure: EUR 12M."},
            {"id": "doc_2", "text": "Exposure trending down since Q2."},
        ]

    async def search_cached(self, query: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        return await self.search(query, top_k=top_k)


class DemoS3:
    async def get_object(self, key: str) -> bytes:
        return b"fake image bytes"


async def main() -> None:
    llm, search, s3 = DemoLLM(), DemoSearch(), DemoS3()

    registry = CapabilityRegistry([
        SearchCapability(llm, search),
        VisionCapability(llm, s3),
        RefsCapability(search),
    ])
    builder = AgentBuilder(
        Deps(llm=llm), registry,
        profile=BANKING_PROFILE, checkpointer=MemorySaver(),
    )
    jobs = JobManager(builder.build(), InMemoryStore())

    job = await jobs.create_job("What is our credit exposure to Acme Corp?")
    print(f"created  {job.job_id}  status={job.status.value}")

    job = await jobs.run_job(job.job_id)
    print(f"finished {job.job_id}  status={job.status.value}")
    print(f"\nplan: {json.dumps(job.plan, indent=2)}")
    print(f"\nartifacts: {sorted(job.results)}")
    for name, result in job.results.items():
        print(f"  - {name}: ok={result['ok']}")
    print(f"\nanswer:\n{job.final_answer}")

    listing = await jobs.list_jobs()
    print(f"\njobs in store: {[(j.job_id[:8], j.status.value) for j in listing]}")


if __name__ == "__main__":
    asyncio.run(main())
