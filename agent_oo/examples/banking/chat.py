"""Interactive chat REPL: type prompts, watch the agent plan and execute jobs.

Run with:  python -m agent_oo.examples.banking.chat

Commands:
  <any text>        run a job for that query
  /image <key>      attach a (fake) image to the NEXT query -> planner may add vision
  /jobs             list all jobs
  /job <id-prefix>  show a job's plan, artifacts, and answer
  /bg <any text>    start the job in the background (try /jobs right after)
  /cancel <id-pfx>  cancel a background job
  /quit             exit

LLM selection (override with --llm=anthropic|openai|fake):
- ANTHROPIC_API_KEY set -> Claude (claude-opus-5) via AnthropicLLMClient.
- else OPENAI_API_KEY set -> OpenAI (gpt-5.1, or any OpenAI-compatible
  endpoint via OPENAI_BASE_URL) via OpenAILLMClient.
- else KeywordLLM — a deterministic fake whose *planning* reacts to your
  wording (deck/slide words pull in `refs`, an attached image pulls in
  `vision`) so plans still vary without any API key.
Search/S3 stay fake in every mode.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from ...core.builder import AgentBuilder
from ...core.deps import Deps
from ...core.registry import CapabilityRegistry
from ...jobs.manager import JobManager
from ...jobs.models import Job, JobStatus
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .profile import BANKING_PROFILE

# ---------------------------------------------------------------- fake clients

class KeywordLLM:
    """Deterministic LLMClient: plans from keywords, answers from context.

    Good enough to *see the orchestration react to your prompt*; swap for a
    real LLMClient implementation to test actual model behaviour.
    """

    REFS_WORDS = ("deck", "slide", "presentation", "past", "previous", "reference")

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "planner" in system.lower():
            return self._plan(user)
        if "Rewrite" in system:
            return json.dumps({"query": user.strip().lower()[:80]})
        if "failed validation" in system:
            return "Refined: " + user.split("Context:")[-1].strip()[:300] + " [doc_0]"
        # generation: answer by echoing what the capabilities produced
        ctx = user.split("Context:")[-1].strip()
        if ctx == "(no context available)":
            return "The context is insufficient to answer this query."
        return f"Based on the retrieved context [doc_0]:\n{ctx[:400]}"

    def _plan(self, query: str) -> str:
        q = query.lower()
        steps: list[dict[str, Any]] = [{"capability": "search", "depends_on": []}]
        if any(w in q for w in self.REFS_WORDS):
            steps.append({"capability": "refs", "depends_on": ["search"]})
        # Always propose vision — the planner validation drops it when no
        # image input is attached (spec.requires_inputs at work).
        steps.append({"capability": "vision", "depends_on": []})
        return json.dumps({
            "steps": steps,
            "rationale": "keyword plan: search always; refs on deck/slide words; vision if image",
        })

    async def vision(self, image_bytes: bytes, prompt: str, **kwargs: Any) -> str:
        return f"(fake analysis of {len(image_bytes)} bytes for: {prompt[:60]})"


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

def make_llm() -> Any:
    """Pick the LLM: --llm=... flag wins, else auto-detect by available key."""
    choice = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--llm=")), None)
    if choice is None and "--real" in sys.argv:
        choice = "anthropic"
    if choice is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            choice = "fake"

    if choice == "anthropic":
        from ...clients import AnthropicLLMClient

        llm = AnthropicLLMClient()
        print(f"[llm: Claude via AnthropicLLMClient — {llm.model}]")
        return llm
    if choice == "openai":
        from ...clients import OpenAILLMClient

        llm = OpenAILLMClient()
        print(f"[llm: OpenAI via OpenAILLMClient — {llm.model}]")
        return llm
    print("[llm: KeywordLLM fake — set ANTHROPIC_API_KEY or OPENAI_API_KEY for a real model]")
    return KeywordLLM()


def build_chat(llm: Any | None = None) -> JobManager:
    llm = llm if llm is not None else make_llm()
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


# ---------------------------------------------------------------- rendering

def show_job(job: Job, *, verbose: bool = True) -> None:
    print(f"  job {job.job_id[:8]}  [{job.status.value}]  {job.query[:60]!r}")
    if not verbose:
        return
    if job.plan:
        steps = " -> ".join(
            s["capability"] + (f"(after {','.join(s['depends_on'])})" if s["depends_on"] else "")
            for s in job.plan["steps"]
        )
        print(f"  plan:      {steps}")
        print(f"  rationale: {job.plan['rationale']}")
    for name, res in job.results.items():
        status = "ok" if res.get("ok") else f"FAILED ({res.get('error')})"
        print(f"  artifact:  {name}: {status}")
    if job.final_answer:
        print("  answer:\n    " + job.final_answer.replace("\n", "\n    "))
    if job.error and job.status is not JobStatus.DONE:
        print(f"  error:     {job.error}")


async def find_job(jobs: JobManager, prefix: str) -> Job | None:
    matches = [j for j in await jobs.list_jobs(limit=100) if j.job_id.startswith(prefix)]
    if len(matches) != 1:
        print(f"  {'no' if not matches else 'ambiguous'} job for prefix {prefix!r}")
        return None
    return await jobs.get_job(matches[0].job_id)


# ---------------------------------------------------------------- REPL

async def repl() -> None:
    jobs = build_chat()
    pending_inputs: dict[str, Any] = {}
    print(__doc__.split("Commands:")[1].split("LLM selection")[0])

    loop = asyncio.get_event_loop()
    while True:
        try:
            line = (await loop.run_in_executor(None, input, "agent> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit", "/q"):
            break
        elif line == "/jobs":
            listing = await jobs.list_jobs(limit=100)
            if not listing:
                print("  (no jobs yet)")
            for j in listing:
                show_job(j, verbose=False)
        elif line.startswith("/job "):
            job = await find_job(jobs, line.split(maxsplit=1)[1])
            if job:
                show_job(job)
        elif line.startswith("/cancel "):
            job = await find_job(jobs, line.split(maxsplit=1)[1])
            if job:
                cancelled = await jobs.cancel_job(job.job_id)
                print(f"  -> {cancelled.status.value}")
        elif line.startswith("/image "):
            key = line.split(maxsplit=1)[1]
            pending_inputs["image_s3_keys"] = [key]
            print(f"  image {key!r} will be attached to the next query")
        elif line.startswith("/bg "):
            job = await jobs.create_job(line[4:].strip(), dict(pending_inputs))
            pending_inputs.clear()
            jobs.start_job(job.job_id)
            print(f"  started in background: {job.job_id[:8]}  (try /jobs, /job {job.job_id[:4]})")
        elif line.startswith("/"):
            print("  unknown command (try /jobs, /job, /bg, /image, /cancel, /quit)")
        else:
            job = await jobs.create_job(line, dict(pending_inputs))
            pending_inputs.clear()
            job = await jobs.run_job(job.job_id)
            show_job(job)

    print("bye")


if __name__ == "__main__":
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        sys.exit(0)
