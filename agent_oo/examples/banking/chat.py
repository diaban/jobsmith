"""Interactive REPL: chat with the agent; complex asks become background jobs.

Run with:  python -m agent_oo.examples.banking.chat

Default mode is a CONVERSATION. The chat agent answers simple messages
directly and proposes a background job for complex ones (launch_job tool);
you approve or decline the launch (human-in-the-loop). When a job finishes,
the next reply includes a short synthesis + the markdown report path.

Commands:
  <any text>        chat (the agent may propose launching a job)
  /jobs             list all jobs
  /job <id-prefix>  show a job's plan, artifacts, and answer
  /bg <any text>    bypass the chat: run that query as a job directly
  /image <key>      attach a (fake) image to the NEXT /bg job -> vision applies
  /cancel <id-pfx>  cancel a background job
  /quit             exit

LLM selection (override with --llm=anthropic|openai|fake):
- ANTHROPIC_API_KEY set -> Claude (chat: langchain-anthropic; jobs: AnthropicLLMClient).
- else OPENAI_API_KEY set -> OpenAI (chat: langchain-openai; jobs: OpenAILLMClient).
- else deterministic fakes — KeywordChatModel proposes a job when your message
  contains analysis-ish words, and KeywordLLM plans from keywords inside jobs.
Search/S3 stay fake in every mode.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import Any, ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from ...chat import ChatSession
from ...core.builder import AgentBuilder
from ...core.deps import Deps
from ...core.registry import CapabilityRegistry
from ...jobs.manager import JobManager
from ...jobs.models import Job, JobStatus
from .capabilities.refs import RefsCapability
from .capabilities.search import SearchCapability
from .capabilities.vision import VisionCapability
from .profile import BANKING_CHAT_PROMPT, BANKING_PROFILE

# ---------------------------------------------------------------- fake clients

class KeywordLLM:
    """Deterministic LLMClient: plans from keywords, answers from context.

    Good enough to *see the orchestration react to your prompt*; swap for a
    real LLMClient implementation to test actual model behaviour.
    """

    REFS_WORDS = ("deck", "slide", "presentation", "past", "previous", "reference")
    DIRECT_WORDS = ("what can you do", "who are you", "hello", "bonjour", "capabilit", "aide")

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "triage" in system.lower():
            direct = any(w in user.lower() for w in self.DIRECT_WORDS)
            return json.dumps({
                "route": "direct" if direct else "plan",
                "rationale": "keyword triage",
            })
        if "Answer the user's message directly" in system:
            return (
                "I'm a demo assistant: I plan and run capability jobs — "
                "search (KB lookup), vision (image analysis), refs (past decks)."
            )
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


class KeywordChatModel(BaseChatModel):
    """Deterministic tool-calling chat model: proposes a job on analysis-ish
    words, otherwise answers directly. Lets the whole chat/HITL/notification
    flow run without any API key."""

    COMPLEX_WORDS: ClassVar[tuple[str, ...]] = (
        "analyse", "analyze", "search", "cherche", "recherche",
        "rapport", "report", "compare", "étude", "review", "deck",
    )

    @property
    def _llm_type(self) -> str:
        return "keyword-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> KeywordChatModel:
        return self

    @staticmethod
    def _reply(text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        # 1. a finished-job notice was injected → synthesize it
        for m in reversed(messages):
            if isinstance(m, SystemMessage) and "background jobs finished" in m.content:
                report = next(
                    (ln.split(": ", 1)[1] for ln in m.content.splitlines()
                     if ln.startswith("Report file: ")), "?",
                )
                return self._reply(
                    "Un job vient de se terminer : l'analyse demandée est disponible. "
                    f"Rapport complet : {report}"
                )
        last = messages[-1]
        # 2. tool result (launch/status/cancel) → relay it
        if isinstance(last, ToolMessage):
            if "DECLINED" in last.content:
                return self._reply("Compris, je ne lance pas ce job.")
            return self._reply(f"C'est noté — {last.content}")
        # 3. user message → job proposal or direct answer
        user = last.content if isinstance(last, HumanMessage) else ""
        if any(w in user.lower() for w in self.COMPLEX_WORDS):
            call = {
                "name": "launch_job",
                "args": {
                    "query": user,
                    "rationale": "Tâche multi-étapes détectée (mots-clés) : "
                                 "un job d'analyse en arrière-plan est préférable.",
                },
                "id": f"call_{uuid.uuid4().hex[:8]}",
            }
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
            )
        return self._reply(
            "Je peux répondre directement, ou lancer des jobs d'analyse en arrière-plan "
            "(recherche, vision, références) — mode fake sans clé API."
        )


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

def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines; real env vars take precedence)."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def pick_provider() -> str:
    """--llm=... flag wins, else auto-detect by available key."""
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
    return choice


def make_llm(choice: str) -> Any:
    """LLMClient for the JOB ENGINE (planner/capabilities/generation)."""
    if choice == "anthropic":
        from ...clients import AnthropicLLMClient

        llm = AnthropicLLMClient()
        print(f"[jobs llm: Claude via AnthropicLLMClient — {llm.model}]")
        return llm
    if choice == "openai":
        from ...clients import OpenAILLMClient

        llm = OpenAILLMClient()
        print(f"[jobs llm: OpenAI via OpenAILLMClient — {llm.model}]")
        return llm
    print("[jobs llm: KeywordLLM fake — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real models]")
    return KeywordLLM()


def make_chat_model(choice: str) -> Any:
    """LangChain chat model for the CHAT AGENT (tool calling handled upstream)."""
    if choice == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            sys.exit("chat with Claude needs:  uv pip install langchain-anthropic")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        print(f"[chat llm: ChatAnthropic — {model}]")
        return ChatAnthropic(model=model, max_tokens=4096)
    if choice == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit("chat with OpenAI needs:  uv pip install langchain-openai")
        model = os.environ.get("OPENAI_MODEL", "gpt-5.1")
        print(f"[chat llm: ChatOpenAI — {model}]")
        return ChatOpenAI(model=model)
    print("[chat llm: KeywordChatModel fake]")
    return KeywordChatModel()


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
    load_dotenv()
    choice = pick_provider()
    jobs = build_chat(make_llm(choice))
    session = ChatSession(
        jobs,
        make_chat_model(choice),
        system_prompt=BANKING_CHAT_PROMPT,
        checkpointer=MemorySaver(),
    )
    agent = session.build()
    chat_cfg = {"configurable": {"thread_id": session.session_id}}
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
            print(f"  image {key!r} will be attached to the next /bg job")
        elif line.startswith("/bg "):
            job = await jobs.create_job(
                line[4:].strip(), dict(pending_inputs), session_id=session.session_id
            )
            pending_inputs.clear()
            jobs.start_job(job.job_id)
            print(f"  started in background: {job.job_id[:8]}  (try /jobs, /job {job.job_id[:4]})")
        elif line.startswith("/"):
            print("  unknown command (try /jobs, /job, /bg, /image, /cancel, /quit)")
        else:
            result = await agent.ainvoke({"messages": [HumanMessage(line)]}, chat_cfg)
            # human-in-the-loop: the agent proposes a job, you approve or not
            while "__interrupt__" in result:
                proposal = result["__interrupt__"][0].value
                print("\n  the agent proposes a background job:")
                print(f"    task     : {proposal.get('query')}")
                print(f"    approach : {proposal.get('rationale')}")
                answer = (await loop.run_in_executor(None, input, "  launch it? [y/N] "))
                approved = answer.strip().lower() in ("y", "yes", "o", "oui")
                result = await agent.ainvoke(Command(resume={"approved": approved}), chat_cfg)
            print("  " + (result["messages"][-1].content or "").replace("\n", "\n  "))

    print("bye")


if __name__ == "__main__":
    try:
        asyncio.run(repl())
    except KeyboardInterrupt:
        sys.exit(0)
