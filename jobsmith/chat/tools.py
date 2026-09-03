"""LangChain tools wrapping the JobManager use-cases for the chat agent.

`launch_job` is human-in-the-loop: it `interrupt()`s with the agent's
rationale before anything runs; the session owner resumes the graph with
`Command(resume={"approved": bool})`. The other tools are read/cancel
operations scoped to the chat session's own jobs.

Carrying the referent across the boundary
-----------------------------------------
The job engine never sees the conversation: it gets a `query` string and an
`inputs` dict. So "analyse that", written after three turns of discussion,
would reach the planner with its referent stripped — and fail silently, by
answering a slightly different question. Two complementary guards:

1. the tool's docstring is the model's instruction — it demands a
   self-contained `query`, which is also what the user approves at the
   interrupt, so the proposal stays readable;
2. a bounded excerpt of the recent turns rides along in
   `inputs[CONVERSATION_INPUT_KEY]` as a safety net for when the model
   under-specifies anyway (the planner reads it as background only).

The excerpt is deliberately small — it is paid for on every job launch, and a
job needs the referent, not the thread.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from ..core.state import CONVERSATION_INPUT_KEY
from ..jobs.manager import JobManager
from ..jobs.models import Job

# Bounds on the conversation excerpt attached to a launch (~400 tokens worst case).
MAX_CONTEXT_TURNS = 6      # most recent user/assistant turns kept
MAX_TURN_CHARS = 400       # each turn truncated to this
MAX_CONTEXT_CHARS = 1500   # hard ceiling on the whole excerpt


def _text_of(message: Any) -> str:
    """Plain text of a message; content blocks are flattened, non-text dropped."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p).strip()
    return ""


def recent_conversation(messages: Iterable[Any]) -> str:
    """Render the last few user/assistant turns as a bounded transcript.

    Only human and assistant *prose* travels: tool calls, tool results and the
    injected [job update] system notices are machinery, not the referent, and
    would cost tokens while adding noise the planner cannot use.
    """
    turns: list[str] = []
    budget = MAX_CONTEXT_CHARS
    for message in reversed(list(messages)):
        if len(turns) >= MAX_CONTEXT_TURNS or budget <= 0:
            break
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage) and not message.tool_calls:
            role = "assistant"
        else:
            continue
        text = _text_of(message)
        if not text:
            continue
        if len(text) > MAX_TURN_CHARS:
            text = text[:MAX_TURN_CHARS].rstrip() + "…"
        line = f"{role}: {text}"
        if len(line) > budget:
            break
        budget -= len(line) + 1
        turns.append(line)
    return "\n".join(reversed(turns))


def _line(job: Job) -> str:
    return f"{job.job_id[:8]} [{job.status.value}] {job.query[:60]!r}"


async def _find(manager: JobManager, session_id: str, prefix: str) -> Job | None:
    """Resolve a job-id prefix among THIS session's jobs only."""
    jobs = await manager.list_jobs(session_id=session_id, limit=100)
    matches = [j for j in jobs if j.job_id.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


def make_job_tools(manager: JobManager, session_id: str) -> list[Any]:
    """Build the job tools bound to one manager + one chat session."""

    @tool
    async def launch_job(
        query: str,
        rationale: str,
        runtime: ToolRuntime,
        inputs: dict[str, Any] | None = None,
    ) -> str:
        """Launch a complex task as a BACKGROUND job (research, analysis,
        anything needing several capability steps).

        `query` is the task for the job engine, which does NOT see this
        conversation: write it so it stands entirely on its own. Resolve every
        referent ("that", "the second option", "same as before") into explicit
        words, and restate the subject, the constraints and what the answer
        should contain. The user approves this exact wording before anything
        runs, so it must also read as a faithful statement of what they asked.

        `rationale` explains to the user why a job is needed and what it will
        do. `inputs` carries structured material the job needs (file refs,
        image keys, ...); the recent conversation turns are attached
        automatically as background — never paste them into `query`.

        Returns immediately; the conversation continues while the job runs.
        """
        job_inputs = dict(inputs or {})
        state = getattr(runtime, "state", None) or {}
        excerpt = recent_conversation(state.get("messages") or [])
        if excerpt:
            job_inputs.setdefault(CONVERSATION_INPUT_KEY, excerpt)

        decision = interrupt({
            "action": "launch_job",
            "query": query,
            "rationale": rationale,
            # what will travel besides the query, so a front-end can show
            # exactly what the user is approving
            "context": job_inputs.get(CONVERSATION_INPUT_KEY, ""),
        })
        if not (isinstance(decision, dict) and decision.get("approved")):
            return "The user DECLINED the launch. Do not launch this job; continue the conversation."
        job = await manager.create_job(query, job_inputs, session_id=session_id)
        manager.start_job(job.job_id)
        return (
            f"Job {job.job_id} launched in the background (short id {job.job_id[:8]}). "
            "Tell the user; you will be notified here when it finishes."
        )

    @tool
    async def job_status(job_id_prefix: str) -> str:
        """Get the status and progress (finished steps) of one of this
        session's jobs, by id prefix."""
        job = await _find(manager, session_id, job_id_prefix)
        if job is None:
            return f"No unique job of this session matches prefix {job_id_prefix!r}."
        parts = [_line(job)]
        if job.plan:
            for step in job.plan["steps"]:
                name = step["capability"]
                mark = "done" if name in job.step_finished_at else "pending"
                parts.append(f"  - {name}: {mark}")
        if job.report_path:
            parts.append(f"report: {job.report_path}")
        if job.error:
            parts.append(f"error: {job.error}")
        return "\n".join(parts)

    @tool
    async def list_my_jobs() -> str:
        """List this session's jobs (id, status, query)."""
        jobs = await manager.list_jobs(session_id=session_id, limit=50)
        return "\n".join(_line(j) for j in jobs) or "No jobs in this session yet."

    @tool
    async def cancel_job(job_id_prefix: str) -> str:
        """Cancel one of this session's running or queued jobs, by id prefix."""
        job = await _find(manager, session_id, job_id_prefix)
        if job is None:
            return f"No unique job of this session matches prefix {job_id_prefix!r}."
        cancelled = await manager.cancel_job(job.job_id)
        return f"Job {job.job_id[:8]} is now {cancelled.status.value}."

    return [launch_job, job_status, list_my_jobs, cancel_job]
