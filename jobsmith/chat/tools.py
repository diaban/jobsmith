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
from datetime import UTC, datetime
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from ..core.state import CONVERSATION_INPUT_KEY, SOURCE_FILES_INPUT_KEY
from ..jobs.manager import JobManager
from ..jobs.models import Job, JobStatus
from ..jobs.report import (
    available_formats,
    document_stem,
    ensure_formats_available,
)

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


# ---------------- Progress, derived from what is already persisted ----------
#
# Nothing here adds bookkeeping to the job engine: `plan` says what the DAG is
# and `step_finished_at` says what has landed, which is enough to say where a
# run currently stands. Shared by the `job_status` tool (pull) and the
# notification middleware (push).


def running_steps(job: Job) -> list[str]:
    """Plan steps whose dependencies have all landed but which have not
    finished yet — i.e. the executor's current wave.

    Empty for a job that is not RUNNING: a cancelled or failed run leaves
    unfinished steps behind, and calling those "running" would be a lie.
    """
    if not job.plan or job.status is not JobStatus.RUNNING:
        return []
    done = job.step_finished_at
    return [
        step["capability"]
        for step in job.plan["steps"]
        if step["capability"] not in done
        and all(dep in done for dep in step.get("depends_on", []))
    ]


def elapsed_since(timestamp: str) -> str:
    """Coarse human duration since an ISO timestamp ("" when unparseable)."""
    try:
        seconds = int((datetime.now(UTC) - datetime.fromisoformat(timestamp)).total_seconds())
    except (TypeError, ValueError):
        return ""
    seconds = max(seconds, 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}"


def progress_line(job: Job) -> str:
    """One compact line: how far an in-flight job has got.

    Deliberately one line: it is re-injected (fresh, never accumulated) into a
    model request whenever it changes, so its cost is paid again every time.
    """
    head = f"{job.job_id[:8]} {job.query[:50]!r}"
    tail = f" · {age} elapsed" if (age := elapsed_since(job.created_at)) else ""
    if not job.plan:
        return f"{head}: {job.status.value}, planning{tail}"
    steps = [s["capability"] for s in job.plan["steps"]]
    done = [name for name in steps if name in job.step_finished_at]
    parts = [f"{len(done)}/{len(steps)} steps done"]
    if done:
        parts[0] += f" ({', '.join(done)})"
    if running := running_steps(job):
        parts.append(f"running {', '.join(running)}")
    return f"{head}: {' · '.join(parts)}{tail}"


def progress_signature(job: Job) -> str:
    """What must change before a job is worth reporting again: its status, the
    shape of its plan, and how many steps have landed. Elapsed time is
    deliberately excluded — otherwise every single turn would look like news.
    """
    plan_size = len(job.plan["steps"]) if job.plan else 0
    return f"{job.status.value}:{plan_size}:{len(job.step_finished_at)}"


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
        source_files: list[str] | None = None,
        document_name: str | None = None,
        document_title: str | None = None,
        formats: list[str] | None = None,
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
        do. `inputs` carries structured material the job needs (image keys,
        ...); the recent conversation turns are attached automatically as
        background — never paste them into `query`.

        `source_files` names files the job must READ — a path the user gave,
        or the report a previous job of this conversation produced. Write each
        path exactly as it was given to you; never invent one, never guess at
        a directory, and never paste a file's contents into `query`. The user
        sees this list when approving the launch and is handing those files
        over, so a path they did not mention has no business in it.

        `document_name` is what the file should be CALLED — a short filename
        with no extension, no directory and no spaces (`chair_comparison`).
        Use the user's own name when they gave one; when they did not, propose
        a short one from the subject rather than leaving it: the alternative is
        a file named after a job id. `document_title` is the heading INSIDE the
        document, in the language of the request, and is a different decision —
        naming one never names the other. `formats` is which files are written
        (`["markdown"]`, `["markdown", "pdf"]`, ...); the FIRST is the main
        deliverable. Ask for what the user asked for and nothing more, and
        never promise a file in your own prose — only these arguments produce
        one. The user sees all three when approving.

        Returns immediately; the conversation continues while the job runs.
        """
        job_inputs = dict(inputs or {})
        state = getattr(runtime, "state", None) or {}
        excerpt = recent_conversation(state.get("messages") or [])
        if excerpt:
            job_inputs.setdefault(CONVERSATION_INPUT_KEY, excerpt)
        # A named file travels as an INPUT, never as prose the engine would
        # have to parse back out of the query. Empty means "none": the key is
        # left out entirely, so the planner drops the reading step instead of
        # planning one that can only report that nothing was given.
        sources = [ref for ref in (str(f).strip() for f in source_files or []) if ref]
        if sources:
            job_inputs[SOURCE_FILES_INPUT_KEY] = sources

        # Refused BEFORE the card, not after the run: a format nothing can
        # render here and a name that is not a filename are both things the
        # model can fix on the spot, and the answer goes back to it as text.
        # `PathRefused` is a `ValueError`, so one arm covers both.
        try:
            wanted = ensure_formats_available(formats or [])
            given = (document_name or "").strip()
            name = document_stem(given) if given else ""
        except ValueError as refused:
            return (f"NOT launched: {refused}. Nothing ran. Tell the user what "
                    f"is possible — formats available here: "
                    f"{', '.join(available_formats())} — and propose a launch again.")
        title = (document_title or "").strip()

        decision = interrupt({
            "action": "launch_job",
            "query": query,
            "rationale": rationale,
            # what the document will be called, be titled, and be written as:
            # a file promised in prose and never written is what #55 is about,
            # and these are the only three things that produce one
            "document_name": name,
            "document_title": title,
            "formats": wanted,
            # what will travel besides the query, so a front-end can show
            # exactly what the user is approving
            "context": job_inputs.get(CONVERSATION_INPUT_KEY, ""),
            # the files it will be allowed to open — the same guard the query
            # gets, and for the stronger reason: this is the user handing
            # something over, not the model restating what they asked
            "sources": sources,
        })
        if not (isinstance(decision, dict) and decision.get("approved")):
            return "The user DECLINED the launch. Do not launch this job; continue the conversation."
        job = await manager.create_job(
            query, job_inputs, session_id=session_id,
            document_name=name, document_title=title, formats=wanted)
        manager.start_job(job.job_id)
        return (
            f"Job {job.job_id} launched in the background (short id {job.job_id[:8]}). "
            "Tell the user; you will be notified here when it finishes."
        )

    @tool
    async def job_status(job_id_prefix: str) -> str:
        """Get the detailed status of one of this session's jobs, by id prefix:
        every plan step marked done / running / pending, how long it has been
        going, the report path once there is one.

        Call it when the user wants more detail than the short [job progress]
        notice carries, or asks about a job that notice does not mention (an
        older, already finished one)."""
        job = await _find(manager, session_id, job_id_prefix)
        if job is None:
            return f"No unique job of this session matches prefix {job_id_prefix!r}."
        parts = [_line(job)]
        if age := elapsed_since(job.created_at):
            parts[0] += f" · {age} elapsed"
        if job.plan:
            wave = set(running_steps(job))
            for step in job.plan["steps"]:
                name = step["capability"]
                if name in job.step_finished_at:
                    mark = f"done at {job.step_finished_at[name]}"
                else:
                    mark = "running" if name in wave else "pending"
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
        # cancel_job re-reads the record and can come back empty; the one we
        # just found is the honest fallback.
        return f"Job {job.job_id[:8]} is now {(cancelled or job).status.value}."

    return [launch_job, job_status, list_my_jobs, cancel_job]
