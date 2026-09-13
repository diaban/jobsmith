"""LangChain tools wrapping the JobManager use-cases for the chat agent.

`launch_job` runs a task **in the turn** and promotes it to the background
when it turns out to be slow. The other tools are read/cancel operations
scoped to the chat session's own jobs.

Synchronous by default, background by promotion (#83)
-----------------------------------------------------
Every task starts the same way — `create_job` + `start_job`, so it is a
tracked, cancellable, persisted Job from the first instant and it outlives
the turn whatever happens next. Then the tool simply **waits on it**, up to
`pick_sync_timeout()`. Promotion is therefore "stop waiting": nothing is
cancelled, nothing is restarted, and a run that is promoted is byte-for-byte
the run that was about to finish.

That shape is the whole design. The alternative — decide up front whether
this looks like a long task — is a prediction, and the cost of a run is set
by the *plan*, which does not exist when that decision would be taken. A
model guessing a duration from a sentence is wrong in both directions: a
ten-second question announced as a background job and answered a turn later,
or a four-minute run holding the prompt hostage. The clock needs no model
and cannot be wrong about what it observed.

The notice, and what used to hang on the approval card
------------------------------------------------------
Nothing interrupts on the nominal path: the agent says what it is doing
instead of asking whether it may. Three guarantees used to hang on that y/N
card, and all three survive as **visibility** — they ride on the
`job_started` notice this tool writes into the turn before the run begins:
the reformulated query, the files the run may open (#60), and what the
document will be called, titled and written as (#55). The notice also
carries the `job_id`, which the card could not: **cancellation is the undo
the approval was the gate for**, so the thing to cancel has to be nameable.

The gate itself is kept, unused, behind `pick_approval_required()` — a
deployment that wants it back, and the seam a future capability that spends
money or does something irreversible will hang off. Removing it would mean
rebuilding it.

Carrying the answer back without rewriting it
---------------------------------------------
A tool result is read by the model, which then writes the user's reply from
it — so returning a 2 000-word answer is asking for a paraphrase, and a
paraphrase is the one thing a deliverable must not be. The answer therefore
does **not** come back through the tool result: it is written straight into
the turn as `Token`s (`chat/runner.py`'s custom channel), verbatim, and the
tool result tells the model it has already been delivered and asks for at
most one short sentence. Where the answer *lives* afterwards — the
conversation or the file — is a separate question and deliberately not
settled here (#85); what this fixes is that nothing between the generator
and the reader rewrites it.

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

import asyncio
import os
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from ..core.state import CONVERSATION_INPUT_KEY, SOURCE_FILES_INPUT_KEY, TERMINAL_UNANSWERED
from ..jobs.manager import JobManager
from ..jobs.models import Job, JobStatus
from ..jobs.report import (
    available_formats,
    document_stem,
    ensure_formats_available,
)
from .runner import CUSTOM_ANSWER, CUSTOM_JOB_STARTED

# Bounds on the conversation excerpt attached to a launch (~400 tokens worst case).
MAX_CONTEXT_TURNS = 6      # most recent user/assistant turns kept
MAX_TURN_CHARS = 400       # each turn truncated to this
MAX_CONTEXT_CHARS = 1500   # hard ceiling on the whole excerpt

# How long a task may hold the conversation before it is promoted to the
# background. Twenty seconds because this is a *human patience* budget and
# nothing else: below it a person waits at a prompt the way they wait for a
# slow command, and past it a conversation that has not said anything reads
# as hung. It is deliberately not tuned to what a run costs — the point of
# waiting on the clock is that it observes rather than predicts.
#
# It is also the one number that decides which mode is the exception. Against
# a real provider a multi-step plan is minutes, so most real tasks promote
# and the background stays what it always was; what changes is that the short
# ones — a single grounded step, a small read — now answer in the turn
# instead of costing a card, a wait and a later notice.
DEFAULT_SYNC_TIMEOUT = 20.0


def pick_sync_timeout(explicit: float | None = None) -> float:
    """How long to wait before promoting: argument > $JOBSMITH_SYNC_TIMEOUT > 20s.

    Same precedence shape as `pick_db` / `pick_search_depth`, minus the
    command-line flag: this is read inside a daemon that a client cannot pass
    flags to, so the environment is where a deployment says it. `0` is
    meaningful and supported — never wait, i.e. every task goes to the
    background, which is exactly the behaviour this issue replaced. A value
    that is not a number is refused loudly rather than silently ignored: a
    deployment that meant to change this must not discover it did not.
    """
    if explicit is not None:
        return max(float(explicit), 0.0)
    raw = os.environ.get("JOBSMITH_SYNC_TIMEOUT")
    if not raw:
        return DEFAULT_SYNC_TIMEOUT
    try:
        return max(float(raw), 0.0)
    except ValueError:
        print(f"[chat: $JOBSMITH_SYNC_TIMEOUT={raw!r} is not a number — "
              f"using {DEFAULT_SYNC_TIMEOUT:.0f}s]", file=sys.stderr)
        return DEFAULT_SYNC_TIMEOUT


def pick_approval_required(explicit: bool | None = None) -> bool:
    """Whether a launch still asks y/N first: argument > $JOBSMITH_APPROVE_JOBS > no.

    Off by default — that is #83. On, the tool `interrupt()`s exactly as it
    always did and the whole approval round trip (`stream_approval`, the
    API's route, the REPL prompt, the TUI card) is the nominal path again.

    It is kept for two reasons that are really one. A deployment may want the
    gate back, and a capability that spends money or does something
    irreversible will want it for itself one day — this is the v1 of that,
    per deployment rather than per capability, and the mechanism it keeps
    alive is the expensive half to rebuild.
    """
    if explicit is not None:
        return explicit
    return (os.environ.get("JOBSMITH_APPROVE_JOBS") or "").strip().lower() in (
        "1", "true", "yes", "on")


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


# ---------------- Running a task inside the turn -----------------------------


def _writer() -> Callable[[dict[str, Any]], None]:
    """The channel a tool has into the turn it is running inside.

    LangGraph's custom stream, which `chat/runner.py` is the only reader of.
    Outside a run — a unit test calling the tool directly, or an `ainvoke`
    with no stream attached — there is nothing listening, so the writes go
    nowhere. That must not be an error: what a tool *does* cannot depend on
    whether anyone is watching it.
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:                       # pragma: no cover - no run context
        return lambda _payload: None


def _files_of(job: Job) -> str:
    return ", ".join(output.path for output in job.outputs)


def _delivered(job: Job) -> str:
    """What the model is told about a job that finished inside the turn.

    Deliberately NOT the answer. The user has already read it — the tool
    wrote it into the turn verbatim — and handing the model the same 2 000
    words back with "tell the user" is asking for the paraphrase this whole
    change exists to avoid. So the result says what happened, where the file
    is, and asks for one short sentence at most.

    The branches are `JobNotificationMiddleware._notice_for`'s, for the same
    reasons and with the same care: a stop is not a failure, a run that could
    not answer must not be announced as one that did (#59), and files a
    stopped run left behind are named as partial material rather than hidden
    (#41). They are said differently here only because the answer has already
    been delivered, which is the one thing the completion notice cannot say.
    """
    short = job.job_id[:8]
    if job.status is JobStatus.CANCELLED:
        stopped = [f"Job {short} was CANCELLED before it finished: no answer, no report."]
        if job.outputs:
            stopped.append("Files its steps left behind (partial material, not a "
                           f"report): {_files_of(job)}")
        if job.error:
            stopped.append(f"Also worth passing on: {job.error}")
        stopped.append("Tell the user it was stopped.")
        return "\n".join(stopped)

    if job.status is not JobStatus.DONE:
        failed = [f"Job {short} FAILED: {job.error}"]
        if job.outputs:
            failed.append("Files its steps left behind (partial material, not a "
                          f"report): {_files_of(job)}")
        failed.append("Tell the user it failed and why, in one or two sentences.")
        return "\n".join(failed)

    if job.terminal_kind == TERMINAL_UNANSWERED:
        unanswered = [
            f"Job {short} ran to the end but COULD NOT ANSWER: the material it "
            "gathered does not answer the request. Its explanation of what was "
            "missing has ALREADY been shown to the user, in full — do not repeat "
            "or summarize it. Say plainly, in one sentence, that there is no result."
        ]
        if job.report_path:
            unanswered.append(f"The explanation is also saved at: {job.report_path}")
        return "\n".join(unanswered)

    done = [
        f"Job {short} finished. Its answer has ALREADY been shown to the user, in "
        "full and word for word — do NOT repeat it, summarize it, rephrase it or "
        "comment on its content. Reply with at most one short sentence."
    ]
    if job.report_path:
        done.append(f"The deliverable is saved at: {job.report_path} — worth naming.")
    elif not job.deliverable_expected:
        # No file, and nothing went wrong: none was asked for (#84). Said out
        # loud because the alternative is a model that invents a path, or
        # apologises for a failure that did not happen.
        done.append("No file was written and none was asked for — do not "
                    "mention a document, and do not apologise for its absence.")
    elif job.error:
        done.append(f"No deliverable file was saved ({job.error}) — say so.")
    if len(job.outputs) > 1:
        done.append("Other files it produced: " + ", ".join(
            o.path for o in job.outputs if o.path != job.report_path))
    return "\n".join(done)


def _promoted(job: Job, waited: float) -> str:
    """What the model is told about a run that outlived the wait.

    The turn has to end saying so — the answer is not coming in it — and the
    completion notice (`JobNotificationMiddleware`) is what brings the answer
    back on a later turn. Nothing about the run changed: it was a background
    task from the first instant, and all that stopped is the waiting.
    """
    short = job.job_id[:8]
    return (
        f"Job {job.job_id} is still running after {waited:.0f}s, so it has been "
        f"moved to the BACKGROUND (short id {short}). It keeps running. Tell the "
        "user it is running in the background, that you will report back here "
        f"when it finishes, and that they can stop it (job {short}). There is no "
        "result yet: do not invent one and do not describe what it might say."
    )


def make_job_tools(
    manager: JobManager,
    session_id: str,
    *,
    sync_timeout: float | None = None,
    approval_required: bool | None = None,
) -> list[Any]:
    """Build the job tools bound to one manager + one chat session.

    Both knobs are resolved once, here, rather than per call: they are a
    property of the deployment the session was built in, and re-reading the
    environment mid-conversation would make two turns of one conversation
    behave differently.
    """
    timeout = pick_sync_timeout(sync_timeout)
    wants_approval = pick_approval_required(approval_required)

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
        """Run a task on the job engine: research, analysis, reading files,
        anything needing several capability steps or material you do not have.

        Call it for ANY request you cannot answer well from your own
        knowledge. It runs the task now and normally answers within this
        turn; a task that turns out to be slow moves to the background on its
        own and is reported back here when it finishes. You never have to
        decide which of the two it will be, and you must not pretend to know.

        `query` is the task for the job engine, which does NOT see this
        conversation: write it so it stands entirely on its own. Resolve every
        referent ("that", "the second option", "same as before") into explicit
        words, and restate the subject, the constraints and what the answer
        should contain. The user is shown this exact wording before anything
        runs, so it must also read as a faithful statement of what they asked.

        `rationale` explains to the user in one line what the task will do.
        `inputs` carries structured material the job needs (image keys,
        ...); the recent conversation turns are attached automatically as
        background — never paste them into `query`.

        `source_files` names files the job must READ — a path the user gave,
        or the report a previous job of this conversation produced. Write each
        path exactly as it was given to you; never invent one, never guess at
        a directory, and never paste a file's contents into `query`. The user
        is shown this list and is handing those files over, so a path they did
        not mention has no business in it.

        `formats` decides WHETHER there is a file and which. Pass the formats
        the user asked for (`["markdown"]`, `["markdown", "pdf"]`, ...) when
        they want a document — a report, a file to keep, something to send or
        to print; the FIRST one is the main deliverable. Pass `[]` when they
        asked for an answer and explicitly no file. Leave it out when they
        said nothing about it, and the run decides: a task that actually
        researches or analyses something leaves a document, a question
        answered on the spot does not. Ask for what the user asked for and
        nothing more, and never promise a file in your own prose — only this
        argument produces one.

        `document_name` is what that file should be CALLED — a short filename
        with no extension, no directory and no spaces (`chair_comparison`).
        Use the user's own name when they gave one; when they did not, propose
        a short one from the subject rather than leaving it: the alternative is
        a file named after a job id. Do not name a document you asked for none
        of. `document_title` is the heading INSIDE the document, in the
        language of the request, and is a different decision — naming one
        never names the other. The user is shown all three before the run
        starts.

        When the task finishes in this turn, its answer is delivered to the
        user WORD FOR WORD by this tool: your own reply must then be at most
        one short sentence, and must never repeat, summarize or rephrase it.
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
            # `None` and `[]` are two different answers here (#84): "you
            # decide" and "no document". `formats or []` would collapse them.
            wanted = ensure_formats_available(formats)
            given = (document_name or "").strip()
            name = document_stem(given) if given else ""
        except ValueError as refused:
            return (f"NOT launched: {refused}. Nothing ran. Tell the user what "
                    f"is possible — formats available here: "
                    f"{', '.join(available_formats())} — and propose a launch again.")
        title = (document_title or "").strip()

        # What the run is about to do — the same payload either way, because
        # a notice and a question must show the user the same three things.
        about = {
            "query": query,
            "rationale": rationale,
            # what the document will be called, be titled, and be written as:
            # a file promised in prose and never written is what #55 is about,
            # and these are the only three things that produce one
            "document_name": name,
            "document_title": title,
            "formats": wanted,
            # the files it will be allowed to open — the same guard the query
            # gets, and for the stronger reason: this is the user handing
            # something over, not the model restating what they asked
            "sources": sources,
        }

        # The gate, when a deployment asked for it back. Off by default: the
        # agent says what it is doing rather than asking whether it may, and
        # cancellation is the undo (see the module docstring).
        if wants_approval:
            decision = interrupt(about | {
                "action": "launch_job",
                # what will travel besides the query, so a front-end can show
                # exactly what the user is approving
                "context": job_inputs.get(CONVERSATION_INPUT_KEY, ""),
            })
            if not (isinstance(decision, dict) and decision.get("approved")):
                return ("The user DECLINED the launch. Do not launch this job; "
                        "continue the conversation.")

        job = await manager.create_job(
            query, job_inputs, session_id=session_id,
            document_name=name, document_title=title, formats=wanted)
        write = _writer()
        # Said BEFORE anything runs, and carrying the job id the approval card
        # never had: this is what the user reads to catch a query whose
        # referent has gone, a file they did not hand over, or a document they
        # did not ask for — and the id is what makes stopping it possible.
        write({"event": CUSTOM_JOB_STARTED, "job_id": job.job_id} | about)

        # Background from the first instant, then waited on. Promotion is
        # "stop waiting", so nothing is cancelled and nothing is restarted:
        # the promoted run IS the run that was about to finish, and a turn
        # that dies (a UI cancelling its worker) does not take it with it.
        task = manager.start_job(job.job_id)
        done, _still_running = await asyncio.wait({task}, timeout=timeout)
        if not done:
            return _promoted(job, timeout)

        # `_drive` folds every ordinary failure into the record, so retrieving
        # the exception is about the extraordinary one — and about not leaving
        # an un-retrieved task exception behind whatever happened.
        if not task.cancelled() and (crashed := task.exception()) is not None:
            return (f"Job {job.job_id[:8]} crashed while running: {crashed}. "
                    "Tell the user it did not run.")

        settled = await manager.get_job(job.job_id) or job
        # Reported right here, so the completion notice does not announce, one
        # turn later, a job the user has already been handed the answer to.
        await manager.mark_announced(settled.job_id)
        if settled.final_answer:
            # Verbatim, into the turn itself. The trailing blank line keeps
            # the model's own sentence from running into the last one of the
            # answer — front-ends append tokens to a single growing answer.
            write({"event": CUSTOM_ANSWER, "text": settled.final_answer + "\n\n"})
        return _delivered(settled)

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
