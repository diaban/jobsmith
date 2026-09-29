"""ChatSession: the conversational front agent above the job engine.

Composition (all prebuilt LangChain/LangGraph, no homemade tool plumbing):

    create_agent(model, job tools, system_prompt, middleware, checkpointer)

- "Complexity detection" IS the function calling: the system prompt tells the
  model to answer simple things directly and call `launch_job` for anything
  that needs the engine. The tool runs the task in the turn and promotes it
  to the background when it turns out to be slow (see chat/tools.py) — the
  model decides *whether* the engine is needed, never *how long* it will take.
- Job completions AND in-flight progress are surfaced by
  `JobNotificationMiddleware`, which wraps the model call and injects transient
  system notices for this session's jobs. Wrapping the *request* keeps them out
  of the persisted thread: what stays in the conversation is the agent's own
  reply.
"""
from __future__ import annotations

import os
import sys
import uuid
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage

from ..adapters.langchain import JobDeliveryMiddleware
from ..adapters.langchain import JobProgressMiddleware as BaseProgressMiddleware
from ..dag.jobs import DagJob, DagJobs, session_address
from ..dag.state import TERMINAL_UNANSWERED
from ..engine.models import Job, JobStatus
from .runner import CUSTOM_ANSWER
from .tools import (
    make_job_tools,
    progress_line,
    progress_signature,
    stream_writer,
)

DEFAULT_CHAT_SYSTEM_PROMPT = """You are an assistant that runs real tasks on a job engine.

- Answer greetings, simple questions, and questions about what you can do directly.
- For any request needing research, analysis, reading a file, or several
  processing steps, call launch_job: put the task in `query` and one line on
  what it will do in `rationale`. Do not ask for permission first and do not
  guess how long it will take — the task runs now and moves to the background
  by itself if it turns out to be slow.
- When launch_job says the answer has already been shown to the user, it means
  exactly that: it was delivered word for word. Reply with at most one short
  sentence (naming the file is usually enough). Never repeat, summarize or
  rephrase the answer, and never comment on its content.
- When launch_job says the task moved to the background, say so plainly. There
  is no result yet: do not invent one.
- When a [job update] notice appears, follow what it says about each job: one
  whose answer has already been shown to the user gets at most one short
  sentence (never a repeat, a summary or a comment on it); one that hands you
  its answer gets a short synthesis (2-3 sentences) and the report file path.
- A [job progress] notice means a job is STILL RUNNING: there is no result yet.
  Use it to answer "how is it going?", or to add one short clause when it is
  genuinely useful ("(the research step is done, analysis is running)"). Never
  present it as an answer, and never make the whole reply about it.
- Use job_status / list_my_jobs / cancel_job to manage jobs when asked; a
  running task can be stopped, which is how a user undoes one.
- To leave part of a running task out ("skip the critique", "no slides after
  all"), call skip_steps: the job runs on without those steps, where
  cancel_job would stop all of it."""

NOTICE_MARKER = "background jobs finished"
PROGRESS_MARKER = "background jobs still running"

MAX_PROGRESS_JOBS = 5   # jobs detailed in one progress notice; the rest are counted

# How long an answer may be and still be handed to the reader *in the
# conversation* rather than as a path to a file (#85).
#
# Two thousand characters, ~300 words, because this is a **reading** budget
# and nothing else: it is roughly a screenful and a half of a terminal, which
# is what a person reads where they are standing before they would rather
# open a document and keep it. Below it, scrolling back up the transcript is
# how you re-read the answer; past it, the transcript stops being a place the
# answer can live and the file is the better container — which is the whole
# of what this issue asked to decide.
#
# It is deliberately not tuned to what a run cost or how long it took: a
# four-minute run can answer in two sentences and a ten-second one can return
# a table. The length of the *answer* is the only thing that bears on where
# it should be read, and it is known exactly at the moment the choice is
# taken — unlike #83's duration, which is why that one had to be observed by
# a clock and this one does not.
DEFAULT_INLINE_ANSWER_MAX = 2000


def pick_inline_answer_max(explicit: int | None = None) -> int:
    """Argument > `$JOBSMITH_INLINE_ANSWER_MAX` > 2 000 characters.

    The precedence of `pick_sync_timeout` / `pick_db` / `pick_search_depth`,
    minus the command-line flag, for the same reason: this is read inside a
    daemon a client cannot pass flags to. `0` is meaningful and supported —
    never write an answer into the conversation, i.e. exactly the behaviour
    this replaced — and it still cannot silence a run that produced **no
    file**, which is a rule about losing text rather than about length (see
    `_deliver`). A value that is not a number is said on stderr rather than
    quietly ignored.
    """
    if explicit is not None:
        return max(int(explicit), 0)
    raw = os.environ.get("JOBSMITH_INLINE_ANSWER_MAX")
    if not raw:
        return DEFAULT_INLINE_ANSWER_MAX
    try:
        return max(int(raw), 0)
    except ValueError:
        print(f"[chat: $JOBSMITH_INLINE_ANSWER_MAX={raw!r} is not a number — "
              f"using {DEFAULT_INLINE_ANSWER_MAX}]", file=sys.stderr)
        return DEFAULT_INLINE_ANSWER_MAX


class JobNotificationMiddleware(JobDeliveryMiddleware):
    """Tells the model of each ending of this session's jobs, in the DAG's
    words, with the answer handed to the reader first when it belongs in the
    conversation (#85).

    Exactly once in the thread is the adapter's (`adapters/langchain`,
    docs/design/core-v1.md G5): this class only says what each ending is.
    Progress is `JobProgressMiddleware`'s.
    """

    def __init__(
        self,
        manager: DagJobs,
        session_id: str,
        *,
        inline_answer_max: int | None = None,
    ):
        super().__init__(manager.engine, session_address(session_id))
        self.manager = manager
        self.inline_answer_max = pick_inline_answer_max(inline_answer_max)

    def _deliver(self, job: DagJob) -> bool:
        """Write a finished job's answer into the turn, verbatim, when the
        conversation is where it belongs (#85).

        A task that finishes **inside** the turn has had a verbatim channel
        since #83: `launch_job` writes `final_answer` onto the custom stream
        and the front-ends render it as the turn's answer. A task that was
        **promoted** to the background had none — its answer reached the model
        as an instruction to synthesize, so the only place the text survived
        word for word was the file. That is the asymmetry this closes, and it
        closes it with the channel that already exists rather than a second
        one: same payload, same reader (`chat/runner.py::_from_custom`), same
        `Token` on the way out. Nothing between the generator and the reader
        rewrites it, which is the property, not the mechanism.

        Two decisions live here. **Length decides**, because a transcript is
        a place you scroll and a document is a place you keep:
        `inline_answer_max` is the reading budget, and past it the path is
        the better answer. **Length only decides when there is something to
        decide between** — a run that wrote no file has nowhere else to put
        its text, so it is delivered whatever its length: the alternative is
        an answer that exists in no channel at all, which is the defect, not
        a policy about it.

        `stream_writer` crosses a module line on purpose: it is the ONE way
        into a turn that a tool or a middleware has (`get_stream_writer`,
        guarded for the case where nothing is listening), and a second copy of
        it here would be a second answer to "is anyone watching this run".

        The write happens *before* the model call it rides with, so the
        reader sees the answer and then the model's one sentence about it. A
        model call that then fails leaves the job unannounced and the answer
        shown — the same trade the notice itself already makes, and the
        harmless half of it.
        """
        answer = (job.final_answer or "").strip()
        if job.status is not JobStatus.DONE or not answer:
            return False
        if job.report_path and len(answer) > self.inline_answer_max:
            return False
        # The trailing blank line keeps the model's own sentence from running
        # into the last line of the answer: front-ends append tokens to one
        # growing answer (`chat/tools.py` writes it the same way).
        stream_writer()({"event": CUSTOM_ANSWER, "text": answer + "\n\n"})
        return True

    @staticmethod
    def _notice_for(job: DagJob, *, delivered: bool) -> str:
        """What the model is told about one finished job.

        A DONE job can have no main deliverable for two unrelated reasons,
        and telling the user the wrong one is the defect each branch exists
        to avoid. The run answered and only the **write failed**, which is
        why the manager keeps it DONE with `job.error` set — rendering
        `report_path` unconditionally then announces "Report file: None" and
        never says why, so the answer, the thing that survived, arrives next
        to a lie. Or **no document was ever asked for** (#84), where the
        write-failed wording would report a failure that did not happen. So
        the path is stated when there is one, the absence is named for what
        it is when there is not, and either way the answer is delivered —
        which is the whole point of staying DONE.

        A job that did NOT reach an answer can still have left files behind
        (#41: the manager collects them at every terminal). Announcing the
        end and saying nothing about them would recreate, inside the
        conversation, the very defect the branch above was fixed for — a file
        the user has no way to learn about. So they are named here too, as
        what they are: partial material from a run that did not finish, never
        a report.

        And it says which ending it was. A cancelled job is not a failed one —
        usually the model cancelled it *because the user asked* — and this
        text is the model's instruction for what to tell the user, so calling
        a stop a failure would put an untruth in the conversation. The same
        holds for a job that ran to the end and declared it could not answer
        (#59): it is DONE and it has a file, but announcing it like a job that
        answered is how someone who waited three minutes finds out only by
        reading the report. It gets its own branch, before the DONE one, and
        the instruction says plainly what the file is — an explanation of what
        was missing, not a result.

        `delivered` says whether `_deliver` has already written the text into
        the turn (#85). It changes one thing in the two branches that have an
        answer at all — whether the model is asked to relay it or told to keep
        its hands off it — and the wording is `chat/tools.py::_delivered`'s,
        deliberately: the reader cannot tell whether a task ran in the turn or
        came back from the background, and being told the same thing twice in
        two registers is how they would find out for no reason.
        """
        if job.status is not JobStatus.DONE:
            what = job.query[:60]
            lines = [
                f"Job {job.job_id[:8]} ({what!r}) was CANCELLED before it "
                f"finished, so it has no answer and no report."
                if job.status is JobStatus.CANCELLED else
                f"Job {job.job_id[:8]} ({what!r}) FAILED: {job.error}"
            ]
            if job.outputs:
                lines.append(
                    "Steps of this job still produced files before it stopped — "
                    "mention them as partial material, not as a report: "
                    + ", ".join(o.path for o in job.outputs))
            # A cancelled job carries no failure message, but it CAN carry a
            # delivery one (a step that declared a file it did not leave now
            # lands in `job.error` at every terminal). Dropping it here would
            # re-hide, in the conversation, what the manager just insisted on
            # saying out loud.
            if job.error and job.status is JobStatus.CANCELLED:
                lines.append(f"Also worth passing on: {job.error}")
            return "\n".join(lines)

        if job.terminal_kind == TERMINAL_UNANSWERED:
            lines = [
                f"Job {job.job_id[:8]} ({job.query[:60]!r}) finished but COULD NOT "
                "ANSWER: the material it gathered does not answer the request. "
                + ("Its explanation of what was missing has ALREADY been shown to "
                   "the user, in full — do not repeat or summarize it. Say plainly, "
                   "in one sentence, that there is no result."
                   if delivered else
                   "Tell the user plainly that it produced no answer, then relay "
                   "what was missing — never present the text below as a result.")
            ]
            if job.report_path:
                lines.append(
                    f"File explaining what was missing: {job.report_path}")
            if len(job.outputs) > 1:
                lines.append("Other files this job left: " + ", ".join(
                    o.path for o in job.outputs if o.path != job.report_path))
            if not delivered:
                lines.append(
                    f"What it reported (summarize, do not paste):\n{job.final_answer}")
            return "\n".join(lines)

        lines = [f"Job {job.job_id[:8]} ({job.query[:60]!r}) is DONE."]
        if delivered:
            lines.append(
                "Its answer has ALREADY been shown to the user, in full and word "
                "for word — do NOT repeat it, summarize it, rephrase it or comment "
                "on its content. Reply with at most one short sentence.")
        if job.report_path:
            lines.append(f"Report file: {job.report_path}")
        elif not job.deliverable_expected:
            # No file, and no failure: none was asked for (#84). The branch
            # below would report a write that never happened, which is the
            # same class of untruth as announcing a cancellation as a failure.
            lines.append("No file was written and none was asked for — do not "
                         "mention a document, and do not apologise for it.")
        else:
            reason = job.error or "the deliverable could not be written"
            lines.append(
                f"No report file was saved ({reason}) — say so."
                if delivered else
                f"No report file was saved ({reason}) — say so, then deliver the "
                "answer below anyway.")
            # A write can fail after earlier formats landed: those files
            # exist and are worth naming, even with the main one missing.
            if job.outputs:
                lines.append("Files that were written: "
                             + ", ".join(o.path for o in job.outputs))
        if not delivered:
            lines.append(
                f"Full answer (synthesize it, do not paste it):\n{job.final_answer}")
        return "\n".join(lines)

    async def notice(self, endings: list[Job]) -> SystemMessage:
        # In full: a notice says what each job produced. Delivered first, then
        # described: the answer is written into the turn before the model is
        # told what to say about it, so a reader sees the result and then the
        # sentence introducing it (#85).
        jobs = [DagJob(await self.jobs.get_job(job.job_id) or job) for job in endings]
        return SystemMessage(
            f"[job update] The following {NOTICE_MARKER}. Announce each to the "
            "user now, following the instruction each one carries: some have "
            "already delivered their answer to the user word for word and only "
            "need a sentence, others need you to relay what happened and name "
            "the file.\n\n"
            + "\n\n".join(self._notice_for(job, delivered=self._deliver(job))
                           for job in jobs)
        )


class JobProgressMiddleware(BaseProgressMiddleware):
    """Tells the model how this session's running jobs are doing, in the DAG's
    terms: steps of the plan done out of all, and what is running now
    (`progress_line`). Pushed only when that moved (`progress_signature`),
    never in the thread — the adapter's (`adapters/langchain`). Pull detail
    with `job_status`.
    """

    def __init__(self, manager: DagJobs, session_id: str):
        super().__init__(manager.engine, session_address(session_id),
                         max_jobs=MAX_PROGRESS_JOBS)
        self.manager = manager

    async def load(self, job: Job) -> DagJob | None:
        # In full: a summary has the finished steps but not the plan, and the
        # plan is what lets the line say "2/4 steps done" rather than "running".
        return await self.manager.get_job(job.job_id)

    def signature(self, job: Any) -> str:
        return progress_signature(job)

    def line(self, job: Any) -> str:
        return progress_line(job)

    async def notice(self, lines: list[str]) -> SystemMessage:
        return SystemMessage(
            f"[job progress] The following {PROGRESS_MARKER} — no results yet, "
            "do not announce them as finished. Mention the state only if the "
            "user asks or it is genuinely useful.\n"
            + "\n".join(lines))


class ChatSession:
    def __init__(
        self,
        manager: DagJobs,
        model: Any,
        *,
        session_id: str | None = None,
        system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
        checkpointer: Any = None,
        sync_timeout: float | None = None,
        approval_required: bool | None = None,
        inline_answer_max: int | None = None,
    ):
        self.manager = manager
        self.model = model
        self.session_id = session_id or uuid.uuid4().hex
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        # Both default to None, i.e. "ask the deployment" (`pick_*` in
        # chat/tools.py). Injectable because a test has to be able to force
        # either side of the clock, and because a composition root that wants
        # to decide for itself should not have to set an environment variable.
        self.sync_timeout = sync_timeout
        self.approval_required = approval_required
        self.inline_answer_max = inline_answer_max

    def build(self):
        return create_agent(
            self.model,
            tools=make_job_tools(self.manager, self.session_id,
                                 sync_timeout=self.sync_timeout,
                                 approval_required=self.approval_required),
            system_prompt=self.system_prompt,
            # Completion first (outermost), progress after it: an instruction
            # to act on now before background awareness.
            middleware=[JobNotificationMiddleware(
                            self.manager, self.session_id,
                            inline_answer_max=self.inline_answer_max),
                        JobProgressMiddleware(self.manager, self.session_id)],
            checkpointer=self.checkpointer,
        )
