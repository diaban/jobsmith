"""ChatSession: the conversational front agent above the job engine.

Composition (all prebuilt LangChain/LangGraph, no homemade tool plumbing):

    create_agent(model, job tools, system_prompt, middleware, checkpointer)

- "Complexity detection" IS the function calling: the system prompt tells the
  model to answer simple things directly and call `launch_job` for complex
  tasks. The tool interrupt()s for human approval (see chat/tools.py).
- Job completions AND in-flight progress are surfaced by
  `JobNotificationMiddleware`, which wraps the model call and injects transient
  system notices for this session's jobs. Wrapping the *request* keeps them out
  of the persisted thread: what stays in the conversation is the agent's own
  reply.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from ..jobs.manager import JobManager
from ..jobs.models import Job, JobStatus
from .tools import make_job_tools, progress_line, progress_signature

DEFAULT_CHAT_SYSTEM_PROMPT = """You are an assistant that can launch background jobs for complex tasks.

- Answer greetings, simple questions, and questions about what you can do directly.
- For any request needing research, analysis, or several processing steps, call
  launch_job: put the task in `query` and explain your reasoning in `rationale`
  (the user must approve the launch).
- Jobs run in the background — after launching one, keep chatting normally.
- When a [job update] notice appears, give the user a short synthesis
  (2-3 sentences) of the result and the path to the markdown report file.
- A [job progress] notice means a job is STILL RUNNING: there is no result yet.
  Use it to answer "how is it going?", or to add one short clause when it is
  genuinely useful ("(the research step is done, analysis is running)"). Never
  present it as an answer, and never make the whole reply about it.
- Use job_status / list_my_jobs / cancel_job to manage jobs when asked."""

NOTICE_MARKER = "background jobs finished"
PROGRESS_MARKER = "background jobs still running"

IN_FLIGHT = (JobStatus.QUEUED, JobStatus.RUNNING)
MAX_PROGRESS_JOBS = 5   # jobs detailed in one progress notice; the rest are counted


class JobNotificationMiddleware(AgentMiddleware):
    """Injects job notices into the model request, then records them as seen —
    only once the model has actually received them.

    Two notices, for two different needs:

    - **completion** must never be missed, so it is pushed and marked announced
      exactly once, persisted on the job itself;
    - **progress** is pushed too, but only on the turns where it *changed*
      (`progress_signature`), and remembered in memory only. Pushing it every
      turn would spend tokens to repeat yesterday's news; leaving it to a tool
      call would mean the model only knows when the user thinks to ask — and
      the whole point is that the wait is opaque. So: push the one-line digest
      when something moved, and keep `job_status` as the pull path for detail.

    Both ride on `request.override(...)` rather than state, so nothing
    accumulates in the persisted thread: a job reporting progress on five
    consecutive turns leaves zero notices behind it.
    """

    def __init__(self, manager: JobManager, session_id: str):
        super().__init__()
        self.manager = manager
        self.session_id = session_id
        # job_id → last progress signature the model was shown. In memory
        # because it is a conversational nicety, not a guarantee: after a
        # daemon restart the worst case is one repeated progress line.
        self._reported: dict[str, str] = {}

    @staticmethod
    def _notice_for(job: Job) -> str:
        if job.status is JobStatus.DONE:
            return (
                f"Job {job.job_id[:8]} ({job.query[:60]!r}) is DONE.\n"
                f"Report file: {job.report_path}\n"
                f"Full answer (synthesize it, do not paste it):\n{job.final_answer}"
            )
        return f"Job {job.job_id[:8]} ({job.query[:60]!r}) FAILED: {job.error}"

    async def _finished_notice(self) -> tuple[SystemMessage | None, list[Job]]:
        finished = await self.manager.list_finished_unannounced(self.session_id)
        if not finished:
            return None, []
        return SystemMessage(
            f"[job update] The following {NOTICE_MARKER}. Announce each to the "
            "user now: a short synthesis plus the report file path.\n\n"
            + "\n\n".join(self._notice_for(job) for job in finished)
        ), finished

    async def _in_flight(self) -> tuple[list[Job], int]:
        """The session's running/queued jobs, newest first, loaded in full.

        `list_jobs` returns index summaries, which carry the status and the
        finished steps but not the plan — so the few newest in-flight jobs are
        re-read in full, which is what lets the notice say "2/4 steps done"
        instead of just "running". The cap bounds both the store reads and the
        tokens: beyond it the notice only counts.
        """
        summaries = await self.manager.list_jobs(session_id=self.session_id, limit=100)
        in_flight = [job for job in summaries if job.status in IN_FLIGHT]
        loaded = [await self.manager.get_job(job.job_id) for job in in_flight[:MAX_PROGRESS_JOBS]]
        # A job can settle between the listing and the reload; leave it to the
        # completion notice rather than reporting it as still running.
        shown = [job for job in loaded if job is not None and job.status in IN_FLIGHT]
        return shown, max(len(in_flight) - len(shown), 0)

    async def _progress_notice(self) -> tuple[SystemMessage | None, list[Job]]:
        shown, others = await self._in_flight()
        moved = [
            job for job in shown
            if progress_signature(job) != self._reported.get(job.job_id)
        ]
        if not moved:
            return None, shown
        lines = [progress_line(job) for job in moved]
        if others:
            lines.append(f"(+{others} more still running)")
        return SystemMessage(
            f"[job progress] The following {PROGRESS_MARKER} — no results yet, "
            "do not announce them as finished. Mention the state only if the "
            "user asks or it is genuinely useful.\n"
            + "\n".join(lines)
        ), shown

    @staticmethod
    def _inject(messages: list[Any], notices: list[SystemMessage]) -> list[Any]:
        """Place transient notices directly after the leading system prompt.

        NOT a stylistic choice — a provider constraint, so do not "helpfully"
        move them later in the list. `langchain_anthropic._format_messages`
        raises "Received multiple non-consecutive system messages" for any
        SystemMessage that is not adjacent to the leading system block: a
        notice appended last, or slotted just before the final turn, kills the
        whole turn on Claude (the default provider whenever ANTHROPIC_API_KEY
        is set). Anthropic hoists every system message into the top-level
        `system` parameter anyway, so adjacency loses nothing there, and
        OpenAI accepts either placement.

        What the placement still buys, and why it is the right compromise: the
        conversation itself is left untouched, so the user's own message stays
        the last message — the turn being answered — and a status line is never
        mistaken for the thing to reply to. Order within the notices carries
        the rest of the intent: completion (an instruction to act on now)
        before progress (background awareness).
        """
        head = 0
        while head < len(messages) and isinstance(messages[head], SystemMessage):
            head += 1
        return [*messages[:head], *notices, *messages[head:]]

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        finished_notice, finished = await self._finished_notice()
        progress_notice, in_flight_shown = await self._progress_notice()
        notices = [n for n in (finished_notice, progress_notice) if n is not None]
        if not notices:
            return await handler(request)

        response = await handler(
            request.override(messages=self._inject(list(request.messages), notices))
        )
        # Only now that the model has actually seen them: mark completions
        # announced, and re-baseline progress (rebuilt from the in-flight set,
        # so a job that settles drops out of the map instead of lingering).
        for job in finished:
            await self.manager.mark_announced(job.job_id)
        self._reported = {job.job_id: progress_signature(job) for job in in_flight_shown}
        return response


class ChatSession:
    def __init__(
        self,
        manager: JobManager,
        model: Any,
        *,
        session_id: str | None = None,
        system_prompt: str = DEFAULT_CHAT_SYSTEM_PROMPT,
        checkpointer: Any = None,
    ):
        self.manager = manager
        self.model = model
        self.session_id = session_id or uuid.uuid4().hex
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

    def build(self):
        return create_agent(
            self.model,
            tools=make_job_tools(self.manager, self.session_id),
            system_prompt=self.system_prompt,
            middleware=[JobNotificationMiddleware(self.manager, self.session_id)],
            checkpointer=self.checkpointer,
        )
