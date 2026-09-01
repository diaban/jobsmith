"""ChatSession: the conversational front agent above the job engine.

Composition (all prebuilt LangChain/LangGraph, no homemade tool plumbing):

    create_agent(model, job tools, system_prompt, middleware, checkpointer)

- "Complexity detection" IS the function calling: the system prompt tells the
  model to answer simple things directly and call `launch_job` for complex
  tasks. The tool interrupt()s for human approval (see chat/tools.py).
- Job completions are surfaced by `JobNotificationMiddleware`, which wraps the
  model call and appends a transient system notice (final answer + report
  path) for finished, unannounced jobs of this session. Wrapping the *request*
  keeps the notice out of the persisted thread: what stays in the conversation
  is the agent's own synthesis reply.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage

from ..jobs.manager import JobManager
from ..jobs.models import Job, JobStatus
from .tools import make_job_tools

DEFAULT_CHAT_SYSTEM_PROMPT = """You are an assistant that can launch background jobs for complex tasks.

- Answer greetings, simple questions, and questions about what you can do directly.
- For any request needing research, analysis, or several processing steps, call
  launch_job: put the task in `query` and explain your reasoning in `rationale`
  (the user must approve the launch).
- Jobs run in the background — after launching one, keep chatting normally.
- When a [job update] notice appears, give the user a short synthesis
  (2-3 sentences) of the result and the path to the markdown report file.
- Use job_status / list_my_jobs / cancel_job to manage jobs when asked."""

NOTICE_MARKER = "background jobs finished"


class JobNotificationMiddleware(AgentMiddleware):
    """Injects finished-job notices into the model request, then marks them
    announced — only once the model has actually seen them."""

    def __init__(self, manager: JobManager, session_id: str):
        super().__init__()
        self.manager = manager
        self.session_id = session_id

    @staticmethod
    def _notice_for(job: Job) -> str:
        if job.status is JobStatus.DONE:
            return (
                f"Job {job.job_id[:8]} ({job.query[:60]!r}) is DONE.\n"
                f"Report file: {job.report_path}\n"
                f"Full answer (synthesize it, do not paste it):\n{job.final_answer}"
            )
        return f"Job {job.job_id[:8]} ({job.query[:60]!r}) FAILED: {job.error}"

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        finished = await self.manager.list_finished_unannounced(self.session_id)
        if not finished:
            return await handler(request)

        notice = SystemMessage(
            f"[job update] The following {NOTICE_MARKER}. Announce each to the "
            "user now: a short synthesis plus the report file path.\n\n"
            + "\n\n".join(self._notice_for(job) for job in finished)
        )
        response = await handler(request.override(messages=[*request.messages, notice]))
        for job in finished:
            await self.manager.mark_announced(job.job_id)
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
