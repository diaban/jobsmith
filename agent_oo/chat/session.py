"""ChatSession: the conversational front agent above the job engine.

Composition (all prebuilt LangGraph, no homemade tool plumbing):

    create_react_agent(model, job tools, prompt, pre_model_hook, checkpointer)

- "Complexity detection" IS the function calling: the system prompt tells the
  model to answer simple things directly and call `launch_job` for complex
  tasks. The tool interrupt()s for human approval (see chat/tools.py).
- Job completions are surfaced by `notify_finished_jobs`, a pre-model hook
  that injects a transient system notice (final answer + report path) for
  finished, unannounced jobs of this session, then marks them announced via
  the manager (persistence stays in the jobs layer). The notice is passed as
  `llm_input_messages` — ephemeral input; what persists in the conversation
  is the agent's own synthesis reply.
"""
from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

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

    # -------- pre-model hook --------

    @staticmethod
    def _notice_for(job: Job) -> str:
        if job.status is JobStatus.DONE:
            return (
                f"Job {job.job_id[:8]} ({job.query[:60]!r}) is DONE.\n"
                f"Report file: {job.report_path}\n"
                f"Full answer (synthesize it, do not paste it):\n{job.final_answer}"
            )
        return f"Job {job.job_id[:8]} ({job.query[:60]!r}) FAILED: {job.error}"

    async def notify_finished_jobs(self, state: dict) -> dict:
        finished = await self.manager.list_finished_unannounced(self.session_id)
        if not finished:
            # llm_input_messages is a persisted channel: reset it every turn,
            # or a previous turn's injected notice would be replayed.
            return {"llm_input_messages": list(state["messages"])}
        notices = []
        for job in finished:
            notices.append(self._notice_for(job))
            await self.manager.mark_announced(job.job_id)
        notice = SystemMessage(
            "[job update] The following background jobs finished. Announce each to "
            "the user now: a short synthesis plus the report file path.\n\n"
            + "\n\n".join(notices)
        )
        return {"llm_input_messages": list(state["messages"]) + [notice]}

    # -------- build --------

    def build(self):
        return create_react_agent(
            self.model,
            make_job_tools(self.manager, self.session_id),
            prompt=self.system_prompt,
            pre_model_hook=self.notify_finished_jobs,
            checkpointer=self.checkpointer,
        )
