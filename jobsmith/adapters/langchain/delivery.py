"""A LangChain agent told of each ending of its jobs, exactly once in its thread.

The engine delivers **at least once, keyed by job id** (`engine/delivery.py`);
this side makes it exactly once where it counts, the conversation. The jobs
are addressed to the agent's thread (`reply_to`, a kind the engine serves by
`Pulled`), and every model call pulls what ended since.

The guarantee rests on one channel of the thread's own state,
`DELIVERED_CHANNEL`: job id → when it was told. An id enters it in the SAME
update as the message that told the model — its answer to a notice
(`ExtendedModelResponse`), or a launch tool's result — so it is in the
checkpoint exactly when that message is. An ending already there is never
told again; one LATER than its stamp (the job was resumed since) is news.
The job's `delivered_at` is only the index of the channel, marked after the
model's answer is checkpointed (`aafter_model`) and so never ahead of it.

A crash before that checkpoint tells the job again on the next call — the
turn that died left nothing in the thread. A crash after it leaves the mark
to the next call's `aafter_model`, and tells nothing. What is left is a turn
streamed to a screen that died before its checkpoint, which the next rewrites.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, ExtendedModelResponse
from langchain_core.messages import SystemMessage
from langgraph.types import Command

from ...engine.manager import JobManager
from ...engine.models import Job, JobStatus, now_iso

DELIVERED_CHANNEL = "delivered_jobs"


def _merge(record: dict[str, str], more: dict[str, str]) -> dict[str, str]:
    return {**record, **more}


class DeliveredState(AgentState):
    """The thread's record of the endings it was told of: job id → when."""

    delivered_jobs: NotRequired[Annotated[dict[str, str], _merge]]


def told(job: Job | Any, record: dict[str, str]) -> bool:
    """This ending of the job is already in the thread."""
    return record.get(job.job_id, "") > job.updated_at


def inject(messages: list[Any], notices: list[SystemMessage]) -> list[Any]:
    """Place transient notices directly after the leading system prompt.

    NOT a stylistic choice — a provider constraint, so do not "helpfully" move
    them later in the list. `langchain_anthropic._format_messages` raises
    "Received multiple non-consecutive system messages" for any SystemMessage
    that is not adjacent to the leading system block: a notice appended last,
    or slotted just before the final turn, kills the whole turn on Claude.
    Anthropic hoists every system message into the top-level `system`
    parameter anyway, so adjacency loses nothing there, and OpenAI accepts
    either placement.

    What the placement still buys: the conversation itself is left untouched,
    so the user's own message stays the last one — the turn being answered —
    and a status line is never mistaken for the thing to reply to. Notices
    injected by an outer middleware come first.
    """
    head = 0
    while head < len(messages) and isinstance(messages[head], SystemMessage):
        head += 1
    return [*messages[:head], *notices, *messages[head:]]


class JobDeliveryMiddleware(AgentMiddleware):
    """Tells the model of each ending of the jobs addressed to `reply_to`.

    The notice rides on the model REQUEST (`request.override`), never in the
    thread's messages: only the channel records it, so nothing accumulates.
    Override `notice` to say it in an application's own words.
    """

    state_schema = DeliveredState

    def __init__(self, jobs: JobManager, reply_to: dict[str, Any]):
        super().__init__()
        self.jobs = jobs
        self.reply_to = reply_to

    async def notice(self, endings: list[Job]) -> SystemMessage:
        """What the model is told of these endings (summaries, newest first)."""
        lines = []
        for job in endings:
            line = f"Job {job.job_id[:8]} ({job.label[:60]!r}) is {job.status.value.upper()}."
            if job.status is JobStatus.DONE:
                line += f" Its result: {json.dumps(job.result, ensure_ascii=False)[:2000]}"
            elif job.error:
                line += f" Why: {job.error}"
            lines.append(line)
        return SystemMessage("[job update] These jobs you launched have ended. Tell the "
                             "user now, from what each one says.\n" + "\n".join(lines))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        record = (request.state or {}).get(DELIVERED_CHANNEL) or {}
        # Told already, in a turn whose mark never came: say nothing —
        # `aafter_model` marks it once this call is checkpointed.
        endings = [job for job in await self.jobs.pending_deliveries(self.reply_to)
                   if not told(job, record)]
        if not endings:
            return await handler(request)
        message = await self.notice(endings)
        response = await handler(
            request.override(messages=inject(list(request.messages), [message])))
        now = now_iso()
        return ExtendedModelResponse(model_response=response, command=Command(
            update={DELIVERED_CHANNEL: {job.job_id: now for job in endings}}))

    async def aafter_model(self, state: Any, runtime: Any) -> None:
        """The model's answer is checkpointed: index what the thread now records."""
        record = state.get(DELIVERED_CHANNEL) or {}
        if record:
            for job in await self.jobs.pending_deliveries(self.reply_to):
                if told(job, record):
                    await self.jobs.mark_delivered(job.job_id)
        return None
