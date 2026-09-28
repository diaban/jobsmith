"""Launching a job from a conversation: answered in the turn when it is quick,
promoted to the background when it is not (0083), and told once either way.

The tool runs the job for up to `timeout` (`JobManager.run_for`). An ending
reached in the turn goes into the thread as the tool's result, WITH its
entry in `DELIVERED_CHANNEL`, in one update (`told_in_thread`) — so the
completion notice never tells it again. A job still running when the clock
runs out is promoted: nothing is stopped, and its ending is told later by
`JobDeliveryMiddleware`, which pulls what is addressed to the same thread.

What the job IS comes from the application: the graph it runs, the arguments
the model fills (`args_schema`), and how they become the graph's input.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, create_model

from ...engine.manager import JobManager
from ...engine.models import Job, JobStatus, now_iso
from .delivery import DELIVERED_CHANNEL

IN_FLIGHT = (JobStatus.QUEUED, JobStatus.RUNNING)


def told_in_thread(job_id: str, text: str, tool_call_id: str) -> Command:
    """A launch tool's result for a job that ended in the turn: the message,
    and the thread's record that this ending was told, as one update."""
    return Command(update={
        "messages": [ToolMessage(text, tool_call_id=tool_call_id)],
        DELIVERED_CHANNEL: {job_id: now_iso()},
    })


def ended(job: Job) -> str:
    """The default result of a job that ended in the turn."""
    if job.status is JobStatus.DONE:
        return f"Job {job.job_id[:8]} is DONE. Its result: " + json.dumps(
            job.result, ensure_ascii=False)
    return f"Job {job.job_id[:8]} is {job.status.value.upper()}: {job.error or 'no reason given'}"


def promoted(job: Job) -> str:
    """The default result of a job still running when the turn stopped waiting."""
    return (f"Job {job.job_id[:8]} is still running and moved to the background: "
            "there is no result yet, do not invent one. You will be told when it ends.")


def launch_tool(
    jobs: JobManager,
    *,
    reply_to: dict[str, Any],
    args_schema: type[BaseModel],
    description: str,
    name: str = "launch_job",
    graph: str | None = None,
    input_of: Callable[[dict[str, Any]], Any] = lambda args: args,
    label_of: Callable[[dict[str, Any]], str] = lambda args: "",
    timeout: float = 20.0,
    told: Callable[[Job], str] = ended,
    still_running: Callable[[Job], str] = promoted,
) -> BaseTool:
    """A tool that runs `graph` on the model's arguments, addressed to `reply_to`."""
    schema = create_model(f"{args_schema.__name__}Call", __base__=args_schema,
                          tool_call_id=(Annotated[str, InjectedToolCallId], ...))

    async def launch(tool_call_id: str, **args: Any) -> str | Command:
        job = await jobs.create_job(input_of(args), graph=graph, label=label_of(args),
                                    reply_to=reply_to)
        job = await jobs.run_for(job.job_id, timeout)
        if job.status in IN_FLIGHT:
            return still_running(job)
        return told_in_thread(job.job_id, told(job), tool_call_id)

    return StructuredTool.from_function(coroutine=launch, name=name, description=description,
                                        args_schema=schema)
