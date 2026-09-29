"""Driving a graph run, and translating it into job updates.

This is the **only** module that knows the shape of LangGraph's stream, and
it knows nothing about the graph it drives (docs/design/core-v1.md, "The
contract a graph signs"). It streams three modes, sub-graphs included, and
yields three updates:

    NodeFinished   a node of the ROOT graph finished — its name, no payload
    Fact           a node, at any depth, published `key = value`
                   (`engine/facts.py`)
    Output         what the run returned: the last root `values`, restricted
                   to the graph's output channels — exactly what `ainvoke`
                   would have returned. Only a run that completed has one.
    Interrupted    the run paused at a LangGraph `interrupt()` instead: what
                   it asks, and no Output (#167)

A sub-graph's own steps and states are its business and never surface; its
facts do, because `subgraphs=True` is what carries them to the parent's
stream (checked on langgraph 1.2.11: without it they are lost).

Reading progress from the stream (rather than instrumenting nodes) is what
keeps graph nodes job-agnostic: they do not know a Job exists.

There are two ways in — `stream()` starts a run from its input, `resume()`
re-enters the thread's checkpoint — and both are translated by the same code,
so the JobManager folds a resumed run exactly like a first one.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langgraph.types import Command

from .facts import FACT_KEY
from .usage import ModelCallUsage


@dataclass(frozen=True)
class NodeFinished:
    """A node of the root graph finished."""
    node: str


@dataclass(frozen=True)
class Fact:
    """A node published a named fact (`engine/facts.publish`)."""
    key: str
    value: Any


@dataclass(frozen=True)
class Output:
    """The run completed and returned this."""
    value: Any


@dataclass(frozen=True)
class Interrupted:
    """The run paused at an `interrupt()`: what it asks, one value per interrupt."""
    asked: list[Any]


JobUpdate = NodeFinished | Fact | Output | Interrupted

# LangGraph's control channels in a task's pending writes (private in
# `langgraph._internal._constants`, checked on 1.2.11): what a task writes
# about itself, not output. `_ERROR` is how a stop or a failure is kept.
_ERROR = "__error__"
_CONTROL = {_ERROR, "__interrupt__", "__resume__", "__error_source_node__"}


class GraphRunner:
    """Runs a job's graph and yields what happened, in job terms."""

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def _config(job_id: str) -> dict[str, Any]:
        # job_id doubles as the LangGraph thread_id: the checkpoint of a
        # cancelled or interrupted run stays addressable for a future resume.
        # The callback books what LangChain models spend (`usage.py`).
        return {"configurable": {"thread_id": job_id}, "callbacks": [ModelCallUsage()]}

    async def stream(self, job_id: str, input: Any) -> AsyncIterator[JobUpdate]:
        """Start a run from `input`, which reaches the graph as it is."""
        async for update in self._translate(input, job_id):
            yield update

    async def _repaired(self, job_id: str) -> None:
        """Before re-entering a checkpoint: if a stop caught a task between its
        writes (`_half_written`), fork the checkpoint without the writes it
        holds, so that task runs again instead of being counted as done (#202).
        Every task of that superstep then runs again, the finished ones
        included: the price of repairing it through LangGraph's public API."""
        if await self._half_written(await self.graph.aget_state(self._config(job_id))):
            await self.graph.aupdate_state(self._config(job_id), None, as_node="__copy__")

    async def _half_written(self, snapshot: Any) -> tuple[str, ...]:
        """Tasks a stop (or an error) caught after they wrote part of their
        output — their state, say, but not the edge to the next node. LangGraph
        (checked on 1.2.11) persists that part with the error, and a resume
        then re-applies it, counts the task as done and never writes the edge:
        the graph ends there, and `next` already says nothing is left (#202)."""
        saved = await self.graph.checkpointer.aget_tuple(snapshot.config)
        channels: dict[str, set[str]] = {}
        for task_id, channel, _ in (saved.pending_writes if saved else None) or ():
            channels.setdefault(task_id, set()).add(channel)
        return tuple(task.name for task in snapshot.tasks
                     if _ERROR in (written := channels.get(task.id, set()))
                     and written - _CONTROL)

    async def resume(self, job_id: str) -> AsyncIterator[JobUpdate]:
        """Re-enter the thread's checkpoint instead of starting a new run.

        `None` as input is LangGraph's "carry on from where you stopped": the
        last completed superstep is replayed from the checkpoint and only the
        tasks that were still pending are executed. A node interrupted
        mid-flight therefore runs again from its start, while the ones that
        had already finished are *not* re-emitted — which is why the caller
        must keep what it loaded from the repository.

        Only call this when `pending()` is non-empty: on a thread with no
        checkpoint LangGraph raises (it has no input to start from).
        """
        await self._repaired(job_id)
        async for update in self._translate(None, job_id):
            yield update

    async def answer(self, job_id: str, answer: Any) -> AsyncIterator[JobUpdate]:
        """Re-enter a run paused at an `interrupt()`, which returns `answer`."""
        await self._repaired(job_id)
        async for update in self._translate(Command(resume=answer), job_id):
            yield update

    async def update(self, job_id: str, values: dict[str, Any]) -> None:
        """Write `values` into the thread's checkpoint, through the graph's own
        reducers — the next run from it starts from what they make of it."""
        await self._repaired(job_id)
        await self.graph.aupdate_state(self._config(job_id), values)

    async def pending(self, job_id: str) -> tuple[str, ...]:
        """Nodes the thread would run next — what a resume would execute.

        Empty means there is nothing to resume: either no checkpoint exists
        (the run never started) or the graph already reached its end. A task
        a stop caught between its writes is pending too (`_half_written`).
        """
        snapshot = await self.graph.aget_state(self._config(job_id))
        return tuple(dict.fromkeys((*(snapshot.next or ()), *await self._half_written(snapshot))))

    async def _translate(self, input: Any, job_id: str) -> AsyncIterator[JobUpdate]:
        """LangGraph's stream → the three updates above."""
        output: Any = None
        returned = False                    # a root `values` was seen
        asked: list[Any] | None = None      # the run paused: what it asks
        async for namespace, mode, chunk in self.graph.astream(
            input,
            config=self._config(job_id),
            stream_mode=["updates", "custom", "values"],
            subgraphs=True,
            output_keys=self.graph.output_channels,
        ):
            if mode == "custom":
                if isinstance(chunk, dict) and FACT_KEY in chunk:
                    yield Fact(chunk[FACT_KEY], chunk.get("value"))
            elif namespace:
                continue                    # a sub-graph's own steps and states
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    asked = [pause.value for pause in chunk["__interrupt__"]]
                for node in chunk:
                    if not node.startswith("__"):       # LangGraph's own markers
                        yield NodeFinished(node)
            elif mode == "values":
                output, returned = chunk, True
        if asked is not None:
            # Its last `values` carries the interrupts, not a result: a paused
            # run returned nothing, and it read as "its result is not JSON".
            yield Interrupted(asked)
        elif returned:
            yield Output(output)
