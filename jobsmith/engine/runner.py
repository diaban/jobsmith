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


JobUpdate = NodeFinished | Fact | Output


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
        async for update in self._translate(None, job_id):
            yield update

    async def pending(self, job_id: str) -> tuple[str, ...]:
        """Nodes the thread would run next — what a resume would execute.

        Empty means there is nothing to resume: either no checkpoint exists
        (the run never started) or the graph already reached its end.
        """
        snapshot = await self.graph.aget_state(self._config(job_id))
        return tuple(snapshot.next or ())

    async def _translate(self, input: Any, job_id: str) -> AsyncIterator[JobUpdate]:
        """LangGraph's stream → the three updates above."""
        output: Any = None
        returned = False                    # a root `values` was seen
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
                for node in chunk:
                    if not node.startswith("__"):       # LangGraph's own markers
                        yield NodeFinished(node)
            elif mode == "values":
                output, returned = chunk, True
        if returned:
            yield Output(output)
