"""Driving a graph run, and translating it into domain updates.

This is the **only** module that knows the shape of LangGraph's
`astream(stream_mode="updates")` events — `{node_name: state_update}`, node
names like `cap_<capability>`, terminal node names. Everything above it reacts
to the small typed updates below, so the JobManager never parses graph output
and a test can drive it with a fake runner.

Reading progress from the stream (rather than instrumenting nodes) is what
keeps graph nodes job-agnostic: they do not know a Job exists.

There are two ways in — `stream()` starts a run from the job's query,
`resume()` re-enters the thread's checkpoint — and both are translated by the
same code, so the JobManager folds a resumed run exactly like a first one.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..core.state import CapabilityResult, NodeError, Plan

_TERMINAL_NODES = ("post_process", "escalate", "user_error")


@dataclass(frozen=True)
class PlanReady:
    """The planner validated a DAG."""
    plan: Plan


@dataclass(frozen=True)
class StepFinished:
    """One capability produced its result (ok or not)."""
    capability: str
    result: CapabilityResult


@dataclass(frozen=True)
class NodeErrors:
    """Errors a node accumulated; recoverable ones do not stop the run."""
    errors: list[NodeError]


@dataclass(frozen=True)
class Terminal:
    """The run reached a terminal node."""
    terminal_kind: str | None
    final_answer: str | None
    user_error_message: str | None


JobUpdate = PlanReady | StepFinished | NodeErrors | Terminal


class GraphRunner:
    """Runs a job's graph and yields what happened, in domain terms."""

    def __init__(self, graph: Any):
        self.graph = graph

    @staticmethod
    def _config(job_id: str) -> dict[str, Any]:
        # job_id doubles as the LangGraph thread_id: the checkpoint of a
        # cancelled or interrupted run stays addressable for a future resume.
        return {"configurable": {"thread_id": job_id}}

    async def stream(
        self, job_id: str, query: str, inputs: dict[str, Any]
    ) -> AsyncIterator[JobUpdate]:
        async for update in self._translate(self.graph.astream(
            {"query": query, "inputs": inputs, "job_id": job_id},
            config=self._config(job_id),
            stream_mode="updates",
        )):
            yield update

    async def resume(self, job_id: str) -> AsyncIterator[JobUpdate]:
        """Re-enter the thread's checkpoint instead of starting a new run.

        `None` as input is LangGraph's "carry on from where you stopped": the
        last completed superstep is replayed from the checkpoint and only the
        tasks that were still pending are executed. A capability interrupted
        mid-flight therefore runs again from its start, while the steps that
        had already finished are *not* re-emitted — which is why the caller
        must keep the results it loaded from the repository.

        Only call this when `pending()` is non-empty: on a thread with no
        checkpoint LangGraph raises (it has no input to start from).
        """
        async for update in self._translate(
            self.graph.astream(None, config=self._config(job_id), stream_mode="updates")
        ):
            yield update

    async def pending(self, job_id: str) -> tuple[str, ...]:
        """Nodes the thread would run next — what a resume would execute.

        Empty means there is nothing to resume: either no checkpoint exists
        (the run never started) or the graph already reached a terminal node.
        """
        snapshot = await self.graph.aget_state(self._config(job_id))
        return tuple(snapshot.next or ())

    async def _translate(self, stream: AsyncIterator[dict]) -> AsyncIterator[JobUpdate]:
        """LangGraph `updates` events → the domain updates above."""
        async for update in stream:
            for node, value in update.items():
                if not isinstance(value, dict):
                    continue
                if value.get("errors"):
                    yield NodeErrors(list(value["errors"]))
                if node == "planner" and value.get("plan"):
                    yield PlanReady(value["plan"])
                elif node.startswith("cap_"):
                    for capability, result in (value.get("results") or {}).items():
                        yield StepFinished(capability, result)
                elif node in _TERMINAL_NODES:
                    yield Terminal(
                        value.get("terminal_kind"),
                        value.get("final_answer"),
                        value.get("user_error_message"),
                    )
