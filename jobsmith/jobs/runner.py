"""Driving a graph run, and translating it into domain updates.

This is the **only** module that knows the shape of LangGraph's
`astream(stream_mode="updates")` events — `{node_name: state_update}`, node
names like `cap_<capability>`, terminal node names. Everything above it reacts
to the small typed updates below, so the JobManager never parses graph output
and a test can drive it with a fake runner.

Reading progress from the stream (rather than instrumenting nodes) is what
keeps graph nodes job-agnostic: they do not know a Job exists.
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

    async def stream(
        self, job_id: str, query: str, inputs: dict[str, Any]
    ) -> AsyncIterator[JobUpdate]:
        # job_id doubles as the LangGraph thread_id: the checkpoint of a
        # cancelled or interrupted run stays addressable for a future resume.
        async for update in self.graph.astream(
            {"query": query, "inputs": inputs, "job_id": job_id},
            config={"configurable": {"thread_id": job_id}},
            stream_mode="updates",
        ):
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
