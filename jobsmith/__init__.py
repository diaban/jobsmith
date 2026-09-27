"""jobsmith — run any LangGraph graph as a durable job, and the planner DAG built on it.

The job engine (`jobsmith.engine`) runs a graph as a persistent, trackable,
cancellable, resumable Job: the input reaches the graph as given, what it
returns becomes the job's result, and the facts it publishes on the way are
recorded as they come (docs/design/core-v1.md).

    jobs = JobManager(GraphSpec("sum", my_graph), store)
    job = await jobs.create_job({"a": 1, "b": 2}, label="1 + 2")
    job = await jobs.run_job(job.job_id)          # job.result == what it returned

The planner DAG (`jobsmith.dag`) is one graph shipped with it: a registry-driven
planner emits a DAG of pluggable capabilities, a wave-based executor runs them,
a generation pipeline merges their results into an answer.

    registry = CapabilityRegistry([MyCapability(...), ...])
    graph = AgentBuilder(Deps(llm=my_llm), registry,
                         profile=AgentProfile(), checkpointer=...).build()
    jobs = DagJobs(JobManager(dag_spec(graph), store))
    job = await jobs.create_job("do something", inputs={...})

See jobsmith/agents/banking for a complete domain agent.

Names are imported on first use, so importing the engine never loads the DAG.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                       # the names, for editors and type checkers
    from .dag.builder import AgentBuilder, build_agent
    from .dag.capability import (
        Capability,
        CapabilityBaseState,
        CapabilityOutputState,
        CapabilitySpec,
    )
    from .dag.deps import Deps, LLMClient
    from .dag.document import DocumentIntent
    from .dag.jobs import DagJobs, dag_spec
    from .dag.profile import AgentProfile
    from .dag.registry import CapabilityRegistry
    from .dag.router import Router
    from .dag.state import AgentState, CapabilityResult, NodeError, Plan, PlanStep
    from .engine.graph import GraphSpec, JobFailed
    from .engine.manager import JobManager
    from .engine.models import Job, JobStatus


_EXPORTS = {
    "AgentBuilder": "dag.builder", "build_agent": "dag.builder",
    "Capability": "dag.capability", "CapabilityBaseState": "dag.capability",
    "CapabilityOutputState": "dag.capability", "CapabilitySpec": "dag.capability",
    "Deps": "dag.deps", "LLMClient": "dag.deps",
    "DocumentIntent": "dag.document",
    "AgentProfile": "dag.profile",
    "CapabilityRegistry": "dag.registry",
    "Router": "dag.router",
    "AgentState": "dag.state", "CapabilityResult": "dag.state", "NodeError": "dag.state",
    "Plan": "dag.state", "PlanStep": "dag.state",
    "DagJobs": "dag.jobs", "dag_spec": "dag.jobs",
    "JobManager": "engine.manager",
    "Job": "engine.models", "JobStatus": "engine.models",
    "GraphSpec": "engine.graph", "JobFailed": "engine.graph",
}

__all__ = [
    "AgentBuilder",
    "AgentProfile",
    "AgentState",
    "Capability",
    "CapabilityBaseState",
    "CapabilityOutputState",
    "CapabilityRegistry",
    "CapabilityResult",
    "CapabilitySpec",
    "DagJobs",
    "Deps",
    "DocumentIntent",
    "GraphSpec",
    "Job",
    "JobFailed",
    "JobManager",
    "JobStatus",
    "LLMClient",
    "NodeError",
    "Plan",
    "PlanStep",
    "Router",
    "build_agent",
    "dag_spec",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'jobsmith' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)
