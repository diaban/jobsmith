"""jobsmith — domain-agnostic agent planner/executor framework on LangGraph.

A user message comes in; a registry-driven planner emits a DAG of pluggable
capabilities (self-describing agentic sub-graphs); a wave-based executor runs
them in parallel where possible; a generation pipeline merges their results
into an answer. Each run is a persistent, trackable, cancellable Job.

Quick start:

    registry = CapabilityRegistry([MyCapability(...), ...])
    graph = AgentBuilder(Deps(llm=my_llm), registry,
                         profile=AgentProfile(), checkpointer=...).build()
    jobs = JobManager(graph, store)
    job = await jobs.create_job("do something", inputs={...})
    job = await jobs.run_job(job.job_id)

See jobsmith/agents/banking for a complete domain agent.
"""
from .dag.builder import AgentBuilder, build_agent
from .dag.capability import (
    Capability,
    CapabilityBaseState,
    CapabilityOutputState,
    CapabilitySpec,
)
from .dag.deps import Deps, LLMClient
from .dag.document import DocumentIntent
from .dag.profile import AgentProfile
from .dag.registry import CapabilityRegistry
from .dag.router import Router
from .dag.state import AgentState, CapabilityResult, NodeError, Plan, PlanStep
from .engine.manager import JobManager
from .engine.models import Job, JobStatus

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
    "Deps",
    "DocumentIntent",
    "Job",
    "JobManager",
    "JobStatus",
    "LLMClient",
    "NodeError",
    "Plan",
    "PlanStep",
    "Router",
    "build_agent",
]
