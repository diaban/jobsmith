"""Job model: a persistent, trackable orchestration run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..core.state import CapabilityResult, Plan


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobOutput:
    """A file the job produced FOR THE HUMAN — the deliverable.

    A job can have several: the main report plus annexes (a chart a
    capability drew, an exported table), and later the same content in
    other formats. `role` is "main" or "annex"; `format` is free-form
    ("markdown", "html", "pdf", ...).
    """

    path: str
    format: str = "markdown"
    title: str = ""
    role: str = "main"
    produced_by: str | None = None      # capability name, when a step made it

    @property
    def name(self) -> str:
        return Path(self.path).name


@dataclass
class Job:
    """Snapshot view of a job. `job_id` doubles as the LangGraph thread_id."""
    job_id: str
    status: JobStatus
    query: str
    inputs: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None           # chat session that launched it, if any
    created_at: str = ""                    # ISO timestamps
    updated_at: str = ""
    plan: Plan | None = None
    results: dict[str, CapabilityResult] = field(default_factory=dict)
    step_finished_at: dict[str, str] = field(default_factory=dict)  # cap name → ISO ts
    final_answer: str | None = None
    terminal_kind: str | None = None
    error: str | None = None
    outputs: list[JobOutput] = field(default_factory=list)   # the deliverables
    announced: bool = False                 # completion surfaced in its chat session
    # What the run spent, all steps together (core.usage.Usage.to_dict()).
    # Kept as a plain dict: it is persisted, served over HTTP and rendered as
    # is, and the per-step breakdown lives in each result's `meta["usage"]`.
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def report_path(self) -> str | None:
        """Path of the main deliverable (kept as the common shortcut)."""
        main = next((o for o in self.outputs if o.role == "main"), None)
        return main.path if main else None

    def step_usage(self, capability: str) -> dict[str, Any]:
        """What one step spent — empty when it made no LLM call, or predates
        usage tracking."""
        return ((self.results.get(capability) or {}).get("meta") or {}).get("usage") or {}

    def to_dict(self) -> dict[str, Any]:
        """Full view for API/CLI consumers (asdict would drop the properties)."""
        return asdict(self) | {"status": self.status.value, "report_path": self.report_path}

    def summary(self) -> dict[str, Any]:
        """The record stored in the ("jobs", "index") namespace."""
        return {
            "status": self.status.value,
            "query": self.query,
            "inputs": self.inputs,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "step_finished_at": self.step_finished_at,
            "terminal_kind": self.terminal_kind,
            "final_answer": self.final_answer,
            "error": self.error,
            "outputs": [asdict(o) for o in self.outputs],
            "report_path": self.report_path,      # derived, for consumers
            "announced": self.announced,
            "usage": self.usage,
        }
