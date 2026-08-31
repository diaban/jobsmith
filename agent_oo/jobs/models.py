"""Job model: a persistent, trackable orchestration run."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..core.state import CapabilityResult, Plan


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


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
    report_path: str | None = None          # markdown report written on completion
    announced: bool = False                 # completion surfaced in its chat session

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
            "report_path": self.report_path,
            "announced": self.announced,
        }
