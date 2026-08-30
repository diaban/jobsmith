"""Job model: a persistent, trackable orchestration run."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..core.state import CapabilityResult, Plan


class JobStatus(str, Enum):
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
    created_at: str = ""                    # ISO timestamps
    updated_at: str = ""
    plan: Optional[Plan] = None
    results: dict[str, CapabilityResult] = field(default_factory=dict)
    final_answer: Optional[str] = None
    terminal_kind: Optional[str] = None
    error: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        """The record stored in the ("jobs", "index") namespace."""
        return {
            "status": self.status.value,
            "query": self.query,
            "inputs": self.inputs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "terminal_kind": self.terminal_kind,
            "final_answer": self.final_answer,
            "error": self.error,
        }
