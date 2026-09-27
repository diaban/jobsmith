"""Job model: a persistent, trackable orchestration run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Snapshot view of a job. `job_id` doubles as the LangGraph thread_id.

    What the engine knows of a run, and nothing about what the run is for
    (docs/design/core-v1.md, "The record"): which graph ran, what it was
    given, what it returned, and what it published on the way.
    """
    job_id: str
    status: JobStatus
    graph: str = ""                         # the GraphSpec that runs it
    label: str = ""                         # the one line a listing shows
    input: Any = None                       # what the graph was given, as given
    result: Any = None                      # what `GraphSpec.result` made of its output
    error: str | None = None                # why it FAILED (never empty then)
    session_id: str | None = None           # chat session that launched it, if any
    created_at: str = ""                    # ISO timestamps
    updated_at: str = ""
    # What the run published while it ran (`engine/facts.py`): the graph's own
    # words, meaningless here. Values only in the full view (`load`), since
    # they can be large; when each arrived travels in every summary.
    facts: dict[str, Any] = field(default_factory=dict)
    facts_at: dict[str, str] = field(default_factory=dict)     # key → ISO ts
    steps: dict[str, str] = field(default_factory=dict)        # root node → ISO ts
    announced: bool = False                 # completion surfaced in its chat session
    # What the run spent, all steps together (engine.usage.Usage.to_dict()).
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Full view, facts included."""
        return asdict(self) | {"status": self.status.value}

    def summary(self) -> dict[str, Any]:
        """The record stored in the index namespace — everything but fact values."""
        return {
            "status": self.status.value,
            "graph": self.graph,
            "label": self.label,
            "input": self.input,
            "result": self.result,
            "error": self.error,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "facts_at": self.facts_at,
            "steps": self.steps,
            "announced": self.announced,
            "usage": self.usage,
        }
