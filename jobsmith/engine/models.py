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
    NEEDS_INPUT = "needs_input"             # paused at an interrupt, waiting for an answer


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
    asked: Any = None                       # what it asks while NEEDS_INPUT (JSON)
    # Where its ending goes (`engine/delivery.py`): JSON, `{"kind": …}`, and
    # the flat key its deliverer indexes it by; stamped once delivered.
    reply_to: dict[str, Any] = field(default_factory=lambda: {"kind": "none"})
    reply_key: str = "none"
    delivered_at: str | None = None
    attempt: int = 1                        # counted up by each resume: which ending this is
    created_at: str = ""                    # ISO timestamps
    updated_at: str = ""
    # What the run published while it ran (`engine/facts.py`): the graph's own
    # words, meaningless here. Values only in the full view (`load`), since
    # they can be large; when each arrived travels in every summary.
    facts: dict[str, Any] = field(default_factory=dict)
    facts_at: dict[str, str] = field(default_factory=dict)     # key → ISO ts
    steps: dict[str, str] = field(default_factory=dict)        # root node → ISO ts
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
            "asked": self.asked,
            "reply_to": self.reply_to,
            "reply_key": self.reply_key,
            "delivered_at": self.delivered_at,
            "attempt": self.attempt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # copies: a run keeps adding to these, and a store that keeps
            # the object it was given (memory) must not see it move (#171)
            "facts_at": dict(self.facts_at),
            "steps": dict(self.steps),
            "usage": self.usage,
        }
