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
    """Snapshot view of a job. `job_id` doubles as the LangGraph thread_id."""
    job_id: str
    status: JobStatus
    query: str
    inputs: dict[str, Any] = field(default_factory=dict)
    # What the requester asked the DOCUMENT to be (#55) — facts about the job,
    # decided once and recorded, never re-derived at write time. All three are
    # optional and each defaults on its own: a name does not decide a title,
    # and neither decides a format.
    #   document_name   filename stem, no extension and no separator (the
    #                   formats decide the extensions); "" ⇒ the job id
    #   document_title  the heading inside the document; "" ⇒ derived from the
    #                   request (`document_title()` in report.py, #54)
    #   formats         which Reporters render it, first one is the `main`
    #                   deliverable. Three states, and the third is #84's:
    #                   None ⇒ the request said nothing, so whoever writes it
    #                   decides; a non-empty list ⇒ exactly these; **[] ⇒ no
    #                   document at all**, which is a request a caller can now
    #                   make and which nothing could say before.
    document_name: str = ""
    document_title: str = ""
    formats: list[str] | None = None
    session_id: str | None = None           # chat session that launched it, if any
    created_at: str = ""                    # ISO timestamps
    updated_at: str = ""
    # What the run published while it ran (`engine/facts.py`): the graph's own
    # words, meaningless here. Values only in the full view (`load`), since
    # they can be large; when each arrived travels in every summary.
    facts: dict[str, Any] = field(default_factory=dict)
    facts_at: dict[str, str] = field(default_factory=dict)     # key → ISO ts
    steps: dict[str, str] = field(default_factory=dict)        # root node → ISO ts
    final_answer: str | None = None
    terminal_kind: str | None = None
    error: str | None = None
    # Was a deliverable meant to be written at all (#84)? False says the
    # absence of one is the *decision* and not a failure — the request asked
    # for no document (`formats == []`), or it said nothing about one
    # (`formats is None` once the engine's document step found nothing in
    # the sentence either, #96): silence is not a request for a file.
    # It is a fact about the ending, like `terminal_kind`, and it exists
    # because `report_path is None` already means two other things: the run
    # did not answer, and the write failed (`error` says which). A caller
    # that cannot tell those three apart tells the user the wrong one.
    # It only ever goes True → False: a job never gains a deliverable it
    # already said it would not write.
    deliverable_expected: bool = True
    announced: bool = False                 # completion surfaced in its chat session
    # What the run spent, all steps together (engine.usage.Usage.to_dict()).
    # Kept as a plain dict: it is persisted, served over HTTP and rendered as
    # is, and the per-step breakdown lives in each result's `meta["usage"]`.
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Full view, facts included."""
        return asdict(self) | {"status": self.status.value}

    def summary(self) -> dict[str, Any]:
        """The record stored in the ("jobs", "index") namespace."""
        return {
            "status": self.status.value,
            "query": self.query,
            "inputs": self.inputs,
            "document_name": self.document_name,
            "document_title": self.document_title,
            "formats": self.formats,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "facts_at": self.facts_at,
            "steps": self.steps,
            "terminal_kind": self.terminal_kind,
            "final_answer": self.final_answer,
            "error": self.error,
            "deliverable_expected": self.deliverable_expected,
            "announced": self.announced,
            "usage": self.usage,
        }
