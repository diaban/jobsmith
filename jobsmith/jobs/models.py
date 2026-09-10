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

    A job can have several. `role` says what each one is:

    - "main"      the deliverable, exactly one per job — what
                  `report_path`, `jobsmith report` and `/report` point at
    - "alternate" the same report rendered in another format (asked for
                  with `JOBSMITH_REPORT_FORMAT=markdown,html`)
    - "annex"     per-step material a capability produced (a chart, an
                  exported table) — supporting material, not the report

    `format` is free-form ("markdown", "html", "pdf", ...); `produced_by`
    names the capability, when a step made the file.
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
    # What the requester asked the DOCUMENT to be (#55) — facts about the job,
    # decided once and recorded, never re-derived at write time. All three are
    # optional and each defaults on its own: a name does not decide a title,
    # and neither decides a format.
    #   document_name   filename stem, no extension and no separator (the
    #                   formats decide the extensions); "" ⇒ the job id
    #   document_title  the heading inside the document; "" ⇒ derived from the
    #                   request (`document_title()` in report.py, #54)
    #   formats         which Reporters render it, first one is the `main`
    #                   deliverable; empty ⇒ whatever the deployment composed
    document_name: str = ""
    document_title: str = ""
    formats: list[str] = field(default_factory=list)
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

    def ordered_results(self) -> list[tuple[str, CapabilityResult]]:
        """Results in PLAN order — the only deterministic order there is.

        `results` is filled by parallel waves, so its insertion order is
        arrival order (see the caveat in `core/state.py`). Anything a human
        reads — the report's annexes, the files a step produced — must be
        stable across two runs of the same plan, so it is ordered here once
        rather than in each consumer. A result with no plan step (a plan that
        never made it to the store) keeps its dict order, at the end.
        """
        order = [step["capability"] for step in (self.plan or {}).get("steps", [])] \
            if self.plan else []
        names = sorted(self.results, key=lambda n: order.index(n) if n in order else len(order))
        return [(name, self.results[name]) for name in names]

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
            "document_name": self.document_name,
            "document_title": self.document_title,
            "formats": self.formats,
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
