"""The run writes its own document: the last step of a DAG run that answered.

A run that reached `post_process` (it answered) or `unanswered` (it declared
it could not, #59) and whose request asked for a file ends here. What was
asked is `document_formats`: named by the caller, or read out of the sentence
by `document_intent` (#90); empty or absent is no file (#96). The Reporters
render it; the step declares each file it wrote (`artifacts.store.declare`)
and returns what failed (`document_error`), neither of which the job engine
understands (core split: docs/design/core-v1.md).

A write that fails does not fail the run (#28): the answer is in the state,
only the file is missing, so the step returns the error instead of raising —
a raise would make an answered run FAILED, and resumable into the same write.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..artifacts.store import declare
from ..engine.usage import current_ledger
from .report import Reporter, ReportWriteError, compose_reporters
from .state import AgentState, CapabilityResult, Plan


@dataclass(frozen=True)
class RunView:
    """What a Reporter reads of a run, taken from the graph's own state.

    The run's facts only. When the job was created, its session and when each
    step finished belong to the job record, not to the run, and stay empty
    here: a Reporter shows them only under `with_provenance` (#85), which no
    deployment turns on for the file a run writes.
    """

    job_id: str
    query: str
    document_name: str = ""
    document_title: str = ""
    final_answer: str | None = None
    terminal_kind: str | None = None
    plan: Plan | None = None
    results: dict[str, CapabilityResult] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    created_at: str = ""
    step_finished_at: dict[str, str] = field(default_factory=dict)

    def ordered_results(self) -> list[tuple[str, CapabilityResult]]:
        """Results in plan order — see `Job.ordered_results` for why."""
        order = [step["capability"] for step in (self.plan or {}).get("steps", [])]
        names = sorted(self.results,
                       key=lambda n: order.index(n) if n in order else len(order))
        return [(name, self.results[name]) for name in names]

    def step_usage(self, capability: str) -> dict[str, Any]:
        return ((self.results.get(capability) or {}).get("meta") or {}).get("usage") or {}


class DocumentWriter:
    """Graph node: render the run's document in the formats its request asked for.

    `reporter_for` composes a Reporter for a job's formats (the composition
    root passes one that knows its registry); `reporter`, when set, writes
    every document whatever the formats — the seam tests and single-format
    embedders use. `reports_dir` is where deliverables land.
    """

    def __init__(
        self,
        reporter_for: Callable[[Sequence[str]], Reporter] | None = None,
        reports_dir: str | Path = "artifacts",
        *,
        reporter: Reporter | None = None,
    ):
        self.reporter_for = reporter_for or (lambda formats: compose_reporters(formats))
        self.reports_dir = Path(reports_dir)
        self.reporter = reporter

    async def run(self, state: AgentState) -> dict[str, Any]:
        # Seeded by the caller, or chosen by `document_intent`; None and [] are no file.
        formats = state.get("document_formats") or []
        if not formats:
            return {}
        ledger = current_ledger()
        view = RunView(
            job_id=state.get("job_id", ""),            # seeded at entry by the job
            query=state["query"],
            document_name=state.get("document_name", ""),
            document_title=state.get("document_title", ""),
            final_answer=state.get("final_answer"),    # post_process / unanswered
            terminal_kind=state.get("terminal_kind"),
            plan=state.get("plan"),
            results=state.get("results", {}),
            usage=ledger.total().to_dict() if ledger is not None else {},
        )
        reporter: Any = None
        try:
            reporter = self.reporter or self.reporter_for(formats)
            outputs, error = list(reporter.write(view, self.reports_dir)), None
        except Exception as e:
            failure = e if isinstance(e, ReportWriteError) else ReportWriteError(
                getattr(reporter, "format", None) or ",".join(formats), e)
            outputs, error = failure.outputs, str(failure)   # keep what made it to disk
        for output in outputs:
            declare(output)
        return {"document_error": error}
