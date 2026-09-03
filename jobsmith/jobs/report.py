"""Job outputs: what a finished job hands back to the human who asked.

Two layers, so a new format never re-implements the layout:

    JobDocument   plain data — the deliverable's structure (answer,
                  provenance, optional annexes), built once from a Job
    Reporter      serializes that document to a file and returns a JobOutput

`MarkdownReport` and `HtmlReport` (report_html.py) are the built-in ones:
same document, same `build_document`, two serializers. PDF/PPTX would be more
of them — that is the whole point of the split.

One job, one deliverable: `Reporter.write` returns a single `JobOutput`, so a
format is a *choice* (`make_reporter("html")`), not an addition. Handing back
markdown *and* HTML for the same run is a protocol change
(`write() -> list[JobOutput]`), not a new Reporter.

Per-step material is asked of the capabilities themselves
(`Capability.render_report`), never introspected here: the registry is
optional, and without it the annexes are simply left out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..core.usage import Usage
from .models import Job, JobOutput


@dataclass
class PlanRow:
    capability: str
    depends_on: list[str]
    status: str
    finished_at: str
    usage: Usage = field(default_factory=Usage)   # what this step spent


@dataclass
class JobDocument:
    """Format-independent shape of a job's deliverable."""

    title: str
    request: str
    job_id: str
    created_at: str
    finished_at: str
    answer: str
    session_id: str | None = None
    plan_rationale: str = ""
    plan: list[PlanRow] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)   # the whole run's spend
    annexes: list[tuple[str, str]] = field(default_factory=list)  # (heading, markdown)

    @property
    def dag_edges(self) -> list[tuple[str, str]]:
        return [(dep, row.capability) for row in self.plan for dep in row.depends_on]


def build_document(job: Job, registry: Any = None, *, with_annexes: bool = False) -> JobDocument:
    """Turn a finished Job into the document a Reporter serializes."""
    doc = JobDocument(
        title=job.query[:80],
        request=job.query,
        job_id=job.job_id,
        session_id=job.session_id,
        created_at=job.created_at,
        finished_at=datetime.now(UTC).isoformat(),
        answer=job.final_answer or "_(no answer)_",
        plan_rationale=(job.plan or {}).get("rationale", "") if job.plan else "",
        usage=Usage.from_dict(job.usage),
    )
    for step in (job.plan or {}).get("steps", []) if job.plan else []:
        name = step["capability"]
        result = job.results.get(name)
        doc.plan.append(PlanRow(
            capability=name,
            depends_on=list(step["depends_on"]),
            status="ok" if result and result.get("ok") else (
                f"failed ({result.get('error')})" if result else "not run"
            ),
            finished_at=job.step_finished_at.get(name, "—"),
            usage=Usage.from_dict(job.step_usage(name)),
        ))
    if with_annexes:
        doc.annexes = _annexes(job, registry)
    return doc


def _annexes(job: Job, registry: Any) -> list[tuple[str, str]]:
    """Ask each capability to present its own result (never guess here)."""
    if registry is None:
        return []
    order = [row["capability"] for row in (job.plan or {}).get("steps", [])] if job.plan else []
    names = sorted(job.results, key=lambda n: order.index(n) if n in order else 99)
    sections: list[tuple[str, str]] = []
    for name in names:
        try:
            body = registry.get(name).render_report(job.results[name])
        except KeyError:            # capability gone from the registry since the run
            body = None
        if body:
            sections.append((name, body))
    return sections


def format_cost(usage: Usage) -> str:
    """`~$0.4212`, or an empty string when no model in the tally has a price."""
    if usage.cost_usd is None:
        return ""
    return f"~${usage.cost_usd:.4f}" if usage.cost_usd < 1 else f"~${usage.cost_usd:.2f}"


def format_usage(usage: Usage) -> str:
    """One line a human can act on: how many calls, how many tokens, how much."""
    if not usage:
        return "not recorded"
    cached = f" (+{usage.cached_input_tokens:,} cached)" if usage.cached_input_tokens else ""
    parts = [
        f"{usage.calls} LLM call{'s' if usage.calls != 1 else ''}",
        f"{usage.input_tokens:,} in{cached} / {usage.output_tokens:,} out tokens",
    ]
    cost = format_cost(usage)
    if cost:
        parts.append(f"{cost} est.")
    if usage.models:
        parts.append(", ".join(usage.models))
    return " — ".join(parts)


def format_step_usage(usage: Usage) -> str:
    """Compact cell for the plan table — enough to spot the expensive step."""
    if not usage:
        return "—"
    tokens = usage.total_tokens
    size = f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)
    cost = format_cost(usage)
    return f"{size} tok · {cost}" if cost else f"{size} tok"


class Reporter(Protocol):
    """Produces one output file for a job."""

    format: str
    extension: str

    def write(self, job: Job, directory: Path) -> JobOutput: ...


class FileReporter:
    """Everything the built-in Reporters share: build the document, write one
    file for it, describe that file as a `JobOutput`.

    A subclass supplies `format`, `extension` and `render(document)` — which
    is genuinely all that differs between two formats of the same deliverable.

    `with_annexes` is a policy, not a structure: per-step material lives in
    the store and is served by the API/CLI, so the document stays a
    deliverable by default. Turn it on for a self-contained archive.
    """

    format = "text"
    extension = "txt"
    title = "Job report"

    def __init__(self, registry: Any = None, *, with_annexes: bool = False):
        self.registry = registry
        self.with_annexes = with_annexes

    def write(self, job: Job, directory: Path) -> JobOutput:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job.job_id}.{self.extension}"
        document = build_document(job, self.registry, with_annexes=self.with_annexes)
        path.write_text(self.render(document), encoding="utf-8")
        return JobOutput(
            path=str(path), format=self.format, title=self.title, role="main"
        )

    def render(self, doc: JobDocument) -> str:
        raise NotImplementedError


class MarkdownReport(FileReporter):
    """The default deliverable: one markdown file per job."""

    format = "markdown"
    extension = "md"

    def render(self, doc: JobDocument) -> str:
        lines = [f"# {doc.title}", "", doc.answer, ""]

        lines += ["---", "", "## About this job", "",
                  f"- **Request**: {doc.request}",
                  f"- **Job**: `{doc.job_id}`",
                  f"- **Started**: {doc.created_at}",
                  f"- **Finished**: {doc.finished_at}"]
        if doc.session_id:
            lines.append(f"- **Session**: `{doc.session_id}`")
        # Cost belongs with the provenance: whoever reads the report is the
        # one paying for it. Estimated from a price table, never a bill.
        lines.append(f"- **Usage**: {format_usage(doc.usage)}")

        if doc.plan:
            lines += ["", "### Steps", ""]
            if doc.plan_rationale:
                lines += [f"_{doc.plan_rationale}_", ""]
            lines += ["| step | depends on | status | usage | finished at |",
                      "|---|---|---|---|---|"]
            lines += [
                f"| {row.capability} | {', '.join(row.depends_on) or '—'} "
                f"| {row.status} | {format_step_usage(row.usage)} | {row.finished_at} |"
                for row in doc.plan
            ]
            lines += ["", "```mermaid", "flowchart LR"]
            edges = doc.dag_edges
            lines += [f"  {src} --> {dst}" for src, dst in edges]
            # isolated steps only: a root that already feeds someone is drawn by its edge
            connected = {n for edge in edges for n in edge}
            lines += [f"  {row.capability}" for row in doc.plan
                      if row.capability not in connected]
            lines += ["```"]

        for heading, body in doc.annexes:
            lines += ["", "<details>", f"<summary>Step output — {heading}</summary>", "",
                      body, "", "</details>"]
        return "\n".join(lines) + "\n"


def make_reporter(
    report_format: str = "markdown",
    registry: Any = None,
    *,
    with_annexes: bool = False,
) -> Reporter:
    """Pick a Reporter by format name — the seam a composition root uses to
    choose what a job hands back. Unknown names fail loudly: silently writing
    markdown for someone who asked for HTML is worse than a traceback.
    """
    # Local import: report_html builds on this module, so importing it at the
    # top would be a cycle. Selecting a format is not a hot path.
    from .report_html import HtmlReport

    reporters: dict[str, type[FileReporter]] = {
        "markdown": MarkdownReport, "md": MarkdownReport,
        "html": HtmlReport, "htm": HtmlReport,
    }
    cls = reporters.get((report_format or "").strip().lower())
    if cls is None:
        raise ValueError(
            f"unknown report format {report_format!r} (known: markdown, html)"
        )
    return cls(registry, with_annexes=with_annexes)
