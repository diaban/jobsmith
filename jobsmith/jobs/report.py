"""Job outputs: what a finished job hands back to the human who asked.

Two layers, so a new format never re-implements the layout:

    JobDocument   plain data — the deliverable's structure (answer,
                  provenance, optional annexes), built once from a Job
    Reporter      serializes that document to file(s) and returns the
                  `JobOutput`s describing them

`MarkdownReport` and `HtmlReport` (report_html.py) are the built-in ones:
same document, same `build_document`, two serializers. PDF/PPTX would be more
of them — that is the whole point of the split.

A job can hand back **several** deliverables: `Reporter.write` returns a
`list[JobOutput]`, and `MultiReporter` composes one Reporter per requested
format (`compose_reporters("markdown,html")`) so the manager still holds a
single reporter object. Exactly one of the outputs is `role="main"` — the
first format asked for, the one `Job.report_path` and `GET /jobs/{id}/report`
point at; the others are `role="alternate"`, the same document rendered
again. They are NOT annexes: an annex is per-step material a capability
produced, not a second copy of the report.

Per-step material is asked of the capabilities themselves
(`Capability.render_report`), never introspected here: the registry is
optional, and without it the annexes are simply left out.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
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


class ReportWriteError(RuntimeError):
    """A deliverable could not be written — carrying whatever WAS written.

    The run is not the file: by the time a Reporter runs, the job has an
    answer and it is persisted. So a failed write is not a failed job, and
    the manager needs two things from it — a message naming which format
    broke and why, and the outputs already on disk. Without the latter, a
    markdown file written before the HTML one raised would exist with no
    `JobOutput` describing it: a deliverable nobody can find, which is the
    whole reason `write` returns a list.
    """

    def __init__(
        self,
        report_format: str,
        cause: BaseException,
        outputs: Sequence[JobOutput] = (),
    ):
        self.report_format = report_format
        self.cause = cause
        self.outputs = list(outputs)
        super().__init__(
            f"the {report_format} deliverable could not be written: "
            f"{type(cause).__name__}: {cause}"
        )


class Reporter(Protocol):
    """Produces a job's deliverable file(s).

    `write` returns a list because one run may be asked for several formats
    (see `MultiReporter`); a Reporter that knows one format returns one
    element. The list is what lands in `Job.outputs`, so a file a Reporter
    writes without describing it here is a file nobody can find.
    """

    format: str
    extension: str

    def write(self, job: Job, directory: Path) -> list[JobOutput]: ...


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

    def write(self, job: Job, directory: Path) -> list[JobOutput]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job.job_id}.{self.extension}"
        document = build_document(job, self.registry, with_annexes=self.with_annexes)
        path.write_text(self.render(document), encoding="utf-8")
        # Always "main": a lone Reporter IS the deliverable. Deciding which
        # one wins when several are asked for belongs to whoever composed
        # them, not to a format that cannot see its siblings.
        return [JobOutput(
            path=str(path), format=self.format, title=self.title, role="main"
        )]

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


class MultiReporter:
    """Several formats of the same deliverable, behind one Reporter.

    The manager holds exactly one reporter and assigns whatever it returns
    to `Job.outputs`, so asking for markdown *and* HTML is a composition
    concern, not a manager change.

    The invariant it owns: **exactly one output is `main`** — the first one
    produced, i.e. the first format requested. `Job.report_path`, the CLI's
    `report` command and `GET /jobs/{id}/report` (whose content type follows
    that output's format) all read it, so a second `main` would make which
    file is *the* deliverable a matter of dict order. Later renderings are
    demoted to `alternate`; anything a Reporter already labelled otherwise
    (an annex) is left alone.
    """

    def __init__(self, reporters: Sequence[Reporter]):
        if not reporters:
            raise ValueError("MultiReporter needs at least one reporter")
        self.reporters = list(reporters)

    @property
    def format(self) -> str:
        """The main deliverable's format — what /report announces."""
        return self.reporters[0].format

    @property
    def extension(self) -> str:
        return self.reporters[0].extension

    def write(self, job: Job, directory: Path) -> list[JobOutput]:
        outputs: list[JobOutput] = []
        for reporter in self.reporters:
            try:
                written = reporter.write(job, directory)
            except ReportWriteError as failed:      # a composed composite
                raise ReportWriteError(
                    failed.report_format, failed.cause, outputs + failed.outputs
                ) from failed.cause
            except Exception as cause:
                # The files already written stay the job's deliverables:
                # they exist, and only a JobOutput makes them findable.
                raise ReportWriteError(reporter.format, cause, outputs) from cause
            for output in written:
                if outputs and output.role == "main":
                    output = replace(output, role="alternate")
                outputs.append(output)
        return outputs


def parse_report_formats(spec: str) -> list[str]:
    """`"markdown, html"` → `["markdown", "html"]`, order preserved.

    Order is meaning here: the first name is the main deliverable.
    """
    return [name.strip() for name in (spec or "").split(",") if name.strip()]


def compose_reporters(
    formats: str | Iterable[str] = "markdown",
    registry: Any = None,
    *,
    with_annexes: bool = False,
) -> Reporter:
    """One Reporter for one or more format names — the seam a composition
    root uses to say what a job hands back.

    A single name gives back that format's Reporter unchanged (a run then
    produces exactly the one file it always did); several give a
    `MultiReporter` whose first name is the main deliverable. Aliases of the
    same format are collapsed (`"markdown,md"` is one file, not the same file
    written twice), and an unknown name anywhere in the list still raises —
    `make_reporter` is the per-format factory and stays the one that decides.
    """
    names = parse_report_formats(formats) if isinstance(formats, str) else list(formats)
    reporters: list[Reporter] = []
    seen: set[type] = set()
    for name in names or ["markdown"]:
        reporter = make_reporter(name, registry, with_annexes=with_annexes)
        if type(reporter) in seen:
            continue
        seen.add(type(reporter))
        reporters.append(reporter)
    return reporters[0] if len(reporters) == 1 else MultiReporter(reporters)
