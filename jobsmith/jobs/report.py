"""Job outputs: what a finished job hands back to the human who asked.

Two layers, so a new format never re-implements the layout:

    JobDocument   plain data — the deliverable's structure (answer,
                  provenance, optional annexes), built once from a Job
    Reporter      serializes that document to file(s) and returns the
                  `JobOutput`s describing them

`MarkdownReport`, `HtmlReport` (report_html.py) and `PdfReport`
(report_pdf.py) are the built-in ones: one document, one `build_document`,
three serializers. PPTX would be another — that is the whole point of the
split. `PdfReport` is the case that proves it: it renders the *same page*
`HtmlReport` does and prints it, sharing the layout as a pure function of the
document rather than as a file another Reporter must have written first.

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

from ..core.state import TERMINAL_UNANSWERED
from ..core.usage import Usage
from .models import Job, JobOutput

# What a deliverable opens with when the run declared it could not answer
# (#59). One sentence, above the text, in every format: a file that reads like
# a report is exactly how a refusal went unnoticed until someone had read it
# to the end. It is framework wording rather than a profile message on purpose
# — it describes the run, not the domain, like the headings around it.
UNANSWERED_NOTICE = (
    "This run could not answer the request. What follows says what was "
    "missing, not what was asked for."
)


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
    # Did the run answer, or declare that it could not? A Reporter renders
    # `UNANSWERED_NOTICE` when it did not. Carried as a fact about the run
    # rather than as rendered text, so each format words it in its own markup.
    answered: bool = True

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
        answered=job.terminal_kind != TERMINAL_UNANSWERED,
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
    sections: list[tuple[str, str]] = []
    for name, result in job.ordered_results():
        try:
            body = registry.get(name).render_report(result)
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

    `format` and `extension` are declared read-only because reading is all
    this protocol ever does with them. Declared as plain variables they
    would be *mutable* members (PEP 544), which a computed one cannot
    satisfy — `MultiReporter` derives both from the reporter it puts first,
    so it would not conform, and `compose_reporters` could not honestly
    promise a `Reporter`. A class attribute still satisfies a read-only
    member, which is why `FileReporter` needs no change; the converse is
    what does not hold.
    """

    @property
    def format(self) -> str: ...

    @property
    def extension(self) -> str: ...

    def write(self, job: Job, directory: Path) -> list[JobOutput]: ...


class FileReporter:
    """Everything the built-in Reporters share: build the document, write one
    file for it, describe that file as a `JobOutput`.

    A subclass supplies `format`, `extension` and `render(document)` — which
    is genuinely all that differs between two *text* formats of the same
    deliverable. A format whose file is bytes overrides `serialize` instead:
    that is the one step `write` delegates, so naming the file, building the
    document and the `role="main"` decision are never copied along with it.

    `binary` says which of the two a format is, for callers that must hand
    the file back rather than write it (`is_binary_format`).

    `with_annexes` is a policy, not a structure: per-step material lives in
    the store and is served by the API/CLI, so the document stays a
    deliverable by default. Turn it on for a self-contained archive.
    """

    format = "text"
    extension = "txt"
    title = "Job report"
    binary = False          # is the file bytes rather than text?

    def __init__(self, registry: Any = None, *, with_annexes: bool = False):
        self.registry = registry
        self.with_annexes = with_annexes

    def write(self, job: Job, directory: Path) -> list[JobOutput]:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{job.job_id}.{self.extension}"
        document = build_document(job, self.registry, with_annexes=self.with_annexes)
        self.serialize(document, path)
        # Always "main": a lone Reporter IS the deliverable. Deciding which
        # one wins when several are asked for belongs to whoever composed
        # them, not to a format that cannot see its siblings.
        return [JobOutput(
            path=str(path), format=self.format, title=self.title, role="main"
        )]

    def serialize(self, document: JobDocument, path: Path) -> None:
        """Put the document on disk, at the path `write` decided.

        The default is a text file holding `render(document)`, which is what
        every text format wants. A binary format overrides this one method —
        it is the only step of `write` that cares whether a file is text.
        """
        path.write_text(self.render(document), encoding="utf-8")

    def render(self, doc: JobDocument) -> str:
        raise NotImplementedError


class MarkdownReport(FileReporter):
    """The default deliverable: one markdown file per job."""

    format = "markdown"
    extension = "md"

    def render(self, doc: JobDocument) -> str:
        lines = [f"# {doc.title}", ""]
        if not doc.answered:
            lines += [f"> **{UNANSWERED_NOTICE}**", ""]
        lines += [doc.answer, ""]

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


def _reporter_classes() -> dict[str, type[FileReporter]]:
    """Every format name this build knows, and the class that serves it.

    A local import, not a module constant: the other Reporters build on this
    module, so importing them at the top would be a cycle. Neither of them
    imports its engine at module scope, so listing PDF here costs nothing to
    someone who never asked for one — `.[pdf]` is probed when a `PdfReport`
    is actually constructed.
    """
    from .report_html import HtmlReport
    from .report_pdf import PdfReport

    return {
        "markdown": MarkdownReport, "md": MarkdownReport,
        "html": HtmlReport, "htm": HtmlReport,
        "pdf": PdfReport,
    }


def is_binary_format(report_format: str | None) -> bool:
    """Is a deliverable of this format bytes rather than text?

    Asked of a *finished job's* declared format — long after the Reporter
    that wrote it is gone, possibly on a machine where that Reporter's extra
    is not installed — which is why it is a lookup by name and not a question
    put to a Reporter instance. The classes stay the single source of truth;
    only the name travels.

    An unknown format is read as text, the same benefit of the doubt
    `REPORT_MEDIA_TYPES` and the evals' extractor already give it.
    """
    cls = _reporter_classes().get((report_format or "").strip().lower())
    return bool(cls and cls.binary)


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
    known = _reporter_classes()
    cls = known.get((report_format or "").strip().lower())
    if cls is None:
        raise ValueError(
            f"unknown report format {report_format!r} "
            f"(known: {', '.join(sorted(known))})"
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
    `MultiReporter` whose first name is the main deliverable. An unknown name
    anywhere in the list still raises — `make_reporter` is the per-format
    factory and stays the one that decides.

    A Reporter writes `{job_id}.{extension}`, so the extension is what makes
    two of them collide, and the two ways to name one twice are not the same
    mistake:

    - the SAME Reporter twice (an alias: `"markdown,md"`) is a harmless way
      of saying one format — collapsed silently, one file;
    - two DIFFERENT Reporters claiming one extension would write the same
      path, the second overwriting the first, and `Job.outputs` would then
      list two entries for one file. That is a composition error and raises
      here, like an unknown name — a trap laid for whoever adds a second
      markdown flavour, sprung at composition rather than in a job. (PDF,
      which landed since, writes `.pdf` and collides with nobody — it is a
      peer of the HTML Reporter, not a second rendering of it.)
    """
    names = parse_report_formats(formats) if isinstance(formats, str) else list(formats)
    reporters: list[Reporter] = []
    seen: dict[str, tuple[type, str]] = {}     # extension -> (class, format name)
    for name in names or ["markdown"]:
        reporter = make_reporter(name, registry, with_annexes=with_annexes)
        known = seen.get(reporter.extension)
        if known is not None:
            if known[0] is type(reporter):
                continue                        # an alias of a format already asked for
            raise ValueError(
                f"report formats {known[1]!r} and {reporter.format!r} both write "
                f".{reporter.extension} files — one would overwrite the other"
            )
        seen[reporter.extension] = (type(reporter), reporter.format)
        reporters.append(reporter)
    return reporters[0] if len(reporters) == 1 else MultiReporter(reporters)
