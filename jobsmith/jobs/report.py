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

**A deliverable is an answer, not a record of a run** (#85). It used to carry,
under the answer, the request quoted in full, the job id, two timestamps, the
session id, the token bill, a plan table with a per-step cost column and a
drawing of the DAG — the product talking about itself inside the thing the
reader opens, and the part nobody asked for. All of it is still *recorded*,
and `GET /jobs/{id}` / `jobsmith job <id>` serve it in full; what the document
keeps is `job_reference`, one line naming the job, which is all it takes to
get from the file back to the record. `with_provenance` puts the section back
for whoever wants a self-contained archive — the same switch `with_annexes`
already is, and for the same reason: a policy, not a structure.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..core.paths import PathRefused, safe_name
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
    # Does this document *show* the record of the run it came from — the
    # request, the timings, the session, the bill, the plan (#85)? Off by
    # default: a reader opened a deliverable, not a trace. The data above is
    # populated either way, because it is what the job *is*; this says
    # whether a Reporter renders it, which is a policy and belongs with
    # `with_annexes`. `job_reference` is what every document carries instead.
    provenance: bool = False

    @property
    def dag_edges(self) -> list[tuple[str, str]]:
        return [(dep, row.capability) for row in self.plan for dep in row.depends_on]


def job_reference(job_id: str) -> str:
    """The one line every deliverable carries about the run behind it (#85).

    The provenance section it replaced answered a question nobody had asked
    inside the document: a reader who wants the plan, the timings or the bill
    wants the *record*, and the record is already served whole by
    `GET /jobs/{id}` and `jobsmith job <id>`. What the file cannot get from
    somewhere else is the way back — which run wrote it — so that is what
    stays, and it stays on every document rather than behind a flag, because
    a file nobody can trace is the same class of defect as a file nobody can
    find (#28).

    Plain text and no markup at all, like `UNANSWERED_NOTICE`: it describes
    the run rather than the domain, each format wraps it in its own markup,
    and a backtick meant for markdown is a literal backtick on an HTML page.
    """
    return (f"Produced by job {job_id} — jobsmith job {job_id[:8]} shows the "
            f"request, the plan, the steps and what it cost.")


#: What a deliverable is called when its request says nothing usable. Shared
#: with `FileReporter.title`, so the file and the document agree.
DEFAULT_TITLE = "Job report"

#: How long a derived title may be. The request is the only source there is
#: today; the day a job carries a title of its own (#55), this stays as the
#: floor under it.
TITLE_MAX = 80


def document_title(request: str, *, limit: int = TITLE_MAX) -> str:
    """The deliverable's title, derived from the request without cutting a word.

    It used to be `job.query[:80]` — a raw slice, landing wherever the count
    landed, with nothing to say it was cut (#54):

        # Réaliser un comparatif détaillé des chaises ergonomiques adaptées à un utilisate

    The word is *utilisateur*. This is the first line of the deliverable, and
    the one part a reader sees before deciding whether to read the rest.

    So: whitespace collapses (a request spanning lines would otherwise break
    the markdown heading in two), a request that fits is left exactly as it
    is, and a longer one is cut at the last word boundary that fits and marked
    with an ellipsis. A single word longer than the limit has no boundary to
    cut on and is cut anyway — a title too long to be one is the worse defect,
    and the ellipsis still says it was cut.
    """
    text = " ".join((request or "").split())
    if not text:
        return DEFAULT_TITLE
    if len(text) <= limit:
        return text
    head = text[:limit - 1]                     # leave room for the ellipsis
    cut = head.rsplit(" ", 1)[0] if " " in head else head
    return cut.rstrip(" ,;:.-—–") + "…"


#: How long a requested filename may be. Nothing here is a filesystem limit
#: (255 bytes is), it is what a filename stops being at: past this, a name is
#: a sentence and the person who wrote it meant the title.
NAME_MAX = 100


def known_extensions() -> dict[str, str]:
    """`{format name: extension}` for every Reporter that ships — one lookup.

    Derived from the same registry `make_reporter` reads, so a Reporter added
    later is known here the day it is added, with nothing to keep in step.
    """
    return {name: cls.extension for name, cls in _reporter_classes().items()}


def document_stem(name: str) -> str:
    """A requested filename, as the one path component a Reporter may use.

    The caller says what the file is *called*; where it goes is this code's
    decision, so `safe_name` refuses anything carrying a separator instead of
    flattening it (`core/paths.py` answers that once, for annexes too).

    The extension is dropped when it names a format we know: a user asking for
    `rapport.md` and a PDF of the same thing is asking for one document under
    two extensions, and the formats — not the name — decide those. An unknown
    suffix is left alone: `v1.2` is a name, not a format.

    Too long is refused rather than cut. A name the user chose, silently
    shortened, is the defect #54 was about in the title; a refusal is seen by
    whoever wrote the name while they can still write another.
    """
    stem = safe_name(name, "document name")
    head, dot, suffix = stem.rpartition(".")
    if dot and suffix.lower() in set(known_extensions().values()) and head:
        stem = head
    if len(stem) > NAME_MAX:
        raise PathRefused(
            f"document name is {len(stem)} characters, at most {NAME_MAX} "
            f"(a filename, not a title): {name!r}")
    return stem


def deliverable_filenames(
    document_name: str, formats: Iterable[str] | None
) -> list[str]:
    """What the files will be called, for a front-end showing what is approved.

    A fact, not a sentence: the format→extension mapping is this module's, and
    two front-ends deriving it themselves would be two mappings. Empty when
    the job has no name of its own — the file is then `{job_id}.{extension}`
    and there is no job id yet at the moment this is shown, so a caller says
    what it can honestly say: the formats.

    `formats` carries the three states of `Job.formats`, and **only a list of
    names promises a file**. `[]` is a request for no document at all (#84);
    `None` is a request that said nothing, which since #96 means no document
    either — unless the engine's own document step reads one out of the
    sentence, which has not happened yet at the moment this is shown. This
    used to guess markdown for `None`, which was the deployment's default
    back when silence meant "a file in the default format"; it would now print
    a filename for a file nobody will write — a promise in exactly the place
    #55 built to stop making them.
    """
    wanted = [f.strip().lower() for f in formats or [] if f and f.strip()]
    if not document_name or not wanted:
        return []
    extensions = known_extensions()
    return [f"{document_name}.{extensions.get(f, f)}" for f in wanted]


def available_formats(registry: Any = None) -> list[str]:
    """The format names this deployment OFFERS, canonical, sorted — loading nothing.

    Not the same list as `known_extensions()`: a Reporter that ships can still
    be unavailable here — `.[pdf]` is an optional extra. Answered from each
    class's `installed()`, never by constructing it (#108): constructing the
    PDF Reporter imports its engine, ~4 s, and this is asked at every startup
    (the formats `document_intent` may choose from) and on every refusal.

    Offered is not proved: with the distribution present and pango missing,
    PDF is offered until the first request that wants one probes the engine
    — `create_job` refuses that request (`ensure_formats_available`), and the
    document step drops the format (`renderable_formats`) — after which it is
    no longer offered. `registry` is accepted for symmetry with the composers.
    """
    del registry            # what a format needs to be offered is not in it
    names: list[str] = []
    for cls in _reporter_classes().values():
        if cls.format not in names and cls.installed():
            names.append(cls.format)
    return sorted(names)


def renderable_formats(formats: Iterable[str], registry: Any = None) -> list[str]:
    """Those of `formats` that can really be rendered here, in the order given.

    The proving twin of `available_formats`: each is composed, which probes
    an engine (once per process), and dropped if that fails or the name is
    unknown. For a caller that must not refuse and must not promise — the
    document step, which chooses a format mid-run where nobody is listening
    (#90), and the evals, which skip a case this machine cannot render.
    """
    chosen: list[str] = []
    for name in formats:
        try:
            reporter = make_reporter(name, registry)
        except Exception:
            continue
        if reporter.format not in chosen:
            chosen.append(reporter.format)
    return chosen


#: The name a caller uses for "a document, in whatever format this deployment
#: writes one" (#96). A request can want a file without naming its format —
#: "write me a report" — and the answer to *which* format is then the
#: deployment's (`$JOBSMITH_REPORT_FORMAT`, `app/agent.py::pick_report_formats`).
#: It is resolved into those names before a job is recorded, so the record
#: always says what will actually be written; a job never carries the alias.
DEFAULT_FORMATS_ALIAS = "default"


def ensure_formats_available(
    formats: Iterable[str] | str | None,
    *,
    registry: Any = None,
    default: Sequence[str] | None = None,
) -> list[str] | None:
    """The requested formats, or a `ValueError` saying which cannot be had.

    Composing IS the check — `make_reporter` refuses an unknown name and
    `PdfReport` probes its engine in `__init__` — so this asks the question by
    building the answer and throwing it away. It is therefore where the PDF
    engine is first loaded (#108): only for a request that names PDF (or
    whose deployment default is PDF), never to compose the app. Blocking on
    that first probe — seconds — so async callers run it in a thread. That matters for `.[pdf]`, this
    project's one deployment constraint: a format nothing can render here must
    be refused where the person who asked can still see it (the notice
    `chat/tools.py` writes, `create_job`), never at the end of a run that
    spent three minutes first.

    It answers in the same three states it is asked in (#84), and passes them
    through unchanged: `None` for a request that said nothing, `[]` for one
    that asked for **no document**, the names otherwise. `[]` composes
    nothing on purpose — there is no Reporter to check, which is the whole
    content of the answer. **`None` is never turned into a format here**
    (#96): a request that said nothing gets no document, so there is no
    default to reach for on its behalf.

    `DEFAULT_FORMATS_ALIAS` is the one name that is not a format: it says
    "a document, in this deployment's format" and is replaced by `default`
    (the deployment's list, which only the caller knows — a module-level
    fallback here would be the silent markdown this function stopped being).
    Asking for it where no default was given is refused like an unknown name.
    """
    if formats is None:
        return None
    names = parse_report_formats(formats) if isinstance(formats, str) else list(formats)
    if not names:
        return []
    resolved: list[str] = []
    for name in names:
        if str(name).strip().lower() == DEFAULT_FORMATS_ALIAS:
            if not default:
                raise ValueError(
                    f"{DEFAULT_FORMATS_ALIAS!r} names this deployment's document "
                    "format, and none was given here")
            expansion = list(default)
        else:
            expansion = [name]
        resolved.extend(n for n in expansion if n not in resolved)
    try:
        compose_reporters(resolved, registry)
    except RuntimeError as unavailable:
        # An engine that cannot load (`report_pdf.engine`) is a refusal like
        # an unknown name, and every door says refusals as `ValueError` — the
        # chat tool's "NOT launched", the API's 400, `DaemonClient`'s mapping
        # back. Raised as `RuntimeError` it was a crashed tool and a 500; it
        # went unseen while the probe ran at startup, and is the path now
        # that the first PDF request is where it runs (#108).
        raise ValueError(str(unavailable)) from unavailable
    return resolved


def build_document(
    job: Job,
    registry: Any = None,
    *,
    with_annexes: bool = False,
    with_provenance: bool = False,
) -> JobDocument:
    """Turn a finished Job into the document a Reporter serializes.

    The job's facts are read whatever the flags say — they are what the job
    *is*, and a document that holds them can be rendered either way.
    `with_provenance` decides whether the reader is *shown* them (#85);
    `with_annexes` is the older half of the same policy, and it gates the
    annexes rather than a flag because building them costs a call into every
    capability that ran.
    """
    doc = JobDocument(
        # The name is not the title (#55): a job that asked for neither
        # derives both from the same request, and asking for one never
        # silently decides the other.
        title=job.document_title.strip() or document_title(job.query),
        request=job.query,
        job_id=job.job_id,
        session_id=job.session_id,
        created_at=job.created_at,
        finished_at=datetime.now(UTC).isoformat(),
        answer=job.final_answer or "_(no answer)_",
        answered=job.terminal_kind != TERMINAL_UNANSWERED,
        provenance=with_provenance,
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

    `with_annexes` and `with_provenance` are policies, not structures, and
    they are the same policy twice: per-step material and the record of the
    run both live in the store and are served by the API/CLI, so the document
    stays a *deliverable* by default (#85). Turn either on for a
    self-contained archive; they are independent because one is what the
    steps produced and the other is what the run was.
    """

    format = "text"
    extension = "txt"
    title = DEFAULT_TITLE
    binary = False          # is the file bytes rather than text?

    def __init__(
        self,
        registry: Any = None,
        *,
        with_annexes: bool = False,
        with_provenance: bool = False,
    ):
        self.registry = registry
        self.with_annexes = with_annexes
        self.with_provenance = with_provenance

    @classmethod
    def installed(cls) -> bool:
        """May this format be offered here — answered WITHOUT constructing it.

        Constructing is the real check (a Reporter with an engine probes it
        in `__init__`), and that can cost seconds; offering a format is asked
        at every startup (#108). A pure-Python Reporter is always installed.
        """
        return True

    def write(self, job: Job, directory: Path) -> list[JobOutput]:
        path = self.path_for(job, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        document = build_document(job, self.registry, with_annexes=self.with_annexes,
                                  with_provenance=self.with_provenance)
        self.serialize(document, path)
        # Always "main": a lone Reporter IS the deliverable. Deciding which
        # one wins when several are asked for belongs to whoever composed
        # them, not to a format that cannot see its siblings.
        return [JobOutput(
            path=str(path), format=self.format, title=self.title, role="main"
        )]

    def path_for(self, job: Job, directory: Path) -> Path:
        """Where this job's deliverable goes, and what it is called (#55).

        A job id is unique and says nothing; a requested name says everything
        and is unique to nobody. So a named deliverable lands in the job's own
        directory — where its annexes already are (`LocalArtifactStore`) — and
        two jobs called `rapport` keep two files instead of the second silently
        overwriting the first, which is the same class of defect as a
        deliverable nobody can find (#28).

        An unnamed job keeps `{job_id}.{extension}` exactly where it always
        was: nothing about it needed a directory of its own.

        The name reaching here is already one legal path component —
        `create_job` refuses anything else, in front of whoever asked.
        """
        if job.document_name:
            return directory / job.job_id / f"{job.document_name}.{self.extension}"
        return directory / f"{job.job_id}.{self.extension}"

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

        lines += ["---", ""]
        if not doc.provenance:
            # The answer, then one line saying where it came from (#85).
            lines += [f"_{job_reference(doc.job_id)}_", ""]
            return "\n".join(lines + self._annexes(doc)) + "\n"

        lines += ["## About this job", "",
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
        return "\n".join(lines + self._annexes(doc)) + "\n"

    @staticmethod
    def _annexes(doc: JobDocument) -> list[str]:
        lines: list[str] = []
        for heading, body in doc.annexes:
            lines += ["", "<details>", f"<summary>Step output — {heading}</summary>", "",
                      body, "", "</details>"]
        return lines


def _reporter_classes() -> dict[str, type[FileReporter]]:
    """Every format name this build knows, and the class that serves it.

    A local import, not a module constant: the other Reporters build on this
    module, so importing them at the top would be a cycle. Neither of them
    imports its engine at module scope, so listing PDF here costs nothing to
    someone who never asked for one — `.[pdf]` is probed when a `PdfReport`
    is actually constructed, and offering it (`available_formats`) asks the
    class, not an instance (#108).
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
    with_provenance: bool = False,
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
    return cls(registry, with_annexes=with_annexes, with_provenance=with_provenance)


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
    with_provenance: bool = False,
) -> Reporter:
    """One Reporter for one or more format names — the seam a composition
    root uses to say what a job hands back.

    A single name gives back that format's Reporter unchanged (a run then
    produces exactly the one file it always did); several give a
    `MultiReporter` whose first name is the main deliverable. An unknown name
    anywhere in the list still raises — `make_reporter` is the per-format
    factory and stays the one that decides.

    **An empty list raises too** (#84). It used to mean markdown, because
    empty meant "the caller said nothing"; it now means "no document", and a
    Reporter is the last place that decision can be honoured — this function
    exists to answer *which* file, and a caller that wants none must not ask.
    Silently writing markdown for a job that asked for no file is the same
    mistake as writing it for one that asked for HTML.

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
    if not names:
        raise ValueError(
            "no report format to compose — an empty list means no document at "
            "all (#84), which is decided before a Reporter is asked for one")
    reporters: list[Reporter] = []
    seen: dict[str, tuple[type, str]] = {}     # extension -> (class, format name)
    for name in names:
        reporter = make_reporter(name, registry, with_annexes=with_annexes,
                                 with_provenance=with_provenance)
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
