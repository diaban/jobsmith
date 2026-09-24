# 0108 — The PDF engine is probed on the first request that needs it, not to compose the app

- **Issue:** #108 · **PR:** #111
- **Status:** accepted · partially supersedes [0034](0034-pdf-deliverable.md)
- **Rule in `CLAUDE.md`:** "**The PDF is that page printed**; offered when installed (`find_spec`), engine probed on first need …" (Jobs layer)
- **See also:** [0055](0055-document-name-title-format.md), [0090](0090-document-intent-node.md), [0009](0009-reporters-and-html.md)

## Context

`build_app(db="memory", llm="fake")` took **3.8–3.9 s** and left `weasyprint`
in `sys.modules`; `python -X importtime -c "import weasyprint"` alone is
3.88 s. Wherever `.[pdf]` is installed, every command that composes the app
(`jobsmith jobs --local` took **4.2 s**), every daemon start and every test
process paid it, whether or not anyone ever asked for a PDF.

Two causes, one on top of the other:

- 0034 probes the engine when a `PdfReport` is **constructed**, so a format
  nothing can render fails before a job is run;
- since #90 the composition root asks `available_formats(registry)` for the
  formats `document_intent` may choose from, and that function answered by
  **constructing every Reporter** — i.e. importing the engine.

Only the second was ever needed at startup, and it does not need the engine.

## Decision

Two questions are kept apart, answered at two different moments.

- **Is PDF offered?** — `FileReporter.installed()`, a class method;
  `PdfReport.installed()` is `report_pdf.engine_installed()`, a
  `importlib.util.find_spec("weasyprint")` that loads nothing.
  `available_formats` asks the classes, never an instance. This is what the
  document step, the chat's refusal text and anything else that lists
  formats reads.
- **Can PDF render?** — `report_pdf.engine()`, the full import with 0034's
  two distinct messages (missing distribution → `pip install 'jobsmith[pdf]'`;
  distribution present but pango/cairo missing → the system-libraries
  message). Still called from `PdfReport.__init__`, so *composing is still
  the check*; what moved is when anything composes one:
  - **`create_job`**, via `ensure_formats_available`, for a request that
    names PDF — before any work, where #55 put both document refusals. Run
    in a thread (`asyncio.to_thread`, also in `chat/tools.py`), because the
    first probe is seconds of import that must not stall every other session
    on the daemon's loop.
  - **the document step**, when the model chose a format mid-run
    (`DocumentIntent.confirm` = `renderable_formats`, handed down by
    `build_app` through `AgentBuilder(confirm_document_formats=...)`): the
    choice is proved before it is written, and a format that cannot render is
    dropped like any other name outside the list — the node still cannot
    refuse. `core/` still never learns what a Reporter is: it gets a callable.
  - **startup**, only when the deployment's default is PDF (see below).
- **Cached once probed**, both ways (`report_pdf._probed`). A success because
  the import is the whole cost. A failure because neither cause is fixed
  without a restart, and because it lets the offer become truthful: once a
  probe has failed, `engine_installed()` answers False and PDF stops being
  offered — to the document step's prompt on the next composition, and in
  the chat's "formats available here".
- **An engine that cannot load is refused as `ValueError`.**
  `ensure_formats_available` translates the `RuntimeError` from composition.
  Every door says a refusal as `ValueError` — the chat tool's "NOT launched"
  (it catches `ValueError` only), the API's 400, `DaemonClient`'s mapping back
  — so as `RuntimeError` it was a crashed tool call and a 500. It went unseen
  because the probe used to run at startup; it is the main path now.

### Pango missing, weasyprint installed: offered, then proved

On such a machine `find_spec` says installed, so PDF **is offered** until
the first request that wants one. The alternative was to probe to decide the
offer — which is exactly the 4 s this record removes, paid by every machine
that *can* render so that the rare one that cannot gets a shorter prompt.
Offering it costs nothing that 0034 protects:

- a caller that names PDF (chat, `POST /jobs`, `jobsmith run` with formats)
  is refused in `create_job` with the libraries message, before any work;
- a request that asks in words is read by the document step, which proves the
  choice before writing it and drops it — the run carries on with no PDF,
  said on the record like every other fail-open of that node (0090), and it
  finds out at its start, not its end;
- after that first probe the offer is withdrawn for the process's lifetime.

### `$JOBSMITH_REPORT_FORMAT=pdf` keeps the eager probe

The deployment default is a startup setting: an operator who made PDF *the*
document format asked for PDF before any request did, so startup is the
first real need, and failing there is still right — the daemon says it at
launch rather than on the first "write me a report". `build_app` keeps
`ensure_formats_available(default_formats)`; it loads the engine only when
that list contains `pdf` (the default, markdown, composes nothing heavy).
Measured: `jobs --local` with `JOBSMITH_REPORT_FORMAT=pdf` is still 4.3 s.

## Alternatives and why not

- **Probe in a background thread at startup** — the cost moves off the
  critical path of a daemon but not of a one-shot command, which exits
  before or while the import runs; and every test process would still pay it.
- **Probe at write time only** — the defect 0034 was written against: a job
  that discovers at its end, three minutes and a ledger later, that its format
  cannot be rendered.
- **Offer PDF only after a successful probe** — truthful offer, but the
  document step could then never choose PDF in a fresh process unless
  something else had probed first: the first "give me that as a PDF" of every
  process would get no file on a machine that renders perfectly.
- **Make the offer list lazy in `core/`** (a callable evaluated per run) —
  would withdraw the offer within the same composition after a failed probe;
  one more concept in `core/` for the one machine that is misconfigured, whose
  requests are already dropped correctly by `confirm`. Not taken.

## Measured

WSL2, Python 3.12, every extra installed, pango present; `XDG_DATA_HOME` a
temp dir, provider keys blank, `db="memory"`. `build_app` timed around
`await build_app(db="memory", llm="fake")` in a fresh interpreter, reading
`"weasyprint" in sys.modules` afterwards; the CLI timed with `/usr/bin/time`
from a neutral cwd, "before" being `git archive b099393` on `PYTHONPATH`;
3–5 runs each, ranges shown.

| | before (b099393) | after |
|---|---|---|
| `build_app(db="memory", llm="fake")` | 3.38–3.86 s, `weasyprint` loaded | **0.10 s**, not loaded |
| `jobsmith --llm=fake --db=memory --local jobs` | 4.21–4.26 s | **0.86–1.04 s** |
| same, `JOBSMITH_REPORT_FORMAT=pdf` | — | 4.34–4.39 s (eager, by decision) |

Falsified against `tests/test_pdf_lazy_probe.py` alone, each mutation restored
before the next:

| mutation | failed |
|---|---|
| `available_formats` constructs each Reporter again (the pre-#108 body) | `test_composing_the_app_does_not_load_the_pdf_engine`, `test_the_offer_is_answered_without_the_engine`, `test_the_probe_runs_once_and_a_failure_stops_the_offer` |
| engine failure escapes `ensure_formats_available` as `RuntimeError` | both `test_a_pdf_request_is_refused_in_create_job_with_the_right_message` cases, `test_the_probe_runs_once…`, `test_the_chat_says_not_launched…` |
| missing libraries reported with the missing-distribution message | `…refused_in_create_job…[no_libraries-pango]`, `test_the_chat_says_not_launched…` |
| no cache (`engine()` re-imports every time) | `test_the_probe_runs_once_and_a_failure_stops_the_offer` |
| the document step trusts the offer (`confirm` skipped) | `test_the_document_step_drops_a_pdf_it_cannot_render` |
| `build_app` no longer probes a PDF default | `test_a_deployment_whose_default_is_pdf_still_probes_at_startup` |
| `PdfReport.serialize` writes the HTML text | `test_a_pdf_request_where_the_engine_runs_still_renders` |

The startup tests run `build_app` in a subprocess: this process's
`sys.modules` holds whatever earlier tests imported.

## Consequences

- The first PDF request of a process pays the ~4 s import inside
  `create_job` (in a thread). The job has not started, so nobody's run waits
  on it but that caller's.
- `renderable_formats(names)` is the proving twin of `available_formats`,
  for callers that must not promise: the document step and the evals harness
  (`Observation.formats_available`), which now proves what it offers — so a
  structural eval run still loads the engine once, as it did before.
- A test that simulates a broken engine must reset `report_pdf._probed`
  (both PDF test files do it in an autouse fixture).
- `slide_deck`'s `python-pptx` probe at startup is untouched: it is a
  capability's registration (0035), and it is not where the 4 s was.

## Supersedes / superseded by

Partially supersedes [0034](0034-pdf-deliverable.md): "the engine is probed
when the Reporter is constructed … a format nothing can render fails at
startup" — the Reporter still probes when constructed, but composing the app
no longer constructs one unless the deployment's default is PDF; the refusal
moves to `create_job` and the document step, both before any work.
