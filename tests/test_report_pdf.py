"""The PDF deliverable — and, first of all, whether the engine can run here.

`.[pdf]` is the only extra in this project with a **system**-level dependency:
WeasyPrint renders through pango/cairo, loaded with cffi at import time. So
the guard below is deliberately not `pytest.importorskip`: that would turn
"installed but the system libraries are missing" into a skip, and CI — which
installs every extra — would go green having tested nothing, which is the one
question this suite exists to answer. `find_spec` only *locates* the
distribution, so the import runs for real wherever the extra is installed and
a broken engine is a red suite, not a silent one.

Everything that is not the engine is tested without it: that a PDF is the
HTML page and not a third layout, that a binary Reporter composes like any
other, and that a job refuses to hand its bytes back as text. Those are
properties of the framework, and they hold on a machine that cannot render.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import FakeLLM
from support import (
    PDF_MARKS,
    StubPdf,
    done_job,
    make_document,
    make_manager,
    no_distribution,
    no_libraries,
    requires_pdf,
)

from jobsmith.core.deps import Deps
from jobsmith.core.document import DocumentIntent
from jobsmith.jobs import report_pdf
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.report import (
    MultiReporter,
    ReportWriteError,
    available_formats,
    compose_reporters,
    is_binary_format,
    make_reporter,
    renderable_formats,
)
from jobsmith.jobs.report_html import HtmlReport, dag_svg
from jobsmith.jobs.report_pdf import DAG_STYLE, PAGED_STYLE, PdfReport


@pytest.fixture(autouse=True)
def unprobed(monkeypatch):
    """Each test starts before the engine was probed: the outcome is cached
    for the process (#108), and a test that simulates a broken engine must
    neither see a real probe's success nor leave its failure behind."""
    monkeypatch.setattr(report_pdf, "_probed", None)


# ------------------------------------------------------------------ the engine


@requires_pdf
def test_weasyprint_renders_a_pdf():
    """Installing is not rendering: the wheel has no system library in it, and
    a machine without pango only finds out when bytes are asked for."""
    import weasyprint

    data = weasyprint.HTML(string="<h1>smoke</h1><p>a paragraph.</p>").write_pdf()
    assert data and data.startswith(b"%PDF-")


@requires_pdf
def test_write_produces_a_pdf_file_and_describes_it(tmp_path):
    (output,) = PdfReport().write(done_job("j20"), tmp_path)

    assert (output.format, output.role) == ("pdf", "main")
    assert output.path.endswith("j20.pdf")
    data = Path(output.path).read_bytes()
    assert data.startswith(b"%PDF-") and len(data) > 1000
    with pytest.raises(UnicodeDecodeError):     # why /report cannot serve it
        Path(output.path).read_text(encoding="utf-8")


@requires_pdf
def test_the_plan_is_drawn_in_colour_and_not_as_black_boxes(tmp_path):
    """The DAG is styled by the page's own sheet, which a print engine applies
    to HTML boxes only. Without the stylesheet carried inside the `<svg>` the
    nodes fall back to the SVG defaults — black fill, black text on it — so
    the fill operator below is the difference between a diagram and a row of
    blocks."""
    import weasyprint

    page = PdfReport().render(make_document())
    pdf = weasyprint.HTML(string=page).write_pdf(uncompressed_pdf=True)
    assert b"1 1 1 rg" in pdf                       # a node's white fill (#ffffff)
    assert b"0.184314 0.619608 0.266667 RG" in pdf  # an "ok" node's green stroke


@requires_pdf
def test_the_annexes_are_printed_open(tmp_path):
    """`<details>` has no reader to click it on paper. An archive asked for
    with `with_annexes` must contain them, not a row of summaries."""
    import weasyprint

    reporter = PdfReport(with_annexes=True)
    page = reporter.render(make_document(annexes=[("research", "the finding")]))
    document = weasyprint.HTML(string=page).render()

    def texts(box, found):
        for child in getattr(box, "children", []) or []:
            if getattr(child, "text", None):
                found.append(child.text)
            texts(child, found)
        return found

    printed = " ".join(texts(document.pages[0]._page_box, []))
    assert "the finding" in printed


# --------------------------------------------------- the shape, without the engine


def test_the_pdf_is_the_html_page_and_not_a_third_layout(monkeypatch):
    """The whole argument of #34: `HtmlReport.render` is a pure function of
    the document, so the PDF can share the layout without depending on the
    HTML *file*. What PdfReport adds is stylesheets — paged media, and the
    DAG's colours — and nothing else. If this ever fails, a second layout has
    started growing."""
    monkeypatch.setattr(report_pdf, "_engine", lambda: None)  # rendering uses none
    doc = make_document()
    printed = PdfReport().render(doc)
    assert PAGED_STYLE in printed and f"<style>{DAG_STYLE}</style>" in printed
    stripped = printed.replace(PAGED_STYLE, "").replace(f"<style>{DAG_STYLE}</style>", "")
    assert stripped == HtmlReport().render(doc)


def test_the_html_deliverable_is_untouched_by_the_seams_pdf_needed():
    """Both seams default to empty, so the browser's page is unchanged."""
    doc = make_document()
    page = HtmlReport().render(doc)
    assert "@page" not in page and "<style></style>" not in page
    assert "<style>" not in dag_svg(doc)          # only a print engine needs one


def test_binary_is_declared_by_the_reporter_not_guessed_from_the_name():
    assert is_binary_format("pdf") and is_binary_format(" PDF ")
    for text_format in ("markdown", "md", "html", "text", "", None, "pptx"):
        assert not is_binary_format(text_format)


def test_make_reporter_knows_pdf_and_composes_it_like_any_other(monkeypatch):
    monkeypatch.setattr(report_pdf, "_engine", lambda: object())
    assert isinstance(make_reporter("pdf"), PdfReport)
    assert isinstance(compose_reporters("pdf"), PdfReport)
    # .pdf collides with no other extension: a peer, not a rendering of HTML
    assert isinstance(compose_reporters("markdown,html,pdf"), MultiReporter)


def test_a_pdf_that_cannot_be_written_still_records_the_markdown(tmp_path):
    """#28's property, with a binary Reporter this time: the file written
    before the failing one exists, and only a `JobOutput` makes it findable."""

    class Broken(StubPdf):
        def serialize(self, document, path):
            raise OSError("no such font")

    reporter = MultiReporter([make_reporter("markdown"), Broken()])
    with pytest.raises(ReportWriteError) as failed:
        reporter.write(done_job("j21"), tmp_path)

    assert failed.value.report_format == "pdf"
    assert [(o.format, o.role) for o in failed.value.outputs] == [("markdown", "main")]
    assert Path(failed.value.outputs[0].path).is_file()


# ----------------------------------------- loading the engine: on first need → 0034, 0108

def _compose_in_a_fresh_process(tmp_path: Path, **env: str) -> dict:
    """`sys.modules` here already holds what earlier tests imported."""
    script = textwrap.dedent("""
        import asyncio, json, sys
        from jobsmith.app.agent import build_app
        from jobsmith.jobs.report import available_formats

        async def main():
            app = await build_app(db="memory", llm="fake")
            offered = available_formats(app.registry)
            await app.aclose()
            return offered

        offered = asyncio.run(main())
        print(json.dumps({"loaded": "weasyprint" in sys.modules, "offered": offered}))
    """)
    environ = {**os.environ, "XDG_DATA_HOME": str(tmp_path),
               "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "TAVILY_API_KEY": "",
               "JOBSMITH_REPORT_FORMAT": "", **env}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=environ, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_composing_the_app_does_not_load_the_pdf_engine(tmp_path):
    seen = _compose_in_a_fresh_process(tmp_path)
    assert seen["loaded"] is False
    if report_pdf.engine_installed():
        assert "pdf" in seen["offered"]      # loading nothing did not cost the offer


@requires_pdf
def test_a_deployment_whose_default_is_pdf_still_probes_at_startup(tmp_path):
    assert _compose_in_a_fresh_process(tmp_path, JOBSMITH_REPORT_FORMAT="pdf")["loaded"]


def test_the_offer_is_answered_without_the_engine(monkeypatch):
    def never():
        raise AssertionError("offering pdf must not probe the engine")

    monkeypatch.setattr(report_pdf, "_engine", never)
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: object())
    assert "pdf" in available_formats()
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: None)
    assert "pdf" not in available_formats()


@pytest.mark.parametrize(("break_it", "message"), [
    (no_distribution, r"jobsmith\[pdf\]"),     # one `pip install` fixes it...
    (no_libraries, "pango"),                   # ...and cannot fix this one
])
async def test_a_pdf_request_is_refused_in_create_job_saying_which_fix(
    store, checkpointer, tmp_path, monkeypatch, break_it, message
):
    break_it(monkeypatch)
    mgr = make_manager(store, checkpointer, tmp_path)
    with pytest.raises(ValueError, match=message):
        await mgr.create_job("compare the chairs", formats=["markdown", "pdf"])
    assert await mgr.list_jobs() == []


async def test_the_probe_runs_once_and_a_failure_stops_the_offer(
    store, checkpointer, tmp_path, monkeypatch
):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError(report_pdf._MISSING_LIBRARIES)

    monkeypatch.setattr(report_pdf, "_engine", broken)
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: object())
    assert "pdf" in available_formats()      # installed, not yet proved
    mgr = make_manager(store, checkpointer, tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match="pango"):
            await mgr.create_job("compare the chairs", formats=["pdf"])
    assert calls == [1]
    assert "pdf" not in available_formats()
    with pytest.raises(RuntimeError, match="pango"):
        make_reporter("pdf")                  # never composed once proved unrenderable


@requires_pdf
async def test_a_pdf_request_where_the_engine_runs_renders(store, checkpointer, tmp_path):
    mgr = make_manager(store, checkpointer, tmp_path)
    done = await mgr.run_job((await mgr.create_job("compare the chairs", formats=["pdf"])).job_id)

    assert done.status is JobStatus.DONE and done.error is None
    assert Path(done.report_path).read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize("renders", [
    False, pytest.param(True, marks=PDF_MARKS)])
async def test_the_document_step_keeps_a_pdf_only_if_it_renders(monkeypatch, renders):
    if not renders:
        no_libraries(monkeypatch)
    reply = json.dumps({"document": "named", "formats": ["pdf"]})
    step = DocumentIntent(Deps(llm=FakeLLM({"document step": reply})),
                          ("html", "markdown", "pdf"), confirm=renderable_formats)
    decided = await step.run({"query": "as a pdf"})
    assert decided == ({"document_formats": ["pdf"]} if renders else {})
