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

import importlib.util
import sys
from pathlib import Path

import pytest
from test_report_html import done_job, make_document

from jobsmith.jobs import report_pdf
from jobsmith.jobs.report import (
    FileReporter,
    JobDocument,
    MultiReporter,
    ReportWriteError,
    compose_reporters,
    is_binary_format,
    make_reporter,
)
from jobsmith.jobs.report_html import HtmlReport, dag_svg
from jobsmith.jobs.report_pdf import DAG_STYLE, PAGED_STYLE, PdfReport

pdf_installed = importlib.util.find_spec("weasyprint") is not None
requires_pdf = pytest.mark.skipif(
    not pdf_installed, reason="the optional .[pdf] extra is not installed"
)


class StubPdf(FileReporter):
    """A binary Reporter with no engine behind it — the shape, not the render.

    Everything about a binary deliverable that the framework must handle is
    here: bytes on disk, a format that declares itself binary. Used where the
    property under test belongs to the port or the manager rather than to
    WeasyPrint, so those tests hold on a machine with no engine at all.
    """

    format = "pdf"
    extension = "pdf"
    binary = True

    def serialize(self, document: JobDocument, path: Path) -> None:
        path.write_bytes(b"%PDF-1.7\n\xe2\xe3\xcf\xd3 not text\n%%EOF\n")


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


# ----------------------------------------------------- the deployment constraint


def test_the_engine_is_probed_when_the_reporter_is_composed(monkeypatch):
    """A format nothing can render must not be composed — the same rule the
    registry applies to a capability nothing can serve. Failing here means a
    daemon says so at startup instead of at the end of the first job that
    asked for a PDF."""
    def no_engine():
        raise RuntimeError("boom")

    monkeypatch.setattr(report_pdf, "_engine", no_engine)
    with pytest.raises(RuntimeError, match="boom"):
        make_reporter("pdf")


def test_a_missing_distribution_and_missing_libraries_are_different_messages(monkeypatch):
    """One `pip install` fixes the first and cannot fix the second, so a
    deployer must be able to tell them apart from the message alone."""
    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setitem(sys.modules, "weasyprint", None)   # import halts: ImportError
    with pytest.raises(RuntimeError, match=r"jobsmith\[pdf\]"):
        report_pdf._engine()

    class NoLibraries:
        """What cffi does when it cannot dlopen pango: OSError, at import."""

        def find_spec(self, name, path=None, target=None):
            if name == "weasyprint":
                raise OSError("cannot load library 'libpango-1.0.so.0'")
            return None

    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoLibraries(), *sys.meta_path])
    with pytest.raises(RuntimeError, match="pango"):
        report_pdf._engine()
