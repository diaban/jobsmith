"""PDF deliverable: the same page the HTML Reporter writes, printed.

The objection this module had to answer (#9, then #34) was that a PDF could
only be one of two bad things: a post-processor over another Reporter's
*file* — a dependency between Reporters the protocol does not express — or a
third re-implementation of the layout, which is what the
document/serializer split exists to prevent.

Both assume the PDF needs the HTML file. It does not. `HtmlReport.render` is
a pure function of a `JobDocument`, so `PdfReport` renders that same page in
memory and hands the string to a print engine. It is a peer Reporter: it
depends on no other Reporter having run, and it owns no layout of its own —
which is why it *subclasses* `HtmlReport`. What differs between them is one
step, the one `FileReporter` delegates: how the rendered document reaches
disk. A change to the HTML layout is a change to this deliverable too, and
that is the intent, not an accident.

Two things it does own. Stylesheets a screen has no use for — `PAGED_STYLE`,
because a page does not scroll, and `DAG_STYLE`, because a print engine
applies the document's CSS to HTML boxes only and would otherwise draw the
plan as black blocks — and the engine's absence, which is a **deployment** fact rather than a
Python one: WeasyPrint renders through pango/cairo, loaded at import time, so
a daemon asked for PDFs needs those libraries where it runs. That is why the
engine is probed when a `PdfReport` is *constructed* (at composition, from
`compose_reporters`) rather than when a job finally asks for its file: a
format nothing can render must not be composed, exactly as a capability
nothing can serve stays out of the registry.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

from .report import JobDocument
from .report_html import HtmlReport

# Rules the screen sheet has no use for. Everything else — colours, tables,
# the DAG, the markdown subset — is inherited unchanged.
PAGED_STYLE = """
@page { size: A4; margin: 16mm 15mm 18mm; }
/* The sheet already IS the frame: the screen layout's centring column and
   its outer padding would only narrow it a second time. */
main { max-width: none; margin: 0; padding: 0; }
body { font-size: 10.5pt; background: #ffffff; }
h1 { font-size: 1.5rem; margin-bottom: 1rem; }
h1, h2, h3 { break-after: avoid; }
/* A plan row or the DAG cut in half by a page break is unreadable, and both
   are small enough to be moved whole. */
.steps tr, .dag, pre { break-inside: avoid; }
/* Nobody clicks a triangle on paper. An annex is why a self-contained
   archive was asked for, so it is printed open, and the marker goes. */
details > * { display: block; }
details { break-inside: auto; }
summary { list-style: none; }
"""

# The DAG's colours, which a print engine cannot take from the page's own
# sheet (see `dag_svg`). They are the light half of `STYLE`'s palette, spelled
# out: a sheet of paper has no colour scheme to prefer.
DAG_STYLE = """
rect { fill: #ffffff; stroke: #e2e2df; stroke-width: 1.5 }
.ok rect { stroke: #2f9e44 }
.failed rect { stroke: #e03131 }
text { fill: #1d1d1f; text-anchor: middle; font: 13px monospace }
.edge { fill: none; stroke: #6b6b70; stroke-width: 1.5; marker-end: url(#arrow) }
marker path { fill: #6b6b70 }
"""

_MISSING_PACKAGE = (
    "the pdf deliverable needs the optional engine: pip install 'jobsmith[pdf]'"
)
_MISSING_LIBRARIES = (
    "weasyprint is installed but cannot load its system libraries (pango, "
    "cairo). A PDF deliverable is the one format with a dependency outside "
    "Python: install them where the agent runs — on Debian/Ubuntu, "
    "libpango-1.0-0 and libpangoft2-1.0-0 (see README)"
)


def _engine() -> ModuleType:
    """The print engine, imported on use.

    Kept out of module scope so that listing PDF among the known formats
    (`report._reporter_classes`) costs nothing to someone who never asks for
    one. The two ways it can be unavailable are one import apart and are not
    the same problem, so they are not the same message: no distribution, and
    a distribution whose system libraries are missing — the second raises
    `OSError` from cffi's `dlopen`, and is the failure a `pip install` will
    not fix.
    """
    try:
        import weasyprint
    except ImportError as missing:
        raise RuntimeError(_MISSING_PACKAGE) from missing
    except OSError as missing:                  # cffi could not dlopen pango/cairo
        raise RuntimeError(_MISSING_LIBRARIES) from missing
    return weasyprint


class PdfReport(HtmlReport):
    """The deliverable as a PDF: same document, same layout, paged."""

    format = "pdf"
    extension = "pdf"
    binary = True
    extra_style = PAGED_STYLE
    dag_style = DAG_STYLE

    def __init__(self, registry: object = None, *, with_annexes: bool = False):
        super().__init__(registry, with_annexes=with_annexes)
        self._weasyprint = _engine()

    def serialize(self, document: JobDocument, path: Path) -> None:
        """The one step that differs from the HTML Reporter."""
        self._weasyprint.HTML(string=self.render(document)).write_pdf(target=str(path))
