"""The PDF deliverable — and, first of all, whether the engine can run here.

`.[pdf]` is the only extra in this project with a **system**-level dependency:
WeasyPrint renders through pango/cairo, loaded with cffi at import time. So
the guard below is deliberately not `pytest.importorskip`: that would turn
"installed but the system libraries are missing" into a skip, and CI — which
installs every extra — would go green having tested nothing, which is the one
question this suite exists to answer. `find_spec` only *locates* the
distribution, so the import runs for real wherever the extra is installed and
a broken engine is a red suite, not a silent one.
"""
from __future__ import annotations

import importlib.util

import pytest

pdf_installed = importlib.util.find_spec("weasyprint") is not None
requires_pdf = pytest.mark.skipif(
    not pdf_installed, reason="the optional .[pdf] extra is not installed"
)


@requires_pdf
def test_weasyprint_renders_a_pdf():
    """Installing is not rendering: the wheel has no system library in it, and
    a machine without pango only finds out when bytes are asked for."""
    import weasyprint

    data = weasyprint.HTML(string="<h1>smoke</h1><p>a paragraph.</p>").write_pdf()
    assert data and data.startswith(b"%PDF-")
