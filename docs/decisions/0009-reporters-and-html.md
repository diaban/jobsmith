# 0009 — The Reporter seam, and an HTML deliverable with no dependency

- **Issue:** #9 · **PR:** #21
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0034](0034-pdf-deliverable.md), [0076](0076-nested-lists.md)

## From “Jobs layer (`jobs/`)”

**Producing the deliverable is a Reporter's job**, not the manager's (`jobs/report.py`): `build_document(job, registry)` makes a format-independent `JobDocument`, a Reporter serializes it and returns the `JobOutput`s describing what it wrote. Three ship — `MarkdownReport` (default), `HtmlReport` (`jobs/report_html.py`) and `PdfReport` (`jobs/report_pdf.py`) — sharing `FileReporter` (build the document, write one file, describe it); a subclass supplies `format`, `extension` and `render(doc)`, or **`serialize(document, path)`** when the file is bytes rather than text. That one overridable step is the whole of what a binary format changes: the naming of the file, the `build_document` call and the `role="main"` decision stay in `write`, uncopied. `make_reporter(format, registry)` selects one from `_reporter_classes()` (unknown name ⇒ `ValueError`), and `FileReporter.binary` — read by name through `is_binary_format(format)` — is how a caller holding only a finished job's declared format knows it cannot print the file.

**The HTML report takes no dependency and makes no request** (`jobs/report_html.py`): inline CSS (light/dark), and since a browser renders neither mermaid nor a fenced block, the same DAG edges are drawn as an inline SVG (`dag_svg`, longest-path columns left-to-right — the plan's step order is not assumed topological). The answer and the annexes are markdown by contract, so a deliberately small subset renderer (`markdown_to_html`: headings, lists, fences, rules, emphasis) converts them — **everything is `html.escape`d first and only our own tags are added afterwards**, which is why model output cannot open a tag; links are not rendered on purpose (an anchor means sanitizing `javascript:` URLs). Grow that renderer no further: if a report needs tables or links, take a dependency then.
