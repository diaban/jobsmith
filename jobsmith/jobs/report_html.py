"""HTML deliverable: the same `JobDocument`, serialized for a browser.

Why HTML gets its own module and not a `pandoc` call: the deliverable must
stay a single self-contained file with **no dependency and no network** — a
job report is often read from a laptop, mailed, or served by a future
artifacts tab, and a CDN reference that 404s in five years is not a
deliverable. So everything here is stdlib: inline CSS, an inline SVG for the
plan DAG, and a deliberately small markdown subset for prose.

Two things this file is careful about:

* **Escaping.** The answer and every annex are LLM-generated text landing in
  a document. Everything is `html.escape`d *first*; the markdown pass only
  ever adds tags of our own afterwards, so no input can open one. Links are
  not rendered on purpose — an anchor means sanitizing `javascript:` URLs,
  which is a decision, not an oversight.
* **The DAG.** Markdown emits a mermaid block, which GitHub and editors
  render. A browser does not, and an unrendered mermaid block is gibberish,
  so the same edges are drawn here as a real inline SVG — layered
  left-to-right, exactly what the mermaid `flowchart LR` says.
"""
from __future__ import annotations

import re
from html import escape

from .report import (
    FileReporter,
    JobDocument,
    PlanRow,
    format_step_usage,
    format_usage,
)

# ------------------------------------------------------------------ markdown

_INLINE = re.compile(
    r"`([^`]+)`"                          # `code`
    r"|\*\*(.+?)\*\*"                     # **strong**
    r"|(?<![\w*])\*([^*\n]+)\*(?![\w*])"  # *em*
    r"|(?<![\w_])_([^_\n]+)_(?![\w_])"    # _em_
)
_HEADING = re.compile(r"(#{1,6})\s+(.*)")
_BULLET = re.compile(r"[-*+]\s+(.*)")
_ORDERED = re.compile(r"\d+[.)]\s+(.*)")
_RULE = re.compile(r"(-{3,}|\*{3,}|_{3,})")


def _inline(escaped: str) -> str:
    """Inline markdown over *already escaped* text — we only add our own tags."""
    def sub(match: re.Match[str]) -> str:
        code, strong, star_em, under_em = match.groups()
        if code is not None:
            return f"<code>{code}</code>"
        if strong is not None:
            return f"<strong>{strong}</strong>"
        return f"<em>{star_em if star_em is not None else under_em}</em>"

    return _INLINE.sub(sub, escaped)


def markdown_to_html(text: str) -> str:
    """A small, safe markdown subset: headings, lists, fenced code, rules,
    paragraphs and inline emphasis.

    Small on purpose. It exists because two things in a job report are
    markdown by contract — the generated answer and whatever
    `Capability.render_report` returns — and dumping them into a `<pre>` would
    make the HTML report worse than the markdown one. It is not a markdown
    engine; if a report ever needs tables or links, take a dependency then.

    Headings are demoted one level: `#` in the answer is subordinate to the
    report's own `<h1>` title.
    """
    lines = text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None

    def close_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append("<p>" + "<br>".join(_inline(escape(x)) for x in paragraph) + "</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def open_list(tag: str) -> None:
        nonlocal list_tag
        if list_tag != tag:
            close_list()
            out.append(f"<{tag}>")
            list_tag = tag

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            close_paragraph()
            close_list()
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1                       # the closing fence (or EOF)
            out.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
            continue

        index += 1
        if not stripped:
            close_paragraph()
            close_list()
            continue
        if _RULE.fullmatch(stripped):
            close_paragraph()
            close_list()
            out.append("<hr>")
            continue
        heading = _HEADING.fullmatch(stripped)
        if heading:
            close_paragraph()
            close_list()
            level = min(len(heading.group(1)) + 1, 6)
            out.append(f"<h{level}>{_inline(escape(heading.group(2)))}</h{level}>")
            continue
        bullet = _BULLET.fullmatch(stripped)
        ordered = _ORDERED.fullmatch(stripped)
        if bullet or ordered:
            close_paragraph()
            open_list("ul" if bullet else "ol")
            item = (bullet or ordered).group(1)
            out.append(f"<li>{_inline(escape(item))}</li>")
            continue
        paragraph.append(stripped)

    close_paragraph()
    close_list()
    return "\n".join(out)


# ------------------------------------------------------------------- the DAG

NODE_HEIGHT = 34
NODE_MIN_WIDTH = 96
GAP_X = 52
GAP_Y = 16
PAD = 12
CHAR_WIDTH = 7.6          # ~13px monospace, wide enough for snake_case names


def _depths(doc: JobDocument) -> dict[str, int]:
    """Longest-path depth per step. Relaxed to a fixpoint rather than assuming
    the plan lists its steps in topological order — it does not have to."""
    depth = {row.capability: 0 for row in doc.plan}
    for _ in range(len(doc.plan)):
        changed = False
        for row in doc.plan:
            for dep in row.depends_on:
                if dep in depth and depth[dep] + 1 > depth[row.capability]:
                    depth[row.capability] = depth[dep] + 1
                    changed = True
        if not changed:
            break
    return depth


def dag_svg(doc: JobDocument) -> str:
    """The plan as an inline SVG flowchart, left to right. Empty string when
    there is no plan to draw."""
    if not doc.plan:
        return ""
    depth = _depths(doc)
    columns: list[list[PlanRow]] = []
    for row in doc.plan:
        while len(columns) <= depth[row.capability]:
            columns.append([])
        columns[depth[row.capability]].append(row)

    width = max(NODE_MIN_WIDTH,
                int(max(len(row.capability) for row in doc.plan) * CHAR_WIDTH) + 20)
    tallest = max(len(column) for column in columns)
    height = tallest * NODE_HEIGHT + (tallest - 1) * GAP_Y

    box: dict[str, tuple[float, float]] = {}       # capability -> (x, y) top-left
    nodes: list[str] = []
    for col, column in enumerate(columns):
        span = len(column) * NODE_HEIGHT + (len(column) - 1) * GAP_Y
        top = PAD + (height - span) / 2            # centre the shorter columns
        for i, row in enumerate(column):
            x = PAD + col * (width + GAP_X)
            y = top + i * (NODE_HEIGHT + GAP_Y)
            box[row.capability] = (x, y)
            nodes.append(
                f'<g class="node {_status_class(row.status)}">'
                f'<rect x="{x:.0f}" y="{y:.0f}" width="{width}" '
                f'height="{NODE_HEIGHT}" rx="6"/>'
                f'<text x="{x + width / 2:.0f}" y="{y + NODE_HEIGHT / 2 + 4:.0f}">'
                f"{escape(row.capability)}</text></g>"
            )

    edges: list[str] = []
    for src, dst in doc.dag_edges:
        if src not in box or dst not in box:
            continue
        x1, y1 = box[src]
        x2, y2 = box[dst]
        x1, y1 = x1 + width, y1 + NODE_HEIGHT / 2
        y2 += NODE_HEIGHT / 2
        bend = max(GAP_X * 0.6, (x2 - x1) * 0.5)
        edges.append(
            f'<path class="edge" d="M {x1:.0f} {y1:.0f} '
            f'C {x1 + bend:.0f} {y1:.0f}, {x2 - bend:.0f} {y2:.0f}, '
            f'{x2 - 8:.0f} {y2:.0f}"/>'
        )

    total_w = PAD * 2 + len(columns) * width + (len(columns) - 1) * GAP_X
    total_h = PAD * 2 + height
    return (
        f'<svg class="dag" viewBox="0 0 {total_w} {total_h}" width="{total_w}" '
        f'height="{total_h}" role="img" aria-label="Execution plan">'
        "<title>Execution plan</title>"
        '<defs><marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 8 4 L 0 8 z"/></marker></defs>'
        + "".join(edges) + "".join(nodes) + "</svg>"
    )


def _status_class(status: str) -> str:
    if status == "ok":
        return "ok"
    return "failed" if status.startswith("failed") else "skipped"


# -------------------------------------------------------------- the reporter

STYLE = """
:root {
  color-scheme: light dark;
  --bg: #fbfbfa; --fg: #1d1d1f; --muted: #6b6b70; --line: #e2e2df;
  --card: #ffffff; --accent: #3b5bdb; --ok: #2f9e44; --failed: #e03131;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #16171a; --fg: #e8e8ea; --muted: #9a9aa2; --line: #2c2d32;
    --card: #1d1e22; --accent: #91a7ff; --ok: #51cf66; --failed: #ff8787;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.75rem; line-height: 1.25; margin: 0 0 1.5rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 .75rem; }
h3, h4, h5, h6 { font-size: 1rem; margin: 1.5rem 0 .5rem; }
p, li { overflow-wrap: anywhere; }
hr { border: 0; border-top: 1px solid var(--line); margin: 2.5rem 0; }
a { color: var(--accent); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .875em;
  background: var(--card); border: 1px solid var(--line); border-radius: 4px;
  padding: 0 .25em; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: .875rem 1rem; overflow-x: auto; }
pre code { background: none; border: 0; padding: 0; }
.about { color: var(--muted); font-size: .9375rem; }
.about dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem;
  margin: 0; }
.about dt { font-weight: 600; color: var(--fg); }
.about dd { margin: 0; overflow-wrap: anywhere; }
.rationale { font-style: italic; }
.steps { width: 100%; border-collapse: collapse; margin: .5rem 0 1.5rem;
  font-size: .875rem; }
.steps th, .steps td { text-align: left; padding: .375rem .5rem;
  border-bottom: 1px solid var(--line); white-space: nowrap; }
.steps th { color: var(--fg); }
.status-ok { color: var(--ok); }
.status-failed { color: var(--failed); }
.scroll-x { overflow-x: auto; }
.dag { max-width: 100%; height: auto; }
.dag rect { fill: var(--card); stroke: var(--line); stroke-width: 1.5; }
.dag .ok rect { stroke: var(--ok); }
.dag .failed rect { stroke: var(--failed); }
.dag text { fill: var(--fg); text-anchor: middle;
  font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; }
.dag .edge { fill: none; stroke: var(--muted); stroke-width: 1.5;
  marker-end: url(#arrow); }
.dag marker path { fill: var(--muted); }
details { border-top: 1px solid var(--line); padding: .75rem 0; }
summary { cursor: pointer; font-weight: 600; }
"""


class HtmlReport(FileReporter):
    """The deliverable as a self-contained HTML page: same document, same
    order (answer first, provenance after), no external resource."""

    format = "html"
    extension = "html"

    def render(self, doc: JobDocument) -> str:
        parts = [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(doc.title)}</title>",
            f"<style>{STYLE}</style>",
            "</head><body><main>",
            f"<h1>{escape(doc.title)}</h1>",
            f'<section class="answer">{markdown_to_html(doc.answer)}</section>',
            "<hr>",
            '<section class="about">',
            "<h2>About this job</h2>",
            self._about(doc),
        ]
        if doc.plan:
            parts.append("<h3>Steps</h3>")
            if doc.plan_rationale:
                parts.append(f'<p class="rationale">{escape(doc.plan_rationale)}</p>')
            parts.append(self._steps(doc))
            parts.append(f'<div class="scroll-x">{dag_svg(doc)}</div>')
        parts.append("</section>")
        for heading, body in doc.annexes:
            parts += [
                "<details>",
                f"<summary>Step output — {escape(heading)}</summary>",
                markdown_to_html(body),
                "</details>",
            ]
        parts.append("</main></body></html>")
        return "\n".join(parts) + "\n"

    def _about(self, doc: JobDocument) -> str:
        rows = [
            ("Request", escape(doc.request)),
            ("Job", f"<code>{escape(doc.job_id)}</code>"),
            ("Started", escape(doc.created_at)),
            ("Finished", escape(doc.finished_at)),
        ]
        if doc.session_id:
            rows.append(("Session", f"<code>{escape(doc.session_id)}</code>"))
        # Cost belongs with the provenance: whoever reads the report pays for it.
        rows.append(("Usage", escape(format_usage(doc.usage))))
        cells = "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in rows)
        return f"<dl>{cells}</dl>"

    def _steps(self, doc: JobDocument) -> str:
        head = ("<tr><th>step</th><th>depends on</th><th>status</th>"
                "<th>usage</th><th>finished at</th></tr>")
        body = "".join(
            "<tr>"
            f"<td><code>{escape(row.capability)}</code></td>"
            f"<td>{escape(', '.join(row.depends_on)) or '—'}</td>"
            f'<td class="status-{_status_class(row.status)}">{escape(row.status)}</td>'
            f"<td>{escape(format_step_usage(row.usage))}</td>"
            f"<td>{escape(row.finished_at)}</td>"
            "</tr>"
            for row in doc.plan
        )
        return f'<div class="scroll-x"><table class="steps">{head}{body}</table></div>'
