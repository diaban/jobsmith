"""The HTML deliverable: another Reporter over the same JobDocument.

Three properties matter here and nothing else does (wording and styling are
free to change): the page cannot be injected into by the model's own output,
it carries the same provenance the markdown one does, and the plan is drawn
as something a browser actually renders.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from jobsmith.app import build_app
from jobsmith.app.agent import pick_report_format
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.core.usage import Usage
from jobsmith.jobs.models import Job, JobStatus
from jobsmith.jobs.report import (
    JobDocument,
    MarkdownReport,
    PlanRow,
    build_document,
    make_reporter,
)
from jobsmith.jobs.report_html import HtmlReport, dag_svg, markdown_to_html


def make_document(**over) -> JobDocument:
    doc = JobDocument(
        title="compare A and B",
        request="compare A and B",
        job_id="j1",
        created_at="2026-09-01T00:00:00Z",
        finished_at="2026-09-01T00:03:00Z",
        answer="## Verdict\n\nA beats B.",
        plan_rationale="chain of three",
        plan=[
            PlanRow("research", [], "ok", "t1",
                    Usage(input_tokens=12_000, output_tokens=3_000, calls=2,
                          cost_usd=0.135, models=("claude-opus-5",))),
            PlanRow("analysis", ["research"], "ok", "t2"),
            PlanRow("critique", ["analysis"], "failed (boom)", "t3"),
            PlanRow("aside", [], "not run", "—"),
        ],
        usage=Usage(input_tokens=20_000, output_tokens=5_000, calls=6,
                    cost_usd=0.225, models=("claude-opus-5",)),
    )
    for key, value in over.items():
        setattr(doc, key, value)
    return doc


def tags_of(html: str) -> list[str]:
    found: list[str] = []

    class Collect(HTMLParser):
        def handle_starttag(self, tag, attrs):
            found.append(tag)

    Collect().feed(html)
    return found


def node_x(svg: str, name: str) -> float:
    """Left edge of one node box — where the layout actually put it."""
    match = re.search(rf'<rect x="([0-9.]+)"[^>]*/><text[^>]*>{name}</text>', svg)
    assert match, f"{name} not drawn"
    return float(match.group(1))


# ------------------------------------------------------------------ escaping

HOSTILE = (
    "<script>alert('pwned')</script>\n\n"
    "Tom & Jerry, 3 < 4, a \"quote\" and an <img src=x onerror=alert(1)>."
)


def test_model_output_cannot_open_a_tag():
    """The answer is LLM-generated text landing in a document: it must be
    readable as text and inert as markup, everywhere it appears."""
    doc = make_document(
        answer=HOSTILE,
        title="<script>t</script>",
        request="<b>req</b>",
        plan_rationale="<i>why</i>",
        annexes=[("<script>cap</script>", HOSTILE)],
    )
    page = HtmlReport().render(doc)

    tags = tags_of(page)
    assert "script" not in tags and "img" not in tags and "b" not in tags
    assert "<script>" not in page and "<img" not in page
    # ...and the text is still there, escaped, not dropped
    assert "&lt;script&gt;alert(&#x27;pwned&#x27;)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(1)&gt;" in page     # inert, still readable
    assert "Tom &amp; Jerry, 3 &lt; 4" in page


def test_a_code_fence_in_the_answer_stays_inert():
    page = HtmlReport().render(make_document(answer="```\n<script>x</script>\n```"))
    assert "<pre><code>&lt;script&gt;x&lt;/script&gt;</code></pre>" in page
    assert "script" not in tags_of(page)


def test_a_capability_name_is_escaped_in_the_table_and_the_diagram():
    doc = make_document(plan=[PlanRow("<script>", [], "ok", "t1")])
    page = HtmlReport().render(doc)
    assert "script" not in tags_of(page)
    assert "&lt;script&gt;" in page


# ------------------------------------------------------------------ markdown


def test_markdown_subset_covers_what_a_report_actually_contains():
    html = markdown_to_html(
        "# Title\n"
        "\n"
        "A paragraph with **bold**, _emphasis_ and `code`.\n"
        "\n"
        "- first angle\n"
        "- second angle\n"
        "\n"
        "1. step one\n"
        "\n"
        "---\n"
        "\n"
        '```json\n{"score": 0.9}\n```\n'
    )
    assert "<h2>Title</h2>" in html                 # demoted under the report's h1
    assert "<strong>bold</strong>" in html
    assert "<em>emphasis</em>" in html
    assert "<code>code</code>" in html
    assert "<ul>\n<li>first angle</li>\n<li>second angle</li>\n</ul>" in html
    assert "<ol>\n<li>step one</li>\n</ol>" in html
    assert "<hr>" in html
    assert '<pre><code>{&quot;score&quot;: 0.9}</code></pre>' in html
    assert "```" not in html                        # no raw fence left behind


def test_emphasis_inside_code_is_left_alone():
    assert markdown_to_html("`a_b_c`") == "<p><code>a_b_c</code></p>"


def test_an_unclosed_fence_does_not_swallow_the_rest_as_markup():
    html = markdown_to_html("text\n\n```\n<b>x</b>")
    assert "<pre><code>&lt;b&gt;x&lt;/b&gt;</code></pre>" in html


# ----------------------------------------------------------------- the DAG


def test_the_plan_is_drawn_as_svg_not_as_an_unrendered_mermaid_block():
    """A browser renders neither mermaid nor a fenced block; the same edges
    the markdown report emits are drawn here as real geometry."""
    page = HtmlReport().render(make_document())
    assert "```mermaid" not in page and "flowchart LR" not in page
    assert "<svg" in page and page.count('class="edge"') == 2   # two dependencies

    svg = dag_svg(make_document())
    for name in ("research", "analysis", "critique", "aside"):
        assert f">{name}</text>" in svg          # isolated step included too
    assert 'class="node ok"' in svg and 'class="node failed"' in svg


def test_a_step_is_placed_after_the_steps_it_depends_on():
    """Layout, not decoration: a dependency must sit in an earlier column,
    even when the plan does not list its steps in topological order."""
    doc = make_document(plan=[
        PlanRow("analysis", ["research"], "ok", "t2"),   # listed before its dep
        PlanRow("research", [], "ok", "t1"),
    ])
    svg = dag_svg(doc)
    assert node_x(svg, "research") < node_x(svg, "analysis")


def test_no_plan_no_diagram():
    doc = make_document(plan=[], plan_rationale="")
    assert dag_svg(doc) == ""
    page = HtmlReport().render(doc)
    assert "<svg" not in page and "About this job" in page


# -------------------------------------------------------- the deliverable


def test_the_page_carries_the_same_provenance_as_the_markdown_one():
    doc = make_document()
    page = HtmlReport().render(doc)
    for expected in (doc.title, doc.request, doc.job_id, doc.created_at,
                     "A beats B.", "About this job", "6 LLM calls", "~$0.2250 est.",
                     "chain of three", "15.0k tok"):
        assert expected in page
    assert page.startswith("<!doctype html>")
    assert "<title>compare A and B</title>" in page
    # self-contained: no network, whatever the reader's browser policy is
    assert "http://" not in page and "https://" not in page and "<script" not in page


def test_write_produces_one_html_output_and_report_path_points_at_it(tmp_path):
    job = Job(job_id="j9", status=JobStatus.DONE, query="analyse the thing",
              created_at="2026-09-01T00:00:00Z", final_answer="Here it is.")
    output = HtmlReport().write(job, tmp_path)

    assert output.format == "html" and output.role == "main"
    assert output.path == str(tmp_path / "j9.html")
    job.outputs = [output]
    assert job.report_path == output.path
    assert "Here it is." in (tmp_path / "j9.html").read_text(encoding="utf-8")


def test_annexes_are_folded_in_when_asked(tmp_path):
    class Cap:
        def render_report(self, result):
            return "**notes**\n\n- first angle"

    class Registry:
        def get(self, name):
            return Cap()

    job = Job(job_id="j8", status=JobStatus.DONE, query="q",
              created_at="", final_answer="a",
              plan={"steps": [{"capability": "research", "depends_on": []}]},
              results={"research": {"ok": True, "data": {}}})
    page = (tmp_path / "j8.html")
    HtmlReport(Registry(), with_annexes=True).write(job, tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "<summary>Step output — research</summary>" in text
    assert "<strong>notes</strong>" in text and "<li>first angle</li>" in text


# ------------------------------------------------------------- the selection


def test_make_reporter_picks_a_format_and_refuses_an_unknown_one():
    assert isinstance(make_reporter(), MarkdownReport)
    assert isinstance(make_reporter("markdown"), MarkdownReport)
    assert isinstance(make_reporter("HTML"), HtmlReport)
    assert make_reporter("html", "reg", with_annexes=True).registry == "reg"
    with pytest.raises(ValueError, match="unknown report format"):
        make_reporter("pdf")


def test_pick_report_format_prefers_the_argument_then_the_env(monkeypatch):
    monkeypatch.delenv("JOBSMITH_REPORT_FORMAT", raising=False)
    assert pick_report_format() == "markdown"
    monkeypatch.setenv("JOBSMITH_REPORT_FORMAT", "html")
    assert pick_report_format() == "html"
    assert pick_report_format("markdown") == "markdown"


async def test_the_composed_agent_can_hand_back_html(tmp_path):
    """End to end, keyless: the format chosen at composition is the file the
    job actually writes, and `report_path` still points at it."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(), db="memory",
                          reports_dir=str(tmp_path / "artifacts"), report_format="html")
    try:
        job = await app.manager.create_job("study the topic in depth")
        done = await app.manager.run_job(job.job_id)
        assert done.report_path.endswith(".html")
        assert [(o.format, o.role) for o in done.outputs] == [("html", "main")]
        page = Path(done.report_path).read_text(encoding="utf-8")
        assert page.startswith("<!doctype html>") and "<svg" in page
    finally:
        await app.aclose()


def test_both_reporters_read_the_same_document(tmp_path):
    """The point of the split: one document, two serializations, no layout
    logic duplicated — and the markdown one is untouched by any of this."""
    job = Job(job_id="j7", status=JobStatus.DONE, query="q",
              created_at="", final_answer="An answer.",
              plan={"steps": [{"capability": "research", "depends_on": []}]},
              results={"research": {"ok": True, "data": {}}},
              step_finished_at={"research": "t1"})
    doc = build_document(job)

    markdown = MarkdownReport().render(doc)
    html = HtmlReport().render(doc)
    assert "```mermaid" in markdown and "<svg" in html
    for expected in ("An answer.", "research", "j7"):
        assert expected in markdown and expected in html
