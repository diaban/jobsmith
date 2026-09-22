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
from jobsmith.app.agent import pick_report_formats
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.core.usage import Usage
from jobsmith.jobs import report as report_module
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.jobs.report import (
    JobDocument,
    MarkdownReport,
    MultiReporter,
    PlanRow,
    build_document,
    compose_reporters,
    make_reporter,
)
from jobsmith.jobs.report_html import HtmlReport, dag_svg, markdown_to_html


def make_document(**over) -> JobDocument:
    """The archive-shaped document: `provenance=True` unless a test says
    otherwise.

    Since #85 a deliverable shows one line about its run and nothing else, so
    most of what this file asserts — the plan table, the DAG, the *About this
    job* list — is the `with_provenance` rendering. The default is what
    `test_the_page_a_reader_opens_carries_one_line_about_the_run` covers.
    """
    doc = JobDocument(
        provenance=True,
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


# ------------------------------------------------------------- nested lists


def lists_of(html: str) -> list[str]:
    """The list structure alone: every `ul`/`ol`/`li` tag in document order,
    an end tag written `/ul`. What nesting *is*, with the prose taken out."""
    found: list[str] = []

    class Collect(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in ("ul", "ol", "li"):
                found.append(tag)

        def handle_endtag(self, tag):
            if tag in ("ul", "ol", "li"):
                found.append("/" + tag)

    Collect().feed(html)
    return found


def test_an_indented_item_is_nested_under_the_one_that_introduces_it():
    """#76: `line.strip()` before the bullet match lost the indentation, so a
    three-level specification came out as one flat list and the line that
    introduced a group became its sibling."""
    html = markdown_to_html(
        "- For each chair, obtain:\n"
        "  - exact name\n"
        "    - seat height\n"
        "- Verification plan:\n"
    )
    assert lists_of(html) == [
        "ul",
        "li", "ul",                       # "obtain:" holds what follows it
        "li", "ul",
        "li", "/li",                      # seat height, two levels down
        "/ul", "/li",
        "/ul", "/li",
        "li", "/li",                      # the plan is a sibling of the first
        "/ul",
    ]
    assert "<li>seat height</li>" in html


def test_a_flat_list_is_unchanged():
    """The fix is for the nested case; a list with no indentation must render
    exactly as it did, byte for byte."""
    assert markdown_to_html("- a\n- b\n") == "<ul>\n<li>a</li>\n<li>b</li>\n</ul>"
    assert markdown_to_html("1. a\n2. b\n") == "<ol>\n<li>a</li>\n<li>b</li>\n</ol>"


@pytest.mark.parametrize("indent", ["  ", "   ", "\t", " \t"])
def test_the_indentation_rule_is_two_spaces_or_a_tab_per_level(indent):
    """Written down, not inferred per document: one level every two spaces, a
    tab advancing to the next boundary, anything between rounding down."""
    assert lists_of(markdown_to_html(f"- a\n{indent}- b\n")) == [
        "ul", "li", "ul", "li", "/li", "/ul", "/li", "/ul"]


def test_one_space_is_not_a_level():
    assert lists_of(markdown_to_html("- a\n - b\n")) == [
        "ul", "li", "/li", "li", "/li", "/ul"]


def test_a_jump_of_several_levels_does_not_unbalance_the_tags():
    """The depth comes from the stack of open lists, never from the document:
    an item indented three levels below a flat one opens exactly one."""
    html = markdown_to_html("- a\n      - way in\n- back\n")
    assert lists_of(html) == [
        "ul", "li", "ul", "li", "/li", "/ul", "/li", "li", "/li", "/ul"]
    assert html.count("<ul>") == html.count("</ul>")
    assert html.count("<li>") == html.count("</li>")


def test_a_list_that_starts_indented_opens_at_the_top_level():
    assert lists_of(markdown_to_html("    - alone\n")) == ["ul", "li", "/li", "/ul"]


def test_a_numbered_sublist_is_a_list_of_its_own_inside_the_item():
    html = markdown_to_html("- a\n  - b\n  1. c\n")
    assert lists_of(html) == [
        "ul", "li",
        "ul", "li", "/li", "/ul",         # the bullets
        "ol", "li", "/li", "/ol",         # ...then the numbers, same level
        "/li", "/ul",
    ]


def test_a_nested_item_is_escaped_like_every_other():
    """Nesting adds tags of ours; it must add nothing of the model's."""
    html = markdown_to_html("- a\n  - <script>x</script>\n")
    assert "script" not in tags_of(html)
    assert "&lt;script&gt;x&lt;/script&gt;" in html


def test_a_nested_list_survives_into_the_printed_page():
    """The deliverable, not the helper: HTML is what the PDF prints."""
    page = HtmlReport().render(make_document(answer="- top\n  - under\n"))
    assert "<li>top\n<ul>\n<li>under</li>\n</ul>\n</li>" in page


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
    [output] = HtmlReport().write(job, tmp_path)

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
    # "pdf" is a known format since #34; "pptx" is deliberately not one and
    # will not become one — a deck is a capability, because designing it is a
    # generation and a Reporter does not think (#35)
    with pytest.raises(ValueError, match="unknown report format"):
        make_reporter("pptx")


def test_pick_report_formats_prefers_the_argument_then_the_env(monkeypatch):
    monkeypatch.delenv("JOBSMITH_REPORT_FORMAT", raising=False)
    assert pick_report_formats() == ["markdown"]
    monkeypatch.setenv("JOBSMITH_REPORT_FORMAT", "html")
    assert pick_report_formats() == ["html"]
    assert pick_report_formats("markdown") == ["markdown"]


def test_pick_report_formats_reads_a_comma_separated_list(monkeypatch):
    """One run, several deliverables — order is meaning: the first is main."""
    monkeypatch.setenv("JOBSMITH_REPORT_FORMAT", "markdown, html")
    assert pick_report_formats() == ["markdown", "html"]
    assert pick_report_formats("html,markdown") == ["html", "markdown"]
    # junk in the list degrades to the default rather than to an empty reporter
    monkeypatch.setenv("JOBSMITH_REPORT_FORMAT", " , ")
    assert pick_report_formats() == ["markdown"]


async def test_the_composed_agent_can_hand_back_html(tmp_path):
    """End to end, keyless: the format chosen at composition is the file the
    job actually writes, and `report_path` still points at it."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(), db="memory",
                          reports_dir=str(tmp_path / "artifacts"), report_format="html")
    try:
        job = await app.manager.create_job("study the topic in depth")
        done = await app.manager.run_job(job.job_id)
        assert done.report_path.endswith(".html")
        # deliverables only: a pack that also produces a file (`slide_deck`,
        # when `.[pptx]` is installed) adds annexes after them, which is #35's
        # business and not this test's
        deliverables = [o for o in done.outputs if o.role != "annex"]
        assert [(o.format, o.role) for o in deliverables] == [("html", "main")]
        page = Path(done.report_path).read_text(encoding="utf-8")
        # the deliverable a reader opens: the answer, then one line back to
        # the record (#85) — no About list, no steps table, no DAG
        assert page.startswith("<!doctype html>")
        assert '<p class="job-ref">' in page and job.job_id in page
        assert "<svg" not in page and "About this job" not in page
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
    doc = build_document(job, with_provenance=True)

    markdown = MarkdownReport().render(doc)
    html = HtmlReport().render(doc)
    assert "```mermaid" in markdown and "<svg" in html
    for expected in ("An answer.", "research", "j7"):
        assert expected in markdown and expected in html

    # ...and the default rendering is the same document without the record
    plain = build_document(job)
    for rendered in (MarkdownReport().render(plain), HtmlReport().render(plain)):
        assert "An answer." in rendered and "j7" in rendered
        assert "research" not in rendered and "About this job" not in rendered


# ------------------------------------------------- several formats at once


def done_job(job_id: str = "j10") -> Job:
    return Job(job_id=job_id, status=JobStatus.DONE, query="compare A and B",
               created_at="2026-09-01T00:00:00Z", final_answer="A beats B.")


def test_one_format_still_composes_to_that_one_reporter():
    """The ordinary case must not grow a wrapper: a single name gives back
    the same Reporter object it always did, so a one-format run is unchanged."""
    assert isinstance(compose_reporters("markdown"), MarkdownReport)
    assert isinstance(compose_reporters("html"), HtmlReport)
    assert isinstance(compose_reporters(["HTML"]), HtmlReport)
    assert compose_reporters("html", "reg", with_annexes=True).registry == "reg"


def test_two_formats_write_two_files_and_exactly_one_is_main(tmp_path):
    job = done_job()
    outputs = compose_reporters("markdown,html").write(job, tmp_path)

    assert [(o.format, o.role) for o in outputs] == [
        ("markdown", "main"), ("html", "alternate")]
    assert [Path(o.path).suffix for o in outputs] == [".md", ".html"]
    for output in outputs:                       # both really landed on disk
        assert "A beats B." in Path(output.path).read_text(encoding="utf-8")

    job.outputs = outputs
    assert job.report_path == outputs[0].path    # the main one, unambiguously


def test_the_first_format_asked_for_is_the_main_deliverable(tmp_path):
    """Order is the decision — and `format` is what /report announces."""
    reporter = compose_reporters("html,markdown")
    assert reporter.format == "html" and reporter.extension == "html"

    job = done_job("j11")
    job.outputs = reporter.write(job, tmp_path)
    assert [(o.format, o.role) for o in job.outputs] == [
        ("html", "main"), ("markdown", "alternate")]
    assert job.report_path.endswith(".html")


def test_aliases_of_one_format_do_not_write_the_same_file_twice(tmp_path):
    reporter = compose_reporters("markdown,md")
    assert isinstance(reporter, MarkdownReport)
    assert len(reporter.write(done_job("j12"), tmp_path)) == 1


def test_two_formats_claiming_one_extension_refuse_to_compose(monkeypatch):
    """The path is `{job_id}.{extension}`, so the extension is the collision
    key: two different Reporters sharing one would write the same file, the
    second overwriting the first, and the job would list two outputs for one
    file. Fail at composition, not silently at report time."""

    class MarkdownLite(MarkdownReport):     # a second flavour, same extension
        format = "md-lite"

    real = report_module.make_reporter

    def factory(name="markdown", registry=None, **policy):
        if (name or "").strip().lower() == "md-lite":
            return MarkdownLite(registry, **policy)
        return real(name, registry, **policy)

    monkeypatch.setattr(report_module, "make_reporter", factory)

    with pytest.raises(ValueError, match=r"both write \.md files"):
        compose_reporters("markdown,md-lite")
    # an alias of the SAME reporter is still just a repetition, not a clash
    assert isinstance(compose_reporters("md-lite,md-lite"), MarkdownLite)
    assert isinstance(compose_reporters("md-lite,html"), MultiReporter)


def test_an_unknown_name_anywhere_in_the_list_still_fails_loudly():
    # "pptx" again: the `slide_deck` capability writes one, and asking a
    # Reporter for it is still the error it always was
    with pytest.raises(ValueError, match="unknown report format"):
        compose_reporters("markdown,pptx")
    with pytest.raises(ValueError, match="unknown report format"):
        compose_reporters("pptx,markdown")
    with pytest.raises(ValueError, match="at least one reporter"):
        MultiReporter([])


def test_an_annex_is_never_demoted_to_a_sibling_format(tmp_path):
    """Only a second *main* is a rival for `report_path`. A Reporter that
    also emits per-step material keeps that material labelled as an annex."""

    class WithAnnex:
        format, extension = "markdown", "md"

        def write(self, job, directory):
            return [JobOutput(path="a.md", format="markdown"),
                    JobOutput(path="chart.svg", format="svg", role="annex",
                              produced_by="research")]

    outputs = MultiReporter([WithAnnex(), WithAnnex()]).write(done_job("j13"), tmp_path)
    assert [o.role for o in outputs] == ["main", "annex", "alternate", "annex"]


async def test_the_composed_agent_can_hand_back_both_formats(tmp_path):
    """End to end, keyless: `JOBSMITH_REPORT_FORMAT=markdown,html` is one run,
    two recorded deliverables — and `report_path` still points at the first."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(), db="memory",
                          reports_dir=str(tmp_path / "artifacts"),
                          report_format="markdown,html")
    try:
        job = await app.manager.create_job("study the topic in depth")
        done = await app.manager.run_job(job.job_id)
        deliverables = [o for o in done.outputs if o.role != "annex"]   # see above
        assert [(o.format, o.role) for o in deliverables] == [
            ("markdown", "main"), ("html", "alternate")]
        assert done.report_path.endswith(".md")
        paths = [Path(o.path) for o in deliverables]
        assert all(p.exists() for p in paths) and len({str(p) for p in paths}) == 2
        assert paths[1].read_text(encoding="utf-8").startswith("<!doctype html>")
    finally:
        await app.aclose()
