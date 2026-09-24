"""The eval harness itself — and the structural tier it runs.

Two things are pinned here:

1. **The deterministic tier stays at 100%.** That is the CI gate the harness
   exists for: it runs on the fakes, needs no API key, and turns a broken
   router/planner/report prompt into a failing test instead of a code review
   someone has to eyeball. The LLM tier is deliberately absent — it is
   stochastic and must never be able to fail CI.
2. **The checks themselves are correct.** A scorer that never fires is worse
   than no scorer, so each property is fed an observation that violates it.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from evals import cases_for, run_suite, score
from evals.cases import GOLDEN_CASES, EvalCase
from evals.deliverable import ensure_readable, extract
from evals.harness import Observation
from evals.results import load_baseline, render_summary, summarize, write_result
from evals.scoring import CHECK_NAMES, step_failure_rate

# ---------------------------------------------------------------- the tier


@pytest.fixture(scope="module")
def structural_run():
    """One real pass of the structural tier, shared by the assertions below."""

    async def go():
        cases = cases_for("structural")
        started = time.perf_counter()
        observations, context = await run_suite(cases, provider="fake")
        return cases, summarize(cases, observations, tier="structural", context=context,
                                duration_s=time.perf_counter() - started)

    return asyncio.run(go())


@pytest.fixture(scope="module")
def html_structural_run():
    """The same tier, scored on the HTML Reporter instead of the markdown one."""

    async def go():
        cases = cases_for("structural")
        started = time.perf_counter()
        observations, context = await run_suite(
            cases, provider="fake", report_format="html")
        return cases, summarize(cases, observations, tier="structural", context=context,
                                duration_s=time.perf_counter() - started)

    return asyncio.run(go())


def test_structural_tier_is_perfect(structural_run):
    """The fakes are deterministic: anything under 100% is a real regression."""
    _, result = structural_run
    assert result.pass_rate == 1.0, "\n".join(
        f"{f['case']} {f['check']}: {f['detail']}" for f in result.failures
    )


def test_structural_tier_actually_exercises_every_check(structural_run):
    """A check nothing applies to would silently protect nothing."""
    _, result = structural_run
    never_applied = [n for n in CHECK_NAMES if not result.checks[n]["applicable"]]
    assert never_applied == []


def test_the_score_does_not_depend_on_the_deliverable_format(
    structural_run, html_structural_run
):
    """What the module docstring of `scoring.py` promises, actually pinned.

    The report checks used to be markdown-shaped while claiming otherwise:
    `# ` is not how HTML opens and escaping moves the strings they search
    for, so the golden set scored 13 checks lower purely on the format. Same
    runs, same score, is the property — and a docstring that promises it
    needs a test, or the next reformatting quietly breaks it again.
    """
    _, markdown = structural_run
    _, html = html_structural_run
    assert (markdown.report_format, html.report_format) == ("markdown", "html")
    assert html.pass_rate == 1.0, "\n".join(
        f"{f['case']} {f['check']}: {f['detail']}" for f in html.failures
    )
    assert html.checks == markdown.checks


def test_the_answer_is_scored_on_more_runs_than_the_file(structural_run):
    """#96 made most runs file-less, which is exactly how the answer checks
    could go dark while the tier kept reading 100%: gated on a file, they
    fell from 7 applicable runs to 2 and nothing failed. "Every check applies
    somewhere" (above) cannot see that — it held at 2 — so this pins the
    shape instead: the checks that read the ANSWER apply to every planned run
    that answered, and those are strictly more than the runs with a file."""
    cases, result = structural_run
    answered = [c for c in cases
                if c.expect_route == "plan" and c.expect_terminal == "answer"]
    for name in ("report_reader_facing", "report_answers_request"):
        assert result.checks[name]["applicable"] == len(answered), name
        assert (result.checks[name]["applicable"]
                > result.checks["report_written"]["applicable"]), name
    # ...and the golden set still carries silent requests AND requests for a
    # file, on both terminals — or one half of the contract goes unmeasured
    claims = {(c.expect_terminal, c.expect_document) for c in cases}
    assert {("answer", True), ("answer", False), ("unanswered", True)} <= claims


def test_structural_tier_covers_both_routes_and_the_guard(structural_run):
    cases, result = structural_run
    routes = {c.expect_route for c in cases}
    assert {"plan", "direct", None} <= routes
    assert any(c.expect_terminal == "user_error" for c in cases)
    assert result.registry, "the composed agent registered no capability"


def test_case_ids_are_unique():
    ids = [c.id for c in GOLDEN_CASES]
    assert len(ids) == len(set(ids))


def test_check_names_match_the_scorers():
    """CHECK_NAMES drives the summary's column order — it must not drift."""
    produced = [c.name for c in score(GOLDEN_CASES[0], Observation("x", "x"))]
    assert produced == list(CHECK_NAMES)


# ---------------------------------------------------------------- the checks

PLAN_CASE = EvalCase(id="c", query="q", expect_route="plan", min_steps=2,
                     must_include=("analysis",), must_exclude=("slide_deck",))

ANSWER = "a final answer long enough to pass the length floor"

#: What a retrieval step brought back, and what a step that never read it
#: writes instead: plausible, about the same subject, and built from none of
#: it. The pair is the whole of #81 — the second is what the run that opened
#: the issue produced, at 14.6k characters, while the first sat in `results`.
RETRIEVED = (
    "The pilot note records the deployment constraints the team measured: the "
    "index must stay under two gigabytes, a query must answer within four "
    "hundred milliseconds, and the nightly rebuild must finish inside a "
    "two-hour window. Memory binds first, peaking at seven gigabytes."
)
RECALLED = (
    "Deployments of this kind are usually bounded by three things: how large "
    "the artefact grows, how quickly it can be served, and how long a full "
    "refresh takes. Figures vary widely between installations."
)


def _obs(**kwargs) -> Observation:
    base = {
        "case_id": "c",
        "query": "q",
        "job_id": "job1",
        "route": "plan",
        "registry": ("research", "analysis"),
        "terminal_kind": "answer",
        "final_answer": ANSWER,
        "plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "analysis", "depends_on": ["research"]},
        ],
        "results": {"research": {"ok": True}, "analysis": {"ok": True}},
    }
    base.update(kwargs)
    obs = Observation(**base)
    if "report_text" not in kwargs:
        obs.report_text = (
            f"# title\n\n{obs.final_answer}\n\n---\n\n"
            "_Produced by job job1 — jobsmith job job1 shows the request, the "
            "plan, the steps and what it cost._\n"
        )
    return obs


def _status(case: EvalCase, obs: Observation, name: str) -> str:
    return next(c.status for c in score(case, obs) if c.name == name)


def test_a_clean_observation_passes_everything():
    statuses = {c.name: c.status for c in score(PLAN_CASE, _obs())}
    assert "fail" not in statuses.values(), statuses


@pytest.mark.parametrize(
    ("check", "broken"),
    [
        ("router_route", {"route": "direct"}),
        ("terminal_kind", {"terminal_kind": "escalated"}),
        ("plan_present", {"plan_steps": []}),
        ("plan_names_known", {"plan_steps": [{"capability": "nope", "depends_on": []}]}),
        ("plan_no_duplicates", {"plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "research", "depends_on": []},
        ]}),
        ("plan_deps_satisfiable", {"plan_steps": [
            {"capability": "research", "depends_on": ["ghost"]},
            {"capability": "analysis", "depends_on": []},
        ]}),
        ("plan_acyclic", {"plan_steps": [
            {"capability": "research", "depends_on": ["analysis"]},
            {"capability": "analysis", "depends_on": ["research"]},
        ]}),
        ("plan_size", {"plan_steps": [{"capability": "research", "depends_on": []}]}),
        ("plan_required_steps", {"plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "critique", "depends_on": []},
        ]}),
        ("plan_excluded_steps", {"plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "analysis", "depends_on": []},
            {"capability": "slide_deck", "depends_on": []},
        ], "registry": ("research", "analysis", "slide_deck")}),
        ("steps_all_ran", {"results": {"research": {"ok": True}}}),
        ("steps_all_ok", {"results": {"research": {"ok": True}, "analysis": {"ok": False}}}),
        ("report_written", {"report_text": None}),
        ("report_title", {"report_text": "no heading at all\n"}),
        ("report_answer", {"report_text": "# t\n\nsomething else entirely\n"}),
        # #85: the deliverable that names no run cannot be traced back to one
        ("report_provenance", {"report_text": f"# t\n\n{ANSWER}\n"}),
        ("run_completed", {"error": "boom"}),
        ("report_reader_facing", {
            "final_answer": "The comparison holds.\n\n## Next steps\n\n- confirm "
                            "the scope with whoever asked for this"}),
        # a run that declared it could not answer, in a file that says nothing
        # about it — the deliverable then reads exactly like a report (#59)
        ("refusal_declared", {"terminal_kind": "unanswered",
                              "report_text": "# t\n\nsomething plausible\n"}),
        # a run whose retrieval nothing downstream ever read (#81)
        ("grounding_reaches_reasoning", {
            "registry": ("web_search", "research", "analysis"),
            "plan_steps": [
                {"capability": "web_search", "depends_on": []},
                {"capability": "research", "depends_on": ["web_search"]},
                {"capability": "analysis", "depends_on": ["research"]},
            ],
            "results": {
                "web_search": {"ok": True, "data": {"documents": [
                    {"id": "u1", "text": RETRIEVED}]}},
                "research": {"ok": True, "data": {"notes": RECALLED}},
                "analysis": {"ok": True, "data": {"analysis": RECALLED}},
            },
        }),
        # a deliverable with nothing of the subject or the material in it (#73)
        ("report_answers_request", {
            "query": "compare ergonomic chairs for a tall adult",
            "material": ("# Research notes\n\nThe Aeron seats 159 kilos; the "
                         "Embody offers lumbar support and a taller backrest."),
            "final_answer": ("Before anything can be stated, a sourcing exercise "
                             "must establish which measurements exist."),
        }),
        # a run that planned no deck, delivering a text that says one exists (#77)
        ("answer_invents_no_file", {
            "final_answer": ANSWER + "\n\nA slide deck exists alongside this document."}),
        # a refusal that turned into the work plan it was told not to write (#73)
        ("refusal_is_bare", {
            "terminal_kind": "unanswered",
            "final_answer": ("Data to collect and sourcing plan.\n\n"
                             "Exact product name:\nManufacturer:\n"
                             "Maximum recommended weight:\n"),
        }),
    ],
)
def test_each_check_fires_on_its_own_violation(check, broken):
    assert _status(PLAN_CASE, _obs(**broken), check) == "fail"


def test_answer_invents_no_file_catches_the_sentence_that_opened_it():
    # #77, verbatim: no deck was planned and the outputs are the report alone.
    said = ("Il est aussi indiqué qu'un fichier slide deck existe parallèlement "
            "à ce document, mais son contenu n'est pas reproduit ici.")
    obs = _obs(final_answer=f"{ANSWER}\n\n{said}", output_formats=("markdown", "html"))
    assert _status(PLAN_CASE, obs, "answer_invents_no_file") == "fail"


@pytest.mark.parametrize(
    ("said", "changed"),
    [
        # the file exists: naming it is the truth
        ("The slide deck delivered alongside this document has the figures.",
         {"output_formats": ("markdown", "pptx")}),
        ("See the attached PDF.", {"output_formats": ("pdf",)}),
        ("The attached page restates it.", {"output_formats": ("html",)}),
        # the word was the request's, or the material's, before it was the answer's
        ("PDF export is the slower path.", {"query": "compare PDF libraries"}),
        ("The manufacturer's PDF datasheet gives 4.2.",
         {"material": "datasheet (PDF): COP 4.2"}),
    ],
)
def test_answer_invents_no_file_leaves_a_true_or_borrowed_mention_alone(said, changed):
    obs = _obs(final_answer=f"{ANSWER}\n\n{said}", **changed)
    assert _status(PLAN_CASE, obs, "answer_invents_no_file") == "pass"


def test_answer_invents_no_file_does_not_read_ordinary_words_as_files():
    # "travaux annexes" came out of a real run and claims nothing.
    obs = _obs(final_answer=f"{ANSWER}\n\nLes travaux annexes varient; the slides "
                            "of the price curve are steep.")
    assert _status(PLAN_CASE, obs, "answer_invents_no_file") == "pass"


def test_document_format_scores_the_format_the_request_asked_for():
    """#90, from the instrument's end: the case asks, the run answers.

    Against the record it would be unfalsifiable — a job stores the format it
    chose, so asking the job whether it chose right is asking the defect to
    report itself. The case says "html" because the request says html.
    """
    case = EvalCase(id="c", query="compare X and Y, as an html page",
                    expect_format="html")
    available = ("html", "markdown")

    wrong = _obs(report_path="/tmp/j.md", report_format="markdown",
                 formats_available=available)
    assert _status(case, wrong, "document_format") == "fail"

    # the defect in its loudest form: the request named a file, none was written
    assert _status(case, _obs(formats_available=available), "document_format") == "fail"

    right = _obs(report_path="/tmp/j.html", report_format="html",
                 formats_available=available)
    assert _status(case, right, "document_format") == "pass"


def test_document_format_skips_what_this_deployment_cannot_render():
    """`.[pdf]` needs pango where the run happens — a fact about the machine.

    A skip leaves the denominator, so this must be one and not a failure, for
    the same reason `must_include` skips a capability that is not registered.
    """
    case = EvalCase(id="c", query="give me that as a pdf", expect_format="pdf")
    obs = _obs(report_path="/tmp/j.md", report_format="markdown",
               formats_available=("html", "markdown"))
    assert _status(case, obs, "document_format") == "skip"


def test_report_reader_facing_catches_the_run_that_opened_it():
    """#58, as it actually arrived: a deliverable addressed to its producer.

    Both markers below are from the observed deck — a request for a printable
    one-pager that came back as a status report on the run. The check reads
    the *answer*, not the file: the provenance a Reporter adds is about the run
    by design and quotes the request, so scanning the whole document would fire
    on the scaffolding.
    """
    observed = ("Contraintes de livrable.\n\n## Prochaines étapes\n\n"
                "- Option A/B selon la préférence exprimée")
    assert _status(PLAN_CASE, _obs(final_answer=observed), "report_reader_facing") == "fail"
    # the same words in the report's provenance, not in the answer: not a hit
    clean = _obs(report_text=f"# t\n\n{ANSWER}\n\n- Request: prochaines étapes\n"
                             "- Job: job1\n\n| research | analysis |\n")
    assert _status(PLAN_CASE, clean, "report_reader_facing") == "pass"


@pytest.mark.parametrize("name", ["report_reader_facing", "report_answers_request"])
def test_the_answer_checks_read_the_answer_whether_or_not_a_file_was_written(name):
    """Gated on a file, these two went dark for every silent request once
    #96 stopped writing one — and a skip is not a failure, so nothing said
    so. They read the answer, which reaches its reader either way."""
    producer_facing = ("## Next steps\n\n- Option A or option B, please confirm")
    obs = _obs(final_answer=producer_facing, report_text=None, report_path=None,
               deliverable_expected=False,
               query="compare the approaches to caching", material=RETRIEVED)
    silent = EvalCase(id="c", query="compare the approaches to caching",
                      expect_route="plan", expect_document=False)
    assert _status(silent, obs, name) == "fail"
    # ...while the checks that read the FILE rightly skip that run
    assert _status(silent, obs, "report_written") == "skip"


def test_the_answer_checks_skip_a_reply_that_had_no_plan():
    """#80's guard, which falls out of re-gating on the answer: a direct
    reply is a chat turn by construction, and scoring it for a deliverable's
    register is the confusion #80 names — even when a file WAS asked for."""
    direct = EvalCase(id="c", query="q", expect_route="direct", expect_document=True)
    obs = _obs(route="direct", plan_steps=[], results={},
               final_answer="Let me know if you would like more.")
    assert _status(direct, obs, "report_reader_facing") == "skip"
    assert _status(direct, obs, "report_answers_request") == "skip"


def test_report_answers_request_catches_the_run_that_opened_it():
    """#73, as it actually arrived: the deliverable was `critique`, not an answer.

    A request for a comparison of ergonomic chairs; four steps ok; 14.7k
    characters of sourced specifications and 8.3k of analysis with real
    conclusions in the material. The document that came out had sections
    named *Data to collect and sourcing plan*, *Verification plan* and a
    blank template, and not one line about a chair.

    Both halves are asserted, because either alone would be gameable: the
    document must name its subject, and it must carry what the material was
    about. The second is what a restatement of the brief cannot fake — note
    that the failing answer below *does* name the subject.
    """
    case = EvalCase(id="c", query="compare 6 ergonomic chairs for a 100 kg adult")
    material = (
        "# Research notes\n\nAeron size C: announced capacity 159 kg, seat "
        "height 41-53 cm, adjustable lumbar. Steelcase Leap: capacity 181 kg, "
        "wide backrest. Embody: lumbar support, narrower seat.\n\n"
        "# Analysis\n\nThe Leap and the Aeron both cover the height and the "
        "capacity; the Embody trades capacity for lumbar adjustment."
    )
    observed = (
        "## Data and requirements to answer\n\n"
        "## Data to collect and sourcing plan\n\n"
        "## Verification plan\n\n"
        "## Expected structure of the chairs data sheets (to fill in)\n\n"
        "Exact product name:\nManufacturer:\nMaximum recommended weight:\n"
    )
    obs = _obs(query=case.query, material=material, final_answer=observed)
    assert _status(case, obs, "report_answers_request") == "fail"

    # the same material, answered: the subject and what the material said
    answered = _obs(
        query=case.query,
        material=material,
        final_answer=(
            "For a 100 kg adult, the Steelcase Leap is the safest fit: its "
            "announced capacity is 181 kg and its backrest is the widest of "
            "the three. The Aeron size C is announced at 159 kg with a seat "
            "height of 41-53 cm and adjustable lumbar support; the Embody "
            "trades capacity for a narrower seat."
        ),
    )
    assert _status(case, answered, "report_answers_request") == "pass"


def test_a_refusal_that_merely_restates_the_request_is_not_an_answer():
    """The half a subject-term count cannot do on its own.

    This text names every subject word the request used — it is the request,
    written back out — and says nothing the material said. `must not be
    satisfiable by repeating the brief` is measured against the generator's
    input, which is the only place a restatement has nothing to borrow from.
    """
    case = EvalCase(id="c", query="compare ergonomic chairs for a tall adult")
    obs = _obs(
        query=case.query,
        material="Aeron: announced 159 kilos. Leap: wider backrest, 181 kilos.",
        final_answer=("This document concerns ergonomic chairs suitable for a "
                      "tall adult, as requested."),
    )
    assert _status(case, obs, "report_answers_request") == "fail"


def test_grounding_reaches_reasoning_catches_the_run_that_opened_it():
    """#81, as it actually arrived: the plan drew an edge that carried nothing.

    `web_search → research → analysis`, every step ok, and the notes the
    analysis reasoned over were written from the model's memory while the
    retrieved specifications sat unread in `results`. Nothing else here sees
    it: the deliverable was handed the material directly by the merger, so
    `report_answers_request` was green for the whole of that run.
    """
    def observed(research: str, analysis: str) -> Observation:
        return _obs(
            query="list the deployment constraints and say which binds first",
            registry=("web_search", "research", "analysis"),
            plan_steps=[
                {"capability": "web_search", "depends_on": []},
                {"capability": "research", "depends_on": ["web_search"]},
                {"capability": "analysis", "depends_on": ["research"]},
            ],
            results={
                "web_search": {"ok": True, "data": {"documents": [
                    {"id": "u1", "title": "note", "text": RETRIEVED}]}},
                "research": {"ok": True, "data": {"notes": research}},
                "analysis": {"ok": True, "data": {"analysis": analysis}},
            },
        )

    name = "grounding_reaches_reasoning"
    assert _status(PLAN_CASE, observed(RECALLED, RECALLED), name) == "fail"
    # the same plan, with the steps actually reading what was retrieved
    grounded = (
        "Index: must stay under two gigabytes [u1]. Query: four hundred "
        "milliseconds [u1]. Nightly rebuild: inside a two-hour window, "
        "peaking at seven gigabytes of memory on the single node measured."
    )
    assert _status(PLAN_CASE, observed(grounded, grounded), name) == "pass"
    # and one step reading it is enough for the material to be in the run
    assert _status(PLAN_CASE, observed(grounded, RECALLED), name) == "pass"


@pytest.mark.parametrize(
    ("why", "changed"),
    [
        ("no step retrieved anything", {}),
        ("nothing depends on the retrieval", {
            "plan_steps": [
                {"capability": "web_search", "depends_on": []},
                {"capability": "research", "depends_on": []},
            ],
            "results": {
                "web_search": {"ok": True, "data": {"documents": [
                    {"id": "u1", "text": RETRIEVED}]}},
                "research": {"ok": True, "data": {"notes": RECALLED}},
            },
        }),
        ("the retrieval failed, so there is no material", {
            "plan_steps": [
                {"capability": "web_search", "depends_on": []},
                {"capability": "research", "depends_on": ["web_search"]},
            ],
            "results": {
                "web_search": {"ok": False, "error": "nothing usable"},
                "research": {"ok": True, "data": {"notes": RECALLED}},
            },
        }),
    ],
)
def test_grounding_reaches_reasoning_skips_what_it_cannot_measure(why, changed):
    """A skip leaves the denominator, so each of these must be one, not a fail.

    The middle case is the honest limit written into the docstring: this
    measures the edges a plan **draws**. A plan that schedules the retrieval
    and the reasoning side by side draws none, and whether it should have is
    a question about planning.
    """
    assert _status(PLAN_CASE, _obs(**changed), "grounding_reaches_reasoning") == "skip", why


def test_refusal_is_bare_catches_the_shape_the_prompts_forbade_and_got():
    """#73: nothing arbitrated, so the refusal became the maximal forbidden thing."""
    case = EvalCase(id="c", query="summarise the attached report")
    short = _obs(terminal_kind="unanswered",
                 final_answer="Nothing in the material covers what was asked.")
    assert _status(case, short, "refusal_is_bare") == "pass"

    for observed in (
        "Nothing was provided.\n\n" + "filler words repeated. " * 120,   # too long
        "Nothing was provided. I can prepare the template if you would like.",
        "Missing data.\n\nProduct:\nManufacturer:\nWeight:\n",
    ):
        obs = _obs(terminal_kind="unanswered", final_answer=observed)
        assert _status(case, obs, "refusal_is_bare") == "fail", observed[:40]


# --------------------------------------------------- reading a deliverable

def test_extract_reads_html_as_the_text_a_reader_sees():
    doc = extract(
        "<!doctype html><html><head><title>t</title><style>h1{color:red}</style></head>"
        "<body><h1>Compare the options</h1>"
        "<p>Rock &amp; roll &lt;is&gt; the answer</p>"
        "<ul><li><code>web_search</code> ran</li></ul></body></html>",
        "html",
    )
    assert doc.title == "Compare the options"       # not the first line, which is a doctype
    assert doc.contains("Rock & roll <is> the answer")   # escaping undone
    assert doc.contains("web_search")
    assert "color" not in doc.text                  # a stylesheet is not content


def test_extract_flattens_markup_so_both_formats_carry_the_same_text():
    same = "- **web_search** produced 3 notes"
    markdown = extract(f"# T\n\n{same}\n", "markdown")
    html = extract("<h1>T</h1><ul><li><strong>web_search</strong> produced 3 notes</li></ul>",
                   "html")
    assert markdown.title == html.title == "T"
    assert markdown.text == html.text
    # and the needle is flattened the same way, whichever side it came from
    assert markdown.contains(same) and html.contains(same)


def test_an_unknown_format_degrades_to_plain_text():
    """A future Reporter scores on its content from day one; only the title waits."""
    doc = extract("just words, no markup at all", "pptx")
    assert doc.contains("just words") and doc.title == ""


def test_a_text_format_is_readable_and_a_binary_one_is_refused():
    """What `extract` covers, and where the line is drawn — by name, once.

    `is_binary_format` is the lookup, so a Reporter added later declares
    which side it is on and this needs no second list of what is text.
    """
    assert ensure_readable("markdown") == "markdown"
    assert ensure_readable("PPTX ") == "pptx"          # unknown: read as text
    with pytest.raises(ValueError, match="read the deliverable as text"):
        ensure_readable("pdf")


def test_a_binary_deliverable_is_refused_before_a_single_case_runs():
    """Not scored-then-discarded: refused at the entry of the suite."""
    with pytest.raises(ValueError, match="pdf"):
        asyncio.run(run_suite(cases_for("structural"), provider="fake",
                              report_format="pdf"))


HTML_HEAD = "<!doctype html><html><body>"


@pytest.mark.parametrize(
    ("check", "body"),
    [
        ("report_title", "<p>no heading at all</p>"),
        ("report_answer", "<h1>t</h1><p>something else entirely</p>"),
        ("report_provenance", f"<h1>t</h1><p>{ANSWER}</p>"),
    ],
)
def test_the_report_checks_still_fire_on_an_html_deliverable(check, body):
    """Reading through the markup must not become reading past everything: a
    stripper permissive enough to pass any HTML would score nothing at all."""
    obs = _obs(report_text=f"{HTML_HEAD}{body}</body></html>", report_format="html")
    assert _status(PLAN_CASE, obs, check) == "fail"


def test_an_html_deliverable_with_everything_in_it_passes():
    obs = _obs(
        report_format="html",
        report_text=(
            f"{HTML_HEAD}<h1>title</h1><p>{ANSWER}</p><hr>"
            '<p class="job-ref">Produced by job job1 — jobsmith job job1 shows '
            "the request, the plan, the steps and what it cost.</p>"
            "</body></html>"
        ),
    )
    assert "fail" not in {c.name: c.status for c in score(PLAN_CASE, obs)}.values()


def test_a_broken_run_skips_the_downstream_checks():
    """One harness exception must not be counted as seventeen prompt failures."""
    checks = {c.name: c.status for c in score(PLAN_CASE, _obs(error="boom"))}
    assert checks["run_completed"] == "fail"
    assert set(checks.values()) == {"fail", "skip"}
    assert sum(1 for s in checks.values() if s == "fail") == 1


def test_inapplicable_checks_do_not_dilute_the_score():
    direct = EvalCase(id="c", query="q", expect_route="direct")
    obs = _obs(route="direct", plan_steps=[], results={})
    checks = score(direct, obs)
    assert all(c.status != "fail" for c in checks)
    assert any(c.status == "skip" for c in checks)


def test_step_failure_rate():
    assert step_failure_rate(_obs()) == 0.0
    assert step_failure_rate(_obs(results={"a": {"ok": True}, "b": {"ok": False}})) == 0.5
    assert step_failure_rate(_obs(results={})) is None


# ---------------------------------------------------------------- results io

def _result(cases, observations):
    return summarize(cases, observations, tier="structural",
                     context={"agent": "default", "provider": "fake",
                              "registry": ["research", "analysis"], "repeat": 1},
                     duration_s=0.1)


def test_results_are_written_and_found_as_a_baseline(tmp_path):
    cases = [PLAN_CASE]
    first = _result(cases, [_obs()])
    path = write_result(first, tmp_path)
    assert json.loads(path.read_text())["tier"] == "structural"

    second = _result(cases, [_obs(route="direct")])
    baseline = load_baseline(second, tmp_path)
    assert baseline is not None
    assert baseline.run_id == first.run_id
    assert baseline.pass_rate == 1.0


def test_a_baseline_from_another_provider_is_not_comparable(tmp_path):
    other = _result([PLAN_CASE], [_obs()])
    other.provider = "anthropic"
    write_result(other, tmp_path)
    mine = _result([PLAN_CASE], [_obs()])
    assert load_baseline(mine, tmp_path) is None


def test_a_baseline_over_a_different_case_set_is_not_comparable(tmp_path):
    """A `--case` slice must not become the baseline of a full run."""
    slice_run = _result([PLAN_CASE], [_obs()])
    write_result(slice_run, tmp_path)
    full = _result([PLAN_CASE, EvalCase(id="other", query="q")], [_obs()])
    assert load_baseline(full, tmp_path) is None


def test_the_summary_shows_the_delta_against_a_baseline():
    cases = [PLAN_CASE]
    before = _result(cases, [_obs()])
    after = _result(cases, [_obs(route="direct")])
    text = render_summary(after, before)
    assert "baseline:" in text
    assert "router_route" in text
    assert "-100.0" in text                      # the check that regressed
    assert "expected 'plan', got 'direct'" in text


# ---------------------------------------------------------------- the cli

def test_cli_runs_and_gates_the_structural_tier(capsys):
    from evals.__main__ import main

    assert main(["--llm", "fake", "--no-write"]) == 0
    assert "overall" in capsys.readouterr().out


def test_the_cli_refuses_a_binary_format_and_stores_nothing(tmp_path, capsys):
    """The refusal is worth having only because of what it does NOT leave behind.

    A `--report-format pdf` run scores 2/10 — every report check failing on
    the format rather than on the agent — and `load_baseline` matches on
    tier/agent/provider/cases and deliberately not on the format, so that
    record would become the next markdown run's baseline and print a
    regression nobody caused. So the assertion that matters is the empty
    results directory, not the message.
    """
    from evals.__main__ import main

    code = main(["--llm", "fake", "--report-format", "pdf",
                 "--results-dir", str(tmp_path)])
    # First, and on purpose: what the run left on disk. A refusal that still
    # stored its record would pass every assertion about the message.
    assert list(tmp_path.iterdir()) == []
    assert code == 2
    assert "read the deliverable as text" in capsys.readouterr().err


def test_a_refused_run_cannot_become_a_baseline(tmp_path):
    """The same property from the other end: what the NEXT run compares against.

    The refused run covers the whole golden set on the fake provider, which is
    exactly what makes it comparable — same tier, agent, provider and cases —
    so if it were stored, `load_baseline` would hand it to the next markdown
    run and print a Δ of several points that no prompt caused. The earlier
    markdown record must still be the one that comes back.
    """
    from evals.__main__ import main

    golden = cases_for("structural")
    earlier = _result(golden, [_obs(case_id=golden[0].id)])
    write_result(earlier, tmp_path)
    code = main(["--llm", "fake", "--report-format", "pdf",
                 "--results-dir", str(tmp_path)])

    baseline = load_baseline(_result(golden, [_obs(case_id=golden[0].id)]), tmp_path)
    assert baseline is not None
    assert baseline.report_format == "markdown"
    assert baseline.run_id == earlier.run_id
    assert code == 2


def test_cli_lists_the_golden_set(capsys):
    from evals.__main__ import main

    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(case.id in out for case in GOLDEN_CASES)
