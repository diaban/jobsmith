"""The golden set: requests plus the properties their run must hold.

A case never declares an expected *answer* — only what must be true of the
run whatever the model says. Keep every query domain-neutral: the harness is
agent-agnostic and `make leak-check` scans this package.

Tiers (`EvalCase.tiers`) say where a case is meaningful:

- `structural` — the deterministic fakes can satisfy it. `KeywordLLM` routes
  on a small keyword list and chains every registered capability, so only
  cases whose expectation survives that crudeness belong here.
- `llm` — needs a real model. Cases that a keyword fake would answer by
  accident, or fail by accident, are llm-only: scoring them against the fake
  would measure the fake, not the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

STRUCTURAL = "structural"
LLM = "llm"
BOTH = (STRUCTURAL, LLM)


@dataclass(frozen=True)
class EvalCase:
    """One request and the properties its run must satisfy."""

    id: str
    query: str
    #: "plan" | "direct" | None (None = the case makes no claim about triage)
    expect_route: str | None = None
    #: "answer" | "unanswered" | "user_error" | "escalated"
    expect_terminal: str = "answer"
    #: bounds on the planned DAG — only checked when set
    min_steps: int = 0
    max_steps: int | None = None
    #: capabilities the plan should contain, IF the composed agent has them
    #: (a registry is agent- and configuration-dependent, so absent names are
    #: skipped rather than failed)
    must_include: tuple[str, ...] = ()
    #: capabilities the plan must NOT contain, IF registered — the mirror of
    #: `must_include`, and the only way to score a spec that is too easy to
    #: reach (#61): a step planned for a request it does not serve delivers
    #: the wrong file, and the plan is where that is visible.
    must_exclude: tuple[str, ...] = ()
    #: Should this request leave a document behind (#84)? None makes no
    #: claim. It is the case's own reading of the request, deliberately not
    #: read back off the job: a run records what it decided, and "was that
    #: the right decision" is a question only the request can answer.
    expect_document: bool | None = None
    #: The format the request asked for IN WORDS (#90) — "the engine reads
    #: the sentence" is only measurable against the case, for the same reason
    #: `expect_document` is: the job records what it decided. `None` makes no
    #: claim, and a format this deployment cannot render is skipped rather
    #: than failed (`.[pdf]` needs pango where the run happens), exactly as an
    #: unregistered capability is in `must_include`.
    expect_format: str | None = None
    tiers: tuple[str, ...] = BOTH
    inputs: dict = field(default_factory=dict)
    note: str = ""


#: A file a case can name, and the reference it names it by.
#:
#: `read_files` opens a path only inside a root the deployment declared
#: readable, which under `build_app` is the reports directory — a scratch one
#: per run here. A case therefore cannot carry a path: it carries this
#: placeholder, and the harness writes the fixture into that directory and
#: swaps the real path in. A hard-coded path would be a path to one machine.
#:
#: The content is deliberately short, specific and domain-neutral: short so
#: what a step did with it is legible in the step's own output, specific so
#: that "this text reached the reasoning" is a question a bag of words can
#: answer at all.
FIXTURE_REF = "{fixture}"
FIXTURE_NAME = "deployment-note.md"
FIXTURE_TEXT = """# Pilot note

This note records the deployment constraints the pilot team measured. The
index must stay under two gigabytes, a query must answer within four hundred
milliseconds, and the nightly rebuild must finish inside a two-hour window.
Memory binds first: the rebuild peaks at seven gigabytes on the single node.
"""


GOLDEN_CASES: tuple[EvalCase, ...] = (
    # ---------------- triage: obviously direct ----------------
    EvalCase(
        id="direct_capabilities",
        query="what can you do?",
        expect_route="direct",
        expect_document=False,
        note=(
            "a question about the assistant itself needs no capability at all "
            "— and no document either: the file it used to leave was a chat "
            "turn with a provenance section stapled to it (#84)"
        ),
    ),
    EvalCase(
        id="direct_greeting",
        query="hello there",
        expect_route="direct",
        expect_document=False,
        note="a greeting must not cost a planning round-trip, nor leave a file",
    ),
    EvalCase(
        id="direct_identity",
        query="who are you, and how do you work?",
        expect_route="direct",
        expect_document=False,
    ),
    EvalCase(
        id="direct_capabilities_as_file",
        query="what can you do? put the answer in a markdown file",
        expect_route="direct",
        expect_document=True,
        expect_format="markdown",
        note=(
            "the direct route keeps a question about the assistant — it alone "
            "is shown the registry — and the file asked for is written; that "
            "reply is then a document, read apart from the turn (#80)"
        ),
    ),
    EvalCase(
        id="direct_thanks",
        query="thanks, that is all I needed for now",
        expect_route="direct",
        tiers=(LLM,),
        note="no keyword the fake router knows — only a real model can get this right",
    ),
    EvalCase(
        id="direct_trivial_fact",
        query="how many days are there in a leap year?",
        expect_route="direct",
        tiers=(LLM,),
        note="answerable inline; planning a DAG for it is over-triage",
    ),

    # ---------------- triage: obviously complex ----------------
    EvalCase(
        id="plan_compare_and_recommend",
        query=(
            "compare hexagonal and layered architectures for a long-running "
            "agent, then recommend one for a team of three and justify it"
        ),
        expect_route="plan",
        min_steps=2,
        expect_document=False,
        note=(
            "two distinct asks (compare, then recommend) should decompose — "
            "and nothing asks for a file: however much it plans, a silent "
            "request gets its answer and no document (#96)"
        ),
    ),
    EvalCase(
        id="plan_research_and_critique",
        query=(
            "research how teams evaluate LLM prompt changes today and write a "
            "critical review of the approaches you find"
        ),
        expect_route="plan",
        min_steps=2,
    ),
    EvalCase(
        id="plan_tradeoff_analysis",
        query=(
            "analyse the trade-offs between running background work in-process "
            "and in a dedicated worker, then challenge your own conclusion"
        ),
        expect_route="plan",
        min_steps=2,
        must_include=("analysis",),
        expect_document=False,
        note="an explicit 'analyse' should reach the analysis capability when it exists",
    ),
    EvalCase(
        id="plan_report_request",
        query=(
            "produce a report on the risks and the benefits of putting a "
            "language model in a customer-facing workflow"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=True,
        must_exclude=("read_files",),
        note=(
            "no file was named, so the step that reads one must be dropped as "
            "inapplicable — true of the fakes too, which chain the whole "
            "registry and let plan validation do the dropping. It asks for "
            "a report and names no format: since #96 that is what earns a "
            "file in the deployment's default format, where silence earns none"
        ),
    ),
    EvalCase(
        id="plan_survey_with_failure_modes",
        query=(
            "summarise the main approaches to retrieval-augmented generation "
            "and where each of them fails in practice"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=False,
        note="a summary is an answer, not a file (#96)",
    ),
    EvalCase(
        id="plan_document_requested_without_format",
        query=(
            "research how teams version their database schemas, compare the "
            "approaches, and put the result in a document I can keep"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=True,
        note=(
            "a document asked for in words, with no format named (#96): the "
            "one request `$JOBSMITH_REPORT_FORMAT` still answers. Since a "
            "silent request gets no file, it is also one of the few runs the "
            "checks that read the FILE can be scored on"
        ),
    ),

    # ---------------- the shape of the deliverable ----------------
    EvalCase(
        id="plan_summary_saved_as_a_file",
        query=(
            "research how teams choose between optimistic and pessimistic "
            "locking, summarise the trade-offs, and save that as a file"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=True,
        note=(
            "a file asked for in so many words, no format named, next to a "
            "'summarise' that on its own asks for an answer (#125): the "
            "document step read the pair as 'unspecified' and the run left "
            "no file, with nothing on the record to say one was asked for"
        ),
    ),
    EvalCase(
        id="trivial_fact_saved_to_a_file",
        query="how many days are there in a leap year? save the answer to a file",
        expect_document=True,
        tiers=(LLM,),
        note=(
            "the issue's own request (#125): a fact the router may answer "
            "directly, asked for as a file. No route is claimed — the "
            "document step runs before triage, so the file is owed on either "
            "route. Once, with no file written, the reply told the user to "
            "run `echo 366 > leap_year_days.txt` themselves (#80)"
        ),
    ),
    EvalCase(
        id="plan_file_is_the_subject",
        query=(
            "compare how ext4 and btrfs store a file on disk, and recommend "
            "one for a developer laptop"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=False,
        note=(
            "the mirror of the two above (#125): a file the request is ABOUT "
            "is not a file it asked to be left behind, so the rule that reads "
            "'save it to a file' must not fire on the word alone (#96)"
        ),
    ),
    EvalCase(
        id="plan_html_page_requested",
        query=(
            "compare two ways of scheduling recurring background work, "
            "recommend one, and give me the result as an html page I can "
            "open in a browser"
        ),
        expect_route="plan",
        min_steps=1,
        expect_document=True,
        expect_format="html",
        note=(
            "the request names its format in words and nothing else reads it "
            "(#90): the chat model fills that argument, so `jobsmith run`, "
            "`/bg` and `POST /jobs` all delivered markdown to someone who "
            "asked for something else. html rather than pdf on purpose — pdf "
            "needs pango where the run happens, and a golden case must not "
            "score a deployment"
        ),
    ),
    EvalCase(
        id="plan_printable_one_pager",
        query=(
            "research how teams roll out feature flags safely, compare the "
            "approaches, and give me the result as a one-page printable "
            "summary, ready to print as a PDF"
        ),
        expect_route="plan",
        min_steps=1,
        must_exclude=("slide_deck",),
        expect_format="pdf",
        tiers=(LLM,),
        note=(
            "a document to read is not a presentation: this request was "
            "planned as a deck and delivered a PowerPoint (#61). Only a real "
            "model can be measured on it — the keyword fake chains whatever "
            "is registered and would fail it for a reason that is not the "
            "capability's description. Compound on purpose: a decision the "
            "planner makes can only be measured on a request that reaches it"
        ),
    ),

    EvalCase(
        id="plan_verifiable_comparison",
        query=(
            "compare the main open-source vector databases on their published "
            "limits — index size, query latency, memory footprint — and say "
            "which one fits a single-node deployment"
        ),
        expect_route="plan",
        expect_terminal="answer",
        min_steps=1,
        tiers=(LLM,),
        note=(
            "a request that asks for figures is the one that tips a generator "
            "into refusing (#73): the notes come back hedged, and the model "
            "reads 'I cannot verify' as 'the material says nothing'. The run "
            "that opened the issue had 14k characters of sourced "
            "specifications and delivered a data-collection plan. Only a real "
            "model can be measured on it — the keyword fake has no notion of "
            "verification, and would reach the same terminal either way"
        ),
    ),

    # ---------------- the material has to reach the reasoning ----------------
    EvalCase(
        id="plan_named_file_grounds_the_steps",
        query=(
            "read the note I named, list the deployment constraints it "
            "records, and say which one binds first"
        ),
        expect_route="plan",
        min_steps=2,
        must_include=("read_files",),
        expect_document=False,
        inputs={"source_files": [FIXTURE_REF]},
        note=(
            "the request names a file, so the run has real material — and "
            "`grounding_reaches_reasoning` asks whether the steps that ran "
            "after the retrieval contain any of it (#81). Compound on "
            "purpose: the property only exists where something is planned "
            "downstream of the retrieval"
        ),
    ),

    # ---------------- a request the material cannot answer ----------------
    EvalCase(
        id="unanswerable_missing_material",
        query=(
            "summarise the attached quarterly report and list the three risks "
            "it names"
        ),
        expect_route="plan",
        expect_terminal="unanswered",
        min_steps=1,
        note=(
            "nothing was attached: the run must declare that it could not "
            "answer, rather than write a speculative report that reads like "
            "one (#59). It asks for no file, so the refusal is scored on the "
            "answer alone (`refusal_is_bare`)"
        ),
    ),
    EvalCase(
        id="unanswerable_as_a_document",
        query=(
            "write a report on the attached survey results and list the three "
            "findings it names"
        ),
        expect_route="plan",
        expect_terminal="unanswered",
        min_steps=1,
        expect_document=True,
        note=(
            "the same refusal, asked for as a document: the file must say the "
            "run could not answer (`refusal_declared`). Since #96 the case "
            "above writes no file, so without this one the declaration in the "
            "deliverable would be scored on nothing at all"
        ),
    ),

    # ---------------- guards ----------------
    EvalCase(
        id="guard_blank_query",
        query="   ",
        expect_route=None,
        expect_terminal="user_error",
        note="input validation must reject before any model call — no route, no plan",
    ),
)


def cases_for(tier: str, *, only: tuple[str, ...] = ()) -> list[EvalCase]:
    """The cases meaningful in `tier`, optionally filtered by id."""
    selected = [c for c in GOLDEN_CASES if tier in c.tiers]
    if only:
        wanted = set(only)
        unknown = wanted - {c.id for c in GOLDEN_CASES}
        if unknown:
            raise KeyError(f"unknown case id(s): {', '.join(sorted(unknown))}")
        selected = [c for c in selected if c.id in wanted]
    return selected
