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
    #: "answer" | "user_error" | "escalated"
    expect_terminal: str = "answer"
    #: bounds on the planned DAG — only checked when set
    min_steps: int = 0
    max_steps: int | None = None
    #: capabilities the plan should contain, IF the composed agent has them
    #: (a registry is agent- and configuration-dependent, so absent names are
    #: skipped rather than failed)
    must_include: tuple[str, ...] = ()
    tiers: tuple[str, ...] = BOTH
    inputs: dict = field(default_factory=dict)
    note: str = ""


GOLDEN_CASES: tuple[EvalCase, ...] = (
    # ---------------- triage: obviously direct ----------------
    EvalCase(
        id="direct_capabilities",
        query="what can you do?",
        expect_route="direct",
        note="a question about the assistant itself needs no capability at all",
    ),
    EvalCase(
        id="direct_greeting",
        query="hello there",
        expect_route="direct",
        note="a greeting must not cost a planning round-trip",
    ),
    EvalCase(
        id="direct_identity",
        query="who are you, and how do you work?",
        expect_route="direct",
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
        note="two distinct asks (compare, then recommend) should decompose",
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
    ),
    EvalCase(
        id="plan_survey_with_failure_modes",
        query=(
            "summarise the main approaches to retrieval-augmented generation "
            "and where each of them fails in practice"
        ),
        expect_route="plan",
        min_steps=1,
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
