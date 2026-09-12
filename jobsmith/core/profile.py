"""AgentProfile: the domain-tunable surface of the framework.

Everything user-facing or prompt-shaped lives here — core node classes read
their prompts, messages, and validation rules from the profile instead of
hardcoding them. The defaults are neutral English; a domain ships its own
profile (see jobsmith/agents/).

Validation rules are plain callables:
- InputRule(state)  -> user-facing rejection message, or None if OK
- OutputRule(state) -> issue code, or None if OK
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .state import AgentState

InputRule = Callable[[AgentState], "str | None"]
OutputRule = Callable[[AgentState], "str | None"]


# ---------- Default prompts / messages ----------

# NOTE: wording matters for tests — FakeLLM/KeywordLLM script responses by
# system-prompt substring, so each prompt keeps a distinctive marker
# ("triage" here, "planner" below, "ONLY the provided" in the generator, ...).
DEFAULT_ROUTER_TEMPLATE = """You are the triage step of an assistant agent.
Read the user's message and choose exactly one route.

Routes:
{routes}

For reference, the capabilities the "plan" route can orchestrate:
{capabilities}

Return ONLY a JSON object, no prose, no markdown fences:
{{"route": "<route name>", "rationale": "<short explanation>"}}"""

DEFAULT_PLANNER_TEMPLATE = """You are the planner of an assistant agent.
Given a user's request, decide which of the available capabilities are needed
and in what order. Output a JSON object describing a DAG.

Available capabilities:
{capabilities}

Schema:
{{
  "steps": [
    {{"capability": "<name>", "depends_on": [<other capability names>]}}
  ],
  "rationale": "<short explanation>"
}}

Rules:
- Include only capabilities that are actually needed.
- depends_on values must refer to other steps in the same plan.
- The DAG must be acyclic.
- The request may be preceded by an excerpt of the conversation it came from.
  Use it only to resolve what the request refers to; plan for the request.
- Return ONLY the JSON object, no prose, no markdown fences."""

# ---------- The generator's structural declaration ----------

# A generator held to "use ONLY the provided context" must sometimes answer
# that the context does not let it answer. Saying that in prose is right for
# the reader and useless to the graph: the run then ends exactly like one that
# answered (#59). So the same decision is asked for as DATA — one marker line,
# emitted only in that case — which `core/generation.py` reads and turns into
# a terminal of its own, the way the router and the planner return decisions
# rather than sentences.
#
# It is FAIL-OPEN by construction: the marker is asked for only on the refusal
# path, so a model that never emits it produces exactly today's run. Nothing
# greps the prose — an answer that merely *reads* like a refusal, in whatever
# language it was written in, is still an answer.
NO_ANSWER_MARKER = "NO_ANSWER:"

# The BAR and the SHAPE are both stated here (#73), and neither was before.
#
# The bar: a run that had 14k characters of sourced specifications declared it
# could not answer, because the notes it was handed opened by saying their
# figures were unverified. "Insufficient" had been left to the model, which
# read it as "unverified" — and against a pack whose research step flags its
# own uncertainty, that reading refuses every request that asks for rigour.
# Unverified material is material: it is answered, with the doubt marked on
# the statement it bears on.
#
# The shape: the prompts that carry this line forbid exactly the register a
# refusal needs (nothing on the state of the work, no templates, no requests
# for input) while asking for a refusal. Nothing arbitrated, so the model
# produced the maximal version of the forbidden thing — a data-collection plan
# and a blank template, closing with an offer to prepare it. So the refusal is
# specified as a shape, and this line says outright that it is the one place
# those rules are lifted.
NO_ANSWER_INSTRUCTION = (
    "- The bar for saying you cannot answer is that the material says NOTHING "
    "about the subject. Material that is partial, indicative, second-hand or "
    "unverified IS material: answer with it, and mark the doubt on the "
    "statement it bears on. A qualified answer is an answer.\n"
    f"- Only when that bar is met: make the FIRST line of your reply exactly "
    f"`{NO_ANSWER_MARKER} <one short sentence naming what is missing>`, and "
    "below it, in a few short paragraphs at most, say what was asked, what "
    "the material did and did not support, and whatever is known anyway. "
    "Nothing else — no plan for gathering what is missing, no template or "
    "blank fields, no offer of further work. On that path, and only there, "
    "that shape replaces the rules above about what the document must "
    "contain; everywhere else they hold and this line must not appear."
)

# The audience clause is not decoration (#58): the material a generator is
# handed was written for the run — notes, findings, a review of the work — and
# a prompt that only says "answer" lets that shape through to a reader who was
# never in the room. Neutral enough to stay a core default: every agent's
# deliverable is read by whoever asked for it.
#
# The OBLIGATION is the other half, and it was missing (#73). #58 gave the
# document a reader and a list of banned registers; nothing said what it must
# CONTAIN. A model handed prohibitions and no substance fills the gap with
# what is at hand — the working material — so the first thing this prompt now
# states is the answer, in the subject's own terms, from the material.
DEFAULT_GENERATOR_PROMPT = (
    "You are writing a document for the person who asked for it: they were not "
    "part of the work that produced it, and they want the subject rather than "
    "a report on the work.\n"
    "What it must contain: the answer to the query, first, in the subject's "
    "own terms. Use ONLY the provided context, and use what it holds — state "
    "what it establishes rather than what would have to be checked. A "
    "document describing what would be needed in order to answer has not "
    "answered.\n"
    "Where the material is partial or unverified, mark the doubt on the "
    "statement it bears on ('reported as X, unconfirmed'), never as a preamble "
    "that disqualifies everything below it.\n"
    "Be concise and precise. Address that reader and never the producer: no "
    "next steps, no open questions, no options to choose between, no requests "
    "for input, no templates or blank fields to fill in.\n"
    "The query may also say what document is wanted — a name, a length, a "
    "format, a language. Carry it out; never restate it in the prose.\n"
    + NO_ANSWER_INSTRUCTION
)

DEFAULT_REFINER_TEMPLATE = (
    "You previously produced an answer that failed validation.\n"
    "Validation issues: {issues}\n"
    "Produce a corrected version fixing these issues. Keep using only the "
    "provided context, and keep what a deliverable must contain: the answer "
    "to the query, first, in the subject's own terms, from that context."
)

DEFAULT_DIRECT_ANSWER_TEMPLATE = (
    "You are an assistant. Answer the user's message directly, concisely and "
    "helpfully — it needs no external context. If asked what you can do, "
    "describe the capabilities below in plain language:\n{capabilities}"
)

DEFAULT_USER_ERROR_MESSAGE = "An error occurred while processing your request."
DEFAULT_ESCALATION_MESSAGE = "Your request has been forwarded for review."
DEFAULT_EMPTY_QUERY_MESSAGE = "Your query is empty."
DEFAULT_QUERY_TOO_LONG_MESSAGE = "Query too long (max {max_len} characters)."


# ---------- Default validation rules ----------

def rule_nonempty_query(*, message: str = DEFAULT_EMPTY_QUERY_MESSAGE) -> InputRule:
    def rule(state: AgentState) -> str | None:
        return message if not (state.get("query") or "").strip() else None
    return rule


def rule_max_query_len(max_len: int = 4000, *, message: str | None = None) -> InputRule:
    msg = message or DEFAULT_QUERY_TOO_LONG_MESSAGE.format(max_len=max_len)

    def rule(state: AgentState) -> str | None:
        return msg if len((state.get("query") or "").strip()) > max_len else None
    return rule


def rule_nonempty_answer(state: AgentState) -> str | None:
    return "empty_answer" if not (state.get("draft_answer") or "") else None


def rule_min_answer_len(min_len: int = 20) -> OutputRule:
    def rule(state: AgentState) -> str | None:
        return "answer_too_short" if len(state.get("draft_answer") or "") < min_len else None
    return rule


# ---------- Profile ----------

@dataclass(frozen=True)
class AgentProfile:
    router_prompt_template: str = DEFAULT_ROUTER_TEMPLATE
    planner_prompt_template: str = DEFAULT_PLANNER_TEMPLATE
    direct_answer_prompt_template: str = DEFAULT_DIRECT_ANSWER_TEMPLATE
    generator_system_prompt: str = DEFAULT_GENERATOR_PROMPT
    refiner_prompt_template: str = DEFAULT_REFINER_TEMPLATE
    user_error_message: str = DEFAULT_USER_ERROR_MESSAGE
    escalation_message: str = DEFAULT_ESCALATION_MESSAGE
    context_empty_message: str = "(no context available)"
    input_rules: tuple[InputRule, ...] = field(
        default_factory=lambda: (rule_nonempty_query(), rule_max_query_len())
    )
    output_rules: tuple[OutputRule, ...] = field(
        default_factory=lambda: (rule_nonempty_answer, rule_min_answer_len())
    )
    max_refine: int = 2
    generation_temperature: float = 0.2
