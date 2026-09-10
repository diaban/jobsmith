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

NO_ANSWER_INSTRUCTION = (
    "- If the provided material does not let you answer, do NOT improvise or "
    f"speculate: make the FIRST line of your reply exactly `{NO_ANSWER_MARKER} "
    "<one short sentence naming what is missing>`, then explain below it what "
    "would be needed. Use that line only in that case, and nowhere else."
)

# The audience clause is not decoration (#58): the material a generator is
# handed was written for the run — notes, findings, a review of the work — and
# a prompt that only says "answer" lets that shape through to a reader who was
# never in the room. Neutral enough to stay a core default: every agent's
# deliverable is read by whoever asked for it.
DEFAULT_GENERATOR_PROMPT = (
    "You are writing a document for the person who asked for it: they were not "
    "part of the work that produced it, and they want the subject rather than "
    "a report on the work. Answer the user's query using ONLY the provided "
    "context. If the context is insufficient, say so explicitly. Be concise "
    "and precise. Address that reader and never the producer: no next steps, "
    "no open questions, no options to choose between, no requests for input.\n"
    + NO_ANSWER_INSTRUCTION
)

DEFAULT_REFINER_TEMPLATE = (
    "You previously produced an answer that failed validation.\n"
    "Validation issues: {issues}\n"
    "Re-write the answer fixing these issues. Keep using only the provided "
    "context."
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
