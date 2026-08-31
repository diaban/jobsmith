"""AgentProfile: the domain-tunable surface of the framework.

Everything user-facing or prompt-shaped lives here — core node classes read
their prompts, messages, and validation rules from the profile instead of
hardcoding them. The defaults are neutral English; a domain ships its own
profile (see the bundled example under agent_oo/examples/).

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
- Return ONLY the JSON object, no prose, no markdown fences."""

DEFAULT_GENERATOR_PROMPT = (
    "You are an assistant. Answer the user's query using ONLY the provided "
    "context. Cite sources inline as [doc_id] when relevant. If the context "
    "is insufficient, say so explicitly. Be concise and precise."
)

DEFAULT_REFINER_TEMPLATE = (
    "You previously produced an answer that failed validation.\n"
    "Validation issues: {issues}\n"
    "Re-write the answer fixing these issues. Keep using only the provided "
    "context and inline [doc_id] citations."
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
