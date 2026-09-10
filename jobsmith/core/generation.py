"""Generation pipeline.

Six classes:
- ContextMerger:   deterministic node — asks each capability to render its own
  result (render_context), iterating in PLAN order for determinism
- Generator:       LLM call to produce the draft answer
- DirectResponder: the router's "direct" route — answers without capabilities
- Refiner:         LLM call to fix a rejected draft
- PostProcessor:   marks the terminal answer (persistence lives in the job layer)
- UnansweredEmitter: marks the terminal of a run that declared it could not
  answer (the other half of the same decision)
"""
from __future__ import annotations

from .deps import Deps
from .profile import NO_ANSWER_MARKER, AgentProfile
from .registry import CapabilityRegistry
from .state import TERMINAL_UNANSWERED, AgentState, NodeError

# What DirectResponder renders where the capability list would go when the
# registry is empty. Not a profile message: nothing here is shown to the human,
# and "the registry is empty" is a fact about the composition, not the domain.
# Without it the prompt would say "describe the capabilities below" and then
# show nothing, which reads as an invitation to invent some.
NO_CAPABILITIES_TEXT = "- (none — this assistant has no capabilities registered)"


class ContextMerger:
    def __init__(self, registry: CapabilityRegistry, profile: AgentProfile):
        self.registry = registry
        self.empty_message = profile.context_empty_message

    async def run(self, state: AgentState) -> dict:
        plan = state.get("plan")
        results = state.get("results", {})
        parts: list[str] = []
        # Iterate in plan order, NOT results-dict order (see state.py determinism caveat)
        for step in (plan["steps"] if plan else []):
            name = step["capability"]
            result = results.get(name)
            if not result or not result.get("ok"):
                continue
            text = self.registry.get(name).render_context(result)
            if text:
                parts.append(text)
        return {"merged_context": "\n\n".join(parts) if parts else self.empty_message}


# Characters a model wraps a line in when it cannot resist formatting one:
# `**NO_ANSWER: ...**`, `> NO_ANSWER: ...`, `# NO_ANSWER: ...`. Stripped before
# the marker is looked for, and after the reason is cut off it.
_DECORATION = " \t`*_#>-"


def split_declaration(reply: str) -> tuple[str, bool]:
    """Split the generator's structural declaration off its prose.

    Returns `(text, answered)`. The generator is asked for one marker line —
    and asked for it ONLY when it cannot answer (`NO_ANSWER_INSTRUCTION`), so
    the absence of a declaration is the overwhelmingly common case and means
    exactly what it meant before this existed: the reply is the answer.

    This is a protocol, not a heuristic: only the FIRST line is looked at, and
    only for the marker the prompt asked for. An answer that merely reads like
    a refusal ("the sources disagree, so no figure can be given") is still an
    answer — searching prose for regret is what this exists instead of.

    What follows the marker on that line is the model's one-line reason. It is
    folded back into the text rather than carried as a field of its own: the
    explanation below it is the same thing said at length, and the reason is
    all there is to keep when the model wrote nothing else.
    """
    head, _, rest = reply.lstrip().partition("\n")
    declaration = head.strip().strip(_DECORATION)
    if not declaration.upper().startswith(NO_ANSWER_MARKER):
        return reply, True
    reason = declaration[len(NO_ANSWER_MARKER):].strip().strip(_DECORATION).strip()
    return rest.strip() or reason, False


class Generator:
    def __init__(self, deps: Deps, profile: AgentProfile):
        self.deps = deps
        self.system_prompt = profile.generator_system_prompt
        self.temperature = profile.generation_temperature

    async def run(self, state: AgentState) -> dict:
        try:
            answer = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {state['query']}\n\n"
                            f"Context:\n{state.get('merged_context', '')}"
                        ),
                    },
                ],
                temperature=self.temperature,
            )
            # Both keys, always: `answered` is re-decided on every generation,
            # so a refusal that a later attempt fixed cannot outlive the draft
            # it was about (the refine cycle re-enters this node).
            text, answered = split_declaration(answer)
            return {"draft_answer": text, "answered": answered}
        except Exception as e:
            err: NodeError = {
                "source": "generation",
                "kind": "generation_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}


class DirectResponder:
    """Answers the user's message with no capability run (router route "direct").

    The registry is rendered into the system prompt so the model can describe
    what the agent is able to do ("what can you do?"). It also sets
    `merged_context`, so the shared refine cycle has material if the draft
    fails output validation.

    It is reached by two paths: the router's "direct" triage, and — since the
    router routes an empty registry here structurally — a plan that came back
    with nothing to run. So an EMPTY registry is a supported case, not an
    accident: the capability list degrades to `NO_CAPABILITIES_TEXT` rather
    than to a blank section.
    """

    def __init__(self, deps: Deps, registry: CapabilityRegistry, profile: AgentProfile):
        self.deps = deps
        self.registry = registry
        self.prompt_template = profile.direct_answer_prompt_template
        self.temperature = profile.generation_temperature

    def _capabilities_text(self) -> str:
        lines = [f"- {spec.name}: {spec.description}" for spec in self.registry.specs()]
        return "\n".join(lines) if lines else NO_CAPABILITIES_TEXT

    def system_prompt(self) -> str:
        return self.prompt_template.format(capabilities=self._capabilities_text())

    async def run(self, state: AgentState) -> dict:
        try:
            answer = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": state["query"]},
                ],
                temperature=self.temperature,
            )
            return {
                "draft_answer": answer,
                "merged_context": f"Assistant capabilities:\n{self._capabilities_text()}",
            }
        except Exception as e:
            err: NodeError = {
                "source": "direct_answer",
                "kind": "direct_answer_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}


class Refiner:
    def __init__(self, deps: Deps, profile: AgentProfile):
        self.deps = deps
        self.prompt_template = profile.refiner_prompt_template
        self.temperature = profile.generation_temperature

    async def run(self, state: AgentState) -> dict:
        issues = ", ".join(state.get("validation_issues", []))
        try:
            answer = await self.deps.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self.prompt_template.format(issues=issues),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original query: {state['query']}\n\n"
                            f"Previous draft:\n{state.get('draft_answer', '')}\n\n"
                            f"Context:\n{state.get('merged_context', '')}"
                        ),
                    },
                ],
                temperature=self.temperature,
            )
            return {
                "draft_answer": answer,
                "refine_count": state.get("refine_count", 0) + 1,
            }
        except Exception as e:
            err: NodeError = {
                "source": "refine",
                "kind": "refine_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}


class PostProcessor:
    """Marks the terminal answer. Persistence is the job layer's concern —
    JobManager observes this node's update via astream and stores the answer."""

    async def run(self, state: AgentState) -> dict:
        # `draft_answer` is guaranteed by graph order, not by the schema: this
        # node is reachable only through validate_output's passing branch, and
        # the default output rules reject an empty draft. A profile that drops
        # those rules is the only way here without one — `None` then flows on
        # as the "no answer" the job layer already models (Job.final_answer is
        # `str | None`, and the reporter renders it as "(no answer)").
        return {"final_answer": state.get("draft_answer"), "terminal_kind": "answer"}


class UnansweredEmitter:
    """Terminal of a run whose generator declared it could not answer.

    A twin of `PostProcessor`, and deliberately not a branch inside it: which
    terminal a run reaches is a control-flow decision, so it lives in the
    builder's path map (`_route_validate_output`) where the empty plan and the
    refine cycle already live, and each terminal node states one outcome.

    It emits the text anyway. The draft is not a failure to be discarded — it
    is the run explaining what it would have needed, which is the most useful
    thing a job that could not answer can hand back. `terminal_kind` is what
    says how to read it, so no reader has to infer the outcome from the prose.
    """

    async def run(self, state: AgentState) -> dict:
        # Same read as PostProcessor's: `draft_answer` is guaranteed by graph
        # order (generation ran), not by the schema.
        return {
            "final_answer": state.get("draft_answer"),
            "terminal_kind": TERMINAL_UNANSWERED,
        }
