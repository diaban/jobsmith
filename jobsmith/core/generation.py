"""Generation pipeline.

Five classes:
- ContextMerger:   deterministic node — asks each capability to render its own
  result (render_context), iterating in PLAN order for determinism
- Generator:       LLM call to produce the draft answer
- DirectResponder: the router's "direct" route — answers without capabilities
- Refiner:         LLM call to fix a rejected draft
- PostProcessor:   marks the terminal answer (persistence lives in the job layer)
"""
from __future__ import annotations

from .deps import Deps
from .profile import AgentProfile
from .registry import CapabilityRegistry
from .state import AgentState, NodeError


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
            return {"draft_answer": answer}
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
    """

    def __init__(self, deps: Deps, registry: CapabilityRegistry, profile: AgentProfile):
        self.deps = deps
        self.registry = registry
        self.prompt_template = profile.direct_answer_prompt_template
        self.temperature = profile.generation_temperature

    def _capabilities_text(self) -> str:
        return "\n".join(
            f"- {spec.name}: {spec.description}" for spec in self.registry.specs()
        )

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
        return {"final_answer": state["draft_answer"], "terminal_kind": "answer"}
