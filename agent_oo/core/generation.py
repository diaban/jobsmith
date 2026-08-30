"""Generation pipeline.

Four classes:
- ContextMerger: deterministic node — asks each capability to render its own
  result (render_context), iterating in PLAN order for determinism
- Generator:     LLM call to produce the draft answer
- Refiner:       LLM call to fix a rejected draft
- PostProcessor: persists to the long-term store
"""
from __future__ import annotations

from typing import Any

from .deps import Deps
from .registry import CapabilityRegistry
from .state import AgentState, NodeError


class ContextMerger:
    def __init__(self, registry: CapabilityRegistry, *, empty_message: str = "(no context available)"):
        self.registry = registry
        self.empty_message = empty_message

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
    SYSTEM_PROMPT = (
        "You are a banking assistant. Answer the banker's query using ONLY the "
        "provided context. Cite sources inline as [doc_id] when relevant. "
        "If the context is insufficient, say so explicitly. Be concise and precise."
    )

    def __init__(self, deps: Deps, *, temperature: float = 0.2):
        self.deps = deps
        self.temperature = temperature

    async def run(self, state: AgentState) -> dict:
        try:
            answer = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
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


class Refiner:
    SYSTEM_PROMPT_TEMPLATE = (
        "You previously produced an answer that failed validation.\n"
        "Validation issues: {issues}\n"
        "Re-write the answer fixing these issues. Keep using only the provided "
        "context and inline [doc_id] citations."
    )

    def __init__(self, deps: Deps, *, temperature: float = 0.2):
        self.deps = deps
        self.temperature = temperature

    async def run(self, state: AgentState) -> dict:
        issues = ", ".join(state.get("validation_issues", []))
        try:
            answer = await self.deps.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT_TEMPLATE.format(issues=issues),
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
    def __init__(self, store: Any):
        self.store = store

    async def run(self, state: AgentState) -> dict:
        final = state["draft_answer"]
        await self.store.aput(
            ("answers", state["job_id"]),
            state["job_id"],
            {
                "query": state["query"],
                "answer": final,
                "plan": state.get("plan"),
            },
        )
        return {"final_answer": final, "terminal_kind": "answer"}
