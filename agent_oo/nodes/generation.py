"""Generation pipeline — object-oriented version.

Three classes:
- ContextMerger: deterministic node, no LLM
- Generator:     LLM call to produce the draft answer
- Refiner:       LLM call to fix a rejected draft
- PostProcessor: persists to the long-term store
"""
from __future__ import annotations

from typing import Any

from ..deps import Deps
from ..state import AgentState, NodeError


class ContextMerger:
    def __init__(self, deps: Deps):
        self.deps = deps

    @staticmethod
    def _format(state: AgentState) -> str:
        parts: list[str] = []
        sr = state.get("search_result")
        if sr:
            parts.append("# Search results")
            for i, d in enumerate(sr["docs"]):
                parts.append(f"[{d.get('id', f'doc_{i}')}] {d.get('text', '')}")
        vr = state.get("vision_result")
        if vr:
            parts.append("# Image analysis")
            parts.append(vr["description"])
        rr = state.get("refs_result")
        if rr:
            parts.append("# References")
            for i, r in enumerate(rr["refs"]):
                parts.append(f"[{r.get('id', f'ref_{i}')}] {r.get('summary', '')}")
        return "\n\n".join(parts) if parts else "(no context available)"

    async def run(self, state: AgentState) -> dict:
        return {"merged_context": self._format(state)}


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
                "subgraph": "generation",
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
                "subgraph": "refine",
                "kind": "refine_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}


class PostProcessor:
    def __init__(self, deps: Deps, store: Any):
        self.deps = deps
        self.store = store

    async def run(self, state: AgentState) -> dict:
        final = state["draft_answer"]
        await self.store.aput(
            namespace=("answers", state["thread_id"]),
            key=state["thread_id"],
            value={
                "query": state["query"],
                "answer": final,
                "plan": state.get("plan"),
            },
        )
        return {"final_answer": final, "terminal_kind": "answer"}
