"""RESEARCH capability (LLM-only).

Internal shape: decompose the request into aspects → produce structured
notes per aspect → emit. No external source: the model's own knowledge,
with uncertainty flagged in the notes.
"""
from __future__ import annotations

import json
from typing import Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.deps import LLMClient
from ...core.state import CapabilityResult
from ._step import SUBJECT_ONLY_RULE


class ResearchState(CapabilityBaseState, total=False):
    aspects: list[str]
    notes: str


class ResearchCapability(Capability):
    """Break the request into key aspects and write research notes."""

    spec = CapabilitySpec(
        name="research",
        description=(
            "break the request into its key aspects and produce structured "
            "research notes from the model's own knowledge (no external sources)"
        ),
        output_schema={
            "type": "object",
            "properties": {
                "aspects": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string"},
            },
        },
    )

    DECOMPOSE_SYSTEM = (
        "Identify the key aspects to investigate to fulfil the user's request. "
        'Return JSON: {"aspects": ["<short aspect>", ...]} with 2 to 5 entries. '
        "No prose, no markdown."
    )
    #: "Flag any uncertainty explicitly" is where a wholesale hedge is born
    #: (#73). Asked for it without being told *where* to put it, the model
    #: opened its notes with "these values are indicative and must all be
    #: verified" and marked every figure `[to confirm]` — including the ones
    #: it was sure of. Downstream, that reads as material nobody can use: the
    #: generator of the run this came from declared it could not answer over
    #: 14k characters of sourced specifications. So the instruction now says
    #: where the mark goes and what must not be marked.
    NOTES_SYSTEM = (
        "Write structured research notes for the request: one short markdown "
        "section per listed aspect, from your own knowledge. Be factual, and "
        "state what you know as what you know. Where a specific point is "
        "uncertain, mark the doubt on that point ('reported as X, "
        "unconfirmed'). Do not open with a general warning that the notes are "
        "unverified, and do not mark what you are confident about: a blanket "
        "hedge makes usable notes unusable."
    )

    def __init__(self, llm: LLMClient, *, max_aspects: int = 5):
        self.llm = llm
        self.max_aspects = max_aspects

    # -------------------- Nodes --------------------

    async def decompose(self, state: ResearchState) -> dict:
        try:
            raw = await self.llm.chat(
                messages=[
                    {"role": "system",
                     "content": self.DECOMPOSE_SYSTEM + SUBJECT_ONLY_RULE},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            aspects = [str(a) for a in json.loads(raw).get("aspects", []) if str(a).strip()]
        except Exception:
            aspects = []
        # lenient by design: an unparseable reply degrades to one broad aspect
        return {"aspects": aspects[: self.max_aspects] or [state["query"]]}

    async def investigate(self, state: ResearchState) -> dict:
        # `aspects` is written by decompose, which always returns a non-empty
        # list — the fallback here is the same one it uses, so an aspect list
        # that somehow never arrived degrades to the request itself.
        aspects = state.get("aspects") or [state["query"]]
        try:
            notes = await self.llm.chat(
                messages=[
                    {"role": "system",
                     "content": self.NOTES_SYSTEM + SUBJECT_ONLY_RULE},
                    {
                        "role": "user",
                        "content": (
                            f"Request: {state['query']}\n\n"
                            "Aspects:\n" + "\n".join(f"- {a}" for a in aspects)
                        ),
                    },
                ],
                temperature=0.2,
            )
        except Exception:
            notes = ""
        return {"notes": notes}

    async def emit_success(self, state: ResearchState) -> dict:
        # Reached only through route_after_notes == "success", i.e. with
        # non-empty notes; decompose has likewise already written the aspects.
        aspects = state.get("aspects") or []
        return self._emit_success(
            {"aspects": aspects, "notes": state.get("notes") or ""},
            meta={"aspect_count": len(aspects)},
        )

    async def emit_failure(self, state: ResearchState) -> dict:
        return self._emit_failure("research produced no notes")

    # -------------------- Router --------------------

    def route_after_notes(self, state: ResearchState) -> Literal["success", "failure"]:
        return "success" if (state.get("notes") or "").strip() else "failure"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        notes = result.get("data", {}).get("notes")
        return f"# Research notes\n\n{notes}" if notes else None

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(ResearchState)
        g.add_node("decompose", self.decompose)
        g.add_node("investigate", self.investigate)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)

        g.set_entry_point("decompose")
        g.add_edge("decompose", "investigate")
        g.add_conditional_edges("investigate", self.route_after_notes, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
