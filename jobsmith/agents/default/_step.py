"""Shared base for single-LLM-step capabilities that build on upstream results.

It also carries the pack's one shared prompt fragment, `SUBJECT_ONLY_RULE`
(see below) — the steps that produce material and `research` all append it.

Reference pattern: a capability whose sub-graph is one LLM node reading the
best available upstream `results` entry (that's why it should be planned
`depends_on` its upstream — but it degrades to reasoning from the query
alone when the upstream failed or was not planned).
"""
from __future__ import annotations

from typing import ClassVar, Literal

from langgraph.constants import END

from ...core.capability import Capability, CapabilityBaseState, CapabilitySpec
from ...core.deps import LLMClient
from ...core.state import CapabilityResult

#: Appended to the system prompt of every step of this pack (#58).
#:
#: A request usually carries two things: a subject, and instructions about the
#: document to produce ("a one-page printable synthesis, with a deck for the
#: team"). A step handed both treats both as its material and analyses the
#: *task* — which is how a report about chairs grew a section of practical
#: advice for building the deck, and a proposed slide structure the model had
#: invented for it. What document is produced is the run's business: the plan
#: decided it, the generator and `slide_deck` are told who reads it. A step
#: producing material has only the subject to work on.
SUBJECT_ONLY_RULE = (
    "\nThe request may also say what document is wanted — a report, a deck, a "
    "page, a language. That is not part of the subject: work on the subject "
    "alone, and never design or advise on the document itself."
)


class StepState(CapabilityBaseState, total=False):
    output: str


class SingleStepCapability(Capability):
    spec: CapabilitySpec                               # declared by Capability, not a ClassVar
    SYSTEM: ClassVar[str]                              # the node's system prompt
    HEADING: ClassVar[str]                             # markdown heading in render_context
    OUTPUT_KEY: ClassVar[str]                          # key in the emitted data dict
    UPSTREAM: ClassVar[tuple[tuple[str, str], ...]]    # (capability, data key), priority order

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # -------------------- Nodes --------------------

    def _material(self, state: StepState) -> str:
        for cap_name, data_key in self.UPSTREAM:
            result = state.get("results", {}).get(cap_name)
            if not result or not result.get("ok"):
                continue
            # Every CapabilityResult key is NotRequired — a failed step has no
            # `data` at all — so bind the value once instead of asserting twice.
            material = (result.get("data") or {}).get(data_key)
            if material:
                return f"[material from {cap_name}]\n{material}"
        return "(no upstream material available — reason from the request alone)"

    async def work(self, state: StepState) -> dict:
        try:
            output = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM + SUBJECT_ONLY_RULE},
                    {
                        "role": "user",
                        "content": f"Request: {state['query']}\n\n{self._material(state)}",
                    },
                ],
                temperature=0.2,
            )
        except Exception:
            output = ""
        return {"output": output}

    async def emit_success(self, state: StepState) -> dict:
        # Reached only through route_after_work == "success", which gates on a
        # non-empty `output`.
        return self._emit_success({self.OUTPUT_KEY: state.get("output") or ""})

    async def emit_failure(self, state: StepState) -> dict:
        return self._emit_failure(f"{self.spec.name} produced no output")

    def route_after_work(self, state: StepState) -> Literal["success", "failure"]:
        return "success" if (state.get("output") or "").strip() else "failure"

    # -------------------- Context rendering --------------------

    def render_context(self, result: CapabilityResult) -> str | None:
        text = result.get("data", {}).get(self.OUTPUT_KEY)
        return f"# {self.HEADING}\n\n{text}" if text else None

    # -------------------- Compilation --------------------

    def build(self):
        g = self.state_graph(StepState)
        g.add_node("work", self.work)
        g.add_node("emit_success", self.emit_success)
        g.add_node("emit_failure", self.emit_failure)
        g.set_entry_point("work")
        g.add_conditional_edges("work", self.route_after_work, {
            "success": "emit_success",
            "failure": "emit_failure",
        })
        g.add_edge("emit_success", END)
        g.add_edge("emit_failure", END)
        return g.compile()
