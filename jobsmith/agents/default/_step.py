"""Shared base for single-LLM-step capabilities that build on upstream results.

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
            if result and result.get("ok") and result.get("data", {}).get(data_key):
                return f"[material from {cap_name}]\n{result['data'][data_key]}"
        return "(no upstream material available — reason from the request alone)"

    async def work(self, state: StepState) -> dict:
        try:
            output = await self.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM},
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
        return self._emit_success({self.OUTPUT_KEY: state["output"]})

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
