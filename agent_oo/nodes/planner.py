"""Planner step — object-oriented version.

The class owns:
- the system prompt
- plan validation (cycle detection, allowed subgraphs)
- the LLM call

It exposes `run` (the node coroutine) for the parent graph to register.
"""
from __future__ import annotations

import json
from typing import Any

from ..deps import Deps
from ..state import AgentState, NodeError, Plan, SubgraphName


class Planner:

    SYSTEM_PROMPT = """You are the planner of a banking-assistant agent.
Given a banker's query, decide which of these sub-graphs are needed and in what
order. Output a JSON object describing a DAG.

Available sub-graphs:
- "search": RAG over internal knowledge base
- "vision": analyse an attached image (only if images are provided)
- "refs":   retrieve past slides / reference decks

Schema:
{
  "steps": [
    {"subgraph": "search" | "vision" | "refs", "depends_on": [<other subgraph names>]}
  ],
  "rationale": "<short explanation>"
}

Rules:
- Include only sub-graphs that are actually needed.
- depends_on values must refer to other steps in the same plan.
- The DAG must be acyclic.
- Return ONLY the JSON object, no prose, no markdown fences.
"""

    _ALLOWED = {s.value for s in SubgraphName}

    def __init__(self, deps: Deps):
        self.deps = deps

    # -------- Validation --------

    def _validate_plan(self, raw: dict[str, Any], has_image: bool) -> Plan:
        if not isinstance(raw, dict) or "steps" not in raw:
            raise ValueError("plan missing 'steps'")
        steps = raw["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("plan 'steps' must be a non-empty list")

        seen: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for step in steps:
            sg = step.get("subgraph")
            if sg not in self._ALLOWED:
                raise ValueError(f"unknown subgraph: {sg}")
            if sg in seen:
                raise ValueError(f"duplicate subgraph: {sg}")
            if sg == SubgraphName.VISION.value and not has_image:
                continue
            deps = step.get("depends_on") or []
            if not isinstance(deps, list) or any(d not in self._ALLOWED for d in deps):
                raise ValueError(f"bad depends_on for {sg}")
            seen.add(sg)
            cleaned.append({"subgraph": sg, "depends_on": list(deps)})

        # Kahn's algo for cycle detection
        indeg = {s["subgraph"]: len(s["depends_on"]) for s in cleaned}
        adj: dict[str, list[str]] = {s["subgraph"]: [] for s in cleaned}
        for s in cleaned:
            for d in s["depends_on"]:
                if d not in adj:
                    raise ValueError(f"depends_on references unknown step: {d}")
                adj[d].append(s["subgraph"])
        queue = [n for n, d in indeg.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        if visited != len(cleaned):
            raise ValueError("plan contains a cycle")
        if not cleaned:
            raise ValueError("plan is empty after validation")

        return Plan(steps=cleaned, rationale=str(raw.get("rationale", "")))

    # -------- Node --------

    async def run(self, state: AgentState) -> dict:
        has_image = bool(state.get("image_s3_keys"))
        try:
            raw_response = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(raw_response)
            plan = self._validate_plan(parsed, has_image=has_image)
            return {"plan": plan}
        except (json.JSONDecodeError, ValueError) as e:
            err: NodeError = {
                "subgraph": "planner",
                "kind": "planner_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}
