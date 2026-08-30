"""Registry-driven planner.

The class owns:
- the prompt template (rendered from the registry's capability specs)
- plan validation (allowed names, applicability, cycle detection)
- the LLM call

It exposes `run` (the node coroutine) for the parent graph to register.
"""
from __future__ import annotations

import json
from typing import Any

from .deps import Deps
from .profile import DEFAULT_PLANNER_TEMPLATE
from .registry import CapabilityRegistry
from .state import AgentState, NodeError, Plan


class Planner:

    DEFAULT_TEMPLATE = DEFAULT_PLANNER_TEMPLATE

    def __init__(
        self,
        deps: Deps,
        registry: CapabilityRegistry,
        *,
        prompt_template: str | None = None,
    ):
        self.deps = deps
        self.registry = registry
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    # -------- Prompt rendering --------

    def _render_capabilities(self) -> str:
        lines: list[str] = []
        for spec in self.registry.specs():
            line = f'- "{spec.name}": {spec.description}'
            if spec.requires_inputs:
                line += f" (only if these inputs are provided: {', '.join(spec.requires_inputs)})"
            if spec.output_schema:
                line += f"\n  produces: {json.dumps(spec.output_schema)}"
            lines.append(line)
        return "\n".join(lines)

    def system_prompt(self) -> str:
        return self.prompt_template.format(capabilities=self._render_capabilities())

    # -------- Validation --------

    def _validate_plan(self, raw: dict[str, Any], state: AgentState) -> Plan:
        if not isinstance(raw, dict) or "steps" not in raw:
            raise ValueError("plan missing 'steps'")
        steps = raw["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("plan 'steps' must be a non-empty list")

        allowed = set(self.registry.names())
        seen: set[str] = set()
        dropped: set[str] = set()
        cleaned: list[dict[str, Any]] = []
        for step in steps:
            name = step.get("capability")
            if name not in allowed:
                raise ValueError(f"unknown capability: {name}")
            if name in seen or name in dropped:
                raise ValueError(f"duplicate capability: {name}")
            deps = step.get("depends_on") or []
            if not isinstance(deps, list) or any(d not in allowed for d in deps):
                raise ValueError(f"bad depends_on for {name}")
            if not self.registry.get(name).is_applicable(state):
                dropped.add(name)
                continue
            seen.add(name)
            cleaned.append({"capability": name, "depends_on": list(deps)})

        # Prune depends_on entries that reference dropped (inapplicable) steps;
        # references to steps absent from the plan altogether are still errors.
        surviving = {s["capability"] for s in cleaned}
        for s in cleaned:
            kept: list[str] = []
            for d in s["depends_on"]:
                if d in surviving:
                    kept.append(d)
                elif d not in dropped:
                    raise ValueError(f"depends_on references unknown step: {d}")
            s["depends_on"] = kept

        # Kahn's algo for cycle detection
        indeg = {s["capability"]: len(s["depends_on"]) for s in cleaned}
        adj: dict[str, list[str]] = {s["capability"]: [] for s in cleaned}
        for s in cleaned:
            for d in s["depends_on"]:
                adj[d].append(s["capability"])
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
        try:
            raw_response = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(raw_response)
            plan = self._validate_plan(parsed, state)
            return {"plan": plan}
        except (json.JSONDecodeError, ValueError) as e:
            err: NodeError = {
                "source": "planner",
                "kind": "planner_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}
