"""Registry-driven planner.

The class owns:
- the prompt template (rendered from the registry's capability specs)
- plan validation (allowed names, applicability, cycle detection)
- the LLM call

It exposes `run` (the node coroutine) for the parent graph to register.

Validation answers one of three things, never two of them at once: a plan with
steps, an EMPTY plan (every step the model chose was dropped as inapplicable —
nothing to run, nothing wrong), or an unrecoverable `NodeError`. What to do
about an empty plan is a control-flow decision and lives in the builder's path
map, not here.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from ..engine.facts import publish
from .deps import Deps
from .profile import DEFAULT_PLANNER_TEMPLATE
from .registry import CapabilityRegistry
from .state import CONVERSATION_INPUT_KEY, AgentState, NodeError, Plan, PlanStep, step_id

# What a step id may be: the same shape as a capability's name, which is what
# a step with no id of its own is known by.
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def without_steps(plan: Plan, ids: Sequence[str]) -> Plan:
    """`plan` with the steps `ids` removed and every `depends_on` on them
    pruned — what dropping a step from a running plan leaves (#177). Still
    acyclic: removing nodes adds no edge."""
    gone = set(ids)
    return {**plan, "steps": [
        {"id": step_id(step), "capability": step["capability"],
         "depends_on": [d for d in step["depends_on"] if d not in gone]}
        for step in plan["steps"] if step_id(step) not in gone]}


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

    @staticmethod
    def user_message(state: AgentState) -> str:
        """The request, prefixed by the conversation it came from when there is one.

        Chat-launched jobs carry a bounded excerpt of the recent turns in
        `inputs[CONVERSATION_INPUT_KEY]` (see chat/tools.py). It is background
        material only — it exists so a request like "analyse that" still has a
        referent — so the request itself stays clearly marked as the thing to
        plan for.
        """
        conversation = (state.get("inputs") or {}).get(CONVERSATION_INPUT_KEY)
        if not conversation:
            return state["query"]
        return (
            "Conversation this request came from (background, do not plan for it):\n"
            f"{conversation}\n\n"
            f"Request to plan for:\n{state['query']}"
        )

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
        cleaned: list[PlanStep] = []
        for step in steps:
            name = step.get("capability")
            if name not in allowed:
                raise ValueError(f"unknown capability: {name}")
            # A step is known by its id, its capability's name when it gives
            # none: so one capability may run as several steps only when each
            # is told apart by an id — a plan with none, which is every plan
            # the prompt asks for, still refuses a capability twice (0217).
            sid = step.get("id") or name
            if not isinstance(sid, str) or not _ID_RE.match(sid):
                raise ValueError(f"bad step id for {name}: {sid!r}")
            if sid in seen or sid in dropped:
                raise ValueError(f"duplicate step id: {sid}")
            deps = step.get("depends_on") or []
            if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
                raise ValueError(f"bad depends_on for {sid}")
            if not self.registry.get(name).is_applicable(state):
                dropped.add(sid)
                continue
            seen.add(sid)
            cleaned.append({"id": sid, "capability": name, "depends_on": list(deps)})

        # Prune depends_on entries that reference dropped (inapplicable) steps;
        # references to steps absent from the plan altogether are still errors.
        surviving = {step_id(s) for s in cleaned}
        for s in cleaned:
            kept: list[str] = []
            for d in s["depends_on"]:
                if d in surviving:
                    kept.append(d)
                elif d not in dropped:
                    raise ValueError(f"depends_on references unknown step: {d}")
            s["depends_on"] = kept

        # Kahn's algo for cycle detection
        indeg = {step_id(s): len(s["depends_on"]) for s in cleaned}
        adj: dict[str, list[str]] = {step_id(s): [] for s in cleaned}
        for s in cleaned:
            for d in s["depends_on"]:
                adj[d].append(step_id(s))
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

        # `cleaned` can only be empty because every step was dropped as
        # inapplicable: an unknown name, a duplicate id or a bad dependency raises,
        # and an empty `steps` from the model was rejected above. So it is a
        # fact about the request (no image for `vision`), not a broken plan —
        # it travels as an EMPTY PLAN, and `AgentBuilder._route_after_planner`
        # decides where that goes. The decision belongs to the path map, not
        # to a rescue hidden in here.
        return Plan(steps=cleaned, rationale=str(raw.get("rationale", "")))

    # -------- Node --------

    async def run(self, state: AgentState) -> dict:
        try:
            raw_response = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": self.user_message(state)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            parsed = json.loads(raw_response)
            plan = self._validate_plan(parsed, state)
            publish("plan", plan)          # the job shows it as soon as it exists
            return {"plan": plan}
        except (json.JSONDecodeError, ValueError) as e:
            err: NodeError = {
                "source": "planner",
                "kind": "planner_fail",
                "detail": str(e),
                "recoverable": False,
            }
            return {"errors": [err]}
