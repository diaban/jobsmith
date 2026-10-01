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
from collections.abc import Sequence
from typing import Any, cast

from pydantic import ValidationError

from ..engine.facts import publish
from .deps import Deps
from .ir import Program, Step, dependencies, from_plan, parse_ref
from .profile import DEFAULT_PLANNER_TEMPLATE
from .registry import CapabilityRegistry
from .state import CONVERSATION_INPUT_KEY, AgentState, NodeError, Plan, PlanStep, step_id


def without_steps(plan: Plan, ids: Sequence[str]) -> Plan:
    """`plan` with the steps `ids` removed and every `depends_on`, `after`
    and answer material naming them pruned — what dropping a step from a
    running plan leaves (#177). Still acyclic: removing nodes adds no edge. A
    step whose arguments reference a dropped one keeps the reference, and
    fails when the interpreter cannot resolve it (#230)."""
    gone = set(ids)
    steps: list[PlanStep] = []
    for step in plan["steps"]:
        if step_id(step) in gone:
            continue
        kept: PlanStep = {**step, "id": step_id(step),
                          "depends_on": [d for d in step["depends_on"] if d not in gone]}
        if "after" in step:
            kept["after"] = [d for d in step["after"] if d not in gone]
        steps.append(kept)
    amended: Plan = {**plan, "steps": steps}
    answer = (plan.get("result") or {}).get("answer") or {}
    if answer.get("material"):
        material = [ref for ref in answer["material"]
                    if (parse_ref(ref) or ("", []))[0] not in gone]
        amended["result"] = {**plan.get("result", {}), "answer": {**answer, "material": material}}
    return amended


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
        # Today's plans (`capability`, `depends_on`) read as an IR with no
        # args (dag/ir.py); a step with no id has its op's name, so a plan
        # with none still refuses an op twice (0217).
        try:
            program = from_plan(raw)
        except ValidationError as exc:
            error = exc.errors()[0]
            where = ".".join(str(part) for part in error["loc"])
            raise ValueError(f"malformed plan at {where}: {error['msg']}") from None

        allowed = set(self.registry.names())
        seen: set[str] = set()
        dropped: set[str] = set()
        kept: list[Step] = []
        for step in program.steps:
            if step.op not in allowed:
                raise ValueError(f"unknown capability: {step.op}")
            if step.id in seen or step.id in dropped:
                raise ValueError(f"duplicate step id: {step.id}")
            if not self.registry.get(step.op).is_applicable(state):
                dropped.add(step.id)
                continue
            seen.add(step.id)
            kept.append(step)

        # `after` entries naming dropped (inapplicable) steps are pruned; a
        # step absent from the program altogether is still an error, and so
        # is a reference to a dropped step: its argument would have no value.
        cleaned: list[Step] = []
        for step in kept:
            step = step.model_copy(update={"after": [d for d in step.after if d not in dropped]})
            for dep in dependencies(step):
                if dep in dropped:
                    raise ValueError(f"{step.id} references {dep}, which cannot run here")
                if dep not in seen:
                    raise ValueError(f"depends_on references unknown step: {dep}")
            cleaned.append(step)

        # The answer's material names whole steps of the program (#230);
        # a dropped one is pruned like `after`.
        answer = program.result.answer
        if answer.material is not None:
            material = []
            for ref in answer.material:
                head, path = parse_ref(ref) or ("", [])
                if head in dropped:
                    continue
                if head not in seen or path:
                    raise ValueError(f"answer material must name a step of the program: {ref}")
                material.append(ref)
            answer = answer.model_copy(update={"material": material})

        # Kahn's algo for cycle detection, over the derived dependencies
        deps = {step.id: dependencies(step) for step in cleaned}
        indeg = {sid: len(ds) for sid, ds in deps.items()}
        adj: dict[str, list[str]] = {sid: [] for sid in deps}
        for sid, ds in deps.items():
            for d in ds:
                adj[d].append(sid)
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
        result = program.result.model_copy(update={"answer": answer})
        stored = Program(steps=cleaned, result=result, rationale=program.rationale).stored()
        return cast(Plan, stored)

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
