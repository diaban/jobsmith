"""Triage router: decides HOW a user message is handled, before any planning.

This is a dedicated decision node — the planner never decides whether to plan.
v1 routes:
- "plan":   the request needs capabilities → planner emits a DAG (the default)
- "direct": the message needs none (greeting, question about the agent itself)
            → DirectResponder answers immediately

The decision is FAIL-OPEN: any LLM error, bad JSON, or unknown route name
falls back to "plan" — the full pipeline can always handle a message the
direct path could have, the reverse is not true.

That holds only while there is something to plan WITH. An agent whose
capabilities are all conditionally registered can legitimately compose an
EMPTY registry, and then "plan" is not a wider door but a wall: the planner
would be asked to plan against nothing and raise. So the empty registry is
decided structurally — route "direct", with no LLM call at all — and the
fail-open fallback follows it there rather than into the wall.

Adding a route = an entry in `routes` (its prompt description), a node for it,
and a target in the builder's router path map.
"""
from __future__ import annotations

import json

from .deps import Deps
from .profile import DEFAULT_ROUTER_TEMPLATE
from .registry import CapabilityRegistry
from .state import AgentState

DEFAULT_ROUTES: dict[str, str] = {
    "plan": (
        "the request needs one or more of the capabilities below — "
        "a DAG of capability steps will be planned and executed"
    ),
    "direct": (
        "the message needs no capability at all — greetings, small talk, "
        "or questions about the assistant itself (e.g. what it can do)"
    ),
}

FALLBACK_ROUTE = "plan"

# Where the fallback goes instead when the registry can serve nothing: planning
# against an empty registry is a guaranteed hard stop, answering is not.
NO_CAPABILITY_ROUTE = "direct"


class Router:

    DEFAULT_TEMPLATE = DEFAULT_ROUTER_TEMPLATE

    def __init__(
        self,
        deps: Deps,
        registry: CapabilityRegistry,
        *,
        routes: dict[str, str] | None = None,
        prompt_template: str | None = None,
    ):
        self.deps = deps
        self.registry = registry
        self.routes = dict(routes or DEFAULT_ROUTES)
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    # -------- Prompt rendering --------

    def system_prompt(self) -> str:
        routes = "\n".join(f'- "{name}": {desc}' for name, desc in self.routes.items())
        capabilities = "\n".join(
            f"- {spec.name}: {spec.description}" for spec in self.registry.specs()
        )
        return self.prompt_template.format(routes=routes, capabilities=capabilities)

    # -------- Decision --------

    def _can_plan(self) -> bool:
        """Is the "plan" route viable at all? Only if something backs it."""
        return len(self.registry) > 0

    def _fallback_route(self) -> str:
        """The route to take when nothing else decided — see the module docstring."""
        if not self._can_plan() and NO_CAPABILITY_ROUTE in self.routes:
            return NO_CAPABILITY_ROUTE
        return FALLBACK_ROUTE

    # -------- Node --------

    async def run(self, state: AgentState) -> dict:
        if not self._can_plan():
            # Structural, not a judgement call: with no capability to
            # orchestrate there is nothing for the planner to do, whatever the
            # message says. Deterministic, and it costs no LLM call.
            return {"route": self._fallback_route()}
        try:
            raw = await self.deps.llm.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": state["query"]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            route = json.loads(raw).get("route")
            if route not in self.routes:
                route = self._fallback_route()
        except Exception:  # fail-open by design, see module docstring
            route = self._fallback_route()
        return {"route": route}

    # -------- Router (conditional edge) --------

    def route(self, state: AgentState) -> str:
        return state.get("route") or self._fallback_route()
