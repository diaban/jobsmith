"""Triage router: decides HOW a user message is handled, before any planning.

This is a dedicated decision node — the planner never decides whether to plan.
v1 routes:
- "plan":   the request needs capabilities → planner emits a DAG (the default)
- "direct": the message needs none (greeting, question about the agent itself)
            → DirectResponder answers immediately

The decision is FAIL-OPEN: any LLM error, bad JSON, or unknown route name
falls back to "plan" — the full pipeline can always handle a message the
direct path could have, the reverse is not true.

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

    # -------- Node --------

    async def run(self, state: AgentState) -> dict:
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
                route = FALLBACK_ROUTE
        except Exception:  # fail-open by design, see module docstring
            route = FALLBACK_ROUTE
        return {"route": route}

    # -------- Router (conditional edge) --------

    def route(self, state: AgentState) -> str:
        return state.get("route") or FALLBACK_ROUTE
