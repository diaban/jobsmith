"""Capability registry.

The registry is the single source of truth for what the agent can do:
- the planner prompt is rendered from `specs()`
- the executor's Send targets and the builder's node map come from `names()`

It is frozen by `AgentBuilder.build()` because Send targets and conditional
edge path maps must name real compiled-graph nodes: the capability set of a
compiled graph is fixed. Registering a new capability means constructing a
fresh AgentBuilder (compilation is milliseconds).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from .capability import Capability, CapabilitySpec


class CapabilityRegistry:
    def __init__(self, capabilities: Iterable[Capability] = ()):
        self._caps: dict[str, Capability] = {}
        self._frozen = False
        for cap in capabilities:
            self.register(cap)

    def register(self, cap: Capability) -> None:
        if self._frozen:
            raise RuntimeError(
                "registry is frozen (graph already built); create a new AgentBuilder "
                "with a new registry to add capabilities"
            )
        name = cap.spec.name  # spec validates the name format itself
        if name in self._caps:
            raise ValueError(f"capability {name!r} already registered")
        self._caps[name] = cap

    def get(self, name: str) -> Capability:
        try:
            return self._caps[name]
        except KeyError:
            raise KeyError(
                f"unknown capability {name!r}; registered: {sorted(self._caps)}"
            ) from None

    def names(self) -> list[str]:
        return list(self._caps)

    def specs(self) -> list[CapabilitySpec]:
        return [c.spec for c in self._caps.values()]

    def freeze(self) -> None:
        self._frozen = True

    def __contains__(self, name: str) -> bool:
        return name in self._caps

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._caps.values())

    def __len__(self) -> int:
        return len(self._caps)
