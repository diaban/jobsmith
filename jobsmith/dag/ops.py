"""What a step can use: `OpSpec`, the one notion of an op, and its `Effects`.

`docs/design/compiler-v1.md` ("The registry"). Today's `CapabilitySpec` plus
what the compiler will need to know about an op without running it: what it
does to the world (`effects`, declared in step 1b, enforced in step 3), which
argument gives its output type (`output_from`: `extract`'s `fields`), and the
model its resolved arguments must fit (`input_model`), checked by the
interpreter before the step runs. `CapabilitySpec` stays as an alias while
the code migrates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class Effects:
    """What running the op does outside the job. Declared, not yet enforced."""
    read_only: bool = True        # False: writes, sends, pays
    idempotent: bool = True       # running it twice leaves the world as running it once
    cost: Literal["free", "llm", "agent"] = "llm"   # an order of magnitude, for the bound
    approval: bool = False        # a human says yes before it runs
    max_fanout: int = 50          # the largest map over this op


@dataclass(frozen=True, slots=True)
class OpSpec:
    """Self-description used for the planner prompt, the interpreter and job
    metadata. `input_schema`/`output_schema` are the JSON Schemas the planner
    is shown; `input_model`, when an op declares one, is binding: resolved
    arguments it refuses fail the step."""
    name: str                                   # unique; doubles as node-name suffix
    description: str                            # one paragraph, feeds the planner prompt
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    requires_inputs: tuple[str, ...] = ()       # keys that must exist in state["inputs"]
    effects: Effects = field(default_factory=Effects)
    output_from: str | None = None              # the argument that gives the output type
    input_model: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if not NAME_RE.match(self.name):
            raise ValueError(
                f"invalid capability name {self.name!r} (must match {NAME_RE.pattern})"
            )


CapabilitySpec = OpSpec
