"""The program the compiler runs: its IR, as Pydantic models (step 1b).

`docs/design/compiler-v1.md` ("The IR", "What a program delivers"); vocabulary
in `docs/glossary.md`. A program is `steps` (the work the planner chose) and a
`result` (the slots that produce the job result; one `answer` slot until the
result contract, step R).

The models validate; the state stores JSON. The IR is data (0228): the `plan`
channel, the `plan` fact and the HTTP view hold `Program.stored()`, a plain dict
in which each step also carries the two fields today's readers key on —
`capability` (its op) and `depends_on` (its dependencies, derived) — until
they read `op` and derive dependencies themselves.

A reference is a whole argument value, never interpolated: `$<id>` (that
step's result data), `$<id>.<field>[.<field>…]`, `$input.<key>` (the job's
inputs). `$item` is reserved for `map` (step 2). Dependencies are derived from
references plus `after`; nobody writes them.
"""
from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[a-z][a-z0-9_]*$"
REF_PATTERN = r"^\$(input|item|[a-z][a-z0-9_]*)(\.[a-zA-Z0-9_]+)*$"
_REF_RE = re.compile(REF_PATTERN)

Id = Annotated[str, Field(pattern=ID_PATTERN)]
Ref = Annotated[str, Field(pattern=REF_PATTERN)]

INPUT, ITEM = "input", "item"


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Step(_Closed):
    """One work step: an op, its arguments, and what it waits for with no data."""
    id: Id
    op: str
    args: dict[str, Any] = {}
    after: list[Id] = []


class AnswerSlot(_Closed):
    """The text for a human. `material` None reads every successful work step
    result, in program order: today's behaviour."""
    mode: Literal["reply", "brief"] = "reply"
    instruction: str = ""
    material: list[Ref] | None = None


class Result(_Closed):
    answer: AnswerSlot = AnswerSlot()


class Program(_Closed):
    version: Literal[1] = 1
    steps: list[Step]
    result: Result = Result()
    rationale: str = ""

    def stored(self) -> dict[str, Any]:
        """What the `plan` channel holds: JSON, each step with its op as
        `capability` and its derived `depends_on`, for today's readers."""
        data = self.model_dump()
        for step, model in zip(data["steps"], self.steps, strict=True):
            step["capability"] = model.op
            step["depends_on"] = dependencies(model)
        return data


def parse_ref(value: Any) -> tuple[str, list[str]] | None:
    """`("compare", ["items"])` for `"$compare.items"`; None when `value` is
    not a reference (a literal, or a string that only starts with `$`)."""
    if not isinstance(value, str) or not _REF_RE.match(value):
        return None
    head, *path = value[1:].split(".")
    return head, path


def refs_in(value: Any) -> Iterator[str]:
    """Every reference in an argument value, in order, at any depth."""
    if isinstance(value, str):
        if parse_ref(value):
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from refs_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from refs_in(item)


def dependencies(step: Step) -> list[str]:
    """The step ids `step` waits for: those its arguments reference, then
    `after`, each once."""
    heads = (parse_ref(ref) for ref in refs_in(step.args))
    ids = [head for head, _ in filter(None, heads) if head not in (INPUT, ITEM)]
    return list(dict.fromkeys([*ids, *step.after]))


def from_plan(raw: Mapping[str, Any]) -> Program:
    """A program from what a planner wrote: the IR (`op`, `args`, `after`), or
    today's plan (`capability`, `depends_on`), read as an IR with no args."""
    steps = []
    for step in raw.get("steps") or []:
        if not isinstance(step, Mapping):
            raise ValueError(f"a step must be an object: {step!r}")
        if "op" in step:
            steps.append({k: step[k] for k in ("id", "op", "args", "after") if k in step})
        else:
            name = step.get("capability")
            steps.append({"id": step.get("id") or name, "op": name,
                          "after": step.get("depends_on") or []})
    return Program.model_validate({"steps": steps,
                                   "result": raw.get("result") or {},
                                   "rationale": str(raw.get("rationale", ""))})


class UnresolvedReference(ValueError):
    """A reference names nothing the run has: a step with no successful
    result, a missing field, an input the job was not given."""


def resolve(value: Any, *, results: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
    """`value` with every reference replaced by what it names; a literal is
    returned as it is. Raises `UnresolvedReference`."""
    if isinstance(value, list):
        return [resolve(item, results=results, inputs=inputs) for item in value]
    if isinstance(value, dict):
        return {key: resolve(item, results=results, inputs=inputs)
                for key, item in value.items()}
    ref = parse_ref(value)
    if ref is None:
        return value
    head, path = ref
    if head == ITEM:
        raise UnresolvedReference(f"{value}: $item is only defined inside a map")
    if head == INPUT:
        if not path or path[0] not in inputs:
            raise UnresolvedReference(f"{value}: no such input")
        current, rest = inputs[path[0]], path[1:]
    else:
        result = results.get(head) or {}
        if not result.get("ok"):
            raise UnresolvedReference(f"{value}: step {head!r} has no successful result")
        current, rest = result.get("data") or {}, path
    for name in rest:
        if not isinstance(current, Mapping) or name not in current:
            raise UnresolvedReference(f"{value}: no field {name!r}")
        current = current[name]
    return current
