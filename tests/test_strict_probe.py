"""The strict-mode probe normalises the IR schema the way a strict provider
demands, and judges a program by the analysis' codes, not by its wording
(→ docs/design/compiler-v1.md, "The IR", 0219)."""
from __future__ import annotations

import copy
import json

import pytest

from evals.strict_probe import REFUSED, degraded, judge, schema_of, strict

PROGRAM = {"version": 1, "output": ["compare"], "steps": [
    {"id": "search", "op": "web_search", "args": {"query": "q"}},
    {"id": "reads", "op": "analyze", "args": {"material": ["$item"], "instruction": "i"},
     "map": {"over": "$search.items", "max_items": 20, "concurrency": 5, "on_error": "partial"}},
    {"id": "compare", "op": "synthesize", "args": {"items": ["$reads"], "instruction": "c"}},
]}


def _objects(node):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


@pytest.mark.parametrize("variant", ["full", "fallback"])
def test_every_object_is_closed_with_every_key_required_and_no_one_of(variant):
    """OpenAI's strict mode refuses each of these, measured (0219)."""
    schema = strict(schema_of(variant))
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])
    assert "oneOf" not in json.dumps(schema)


def _with(path, value):
    program = copy.deepcopy(PROGRAM)
    step, key = path
    program["steps"][step]["args"][key] = value
    return json.dumps(program)


@pytest.mark.parametrize("text, verdict", [
    (json.dumps(PROGRAM), "valid"),
    (_with((2, "items"), ["$nowhere"]), "unknown_ref"),
    (_with((2, "items"), ["$item"]), "item_outside_map"),
    (_with((0, "query"), "$compare"), "cycle"),
    (_with((2, "items"), "$reads"), "args"),
    ("{not json", "json"),
])
def test_a_program_is_judged_by_the_first_code_it_fails(text, verdict):
    assert judge("full", text)[0] == verdict


def test_the_fallback_checks_the_arguments_it_left_unconstrained():
    step = {"id": "s", "op": "web_search", "args_json": json.dumps({"q": "x"})}
    assert judge("fallback", json.dumps({"version": 1, "steps": [step]}))[0] == "args"
    step["args_json"] = json.dumps({"query": "x"})
    assert judge("fallback", json.dumps({"version": 1, "steps": [step]}))[0] == "valid"


def test_a_shared_step_leaves_the_op_and_args_pairing_to_the_analysis():
    """`{material}` alone is a valid `args` shape, so the schema takes it for
    `analyze`, which needs an instruction: Haiku wrote it 3 times in 40 (0219)."""
    step = {"id": "a", "op": "analyze", "args": {"material": ["$input.notes"]}}
    assert judge("shared", json.dumps({"version": 1, "steps": [step]}))[0] == "op_args"
    step["op"] = "analysis"
    assert judge("shared", json.dumps({"version": 1, "steps": [step]}))[0] == "valid"


def test_degrading_moves_only_the_refused_keywords_and_keeps_pattern():
    """Anthropic refuses bounds and `maxItems` but accepts `pattern` (0219)."""
    text = json.dumps(degraded(strict(schema_of("full"))))
    assert '"pattern"' in text
    assert not any(f'"{key}"' in text for key in REFUSED)
