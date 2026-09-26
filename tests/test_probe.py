"""The node probe: one node of the composed agent, N calls per case, tallied.

It is the instrument the effort budget is measured with, so it must refuse to
exceed that budget and must tally what the node actually wrote.
"""
from __future__ import annotations

import json

import pytest

from evals.probe import main, meets, probe, render


async def test_it_tallies_what_the_node_wrote_against_the_expectation():
    cases = [
        {"id": "asks", "query": "compare A and B, and save that as a file", "expect": "truthy"},
        {"id": "silent", "query": "compare A and B", "expect": "falsy"},
    ]
    asks, silent = await probe("document_intent", cases, read="document_formats",
                               n=2, provider="fake")
    assert (asks.passed, asks.n, asks.counts) == (2, 2, {'["markdown"]': 2})
    assert (silent.passed, silent.n) == (2, 2)
    assert "0/2 -> 2/2" in render({"asks": asks.summary()}, {"asks": {"passed": 0, "n": 2}})


async def test_grep_tallies_a_match_instead_of_the_value():
    (tally,) = await probe("direct_answer", [{"id": "hi", "query": "hello"}],
                           read="draft_answer", grep="ask me anything", n=2, provider="fake")
    assert tally.counts == {"true": 2}


async def test_it_refuses_a_probe_over_the_budget_before_composing_anything():
    with pytest.raises(ValueError, match="over the budget"):
        await probe("router", [{"id": "x", "query": "q"}] * 31, read="route", n=10)


@pytest.mark.parametrize(("value", "expect", "ok"), [
    (["markdown"], "truthy", True), (None, "falsy", True), ("plan", "plan", True),
    ("direct", "plan", False),
])
def test_an_expectation_is_truthy_falsy_or_a_value(value, expect, ok):
    assert meets(value, expect) is ok


def test_compare_prints_before_and_after_without_calling_anything(tmp_path, capsys):
    """`make probe` runs both sides apart, then asks this mode for one table."""
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    before.write_text(json.dumps({"c": {"counts": {"null": 3}, "n": 3, "passed": 1, "errors": 0}}))
    after.write_text(json.dumps({"c": {"counts": {'["md"]': 2, "null": 1}, "n": 3,
                                       "passed": 2, "errors": 1}}))
    assert main(["--compare", str(before), str(after)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("c ") and "1/3 -> 2/3" in line
    assert '["md"] x2, null x1' in line and "(1 errors)" in line


def test_a_probe_needs_a_node_and_a_key_unless_it_compares():
    with pytest.raises(SystemExit):
        main(["-q", "hello"])
