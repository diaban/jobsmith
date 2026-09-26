"""The node probe: one node of the composed agent, N calls per case, tallied.

It is the instrument the effort budget is measured with, so it must refuse to
exceed that budget and must tally what the node actually wrote.
"""
from __future__ import annotations

import pytest

from evals.probe import meets, probe, render


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
