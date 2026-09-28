"""The job engine runs any LangGraph graph, with no change to it
(→ docs/design/core-v1.md, gates G1 and G2).

G2 here: a plain structured graph — no chat, no document, no planner — runs
as a job on the engine alone, and the job's result is exactly what the graph
returned, and — addressed to nobody — it is delivered as it settles. It runs
in a fresh interpreter, so what it did NOT load is part of the proof. (G1, a
ReAct graph, joins at step 8.)
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from jobsmith.engine.graph import GraphSpec
from jobsmith.engine.manager import JobManager

G2 = textwrap.dedent('''
    import asyncio, json, sys
    from typing import TypedDict

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.store.memory import InMemoryStore

    from jobsmith.engine.graph import GraphSpec
    from jobsmith.engine.manager import JobManager

    class In(TypedDict):
        a: int
        b: int

    class Out(TypedDict):
        sum: int

    class State(In, Out, total=False):
        pass

    graph = StateGraph(State, input_schema=In, output_schema=Out)
    graph.add_node("add", lambda state: {"sum": state["a"] + state["b"]})
    graph.add_edge(START, "add")
    graph.add_edge("add", END)

    async def main():
        jobs = JobManager(GraphSpec("sum", graph.compile(checkpointer=MemorySaver())),
                          InMemoryStore())
        job = await jobs.create_job({"a": 1, "b": 2}, label="1 + 2")
        done = await jobs.run_job(job.job_id)
        loaded = sorted(m for m in sys.modules
                        if m.startswith(("jobsmith.dag", "jobsmith.chat", "jobsmith.artifacts",
                                         "jobsmith.agents", "langchain."))
                        or m == "langchain")
        print(json.dumps({"status": done.status.value, "result": done.result,
                          "steps": sorted(done.steps), "delivered": bool(done.delivered_at),
                          "loaded": loaded}))

    asyncio.run(main())
''')


def test_a_structured_job_runs_on_the_engine_alone():
    run = subprocess.run([sys.executable, "-c", G2], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report == {"status": "done", "result": {"sum": 3}, "steps": ["add"],
                      "delivered": True, "loaded": []}


async def test_one_engine_runs_several_graphs_each_by_its_name():
    """The record names its graph, and a job runs the graph it names; the first
    is the default, and a name this process does not have is refused."""
    class State(TypedDict):
        n: int

    def graph(step: int):
        g = StateGraph(State)
        g.add_node("step", lambda state: {"n": state["n"] + step})
        g.add_edge(START, "step")
        g.add_edge("step", END)
        return g.compile(checkpointer=MemorySaver())

    jobs = JobManager(GraphSpec("plus_one", graph(1)), InMemoryStore(),
                      graphs=[GraphSpec("plus_ten", graph(10))])
    default = await jobs.run_job((await jobs.create_job({"n": 1})).job_id)
    named = await jobs.run_job((await jobs.create_job({"n": 1}, graph="plus_ten")).job_id)

    assert (default.graph, default.result) == ("plus_one", {"n": 2})
    assert (named.graph, named.result) == ("plus_ten", {"n": 11})
    with pytest.raises(ValueError, match="plus_hundred"):
        await jobs.create_job({"n": 1}, graph="plus_hundred")
