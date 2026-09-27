"""The job engine runs any LangGraph graph, with no change to it
(→ docs/design/core-v1.md, gates G1 and G2).

G2 here: a plain structured graph — no chat, no document, no planner — runs
as a job on the engine alone, and the job's result is exactly what the graph
returned. It runs in a fresh interpreter, so what it did NOT load is part of
the proof. (Delivery at settle joins this gate at step 7; G1, a ReAct graph,
at step 8.)
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

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
                          "steps": sorted(done.steps), "loaded": loaded}))

    asyncio.run(main())
''')


def test_a_structured_job_runs_on_the_engine_alone():
    run = subprocess.run([sys.executable, "-c", G2], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report == {"status": "done", "result": {"sum": 3}, "steps": ["add"], "loaded": []}
