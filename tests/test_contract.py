"""The job ↔ conversation contract stands without the bench
(→ docs/design/core-v1.md, gate G5).

A LangChain agent built from `adapters/langchain` and the engine alone
launches a plain graph (G2's sum), which outlives the turn and is promoted;
its ending is told on a later turn — and through a crash on each side of the
checkpoint, it is in the thread exactly once. In a fresh interpreter, so what
it did NOT load is part of the proof.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap

G5 = textwrap.dedent('''
    import asyncio, json, sys
    from typing import Any, TypedDict

    from langchain.agents import create_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.store.memory import InMemoryStore
    from pydantic import BaseModel, Field

    from jobsmith.adapters.langchain import JobDeliveryMiddleware, launch_tool
    from jobsmith.engine.delivery import Pulled
    from jobsmith.engine.graph import GraphSpec
    from jobsmith.engine.manager import JobManager

    class In(TypedDict):
        a: int
        b: int

    class Out(TypedDict):
        sum: int

    class State(In, Out, total=False):
        pass

    GATE = {}

    async def add(state):
        await GATE["open"].wait()                 # outlives the turn: promoted
        return {"sum": state["a"] + state["b"]}

    graph = StateGraph(State, input_schema=In, output_schema=Out)
    graph.add_node("add", add)
    graph.add_edge(START, "add")
    graph.add_edge("add", END)

    class Scripted(BaseChatModel):
        replies: list[Any]
        fail_on: set[int] = Field(default_factory=set)
        calls: list[Any] = Field(default_factory=list)

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            n = len(self.calls)
            self.calls.append(list(messages))
            if n in self.fail_on:
                raise RuntimeError("died before the checkpoint")
            return ChatResult(generations=[ChatGeneration(message=self.replies[n])])

    class Sum(BaseModel):
        a: int
        b: int

    async def main():
        GATE["open"] = asyncio.Event()
        jobs = JobManager(GraphSpec("sum", graph.compile(checkpointer=MemorySaver())),
                          InMemoryStore(), deliverers=[Pulled("conversation")])
        address = {"kind": "conversation", "id": "t1"}
        tool = launch_tool(jobs, reply_to=address, args_schema=Sum, name="add",
                           description="Add two numbers on the job engine.",
                           label_of=lambda args: f"{args['a']} + {args['b']}", timeout=0.05)
        model = Scripted(replies=[
            AIMessage("", tool_calls=[{"name": "add", "args": {"a": 1, "b": 2}, "id": "c1"}]),
            AIMessage("It runs in the background."),
            AIMessage("never said: this call dies"),
            AIMessage("The sum is 3."),
            AIMessage("You are welcome."),
        ], fail_on={2})
        agent = create_agent(model, tools=[tool], middleware=[JobDeliveryMiddleware(jobs, address)],
                             checkpointer=MemorySaver())
        cfg = {"configurable": {"thread_id": "t1"}}

        await agent.ainvoke({"messages": [HumanMessage("add 1 and 2")]}, cfg)
        (job,) = await jobs.list_jobs(limit=None)
        promoted = job.status.value
        GATE["open"].set()
        while (job := await jobs.get_job(job.job_id)).status.value != "done":
            await asyncio.sleep(0.01)

        crashes = []
        try:                                      # the model dies: nothing checkpointed
            await agent.ainvoke({"messages": [HumanMessage("any news?")]}, cfg)
        except RuntimeError as e:
            crashes.append(str(e))
        mark = jobs.mark_delivered

        async def dies(job_id):
            raise RuntimeError("died before the mark")

        jobs.mark_delivered = dies                # the answer is checkpointed, the mark dies
        try:
            await agent.ainvoke({"messages": [HumanMessage("any news?")]}, cfg)
        except RuntimeError as e:
            crashes.append(str(e))
        jobs.mark_delivered = mark
        await agent.ainvoke({"messages": [HumanMessage("thanks")]}, cfg)

        thread = (await agent.aget_state(cfg)).values
        job = await jobs.get_job(job.job_id)
        loaded = sorted({m.split(".")[1] for m in sys.modules
                         if m.startswith("jobsmith.") and m.split(".")[1] not in
                         ("adapters", "engine")})
        print(json.dumps({
            "promoted": promoted, "result": job.result, "crashes": crashes,
            "told": [sum(isinstance(m, SystemMessage) and job.job_id[:8] in m.content
                         for m in call) for call in model.calls],
            "in_thread": [m.text for m in thread["messages"] if m.type == "ai"].count("The sum is 3."),
            "recorded": list(thread["delivered_jobs"]), "delivered": bool(job.delivered_at),
            "job_id": job.job_id, "loaded": loaded}))

    asyncio.run(main())
''')


def test_a_conversation_built_from_the_adapter_alone_is_told_once():
    run = subprocess.run([sys.executable, "-c", G5], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])

    assert report["promoted"] == "running" and report["result"] == {"sum": 3}
    assert report["crashes"] == ["died before the checkpoint", "died before the mark"]
    # told on the call that died, again on the next — whose answer stuck — then never
    assert report["told"] == [0, 0, 1, 1, 0]
    assert report["in_thread"] == 1
    assert report["recorded"] == [report["job_id"]] and report["delivered"]
    assert report["loaded"] == []               # no bench: no chat, dag, service, app…


async def test_the_default_notice_says_how_each_job_ended():
    """What an application that overrides nothing tells its model: the result
    of a job that is DONE, the reason of one that is not."""
    from jobsmith.adapters.langchain import JobDeliveryMiddleware
    from jobsmith.engine.models import Job, JobStatus

    done = Job(job_id="aaaaaaaa1", status=JobStatus.DONE, label="1 + 2", result={"sum": 3})
    failed = Job(job_id="bbbbbbbb2", status=JobStatus.FAILED, label="1 / 0", error="division by zero")
    notice = (await JobDeliveryMiddleware(None, {"kind": "none"}).notice([done, failed])).content

    assert "Job aaaaaaaa ('1 + 2') is DONE. Its result: {\"sum\": 3}" in notice
    assert "Job bbbbbbbb ('1 / 0') is FAILED. Why: division by zero" in notice
