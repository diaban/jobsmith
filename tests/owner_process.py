"""A job owner in a process of its own — driven by tests/test_cross_process.py.

    python tests/owner_process.py <db> <reports_dir>

Opens the SQLite file, creates a two-step job (`alpha`, then `slow`, which
sleeps for a minute), prints its id on stdout, then AWAITS `run_job` — the
`jobsmith run --wait` shape, where a cancel that is not this process's own
must come back as a CANCELLED job and not as an exception. Prints the final
status and exits 0. Kept outside `test_*` so pytest does not collect it.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack

from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.app.persistence import open_persistence
from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.ownership import LeasePolicy


class Step(Capability):
    def __init__(self, name: str, delay: float):
        self.spec = CapabilitySpec(name=name, description=f"{name} capability")
        self.delay = delay

    async def work(self, state: CapabilityBaseState) -> dict:
        await asyncio.sleep(self.delay)
        return self._emit_success({"echo": f"{self.spec.name}@{self.delay}"})

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


async def main(db: str, reports: str) -> None:
    async with AsyncExitStack() as stack:
        checkpointer, store = await open_persistence(db, stack)
        llm = FakeLLM({"planner": plan_json("alpha", "slow", deps={"slow": ["alpha"]})},
                      default="A sufficiently long final answer for the job test.")
        caps = [Step("alpha", 0.0), Step("slow", 60.0)]
        graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps),
                            checkpointer=checkpointer)
        mgr = JobManager(graph, store, reports_dir=reports,
                         lease=LeasePolicy(heartbeat=0.1))
        job = await mgr.create_job("a job owned by another process")
        print(job.job_id, flush=True)
        settled = await mgr.run_job(job.job_id)
        print(settled.status.value, flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
