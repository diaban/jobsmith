"""Generic chat REPL over a (JobManager, ChatSession) pair.

Domain-neutral: any composition (the default agent, a domain example) builds
its manager + session and hands them to `run_repl`.

Commands:
  <any text>        chat (the agent may propose launching a job — approve y/N)
  /jobs             list all jobs
  /job <id-prefix>  show a job's plan, artifacts, and answer
  /bg <any text>    bypass the chat: run that query as a job directly
  /image <key>      attach an image input to the NEXT /bg job
  /cancel <id-pfx>  cancel a background job
  /quit             exit
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from ..chat.session import ChatSession
from ..jobs.manager import JobManager
from ..jobs.models import Job, JobStatus

BANNER = "\n".join(
    line for line in (__doc__ or "").splitlines() if line.startswith(("  ", "Commands"))
)


def show_job(job: Job, *, verbose: bool = True) -> None:
    print(f"  job {job.job_id[:8]}  [{job.status.value}]  {job.query[:60]!r}")
    if not verbose:
        return
    if job.plan:
        steps = " -> ".join(
            s["capability"] + (f"(after {','.join(s['depends_on'])})" if s["depends_on"] else "")
            for s in job.plan["steps"]
        )
        print(f"  plan:      {steps}")
        print(f"  rationale: {job.plan['rationale']}")
    for name, res in job.results.items():
        status = "ok" if res.get("ok") else f"FAILED ({res.get('error')})"
        print(f"  artifact:  {name}: {status}")
    if job.report_path:
        print(f"  report:    {job.report_path}")
    if job.final_answer:
        print("  answer:\n    " + job.final_answer.replace("\n", "\n    "))
    if job.error and job.status is not JobStatus.DONE:
        print(f"  error:     {job.error}")


async def find_job(jobs: JobManager, prefix: str) -> Job | None:
    matches = [j for j in await jobs.list_jobs(limit=100) if j.job_id.startswith(prefix)]
    if len(matches) != 1:
        print(f"  {'no' if not matches else 'ambiguous'} job for prefix {prefix!r}")
        return None
    return await jobs.get_job(matches[0].job_id)


async def run_repl(manager: JobManager, session: ChatSession) -> None:
    agent = session.build()
    chat_cfg = {"configurable": {"thread_id": session.session_id}}
    pending_inputs: dict[str, Any] = {}
    print(BANNER + "\n")

    loop = asyncio.get_event_loop()
    while True:
        try:
            line = (await loop.run_in_executor(None, input, "agent> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line in ("/quit", "/exit", "/q"):
            break
        elif line == "/jobs":
            listing = await manager.list_jobs(limit=100)
            if not listing:
                print("  (no jobs yet)")
            for j in listing:
                show_job(j, verbose=False)
        elif line.startswith("/job "):
            job = await find_job(manager, line.split(maxsplit=1)[1])
            if job:
                show_job(job)
        elif line.startswith("/cancel "):
            job = await find_job(manager, line.split(maxsplit=1)[1])
            if job:
                cancelled = await manager.cancel_job(job.job_id)
                print(f"  -> {cancelled.status.value}")
        elif line.startswith("/image "):
            key = line.split(maxsplit=1)[1]
            pending_inputs["image_s3_keys"] = [key]
            print(f"  image {key!r} will be attached to the next /bg job")
        elif line.startswith("/bg "):
            job = await manager.create_job(
                line[4:].strip(), dict(pending_inputs), session_id=session.session_id
            )
            pending_inputs.clear()
            manager.start_job(job.job_id)
            print(f"  started in background: {job.job_id[:8]}  (try /jobs, /job {job.job_id[:4]})")
        elif line.startswith("/"):
            print("  unknown command (try /jobs, /job, /bg, /image, /cancel, /quit)")
        else:
            result = await agent.ainvoke({"messages": [HumanMessage(line)]}, chat_cfg)
            # human-in-the-loop: the agent proposes a job, you approve or not
            while "__interrupt__" in result:
                proposal = result["__interrupt__"][0].value
                print("\n  the agent proposes a background job:")
                print(f"    task     : {proposal.get('query')}")
                print(f"    approach : {proposal.get('rationale')}")
                answer = (await loop.run_in_executor(None, input, "  launch it? [y/N] "))
                approved = answer.strip().lower() in ("y", "yes", "o", "oui")
                result = await agent.ainvoke(Command(resume={"approved": approved}), chat_cfg)
            print("  " + (result["messages"][-1].content or "").replace("\n", "\n  "))

    print("bye")
