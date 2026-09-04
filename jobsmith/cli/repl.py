"""Interactive chat loop, written against the client interface.

Identical experience whether it is backed by a daemon or by an embedded
agent — only the lifetime of the jobs differs (see cli/client.py).

Commands:
  <any text>        chat (the agent may propose launching a job — approve y/N)
  /jobs             list jobs
  /job <id-prefix>  show a job's plan, artifacts and answer
  /report <id-pfx>  print a finished job's markdown report
  /bg <any text>    bypass the chat: run that query as a job directly
  /image <key>      attach an image input to the NEXT /bg job
  /cancel <id-pfx>  cancel a job
  /resume <id-pfx>  restart a stopped job from its checkpoint
  /quit             exit
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..core.usage import Usage
from ..jobs.report import format_step_usage, format_usage
from .client import AgentClient

BANNER = "\n".join(
    line for line in (__doc__ or "").splitlines() if line.startswith(("  ", "Commands"))
)


def show_job(job: dict, *, verbose: bool = True) -> None:
    print(f"  job {job['job_id'][:8]}  [{job['status']}]  {job['query'][:60]!r}")
    if not verbose:
        return
    plan = job.get("plan")
    if plan:
        done = job.get("step_finished_at") or {}
        steps = " -> ".join(
            s["capability"] + ("" if s["capability"] in done else " (pending)")
            for s in plan["steps"]
        )
        print(f"  plan:      {steps}")
        print(f"  rationale: {plan['rationale']}")
    for name, res in (job.get("results") or {}).items():
        status = "ok" if res.get("ok") else f"FAILED ({res.get('error')})"
        spent = format_step_usage(Usage.from_dict((res.get("meta") or {}).get("usage")))
        print(f"  artifact:  {name}: {status}" + (f"  [{spent}]" if spent != "—" else ""))
    if job.get("usage"):
        # what the run cost, right where its steps are listed
        print(f"  usage:     {format_usage(Usage.from_dict(job['usage']))}")
    if job.get("report_path"):
        print(f"  report:    {job['report_path']}")
    if job.get("final_answer"):
        print("  answer:\n    " + job["final_answer"].replace("\n", "\n    "))
    if job.get("error") and job["status"] != "done":
        print(f"  error:     {job['error']}")


async def _resolve(client: AgentClient, prefix: str) -> dict | None:
    job = await client.resolve_job(prefix)
    if job is None:
        print(f"  no single job matching {prefix!r}")
    return job


async def run_repl(client: AgentClient, session_id: str) -> None:
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
            jobs = await client.list_jobs()
            if not jobs:
                print("  (no jobs yet)")
            for job in jobs:
                show_job(job, verbose=False)
        elif line.startswith("/job "):
            job = await _resolve(client, line.split(maxsplit=1)[1])
            if job:
                show_job(job)
        elif line.startswith("/report "):
            job = await _resolve(client, line.split(maxsplit=1)[1])
            if job:
                report = await client.get_report(job["job_id"])
                print(report or "  no report yet (is the job done?)")
        elif line.startswith("/cancel "):
            job = await _resolve(client, line.split(maxsplit=1)[1])
            if job:
                print(f"  -> {(await client.cancel_job(job['job_id']))['status']}")
        elif line.startswith("/resume "):
            job = await _resolve(client, line.split(maxsplit=1)[1])
            if job:
                resumed = await client.resume_job(job["job_id"])
                print("  " + (f"cannot resume: {resumed['error']}" if resumed.get("error")
                              else f"-> {resumed['status']}"))
        elif line.startswith("/image "):
            key = line.split(maxsplit=1)[1]
            pending_inputs["image_s3_keys"] = [key]
            print(f"  image {key!r} will be attached to the next /bg job")
        elif line.startswith("/bg "):
            launched = await client.launch_job(
                line[4:].strip(), session_id=session_id, inputs=dict(pending_inputs) or None
            )
            pending_inputs.clear()
            short = launched["job_id"][:8]
            print(f"  started in background: {short}  (try /jobs, /job {short[:4]})")
        elif line.startswith("/"):
            print("  unknown command (try /jobs, /job, /report, /bg, /image, "
                  "/cancel, /resume, /quit)")
        else:
            reply = await client.send(session_id, line)
            # human-in-the-loop: the agent proposes a job, you approve or not
            while reply.get("type") == "proposal":
                print("\n  the agent proposes a background job:")
                print(f"    task     : {reply.get('query')}")
                print(f"    approach : {reply.get('rationale')}")
                answer = await loop.run_in_executor(None, input, "  launch it? [y/N] ")
                approved = answer.strip().lower() in ("y", "yes", "o", "oui")
                reply = await client.approve(session_id, approved)
            print("  " + (reply.get("content") or "").replace("\n", "\n  "))

    print("bye")
