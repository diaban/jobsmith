"""jobsmith — the command line for the global agent.

    jobsmith serve [--port 8000]     run the daemon: it owns the job engine
    jobsmith chat [--session ID]     converse (resume a conversation by id)
    jobsmith run "<task>"            launch a job directly, without chatting
    jobsmith jobs [--status done]    list jobs
    jobsmith job <id-prefix>         plan, steps, artifacts, answer
    jobsmith report <id-prefix>      print the markdown deliverable
    jobsmith outputs <id-prefix>     list the files the job produced
    jobsmith cancel <id-prefix>      cancel a running job

Every command except `serve` is a CLIENT: it talks to a daemon when one is
running (so jobs outlive the command that launched them, and any other
command can list or cancel them), and otherwise runs the agent embedded in
the process — convenient for a quick try, but jobs then stop when it exits.
`--local` forces embedded mode, `--url` points at another daemon.

`--agent NAME` picks which agent to run (see jobsmith/agents/); it applies
to the process that owns the engine, so pass it to `serve` when a daemon is
running, not to the client commands talking to it.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from ..agents import agent_names
from .client import DEFAULT_URL, AgentClient, open_client
from .repl import run_repl, show_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jobsmith",
        description="A conversational agent that runs complex tasks as background jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--llm", choices=("anthropic", "openai", "fake"),
                        help="LLM provider (default: auto-detected from API keys)")
    parser.add_argument("--db", metavar="SPEC",
                        help="memory | <file.db> | <postgres DSN>  (default: $JOBSMITH_DB)")
    parser.add_argument("--agent", metavar="NAME", choices=agent_names(),
                        help=f"which agent to run: {', '.join(agent_names())} (default: default)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"daemon URL (default: {DEFAULT_URL})")
    parser.add_argument("--local", action="store_true",
                        help="never use a daemon: run the agent in this process")

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the daemon (HTTP API + SSE)").add_argument(
        "--port", type=int, default=8000)

    chat = sub.add_parser("chat", help="interactive conversation (default command)")
    chat.add_argument("--session", metavar="ID", help="resume this conversation")

    run = sub.add_parser("run", help="launch a job directly, no chat")
    run.add_argument("task", help="what the job should do")
    run.add_argument("--wait", action="store_true", help="block until the job finishes")

    jobs = sub.add_parser("jobs", help="list jobs")
    jobs.add_argument("--status", choices=("queued", "running", "done", "failed", "cancelled"))
    jobs.add_argument("--session", metavar="ID", help="only this conversation's jobs")

    for name, help_text in (("job", "show one job in detail"),
                            ("report", "print a job's markdown report"),
                            ("outputs", "list the files a job produced"),
                            ("cancel", "cancel a job")):
        sub.add_parser(name, help=help_text).add_argument("job_id", metavar="ID-PREFIX")
    return parser


# ---------------------------------------------------------------- commands

async def cmd_chat(client: AgentClient, args) -> int:
    session_id = await client.new_session(args.session)
    print(f"[session {session_id}]")
    await run_repl(client, session_id)
    return 0


async def cmd_run(client: AgentClient, args) -> int:
    # Embedded, the job runs in THIS process: returning immediately would kill
    # it before it starts. Only a daemon can outlive the command.
    wait = args.wait or not client.persistent
    if wait and not args.wait:
        print("no daemon: running the job here — it needs this process to stay alive",
              file=sys.stderr)
    launched = await client.launch_job(args.task)
    job_id = launched["job_id"]
    print(f"job {job_id[:8]} launched ({launched['status']})")
    if not wait:
        print(f"follow it with:  jobsmith job {job_id[:8]}")
        return 0
    while True:
        await asyncio.sleep(0.5)
        job = await client.get_job(job_id)
        if job and job["status"] in ("done", "failed", "cancelled"):
            show_job(job)
            return 0 if job["status"] == "done" else 1


async def cmd_jobs(client: AgentClient, args) -> int:
    jobs = await client.list_jobs(status=args.status, session_id=args.session)
    if not jobs:
        print("no jobs")
        return 0
    for job in jobs:
        steps = len(job.get("step_finished_at") or {})
        print(f"{job['job_id'][:8]}  {job['status']:<9}  {steps} step(s)  {job['query'][:52]!r}")
    return 0


async def cmd_job(client: AgentClient, args) -> int:
    job = await client.resolve_job(args.job_id)
    if job is None:
        print(f"no single job matching {args.job_id!r}")
        return 1
    show_job(job)
    return 0


async def cmd_report(client: AgentClient, args) -> int:
    job = await client.resolve_job(args.job_id)
    report = await client.get_report(job["job_id"]) if job else None
    if report is None:
        print("no report available (is the job done?)")
        return 1
    print(report)
    return 0


async def cmd_outputs(client: AgentClient, args) -> int:
    job = await client.resolve_job(args.job_id)
    if job is None:
        print(f"no single job matching {args.job_id!r}")
        return 1
    outputs = job.get("outputs") or []
    if not outputs:
        print("no output yet (is the job done?)")
        return 1
    for output in outputs:
        title = f"  {output['title']}" if output.get("title") else ""
        print(f"{output['role']:<6} {output['format']:<9} {output['path']}{title}")
    return 0


async def cmd_cancel(client: AgentClient, args) -> int:
    job = await client.resolve_job(args.job_id)
    if job is None:
        print(f"no single job matching {args.job_id!r}")
        return 1
    print(f"{job['job_id'][:8]} -> {(await client.cancel_job(job['job_id']))['status']}")
    return 0


COMMANDS = {
    "chat": cmd_chat, "run": cmd_run, "jobs": cmd_jobs, "job": cmd_job,
    "report": cmd_report, "outputs": cmd_outputs, "cancel": cmd_cancel,
}


# ---------------------------------------------------------------- entrypoints

async def serve(args) -> int:
    """The daemon: it owns the job engine, so jobs survive their client."""
    import uvicorn

    from ..api import create_api
    from ..app.agent import build_app

    app = await build_app(db=args.db, agent=args.agent)
    try:
        config = uvicorn.Config(
            create_api(app.service()), host="127.0.0.1", port=args.port
        )
        await uvicorn.Server(config).serve()
    finally:
        await app.aclose()
    return 0


async def run_command(args) -> int:
    client = await open_client(url=args.url, force_local=args.local,
                               db=args.db, agent=args.agent)
    try:
        return await COMMANDS[args.command](client, args)
    finally:
        await client.aclose()


def main(argv: list[str] | None = None) -> int:
    import os

    from ..app.providers import load_dotenv

    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.llm:
        os.environ["JOBSMITH_LLM"] = args.llm   # read by pick_provider, both stacks
    if args.command is None:            # bare `jobsmith` == `jobsmith chat`
        args = build_parser().parse_args([*(argv or []), "chat"])
    try:
        return asyncio.run(serve(args) if args.command == "serve" else run_command(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
