"""Interactive chat loop, written against the client interface.

Identical experience whether it is backed by a daemon or by an embedded
agent — only the lifetime of the jobs differs (see cli/client.py).

Commands:
  <any text>        chat (the agent may propose launching a job — approve y/N)
  /jobs             list jobs
  /job <id-prefix>  show a job's plan, artifacts and answer
  /report <id-pfx>  print a finished job's report (text formats)
  /bg <any text>    bypass the chat: run that query as a job directly
  /image <key>      attach an image input to the NEXT /bg job
  /cancel <id-pfx>  cancel a job
  /resume <id-pfx>  restart a stopped job from its checkpoint
  /quit             exit
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

from ..core.usage import Usage
from ..jobs.report import deliverable_filenames, format_step_usage, format_usage
from ..service import TERMINAL_EVENTS, BinaryDeliverable, ChatStreamError, ServiceUnavailable
from .client import AgentClient

BANNER = "\n".join(
    line for line in (__doc__ or "").splitlines() if line.startswith(("  ", "Commands"))
)


# What a tool call is called in front of a human. The event carries the tool's
# real name (`launch_job`); this is the presentation layer, so this is where it
# becomes something worth reading — the same reason REPORT_MEDIA_TYPES lives in
# the HTTP adapter and not in jobs/report.py. A TUI will word these its own way,
# and neither wording belongs in chat/runner.py.
TOOL_ACTIVITY = {
    "launch_job": "sizing up a background job",
    "job_status": "checking on a job",
    "list_my_jobs": "looking up your jobs",
    "cancel_job": "cancelling a job",
}


def tool_activity(name: str) -> str:
    """Readable prose for a tool name; an unmapped tool still says something."""
    return TOOL_ACTIVITY.get(name, f"running {name}")


class TurnPrinter:
    """Renders a streamed turn: the answer on stdout, the activity on stderr.

    The split is this layer's standing rule (`cli/client.py`): stdout is the
    conversation, so piping `jobsmith chat` gives the answers and nothing
    else, while "what it is doing right now" — which is over the moment it is
    read — goes where every other diagnostic goes.

    Tokens are written as they arrive, so `flush` is not optional: a line
    still being written has no newline to trigger one, and an unflushed
    answer is exactly the silence this replaces.
    """

    def __init__(self, indent: str = "  "):
        self.indent = indent
        self._writing = False        # a line of answer is open, unterminated

    def show(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "token":
            self._write(event.get("text") or "")
        elif kind == "tool_started":
            self._note(f"… {tool_activity(event.get('name') or '')}")
        elif kind == "tool_finished":
            self._note(f"✓ {tool_activity(event.get('name') or '')}")

    def end(self) -> None:
        """Close the answer's line. The terminal event restates the reply the
        tokens already delivered, so printing it would print it twice."""
        if self._writing:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._writing = False

    def _write(self, text: str) -> None:
        if not text:
            return
        if not self._writing:
            sys.stdout.write(self.indent)
            self._writing = True
        sys.stdout.write(text.replace("\n", "\n" + self.indent))
        sys.stdout.flush()

    def _note(self, text: str) -> None:
        self.end()                   # never interleave a note into a sentence
        print(f"{self.indent}{text}", file=sys.stderr, flush=True)


async def render_turn(events: AsyncIterator[dict], printer: TurnPrinter) -> dict:
    """Show a turn as it happens and return the reply it ended on.

    The port's `send` drains the same stream for its terminal; here the
    events are also shown on the way past, which is the only difference
    between a turn you wait for and a turn you watch.
    """
    terminal: dict = {}
    async for event in events:
        printer.show(event)
        if event.get("type") in TERMINAL_EVENTS:
            terminal = event
    printer.end()
    return terminal


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

        # One `try`, and one exception: the port narrowed "the backing is not
        # there" to a single name (`service.py`) precisely so a front-end
        # could answer it without a broad `except` swallowing its own bugs.
        # A KeyError from this code still ends the REPL with a traceback,
        # which is what a defect deserves; a daemon that went away is a fact
        # about the world, and the conversation lives in the checkpointer, so
        # the loop says it and stays open.
        try:
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
                    try:
                        report = await client.get_report(job["job_id"])
                        print(report or "  no report yet (is the job done?)")
                    except BinaryDeliverable as refused:
                        print(f"  {refused}")
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
                printer = TurnPrinter()
                try:
                    reply = await render_turn(client.stream(session_id, line), printer)
                    # human-in-the-loop: the agent proposes a job, you approve or not
                    while reply.get("type") == "proposal":
                        print("\n  the agent proposes a background job:")
                        print(f"    task     : {reply.get('query')}")
                        print(f"    approach : {reply.get('rationale')}")
                        # the files it would be allowed to open: approving the job
                        # is approving this list, so it is never left unsaid
                        if sources := reply.get("sources"):
                            print(f"    reads    : {', '.join(sources)}")
                        # ...and what it would leave behind: the name is the
                        # thing they will look for on disk afterwards (#55)
                        if title := reply.get("document_title"):
                            print(f"    titled   : {title}")
                        if files := deliverable_filenames(
                                str(reply.get("document_name") or ""),
                                reply.get("formats") or []):
                            print(f"    writes   : {', '.join(files)}")
                        elif formats := reply.get("formats"):
                            print(f"    writes   : {', '.join(formats)}")
                        answer = await loop.run_in_executor(None, input, "  launch it? [y/N] ")
                        approved = answer.strip().lower() in ("y", "yes", "o", "oui")
                        reply = await render_turn(
                            client.stream_approval(session_id, approved), printer)
                except ChatStreamError as cut_short:
                    # The turn is already half-printed, so silence would leave a
                    # truncated answer looking finished — which is the one thing
                    # the no-drop rule exists to prevent. Say it, and keep the
                    # session usable; the conversation is in the checkpointer.
                    printer.end()
                    print(f"  [the reply was cut short: {cut_short}]", file=sys.stderr)
        except ServiceUnavailable as gone:
            print(f"  [{gone}]", file=sys.stderr)

    print("bye")
