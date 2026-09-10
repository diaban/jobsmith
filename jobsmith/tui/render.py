"""Everything the TUI draws, as strings — widgets are assembled elsewhere.

Two reasons this is its own module. It is testable without a terminal: a
plan drawn wrong is a string comparison here, not a screenshot. And it is
where the **colour vocabulary** is declared once, so that no widget invents a
variable name: the constants below are standard Textual theme roles, which is
what lets any of the 23 themes in the Ctrl+P picker paint this UI instead of
failing to parse (see `themes.py`).

The `$text-muted` / `$foreground-muted` split is not a style: `$text-*` muted
values are `auto NN%` — a contrast computed against whatever background they
land on. A stylesheet knows that background and composites correctly; content
markup does not, and drops the alpha, so `[$text-disabled]` renders pure white
on a dark theme. Everything in this module travels as markup, so it uses the
concrete `$foreground-*` pair. CSS in `app.py` keeps the `$text-*` names.

Status glyphs appear in the job list and nowhere else. `●` `◐` `○` are
East-Asian-Ambiguous width: a terminal that renders them double-wide shifts
every character under them, which the plan drawing cannot survive and a list
of rows does not care about. So the DAG carries state in colour alone, and
the list — which is also the one place a colour-blind reader has nothing else
to go on — carries both.
"""
from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import Any

from textual.markup import escape

from ..core.state import plan_depths
from ..core.usage import Usage
from ..jobs.report import format_cost, format_usage

# The roles, once. Nothing below names a colour, and nothing invents a variable.
RUNNING, DONE, FAILED = "$text-accent", "$text-success", "$text-error"
CANCELLED, QUEUED = "$foreground-muted", "$foreground-disabled"
CHROME, DIM = "$text-primary", "$foreground-muted"
RULE, ATTENTION = "$foreground-disabled", "$text-warning"

ROLE = {"running": RUNNING, "done": DONE, "failed": FAILED,
        "cancelled": CANCELLED, "queued": QUEUED}

# Every role this module puts into markup, so a test can ask each theme
# whether it defines them. It has to ask: an undefined variable in a
# *stylesheet* raises at parse time and the app does not start, but in
# content markup it is silently ignored and the text renders unstyled. Both
# `ansi-dark` and `ansi-light` really do lack one of these — see the note in
# `tests/test_tui.py` — and nothing would have said so.
MARKUP_ROLES = (RUNNING, DONE, FAILED, CANCELLED, QUEUED, CHROME, DIM, RULE, ATTENTION)
GLYPH = {"running": "◐", "done": "●", "failed": "✕", "cancelled": "⊘", "queued": "○"}

NONE = "—"          # "nothing to show here", everywhere, so a column reads evenly

# What a tool call is called in front of a human. `chat/runner.py` reports the
# tool's real name and refuses to word it, precisely so that the REPL and this
# can word it differently — the REPL narrates a line at a time, this one has a
# status line that is overwritten, so it says what is happening rather than
# what has happened.
TOOL_ACTIVITY = {
    "launch_job": "sizing up a background job",
    "job_status": "checking on a job",
    "list_my_jobs": "looking up your jobs",
    "cancel_job": "cancelling a job",
}


def tool_activity(name: str) -> str:
    """Readable prose for a tool name; an unmapped tool still says something."""
    return TOOL_ACTIVITY.get(name, f"running {name}")


def status_role(status: str) -> str:
    """The theme role a job or step status is painted in."""
    return ROLE.get(status, DIM)


# ------------------------------------------------------------------- times

def _at(timestamp: str | None) -> datetime | None:
    try:
        return datetime.fromisoformat(timestamp) if timestamp else None
    except ValueError:
        return None


def duration(seconds: float) -> str:
    """A span a human reads at a glance: `0.8s`, `12s`, `3m 04s`, `1h 02m`."""
    if seconds < 0:
        return NONE
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds) // 60}m {int(seconds) % 60:02d}s"
    return f"{int(seconds) // 3600}h {(int(seconds) % 3600) // 60:02d}m"


def span(start: str | None, end: str | None) -> str:
    """`end - start` as a duration, or `—` when either end is unknown."""
    first, last = _at(start), _at(end)
    if first is None or last is None:
        return NONE
    return duration((last - first).total_seconds())


def ellipsis(text: str, width: int) -> str:
    """Trim to `width` cells, keeping the truncation visible."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: max(width - 1, 0)] + "…"


# -------------------------------------------------------------- the job list

def job_row(job: dict[str, Any], width: int = 28) -> str:
    """One job in the list: glyph, id, request, then status and step count.

    Two lines, because `list_jobs` answers with *summaries* — no plan, so
    "2 of 4 steps" is not knowable here and the row says what it does know.
    The detail pane is what loads the plan.
    """
    status = str(job.get("status", "queued"))
    steps = len(job.get("step_finished_at") or {})
    query = escape(ellipsis(str(job.get("query", "")), width))
    job_id = str(job.get("job_id", ""))[:8]
    identity = "$foreground" if status == "running" else DIM
    head = f"[{status_role(status)}]{GLYPH.get(status, '·')}[/] [{identity}]{job_id}[/]  {query}"
    tail = f"  [{DIM}]{status:<11}{steps} step(s)[/]"
    return (f"[b]{head}[/b]\n{tail}") if status == "running" else f"{head}\n{tail}"


# ------------------------------------------------------------- the job detail

def step_states(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Each planned step with the state the record implies, in plan order.

    Derived, never stored: the engine records when a step *finished*
    (`step_finished_at`) and what it produced (`results`), and this reads a
    status out of the two. A step that has not finished is `queued` unless
    the job is running and every dependency has landed, which is the one case
    where it must be on the executor's current wave.

    `took` is the same kind of derivation and is honest about being one: with
    no start timestamp anywhere in the record, it is the window between the
    step's dependencies landing (or the job starting) and the step landing —
    an upper bound on the step's own time, not a measurement of it.
    """
    plan = job.get("plan") or {}
    finished: dict[str, str] = job.get("step_finished_at") or {}
    results: dict[str, Any] = job.get("results") or {}
    job_status = str(job.get("status", "queued"))
    rows: list[dict[str, Any]] = []
    for step in plan.get("steps", []):
        name = step["capability"]
        deps = list(step.get("depends_on") or [])
        result = results.get(name) or {}
        if name in finished:
            status = "done" if result.get("ok") else "failed"
        elif job_status == "running" and all(dep in finished for dep in deps):
            status = "running"
        else:
            status = "queued"
        started = max((finished[dep] for dep in deps if dep in finished),
                      default=str(job.get("created_at") or ""))
        rows.append({
            "capability": name,
            "depends_on": deps,
            "status": status,
            "took": span(started, finished.get(name)),
            "usage": Usage.from_dict((result.get("meta") or {}).get("usage")),
            "error": result.get("error") or "",
        })
    return rows


STEP_LABEL = {"done": "ok", "failed": "failed", "running": "running", "queued": "queued"}


def steps_table(job: dict[str, Any]) -> str:
    """The plan as a table: what each step did, how long, and what it cost."""
    rows = step_states(job)
    if not rows:
        return f"[{DIM}]no plan yet[/]"
    name_width = max(12, *(len(row["capability"]) for row in rows)) + 1
    head = (f"{'step':<{name_width}}{'status':<9}{'took':>8}"
            f"{'tokens':>10}{'cost':>11}")
    out = [f"[{DIM}]{head}[/]", f"[{RULE}]{'─' * len(head)}[/]"]
    for row in rows:
        usage: Usage = row["usage"]
        tokens = f"{usage.total_tokens:,}" if usage else NONE
        cost = format_cost(usage) if usage else ""
        body = "$foreground" if row["status"] == "done" else DIM
        out.append(
            f"{row['capability']:<{name_width}}"
            f"[{status_role(row['status'])}]{STEP_LABEL[row['status']]:<9}[/]"
            f"[{body}]{row['took']:>8}{tokens:>10}[/][{DIM}]{cost or NONE:>11}[/]"
        )
    return "\n".join(out)


# The box character for a cell is chosen from every direction that reaches it,
# which is what makes a crossing a junction rather than whichever segment was
# drawn last.
_BOX = {
    frozenset("EW"): "─", frozenset("NS"): "│", frozenset("SE"): "┌",
    frozenset("SW"): "┐", frozenset("NE"): "└", frozenset("NW"): "┘",
    frozenset("NSE"): "├", frozenset("NSW"): "┤", frozenset("SEW"): "┬",
    frozenset("NEW"): "┴", frozenset("NSEW"): "┼", frozenset("E"): "─",
    frozenset("W"): "─", frozenset("N"): "│", frozenset("S"): "│",
}
# Space between a column's widest name and the next column: one cell of stub,
# one vertical per member of the column (so two edges leaving the same column
# never share a line), and two cells for the arriving edge to turn in.
_TRUNK_PAD = 1
_ARRIVAL_PAD = 2


def dag(job: dict[str, Any]) -> str:
    """The plan drawn on a character grid: waves as columns, edges between.

    The columns are `core.state.plan_depths` — the same longest-path layout
    the HTML deliverable uses, so the two drawings of one plan agree. Nodes
    are placed first; then each edge accumulates a *set of directions* per
    cell it crosses, and the box character is picked once at the end.

    Two things the accumulation cannot decide on its own, because it only
    answers "what meets here" and these are questions of routing:

    * **Each source gets its own vertical.** Two steps in one column shared a
      trunk whenever their names were the same length, and their two edges
      then merged into a single line that read as one edge going somewhere it
      did not.
    * **An edge spanning more than one wave detours through the row below its
      target.** Drawn straight it crossed the *names* of the steps in
      between — and since a name is never overwritten, it came out looking
      like it entered one step and left the other. Names sit on even rows
      only, so an odd row is always free.

    What is left, named rather than half-fixed: an edge *arriving* at a step
    has to cross the band of verticals leaving that step's own column, so two
    edges that genuinely cross there merge into one junction (`┴`) and the
    reader cannot tell which side continues. Removing that needs lane
    routing — a track per edge — which is a layout algorithm, not a guard. It
    cannot arise in the shape a plan usually has (a fan-out and a fan-in,
    drawn correctly); it needs two steps of one wave whose edges cross.

    Status is colour only. See the module docstring: the glyphs that would
    carry it are ambiguous-width, and one double-wide cell shifts every
    character under it.
    """
    rows = step_states(job)
    if not rows:
        return f"[{DIM}]no plan yet[/]"
    depth = plan_depths((row["capability"], row["depends_on"]) for row in rows)

    columns: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        columns.setdefault(depth[row["capability"]], []).append(row)

    column_x: dict[int, int] = {}
    trunk_x: dict[str, int] = {}        # capability -> the vertical its edges leave on
    x = 0
    for column in sorted(columns):
        members = columns[column]
        widest = max(len(row["capability"]) for row in members)
        column_x[column] = x
        for index, row in enumerate(members):
            trunk_x[row["capability"]] = x + widest + _TRUNK_PAD + index
        x += widest + _TRUNK_PAD + len(members) + _ARRIVAL_PAD
    width = x

    char: dict[tuple[int, int], str] = {}
    tag: dict[tuple[int, int], str] = {}
    dirs: dict[tuple[int, int], set[str]] = {}
    place: dict[str, tuple[int, int, int]] = {}     # name -> (x, y, length)

    def write(text: str, cx: int, cy: int, style: str) -> None:
        for offset, character in enumerate(text):
            char[(cy, cx + offset)] = character
            tag[(cy, cx + offset)] = style

    def link(cx: int, cy: int, *directions: str) -> None:
        dirs.setdefault((cy, cx), set()).update(directions)

    def hline(cy: int, x0: int, x1: int) -> None:
        for cx in range(x0, x1 + 1):
            link(cx, cy, "E", "W")

    def vline(cx: int, y0: int, y1: int) -> None:
        """Turns at both ends included; the caller adds where it goes next."""
        if y0 == y1:
            return
        step = 1 if y1 > y0 else -1
        link(cx, y0, "S" if step > 0 else "N")
        for cy in range(y0 + step, y1, step):
            link(cx, cy, "N", "S")
        link(cx, y1, "N" if step > 0 else "S")

    for column, members in columns.items():
        for index, row in enumerate(members):
            name, cy = row["capability"], index * 2
            write(name, column_x[column], cy, status_role(row["status"]))
            place[name] = (column_x[column], cy, len(name))

    for row in rows:
        target = row["capability"]
        for dep in row["depends_on"]:
            if dep not in place:
                continue
            ax, ay, alength = place[dep]
            bx, by, _ = place[target]
            trunk = trunk_x[dep]
            hline(ay, ax + alength, trunk - 1)      # stub from the name
            link(trunk, ay, "W")
            if depth[target] - depth[dep] > 1:
                # Spans a wave: run below the target row, where no name is.
                detour = by + 1
                vline(trunk, ay, detour)
                link(trunk, detour, "E")
                hline(detour, trunk + 1, bx - 2)
                link(bx - 1, detour, "W")
                vline(bx - 1, detour, by)
                link(bx - 1, by, "E")
            else:
                vline(trunk, ay, by)
                link(trunk, by, "E")
                hline(by, trunk + 1, bx - 1)

    for (cy, cx), directions in dirs.items():
        if (cy, cx) not in char:                   # never draw over a name
            char[(cy, cx)] = _BOX[frozenset(directions)]
            tag[(cy, cx)] = RULE

    height = max(cy for cy, _ in char) + 1
    lines: list[str] = []
    for cy in range(height):
        out: list[str] = []
        current: str | None = None
        for cx in range(width):
            style = tag.get((cy, cx))
            if style != current:
                out.append("[/]" if current else "")
                out.append(f"[{style}]" if style else "")
                current = style
            out.append(char.get((cy, cx), " "))
        if current:
            out.append("[/]")
        lines.append("".join(out).rstrip())
    return "\n".join(lines)


def outputs_block(
    outputs: list[dict[str, Any]], *, missing: Collection[str] = (), where: str = ""
) -> str:
    """The files the job produced, deliverables first — the order it records.

    Fed from `list_outputs`, not from `get_job`: `Job.to_dict()` is
    `dataclasses.asdict`, which serialises `JobOutput`'s fields and **drops
    `name`, because it is a property**. Only `list_outputs` puts it back, and
    reading that key off the job rendered a blank where every filename should
    have been — with the layout snapshot then freezing the blank as correct.

    The path is shown, and `where` says whose disk it is on. That is the port
    being taken at its word: `find_output` answers with a locator on the
    machine that RAN the job, which is this one only when the service is
    embedded. A daemon's path is true and unopenable, so the pane prints it
    as a location and names the download route rather than implying a file
    the reader can reach. `missing` is the other half of the same promise —
    a file deleted since the job finished is said to be gone rather than
    drawn as a path to nothing.
    """
    if not outputs:
        return f"[{DIM}]no file yet[/]"
    lines = [f"[{DIM}]files[/]"]
    for output in outputs:
        name = str(output.get("name") or Path(str(output.get("path") or "")).name)
        facts = [str(output.get("role") or ""), str(output.get("format") or "")]
        if output.get("produced_by"):
            facts.append(f"from {output['produced_by']}")
        if output.get("title"):
            facts.append(str(output["title"]))
        gone = name in missing
        lines.append(
            f"[{FAILED if gone else CHROME}]▸[/] [b]{escape(name)}[/b]"
            f"  [{DIM}]{escape(' · '.join(fact for fact in facts if fact))}[/]"
        )
        path = escape(str(output.get("path") or NONE))
        lines.append(f"  [{FAILED}]gone from disk[/] [{DIM}]{path}[/]" if gone
                     else f"  [{DIM}]{path}[/]")
    if where:
        lines.append(f"[{DIM}]{escape(where)}[/]")
    return "\n".join(lines)


def where_files_are(mode: str, job_id: str) -> str:
    """One line under the files saying whose disk those paths are on.

    Not decoration: with a daemon backing they are the daemon's, and a UI
    printing them without saying so would be offering the reader a file it
    cannot open. `mode` is on the port for exactly this kind of question.
    """
    if mode == "embedded":
        return "on this machine"
    return ("on the machine running the daemon — fetch one with "
            f"GET /jobs/{job_id}/outputs/<name>")


def job_headline(job: dict[str, Any]) -> str:
    return f"[b]{escape(ellipsis(str(job.get('query', '')), 100))}[/b]"


def job_meta(job: dict[str, Any]) -> str:
    """One line of provenance under the headline: who, what state, what spend."""
    status = str(job.get("status", "queued"))
    rows = step_states(job)
    done = sum(1 for row in rows if row["status"] in ("done", "failed"))
    progress = f"{done} of {len(rows)} steps" if rows else "no plan"
    parts = [str(job.get("job_id", ""))[:8], status, progress,
             span(job.get("created_at"), job.get("updated_at"))]
    usage = Usage.from_dict(job.get("usage"))
    line = f"[{DIM}]{' · '.join(p for p in parts if p and p != NONE)}[/]"
    return f"{line}\n[{DIM}]{escape(format_usage(usage))}[/]" if usage else line
