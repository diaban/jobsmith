"""The terminal UI: what it draws, and what it does with a turn. → 0048

* **layout** — an SVG snapshot, in one theme; the facts a pane states are
  asserted by name before the picture, never left to the snapshot.
* **colour** — representative themes mounted and rendered for real (→ 0109).
* **behaviour** — asserted on what the screen shows (prompt placeholder,
  activity line, tab bar, bubbles, cards), never on the app's private state.
"""
from __future__ import annotations

import asyncio
import dataclasses
import re
import sys
import time
from typing import Any

import pytest
from conftest import FakeLLM, plan_json
from langchain_core.messages import AIMessage
from support import (
    Gate,
    SlowEcho,
    launch_call,
    make_manager,
    planned_manager,
    planned_service,
    service_over,
)
from textual.content import Content
from textual.widgets import Input, ListView, Static

from jobsmith.core.usage import Usage
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.service import AgentService, ServiceUnavailable
from jobsmith.tui import MISSING, TuiUnavailable
from jobsmith.tui.app import (
    LIVE_LOST,
    UNREACHABLE,
    Bubble,
    JobNoticeCard,
    JobsmithApp,
    ProposalCard,
)
from jobsmith.tui.render import (
    MARKUP_ROLES,
    NONE,
    dag,
    job_row,
    outputs_block,
    step_states,
    steps_table,
    where_files_are,
)
from jobsmith.tui.themes import DEFAULT_THEME, THEMES, pick_theme

SESSION = "s3f9c21e"
T0 = "2026-01-05T10:00:00+00:00"
T1 = "2026-01-05T10:00:04+00:00"
T2 = "2026-01-05T10:00:19+00:00"
T3 = "2026-01-05T10:00:31+00:00"

PLAN = {
    "steps": [
        {"capability": "documents", "depends_on": []},
        {"capability": "research", "depends_on": ["documents"]},
        {"capability": "web_search", "depends_on": ["documents"]},
        {"capability": "analysis", "depends_on": ["research", "web_search"]},
        {"capability": "critique", "depends_on": ["analysis"]},
    ],
    "rationale": "ground it, then read it two ways, then check the reading",
}


def spend(tokens: int, cost: float) -> dict[str, Any]:
    return Usage(input_tokens=tokens, calls=1, cost_usd=cost, models=("fake-1",)).to_dict()


def canned_jobs() -> list[Job]:
    """Five jobs, one per status — real records with frozen timestamps."""
    running = Job(
        job_id="9b7e3011-0000-4000-8000-000000000001",
        status=JobStatus.RUNNING,
        query="how well is sixel supported across terminal emulators?",
        created_at=T0, updated_at=T2, plan=PLAN,
        step_finished_at={"documents": T1, "research": T2, "web_search": T2},
        results={
            "documents": {"ok": True, "data": {}, "meta": {"usage": spend(3104, 0.009)}},
            "research": {"ok": True, "data": {}, "meta": {"usage": spend(24077, 0.071)}},
            "web_search": {"ok": True, "data": {}, "meta": {"usage": spend(8812, 0.026)}},
        },
        usage=spend(35993, 0.106),
        outputs=[JobOutput(path="/tmp/a/sixel-matrix.svg", format="svg", role="annex",
                           title="support matrix", produced_by="web_search")],
    )
    done = Job(
        job_id="4f2a1c22-0000-4000-8000-000000000002",
        status=JobStatus.DONE, query="compare the vendor proposals",
        created_at=T0, updated_at=T3, plan=PLAN,
        step_finished_at={s["capability"]: T3 for s in PLAN["steps"]},
        results={s["capability"]: {"ok": True, "data": {}} for s in PLAN["steps"]},
        final_answer="The second proposal is cheaper and shorter to exit.",
        outputs=[JobOutput(path="/tmp/a/4f2a1c22.md", format="markdown", role="main")],
    )
    failed = Job(
        job_id="1d5c8833-0000-4000-8000-000000000003",
        status=JobStatus.FAILED, query="summarise the Q3 board deck",
        created_at=T0, updated_at=T1, plan=PLAN,
        step_finished_at={"documents": T1},
        results={"documents": {"ok": False, "error": "no such directory"}},
        error="documents: no such directory",
    )
    cancelled = Job(job_id="c04ab244-0000-4000-8000-000000000004",
                    status=JobStatus.CANCELLED, query="migration plan, events backend",
                    created_at=T0, updated_at=T1)
    queued = Job(job_id="7e19f455-0000-4000-8000-000000000005",
                 status=JobStatus.QUEUED, query="audit the report writers",
                 created_at=T0, updated_at=T0)
    return [running, done, failed, cancelled, queued]


class CannedService(AgentService):
    """An `AgentService` whose answers do not move — so a snapshot can.

    It answers `list_jobs` with `summary()` and `get_job` with `to_dict()`,
    the two shapes the real service returns, so the panes are fed exactly
    what they are fed in production.
    """

    mode = "embedded"
    persistent = False

    def __init__(self, jobs: list[Job] | None = None, events: list[dict] | None = None,
                 stream_gate: asyncio.Event | None = None,
                 list_gate: asyncio.Event | None = None,
                 missing_files: set[str] | None = None):
        self.jobs = jobs if jobs is not None else canned_jobs()
        self.events = events or []
        self.cancelled: list[str] = []
        # What `find_output` answers None for: a file recorded by the job and
        # no longer on the disk that ran it.
        self.missing_files = missing_files or set()
        self.queues: list[asyncio.Queue] = []
        self.released: list[asyncio.Queue] = []
        # A gate, when given, is awaited inside the call it names — so a test
        # can hold the UI inside work that is genuinely in flight instead of
        # racing it. Two of them, because holding a turn open must not also
        # hold the job refresh the mount performs.
        self.stream_gate = stream_gate
        self.list_gate = list_gate
        self.listed = 0

    async def new_session(self, session_id: str | None = None) -> str:
        return session_id or SESSION

    async def stream(self, session_id: str, text: str):
        for index, event in enumerate(self.events):
            if self.stream_gate is not None and index == 1:
                await self.stream_gate.wait()
            yield event

    async def stream_approval(self, session_id: str, approved: bool):
        yield {"type": "message", "content": "ok"}

    async def launch_job(self, query, *, session_id=None, inputs=None) -> dict:
        return {"job_id": "new", "status": "queued"}

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
        self.listed += 1
        if self.list_gate is not None:
            await self.list_gate.wait()
        return [job.summary() | {"job_id": job.job_id} for job in self.jobs]

    async def get_job(self, job_id: str) -> dict | None:
        return next((j.to_dict() for j in self.jobs if j.job_id == job_id), None)

    async def cancel_job(self, job_id: str) -> dict:
        self.cancelled.append(job_id)
        return {"job_id": job_id, "status": "cancelled"}

    async def resume_job(self, job_id: str) -> dict:
        return {"job_id": job_id, "status": "queued"}

    async def get_report(self, job_id: str) -> str | None:
        return None

    async def list_outputs(self, job_id: str) -> list[dict] | None:
        """What the real service answers: `asdict` plus the `name` property."""
        job = next((j for j in self.jobs if j.job_id == job_id), None)
        if job is None:
            return None
        return [dataclasses.asdict(o) | {"name": o.name} for o in job.outputs]

    async def find_output(self, job_id: str, name: str) -> str | None:
        outputs = await self.list_outputs(job_id) or []
        if name in self.missing_files:
            return None
        return next((o["path"] for o in outputs if o["name"] == name), None)

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.queues.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.released.append(queue)


# The one measured gap in the standard vocabulary: the two ansi themes carry
# the 16 terminal colours and generate a smaller variable set than the other
# 23. Written as what each theme is allowed to be missing, so the day Textual
# defines it this test keeps passing rather than inverting.
ANSI_GAP = {"ansi-dark": {"foreground-disabled"}, "ansi-light": {"foreground-disabled"}}

PROPOSAL_TURN = [
    {"type": "tool_started", "name": "launch_job"},
    {"type": "token", "text": "That is worth doing properly rather than from memory. "},
    {"type": "token", "text": "I would run it as a background job."},
    {"type": "proposal",
     "query": "survey sixel support across the major terminal emulators",
     "rationale": "it needs the web and several sources cross-checked"},
]


def canned_app(**kwargs: Any) -> JobsmithApp:
    """The app on a service whose answers do not move, so a snapshot can."""
    service = kwargs.pop("service", None) or CannedService(events=PROPOSAL_TURN)
    return JobsmithApp(service, SESSION, **kwargs)


async def settle(pilot: Any) -> None:
    """Let the workers that load the job list and its detail finish."""
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


async def say(pilot: Any, text: str) -> None:
    """Type a message, send it, and let the turn finish."""
    await pilot.press(*text)
    await pilot.press("enter")
    await settle(pilot)


async def until(pilot: Any, condition: Any, timeout: float = 5.0) -> bool:
    """Let the UI breathe until it says something, or give up saying so."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if condition():
            return True
        await asyncio.sleep(0.02)
    return False


# What the screen shows — the only state these tests read.
IDLE, PROPOSING = "message the agent…", "launch it? [y/N]"


def placeholder(app: JobsmithApp) -> str:
    return app.query_one("#prompt", Input).placeholder


def activity(app: JobsmithApp) -> str:
    return str(app.query_one("#activity", Static).content)


def tabs(app: JobsmithApp) -> str:
    return str(app.query_one("#tabs", Static).content)


def job_rows(app: JobsmithApp) -> int:
    return len(app.query_one("#job-list", ListView))


# ------------------------------------------------------------------- layout


def test_the_jobs_screen_looks_like_this(snap_compare):
    """Job list on the left, the plan, steps and files on the right."""
    async def before(pilot):
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)
        # before the picture, so `--snapshot-update` cannot re-freeze a blank field
        outputs = str(pilot.app.query_one("#detail-outputs", Static).content)
        assert "sixel-matrix.svg" in outputs, outputs
        assert "/tmp/a/sixel-matrix.svg" in outputs, "the file is not located"
        assert "from web_search" in outputs, "an annex does not say which step made it"
        assert "on this machine" in outputs, "the pane does not say whose disk that is"

    assert snap_compare(canned_app(), terminal_size=(126, 38), run_before=before)


@pytest.mark.slow  # a full turn through a streamed proposal, snapshotted
def test_the_chat_screen_looks_like_this(snap_compare):
    """A streamed turn that ended on a proposal, waiting to be answered."""
    async def before(pilot):
        await settle(pilot)
        await say(pilot, "look into sixel support")

    assert snap_compare(canned_app(), terminal_size=(126, 38), run_before=before)


# ------------------------------------------------------------------- colour


async def test_every_theme_is_registered():
    """Ours and Textual's built-ins are all in the picker (one mount, no render)."""
    app = canned_app()
    async with app.run_test():
        themes = list(app.available_themes)
    assert set(THEMES) <= set(themes), "ours are not in the picker"
    assert len(themes) >= 20, "the built-in themes went missing"


# Ours (the only ones a typo in themes.py can break), the two ansi themes
# (reduced variable set) and one built-in for the 20 that share textual-dark's. → 0109
REPRESENTATIVE_THEMES = [*THEMES, "ansi-dark", "ansi-light", "textual-dark"]


@pytest.mark.slow
@pytest.mark.parametrize("name", REPRESENTATIVE_THEMES)
async def test_every_registered_theme_resolves(name):
    """The app starts under the theme (an undefined CSS variable fails at parse
    time) and every markup role resolves (an undefined one is silently
    dropped) — `ANSI_GAP` is the one measured exception. → 0048"""
    app = canned_app(theme=name)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press(*"hello")
        await pilot.press("enter")           # a turn, so the proposal card is up
        await settle(pilot)
        unresolved = {role.lstrip("$") for role in MARKUP_ROLES} - set(
            app.get_css_variables())
        assert unresolved <= ANSI_GAP.get(name, set()), \
            f"{name} does not define {sorted(unresolved)} — that markup renders unstyled"
        assert app.export_screenshot(), f"{name} rendered nothing"
        await pilot.press("f3")              # the other pane, same question
        await settle(pilot)
        assert app.export_screenshot(), f"{name} rendered nothing on the jobs pane"


def test_pick_theme_follows_the_house_precedence(monkeypatch):
    monkeypatch.delenv("JOBSMITH_THEME", raising=False)
    assert pick_theme() == DEFAULT_THEME
    monkeypatch.setenv("JOBSMITH_THEME", "tide-dark")
    assert pick_theme() == "tide-dark"
    monkeypatch.setattr("sys.argv", ["jobsmith", "ui", "--theme=ember-light"])
    assert pick_theme() == "ember-light"
    assert pick_theme("tide-light") == "tide-light"


async def test_an_unknown_theme_name_does_not_take_the_app_down():
    """Assigning a name nothing defines raises inside Textual, so it is
    checked before it is assigned and the run continues on the default."""
    app = canned_app(theme="no-such-theme")
    async with app.run_test(size=(80, 24)) as pilot:
        await settle(pilot)
        assert app.theme == DEFAULT_THEME


# ---------------------------------------------------------------- the panes


async def test_the_detail_pane_follows_the_highlighted_row():
    app = canned_app()
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)
        first = str(app.query_one("#detail-title", Static).content)
        await pilot.press("down")
        await settle(pilot)
        assert str(app.query_one("#detail-title", Static).content) != first
        assert "critique" in str(app.query_one("#detail-steps", Static).content)


async def test_cancelling_asks_twice_and_only_where_the_job_is_shown():
    """F8 arms, a second F8 cancels — and only on the jobs pane, where the job
    is shown (the prompt claims the ctrl keys; → 0048)."""
    service = CannedService(events=PROPOSAL_TURN)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f8", "f8")            # chat pane: refused, twice
        await settle(pilot)
        assert service.cancelled == []

        await pilot.press("f3")
        await settle(pilot)
        await pilot.press("f8")                  # arms, and names the job
        await settle(pilot)
        assert service.cancelled == [], "one press was enough to cancel"

        await pilot.press("f8")
        await settle(pilot)
    assert service.cancelled == [service.jobs[0].job_id]


async def test_moving_off_the_row_drops_an_armed_cancel():
    """Armed for a job, then highlighting another: the second press must not
    inherit the first one's intent."""
    service = CannedService(events=PROPOSAL_TURN)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)
        await pilot.press("f8")                  # armed for the first job
        await pilot.press("down")                # ... now looking at another
        await settle(pilot)
        await pilot.press("f8")                  # re-arms, does not fire
        await settle(pilot)
    assert service.cancelled == []


# ------------------------------------------------------- the streamed turn


def make_service(store, checkpointer, tmp_path, responses, *, approval=False):
    manager = make_manager(store, checkpointer, tmp_path)
    return service_over(manager, responses, approval=approval), manager


ANSWER = "A reasonably long answer that no single chunk should ever carry whole."


async def test_the_answer_is_drawn_as_it_arrives(store, checkpointer, tmp_path, monkeypatch):
    """The bubble passes through several states, each a prefix of the next —
    a turn drawn when it is over leaves exactly one."""
    seen: list[str] = []
    original = Bubble.add

    def spy(self: Bubble, text: str) -> None:
        original(self, text)
        seen.append(self.text)

    monkeypatch.setattr(Bubble, "add", spy)
    service, _ = make_service(store, checkpointer, tmp_path, [AIMessage(content=ANSWER)])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await say(pilot, "hello")

    assert len(seen) > 1, "the answer landed in one block: nothing was streamed"
    assert seen[-1] == ANSWER
    assert all(earlier == later[:len(earlier)] for earlier, later in zip(seen, seen[1:], strict=False)), \
        "the bubble did not grow by prefixes"


@pytest.mark.slow  # drives a real job through the graph
async def test_the_job_notice_says_what_will_run_and_how_to_stop_it(
    store, checkpointer, tmp_path
):
    """The nominal path: a card that is read, not answered — query, files,
    document, job id and how to stop it; the prompt stays idle. → 0083"""
    service, manager = make_service(store, checkpointer, tmp_path, [
        launch_call("survey sixel support", "it needs the web",
                    source_files=["/notes/sixel.md"], document_name="sixel",
                    document_title="Sixel support", formats=["markdown"]),
        AIMessage(content="Saved."),
    ])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await say(pilot, "look into sixel")

        (notice,) = app.query(JobNoticeCard)
        shown = str(notice.content)
        assert "survey sixel support" in shown
        assert "/notes/sixel.md" in shown
        assert "sixel.md" in shown                      # what it will write
        assert "Sixel support" in shown                 # ...and its title
        assert "stops it" in shown, "the undo is not offered anywhere"
        assert "builds on" not in shown, "a run that references no job names one"
        assert not app.query(ProposalCard), "the nominal path still asked"
        assert placeholder(app) == IDLE

        (job,) = await manager.list_jobs()
        assert job.job_id[:8] in shown, "the card cannot name the job to stop"
        # the answer the run produced was written into the conversation
        settled = await manager.get_job(job.job_id)
        assert any(settled.final_answer in b.text for b in app.query(Bubble))


@pytest.mark.slow  # drives a real job through the graph, twice over
@pytest.mark.parametrize("approval", [False, True], ids=["notice", "proposal"])
async def test_the_card_names_the_jobs_a_run_builds_on(
    store, checkpointer, tmp_path, approval
):
    """#104: the earlier jobs a run is handed, on the notice and on the
    proposal alike — each by its short id and the start of its query, escaped
    like every other piece of model or user text on the card."""
    manager = make_manager(store, checkpointer, tmp_path)
    first = await manager.create_job("compare [b]both[/b] chairs", session_id="s-tui")
    second = await manager.create_job("price the standing desk", session_id="s-tui")
    service = service_over(manager, [
        launch_call("a one-pager out of both", "several steps",
                    from_jobs=[first.job_id[:8], second.job_id[:8]]),
        AIMessage(content="Saved."),
    ], approval=approval)
    session_id = await service.new_session("s-tui")

    app = JobsmithApp(service, session_id)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await say(pilot, "one-pager out of both")

        (card,) = app.query(ProposalCard if approval else JobNoticeCard)
        # the text as drawn, not the markup: an unescaped `[b]` would be
        # swallowed as a tag here, and the query would read "compare both"
        shown = Content.from_markup(str(card.content)).plain
        assert f"builds on job {first.job_id[:8]} — compare [b]both[/b] chairs" in shown
        assert f"builds on job {second.job_id[:8]} — price the standing desk" in shown


@pytest.mark.slow  # drives a real job through the graph, gated mid-run
async def test_the_plan_is_on_the_activity_line_while_the_task_runs(
    store, checkpointer, tmp_path
):
    """The plan is said on the activity line while the run is held, and never in
    the conversation. → 0086"""

    gate = Gate("web_search")
    manager = planned_manager(store, checkpointer, tmp_path, gate=gate)
    service = planned_service(manager, responses=[launch_call("analyse it", "several steps"),
                                                  AIMessage(content="Done.")])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id)
    async with app.run_test(size=(120, 30)) as pilot:
        await settle(pilot)
        await pilot.press(*"analyse it")
        await pilot.press("enter")
        assert await until(pilot, lambda: "→" in activity(app)), "the plan was never said"
        assert activity(app) == "… running the task: web_search + documents → research → analysis"
        (job,) = await manager.list_jobs()
        assert job.status is JobStatus.RUNNING, "the plan was shown only once it was over"

        gate.open.set()
        await settle(pilot)
        assert app.query(JobNoticeCard), "no job ran; this proves nothing"
        drawn = [str(w.content) for w in app.query(JobNoticeCard)] + [
            b.text for b in app.query(Bubble)]
        assert not any("→" in text for text in drawn), "the plan leaked into the conversation"
        assert activity(app) == "", "the plan outlived the turn on the activity line"


@pytest.mark.slow  # drives a real job through the graph
@pytest.mark.parametrize(("key", "launched"), [("y", ["survey sixel support"]), ("n", [])])
async def test_a_proposal_is_answered_through_the_ui(
    store, checkpointer, tmp_path, key, launched
):
    """The gate path (`$JOBSMITH_APPROVE_JOBS`, → 0083): a card to answer, and
    a job only on `y`."""
    service, manager = make_service(store, checkpointer, tmp_path, [
        launch_call("survey sixel support", "it needs the web"),
        AIMessage(content=ANSWER),
    ], approval=True)
    app = JobsmithApp(service, await service.new_session())
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await say(pilot, "look into sixel")
        (card,) = app.query(ProposalCard)
        assert "survey sixel support" in str(card.content)
        assert placeholder(app) == PROPOSING
        assert not await manager.list_jobs(), "a job was created before the answer"

        await pilot.press(key, "enter")
        await settle(pilot)
        assert placeholder(app) == IDLE

    assert [job.query for job in await manager.list_jobs()] == launched


@pytest.mark.slow  # a full streamed turn, keystroke by keystroke
async def test_a_second_message_cannot_cut_the_turn_being_written():
    """A second Enter during a turn is refused and its text kept; the first turn
    finishes. → 0048"""
    gate = asyncio.Event()
    service = CannedService(events=PROPOSAL_TURN, stream_gate=gate)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        try:
            await pilot.press(*"first")
            await pilot.press("enter")
            await pilot.pause()
            assert activity(app), "the turn is not in flight; the gate did not hold"

            await pilot.press(*"second")
            await pilot.press("enter")
            await pilot.pause()
            assert app.query_one("#prompt", Input).value == "second", "the text was lost"
            assert [b.text for b in app.query(Bubble)][:1] == ["first"], \
                "the second message was sent while a turn was streaming"
        finally:
            gate.set()      # a failed assert must not leave the app mid-turn
        await settle(pilot)
        assert activity(app) == ""
        answer = "".join(e["text"] for e in PROPOSAL_TURN if e["type"] == "token")
        assert answer in [b.text for b in app.query(Bubble)], "the turn did not finish"


@pytest.mark.slow  # a full streamed turn, keystroke by keystroke
async def test_a_message_during_a_proposal_is_not_read_as_a_refusal():
    """Everything that was not an approval used to decline the job AND lose
    the sentence. Only the refusals decline; a real message is refused with
    the text kept, and a bare Enter is the `N` the prompt advertises."""
    service = CannedService(events=PROPOSAL_TURN)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await say(pilot, "look into sixel")
        assert placeholder(app) == PROPOSING

        await say(pilot, "actually narrow it to macOS")
        assert placeholder(app) == PROPOSING, "an ordinary message answered the proposal"
        assert app.query_one("#prompt", Input).value == "actually narrow it to macOS"

        app.query_one("#prompt", Input).value = ""
        await pilot.press("enter")               # bare Enter: the N in [y/N]
        await settle(pilot)
        assert placeholder(app) == IDLE


async def test_a_refresh_in_flight_is_joined_rather_than_cancelled_or_dropped():
    """Events during a refresh neither cancel it nor get lost: one more refresh
    runs after it. → 0048"""
    gate = asyncio.Event()
    service = CannedService(list_gate=gate)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        try:
            await pilot.pause()
            assert service.listed == 1
            for _ in range(4):                    # events arriving while it waits
                app.refresh_jobs()
                await pilot.pause()
            assert service.listed == 1, "an event cancelled the refresh already running"
        finally:
            gate.set()
        await settle(pilot)
        assert service.listed == 2, "the refresh those events asked for was dropped"
        assert len(app.query_one("#job-list", ListView)) == len(service.jobs)


# --------------------------------------------------------------- live updates


@pytest.mark.slow  # a real job with real (slowed) steps, followed live
async def test_the_screen_follows_a_job_while_it_runs(store, checkpointer, tmp_path):
    """Driven only by `subscribe()` — no keystroke, no F5, no poll: the job
    appears, its plan before any step lands, then `done`."""
    # `alpha` is held until the plan has been seen — a 0.4 s sleep made "before
    # any step landed" a window of wall clock that a GC pause closed (#113)
    alpha = Gate("alpha")
    caps = [alpha, SlowEcho("beta", delay=0.4)]
    llm = FakeLLM(
        {"planner": plan_json("alpha", "beta", deps={"beta": ["alpha"]})},
        default="A sufficiently long final answer for the job test.",
    )
    manager = make_manager(store, checkpointer, tmp_path, caps=caps, llm=llm)
    service = service_over(manager, [])

    def dag_now() -> str:
        return str(app.query_one("#detail-dag", Static).content)

    def meta_now() -> str:
        return str(app.query_one("#detail-meta", Static).content)

    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f3")               # the last keystroke of this test
        await settle(pilot)
        assert len(app.query_one("#job-list", ListView)) == 0

        job = await manager.create_job("a chain", formats=["markdown"])
        manager.start_job(job.job_id)

        assert await until(pilot, lambda: len(app.query_one("#job-list", ListView)) == 1), \
            "the job never appeared: nothing is following the stream"
        # Before any step has landed, which is the reason the manager
        # publishes when the plan does: with only step events, a DAG stays
        # "no plan yet" for the whole of the first step.
        assert await until(pilot, lambda: "alpha" in dag_now() and "0 of 2" in meta_now()), \
            "the plan was not on screen until a step had finished"
        alpha.open.set()                      # only now may the first step land
        assert await until(pilot, lambda: "done" in meta_now()), "the end never arrived"

        steps = str(app.query_one("#detail-steps", Static).content)
        assert "alpha" in steps and "beta" in steps
        files = str(app.query_one("#detail-outputs", Static).content)
        assert job.job_id in files, "the deliverable the run produced was never shown"


async def test_a_file_that_is_no_longer_there_is_not_offered():
    """`find_output` probes rather than trusting the record, on both
    backings. A pane that ignored that answer would draw a path to nothing."""
    service = CannedService(missing_files={"sixel-matrix.svg"})
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)
        files = str(app.query_one("#detail-outputs", Static).content)
        assert "gone from disk" in files and "sixel-matrix.svg" in files


async def test_a_stream_that_ends_is_said_on_the_screen_that_took_the_terminal():
    """The end of the event stream (`None` on the queue) is said in the tab bar."""
    service = CannedService()
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        assert service.queues, "nothing subscribed"
        assert LIVE_LOST not in tabs(app)

        service.queues[0].put_nowait(None)           # the daemon went away
        assert await until(pilot, lambda: LIVE_LOST in tabs(app)), "the end went unnoticed"


class GoneService(CannedService):
    """A backing that can be taken away, the way a daemon can.

    Every call goes through `_reached`, so "the daemon is gone" is one flag
    rather than a stub per method — and it raises what the port says it
    raises (`ServiceUnavailable`), which is the whole point: the UI is
    written against the port, not against httpx.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.gone = False

    def _reached(self) -> None:
        if self.gone:
            raise ServiceUnavailable.reaching("http://daemon", ConnectionRefusedError())

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
        self._reached()
        return await super().list_jobs(status=status, session_id=session_id)

    async def get_job(self, job_id: str) -> dict | None:
        self._reached()
        return await super().get_job(job_id)

    async def list_outputs(self, job_id: str) -> list[dict] | None:
        self._reached()
        return await super().list_outputs(job_id)

    async def find_output(self, job_id: str, name: str) -> str | None:
        self._reached()
        return await super().find_output(job_id, name)

    async def cancel_job(self, job_id: str) -> dict:
        self._reached()
        return await super().cancel_job(job_id)

    async def stream(self, session_id: str, text: str):
        self._reached()
        for event in self.events:
            yield event


async def test_a_backing_that_raises_does_not_take_the_screen_down():
    """A daemon gone mid-session: the app stays up, keeps what it read, and the
    tab bar says it cannot reach the agent. → 0064"""
    service = GoneService()
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        rows = job_rows(app)
        assert rows and UNREACHABLE not in tabs(app)

        service.gone = True                       # the daemon was killed
        await pilot.press("f5")
        await settle(pilot)

        assert app.is_running, "the app went down with the backing"
        assert UNREACHABLE in tabs(app), "the screen did not notice it cannot reach the agent"
        assert job_rows(app) == rows, "what it had already read was thrown away"


async def test_a_screen_that_can_reach_the_agent_again_stops_saying_it_cannot():
    """A call that gets through clears "cannot reach"; it does not clear "live
    updates stopped", since nothing resubscribed."""
    service = GoneService()
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        service.queues[0].put_nowait(None)        # the stream ended...
        service.gone = True                       # ...and the daemon is gone
        await pilot.press("f5")
        await settle(pilot)
        assert UNREACHABLE in tabs(app)

        service.gone = False                      # it came back; nothing resubscribed
        await pilot.press("f5")
        await settle(pilot)
        assert UNREACHABLE not in tabs(app)
        assert LIVE_LOST in tabs(app), "a screen that stopped following must keep saying so"


async def test_a_turn_against_a_backing_that_is_gone_is_answered_in_the_conversation():
    """The person is waiting on a sentence, so the tab bar is not where they
    are looking — and a turn that raised used to end the app instead. Said in
    the conversation, and the prompt stays usable."""
    service = GoneService(events=PROPOSAL_TURN)
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        service.gone = True
        await pilot.click("#prompt")
        await say(pilot, "hello")

        assert app.is_running, "the app went down with the backing"
        said = [b.text for b in app.query(Bubble)]
        assert any("cannot reach the agent at http://daemon" in t for t in said), said
        assert activity(app) == "", "the turn was left looking like it is still running"


async def test_the_subscription_is_released_with_the_app():
    """A subscription is a resource — behind a daemon it holds an HTTP stream
    open — so it is unsubscribed, not dropped."""
    service = CannedService()
    app = JobsmithApp(service, SESSION)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        assert len(service.queues) == 1
    assert service.released == service.queues, "the event stream was left open"


# ------------------------------------------------------------- what it draws


def plain(markup: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", markup)


def plan_of(*steps: tuple[str, list[str]]) -> dict[str, Any]:
    """A running job with this plan and nothing finished — enough to draw."""
    return {"status": "running", "created_at": T0, "updated_at": T0,
            "plan": {"steps": [{"capability": c, "depends_on": list(d)} for c, d in steps],
                     "rationale": ""},
            "step_finished_at": {}, "results": {}, "outputs": []}


def test_a_step_is_running_only_when_its_dependencies_have_landed():
    """Derived from the record, which holds finishes and nothing else: a step
    whose dependency has not landed cannot be on the executor's wave."""
    running = canned_jobs()[0].to_dict()
    states = {row["capability"]: row["status"] for row in step_states(running)}
    assert states == {"documents": "done", "research": "done", "web_search": "done",
                      "analysis": "running", "critique": "queued"}

    failed = canned_jobs()[2].to_dict()
    states = {row["capability"]: row["status"] for row in step_states(failed)}
    assert states["documents"] == "failed"
    assert states["analysis"] == "queued", "a stopped job has no running step"


def test_the_plan_is_drawn_in_waves_with_junctions():
    """Two edges meeting in one cell make a junction, not whichever segment
    was drawn last — which is the whole reason directions are accumulated."""
    drawing = plain(dag(canned_jobs()[0].to_dict()))
    rows = drawing.splitlines()

    assert rows[0].startswith("documents─┬──research")
    assert "web_search" in rows[2], "the second wave's members are not stacked"
    # Two junctions on the first row: documents forking, and the web_search →
    # analysis trunk crossing the research → analysis segment. The second one
    # is the whole reason directions are accumulated per cell — drawn segment
    # by segment it would be whichever of the two was written last.
    assert rows[0].count("┬") == 2
    assert rows[2].strip() == "└──web_search──┘"
    # Status is colour only: the list's glyphs must not leak into the plan.
    assert not any(glyph in drawing for glyph in ("●", "◐", "○", "✕", "⊘"))


def test_an_edge_that_spans_a_wave_does_not_run_through_the_step_between():
    """Drawn straight it crossed `beta`'s name, and a name is never painted
    over — so the edge read as entering one step and leaving another. Names
    are on even rows, so the detour goes below the target."""
    job = plan_of(("alpha", []), ("beta", ["alpha"]), ("gamma", ["alpha", "beta"]))
    rows = plain(dag(job)).splitlines()

    assert rows[0].startswith("alpha")
    assert len(rows) > 1, "the spanning edge was drawn straight through the row"
    assert "─" in rows[1], "the detour row carries no edge"
    # alpha -> gamma leaves alpha's trunk, runs below, and turns up into gamma
    assert rows[1].strip().startswith("└") and rows[1].rstrip().endswith("┘")


def test_two_steps_of_one_wave_do_not_share_a_line():
    """Same name length meant the same trunk column, and two edges then
    merged into one vertical that read as an edge going somewhere it did
    not."""
    job = plan_of(("aa", []), ("bb", []), ("xx", ["bb"]), ("yy", ["aa"]))
    verticals = [i for i, ch in enumerate(plain(dag(job)).splitlines()[1]) if ch == "│"]
    assert len(verticals) == 2, "the two edges leaving that wave share a line"


def test_a_request_cannot_open_a_tag():
    """Job queries are human (and model) text landing in content markup."""
    job = Job(job_id="x" * 8, status=JobStatus.QUEUED, query="[b]not bold[/b]",
              created_at=T0, updated_at=T0)
    row = job_row(job.summary() | {"job_id": job.job_id})
    assert "\\[b]not bold\\[/b]" in row, "the request was not escaped"


def test_the_outputs_pane_names_each_file_and_says_whose_disk_it_is_on():
    """Fed by `list_outputs` — `to_dict()` is `asdict`, which drops `JobOutput.name`."""
    assert "name" not in canned_jobs()[0].to_dict()["outputs"][0], "the guarded shape changed"
    outputs = [dataclasses.asdict(o) | {"name": o.name} for o in canned_jobs()[0].outputs]
    job_id = canned_jobs()[0].job_id

    here = outputs_block(outputs, where=where_files_are("embedded", job_id))
    assert "sixel-matrix.svg" in here and "annex" in here and "from web_search" in here
    assert "/tmp/a/sixel-matrix.svg" in here and "on this machine" in here
    there = outputs_block(outputs, where=where_files_are("daemon", job_id))
    assert "the machine running the daemon" in there
    assert f"/jobs/{job_id}/outputs/<name>" in there, "no way to fetch the bytes"


def test_the_step_table_shows_what_each_step_spent():
    table = steps_table(canned_jobs()[0].to_dict())
    assert "24,077" in table and "~$0.0710" in table
    assert "queued" in table, "a step that has not run must still be listed"


# ------------------------------------------------------------ without textual


async def test_only_a_missing_textual_is_reported_as_a_missing_extra(monkeypatch):
    """A missing `textual` is `TuiUnavailable`; a broken import inside the UI is
    not dressed as one."""
    from jobsmith.tui import run_tui

    monkeypatch.setitem(sys.modules, "textual", None)
    with pytest.raises(TuiUnavailable):
        await run_tui(CannedService(), SESSION)

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "jobsmith.tui.app", None)
    with pytest.raises(ImportError) as raised:
        await run_tui(CannedService(), SESSION)
    assert not isinstance(raised.value, TuiUnavailable), \
        "a broken import inside the UI was reported as a missing extra"


async def test_the_ui_command_says_what_to_install(monkeypatch, capsys):
    """The extra is optional, so its absence is an instruction, not a trace."""
    import jobsmith.tui as tui
    from jobsmith.cli.main import cmd_ui

    async def refuse(*args, **kwargs):
        raise TuiUnavailable(MISSING)

    monkeypatch.setattr(tui, "run_tui", refuse)
    service = CannedService()
    code = await cmd_ui(service, type("Args", (), {"session": None, "theme": None})())

    assert code == 1
    assert ".[tui]" in capsys.readouterr().err


async def test_took_measures_a_real_step_not_the_end_of_the_run(
    store, checkpointer, tmp_path
):
    """From a real run: each dependent step's `took` is about its own sleep, not
    `0.0s`. → 0053"""
    from conftest import FakeLLM, plan_json

    delay = 0.2
    caps = [SlowEcho(n, delay=delay) for n in ("alpha", "beta", "gamma")]
    llm = FakeLLM(
        {"planner": plan_json("alpha", "beta", "gamma",
                              deps={"beta": ["alpha"], "gamma": ["beta"]})},
        default="A sufficiently long final answer for the job test.",
    )
    manager = make_manager(store, checkpointer, tmp_path, caps=caps, llm=llm)
    done = await manager.run_job((await manager.create_job("a chain")).job_id)

    took = {row["capability"]: row["took"] for row in step_states(done.to_dict())}
    assert took["alpha"] != NONE and took["beta"] != NONE and took["gamma"] != NONE
    # `alpha`'s window starts at job creation, so only the dependent steps
    # bound their own time — and each must show roughly its own sleep.
    for name in ("beta", "gamma"):
        assert float(took[name].removesuffix("s")) >= delay * 0.8, took
