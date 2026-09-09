"""The terminal UI: what it draws, and what it does with a turn.

Three kinds of test, and they are deliberately not the same test:

* **layout** — an SVG snapshot of a screen, in ONE theme. A snapshot pins
  where things are, not what colour they are, so a second theme's snapshot
  would only be another file to regenerate.
* **colour** — one test that mounts every registered theme, ours and
  Textual's, and forces a full render. It costs the same for 23 themes as
  for 2, and it is the only thing that catches the failure mode `themes.py`
  is written around: a stylesheet naming a variable the theme does not
  define does not look wrong, it fails to parse and the app never starts.
* **behaviour** — the streamed turn and the approval round trip, driven
  through the real `LocalAgentService` with the scripted chat model, because
  what is being checked is that the UI renders the port's flow rather than
  its own idea of one.

The canned service below answers with real `Job` records: the shapes come
from `jobs/models.py` and only the timestamps are frozen, so a change to what
`get_job` returns reaches these tests instead of passing them.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from conftest import ScriptedChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from test_chat import launch_call
from test_jobs import make_manager
from textual.widgets import Static

from jobsmith.chat import ChatSession
from jobsmith.core.usage import Usage
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.service import AgentService, LocalAgentService
from jobsmith.tui import MISSING, TuiUnavailable
from jobsmith.tui.app import Bubble, JobsmithApp, ProposalCard
from jobsmith.tui.render import MARKUP_ROLES, dag, job_row, step_states, steps_table
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

    def __init__(self, jobs: list[Job] | None = None, events: list[dict] | None = None):
        self.jobs = jobs if jobs is not None else canned_jobs()
        self.events = events or []
        self.cancelled: list[str] = []

    async def new_session(self, session_id: str | None = None) -> str:
        return session_id or SESSION

    async def stream(self, session_id: str, text: str):
        for event in self.events:
            yield event

    async def stream_approval(self, session_id: str, approved: bool):
        yield {"type": "message", "content": "ok"}

    async def launch_job(self, query, *, session_id=None, inputs=None) -> dict:
        return {"job_id": "new", "status": "queued"}

    async def list_jobs(self, *, status=None, session_id=None) -> list[dict]:
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
        return []

    async def find_output(self, job_id: str, name: str) -> str | None:
        return None

    def subscribe(self, *, max_queue: int = 256) -> asyncio.Queue:
        return asyncio.Queue(maxsize=max_queue)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        return None


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
    """The app with the poll disabled — a timer is noise in a snapshot."""
    service = kwargs.pop("service", None) or CannedService(events=PROPOSAL_TURN)
    return JobsmithApp(service, SESSION, poll_seconds=0, **kwargs)


async def settle(pilot: Any) -> None:
    """Let the workers that load the job list and its detail finish."""
    await pilot.pause()
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


# ------------------------------------------------------------------- layout


def test_the_jobs_screen_looks_like_this(snap_compare):
    """Job list on the left, the plan and the step table on the right."""
    async def before(pilot):
        await settle(pilot)
        await pilot.press("f3")
        await settle(pilot)

    assert snap_compare(canned_app(), terminal_size=(126, 38), run_before=before)


def test_the_chat_screen_looks_like_this(snap_compare):
    """A streamed turn that ended on a proposal, waiting to be answered."""
    async def before(pilot):
        await settle(pilot)
        await pilot.press(*"look into sixel support")
        await pilot.press("enter")
        await settle(pilot)

    assert snap_compare(canned_app(), terminal_size=(126, 38), run_before=before)


# ------------------------------------------------------------------- colour


async def test_every_registered_theme_resolves():
    """Mount both panes under every registered theme and render them for real.

    This is the test `themes.py` exists for, and it asks two questions that
    are not the same one.

    *Does the app start?* A stylesheet naming a variable the theme does not
    define raises `UnresolvedVariableError` at parse time — measured: the app
    never composes. `export_screenshot()` is a full render of every widget on
    screen, so it answers that for every theme Ctrl+P can reach.

    *Is the colour the one that was asked for?* A different question, because
    an undefined variable in **content markup** does not raise — it is
    dropped and the text renders unstyled. So the roles are checked by name
    as well. Both `ansi-*` themes really are short one of them: they generate
    a reduced variable set (no `$foreground-disabled`, and `$surface-active`
    is spelt `$surface-actrive` there), which is why the exception below is
    written down rather than papered over. Under those two, queued steps and
    rules lose their dimming and read as ordinary foreground — cosmetic, and
    the price the issue already accepted for standard roles.
    """
    app = canned_app()
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press(*"hello")
        await pilot.press("enter")           # a turn, so the proposal card is up
        await settle(pilot)
        themes = list(app.available_themes)
        assert set(THEMES) <= set(themes), "ours are not in the picker"
        assert len(themes) >= 20, "the built-in themes went missing"
        for name in themes:
            app.theme = name
            await pilot.pause()
            unresolved = {role.lstrip("$") for role in MARKUP_ROLES} - set(
                app.get_css_variables())
            assert unresolved <= ANSI_GAP.get(name, set()), \
                f"{name} does not define {sorted(unresolved)} — that markup renders unstyled"
            assert app.export_screenshot(), f"{name} rendered nothing"
            await pilot.press("f3")          # the other pane, same question
            await settle(pilot)
            assert app.export_screenshot(), f"{name} rendered nothing on the jobs pane"
            await pilot.press("f2")
            await pilot.pause()


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


async def test_cancelling_from_the_ui_reaches_the_port():
    """From the chat pane, prompt focused — which is the case the key choice
    turns on: `Input` claims ctrl+x, ctrl+k, ctrl+w and ctrl+u for editing,
    so a binding on any of them would never reach the app."""
    service = CannedService(events=PROPOSAL_TURN)
    app = JobsmithApp(service, SESSION, poll_seconds=0)
    async with app.run_test(size=(126, 38)) as pilot:
        await settle(pilot)
        await pilot.press("f8")
        await settle(pilot)
    assert service.cancelled == [service.jobs[0].job_id]


# ------------------------------------------------------- the streamed turn


def make_service(store, checkpointer, tmp_path, responses):
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=responses)
    saver = MemorySaver()
    service = LocalAgentService(manager, lambda session_id=None: ChatSession(
        manager, model, session_id=session_id, checkpointer=saver))
    return service, manager


ANSWER = "A reasonably long answer that no single chunk should ever carry whole."


async def test_the_answer_is_drawn_as_it_arrives(store, checkpointer, tmp_path, monkeypatch):
    """Not "the answer appears" — that a *block* would satisfy too.

    Every state the answer bubble passes through is recorded: a turn rendered
    on arrival leaves several, each a growing prefix of the last. A turn
    rendered when it is over leaves exactly one.
    """
    seen: list[str] = []
    original = Bubble.add

    def spy(self: Bubble, text: str) -> None:
        original(self, text)
        seen.append(self.text)

    monkeypatch.setattr(Bubble, "add", spy)
    service, _ = make_service(store, checkpointer, tmp_path, [AIMessage(content=ANSWER)])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id, poll_seconds=0)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await pilot.press(*"hello")
        await pilot.press("enter")
        await settle(pilot)

    assert len(seen) > 1, "the answer landed in one block: nothing was streamed"
    assert seen[-1] == ANSWER
    assert all(earlier == later[:len(earlier)] for earlier, later in zip(seen, seen[1:], strict=False)), \
        "the bubble did not grow by prefixes"


async def test_a_proposal_is_approved_through_the_ui(store, checkpointer, tmp_path):
    """The round trip: the interrupt becomes a card, `y` resumes the graph,
    and a job exists on the other side of it."""
    service, manager = make_service(store, checkpointer, tmp_path, [
        launch_call("survey sixel support", "it needs the web"),
        AIMessage(content=ANSWER),
    ])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id, poll_seconds=0)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await pilot.press(*"look into sixel")
        await pilot.press("enter")
        await settle(pilot)
        card = app.query(ProposalCard)
        assert len(card) == 1, "the proposal was not shown as something to answer"
        assert "survey sixel support" in str(card.first().content)
        assert app.query_one("#prompt").placeholder == "launch it? [y/N]"
        assert not await manager.list_jobs(), "a job was created before approval"

        await pilot.press("y", "enter")
        await settle(pilot)
        assert app.query_one("#prompt").placeholder == "message the agent…"

    jobs = await manager.list_jobs()
    assert [job.query for job in jobs] == ["survey sixel support"]


async def test_declining_a_proposal_creates_nothing(store, checkpointer, tmp_path):
    service, manager = make_service(store, checkpointer, tmp_path, [
        launch_call("survey sixel support", "it needs the web"),
        AIMessage(content="fine, not now"),
    ])
    session_id = await service.new_session()

    app = JobsmithApp(service, session_id, poll_seconds=0)
    async with app.run_test(size=(100, 30)) as pilot:
        await settle(pilot)
        await pilot.press(*"look into sixel")
        await pilot.press("enter")
        await settle(pilot)
        await pilot.press("n", "enter")
        await settle(pilot)

    assert await manager.list_jobs() == []


# ------------------------------------------------------------- what it draws


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
    plain = re.sub(r"\[[^\]]*\]", "", dag(canned_jobs()[0].to_dict()))
    rows = plain.splitlines()

    assert rows[0].startswith("documents──┬───research")
    assert "web_search" in rows[2], "the second wave's members are not stacked"
    # Two junctions on the first row: documents forking, and the web_search →
    # analysis trunk crossing the research → analysis segment. The second one
    # is the whole reason directions are accumulated per cell — drawn segment
    # by segment it would be whichever of the two was written last.
    assert rows[0].count("┬") == 2
    assert rows[2].strip() == "└───web_search──┘"
    # Status is colour only: the list's glyphs must not leak into the plan.
    assert not any(glyph in plain for glyph in ("●", "◐", "○", "✕", "⊘"))


def test_a_request_cannot_open_a_tag():
    """Job queries are human (and model) text landing in content markup."""
    job = Job(job_id="x" * 8, status=JobStatus.QUEUED, query="[b]not bold[/b]",
              created_at=T0, updated_at=T0)
    row = job_row(job.summary() | {"job_id": job.job_id})
    assert "\\[b]not bold\\[/b]" in row, "the request was not escaped"


def test_the_step_table_shows_what_each_step_spent():
    table = steps_table(canned_jobs()[0].to_dict())
    assert "24,077" in table and "~$0.0710" in table
    assert "queued" in table, "a step that has not run must still be listed"


# ------------------------------------------------------------ without textual


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
