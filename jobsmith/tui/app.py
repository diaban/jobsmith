"""The terminal UI: one more adapter over `AgentService`, and the first one
that can show a job while it runs.

It is not a second implementation of the REPL, it is a second *presentation*
of the same flow: `cli/repl.py` and this both drive `stream` /
`stream_approval` and both render the five events of `chat/runner.py`. What
differs is what the two can do while waiting. `run_repl` blocks the loop on
`input()`, so nothing repaints until the human types; Textual owns the event
loop and treats a keystroke as an event, so a running job can be on screen at
the same time as the conversation. That is the whole reason this exists, and
it is why `jobsmith ui` sits **beside** `jobsmith chat` rather than replacing
it: a TUI takes the whole screen and cannot be piped, and this CLI keeps
stdout pipeable on purpose.

Three panes, and one rule each:

* **chat** — the streamed turn, and the approval round trip. Tokens are
  appended to the bubble as they arrive; the terminal event is not printed,
  because the tokens already delivered it.
* **job list** — `list_jobs`, which answers with *summaries*: a row says what
  a summary knows and the detail pane loads the rest.
* **job detail** — `get_job`: the plan drawn as a DAG, a per-step table, the
  files produced. Static, refreshed on a poll; the live DAG driven by
  `subscribe()` is step 3 of #48 and deliberately not here yet.

Every colour in `CSS` is a standard theme role. A stylesheet naming a
variable the current theme does not define does not look wrong — it **fails
to parse**, and the app does not start — so this constraint is what lets all
23 themes in the Ctrl+P picker work (see `themes.py`, `render.py`).
"""
from __future__ import annotations

from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.markup import escape
from textual.widgets import ContentSwitcher, Footer, Header, Input, ListItem, ListView, Static

from ..service import TERMINAL_EVENTS, AgentService, ChatStreamError
from . import render
from .themes import DEFAULT_THEME, THEMES, pick_theme

# How often the job list re-reads itself. A poll, and named as one: the event
# stream is step 3 of #48, and until then a list that only moves when the user
# types would be the very defect this layer exists to fix.
POLL_SECONDS = 2.0

# What counts as "yes" to a proposal — the REPL's own set, so the same answer
# means the same thing in both front-ends. Anything else declines, which is
# what the `[y/N]` prompt says.
APPROVALS = ("y", "yes", "o", "oui")


class Bubble(Static):
    """One speaker's turn, growing as the tokens arrive.

    The body is escaped on every repaint rather than rendered as markup: it is
    model output, and a model that writes `[b]` in a sentence must not be able
    to open a tag in the UI drawing it.
    """

    def __init__(self, speaker: str, style: str, text: str = "") -> None:
        super().__init__(classes="bubble")
        self._speaker, self._style, self._text = speaker, style, text
        self._paint()

    def add(self, text: str) -> None:
        self._text += text
        self._paint()

    @property
    def text(self) -> str:
        return self._text

    def _paint(self) -> None:
        body = escape(self._text) if self._text else f"[{render.DIM}]…[/]"
        self.update(f"[{self._style}]{self._speaker}[/]\n{body}")


class ProposalCard(Static):
    """The human-in-the-loop interrupt, as something to answer.

    The wording is the model's own — `chat/tools.py` demands a self-contained
    query and that is what the human approves — so it is shown verbatim and
    escaped, never summarised.
    """

    def __init__(self, query: str, rationale: str) -> None:
        super().__init__(classes="proposal")
        self.update(
            f"[{render.ATTENTION}]a background job is proposed[/]\n"
            f"[b]{escape(query)}[/b]\n"
            f"[{render.DIM}]{escape(rationale)}[/]\n\n"
            f"[b {render.DONE}]y[/] [{render.DIM}]launch it[/]     "
            f"[b]n[/] [{render.DIM}]not now[/]"
        )


class ChatPane(Vertical):
    """The conversation, an activity line, and the prompt."""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="conversation")
        yield Static("", id="activity")
        yield Input(placeholder="message the agent…", id="prompt")


class JobsPane(Horizontal):
    """The job list, and the detail of whichever row is highlighted."""

    def compose(self) -> ComposeResult:
        with Vertical(id="job-list-pane"):
            yield Static("[b]jobs[/b]", id="job-list-title", classes="heading")
            yield ListView(id="job-list")
        with VerticalScroll(id="job-detail"):
            yield Static("", id="detail-title", classes="heading")
            yield Static("", id="detail-meta")
            yield Static("", id="detail-dag", classes="card")
            yield Static("", id="detail-steps", classes="card")
            yield Static("", id="detail-outputs", classes="card")
            yield Static("", id="detail-answer", classes="card")


class JobsmithApp(App[None]):
    """The application. It holds the service and drives every use case on it."""

    TITLE = "jobsmith"

    CSS = """
    Screen { background: $background; }
    #tabs { height: 1; background: $panel; padding: 0 1; }
    #body { height: 1fr; }
    .heading { color: $text-primary; text-style: bold; }
    .card { background: $panel; padding: 1 2; margin: 0 0 1 0; }

    #conversation { width: 1fr; padding: 1 2 0 2; }
    .bubble { margin: 0 0 1 0; }
    .proposal { background: $panel; border-left: thick $warning; padding: 1 2; margin: 0 0 1 0; }
    #activity { height: 1; padding: 0 2; color: $text-accent; }
    #prompt { border: none; background: $panel; padding: 0 1; }
    #prompt:focus { border: none; }

    #job-list-pane { width: 44; background: $surface; padding: 1 1 0 2; }
    /* A row is two lines, always: wrapping one would shift every row under
       it and the list would stop being scannable. */
    #job-list { background: $surface; border: none; height: 1fr;
                text-wrap: nowrap; text-overflow: ellipsis; }
    #job-list > ListItem { background: $surface; padding: 0 0 1 0; }
    #job-list > ListItem.-highlight { background: $boost; }
    #job-detail { width: 1fr; padding: 1 2 0 2; }
    """

    BINDINGS = [
        Binding("f2", "show('chat')", "chat"),
        Binding("f3", "show('jobs')", "jobs"),
        Binding("f5", "reload", "refresh"),
        Binding("f8", "cancel_job", "cancel job"),
        # Function keys, and not the obvious control letters, because the
        # prompt has first refusal on every key it is focused for: `Input`
        # binds ctrl+x, ctrl+k, ctrl+w, ctrl+u and ctrl+d for editing, so
        # those never reach the app. ctrl+j is worse than taken — it IS the
        # newline control code, so binding it steals Enter from the prompt.
    ]

    def __init__(
        self,
        service: AgentService,
        session_id: str,
        *,
        theme: str | None = None,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        super().__init__()
        self.service = service
        self.session_id = session_id
        self._theme_name = pick_theme(theme)
        self._poll_seconds = poll_seconds
        self._jobs: list[dict[str, Any]] = []
        self._selected: str | None = None      # job_id, so a refresh keeps the row
        self._awaiting_approval = False
        self._answer: Bubble | None = None     # the bubble the tokens land in

    # ------------------------------------------------------------- assembly

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="tabs")
        with ContentSwitcher(initial="chat", id="body"):
            yield ChatPane(id="chat")
            yield JobsPane(id="jobs")
        yield Footer()

    def on_mount(self) -> None:
        for theme in THEMES.values():
            self.register_theme(theme)
        # A name nothing defines would raise here and take the app with it, so
        # an unknown $JOBSMITH_THEME degrades to the default and says so.
        if self._theme_name not in self.available_themes:
            self.notify(f"unknown theme {self._theme_name!r} — using {DEFAULT_THEME}",
                        severity="warning")
            self._theme_name = DEFAULT_THEME
        self.theme = self._theme_name
        self.sub_title = (
            f"{self.service.mode} · session {self.session_id[:8]}"
            + ("" if self.service.persistent else " · jobs stop when you exit")
        )
        self._paint_tabs()
        self.reload()
        if self._poll_seconds:
            self.set_interval(self._poll_seconds, self.reload)
        self.query_one("#prompt", Input).focus()

    # ---------------------------------------------------------------- chrome

    def _paint_tabs(self) -> None:
        current = self.query_one("#body", ContentSwitcher).current
        running = sum(1 for job in self._jobs if job.get("status") == "running")
        # The chat pane hides the job list, so the count is what keeps a
        # running job visible from the conversation — the opaque wait is the
        # defect this layer exists to remove.
        live = (f"  [{render.RUNNING}]{render.GLYPH['running']} {running} running[/]"
                if running else "")
        tabs = "  ".join(
            f"[b {render.CHROME}]{name}[/]" if name == current else f"[{render.DIM}]{name}[/]"
            for name in ("chat", "jobs")
        )
        self.query_one("#tabs", Static).update(tabs + live)

    def action_show(self, pane: str) -> None:
        self.query_one("#body", ContentSwitcher).current = pane
        self._paint_tabs()
        if pane == "chat":
            self.query_one("#prompt", Input).focus()
        else:
            # Opening the pane is a request for what is true now: between two
            # polls a job can have finished, and a plan that was accurate a
            # second ago is exactly the stale picture this layer exists to
            # replace.
            self.reload()
            self.query_one("#job-list", ListView).focus()

    # ----------------------------------------------------------------- chat

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._awaiting_approval:
            self._answer_proposal(text.lower() in APPROVALS)
            return
        self._say("you", "$foreground", text)
        self._turn(text=text)

    def _say(self, speaker: str, style: str, text: str) -> Bubble:
        bubble = Bubble(speaker, style, text)
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(bubble)
        conversation.scroll_end(animate=False)
        return bubble

    def _answer_proposal(self, approved: bool) -> None:
        self._awaiting_approval = False
        self.query_one("#prompt", Input).placeholder = "message the agent…"
        self._say("you", "$foreground", "yes, go ahead" if approved else "not now")
        self._turn(approved=approved)

    @work(exclusive=True, group="turn")
    async def _turn(self, *, text: str | None = None, approved: bool | None = None) -> None:
        """Render one turn as it happens — the REPL's `render_turn`, on screen.

        The events are the port's, so this reads exactly like `cli/repl.py`
        does; only the destinations differ. The terminal event is not shown:
        a `message` restates what the tokens already wrote, and a `proposal`
        becomes a card to answer rather than a line to read.
        """
        events = (self.service.stream(self.session_id, text) if text is not None
                  else self.service.stream_approval(self.session_id, bool(approved)))
        self._answer = None
        terminal: dict[str, Any] = {}
        try:
            async for event in events:
                self._show_event(event)
                if event.get("type") in TERMINAL_EVENTS:
                    terminal = event
        except ChatStreamError as cut_short:
            # Half a turn is on screen and nothing in it says so, which is the
            # one thing the no-drop rule exists to prevent. Say it, and keep
            # the session usable — the conversation is in the checkpointer.
            self._say("jobsmith", render.FAILED, f"the reply was cut short: {cut_short}")
        self._activity("")
        self._answer = None
        if terminal.get("type") == "proposal":
            self._propose(terminal)
        self.reload()

    def _show_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "token":
            if self._answer is None:
                self._answer = self._say("jobsmith", render.CHROME, "")
            self._answer.add(str(event.get("text") or ""))
            self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)
        elif kind == "tool_started":
            self._activity(f"… {render.tool_activity(str(event.get('name') or ''))}")
        elif kind == "tool_finished":
            self._activity(f"✓ {render.tool_activity(str(event.get('name') or ''))}")

    def _activity(self, text: str) -> None:
        self.query_one("#activity", Static).update(text)

    def _propose(self, terminal: dict[str, Any]) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        conversation.mount(ProposalCard(str(terminal.get("query") or ""),
                                        str(terminal.get("rationale") or "")))
        conversation.scroll_end(animate=False)
        self._awaiting_approval = True
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = "launch it? [y/N]"
        prompt.focus()

    # ----------------------------------------------------------------- jobs

    @work(exclusive=True, group="jobs")
    async def reload(self) -> None:
        """Re-read the job list, and the highlighted job's detail with it."""
        self._jobs = await self.service.list_jobs()
        listing = self.query_one("#job-list", ListView)
        index = next((i for i, job in enumerate(self._jobs)
                      if job["job_id"] == self._selected), 0 if self._jobs else None)
        await listing.clear()
        for job in self._jobs:
            listing.append(ListItem(Static(render.job_row(job))))
        self.query_one("#job-list-title", Static).update(
            f"[b]jobs[/b]  [{render.DIM}]{len(self._jobs)}[/]")
        if index is not None:
            listing.index = index
            self._selected = selected = self._jobs[index]["job_id"]
            await self._show_detail(selected)
        self._paint_tabs()

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        index = event.list_view.index
        if index is None or index >= len(self._jobs):
            return
        self._selected = selected = self._jobs[index]["job_id"]
        await self._show_detail(selected)

    async def _show_detail(self, job_id: str) -> None:
        job = await self.service.get_job(job_id)
        if job is None:
            return
        self.query_one("#detail-title", Static).update(render.job_headline(job))
        self.query_one("#detail-meta", Static).update(render.job_meta(job))
        self.query_one("#detail-dag", Static).update(render.dag(job))
        self.query_one("#detail-steps", Static).update(render.steps_table(job))
        self.query_one("#detail-outputs", Static).update(render.outputs_block(job))
        answer = job.get("final_answer") or ""
        error = job.get("error") or ""
        body = f"[{render.DIM}]answer[/]\n{escape(answer)}" if answer else ""
        if error:
            # A job can be DONE and still carry one — a deliverable that could
            # not be written — so the error is shown next to the answer, not
            # instead of it.
            body = (body + "\n\n" if body else "") + f"[{render.FAILED}]{escape(error)}[/]"
        self.query_one("#detail-answer", Static).update(body or f"[{render.DIM}]no answer yet[/]")

    def action_reload(self) -> None:
        self.reload()

    @work(exclusive=True, group="cancel")
    async def action_cancel_job(self) -> None:
        if self._selected is None:
            return
        answer = await self.service.cancel_job(self._selected)
        self.notify(f"{self._selected[:8]} → {answer.get('status', 'unknown')}")
        self.reload()
