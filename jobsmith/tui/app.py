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
* **job detail** — `get_job`: the plan drawn as a DAG, a per-step table, and
  the files produced (`list_outputs`, which is the only call that carries a
  filename). It moves on its own: `subscribe()` says a job changed and the
  pane re-reads. Nothing here accumulates an event, because events are
  dropped under back-pressure by design — see `_watch`.

Every colour in `CSS` is a standard theme role. A stylesheet naming a
variable the current theme does not define does not look wrong — it **fails
to parse**, and the app does not start — so this constraint is what lets all
23 themes in the Ctrl+P picker work (see `themes.py`, `render.py`).
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
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

# What the tab bar says once the event stream is over: the screen is still
# true, it has simply stopped following. A UI has taken the terminal, so the
# stderr note `DaemonClient` prints goes nowhere anybody can read.
LIVE_LOST = "live updates stopped — F5 re-reads"

# What answers a proposal. The approvals are the REPL's own set, so the same
# word means the same thing in both front-ends. The refusals are spelt out
# rather than being "everything else": the REPL can afford that reading
# because its prompt accepts one line and then returns to the conversation,
# while here the same box carries both, and "actually, narrow it to macOS"
# read as a decline would decline the job AND lose the sentence. Anything
# that is neither is refused, and the text stays in the box.
APPROVALS = ("y", "yes", "o", "oui")
REFUSALS = ("n", "no", "non")


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

    def __init__(self, query: str, rationale: str, sources: Sequence[str] = ()) -> None:
        super().__init__(classes="proposal")
        # The files it would be allowed to open, when there are any: approving
        # the job is approving this list, so it is shown, not summarised away.
        reads = (f"[{render.DIM}]reads {escape(', '.join(sources))}[/]\n"
                 if sources else "")
        self.update(
            f"[{render.ATTENTION}]a background job is proposed[/]\n"
            f"[b]{escape(query)}[/b]\n"
            f"[{render.DIM}]{escape(rationale)}[/]\n"
            f"{reads}\n"
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
        Binding("escape", "dismiss_cancel", "", show=False),
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
    ) -> None:
        super().__init__()
        self.service = service
        self.session_id = session_id
        self._theme_name = pick_theme(theme)
        self._jobs: list[dict[str, Any]] = []
        self._selected: str | None = None      # job_id, so a refresh keeps the row
        self._awaiting_approval = False
        self._answer: Bubble | None = None     # the bubble the tokens land in
        self._streaming = False                # a turn is being rendered right now
        self._reloading = False                # a job refresh is in flight
        self._stale = False                    # ... and one more is owed after it
        self._cancel_armed: str | None = None  # job id F8 has asked about once
        self._live: asyncio.Task | None = None  # the subscription, for its lifetime
        self._live_lost = False                # the stream ended; the screen says so
        self._files_of: tuple[str, tuple[str, ...]] | None = None   # probed for these
        self._missing: set[str] = set()        # ... and these were not on disk

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
        self.refresh_jobs()
        self._live = asyncio.create_task(self._watch())
        self.query_one("#prompt", Input).focus()

    async def on_unmount(self) -> None:
        """Release the subscription — it is a resource, not a reference.

        Behind a daemon it holds an HTTP stream open, and `unsubscribe` (sync,
        as the port says) can only cancel the reader; the unwind is awaited by
        `DaemonClient.aclose`. Here the cancelled watcher is awaited so its own
        `finally` — the `unsubscribe` — has actually run before the app goes.
        """
        watcher, self._live = self._live, None
        if watcher is not None:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

    # ------------------------------------------------------- live from the port

    async def _watch(self) -> None:
        """Repaint whenever a job moves, driven by `subscribe()`.

        The one rule that shapes this: an event means **something changed,
        re-read**, never a delta to apply. `subscribe` drops under
        back-pressure on both backings — deliberately, because a missed
        progress tick is superseded by the next one — so a UI that
        accumulated events would be a UI whose picture is wrong exactly when
        it was busy. Re-reading (`list_jobs`, then `get_job` for the row on
        screen) makes a dropped event cost nothing but a slightly later
        repaint, and a burst is coalesced into one refresh rather than one
        each: the screen only ever shows the latest answer anyway.

        `None` is the port's end-of-stream marker, and the reason it exists:
        a daemon that goes away leaves a queue that is quiet in exactly the
        way a calm system is quiet. The stderr note `DaemonClient` prints for
        a person at a shell cannot reach a screen this app has taken over, so
        the fact is carried on the queue and said on the tab bar.

        Not a Textual worker on purpose: `workers.wait_for_complete()` waits
        for every one of them, and this one is over only when the app is.
        """
        queue = self.service.subscribe()
        try:
            while True:
                event = await queue.get()
                while event is not None and not queue.empty():
                    event = queue.get_nowait()      # one repaint per burst
                if event is None:
                    self._live_stopped()
                    return
                self.refresh_jobs()
        except asyncio.CancelledError:
            raise
        except Exception as failed:      # a backing that broke, not a job that did
            self._live_stopped(str(failed))
        finally:
            self.service.unsubscribe(queue)

    def _live_stopped(self, reason: str = "") -> None:
        """Say it, in both places a reader might be looking."""
        self._live_lost = True
        self._paint_tabs()
        self.notify(f"{LIVE_LOST}{f' ({reason})' if reason else ''}", severity="warning")

    # ---------------------------------------------------------------- chrome

    def _paint_tabs(self) -> None:
        current = self.query_one("#body", ContentSwitcher).current
        running = sum(1 for job in self._jobs if job.get("status") == "running")
        # The chat pane hides the job list, so the count is what keeps a
        # running job visible from the conversation — the opaque wait is the
        # defect this layer exists to remove.
        live = (f"  [{render.RUNNING}]{render.GLYPH['running']} {running} running[/]"
                if running else "")
        # A screen that stopped following is still true, and must not look
        # like one that is up to date.
        if self._live_lost:
            live += f"  [{render.FAILED}]{LIVE_LOST}[/]"
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
            self.refresh_jobs()
            self.query_one("#job-list", ListView).focus()

    # ----------------------------------------------------------------- chat

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """One turn at a time, and a proposal is answered before anything else.

        Both refusals keep the typed text in the box, because the alternative
        is losing a sentence the user wrote. Sending while a turn streams used
        to cancel it mid-sentence (`@work(exclusive=True)`), which left half an
        answer on screen with nothing saying it was cut — and, if the cancelled
        turn was about to propose a job, a thread interrupted in the
        checkpointer that the next message would not have answered.
        """
        text = event.value.strip()
        if self._awaiting_approval:
            answer = text.lower()
            if answer in APPROVALS or answer in REFUSALS or not text:
                event.input.value = ""
                self._answer_proposal(answer in APPROVALS)   # bare Enter = the N
            else:
                self.notify("answer the proposal first — y to launch it, n to decline",
                            severity="warning")
            return
        if not text:
            return
        if self._streaming:
            self.notify("still writing — the turn has to finish first",
                        severity="warning")
            return
        event.input.value = ""
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
        self._streaming = True
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
        finally:
            # Also the path a cancellation takes (the app shutting down): the
            # activity line must not be left saying something is happening.
            self._streaming = False
            self._activity("")
        self._answer = None
        if terminal.get("type") == "proposal":
            self._propose(terminal)
        self.refresh_jobs()

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
                                        str(terminal.get("rationale") or ""),
                                        [str(s) for s in terminal.get("sources") or []]))
        conversation.scroll_end(animate=False)
        self._awaiting_approval = True
        prompt = self.query_one("#prompt", Input)
        prompt.placeholder = "launch it? [y/N]"
        prompt.focus()

    # ----------------------------------------------------------------- jobs

    def refresh_jobs(self) -> None:
        """Ask for a refresh; one already in flight is joined, never cancelled.

        A refresh is three calls (`list_jobs`, then `get_job` and
        `list_outputs` for the highlighted row) — round trips against a
        daemon, and events arrive faster than that when a wave lands. Left to
        `exclusive=True` each one would cancel the previous worker, so the
        list would never finish repopulating and a cancellation landing on
        `clear()` would leave it empty.

        Skipping alone is not enough either, and that is what the poll used to
        hide: with a timer re-arming, a request dropped because a refresh was
        running was picked up two seconds later. Nothing re-arms now, so the
        last event of a job — the one that says it is DONE — would be the one
        most likely to be dropped. It is remembered instead, and the refresh
        in flight goes round once more.
        """
        if self._reloading:
            self._stale = True
            return
        self.reload()

    @work(exclusive=True, group="jobs")
    async def reload(self) -> None:
        """Re-read the job list, and the highlighted job's detail with it."""
        self._reloading = True
        try:
            await self._reload()
            while self._stale:
                self._stale = False       # anything arriving now asks again
                await self._reload()
        finally:
            self._reloading = self._stale = False

    async def _reload(self) -> None:
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
        if self._selected != self._jobs[index]["job_id"]:
            self._cancel_armed = None       # armed for a row nobody is on now
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
        await self._show_files(job_id)
        answer = job.get("final_answer") or ""
        error = job.get("error") or ""
        body = f"[{render.DIM}]answer[/]\n{escape(answer)}" if answer else ""
        if error:
            # A job can be DONE and still carry one — a deliverable that could
            # not be written — so the error is shown next to the answer, not
            # instead of it.
            body = (body + "\n\n" if body else "") + f"[{render.FAILED}]{escape(error)}[/]"
        self.query_one("#detail-answer", Static).update(body or f"[{render.DIM}]no answer yet[/]")

    async def _show_files(self, job_id: str) -> None:
        """What the job produced — from `list_outputs`, and honest about where.

        Three things this pane must not do, all of them promises the port
        makes rather than choices made here.

        It must not read the files off `get_job`: that shape is
        `dataclasses.asdict`, which drops `JobOutput.name` because it is a
        property, and reading the missing key is what printed a blank
        filename on every job. `list_outputs` is the call that carries one.

        It must not present a path as something to open. `find_output`
        answers with a locator on the machine that RAN the job, so with a
        daemon backing these are the daemon's paths — true, and unopenable
        from here. The pane says whose disk they are on and names the
        download route.

        And it must not show a file that is no longer there: `find_output`
        answers None for one deleted since the job finished, on both
        backings. That answer is a round trip per file remotely, so it is
        asked once per *set* of files — which changes when a step produces
        one, and never between two events about the same step. F5 asks again
        from scratch, which is what makes a file deleted later reachable.
        """
        outputs = await self.service.list_outputs(job_id) or []
        names = tuple(str(o.get("name") or "") for o in outputs)
        if self._files_of != (job_id, names):
            self._files_of = (job_id, names)
            self._missing = {name for name in names
                             if await self.service.find_output(job_id, name) is None}
        self.query_one("#detail-outputs", Static).update(render.outputs_block(
            outputs,
            missing=self._missing,
            where=render.where_files_are(self.service.mode, job_id) if outputs else "",
        ))

    def action_reload(self) -> None:
        """Re-read everything, including the questions a refresh caches."""
        self._files_of = None
        self.refresh_jobs()

    # -------------------------------------------------------- cancelling one

    def action_cancel_job(self) -> None:
        """Cancel the highlighted job — from the jobs pane, and on the second press.

        Two guards, and both come from the same fact: `_selected` defaults to
        the first row, and `list_jobs` sorts newest first. So from the chat
        pane a single keystroke would stop the job the user had just launched,
        without ever having shown them which one. It is scoped to the pane
        that displays the row, and armed by naming the job before it acts —
        `escape`, or highlighting another row, disarms it.
        """
        if self.query_one("#body", ContentSwitcher).current != "jobs":
            self.notify("open the jobs pane (F3) to cancel a job", severity="warning")
            return
        if self._selected is None:
            return
        if self._cancel_armed != self._selected:
            self._cancel_armed = self._selected
            self.notify(f"press F8 again to cancel {self._selected[:8]}",
                        severity="warning")
            return
        self._cancel_armed = None
        self._cancel(self._selected)

    def action_dismiss_cancel(self) -> None:
        if self._cancel_armed is not None:
            self._cancel_armed = None
            self.notify("cancel dropped")

    @work(exclusive=True, group="cancel")
    async def _cancel(self, job_id: str) -> None:
        answer = await self.service.cancel_job(job_id)
        self.notify(f"{job_id[:8]} → {answer.get('status', 'unknown')}")
        self.refresh_jobs()
