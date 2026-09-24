"""Driving one conversational turn, and translating it into domain events.

This is the chat twin of `jobs/runner.py`, and it exists for the same reason:
it is the **only** module that knows the shape of what LangGraph's `astream`
emits for a chat agent — the `("messages", (chunk, metadata))` pairs, the
`("updates", {node: update})` ones, the `model`/`tools` node names, the
`__interrupt__` key. Everything above it reacts to the small typed events
below, so the service, the HTTP adapter and every UI are written against a
vocabulary that does not move when LangChain reshapes its agent.

Seven events, and no more, because a turn only ever shows seven things:

    Token          the answer being written
    ToolStarted    the model asked for a tool, by its real name
    ToolFinished   that tool answered
    JobStarted     a job began inside this turn, and here is what it will do
    JobPlanned     that job's plan was decided, while the turn waits on it
    Message        the turn is over, here is the whole reply
    Proposal       the turn is over, it is waiting for an approval

The two terminals are not an invention of the stream: they are the duality
`LocalAgentService._reply` already carried as a return value. Streaming moves
it to the end of a flow, which is why `send()` can be *defined* as draining
this stream — one implementation of the turn, not two.

`ToolStarted` carries the tool's real name (`launch_job`), never a phrase for
a human. Turning it into readable prose is the presentation layer's business,
for the same reason `REPORT_MEDIA_TYPES` lives in `api/app.py` and not in
`jobs/report.py`: the CLI and a future TUI word it differently, and neither
wording belongs in the thing that reports the fact.

**A third stream mode, and the sixth event** (#83). A task now runs *inside*
the turn by default, so two things have to reach the reader from a place the
other two modes cannot see — from inside a tool, while it is running: what
the job is about to do (`JobStarted`, the approval card's payload said as a
notice), and the answer the job produced. LangGraph's `custom` mode is that
channel; `_from_custom` is the only place its payloads are read, so the tool
and the front-ends stay strangers.

The job's answer arrives as `Token`, not as an event of its own, and that is
the decision rather than an economy: a front-end already renders tokens as
the turn's answer, and this IS the turn's answer — the one the user asked
for. Nothing between here and the screen rewrites it, which is the whole
point (see `chat/tools.py`: the model is told the answer was delivered and
is asked NOT to restate it, because a model handed 2 000 words paraphrases
them).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage

# `create_agent`'s node names. The updates stream is keyed by them, so this is
# where that coupling is admitted rather than spread over the service.
_MODEL_NODE = "model"
_TOOLS_NODE = "tools"
_INTERRUPT = "__interrupt__"

# The custom-stream vocabulary, declared where it is *read* rather than where
# it is written: `chat/tools.py` imports these, so the protocol between a tool
# and this translator has exactly one definition. Anything else on that
# channel is ignored — a payload nobody here understands is not a turn event.
CUSTOM_JOB_STARTED = "job_started"
CUSTOM_JOB_PLANNED = "job_planned"
CUSTOM_ANSWER = "answer"

# All three are needed and none is redundant: `messages` carries the answer as
# the model writes it, `updates` carries what a token cannot show (which tool
# was called, that the run stopped on an interrupt), and `custom` carries what
# a *tool* has to say while it runs.
_STREAM_MODES = ["messages", "updates", "custom"]


@dataclass(frozen=True)
class Token:
    """A piece of the answer being written in this turn.

    Usually the model's own words. It is also how a job's answer reaches the
    reader (#83): a task that finished inside the turn writes its answer here
    **verbatim**, because it is the answer that was asked for and a model
    asked to relay it would paraphrase it. A front-end needs to know neither
    — it renders the turn's answer as it arrives, which is what this is.
    """
    text: str


@dataclass(frozen=True)
class ToolStarted:
    """The model asked for a tool — its real name, not a phrase."""
    name: str


@dataclass(frozen=True)
class ToolFinished:
    """That tool answered."""
    name: str


@dataclass(frozen=True)
class JobStarted:
    """A job began inside this turn, and here is what it will do.

    The payload the approval card used to carry, said as a **notice** instead
    of asked as a question (#83). Three guarantees hung on that card and all
    three survive here, as visibility rather than as a gate: the reformulated
    `query` (the engine never sees the thread, so a reader is what catches a
    referent that has gone), the `sources` it may open (#60) and the earlier
    jobs it builds on (`from_jobs`, #104), and the document it will write
    (#55). What the card could not carry is `job_id`, because at
    proposal time no job existed — and it is the load-bearing addition, since
    cancellation is now the undo the approval used to be the gate for.

    Not a terminal: a turn that starts a job carries on, and ends on a
    `Message` like any other.
    """
    job_id: str
    query: str
    rationale: str = ""
    sources: list[str] = field(default_factory=list)
    document_name: str = ""
    document_title: str = ""
    # Three states, not two (#84): `None` is a request that named no format,
    # `[]` is a request for **no document**, a list is those formats. A
    # front-end that collapsed them would print a filename for a file nobody
    # is going to write, which is the promise #55 exists to stop making.
    formats: list[str] | None = None
    # The earlier jobs of this conversation it builds on (#74), each as
    # `{"job_id", "query"}` — the start of that job's query, so the user
    # recognises which one the model picked (#104). Empty when none.
    from_jobs: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class JobPlanned:
    """The job started in this turn has a plan, and here it is (#86).

    The plan is the first moment a run has a shape worth showing, and it is
    decided seconds into a turn that may wait twenty. `steps` is the plan as
    the planner left it — `{"capability", "depends_on"}` per step, in plan
    order — and never a sentence: how a DAG reads on one line is each
    front-end's wording, exactly as `ToolStarted` carries a name and not a
    phrase. Not a `Token`: a plan is not the answer, and a token would put it
    into `Message.content`, the reply a caller that waits is handed.

    A notice, not a question: nothing waits on it, and nothing here can
    amend the plan it shows (that is #5).
    """
    job_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Message:
    """Terminal: the turn ended with a reply — the whole of what it produced.

    `content` is the concatenation of every `Token` this turn yielded, in
    order, so a caller that **waits** ends up with exactly the text a caller
    that **renders** saw. That is #50's invariant, and it is not a nicety: a
    turn is driven in one place precisely so the two can never be told
    different things about it.

    It used to be the model's own last message, which was the same string
    while the tokens and the terminal came from that message. #83 broke that
    tie — a job's answer is written into the turn by the tool, and the
    model's reply is then "at most one short sentence" — so a terminal built
    from the model would have dropped the answer entirely for anyone who did
    not render the flow (`POST /sessions/{id}/messages`, the surface a web
    UI will use). Composing from the tokens leaves no second source of truth
    to drift: whatever the reader saw IS the reply.
    """
    content: str


@dataclass(frozen=True)
class Proposal:
    """Terminal: the turn ended on a job awaiting human approval.

    `sources` is the files the job would be allowed to open (`launch_job`'s
    `source_files`). It rides on the terminal for the same reason `query`
    does: what the user approves has to be what they were shown, and a file
    handed over without being named is a second silent decision. A list, not
    a tuple, because this crosses HTTP and both backings must answer with the
    same JSON.
    """
    query: str | None
    rationale: str | None
    sources: list[str] = field(default_factory=list)
    # What the job would call its deliverable, title it, and write it as
    # (#55). Same reason as `sources`: a document named where the user cannot
    # see the name is a second silent decision, and the name is the one thing
    # they will look for on disk afterwards.
    document_name: str = ""
    document_title: str = ""
    formats: list[str] | None = None    # see `JobStarted.formats` (#84)
    from_jobs: list[dict[str, str]] = field(default_factory=list)  # see `JobStarted`


ChatEvent = (Token | ToolStarted | ToolFinished | JobStarted | JobPlanned
             | Message | Proposal)


def _from_custom(payload: Any) -> ChatEvent | None:
    """One custom-stream payload as a turn event, or None if it is not ours.

    Tolerant on purpose: the channel is shared with anything else that
    might one day write to it, and an unknown payload is not a defect in
    the turn. What it must never do is *guess* — an event nobody can name
    is dropped, never rendered as an answer.
    """
    if not isinstance(payload, dict):
        return None
    kind = payload.get("event")
    if kind == CUSTOM_ANSWER:
        return Token(str(payload.get("text") or ""))
    if kind == CUSTOM_JOB_STARTED:
        return JobStarted(
            str(payload.get("job_id") or ""),
            str(payload.get("query") or ""),
            str(payload.get("rationale") or ""),
            [str(ref) for ref in payload.get("sources") or []],
            str(payload.get("document_name") or ""),
            str(payload.get("document_title") or ""),
            _formats(payload.get("formats")),
            _job_references(payload.get("from_jobs")),
        )
    if kind == CUSTOM_JOB_PLANNED:
        return JobPlanned(str(payload.get("job_id") or ""),
                          _plan_steps(payload.get("steps")))
    return None


def _plan_steps(value: Any) -> list[dict[str, Any]]:
    """A payload's plan steps as fresh `{capability, depends_on}` dicts.

    Lists of strings whatever arrived, because this crosses HTTP and both
    backings must answer with the same JSON (#50).
    """
    return [{"capability": str(step.get("capability") or ""),
             "depends_on": [str(dep) for dep in step.get("depends_on") or []]}
            for step in value or [] if isinstance(step, dict)]


def _job_references(value: Any) -> list[dict[str, str]]:
    """The jobs a payload says the run builds on, as plain `{job_id, query}`.

    A list of fresh dicts of strings, whatever arrived: this crosses HTTP, and
    both backings must answer with the same JSON (#50).
    """
    return [{"job_id": str(ref.get("job_id") or ""), "query": str(ref.get("query") or "")}
            for ref in value or [] if isinstance(ref, dict)]


def _formats(value: Any) -> list[str] | None:
    """The document formats of a payload, keeping `None` apart from `[]`.

    The one coercion in this module that must NOT use `or []`: absent means
    "the run decides" and empty means "no document" (#84), and a front-end
    told the second when the first was meant would announce a file that is
    coming, or say nothing about one that is not.
    """
    return None if value is None else [str(fmt) for fmt in value]


def _text_of(message: Any) -> str:
    """The message's text, unstripped.

    Unstripped on purpose: a token is a fragment, and trimming it would glue
    words together. `.text` flattens provider content blocks, so a reply that
    arrives as blocks reads the same as one that arrives as a string.
    """
    text = getattr(message, "text", "")
    return text if isinstance(text, str) else ""


class ChatRunner:
    """Runs one turn of a conversation and yields what happened, in domain terms.

    Two ways in — `stream()` sends a message, `resume()` answers a pending
    approval — and both are translated by the same code, because a
    post-approval reply is a turn like any other.
    """

    def __init__(self, agent: Any):
        self.agent = agent

    @staticmethod
    def _config(session_id: str) -> dict[str, Any]:
        # Same convention as the job engine: the session id IS the thread id,
        # which is what makes a conversation rebuildable from the checkpointer.
        return {"configurable": {"thread_id": session_id}}

    def stream(self, session_id: str, text: str) -> AsyncIterator[ChatEvent]:
        from langchain_core.messages import HumanMessage

        return self._translate(self.agent.astream(
            {"messages": [HumanMessage(text)]},
            config=self._config(session_id),
            stream_mode=_STREAM_MODES,
        ))

    def resume(self, session_id: str, approved: bool) -> AsyncIterator[ChatEvent]:
        """Answer the approval the last turn stopped on, and stream what follows."""
        from langgraph.types import Command

        return self._translate(self.agent.astream(
            Command(resume={"approved": approved}),
            config=self._config(session_id),
            stream_mode=_STREAM_MODES,
        ))

    async def _translate(self, stream: AsyncIterator[Any]) -> AsyncIterator[ChatEvent]:
        """LangGraph's three stream modes → the events above, terminal last.

        See `_STREAM_MODES` for why each one is there. Order is the stream's
        own, so a tool's notice lands where it happened: after the
        `ToolStarted` that announced the call and before the `ToolFinished`
        that closed it.

        Exactly one terminal is always yielded, last. A caller draining this
        for a reply (`AgentService.send`) must never have to decide what an
        exhausted stream with no terminal meant.

        **`Message` is the transcript, not the model's last message.** Every
        `Token` yielded is kept, and the terminal is their concatenation —
        see `Message` for why that has to be the definition rather than a
        convenience.
        """
        transcript: list[str] = []
        # Only ever a stand-in: see `Message`. Kept because a model that does
        # not stream at all would otherwise make the terminal empty.
        last_model_text, proposal = "", None
        async for mode, chunk in stream:
            if mode == "messages":
                message, _metadata = chunk
                # AI messages only: the tools node publishes its ToolMessages
                # here too, and a tool's output is not the answer being written.
                if isinstance(message, AIMessage) and (text := _text_of(message)):
                    transcript.append(text)
                    yield Token(text)
            elif mode == "custom":
                if (event := _from_custom(chunk)) is not None:
                    if isinstance(event, Token):
                        transcript.append(event.text)
                    yield event
            elif mode == "updates":
                for node, value in chunk.items():
                    if node == _INTERRUPT:
                        proposal = value[0].value
                    elif not isinstance(value, dict):
                        continue
                    elif node == _MODEL_NODE:
                        for message in value.get("messages") or []:
                            last_model_text = _text_of(message)
                            for call in getattr(message, "tool_calls", None) or []:
                                yield ToolStarted(call["name"])
                    elif node == _TOOLS_NODE:
                        for message in value.get("messages") or []:
                            yield ToolFinished(getattr(message, "name", "") or "")
        if proposal is not None:
            yield Proposal(
                proposal.get("query"), proposal.get("rationale"),
                list(proposal.get("sources") or []),
                str(proposal.get("document_name") or ""),
                str(proposal.get("document_title") or ""),
                _formats(proposal.get("formats")),
                _job_references(proposal.get("from_jobs")),
            )
        else:
            # The transcript, and the model's last message only when nothing
            # was streamed at all — a model with no `_astream` of its own
            # yields one chunk through LangChain's fallback, but one that
            # emitted none must not turn a written answer into an empty
            # terminal. It can never mask a missing job answer: that arrives
            # as a Token, so a job that answered leaves a non-empty
            # transcript and this branch is not reached.
            yield Message("".join(transcript) if transcript else last_model_text)
