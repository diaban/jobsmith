"""A run that could not answer says so — structurally (#59).

The generator, held to "use ONLY the provided material", sometimes has to
answer that the material does not answer the request. Before this, saying so
had no consequence: `terminal_kind` was "answer", the job was DONE, and the
chat announced a finished job with a report path. These tests pin the whole
path of the declaration — the marker line the generator emits, the terminal it
reaches, the job that keeps everything it produced, the file that opens by
saying what it is, and the notice that reaches the conversation.
"""
from __future__ import annotations

from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.chat.session import JobNotificationMiddleware
from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.generation import split_declaration
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.state import TERMINAL_UNANSWERED
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.jobs.report import UNANSWERED_NOTICE, MarkdownReport, build_document
from jobsmith.jobs.report_html import HtmlReport

REFUSAL = (
    "NO_ANSWER: the quarterly report the request names was never provided\n"
    "\n"
    "To produce this deliverable I would need the file itself, or the figures "
    "it contains. Nothing in the gathered material covers them."
)


class Echo(Capability):
    def __init__(self, name: str = "alpha"):
        self.spec = CapabilitySpec(name=name, description=f"{name} capability")

    async def work(self, state: CapabilityBaseState) -> dict:
        return self._emit_success({"echo": self.spec.name})

    def render_context(self, result):
        return f"# {self.spec.name}\n{result['data']['echo']}"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


def make_graph(checkpointer, answer: str):
    llm = FakeLLM(
        {"planner": plan_json("alpha"), "ONLY the provided": answer},
        default="a plain answer long enough to pass the length floor",
    )
    graph = build_agent(Deps(llm=llm), CapabilityRegistry([Echo()]),
                        checkpointer=checkpointer)
    return graph, llm


async def run_graph(checkpointer, answer: str, thread: str = "u1") -> dict:
    graph, llm = make_graph(checkpointer, answer)
    out = await graph.ainvoke(
        {"query": "summarize the quarterly report", "job_id": thread},
        config={"configurable": {"thread_id": thread}},
    )
    out["_llm"] = llm
    return out


# ------------------------------------------------------- the declaration

def test_a_reply_without_the_marker_is_an_answer():
    """Fail-open: the marker is asked for only on the refusal path, so its
    absence must mean exactly what it meant before this existed."""
    text, answered = split_declaration("Here is the answer.\nWith a second line.")
    assert answered is True
    assert text == "Here is the answer.\nWith a second line."


def test_prose_that_reads_like_a_refusal_is_still_an_answer():
    """The declaration is a protocol, not a search for regret in prose — which
    would fire on a hedged answer and miss a refusal written in French."""
    _, answered = split_declaration(
        "I cannot give a figure without the source document, but the trend is clear."
    )
    assert answered is True


def test_the_marker_is_read_through_the_formatting_a_model_adds():
    text, answered = split_declaration("> **NO_ANSWER: no figures were given**\n\nBody.")
    assert answered is False
    assert text == "Body."


def test_a_declaration_with_nothing_under_it_keeps_its_reason():
    """The reason is folded into the text rather than carried as a field: when
    the model wrote nothing else, it is all there is to hand back."""
    text, answered = split_declaration("NO_ANSWER: the document was not provided")
    assert answered is False
    assert text == "the document was not provided"


# ------------------------------------------------------------ the graph

async def test_a_declared_refusal_reaches_its_own_terminal(checkpointer):
    out = await run_graph(checkpointer, REFUSAL)

    assert out["terminal_kind"] == TERMINAL_UNANSWERED
    assert out["answered"] is False
    # The steps ran and their results are kept: this is not an error channel.
    assert set(out["results"]) == {"alpha"}
    # The text survives, marker gone — it says what would have been needed.
    assert out["final_answer"].startswith("To produce this deliverable")
    assert "NO_ANSWER" not in out["final_answer"]


async def test_a_run_that_answers_is_untouched(checkpointer):
    out = await run_graph(checkpointer, "A sufficiently long final answer, freely given.")
    assert out["terminal_kind"] == "answer"
    assert out["final_answer"] == "A sufficiently long final answer, freely given."


async def test_a_declared_refusal_is_never_refined(checkpointer):
    """A short refusal fails the length rule, and refining it against the same
    context can only repeat it — or invent the answer the declaration exists
    to prevent. So the declaration is settled before the draft's validity."""
    out = await run_graph(checkpointer, "NO_ANSWER: nothing was provided")

    assert out["terminal_kind"] == TERMINAL_UNANSWERED
    assert out.get("refine_count", 0) == 0
    generations = [c for c in out["_llm"].calls
                   if "ONLY the provided" in c["messages"][0].get("content", "")]
    assert len(generations) == 1


# ------------------------------------------------------------- the job

def make_manager(store, checkpointer, tmp_path, answer: str) -> JobManager:
    graph, _ = make_graph(checkpointer, answer)
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts")


async def test_the_job_is_done_keeps_its_work_and_says_it_did_not_answer(
    store, checkpointer, tmp_path
):
    """DONE because nothing failed: the graph ran to the end, every step
    reported, the tokens were spent. FAILED would misreport the work — and
    would throw away the deliverable that explains what was missing."""
    mgr = make_manager(store, checkpointer, tmp_path, REFUSAL)
    job = await mgr.create_job("summarize the quarterly report")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE
    assert done.terminal_kind == TERMINAL_UNANSWERED
    assert done.error is None                  # a refusal is not a failure
    assert set(done.results) == {"alpha"}      # the work is kept (#41)
    assert done.report_path is not None
    written = (tmp_path / "artifacts" / f"{job.job_id}.md").read_text()
    assert UNANSWERED_NOTICE in written
    assert "To produce this deliverable" in written


# ------------------------------------------------------ the deliverable

def _doc(terminal_kind: str):
    return build_document(Job(
        job_id="abcdef0123", status=JobStatus.DONE, query="summarize the report",
        final_answer="The figures were never provided.", terminal_kind=terminal_kind,
    ))


def test_both_formats_open_by_saying_the_run_did_not_answer():
    unanswered, answered = _doc(TERMINAL_UNANSWERED), _doc("answer")
    assert unanswered.answered is False and answered.answered is True

    for reporter in (MarkdownReport(), HtmlReport()):
        rendered = reporter.render(unanswered)
        assert UNANSWERED_NOTICE in rendered, reporter.format
        # above the answer, where a reader meets it first
        assert rendered.index(UNANSWERED_NOTICE) < rendered.index("The figures")
        assert UNANSWERED_NOTICE not in reporter.render(answered), reporter.format


# ---------------------------------------------------------- the notice

def test_the_conversation_is_told_the_job_could_not_answer():
    """Announcing it like a job that answered is how someone who waited three
    minutes finds out only by reading the file."""
    job = Job(job_id="abcdef0123", status=JobStatus.DONE, query="summarize the report",
              terminal_kind=TERMINAL_UNANSWERED,
              final_answer="The figures were never provided.",
              outputs=[JobOutput(path="/tmp/abcdef0123.md")])
    notice = JobNotificationMiddleware._notice_for(job)

    assert "COULD NOT ANSWER" in notice
    assert "is DONE." not in notice              # not announced as a success
    assert "FAILED" not in notice                # nor as a crash
    assert "/tmp/abcdef0123.md" in notice        # the file is still offered
    assert "The figures were never provided." in notice


def test_a_job_that_answered_is_announced_exactly_as_before():
    job = Job(job_id="abcdef0123", status=JobStatus.DONE, query="q",
              terminal_kind="answer", final_answer="The answer.",
              outputs=[JobOutput(path="/tmp/abcdef0123.md")])
    notice = JobNotificationMiddleware._notice_for(job)

    assert notice.startswith("Job abcdef01 ('q') is DONE.")
    assert "Report file: /tmp/abcdef0123.md" in notice
