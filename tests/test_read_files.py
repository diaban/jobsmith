"""A request that names a file: the path policy, the port, and the loop.

Three layers, tested where each one actually decides something:

- `core/paths.py` — what a path is allowed to be. Every refusal here is a
  security property, so each is asserted on its own rather than through a
  capability that might be swallowing it.
- `LocalFileReader` — the adapter, against a real directory, because "a
  symlink out of the root is refused" is a claim about a filesystem.
- `ReadFilesCapability` — against a FAKE reader, because the capability must
  not know what is behind the port. If a test here needs a disk, the
  capability has learned something it should not know.

And, at the end, the loop the issue is about: a job writes a report, a second
job names that report, and the second job actually reads it.
"""
from __future__ import annotations

import asyncio

import pytest
from conftest import FakeLLM, ScriptedChatModel, plan_json
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from test_chat import launch_call
from test_jobs import make_manager

from jobsmith.agents.base import AgentContext
from jobsmith.agents.default import DefaultResources, default_capabilities, readable_roots
from jobsmith.agents.default.read_files import ReadFilesCapability, named_files
from jobsmith.agents.default.sources import Document, DocumentUnavailable, LocalFileReader
from jobsmith.app import build_app
from jobsmith.app.providers import KeywordChatModel, KeywordLLM
from jobsmith.chat import ChatSession
from jobsmith.core.builder import build_agent
from jobsmith.core.deps import Deps
from jobsmith.core.paths import PathRefused, resolve_within, safe_name
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.state import SOURCE_FILES_INPUT_KEY
from jobsmith.jobs.models import JobStatus

# --------------------------------------------------------------- the policy


def test_a_relative_path_inside_the_root_is_allowed(tmp_path):
    (tmp_path / "report.md").write_text("hello")
    assert resolve_within("report.md", [tmp_path]) == (tmp_path / "report.md").resolve()


def test_traversal_out_of_the_root_is_refused(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    with pytest.raises(PathRefused, match="outside the readable area"):
        resolve_within("../secret.txt", [root])


def test_a_symlink_pointing_out_of_the_root_is_traversal(tmp_path):
    """The check is on where a path LANDS, which is the whole reason it resolves.

    A spelling-based guard passes this one: `escape.md` contains no `..` and
    no separator at all.
    """
    root = tmp_path / "artifacts"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("nope")
    (root / "escape.md").symlink_to(secret)
    with pytest.raises(PathRefused, match="outside the readable area"):
        resolve_within("escape.md", [root])


def test_an_absolute_path_is_judged_by_where_it_lands(tmp_path):
    """Not refused on sight: this product prints absolute paths, and the path
    a user pastes back is the one they were shown."""
    root = tmp_path / "artifacts"
    root.mkdir()
    inside = root / "report.md"
    inside.write_text("hello")
    assert resolve_within(str(inside), [root]) == inside.resolve()

    outside = tmp_path / "elsewhere.md"
    outside.write_text("nope")
    with pytest.raises(PathRefused, match="outside the readable area"):
        resolve_within(str(outside), [root])


def test_no_root_means_nothing_is_readable(tmp_path):
    with pytest.raises(PathRefused, match="nothing is readable here"):
        resolve_within("report.md", [])


def test_an_empty_or_null_reference_is_refused(tmp_path):
    with pytest.raises(PathRefused):
        resolve_within("   ", [tmp_path])
    with pytest.raises(PathRefused):
        resolve_within("re\x00port.md", [tmp_path])


def test_the_first_root_containing_the_file_wins(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    (first / "x.md").write_text("from a")
    (second / "x.md").write_text("from b")
    assert resolve_within("x.md", [first, second]) == (first / "x.md").resolve()


def test_a_name_may_not_be_a_location():
    assert safe_name(" report.md ") == "report.md"
    assert safe_name("/report.md") == "report.md"       # a leading slash means the name
    for bad in ("", ".", "..", "a/b", "a\\b"):
        with pytest.raises(PathRefused):
            safe_name(bad, "report name")


# --------------------------------------------------------------- the adapter


async def test_the_reader_returns_the_whole_named_document(tmp_path):
    (tmp_path / "report.md").write_text("# Title\n\nthe body of the report")
    document = await LocalFileReader([tmp_path]).read("report.md")
    assert document.id == "report.md"
    assert document.title == "report.md"
    assert "the body of the report" in document.text
    assert document.source == str((tmp_path / "report.md").resolve())


async def test_the_reader_refuses_a_path_outside_its_roots(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("nope")
    (root / "escape.txt").symlink_to(tmp_path / "secret.txt")
    reader = LocalFileReader([root])
    for ref in ("../secret.txt", "escape.txt", str(tmp_path / "secret.txt")):
        with pytest.raises(DocumentUnavailable, match="outside the readable area"):
            await reader.read(ref)


async def test_the_reader_says_which_way_it_failed(tmp_path):
    reader = LocalFileReader([tmp_path])
    with pytest.raises(DocumentUnavailable, match="no such file"):
        await reader.read("absent.md")
    (tmp_path / "empty.md").write_text("   \n")
    with pytest.raises(DocumentUnavailable, match="empty"):
        await reader.read("empty.md")
    (tmp_path / "picture.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff\xfe binary")
    with pytest.raises(DocumentUnavailable, match="not UTF-8 text"):
        await reader.read("picture.png")


async def test_a_long_file_is_cut_and_the_text_says_so(tmp_path):
    (tmp_path / "long.md").write_text("x" * 5000)
    document = await LocalFileReader([tmp_path], max_chars=100).read("long.md")
    assert document.text.startswith("x" * 100)
    assert "truncated" in document.text, "a shorter document than the one on disk, silently"


async def test_a_multibyte_character_split_by_the_budget_is_not_binary(tmp_path):
    """The byte budget can land mid-character; that is not "the file is not text"."""
    (tmp_path / "accents.md").write_text("é" * 4000)
    document = await LocalFileReader([tmp_path], max_chars=10).read("accents.md")
    assert document.text.startswith("é" * 10)


# ------------------------------------------------------------- the capability


class FakeReader:
    """A DocumentReader that records what it was asked — no filesystem."""

    def __init__(self, docs: dict[str, str] | None = None):
        self.docs = docs or {}
        self.asked: list[str] = []

    async def read(self, ref: str) -> Document:
        self.asked.append(ref)
        try:
            return Document(id=ref, text=self.docs[ref], title=ref, source=f"/fake/{ref}")
        except KeyError:
            raise DocumentUnavailable(f"{ref!r}: no such file") from None


def state(*refs: str) -> dict:
    return {"query": "summarise it", "inputs": {SOURCE_FILES_INPUT_KEY: list(refs)}}


async def test_named_files_are_read_and_offered_to_the_model():
    reader = FakeReader({"a.md": "alpha text", "b.md": "beta text"})
    capability = ReadFilesCapability(reader)
    out = await capability.build().ainvoke(state("a.md", "b.md"))

    result = out["results"]["read_files"]
    assert result["ok"] is True
    assert [d["id"] for d in result["data"]["documents"]] == ["a.md", "b.md"]
    assert reader.asked == ["a.md", "b.md"]
    context = capability.render_context(result)
    assert context is not None and "alpha text" in context and "beta text" in context


async def test_one_unreadable_file_does_not_lose_the_others():
    reader = FakeReader({"a.md": "alpha text"})
    capability = ReadFilesCapability(reader)
    out = await capability.build().ainvoke(state("a.md", "gone.md"))

    result = out["results"]["read_files"]
    assert result["ok"] is True
    assert [d["id"] for d in result["data"]["documents"]] == ["a.md"]
    # the gap is stated, to the model and to the human — a model told nothing
    # about a missing file writes confidently around the hole
    context = capability.render_context(result)
    assert context is not None and "could NOT be read" in context and "gone.md" in context
    report = capability.render_report(result)
    assert report is not None and "not read" in report and "gone.md" in report


async def test_nothing_readable_is_a_failed_step_that_says_why():
    capability = ReadFilesCapability(FakeReader())
    out = await capability.build().ainvoke(state("gone.md"))

    result = out["results"]["read_files"]
    assert result["ok"] is False
    assert "gone.md" in (result["error"] or "")
    (error,) = out["errors"]
    assert error["recoverable"] is True     # the run degrades, it does not stop


async def test_only_the_first_files_are_read():
    reader = FakeReader({f"{i}.md": "text" for i in range(10)})
    capability = ReadFilesCapability(reader, max_files=2)
    await capability.build().ainvoke(state(*[f"{i}.md" for i in range(10)]))
    assert reader.asked == ["0.md", "1.md"]


def test_a_request_that_named_nothing_drops_the_step():
    capability = ReadFilesCapability(FakeReader())
    assert capability.is_applicable({"query": "q", "inputs": {"source_files": ["a.md"]}})
    # the key being present is not enough — it has to name something
    assert not capability.is_applicable({"query": "q", "inputs": {"source_files": []}})
    assert not capability.is_applicable({"query": "q", "inputs": {}})
    assert not capability.is_applicable({"query": "q"})


def test_a_single_reference_is_read_as_generously_as_a_list():
    assert named_files({SOURCE_FILES_INPUT_KEY: "a.md"}) == ["a.md"]
    assert named_files({SOURCE_FILES_INPUT_KEY: [" a.md ", "", None]}) == ["a.md", "None"]
    assert named_files({SOURCE_FILES_INPUT_KEY: 7}) == []
    assert named_files(None) == []


async def test_the_planner_drops_it_when_the_request_names_no_file(checkpointer):
    """End to end: registered, offered to the planner, and dropped as
    inapplicable — never planned as a step that can only report emptiness."""
    llm = FakeLLM({
        "planner": plan_json("read_files", "research", deps={"research": ["read_files"]}),
        "key aspects": '{"aspects": ["one"]}',
        "research notes": "notes",
        "ONLY the provided": "A sufficiently long final answer built from that context.",
    })
    registry = CapabilityRegistry([
        ReadFilesCapability(FakeReader({"a.md": "alpha"})),
        *default_capabilities(AgentContext(llm)),
    ])
    graph = build_agent(Deps(llm=llm), registry, checkpointer=checkpointer)
    out = await graph.ainvoke({"query": "study X", "job_id": "d1"},
                              config={"configurable": {"thread_id": "d1"}})
    assert [s["capability"] for s in out["plan"]["steps"]] == ["research"]
    assert "read_files" not in out["results"]
    assert out["plan"]["steps"][0]["depends_on"] == []   # the dropped name is pruned


# ------------------------------------------------------- composition + policy


def test_the_step_stays_out_of_the_registry_with_no_readable_root():
    """The rule every conditional step here follows: nothing backing it, no
    capability. A hand-assembled context declares no root."""
    names = [c.spec.name for c in default_capabilities(AgentContext(FakeLLM()))]
    assert "read_files" not in names


def test_the_readable_area_is_what_the_deployment_already_exposes(tmp_path):
    ctx = AgentContext(FakeLLM(), readable_roots=(str(tmp_path / "artifacts"),))
    assert readable_roots(ctx, DefaultResources()) == (str(tmp_path / "artifacts"),)
    # the --docs directory is already searchable and quotable, so opening a
    # file in it by name is not a widening
    with_docs = DefaultResources(documents_root=str(tmp_path / "docs"))
    assert readable_roots(ctx, with_docs) == (
        str(tmp_path / "artifacts"), str(tmp_path / "docs"))
    # and nothing at all is nothing at all
    assert readable_roots(AgentContext(FakeLLM()), DefaultResources()) == ()


async def test_a_composed_app_can_read_its_own_artifacts_and_nothing_else(tmp_path):
    """The loop the issue is about, and the fence around it, in one place."""
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the agent")
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(),
                          db="memory", reports_dir=str(tmp_path / "artifacts"))
    try:
        assert "read_files" in app.registry.names()
        reader = app.registry.get("read_files").reader
        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
        (tmp_path / "artifacts" / "mine.md").write_text("a report jobsmith wrote")

        assert (await reader.read("mine.md")).text == "a report jobsmith wrote"
        with pytest.raises(DocumentUnavailable):
            await reader.read(str(secret))
    finally:
        await app.aclose()


async def test_a_job_reads_the_report_the_previous_job_wrote(tmp_path):
    """Write, then read: the product's own loop, through the real path."""
    app = await build_app(llm=KeywordLLM(), chat_model=KeywordChatModel(),
                          db="memory", reports_dir=str(tmp_path / "artifacts"))
    try:
        first = await app.manager.run_job(
            (await app.manager.create_job("study the topic in depth")).job_id)
        assert first.status is JobStatus.DONE and first.report_path

        second = await app.manager.create_job(
            "make a one-pager out of the report",
            {SOURCE_FILES_INPUT_KEY: [first.report_path]},
        )
        second = await app.manager.run_job(second.job_id)
        assert second.status is JobStatus.DONE
        result = second.results["read_files"]
        assert result["ok"] is True, result.get("error")
        (document,) = result["data"]["documents"]
        assert first.job_id in document["text"], "it read the earlier deliverable"
    finally:
        await app.aclose()


# ------------------------------------------------------------- the approval


async def test_the_files_a_job_would_open_are_shown_and_approved(store, checkpointer,
                                                                 tmp_path):
    """A path the user never saw would be a second silent decision."""
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=[
        launch_call("summarise the attached report", "several steps",
                    source_files=["artifacts/abc.md", "  "]),
        AIMessage(content="Launched."),
    ])
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}

    out = await agent.ainvoke({"messages": [HumanMessage("summarise it")]}, cfg)
    (interrupted,) = out["__interrupt__"]
    assert interrupted.value["sources"] == ["artifacts/abc.md"]   # blanks dropped

    await agent.ainvoke(Command(resume={"approved": True}), cfg)
    (job,) = await manager.list_jobs(session_id=session.session_id)
    assert job.inputs[SOURCE_FILES_INPUT_KEY] == ["artifacts/abc.md"]
    for _ in range(200):
        await asyncio.sleep(0.01)
        job = await manager.get_job(job.job_id)
        if job.status in (JobStatus.DONE, JobStatus.FAILED):
            break


async def test_a_launch_that_names_no_file_carries_no_key(store, checkpointer, tmp_path):
    """Absent, not empty: the planner drops the step instead of planning one
    that can only report that nothing was given."""
    manager = make_manager(store, checkpointer, tmp_path)
    model = ScriptedChatModel(responses=[
        launch_call("analyse the alpha data", "several steps"),
        AIMessage(content="Launched."),
    ])
    session = ChatSession(manager, model, checkpointer=MemorySaver())
    agent = session.build()
    cfg = {"configurable": {"thread_id": session.session_id}}

    out = await agent.ainvoke({"messages": [HumanMessage("analyse it")]}, cfg)
    (interrupted,) = out["__interrupt__"]
    assert interrupted.value["sources"] == []

    await agent.ainvoke(Command(resume={"approved": True}), cfg)
    (job,) = await manager.list_jobs(session_id=session.session_id)
    assert SOURCE_FILES_INPUT_KEY not in job.inputs
