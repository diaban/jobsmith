"""What the requester asked the DOCUMENT to be (#55).

A user asked, in the conversation, for a report called
`rapport_chaises_gabarit.md` and a PDF of the same content. The job ran,
answered, and the deliverable **promised both files in its own prose** — while
the product wrote one file called `2c223bc5….md` and no PDF. Nothing
malfunctioned: the request travelled to a layer with no opinion about it.

Three decisions are pinned here, and they are three, not one — a name is not a
title and neither is a format:

- the name reaches the file, and cannot reach anywhere else on the way;
- the title reaches the heading, and a job that asked for neither still gets
  both derived from the request (#54);
- the formats compose per job, and one that cannot be rendered here is refused
  in front of whoever asked rather than three minutes later.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from test_chat import launch_call, make_session
from test_jobs import make_manager

from jobsmith.core.paths import PathRefused
from jobsmith.jobs.report import (
    NAME_MAX,
    available_formats,
    deliverable_filenames,
    document_stem,
)

NAME = "chair_comparison"


# ---------------------------------------------------------------- the name

async def test_the_deliverable_takes_the_name_and_keeps_the_job_its_own_folder(
    store, checkpointer, tmp_path
):
    """A job id is unique and says nothing; a name says everything and is
    unique to nobody — so a named deliverable lands in the job's own
    directory, where its annexes already are."""
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs", document_name=NAME)
    done = await mgr.run_job(job.job_id)

    assert done.report_path == str(
        tmp_path / "artifacts" / job.job_id / f"{NAME}.md")
    assert "final answer" in (tmp_path / "artifacts" / job.job_id / f"{NAME}.md").read_text()
    # and it is the file the outputs point at, not one of two records
    assert [o.name for o in done.outputs] == [f"{NAME}.md"]


async def test_an_unnamed_job_writes_exactly_where_it_always_did(
    store, checkpointer, tmp_path
):
    """Nothing about a job with no name of its own needed a directory."""
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs")
    done = await mgr.run_job(job.job_id)

    assert done.report_path == str(tmp_path / "artifacts" / f"{job.job_id}.md")


async def test_two_jobs_with_one_name_keep_two_files(store, checkpointer, tmp_path):
    """The reason the folder is not cosmetic: a name is not unique, and a
    deliverable silently overwritten is the same defect as one nobody can
    find (#28)."""
    mgr = make_manager(store, checkpointer, tmp_path)
    first = await mgr.run_job(
        (await mgr.create_job("compare the chairs", document_name=NAME)).job_id)
    second = await mgr.run_job(
        (await mgr.create_job("compare them again", document_name=NAME)).job_id)

    assert first.report_path != second.report_path
    assert {p.name for p in (tmp_path / "artifacts").rglob("*.md")} == {f"{NAME}.md"}
    assert len(list((tmp_path / "artifacts").rglob("*.md"))) == 2


@pytest.mark.parametrize("refused", [
    "../escape",                 # traversal, the reason `safe_name` exists
    "sub/dir/report",            # a location, not a name: never flattened
    "/etc/passwd",
    ".",
    "x" * (NAME_MAX + 1),        # a name that long is a title
])
async def test_a_name_that_is_not_a_filename_is_refused_before_a_job_exists(
    store, checkpointer, tmp_path, refused
):
    """The caller says what the file is CALLED; where it goes is ours. And it
    refuses at `create_job`, the last point at which whoever asked is still
    listening."""
    mgr = make_manager(store, checkpointer, tmp_path)
    with pytest.raises(ValueError):          # PathRefused is one
        await mgr.create_job("compare the chairs", document_name=refused)
    assert await mgr.list_jobs() == []


def test_a_known_extension_is_dropped_and_an_unknown_suffix_is_not():
    """`rapport.md` plus a PDF is ONE document under two extensions, and the
    formats decide those. `v1.2` is a name."""
    assert document_stem("rapport.md") == "rapport"
    assert document_stem("rapport.pdf") == "rapport"
    assert document_stem("notes v1.2") == "notes v1.2"
    assert document_stem("  rapport  ") == "rapport"


# ---------------------------------------------------------------- the title

async def test_the_requested_title_is_the_heading_and_the_name_never_decides_it(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs", document_name=NAME,
                               document_title="Comparatif des chaises")
    await mgr.run_job(job.job_id)

    written = (tmp_path / "artifacts" / job.job_id / f"{NAME}.md").read_text()
    assert written.startswith("# Comparatif des chaises\n")
    assert NAME not in written.splitlines()[0]       # the name is not the title


async def test_a_job_that_asked_for_neither_still_derives_both(
    store, checkpointer, tmp_path
):
    """#54 stays the floor under #55: no title asked for, one derived from the
    request rather than a job id or an empty heading."""
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs for a home office")
    done = await mgr.run_job(job.job_id)

    assert done.report_path is not None
    first = open(done.report_path).readline()
    assert first.startswith("# compare the chairs for a home office")


# ---------------------------------------------------------------- the formats

async def test_a_job_composes_the_formats_it_asked_for(store, checkpointer, tmp_path):
    """Formats were already a list (#28); this is only a way to say it — and
    the FIRST one is still the main deliverable."""
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs", document_name=NAME,
                               formats=["html", "markdown"])
    done = await mgr.run_job(job.job_id)

    assert [(o.role, o.format, o.name) for o in done.outputs] == [
        ("main", "html", f"{NAME}.html"),
        ("alternate", "markdown", f"{NAME}.md"),
    ]
    assert done.report_path.endswith(".html")


async def test_a_job_that_asked_for_nothing_uses_what_the_deployment_composed(
    store, checkpointer, tmp_path
):
    """The default path is untouched: no formats asked for, the composed
    reporter writes, and the factory is never consulted."""
    mgr = make_manager(store, checkpointer, tmp_path)
    mgr.reporter_factory = lambda formats: pytest.fail("should not be consulted")
    done = await mgr.run_job((await mgr.create_job("compare the chairs")).job_id)
    assert done.report_path.endswith(".md")


async def test_a_format_nothing_can_render_is_refused_in_front_of_the_asker(
    store, checkpointer, tmp_path
):
    """`.[pdf]` is this project's one deployment constraint, and an unknown
    name is the same question: both must be answered where the person who
    asked can still see it, never at the end of a run that spent its tokens."""
    mgr = make_manager(store, checkpointer, tmp_path)
    with pytest.raises(ValueError, match="unknown report format"):
        await mgr.create_job("compare the chairs", formats=["docx"])
    assert await mgr.list_jobs() == []


def test_the_filenames_shown_are_the_filenames_written():
    """What a front-end puts on the approval card is what lands on disk."""
    assert deliverable_filenames(NAME, ["markdown", "pdf"]) == [
        f"{NAME}.md", f"{NAME}.pdf"]
    # nothing to call them yet: there is no job id at proposal time
    assert deliverable_filenames("", ["markdown"]) == []
    assert "markdown" in available_formats()


# ---------------------------------------------------------------- the chat

CFG = {"configurable": {"thread_id": "doc-1"}}


async def test_the_proposal_shows_the_document_and_the_approval_creates_it(
    store, checkpointer, tmp_path
):
    """The card is where a name stops being a silent decision: what the user
    is shown is what reaches `create_job`."""
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "several steps",
                    document_name="chair_comparison.md",   # the model wrote an extension
                    document_title="Comparatif des chaises",
                    formats=["markdown", "html"]),
        AIMessage(content="launched"),
    ])
    agent = session.build()

    out = await agent.ainvoke(
        {"messages": [HumanMessage("compare the chairs please")]}, CFG)
    (proposal,) = out["__interrupt__"]
    assert proposal.value["document_name"] == NAME        # the extension was dropped
    assert proposal.value["document_title"] == "Comparatif des chaises"
    assert proposal.value["formats"] == ["markdown", "html"]
    assert await session.manager.list_jobs() == []        # nothing created yet

    await agent.ainvoke(Command(resume={"approved": True}), CFG)
    (job,) = await session.manager.list_jobs()
    assert (job.document_name, job.document_title) == (NAME, "Comparatif des chaises")
    assert job.formats == ["markdown", "html"]


async def test_a_refused_document_never_reaches_the_card(store, checkpointer, tmp_path):
    """A format nothing can render here is the model's to fix on the spot, so
    the answer goes back to it as text and no approval is ever asked for."""
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "several steps", formats=["docx"]),
        AIMessage(content="I cannot write docx here."),
    ])
    agent = session.build()

    out = await agent.ainvoke(
        {"messages": [HumanMessage("compare the chairs as a docx")]}, CFG)

    assert "__interrupt__" not in out                     # never proposed
    tool_reply = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert "NOT launched" in tool_reply.content
    assert "markdown" in tool_reply.content               # what IS possible here
    assert await session.manager.list_jobs() == []


def test_the_path_policy_is_the_one_from_the_reading_side():
    """#60 answered this for reading; #55 is the same question for writing,
    and `core/paths.py` says so in its own docstring."""
    with pytest.raises(PathRefused):
        document_stem("../../etc/passwd")
