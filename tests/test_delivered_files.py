"""The generator is told which files the run delivers, and that there are no others (#77).

A run whose plan never touched `slide_deck` delivered a text asserting "a
slide deck exists alongside this document" — in the generator prompt's own
words, since that prompt talked about a produced file by example and never
said which files the run actually produced. These pin the list the generator
(and the refiner, which restates the same contract) now sees, in the three
shapes it has: no file, the requested document, and a file a step declared.

The direct route is told the same list, and — only when that list names the
answer itself — that its reply is that document (#80).
"""
from __future__ import annotations

from typing import Any

from conftest import FakeLLM

from jobsmith.core.artifacts import ArtifactRef, artifact_meta
from jobsmith.core.deps import Deps
from jobsmith.core.generation import (
    DIRECT_DOCUMENT_RULE,
    FILES_HEADING,
    DirectResponder,
    Generator,
    Refiner,
    delivered_files,
    delivered_files_note,
)
from jobsmith.core.profile import AgentProfile
from jobsmith.core.registry import CapabilityRegistry


def _state(**extra: Any) -> dict[str, Any]:
    return {"query": "compare heat pumps and boilers", "merged_context": "material", **extra}


def _deck_state(**extra: Any) -> dict[str, Any]:
    return _state(
        plan={"steps": [{"capability": "research", "depends_on": []},
                        {"capability": "slide_deck", "depends_on": ["research"]}],
              "rationale": ""},
        results={
            "research": {"ok": True, "data": {}},
            "slide_deck": {"ok": True, "data": {},
                           "meta": artifact_meta(ArtifactRef("/out/abc/deck.pptx",
                                                             title="Heating options"))},
        },
        **extra,
    )


def test_a_run_with_no_file_is_told_there_is_none():
    assert delivered_files(_state()) == []
    note = delivered_files_note(_state(document_formats=None))
    assert note.startswith(f"{FILES_HEADING}: none.")
    assert "Name no other file" in note


def test_an_empty_format_list_is_no_file_either():
    # `[]` and silence are two facts on the job, and the same answer here:
    # neither writes a document (#96).
    assert delivered_files(_state(document_formats=[])) == []


def test_the_requested_document_is_listed_in_its_formats():
    files = delivered_files(_state(document_formats=["markdown", "html"]))
    assert files == ["this answer itself, written to a file as markdown, html"]


def test_a_file_a_step_declared_is_listed_by_its_title():
    files = delivered_files(_deck_state())
    assert files == ["Heating options (pptx)"]
    note = delivered_files_note(_deck_state(document_formats=["markdown"]))
    assert "the complete list" in note
    assert "- this answer itself" in note and "- Heating options (pptx)" in note


async def test_the_generator_sees_the_list_between_the_request_and_the_material():
    llm = FakeLLM(default="the answer")
    await Generator(Deps(llm=llm), AgentProfile()).run(_state())
    user = llm.calls[0]["messages"][1]["content"]
    assert f"{FILES_HEADING}: none." in user
    assert user.index("Query:") < user.index(FILES_HEADING) < user.index("Context:")


async def test_the_refiner_sees_the_same_list():
    llm = FakeLLM(default="the answer")
    await Refiner(Deps(llm=llm), AgentProfile()).run(
        _deck_state(draft_answer="draft", validation_issues=["x"]))
    user = llm.calls[0]["messages"][1]["content"]
    assert "- Heating options (pptx)" in user




async def _direct_prompt(**extra: Any) -> str:
    llm = FakeLLM(default="hello")
    await DirectResponder(Deps(llm=llm), CapabilityRegistry([]), AgentProfile()).run(
        _state(**extra))
    return llm.calls[0]["messages"][0]["content"]


async def test_a_direct_reply_is_told_there_is_no_file_and_stays_a_turn():
    prompt = await _direct_prompt(document_formats=None)
    assert f"{FILES_HEADING}: none." in prompt
    assert DIRECT_DOCUMENT_RULE not in prompt


async def test_a_direct_reply_asked_for_as_a_file_is_written_as_that_document():
    """→ 0080: the direct route keeps the request, and writes for the file's reader."""
    prompt = await _direct_prompt(document_formats=["markdown"])
    assert "- this answer itself, written to a file as markdown" in prompt
    assert DIRECT_DOCUMENT_RULE in prompt
