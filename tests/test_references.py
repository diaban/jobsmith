"""A run pointed at something by name — files (`read_files`, → 0060) or
earlier jobs (`prior_jobs`, → 0074): the properties both references share."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from support import chat_turn, launch_call, make_manager

from jobsmith.agents.default.prior_jobs import PriorJobsCapability, referenced_jobs
from jobsmith.agents.default.read_files import ReadFilesCapability, named_files
from jobsmith.chat import JobStarted
from jobsmith.core.state import FROM_JOBS_INPUT_KEY, SOURCE_FILES_INPUT_KEY


class Nothing:
    """A port that is never reached: these tests stop before any read."""


@pytest.mark.parametrize(("capability", "key"), [
    (ReadFilesCapability(Nothing()), SOURCE_FILES_INPUT_KEY),     # type: ignore[arg-type]
    (PriorJobsCapability(Nothing()), FROM_JOBS_INPUT_KEY),        # type: ignore[arg-type]
], ids=["read_files", "prior_jobs"])
def test_a_request_that_names_nothing_drops_the_step(capability, key):
    """The key being present is not the same as naming something."""
    assert capability.spec.requires_inputs == (key,)
    assert capability.is_applicable({"query": "q", "inputs": {key: ["x"]}})
    for inputs in ({key: []}, {}, None):
        state = {"query": "q"} if inputs is None else {"query": "q", "inputs": inputs}
        assert not capability.is_applicable(state)


@pytest.mark.parametrize(("parse", "key", "cases"), [
    (named_files, SOURCE_FILES_INPUT_KEY,
     [("a.md", ["a.md"]), ([" a.md ", "", None], ["a.md", "None"]), (7, [])]),
    (referenced_jobs, FROM_JOBS_INPUT_KEY,
     [("j1", ["j1"]), ([" j1 ", ""], ["j1"]), (3, [])]),
], ids=["read_files", "prior_jobs"])
def test_a_single_reference_is_read_as_generously_as_a_list(parse, key, cases):
    assert parse(None) == []
    for given, read in cases:
        assert parse({key: given}) == read


async def test_a_launch_that_names_nothing_carries_neither_key(store, checkpointer, tmp_path):
    """Absent, not empty: the planner then drops the step rather than planning
    one that can only report that nothing was given — and the notice says none."""
    manager = make_manager(store, checkpointer, tmp_path)
    events, _, session_id = await chat_turn(manager, [
        launch_call("analyse the alpha data", "several steps"), AIMessage(content="Done."),
    ], "analyse it")

    (started,) = [e for e in events if isinstance(e, JobStarted)]
    assert started.sources == [] and started.from_jobs == []
    (job,) = await manager.list_jobs(session_id=session_id)
    assert SOURCE_FILES_INPUT_KEY not in job.inputs and FROM_JOBS_INPUT_KEY not in job.inputs
