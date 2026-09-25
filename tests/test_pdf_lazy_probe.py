"""The PDF engine is loaded when a PDF is asked for, not to compose the app (#108).

`import weasyprint` costs ~4 s, and composing the app used to pay it on every
command, daemon start and test process, because listing what a deployment can
render constructed every Reporter. What 0034 guaranteed survives, moved to
the first real need:

- composing the app loads no engine — and still OFFERS pdf where the extra is
  installed (`available_formats` asks the class, not an instance);
- a request that names PDF is refused in `create_job`, before any work, with
  0034's two distinct messages — as a `ValueError`, the refusal every door
  already says (the chat's "NOT launched", the API's 400);
- the document step, which chooses a format mid-run, proves its choice first
  and drops one that cannot render — no job reaches its end to find out;
- the probe runs once per process, and a failed one stops the offer;
- a deployment whose own default is PDF still fails at startup.

The startup properties run in a subprocess: `sys.modules` in this process
already holds whatever every earlier test imported.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import FakeLLM
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from support import launch_call, make_manager, make_session, requires_pdf

from jobsmith.core.deps import Deps
from jobsmith.core.document import DocumentIntent
from jobsmith.jobs import report_pdf
from jobsmith.jobs.models import JobStatus
from jobsmith.jobs.report import available_formats, renderable_formats

MISSING_PACKAGE = r"jobsmith\[pdf\]"
MISSING_LIBRARIES = "pango"


@pytest.fixture(autouse=True)
def unprobed(monkeypatch):
    """Every test starts in a process that has not probed the engine yet."""
    monkeypatch.setattr(report_pdf, "_probed", None)


def no_distribution(monkeypatch) -> None:
    """`import weasyprint` halts with ImportError: the extra is not installed."""
    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setitem(sys.modules, "weasyprint", None)


def no_libraries(monkeypatch) -> None:
    """The distribution is found, and its import raises OSError — what cffi
    does when it cannot dlopen pango. `find_spec` still finds it: this is the
    machine where PDF is offered and cannot be rendered."""
    real_find_spec = __import__("importlib.util").util.find_spec

    class NoLibraries:
        def find_spec(self, name, path=None, target=None):
            if name == "weasyprint":
                raise OSError("cannot load library 'libpango-1.0.so.0'")
            return None

    monkeypatch.delitem(sys.modules, "weasyprint", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoLibraries(), *sys.meta_path])
    # The offer is answered by locating the distribution, which the finder
    # above would refuse too; on the machine simulated it is located.
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec",
                        lambda name, *a: object() if name == "weasyprint"
                        else real_find_spec(name, *a))


# --------------------------------------------------------------- at startup

def _compose_in_a_fresh_process(tmp_path: Path, **env: str) -> dict:
    script = textwrap.dedent("""
        import asyncio, json, sys
        from jobsmith.app.agent import build_app
        from jobsmith.jobs.report import available_formats

        async def main():
            app = await build_app(db="memory", llm="fake")
            offered = available_formats(app.registry)
            await app.aclose()
            return offered

        offered = asyncio.run(main())
        print(json.dumps({"loaded": "weasyprint" in sys.modules,
                          "offered": offered}))
    """)
    environ = {**os.environ, "XDG_DATA_HOME": str(tmp_path),
               "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", "TAVILY_API_KEY": "",
               "JOBSMITH_REPORT_FORMAT": "", **env}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=environ, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_composing_the_app_does_not_load_the_pdf_engine(tmp_path):
    seen = _compose_in_a_fresh_process(tmp_path)
    assert seen["loaded"] is False
    if report_pdf.engine_installed():
        # ...and loading nothing did not cost the offer.
        assert "pdf" in seen["offered"]


@requires_pdf
def test_a_deployment_whose_default_is_pdf_still_probes_at_startup(tmp_path):
    """`$JOBSMITH_REPORT_FORMAT=pdf` is the deployment asking for PDF before
    any request does, so a machine that cannot render one says so when it
    starts — 0034's rule, kept for the one case it was always about."""
    seen = _compose_in_a_fresh_process(tmp_path, JOBSMITH_REPORT_FORMAT="pdf")
    assert seen["loaded"] is True


def test_the_offer_is_answered_without_the_engine(monkeypatch):
    def never():
        raise AssertionError("offering pdf must not probe the engine")

    monkeypatch.setattr(report_pdf, "_engine", never)
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: object())
    assert "pdf" in available_formats()     # the distribution is there
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: None)
    assert "pdf" not in available_formats()  # it is not


# ----------------------------------------------------------- in create_job

@pytest.mark.parametrize(("break_it", "message"), [
    (no_distribution, MISSING_PACKAGE),
    (no_libraries, MISSING_LIBRARIES),
])
async def test_a_pdf_request_is_refused_in_create_job_with_the_right_message(
    store, checkpointer, tmp_path, monkeypatch, break_it, message
):
    break_it(monkeypatch)
    mgr = make_manager(store, checkpointer, tmp_path)
    with pytest.raises(ValueError, match=message):
        await mgr.create_job("compare the chairs", formats=["markdown", "pdf"])
    assert await mgr.list_jobs() == []       # refused before anything exists


async def test_the_probe_runs_once_and_a_failure_stops_the_offer(
    store, checkpointer, tmp_path, monkeypatch
):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError(report_pdf._MISSING_LIBRARIES)

    monkeypatch.setattr(report_pdf, "_engine", broken)
    monkeypatch.setattr(report_pdf.importlib.util, "find_spec", lambda name, *a: object())
    assert "pdf" in available_formats()      # offered: installed, not yet proved
    mgr = make_manager(store, checkpointer, tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match=MISSING_LIBRARIES):
            await mgr.create_job("compare the chairs", formats=["pdf"])
    assert calls == [1]
    assert "pdf" not in available_formats()  # proved unrenderable: no longer offered


async def test_the_chat_says_not_launched_and_what_is_possible_instead(
    store, checkpointer, tmp_path, monkeypatch
):
    """The chat catches refusals as `ValueError`: an engine that cannot load
    used to escape it as `RuntimeError`, unseen while the probe was at startup."""
    no_libraries(monkeypatch)
    session, _ = make_session(store, checkpointer, tmp_path, [
        launch_call("compare the chairs", "several steps", formats=["pdf"]),
        AIMessage(content="PDF cannot be written here."),
    ])
    out = await session.build().ainvoke(
        {"messages": [HumanMessage("compare the chairs as a pdf")]},
        {"configurable": {"thread_id": "lazy-pdf"}})

    assert await session.manager.list_jobs() == []
    reply = next(m for m in out["messages"] if isinstance(m, ToolMessage)).content
    assert "NOT launched" in reply and MISSING_LIBRARIES in reply
    possible = reply.split("formats available here:")[1]
    assert "markdown" in possible and "pdf" not in possible


@requires_pdf
async def test_a_pdf_request_where_the_engine_runs_still_renders(
    store, checkpointer, tmp_path
):
    mgr = make_manager(store, checkpointer, tmp_path)
    job = await mgr.create_job("compare the chairs", formats=["pdf"])
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE and done.error is None
    assert done.report_path and done.report_path.endswith(".pdf")
    assert Path(done.report_path).read_bytes().startswith(b"%PDF-")


# ------------------------------------------------------ in the document step

def pdf_node(confirm) -> DocumentIntent:
    reply = json.dumps({"document": "named", "formats": ["pdf"]})
    return DocumentIntent(Deps(llm=FakeLLM({"document step": reply})),
                          ("html", "markdown", "pdf"), confirm=confirm)


async def test_the_document_step_drops_a_pdf_it_cannot_render(monkeypatch):
    """Offered because installed, chosen because asked — and proved before it
    is written, so the job does not find out at its end."""
    no_libraries(monkeypatch)
    assert await pdf_node(renderable_formats).run({"query": "as a pdf"}) == {}


@requires_pdf
async def test_the_document_step_keeps_a_pdf_it_can_render():
    assert await pdf_node(renderable_formats).run({"query": "as a pdf"}) == {
        "document_formats": ["pdf"]}
