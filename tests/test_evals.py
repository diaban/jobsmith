"""The eval harness itself — and the structural tier it runs.

Two things are pinned here:

1. **The deterministic tier stays at 100%.** That is the CI gate the harness
   exists for: it runs on the fakes, needs no API key, and turns a broken
   router/planner/report prompt into a failing test instead of a code review
   someone has to eyeball. The LLM tier is deliberately absent — it is
   stochastic and must never be able to fail CI.
2. **The checks themselves are correct.** A scorer that never fires is worse
   than no scorer, so each property is fed an observation that violates it.
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from evals import cases_for, run_suite, score
from evals.cases import GOLDEN_CASES, EvalCase
from evals.harness import Observation
from evals.results import load_baseline, render_summary, summarize, write_result
from evals.scoring import CHECK_NAMES, step_failure_rate

# ---------------------------------------------------------------- the tier


@pytest.fixture(scope="module")
def structural_run():
    """One real pass of the structural tier, shared by the assertions below."""

    async def go():
        cases = cases_for("structural")
        started = time.perf_counter()
        observations, context = await run_suite(cases, provider="fake")
        return cases, summarize(cases, observations, tier="structural", context=context,
                                duration_s=time.perf_counter() - started)

    return asyncio.run(go())


def test_structural_tier_is_perfect(structural_run):
    """The fakes are deterministic: anything under 100% is a real regression."""
    _, result = structural_run
    assert result.pass_rate == 1.0, "\n".join(
        f"{f['case']} {f['check']}: {f['detail']}" for f in result.failures
    )


def test_structural_tier_actually_exercises_every_check(structural_run):
    """A check nothing applies to would silently protect nothing."""
    _, result = structural_run
    never_applied = [n for n in CHECK_NAMES if not result.checks[n]["applicable"]]
    assert never_applied == []


def test_structural_tier_covers_both_routes_and_the_guard(structural_run):
    cases, result = structural_run
    routes = {c.expect_route for c in cases}
    assert {"plan", "direct", None} <= routes
    assert any(c.expect_terminal == "user_error" for c in cases)
    assert result.registry, "the composed agent registered no capability"


def test_case_ids_are_unique():
    ids = [c.id for c in GOLDEN_CASES]
    assert len(ids) == len(set(ids))


def test_check_names_match_the_scorers():
    """CHECK_NAMES drives the summary's column order — it must not drift."""
    produced = [c.name for c in score(GOLDEN_CASES[0], Observation("x", "x"))]
    assert produced == list(CHECK_NAMES)


# ---------------------------------------------------------------- the checks

PLAN_CASE = EvalCase(id="c", query="q", expect_route="plan", min_steps=2,
                     must_include=("analysis",))

ANSWER = "a final answer long enough to pass the length floor"


def _obs(**kwargs) -> Observation:
    base = {
        "case_id": "c",
        "query": "q",
        "job_id": "job1",
        "route": "plan",
        "registry": ("research", "analysis"),
        "terminal_kind": "answer",
        "final_answer": ANSWER,
        "plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "analysis", "depends_on": ["research"]},
        ],
        "results": {"research": {"ok": True}, "analysis": {"ok": True}},
    }
    base.update(kwargs)
    obs = Observation(**base)
    if "report_text" not in kwargs:
        obs.report_text = (
            f"# title\n\n{obs.final_answer}\n\n"
            "- Request: q\n- Job: job1\n\n| research | analysis |\n"
        )
    return obs


def _status(case: EvalCase, obs: Observation, name: str) -> str:
    return next(c.status for c in score(case, obs) if c.name == name)


def test_a_clean_observation_passes_everything():
    statuses = {c.name: c.status for c in score(PLAN_CASE, _obs())}
    assert "fail" not in statuses.values(), statuses


@pytest.mark.parametrize(
    ("check", "broken"),
    [
        ("router_route", {"route": "direct"}),
        ("terminal_kind", {"terminal_kind": "escalated"}),
        ("plan_present", {"plan_steps": []}),
        ("plan_names_known", {"plan_steps": [{"capability": "nope", "depends_on": []}]}),
        ("plan_no_duplicates", {"plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "research", "depends_on": []},
        ]}),
        ("plan_deps_satisfiable", {"plan_steps": [
            {"capability": "research", "depends_on": ["ghost"]},
            {"capability": "analysis", "depends_on": []},
        ]}),
        ("plan_acyclic", {"plan_steps": [
            {"capability": "research", "depends_on": ["analysis"]},
            {"capability": "analysis", "depends_on": ["research"]},
        ]}),
        ("plan_size", {"plan_steps": [{"capability": "research", "depends_on": []}]}),
        ("plan_required_steps", {"plan_steps": [
            {"capability": "research", "depends_on": []},
            {"capability": "critique", "depends_on": []},
        ]}),
        ("steps_all_ran", {"results": {"research": {"ok": True}}}),
        ("steps_all_ok", {"results": {"research": {"ok": True}, "analysis": {"ok": False}}}),
        ("report_written", {"report_text": None}),
        ("report_title", {"report_text": "no heading at all\n"}),
        ("report_answer", {"report_text": "# t\n\nsomething else entirely\n"}),
        ("report_provenance", {"report_text": f"# t\n\n{ANSWER}\n"}),
        ("report_covers_plan", {
            "report_text": f"# t\n\n{ANSWER}\n- Job: job1\n- Request: q\n"}),
        ("run_completed", {"error": "boom"}),
    ],
)
def test_each_check_fires_on_its_own_violation(check, broken):
    assert _status(PLAN_CASE, _obs(**broken), check) == "fail"


def test_a_broken_run_skips_the_downstream_checks():
    """One harness exception must not be counted as seventeen prompt failures."""
    checks = {c.name: c.status for c in score(PLAN_CASE, _obs(error="boom"))}
    assert checks["run_completed"] == "fail"
    assert set(checks.values()) == {"fail", "skip"}
    assert sum(1 for s in checks.values() if s == "fail") == 1


def test_inapplicable_checks_do_not_dilute_the_score():
    direct = EvalCase(id="c", query="q", expect_route="direct")
    obs = _obs(route="direct", plan_steps=[], results={})
    checks = score(direct, obs)
    assert all(c.status != "fail" for c in checks)
    assert any(c.status == "skip" for c in checks)


def test_step_failure_rate():
    assert step_failure_rate(_obs()) == 0.0
    assert step_failure_rate(_obs(results={"a": {"ok": True}, "b": {"ok": False}})) == 0.5
    assert step_failure_rate(_obs(results={})) is None


# ---------------------------------------------------------------- results io

def _result(cases, observations):
    return summarize(cases, observations, tier="structural",
                     context={"agent": "default", "provider": "fake",
                              "registry": ["research", "analysis"], "repeat": 1},
                     duration_s=0.1)


def test_results_are_written_and_found_as_a_baseline(tmp_path):
    cases = [PLAN_CASE]
    first = _result(cases, [_obs()])
    path = write_result(first, tmp_path)
    assert json.loads(path.read_text())["tier"] == "structural"

    second = _result(cases, [_obs(route="direct")])
    baseline = load_baseline(second, tmp_path)
    assert baseline is not None
    assert baseline.run_id == first.run_id
    assert baseline.pass_rate == 1.0


def test_a_baseline_from_another_provider_is_not_comparable(tmp_path):
    other = _result([PLAN_CASE], [_obs()])
    other.provider = "anthropic"
    write_result(other, tmp_path)
    mine = _result([PLAN_CASE], [_obs()])
    assert load_baseline(mine, tmp_path) is None


def test_a_baseline_over_a_different_case_set_is_not_comparable(tmp_path):
    """A `--case` slice must not become the baseline of a full run."""
    slice_run = _result([PLAN_CASE], [_obs()])
    write_result(slice_run, tmp_path)
    full = _result([PLAN_CASE, EvalCase(id="other", query="q")], [_obs()])
    assert load_baseline(full, tmp_path) is None


def test_the_summary_shows_the_delta_against_a_baseline():
    cases = [PLAN_CASE]
    before = _result(cases, [_obs()])
    after = _result(cases, [_obs(route="direct")])
    text = render_summary(after, before)
    assert "baseline:" in text
    assert "router_route" in text
    assert "-100.0" in text                      # the check that regressed
    assert "expected 'plan', got 'direct'" in text


# ---------------------------------------------------------------- the cli

def test_cli_runs_and_gates_the_structural_tier(capsys):
    from evals.__main__ import main

    assert main(["--llm", "fake", "--no-write"]) == 0
    assert "overall" in capsys.readouterr().out


def test_cli_lists_the_golden_set(capsys):
    from evals.__main__ import main

    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(case.id in out for case in GOLDEN_CASES)
