"""Token/cost accounting: the tally, the ledger, attribution, and reporting.

The interesting property is end-to-end: every LLM call a job makes is booked
exactly once, attributed to the graph step that made it, and readable
afterwards from the Job, the store and the report.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import FakeLLM, plan_json
from langgraph.constants import END

from jobsmith.app.providers import KeywordLLM
from jobsmith.clients import AnthropicLLMClient, OpenAILLMClient
from jobsmith.core.builder import build_agent
from jobsmith.core.capability import Capability, CapabilityBaseState, CapabilitySpec
from jobsmith.core.deps import Deps
from jobsmith.core.registry import CapabilityRegistry
from jobsmith.core.usage import (
    UNATTRIBUTED,
    ModelPrice,
    Usage,
    current_ledger,
    current_scope,
    estimate_cost,
    price_for,
    record_usage,
    reset_price_overrides,
    usage_ledger,
)
from jobsmith.jobs.manager import JobManager
from jobsmith.jobs.models import Job, JobOutput, JobStatus
from jobsmith.jobs.report import MarkdownReport, build_document, format_step_usage, format_usage

# ---------------------------------------------------------------- the tally


def test_usage_adds_tokens_calls_and_models():
    a = Usage(input_tokens=100, output_tokens=10, calls=1, cost_usd=0.5, models=("m1",))
    b = Usage(input_tokens=40, cached_input_tokens=7, output_tokens=2, calls=1,
              cost_usd=0.25, models=("m2",))
    total = a + b
    assert (total.input_tokens, total.output_tokens, total.cached_input_tokens) == (140, 12, 7)
    assert total.calls == 2
    assert total.cost_usd == pytest.approx(0.75)
    assert total.models == ("m1", "m2")
    assert total.total_tokens == 159


def test_cost_is_none_only_when_nothing_could_be_priced():
    unpriced = Usage(input_tokens=10, calls=1)
    assert (unpriced + unpriced).cost_usd is None
    # a partial sum is still worth showing — it is an estimate, not a bill
    assert (unpriced + Usage(calls=1, cost_usd=2.0)).cost_usd == pytest.approx(2.0)


def test_usage_is_falsy_without_calls_and_round_trips_through_json():
    assert not Usage()
    assert Usage(calls=1)
    usage = Usage(input_tokens=3, output_tokens=4, cached_input_tokens=1, calls=2,
                  cost_usd=0.125, models=("m",))
    assert Usage.from_dict(json.loads(json.dumps(usage.to_dict()))) == usage
    assert Usage.from_dict(None) == Usage()          # a job that predates tracking


# ---------------------------------------------------------------- prices


def test_price_lookup_is_longest_prefix():
    # a dated snapshot is priced by its family
    assert price_for("claude-opus-5-20260101") == price_for("claude-opus-5")
    # ...and the more specific row wins over the shorter one
    assert price_for("claude-sonnet-4-6").input == 3.0
    assert price_for("claude-sonnet-5").input == 2.0
    assert price_for("some-local-model") is None


def test_cost_math_including_cached_input_discount():
    price = ModelPrice(input=5.0, output=25.0)               # USD per million
    cost = price.cost(Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == pytest.approx(30.0)
    # cache reads default to a 90% discount off the input rate
    assert price.cost(Usage(cached_input_tokens=1_000_000)) == pytest.approx(0.5)
    assert ModelPrice(1.0, 1.0, cached_input=0.25).cost(
        Usage(cached_input_tokens=1_000_000)) == pytest.approx(0.25)
    assert estimate_cost("nothing-known", Usage(input_tokens=10, calls=1)) is None


def test_prices_are_overridable_without_touching_code(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBSMITH_PRICES", '{"my-gateway/llama": {"input": 0.2, "output": 0.6}}')
    reset_price_overrides()
    try:
        assert price_for("my-gateway/llama-70b").output == 0.6
        # an override also replaces a built-in row (vendors reprice)
        monkeypatch.setenv("JOBSMITH_PRICES", '{"claude-opus-5": {"input": 1, "output": 2}}')
        reset_price_overrides()
        assert price_for("claude-opus-5").input == 1.0
        # from a file, too
        prices = tmp_path / "prices.json"
        prices.write_text('{"x-model": {"input": 9, "output": 9}}')
        monkeypatch.setenv("JOBSMITH_PRICES", str(prices))
        reset_price_overrides()
        assert price_for("x-model").input == 9.0
        # a broken table must never break a run
        monkeypatch.setenv("JOBSMITH_PRICES", "{not json")
        reset_price_overrides()
        assert price_for("claude-opus-5").input == 5.0
    finally:
        monkeypatch.delenv("JOBSMITH_PRICES", raising=False)
        reset_price_overrides()


# ---------------------------------------------------------------- the ledger


def test_record_usage_without_a_ledger_is_a_noop():
    assert current_ledger() is None
    usage = record_usage("claude-opus-5", input_tokens=1000, output_tokens=100)
    assert usage.calls == 1                       # still returned to the caller
    assert usage.cost_usd == pytest.approx((1000 * 5.0 + 100 * 25.0) / 1_000_000)


def test_ledger_accumulates_per_scope_and_totals():
    with usage_ledger() as ledger:
        record_usage("claude-opus-5", input_tokens=10, output_tokens=1, scope="planner")
        record_usage("claude-opus-5", input_tokens=20, output_tokens=2, scope="research")
        record_usage("claude-opus-5", input_tokens=30, output_tokens=3, scope="research")
        assert ledger.get("research").calls == 2
        assert ledger.get("research").input_tokens == 50
        assert ledger.get("nobody") == Usage()
        assert set(ledger.by_scope()) == {"planner", "research"}
        assert ledger.total().calls == 3
        assert ledger.total().input_tokens == 60


def test_nested_ledger_never_bills_its_parent():
    with usage_ledger() as outer:
        record_usage("claude-opus-5", output_tokens=1, scope="a")
        with usage_ledger() as inner:
            record_usage("claude-opus-5", output_tokens=1, scope="a")
            assert inner.total().calls == 1
        record_usage("claude-opus-5", output_tokens=1, scope="a")
        assert outer.total().calls == 2


def test_scope_outside_a_graph_is_unattributed():
    assert current_scope() == UNATTRIBUTED


# ---------------------------------------------------------------- adapters


class MeteredLLM(FakeLLM):
    """FakeLLM that reports usage the way a real adapter does."""

    IN, OUT = 1000, 100

    async def chat(self, messages, **kwargs):
        reply = await super().chat(messages, **kwargs)
        record_usage("claude-opus-5", input_tokens=self.IN, output_tokens=self.OUT)
        return reply


CALL_COST = (MeteredLLM.IN * 5.0 + MeteredLLM.OUT * 25.0) / 1_000_000


class Metered(Capability):
    """Capability whose single node makes one or more LLM calls."""

    def __init__(self, name: str, llm, *, calls: int = 1, fail: bool = False):
        self.spec = CapabilitySpec(name=name, description=f"{name} capability")
        self.llm = llm
        self.calls = calls
        self.fail = fail

    async def work(self, state: CapabilityBaseState) -> dict:
        for _ in range(self.calls):
            await self.llm.chat([{"role": "system", "content": f"{self.spec.name} step"},
                                 {"role": "user", "content": state.get("query", "")}])
        if self.fail:
            return self._emit_failure(f"{self.spec.name} broke")
        return self._emit_success({"echo": self.spec.name})

    def render_context(self, result):
        return f"# {self.spec.name}"

    def build(self):
        g = self.state_graph(CapabilityBaseState)
        g.add_node("work", self.work)
        g.set_entry_point("work")
        g.add_edge("work", END)
        return g.compile()


def make_manager(store, checkpointer, tmp_path, caps_spec, *, fail=()):
    llm = MeteredLLM(
        {"planner": plan_json(*[name for name, _ in caps_spec])},
        default="A sufficiently long final answer for the usage test.",
    )
    caps = [Metered(name, llm, calls=calls, fail=name in fail) for name, calls in caps_spec]
    graph = build_agent(Deps(llm=llm), CapabilityRegistry(caps), checkpointer=checkpointer)
    return JobManager(graph, store, reports_dir=tmp_path / "artifacts"), llm


# ---------------------------------------------------------------- end to end


async def test_every_call_is_booked_once_and_attributed_to_its_step(
    store, checkpointer, tmp_path
):
    mgr, llm = make_manager(store, checkpointer, tmp_path, [("alpha", 1), ("beta", 2)])
    job = await mgr.create_job("do the thing")
    done = await mgr.run_job(job.job_id)

    assert done.status is JobStatus.DONE
    # per step: exactly the calls that step made, in its own result meta
    assert done.results["alpha"]["meta"]["usage"]["calls"] == 1
    assert done.results["beta"]["meta"]["usage"]["calls"] == 2
    assert done.results["beta"]["meta"]["usage"]["input_tokens"] == 2 * MeteredLLM.IN
    # the aggregate covers the framework's own calls (router, planner,
    # generation) too — nothing is lost and nothing is double-counted
    assert done.usage["calls"] == len(llm.calls)
    assert done.usage["calls"] > 3
    assert done.usage["input_tokens"] == len(llm.calls) * MeteredLLM.IN
    assert done.usage["cost_usd"] == pytest.approx(len(llm.calls) * CALL_COST)
    assert done.usage["models"] == ["claude-opus-5"]

    # ...and it survives the store, so `jobsmith job <id>` shows it later
    reloaded = await mgr.get_job(job.job_id)
    assert reloaded.usage == done.usage
    assert reloaded.step_usage("alpha")["calls"] == 1
    assert reloaded.step_usage("never-ran") == {}


async def test_a_failed_step_still_reports_what_it_burned(store, checkpointer, tmp_path):
    mgr, llm = make_manager(store, checkpointer, tmp_path, [("alpha", 2)], fail=("alpha",))
    job = await mgr.run_job((await mgr.create_job("do the thing")).job_id)

    result = job.results["alpha"]
    assert result["ok"] is False
    assert result["meta"]["usage"]["calls"] == 2       # the debugging number
    assert job.usage["calls"] == len(llm.calls)


async def test_progress_events_carry_the_running_cost(store, checkpointer, tmp_path):
    mgr, _ = make_manager(store, checkpointer, tmp_path, [("alpha", 1)])
    queue = mgr.subscribe()
    await mgr.run_job((await mgr.create_job("do the thing")).job_id)
    events = [queue.get_nowait() for _ in range(queue.qsize())]
    assert events[-1]["usage"]["calls"] > 0
    # spend is monotonic across a run, so a watcher can trust the last value
    calls = [event["usage"].get("calls", 0) for event in events]
    assert calls == sorted(calls)


async def test_capability_meta_is_untouched_without_a_ledger():
    """A capability run outside a job (a demo, a unit test) must not gain
    empty accounting keys."""
    cap = Metered("alpha", MeteredLLM(default="x"))
    out = await cap.build().ainvoke({"query": "q"})
    assert out["results"]["alpha"]["meta"] == {}


async def test_explicit_capability_meta_wins():
    cap = Metered("alpha", MeteredLLM())
    with usage_ledger():
        record_usage("claude-opus-5", input_tokens=5, scope="alpha")
        emitted = cap._emit_success({"x": 1}, {"usage": "mine", "via_fallback": True})
    assert emitted["results"]["alpha"]["meta"]["usage"] == "mine"
    assert emitted["results"]["alpha"]["meta"]["via_fallback"] is True


# ---------------------------------------------------------------- reporting


def make_job(**over) -> Job:
    job = Job(
        job_id="j1", status=JobStatus.DONE, query="analyse the thing",
        created_at="2026-09-01T00:00:00Z", final_answer="Here it is.",
        plan={"steps": [{"capability": "research", "depends_on": []},
                        {"capability": "critique", "depends_on": ["research"]}],
              "rationale": "chain"},
        results={
            "research": {"ok": True, "data": {"notes": "n"},
                         "meta": {"usage": Usage(input_tokens=12_000, output_tokens=3_000,
                                                 calls=2, cost_usd=0.135,
                                                 models=("claude-opus-5",)).to_dict()}},
            "critique": {"ok": True, "data": {"notes": "n"}},     # made no LLM call
        },
        step_finished_at={"research": "t1", "critique": "t2"},
        usage=Usage(input_tokens=20_000, cached_input_tokens=1_000, output_tokens=5_000,
                    calls=6, cost_usd=0.225, models=("claude-opus-5",)).to_dict(),
    )
    for key, value in over.items():
        setattr(job, key, value)
    return job


def test_formatters_stay_readable():
    usage = Usage(input_tokens=20_000, cached_input_tokens=1_000, output_tokens=5_000,
                  calls=6, cost_usd=0.225, models=("claude-opus-5",))
    line = format_usage(usage)
    assert "6 LLM calls" in line
    assert "20,000 in (+1,000 cached) / 5,000 out tokens" in line
    assert "~$0.2250 est." in line and "claude-opus-5" in line
    assert format_usage(Usage()) == "not recorded"
    assert format_step_usage(Usage()) == "—"
    assert format_step_usage(
        Usage(input_tokens=12_000, output_tokens=3_000, calls=2)) == "15.0k tok"
    assert "$" in format_step_usage(Usage(calls=1, output_tokens=10, cost_usd=1.5))


def test_report_shows_the_cost_in_about_this_job_and_per_step():
    job = make_job()
    document = build_document(job)
    assert document.usage.calls == 6
    assert document.plan[0].usage.calls == 2

    markdown = MarkdownReport().render(document)
    about = markdown.split("## About this job", 1)[1]
    assert "- **Usage**: 6 LLM calls — 20,000 in (+1,000 cached) / 5,000 out tokens" in about
    assert "~$0.2250 est." in about
    # the plan table lets a reader see WHICH step was expensive
    assert "| step | depends on | status | usage | finished at |" in about
    assert "| research | — | ok | 15.0k tok · ~$0.1350 | t1 |" in about
    assert "| critique | research | ok | — | t2 |" in about


def test_report_of_an_untracked_job_says_so(tmp_path):
    job = make_job(usage={}, results={})
    [output] = MarkdownReport().write(job, tmp_path)
    path = output.path
    assert "- **Usage**: not recorded" in open(path).read()


def test_job_summary_round_trips_usage():
    job = make_job()
    assert job.summary()["usage"]["calls"] == 6
    assert job.to_dict()["usage"]["cost_usd"] == pytest.approx(0.225)
    assert JobOutput(path="p").role == "main"       # unrelated fields untouched


def test_cli_job_view_shows_the_cost(capsys):
    from jobsmith.cli.repl import show_job

    show_job(make_job().to_dict())
    out = capsys.readouterr().out
    assert "artifact:  research: ok  [15.0k tok · ~$0.1350]" in out
    assert "artifact:  critique: ok" in out and "critique: ok  [" not in out
    assert "usage:     6 LLM calls — 20,000 in (+1,000 cached) / 5,000 out tokens" in out


# ------------------------------------------------- the real adapters' plumbing


def anthropic_response(usage, model="claude-opus-5"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn", stop_details=None, model=model, usage=usage,
    )


def anthropic_stub(response):
    async def create(**kwargs):
        return response

    return SimpleNamespace(
        messages=SimpleNamespace(create=create),
        beta=SimpleNamespace(messages=SimpleNamespace(create=create)),
    )


async def test_anthropic_adapter_books_tokens_and_the_model_that_served():
    usage = SimpleNamespace(input_tokens=100, output_tokens=20,
                            cache_read_input_tokens=50, cache_creation_input_tokens=10)
    # a server-side fallback answered: bill the model from the RESPONSE
    client = AnthropicLLMClient(client=anthropic_stub(
        anthropic_response(usage, model="claude-sonnet-5")))
    with usage_ledger() as ledger:
        await client.chat([{"role": "user", "content": "hi"}])
        await client.vision(b"png-bytes", "what is this")
    total = ledger.total()
    assert total.calls == 2                                   # chat AND vision
    assert total.input_tokens == 2 * 110                      # cache writes fold into input
    assert total.cached_input_tokens == 2 * 50
    assert total.output_tokens == 2 * 20
    assert total.models == ("claude-sonnet-5",)
    assert total.cost_usd == pytest.approx(
        (220 * 2.0 + 100 * 0.2 + 40 * 10.0) / 1_000_000)      # sonnet-5 rates, cache at 10%


async def test_adapter_without_usage_reports_nothing_rather_than_zeroes():
    client = AnthropicLLMClient(client=anthropic_stub(anthropic_response(None)))
    with usage_ledger() as ledger:
        await client.chat([{"role": "user", "content": "hi"}])
    assert ledger.total() == Usage()


async def test_openai_adapter_keeps_cached_tokens_disjoint_from_input():
    usage = SimpleNamespace(
        prompt_tokens=100,                                     # INCLUDES the cached ones
        completion_tokens=20,
        prompt_tokens_details=SimpleNamespace(cached_tokens=40),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", refusal=None))],
        model="gpt-5.1", usage=usage,
    )

    async def create(**kwargs):
        return response

    stub = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with usage_ledger() as ledger:
        await OpenAILLMClient(client=stub).chat([{"role": "user", "content": "hi"}])
    total = ledger.total()
    assert (total.input_tokens, total.cached_input_tokens, total.output_tokens) == (60, 40, 20)
    assert total.cost_usd is None          # no price shipped for it: tokens, not invention


async def test_the_keyless_fake_reports_plausible_usage():
    """CI has no API key — the cost path must still be exercised end to end."""
    llm = KeywordLLM()
    with usage_ledger() as ledger:
        await llm.chat([{"role": "system", "content": "triage" * 100},
                        {"role": "user", "content": "analyse this"}])
        await llm.vision(b"x" * 1500, "what is this")
    total = ledger.total()
    assert total.calls == 2
    assert total.input_tokens > 100
    assert total.output_tokens > 0
    assert total.cost_usd > 0               # the fake is priced, so the maths runs
    assert total.models == (KeywordLLM.MODEL,)


async def test_two_jobs_running_at_once_never_bill_each_other(store, checkpointer, tmp_path):
    """The failure mode a ContextVar ledger invites, and the one that would be
    silent: two jobs in the same process crediting their tokens to each other.

    The capabilities sleep around their booking so the two runs genuinely
    interleave — without that, the tasks would serialise and the test would
    pass whatever the design.
    """
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver

    from jobsmith.jobs.manager import JobManager

    class Burner(Capability):
        def __init__(self, name: str, tokens: int):
            self.spec = CapabilitySpec(name=name, description=name)
            self.tokens = tokens

        async def work(self, state):
            await asyncio.sleep(0.02)          # let the sibling run interleave
            record_usage("test-model", input_tokens=self.tokens)
            await asyncio.sleep(0.02)
            return self._emit_success({"burned": self.tokens})

        def render_context(self, result):
            return str(result["data"]["burned"])

        def build(self):
            g = self.state_graph(CapabilityBaseState)
            g.add_node("work", self.work)
            g.set_entry_point("work")
            g.add_edge("work", END)
            return g.compile()

    def manager_for(name: str, tokens: int) -> JobManager:
        capability = Burner(name, tokens)
        llm = FakeLLM({"planner": plan_json(name)},
                      default="A sufficiently long final answer for this run.")
        graph = build_agent(Deps(llm=llm), CapabilityRegistry([capability]),
                            checkpointer=MemorySaver())
        return JobManager(graph, store, reports_dir=tmp_path / "artifacts")

    alpha, beta = manager_for("alpha", 100), manager_for("beta", 7)
    job_a = await alpha.create_job("A")
    job_b = await beta.create_job("B")
    done_a, done_b = await asyncio.gather(
        alpha.run_job(job_a.job_id), beta.run_job(job_b.job_id)
    )

    assert done_a.results["alpha"]["meta"]["usage"]["input_tokens"] == 100
    assert done_b.results["beta"]["meta"]["usage"]["input_tokens"] == 7
    # and the job totals stay disjoint too
    assert done_a.usage["input_tokens"] == 100
    assert done_b.usage["input_tokens"] == 7
