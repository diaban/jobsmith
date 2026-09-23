# 0081 — The grounding reaches the reasoning

- **Issue:** #81 · **PR:** #87
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0075](0075-web-pages-not-snippets.md), [0073](0073-deliverable-answers.md)

## From “Agents (`agents/`) — what an agent *is*”

**The grounding reaches the reasoning, and the result says whether it did** (#81). Retrieval was wired to exactly one consumer: `ContextMerger`, i.e. the final generator. `research.investigate` built its messages from `query` and `aspects` alone, `analysis` declared `UPSTREAM = (("research", "notes"),)`, `critique` read analysis-then-research, and `documents` / `web_search` / `read_files` appeared in no `UPSTREAM` at all. Measured on one job: `web_search` returned 14.7k characters of sourced specifications, `research` then wrote 14.6k characters of per-item sheets **from model recall**, and `analysis` drew its conclusions from those recalled sheets — the retrieved table entered the run at the generator, three steps too late. That also reframes two earlier issues: #73's wholesale hedge was *honest* for a recall step, and #75 improved what reached the generator and changed nothing for anything upstream of it. **The fix is one capability, not a new contract**: `research` reads the retrieved material when the plan has any and falls back to its own knowledge when it has none, so the chain the planner already draws (`web_search → research → analysis`) becomes true and the LLM-only agent is unchanged to the prompt. Five decisions, and two things deliberately left alone. **Every source is read, not the first** — `GROUNDING` looks like `_step.py`'s `UPSTREAM` and means the opposite: there the tuple is a priority chain over restatements of one thing (the analysis, else the notes behind it) and reading the second would duplicate content, here the entries are *sources* and a named file plus what the web says today are complementary. `_material`'s first-match rule is therefore left exactly as it is, because nothing about this fix needs it changed. **The two modes are two prompts**: notes written from a source are told the material *is* the source, to carry its figures and ids, and to mark doubt only where the material is thin — a standard the recall prompt cannot be held to, which is the whole of what #73 was arguing about from the other side. **The mode is on the result** (`meta["grounded_on"]`, the steps whose material was read; the heading of `render_context` and `render_report` says it in words), because "the Aeron seats 159 kg" from a spec sheet and the same sentence from memory are not the same claim and nothing downstream could tell them apart. **The material is bounded at 32 000 characters ≈ 8 000 tokens**, shared out document by document so no document is dropped whole and short ones hand their share back, each cut written into the text (the rule `TavilySource._bounded` and `LocalFileReader` already follow): the worst case handed here is 10 × 8 000 from `web_search` alone (#75), 80 000 characters ≈ 20 000 tokens, and that block is *already* re-sent to the generator, so feeding it whole would double the run's largest payload before a note is written. **A refusal crosses the new step too**: `read_files` emits `documents` *and* `unreadable`, and its own docstring already says why — "a refusal is material, not silence", in the generation context "because a model told nothing about the missing file writes confidently over the hole". Putting a step that writes *sourced* notes between the two would have left the gap uncrossed, so `REFUSALS` carries it, in its own labelled block (what is missing is not something to reason from) and with the prompt saying what to do with it: name it as a gap, never write the document up from memory as though it had been read. Only `read_files` has one, and that is a fact about the two ports rather than an omission: a query that matched nothing returned nothing and fails the step, while a file that could not be opened was pointed at by name — `documents` and `web_search` are deliberately absent rather than given an invented equivalent. **The spec had to change with the capability** — it said "from the model's own knowledge (no external sources)", which is what the planner reads and was now false. `analysis` was deliberately **not** touched: it reads `research`, and `research` is now grounded, so the material reaches it through the edge that was already there; adding the retrieval steps to its `UPSTREAM` (or merging every upstream in `_material`) would cost tokens on every run to fix a plan shape nobody has observed, and the new eval check below is what would show it. **It also dissolves the honesty problem the issue raised about the picture**: `depends_on` is still only a scheduling constraint and the framework still has no notion of data flow, but on `retrieval → research` the edge every deliverable draws as a data flow now carries data, so nothing in the report renderer had to change.

## From “Evaluating prompts (`evals/`)”

**One check reads what the steps produced, not what the document says**
(#81). `report_answers_request` was green through the entire run that
opened that issue: the merger hands the retrieved material straight to the
generator, so the deliverable carried it while not one step between the
retrieval and the writing had read it.
`grounding_reaches_reasoning` asks the same question one layer earlier and
against the steps' own `results` — a third of what the retrieved documents
keep coming back to (minus the request's own words) has to appear in what
the steps downstream of the retrieval produced. It recognises a retrieval
step by the **shape** of its result (`data["documents"]`, passages with a
`text`) and never by name: `evals` is agent-agnostic, and a check that knew
`web_search` would score one pack and silently skip every other. Three
honest skips, each leaving the denominator: nothing was retrieved, the
retrieval failed, or **no step of the plan depends on the retrieval** — the
last one being the limit worth naming, since this measures the edges a plan
*draws* and a plan that schedules retrieval beside reasoning draws none.
Like every check built on `frequent_terms` it detects abandonment and never
regurgitation.

**A case can name a file, and it cannot carry a path** (#81). The check
above needs a run with real material, and the only grounding step that is
deterministically available is `read_files` (`documents` and `web_search`
need a `--docs` directory or a key; `documents` searching a fixture corpus
on *every* case would fail the step wherever no term overlapped). So one
golden case names a file — and `read_files` may only open a path inside a
root the deployment declared, which under `build_app` is the scratch
reports directory this suite composes. A case therefore declares
`FIXTURE_REF` in its `inputs`, `harness.write_fixture` writes the fixture
there before the app is composed, and `resolve_inputs` swaps the real path
in: a hard-coded path would be a path to one machine. The fixture text is
short, specific and domain-neutral on purpose — short so that what a step
did with it is legible in that step's own output, specific so that "this
reached the reasoning" is a question a bag of words can answer at all.
