# Glossary

One name per concept, one concept per name. This is the vocabulary of the code, the design notes, the decision records, the issues and every discussion about jobsmith. When a word below is used, it means what its row says; when a concept below is meant, its term is used.

**Why it exists.** Misunderstandings came from one word meaning several things: "DAG" (the `dag/` package, the reference agent, or a program's dependency structure), "generation" (writing the answer, or turning a program into something runnable), "output" (an IR field, `Job.outputs`, an op's result), "agent" (an agent definition, the ReAct baseline, an exploring op). The last section lists the words not to use unqualified.

**How to read the tables.** *Term* is the word to use. *In code* is where it lives today, or the name it takes when code and term still differ (marked **rename**: done by the step that touches that code, never as a bulk rename). *Not to be confused with* names the neighbour it is most often mixed up with. Terms marked *(target)* belong to a design not yet built (`docs/design/compiler-v1.md`).

---

## 1. Layers

| Term | In code | Definition | Not to be confused with |
|---|---|---|---|
| **Engine** | `engine/` | Runs any LangGraph graph as a durable job and delivers its job result. Knows no product word (gate G4). | the compiler |
| **Compiler** | `dag/` (**rename** to `compiler/`) | Turns a request into a checked program and runs it through the interpreter. | the reference agent, built on it |
| **Reference agent** | `agents/default/` | The agent shipped with the framework, built on the compiler. | "the DAG" (do not use) |
| **Adapters** | `adapters/` | The reusable contract between a job and something outside the engine (today: a LangChain conversation). | the bench |
| **Bench** | `chat/`, `tui/`, `cli/`, `api/`, `app/`, `service.py` | The example application, used as a test bench for the framework. | the adapters, which are part of the framework |

The framework is **chat-agnostic**: the engine already is (G4, return-address kinds); the compiler must become so. A conversation is one caller among others.

---

## 2. Running a job (engine)

| Term | In code | Definition | Not to be confused with |
|---|---|---|---|
| **Job** | `Job` (`engine/models.py`) | One request taken in charge, end to end: created, run, settled, delivered. | a step |
| **Attempt** | `Job.attempt` | One try at running a job. A resume, an answer to a pause, a relaunch or an amendment opens a new attempt. | a retry of a step |
| **Graph spec** | `GraphSpec` | The contract a graph signs with the engine: name, compiled graph, result extractor, relaunch bound. One per agent definition, not one per program. | a program |
| **Fact** | `publish(key, value)`, `engine/facts.py` | A named value a graph tells the job while it runs. | a step result |
| **Return address** | `reply_to` | Who the job result is delivered to: a session, a webhook, or nobody (`none`). | the bench's UI |
| **Delivery** | `engine/delivery.py`, `delivered_at` | Handing the job result to the return address, at least once, keyed by job id and attempt. | writing a deliverable |
| **Promotion** | `run_for` | A caller waits for a job up to a clock; past it, the job runs on in the background and is delivered later. | export (§3) |
| **Pause** | `needs_input`, `interrupt()`, `answer_job` | A job waiting for an answer from outside; it does not run while paused. | a cancelled job |
| **Amendment** | `amend_job` | Stopping a running job, changing its checkpoint, and running it on as a new attempt. | a recompilation, which happens inside the run |

---

## 3. Compiling and running a program (compiler)

### Objects

| Term | In code | Definition | Not to be confused with |
|---|---|---|---|
| **Request** | `query` | What the caller asked, in natural language. | the program; an instruction |
| **Program** | `Plan` (**rename** to `Program`, `dag/state.py`) | What the compiler runs for one request: `steps` and a `result`. Replaces "plan". | a graph spec |
| **Step** | `PlanStep` (**rename** to `Step`), known by its `id` | One element of a program: a work step in `steps`, or a result slot. | an op; a job |
| **Work step** | an entry of `steps` | A gathering or reasoning step the planner chooses. | a result slot |
| **Result slot** | `result.<name>` *(target)* | A step that produces part of the job result: the answer, or a deliverable. The compiler decides which slots a program has; the planner fills their `instruction` and `material`. | a work step |
| **Result contract** | *(target)* | What the job must produce: a text for a human (with its mode), a typed object, deliverables, formats. Declared by the caller; inferred from the request by the conversation's adapter. The program's result slots are generated from it. | the return address, which says *to whom* |
| **Instruction** | `instruction` argument | What one step must do, written by the planner for that step. The content; the form comes from the profile. | the request, which a step receives only as background |
| **Background** | — | The request, given to a step as marked context ("the request this job serves; your task is the instruction"). | an instruction |
| **Op** | `CapabilitySpec` (**rename** to `OpSpec`) | A registered operation a step uses. One op may serve several steps. | a step |
| **Op kinds** | `LlmOp`, `ToolOp`, `FnOp`, `ReviewOp`, `Capability` | One model call with an output schema; a LangChain or MCP tool; deterministic code; a human decision; a sub-graph (bounded ReAct). The planner never sees the kind. | — |
| **Agent op** | a `Capability` running a bounded ReAct loop | An op that explores, with declared tools, an iteration bound and a budget. | an agent definition (§5) |
| **Instance** | `reads[3]` | One run of a `map` step over one item, with its own result, fact and retry. | a step |
| **Reference** | `$id`, `$id.field`, `$input.key`, `$item` | What a step consumes: another step's result, an input, or the current item of a `map`. | a dependency written by hand (`after`) |
| **Registry** | `CapabilityRegistry` (**rename** to `OpRegistry`) | The ops a program may use, with their input and output schemas and their effects. Work ops are what `steps` may name; result ops serve result slots. | — |
| **Effects** | `Effects` | What an op does to the world: read-only or not, idempotent or not, cost class, approval, max fan-out. | an op's output schema |
| **Calibrated program** | `agents/<agent>/programs/*.json` *(target)* | A program frozen after it proved itself: named, versioned, reviewed as a diff. Runs with no planner call. | a just-in-time program |

### Phases

| Term | In code | Definition | Not to be confused with |
|---|---|---|---|
| **Planning** | `Planner` | The planner writes the work steps and fills the result slots. | the whole compilation |
| **Static analysis** | `dag/analysis.py` *(target)* | The checks run on a program before its first step, each returning findings. | the `analysis` op |
| **Repair** | *(target)* | Sending a program with its findings back to the planner, at most twice. | a step retry |
| **Interpreter** | `dag/executor.py` (**rename** to `interpreter.py`) | The generic graph that runs a program wave by wave, resolving references and expanding `map`s. The program is data in its state. | the engine |
| **Wave** | — | The steps the interpreter sends together because all their dependencies are done. | an attempt |
| **Recompilation** | `verify` step *(target)* | Rewriting the not-yet-started part of a program during a run, a bounded number of times. | an amendment (§2) |
| **Export** | `jobsmith program export` *(target)* | Turning a just-in-time program that succeeded into a calibrated program. | promotion (§2) |

There is no "lowering": a program is not turned into a graph spec, the interpreter reads it.

---

## 4. What a job produces

The most overloaded area. **Never write "output" alone.**

| Term | In code | Definition | Not to be confused with |
|---|---|---|---|
| **Step result** | `results[id]` | What one step returns. | the job result |
| **Material** | `material` argument, `results_of` | What a step reads from other steps: the step results it references. The answer's material defaults to every successful work step result, in program order. | the answer |
| **Writing** | `generation` pipeline (**rename** to the answer op) | Producing the answer from its material. Replaces "generation". | an authored deliverable |
| **Answer** | `final_answer`; `result.answer` *(target)* | The text for a human. Two modes: **reply** (the answer is the response) and **brief** (an account of the job that presents its deliverables; `BRIEF_RULE` is its seed). A refusal to answer (0059) is one of its outcomes. | a deliverable |
| **Deliverable** | `Job.outputs`, role `main` / `alternate` | A document the job delivers. **Authored**: designed by a model from the material, for its own audience (the slide deck), a sibling of the answer. **Rendered**: a deterministic formatting of an existing text (today's Reporters), appended by the compiler. | an annex |
| **Format** | `markdown`, `html`, `pdf`, `pptx` | How a deliverable is serialized to a file. | a deliverable |
| **Annex** | `Job.outputs`, role `annex` | A file produced by a work step, e.g. a chart. | a deliverable |
| **Job result** | `Job.result` | What the engine delivers to the return address. A program's result slots produce it. | a step result |

Today the slide deck is an annex planned as a work step (0035); in the target it is an authored deliverable in a result slot.

---

## 5. Agents

**Never write "agent" alone.**

| Term | In code | Definition |
|---|---|---|
| **Agent definition** | `AgentDefinition` | What an agent is: an op pack and a profile, or a graph of its own. |
| **Profile** | `AgentProfile` | An agent definition's prompts, messages and rules; the form of each medium (how a reply, a brief, a deck is written). |
| **Reference agent** | `agents/default/` | See §1. |
| **Baseline** | `agents/react/` | The ReAct agent every compiler step is measured against (0206). |
| **Agent op** | see §3 | An op implemented by a bounded ReAct loop. |

## 6. Conversation

| Term | In code | Definition |
|---|---|---|
| **Session** | `session_id`, `reply_to` of kind `session` | One conversation, rebuildable by id. |
| **Turn** | `stream`, `ChatRunner` | One exchange in a session, from a message to its terminal event. |

The conversation is a caller: its adapter infers the result contract from the request (today `document_intent`); an API caller declares it.

---

## 7. Words not to use unqualified

| Word | Say instead |
|---|---|
| "the DAG" | **reference agent**, **compiler**, or **program** (its dependency structure) |
| "plan" | **program** |
| "generation" | **writing** |
| "output" | **step result**, **material**, **answer**, **deliverable**, **annex**, **format** or **job result** |
| "capability" | **op**, or **agent op** for the exploring kind (`Capability` stays the class of a sub-graph op) |
| "agent" | **agent definition**, **reference agent**, **baseline** or **agent op** |
| "run" | **job**, **attempt**, or **wave**, whichever is meant |

---

## Example

The slide-deck failure the strict-mode probe found (0219), in these words:

> The planner placed the deck as a **work step**, alone, with nothing to reference: it read the **request** as its own task. In the target, the deck is an **authored deliverable** in a **result slot** the compiler generates from the **result contract**; the planner fills its **instruction** and its **material** (`$compare`), and the **answer**, in brief mode, references `$deck` to present it.
