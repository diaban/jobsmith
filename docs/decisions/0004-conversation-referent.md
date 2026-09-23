# 0004 — A launched job carries the conversation's referent

- **Issue:** #4 · **PR:** #15
- **Status:** accepted
- **Source:** migrated verbatim from `CLAUDE.md` at `8326b98` (#102). The text is the original; only the headings (which section of `CLAUDE.md` it lived in) and the links were added.
- **See also:** [0074](0074-prior-job-references.md)

## From “Chat layer (`chat/`)”

**The job engine never sees the thread**, so "analyse that" would reach the planner with its referent gone — a silent failure (the report looks fine, it answers a slightly different question). Two guards: the tool docstring *is* the model's instruction and demands a self-contained `query` (that wording is also what the user approves), and `recent_conversation()` attaches a bounded excerpt of the last turns as `inputs[CONVERSATION_INPUT_KEY]` (`core/state.py`, key `"conversation"`) as the safety net. Bounds live in `chat/tools.py` (`MAX_CONTEXT_TURNS`/`MAX_TURN_CHARS`/`MAX_CONTEXT_CHARS`) — this rides on every launch; only human/assistant prose travels (tool calls, tool results and the `[job update]`/`[job progress]` notices are dropped). `Planner.user_message` renders it as clearly-labelled background above the request; the request stays authoritative.
