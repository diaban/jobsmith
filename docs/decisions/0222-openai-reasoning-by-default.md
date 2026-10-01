# 0222 — An OpenAI model is a reasoning model unless it belongs to a classic family

- **Issue:** #222 · **PR:** #223 · **Status:** accepted
- **Rule in `CLAUDE.md`:** none (the comment above `is_reasoning_model` in `dag/clients.py`)

**Context.** `OpenAILLMClient` dropped `temperature` and sent `max_completion_tokens` only for names starting with `gpt-5`, `o1`, `o3`, `o4`. With `gpt-6-luna` it sent `max_tokens` and `temperature=0.0`: both refused (400), so every job call failed. Found by the strict-mode probe (0219).

**Decision.** `is_reasoning_model(name)`: an OpenAI name (`gpt-…`, `o<digit>…`) is a reasoning model unless it starts with a classic family (`gpt-4`, `gpt-3.5`). The classic families are closed; every OpenAI model since gpt-5 is a reasoning model, so the next family needs no change. A name from another OpenAI-compatible server (Ollama, vLLM) is not an OpenAI name and keeps the classic behaviour, as before.

**Alternatives and why not.** Adding `gpt-6` to the list: breaks again at the next family. Retrying on the 400 with the other parameters: one failed call per model and per process, for what a name already tells. Sending `max_completion_tokens` to every model: Ollama-style servers are not known to honour it.

**Measured.** One real call each on `gpt-6-luna`: the jobs client answers after the fix; the chat stack (`ChatOpenAI`, which sets neither parameter) already did. Falsified: the old prefix list fails the new test on `gpt-6-luna`, `gpt-7` and `o5`.
