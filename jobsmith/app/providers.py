"""Provider selection + deterministic fakes for the two LLM stacks.

- `make_llm` → framework `LLMClient` for the JOB ENGINE (planner/capabilities).
- `make_chat_model` → LangChain chat model for the CHAT AGENT (tool calling).
- `pick_provider` resolves once (--llm= flag, then available API keys) so both
  stacks agree; `load_dotenv` is a dependency-free .env loader.
- `KeywordLLM` / `KeywordChatModel` are generic fakes: everything runs with no
  API key. KeywordLLM plans by READING the capability names out of the
  rendered planner prompt (a chain in listed order), so it works with any
  registry — every registered agent alike.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from typing import Any, ClassVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

# ---------------------------------------------------------------- env / choice

def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines; real env vars take precedence)."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def pick_provider() -> str:
    """--llm=... flag wins, else auto-detect by available key."""
    choice = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--llm=")), None)
    if choice is None and "--real" in sys.argv:
        choice = "anthropic"
    if choice is None:
        choice = os.environ.get("JOBSMITH_LLM") or None   # set by the CLI's --llm
    if choice is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            choice = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            choice = "openai"
        else:
            choice = "fake"
    return choice


def make_llm(choice: str) -> Any:
    """LLMClient for the JOB ENGINE (planner/capabilities/generation)."""
    if choice == "anthropic":
        from ..clients import AnthropicLLMClient

        llm = AnthropicLLMClient()
        print(f"[jobs llm: Claude via AnthropicLLMClient — {llm.model}]", file=sys.stderr)
        return llm
    if choice == "openai":
        from ..clients import OpenAILLMClient

        llm = OpenAILLMClient()
        print(f"[jobs llm: OpenAI via OpenAILLMClient — {llm.model}]", file=sys.stderr)
        return llm
    print("[jobs llm: KeywordLLM fake — set ANTHROPIC_API_KEY or OPENAI_API_KEY for real models]", file=sys.stderr)
    return KeywordLLM()


def make_chat_model(choice: str) -> Any:
    """LangChain chat model for the CHAT AGENT (tool calling handled upstream)."""
    if choice == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            sys.exit("chat with Claude needs:  uv pip install langchain-anthropic")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
        print(f"[chat llm: ChatAnthropic — {model}]", file=sys.stderr)
        return ChatAnthropic(model=model, max_tokens=4096)
    if choice == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            sys.exit("chat with OpenAI needs:  uv pip install langchain-openai")
        model = os.environ.get("OPENAI_MODEL", "gpt-5.1")
        print(f"[chat llm: ChatOpenAI — {model}]", file=sys.stderr)
        return ChatOpenAI(model=model)
    print("[chat llm: KeywordChatModel fake]", file=sys.stderr)
    return KeywordChatModel()


# ---------------------------------------------------------------- fakes

class KeywordLLM:
    """Deterministic LLMClient for the job engine.

    Registry-agnostic: the plan is built from the capability names found in
    the planner's own rendered prompt (chained in listed order — inapplicable
    steps are dropped by plan validation as usual).
    """

    _CAP_LINE = re.compile(r'^- "([a-z][a-z0-9_]*)"', re.MULTILINE)
    DIRECT_WORDS = ("what can you do", "who are you", "hello", "bonjour", "capabilit", "aide")

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]

        if "triage" in system.lower():
            direct = any(w in user.lower() for w in self.DIRECT_WORDS)
            return json.dumps({
                "route": "direct" if direct else "plan",
                "rationale": "keyword triage",
            })
        if "Answer the user's message directly" in system:
            return "I plan and run capability jobs in the background; ask me anything."
        if "planner" in system.lower():
            names = self._CAP_LINE.findall(system)
            steps = [
                {"capability": name, "depends_on": [names[i - 1]] if i else []}
                for i, name in enumerate(names)
            ]
            return json.dumps({
                "steps": steps,
                "rationale": "fake plan: chain every listed capability in order",
            })
        if "Rewrite" in system:
            return json.dumps({"query": user.strip().lower()[:80]})
        if "failed validation" in system:
            return "Refined: " + user.split("Context:")[-1].strip()[:300] + " [doc_0]"
        # generation-ish prompts: echo whatever context the pipeline produced
        ctx = user.split("Context:")[-1].strip()
        if ctx == "(no context available)":
            return "The context is insufficient to answer this query."
        return f"Based on the produced context [doc_0]:\n{ctx[:400]}"

    async def vision(self, image_bytes: bytes, prompt: str, **kwargs: Any) -> str:
        return f"(fake analysis of {len(image_bytes)} bytes for: {prompt[:60]})"


class KeywordChatModel(BaseChatModel):
    """Deterministic tool-calling chat model: proposes a job on analysis-ish
    words, otherwise answers directly. Lets the whole chat/HITL/notification
    flow run without any API key."""

    COMPLEX_WORDS: ClassVar[tuple[str, ...]] = (
        "analyse", "analyze", "search", "cherche", "recherche", "research",
        "rapport", "report", "compare", "étude", "review", "deck", "investigate",
    )

    @property
    def _llm_type(self) -> str:
        return "keyword-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> KeywordChatModel:
        return self

    @staticmethod
    def _reply(text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        # 1. a finished-job notice was injected → synthesize it
        for m in reversed(messages):
            if isinstance(m, SystemMessage) and "background jobs finished" in m.content:
                report = next(
                    (ln.split(": ", 1)[1] for ln in m.content.splitlines()
                     if ln.startswith("Report file: ")), "?",
                )
                return self._reply(
                    "A background job just finished — the requested analysis is ready. "
                    f"Full report: {report}"
                )
        last = messages[-1]
        # 2. tool result (launch/status/cancel) → relay it
        if isinstance(last, ToolMessage):
            if "DECLINED" in last.content:
                return self._reply("Understood, I won't launch that job.")
            return self._reply(f"Noted — {last.content}")
        # 3. user message → job proposal or direct answer
        user = last.content if isinstance(last, HumanMessage) else ""
        if any(w in user.lower() for w in self.COMPLEX_WORDS):
            call = {
                "name": "launch_job",
                "args": {
                    "query": user,
                    "rationale": "Multi-step task detected (keywords): "
                                 "a background job fits better than an inline answer.",
                },
                "id": f"call_{uuid.uuid4().hex[:8]}",
            }
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
            )
        return self._reply(
            "I can answer directly, or launch background analysis jobs "
            "(fake mode — no API key set)."
        )
