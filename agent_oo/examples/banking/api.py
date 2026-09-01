"""Run the banking agent as an HTTP API.

    python -m agent_oo.examples.banking.api          # http://127.0.0.1:8000
    python -m agent_oo.examples.banking.api 9000     # custom port

Same provider auto-selection as the REPL (--llm=anthropic|openai|fake).
Interactive docs at /docs; live job events at /events (SSE).
"""
from __future__ import annotations

import sys

from langgraph.checkpoint.memory import MemorySaver

from ...api import create_api
from ...app.providers import load_dotenv, make_chat_model, make_llm, pick_provider
from ...chat import ChatSession
from .chat import build_chat
from .profile import BANKING_CHAT_PROMPT


def main() -> None:
    import uvicorn

    load_dotenv()
    choice = pick_provider()
    manager = build_chat(make_llm(choice))
    chat_model = make_chat_model(choice)

    def session_factory(session_id: str | None = None) -> ChatSession:
        return ChatSession(
            manager,
            chat_model,
            session_id=session_id,
            system_prompt=BANKING_CHAT_PROMPT,
            checkpointer=MemorySaver(),
        )

    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
    uvicorn.run(create_api(manager, session_factory), host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
