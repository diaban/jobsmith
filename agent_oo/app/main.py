"""Entrypoints for the global agent.

    python -m agent_oo               # chat REPL
    python -m agent_oo api [port]    # HTTP API + SSE (needs .[api])

Both accept --llm=anthropic|openai|fake (auto-detected from API keys
otherwise) and load .env from the working directory.
"""
from __future__ import annotations

import asyncio
import sys

from .agent import build_app
from .providers import load_dotenv
from .repl import run_repl


async def chat() -> None:
    app = build_app()
    await run_repl(app.manager, app.new_session())


def serve() -> None:
    import uvicorn

    from ..api import create_api

    app = build_app()
    port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
    uvicorn.run(create_api(app.manager, app.session_factory), host="127.0.0.1", port=port)


def main() -> None:
    load_dotenv()
    try:
        if "api" in sys.argv[1:]:
            serve()
        else:
            asyncio.run(chat())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
