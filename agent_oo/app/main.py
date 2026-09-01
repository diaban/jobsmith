"""Entrypoints for the global agent.

    python -m agent_oo               # chat REPL
    python -m agent_oo api [port]    # HTTP API + SSE (needs .[api])

Flags: --llm=anthropic|openai|fake (else auto-detected from API keys) and
--db=memory|<file.db>|<postgres DSN> (else $AGENT_OO_DB, else memory).
.env is loaded from the working directory.

Both entrypoints run inside ONE asyncio loop: `build_app` opens the
persistence connections there, and the API is served with
`uvicorn.Server.serve()` rather than `uvicorn.run()` — the latter starts its
own loop, which would leave the pool bound to a dead one.
"""
from __future__ import annotations

import asyncio
import sys

from .agent import build_app
from .providers import load_dotenv
from .repl import run_repl


async def chat() -> None:
    app = await build_app()
    try:
        await run_repl(app.manager, app.new_session())
    finally:
        await app.aclose()


async def serve() -> None:
    import uvicorn

    from ..api import create_api

    app = await build_app()
    try:
        port = next((int(a) for a in sys.argv[1:] if a.isdigit()), 8000)
        config = uvicorn.Config(
            create_api(app.manager, app.session_factory),
            host="127.0.0.1",
            port=port,
        )
        await uvicorn.Server(config).serve()
    finally:
        await app.aclose()


def main() -> None:
    load_dotenv()
    try:
        asyncio.run(serve() if "api" in sys.argv[1:] else chat())
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
