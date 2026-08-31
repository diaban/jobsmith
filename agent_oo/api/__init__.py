"""HTTP entrypoint: FastAPI app over a JobManager + chat sessions.

Optional dependency:  pip install -e ".[api]"
"""
from .app import create_api

__all__ = ["create_api"]
