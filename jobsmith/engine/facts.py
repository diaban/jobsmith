"""Named facts a graph tells the job running it, while it runs.

A node calls `publish(key, value)`; the job runner hears it on LangGraph's
`custom` stream, from any depth of sub-graph, and hands it to the manager as
a `Fact`. Nothing about the keys is known here: they are the graph's words
(docs/design/core-v1.md, "Facts").

Outside a job's run — a graph invoked on its own, a node called by a test —
there is nobody to tell, and `publish` does nothing.
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

#: Marks a custom stream event as a fact, among whatever else a graph writes there.
FACT_KEY = "jobsmith_fact"


def publish(key: str, value: Any) -> None:
    """Tell the job running this graph that `key` is now `value`."""
    try:
        writer = get_stream_writer()
    except RuntimeError:              # not inside a graph run
        return
    writer({FACT_KEY: key, "value": value})
