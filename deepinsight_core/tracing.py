"""LangSmith tracing integration for DeepInsight LangGraph workflows.

Usage:
    Set these environment variables to enable tracing:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<your-langsmith-api-key>
        LANGCHAIN_PROJECT=deepinsight  (optional, defaults to "deepinsight")

If the env vars are not set, LangSmith tracing is silently disabled.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

_LANGSMITH_AVAILABLE: Optional[bool] = None


def _check_langsmith() -> bool:
    global _LANGSMITH_AVAILABLE
    if _LANGSMITH_AVAILABLE is None:
        try:
            import langsmith  # noqa: F401
            _LANGSMITH_AVAILABLE = True
        except ImportError:
            _LANGSMITH_AVAILABLE = False
    return _LANGSMITH_AVAILABLE


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is configured and available."""
    if not _check_langsmith():
        return False
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes", "on")


def get_project_name() -> str:
    return os.getenv("LANGCHAIN_PROJECT", "deepinsight")


def get_tracing_metadata(session_id: str = "", query: str = "", model_name: str = "") -> Dict[str, Any]:
    """Build metadata dict for LangSmith trace context."""
    meta: Dict[str, Any] = {"app": "deepinsight", "version": "0.2.0"}
    if session_id:
        meta["session_id"] = session_id
    if query:
        meta["query"] = query[:200]
    if model_name:
        meta["model_name"] = model_name
    return meta


def trace_context(session_id: str = "", query: str = "", model_name: str = ""):
    """Context manager that attaches metadata to the current LangSmith trace.

    Usage:
        with trace_context(session_id="abc", query="...", model_name="..."):
            result = graph.invoke(state)
    """
    if not is_tracing_enabled():
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    from langsmith import traceable

    metadata = get_tracing_metadata(
        session_id=session_id,
        query=query,
        model_name=model_name,
    )

    class _TraceContext:
        def __init__(self, meta):
            self.meta = meta

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    return _TraceContext(metadata)


def install_langsmith_callback() -> Optional[Any]:
    """Install LangSmith as a global LangChain callback handler.

    Returns the callback handler if successful, None otherwise.
    Call this once at application startup.
    """
    if not is_tracing_enabled():
        return None

    try:
        from langsmith import Client

        project = get_project_name()
        client = Client()
        print(f"[LangSmith] Tracing enabled — project: {project}")
        return client
    except Exception as exc:
        print(f"[LangSmith] Failed to initialize: {exc}")
        return None
