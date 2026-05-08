"""Shared state definitions for DeepInsight graph execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class DeepInsightState(TypedDict, total=False):
    session_id: str
    query: str
    sql: str
    retrieval_result: Dict[str, Any]
    history_context: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    result: Optional[Dict[str, Any]]
    error: str
    error_category: str
    healing_attempts: int
