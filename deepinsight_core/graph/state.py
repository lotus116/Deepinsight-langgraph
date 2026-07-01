"""Shared state definitions for DeepInsight graph execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class DeepInsightState(TypedDict, total=False):
    session_id: str
    query: str
    query_understanding: Dict[str, Any]
    sql_plan: Dict[str, Any]
    sql: str
    final_sql: str
    retrieval_result: Dict[str, Any]
    history_context: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    safety_result: Dict[str, Any]
    error_history: List[Dict[str, Any]]
    sql_history: List[str]
    result: Optional[Dict[str, Any]]
    error: str
    error_category: str
    healing_attempts: int
