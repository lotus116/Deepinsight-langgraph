"""Standalone SQL graph runner.

This runner is intentionally small. It gives tests and future API endpoints a
graph-shaped execution path before SQL generation is fully split out of the
legacy agent.
"""

from __future__ import annotations

from typing import Any, Dict

from deepinsight_core.graph.nodes import run_sql_execution_path
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.sql_service import SQLService


class SQLGraphRunner:
    def __init__(self, sql_service: SQLService):
        self.sql_service = sql_service

    def invoke(self, state: DeepInsightState) -> DeepInsightState:
        return run_sql_execution_path(state, self.sql_service)

    def invoke_sql(self, query: str, sql: str, session_id: str = "default") -> DeepInsightState:
        state: DeepInsightState = {
            "session_id": session_id,
            "query": query,
            "sql": sql,
            "events": [],
        }
        return self.invoke(state)

    def build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph is required to build SQLGraphRunner graph") from exc

        def execute_path(state: DeepInsightState) -> DeepInsightState:
            return self.invoke(state)

        graph = StateGraph(DeepInsightState)
        graph.add_node("validate_and_execute_sql", execute_path)
        graph.set_entry_point("validate_and_execute_sql")
        graph.add_edge("validate_and_execute_sql", END)
        return graph.compile()
