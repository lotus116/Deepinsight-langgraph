"""Standalone SQL generation runner."""

from __future__ import annotations

from typing import Any

from deepinsight_core.graph.nodes import generate_sql_node
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.sql_generation_service import SQLGenerationService


class SQLGenerationGraphRunner:
    def __init__(self, generation_service: SQLGenerationService):
        self.generation_service = generation_service

    def invoke(self, state: DeepInsightState) -> DeepInsightState:
        return generate_sql_node(state, self.generation_service)

    def build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph is required to build SQLGenerationGraphRunner graph") from exc

        def generate(state: DeepInsightState) -> DeepInsightState:
            return self.invoke(state)

        graph = StateGraph(DeepInsightState)
        graph.add_node("generate_sql", generate)
        graph.set_entry_point("generate_sql")
        graph.add_edge("generate_sql", END)
        return graph.compile()
