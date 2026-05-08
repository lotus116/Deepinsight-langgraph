"""Composable Text2SQL graph runner.

This runner wires retrieval, generation, validation, and SQL execution together.
It is still opt-in and does not replace the legacy production path.
"""

from __future__ import annotations

from typing import Any

from deepinsight_core.graph.events import step_event
from deepinsight_core.graph.nodes import generate_sql_node, heal_sql_node, retrieve_context_node, run_sql_execution_path
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.rag_service import LegacyRAGAdapter
from deepinsight_core.services.sql_generation_service import SQLGenerationService
from deepinsight_core.services.sql_service import SQLService


class Text2SQLGraphRunner:
    def __init__(
        self,
        rag_adapter: LegacyRAGAdapter,
        generation_service: SQLGenerationService,
        sql_service: SQLService,
        enable_pruning: bool = True,
        max_healing_attempts: int = 1,
        rag_config: dict | None = None,
        rag_llm_client: Any = None,
        rag_model_name: str | None = None,
        rag_db_engine: Any = None,
    ):
        self.rag_adapter = rag_adapter
        self.generation_service = generation_service
        self.sql_service = sql_service
        self.enable_pruning = enable_pruning
        self.max_healing_attempts = max_healing_attempts
        self.rag_config = rag_config
        self.rag_llm_client = rag_llm_client
        self.rag_model_name = rag_model_name
        self.rag_db_engine = rag_db_engine

    def invoke(self, state: DeepInsightState) -> DeepInsightState:
        retrieved = retrieve_context_node(
            state=state,
            rag_adapter=self.rag_adapter,
            enable_pruning=self.enable_pruning,
            config=self.rag_config,
            llm_client=self.rag_llm_client,
            model_name=self.rag_model_name,
            db_engine=self.rag_db_engine,
        )
        generated = generate_sql_node(retrieved, self.generation_service)
        executed = run_sql_execution_path(generated, self.sql_service)
        attempts = 0
        while executed.get("error") and attempts < self.max_healing_attempts:
            healed = heal_sql_node(executed, self.generation_service)
            if healed.get("error") or healed.get("sql") == executed.get("sql"):
                return healed
            executed = run_sql_execution_path(healed, self.sql_service)
            attempts += 1
        return executed

    def invoke_query(self, query: str, session_id: str = "default") -> DeepInsightState:
        state: DeepInsightState = {"session_id": session_id, "query": query, "events": []}
        return self.invoke(state)

    def stream_query(self, query: str, session_id: str = "default") -> Any:
        state: DeepInsightState = {"session_id": session_id, "query": query, "events": []}

        yield step_event("Retrieving context", status="running")
        retrieved = retrieve_context_node(
            state=state,
            rag_adapter=self.rag_adapter,
            enable_pruning=self.enable_pruning,
            config=self.rag_config,
            llm_client=self.rag_llm_client,
            model_name=self.rag_model_name,
            db_engine=self.rag_db_engine,
        )
        for event in retrieved.get("events", [])[len(state.get("events", [])):]:
            yield event
        yield {"type": "_retrieval_result", "retrieval_result": retrieved.get("retrieval_result", {})}
        if retrieved.get("error"):
            return retrieved

        yield step_event("Generating SQL", status="running")
        generated = generate_sql_node(retrieved, self.generation_service)
        for event in generated.get("events", [])[len(retrieved.get("events", [])):]:
            yield event
        if generated.get("error"):
            return generated

        yield step_event("Validating and executing SQL", status="running")
        executed = run_sql_execution_path(generated, self.sql_service)
        for event in executed.get("events", [])[len(generated.get("events", [])):]:
            yield event

        attempts = 0
        while executed.get("error") and attempts < self.max_healing_attempts:
            yield step_event("Repairing SQL", status="running")
            healed = heal_sql_node(executed, self.generation_service)
            for event in healed.get("events", [])[len(executed.get("events", [])):]:
                yield event
            if healed.get("error") or healed.get("sql") == executed.get("sql"):
                return healed

            yield step_event("Validating and executing repaired SQL", status="running")
            previous_count = len(healed.get("events", []))
            executed = run_sql_execution_path(healed, self.sql_service)
            for event in executed.get("events", [])[previous_count:]:
                yield event
            attempts += 1

        return executed

    def build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph is required to build Text2SQLGraphRunner graph") from exc

        def retrieve(state: DeepInsightState) -> DeepInsightState:
            return retrieve_context_node(
                state=state,
                rag_adapter=self.rag_adapter,
                enable_pruning=self.enable_pruning,
                config=self.rag_config,
                llm_client=self.rag_llm_client,
                model_name=self.rag_model_name,
                db_engine=self.rag_db_engine,
            )

        def generate(state: DeepInsightState) -> DeepInsightState:
            return generate_sql_node(state, self.generation_service)

        def execute(state: DeepInsightState) -> DeepInsightState:
            return run_sql_execution_path(state, self.sql_service)

        def heal(state: DeepInsightState) -> DeepInsightState:
            return heal_sql_node(state, self.generation_service)

        def route_after_execute(state: DeepInsightState) -> str:
            if not state.get("error"):
                return "end"
            if int(state.get("healing_attempts", 0)) >= self.max_healing_attempts:
                return "end"
            if state.get("error_category") == "security":
                return "end"
            if state.get("sql"):
                return "heal"
            return "end"

        graph = StateGraph(DeepInsightState)
        graph.add_node("retrieve_context", retrieve)
        graph.add_node("generate_sql", generate)
        graph.add_node("validate_and_execute_sql", execute)
        graph.add_node("heal_sql", heal)
        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", "generate_sql")
        graph.add_edge("generate_sql", "validate_and_execute_sql")
        graph.add_conditional_edges(
            "validate_and_execute_sql",
            route_after_execute,
            {"heal": "heal_sql", "end": END},
        )
        graph.add_edge("heal_sql", "validate_and_execute_sql")
        return graph.compile()
