"""Composable Text2SQL graph runner.

This runner wires retrieval, generation, validation, and SQL execution together.
"""

from __future__ import annotations

from typing import Any

from deepinsight_core.graph.events import step_event, trace_event
from deepinsight_core.graph.nodes import (
    generate_sql_node,
    heal_sql_node,
    plan_sql_node,
    retrieve_context_node,
    run_sql_execution_path,
    understand_query_node,
    validate_result_node,
)
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
        trace_metadata: dict | None = None,
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
        self.trace_metadata = trace_metadata or {}

    def _make_invoke_config(self) -> dict:
        from deepinsight_core.tracing import is_tracing_enabled, get_tracing_metadata

        cfg: dict = {}
        if is_tracing_enabled():
            cfg["metadata"] = self.trace_metadata
            cfg["run_name"] = "text2sql"
            cfg["tags"] = ["deepinsight", "text2sql"]
            if self.trace_metadata.get("session_id"):
                cfg["tags"].append(f"session:{self.trace_metadata['session_id'][:8]}")
        return cfg

    def _trace(self, fn: Any, name: str) -> Any:
        from deepinsight_core.tracing import is_tracing_enabled

        if not is_tracing_enabled():
            return fn
        try:
            from langsmith import traceable
            return traceable(
                fn,
                name=name,
                metadata=self.trace_metadata,
                tags=["deepinsight", name],
            )
        except ImportError:
            return fn

    def invoke(self, state: DeepInsightState) -> DeepInsightState:
        return self._trace(self._invoke_impl, "text2sql_invoke")(state)

    def _invoke_impl(self, state: DeepInsightState) -> DeepInsightState:
        understood = understand_query_node(state)
        retrieved = retrieve_context_node(
            state=understood,
            rag_adapter=self.rag_adapter,
            enable_pruning=self.enable_pruning,
            config=self.rag_config,
            llm_client=self.rag_llm_client,
            model_name=self.rag_model_name,
            db_engine=self.rag_db_engine,
        )
        planned = plan_sql_node(retrieved)
        generated = generate_sql_node(planned, self.generation_service)
        executed = run_sql_execution_path(generated, self.sql_service)
        attempts = 0
        while executed.get("error") and attempts < self.max_healing_attempts:
            healed = heal_sql_node(executed, self.generation_service)
            if healed.get("error") or healed.get("sql") == executed.get("sql"):
                return healed
            executed = run_sql_execution_path(healed, self.sql_service)
            attempts += 1
        if executed.get("result"):
            executed = validate_result_node(executed)
        return executed

    def invoke_query(self, query: str, session_id: str = "default") -> DeepInsightState:
        state: DeepInsightState = {"session_id": session_id, "query": query, "events": []}
        return self.invoke(state)

    def stream_query(self, query: str, session_id: str = "default") -> Any:
        return self._trace(self._stream_query_impl, "text2sql_stream")(query, session_id)

    def _stream_query_impl(self, query: str, session_id: str = "default") -> Any:
        state: DeepInsightState = {"session_id": session_id, "query": query, "events": []}

        def nonterminal_event(event: dict) -> dict:
            if event.get("type") == "error":
                return {"type": "error_log", "content": event.get("msg", "")}
            return event

        yield step_event("Understanding query", status="running")
        understood = understand_query_node(state)
        for event in understood.get("events", [])[len(state.get("events", [])):]:
            yield event

        yield step_event("Retrieving context", status="running")
        retrieved = retrieve_context_node(
            state=understood,
            rag_adapter=self.rag_adapter,
            enable_pruning=self.enable_pruning,
            config=self.rag_config,
            llm_client=self.rag_llm_client,
            model_name=self.rag_model_name,
            db_engine=self.rag_db_engine,
        )
        for event in retrieved.get("events", [])[len(understood.get("events", [])):]:
            yield event
        yield {"type": "_retrieval_result", "retrieval_result": retrieved.get("retrieval_result", {})}
        if retrieved.get("error"):
            yield trace_event(retrieved.get("trace", []))
            return retrieved

        yield step_event("Planning SQL", status="running")
        planned = plan_sql_node(retrieved)
        for event in planned.get("events", [])[len(retrieved.get("events", [])):]:
            yield event

        yield step_event("Generating SQL", status="running")
        generated = generate_sql_node(planned, self.generation_service)
        for event in generated.get("events", [])[len(planned.get("events", [])):]:
            yield event
        if generated.get("error"):
            yield trace_event(generated.get("trace", []))
            return generated

        yield step_event("Validating and executing SQL", status="running")
        executed = run_sql_execution_path(generated, self.sql_service)
        for event in executed.get("events", [])[len(generated.get("events", [])):]:
            yield nonterminal_event(event) if executed.get("error") else event

        attempts = 0
        while executed.get("error") and attempts < self.max_healing_attempts:
            yield step_event("Repairing SQL", status="running")
            healed = heal_sql_node(executed, self.generation_service)
            for event in healed.get("events", [])[len(executed.get("events", [])):]:
                yield nonterminal_event(event) if healed.get("error") else event
            if healed.get("error") or healed.get("sql") == executed.get("sql"):
                message = healed.get("error") or "SQL repair did not change the failed query"
                yield {"type": "error", "msg": message}
                return healed

            yield step_event("Validating and executing repaired SQL", status="running")
            previous_count = len(healed.get("events", []))
            executed = run_sql_execution_path(healed, self.sql_service)
            for event in executed.get("events", [])[previous_count:]:
                yield nonterminal_event(event) if executed.get("error") else event
            attempts += 1

        if executed.get("error"):
            yield {"type": "error", "msg": executed.get("error", "SQL execution failed")}
            yield trace_event(executed.get("trace", []))
            return executed

        if executed.get("result"):
            previous_count = len(executed.get("events", []))
            executed = validate_result_node(executed)
            for event in executed.get("events", [])[previous_count:]:
                yield event

        yield trace_event(executed.get("trace", []))
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
                return "validate_result"
            if int(state.get("healing_attempts", 0)) >= self.max_healing_attempts:
                return "end"
            if state.get("error_category") == "security":
                return "end"
            if state.get("sql"):
                return "heal"
            return "end"

        graph = StateGraph(DeepInsightState)
        graph.add_node("retrieve_context", retrieve)
        graph.add_node("understand_query", understand_query_node)
        graph.add_node("plan_sql", plan_sql_node)
        graph.add_node("generate_sql", generate)
        graph.add_node("validate_and_execute_sql", execute)
        graph.add_node("validate_result", validate_result_node)
        graph.add_node("heal_sql", heal)
        graph.set_entry_point("understand_query")
        graph.add_edge("understand_query", "retrieve_context")
        graph.add_edge("retrieve_context", "plan_sql")
        graph.add_edge("plan_sql", "generate_sql")
        graph.add_edge("generate_sql", "validate_and_execute_sql")
        graph.add_conditional_edges(
            "validate_and_execute_sql",
            route_after_execute,
            {"heal": "heal_sql", "validate_result": "validate_result", "end": END},
        )
        graph.add_edge("heal_sql", "validate_and_execute_sql")
        graph.add_edge("validate_result", END)
        return graph.compile()
