"""Composable graph nodes for the refactored workflow."""

from __future__ import annotations

from typing import Any, Dict, Optional

from deepinsight_core.graph.events import (
    append_events,
    code_chunk_event,
    code_start_event,
    error_event,
    result_event,
    step_event,
    token_usage_event,
)
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.rag_service import LegacyRAGAdapter, RetrievedContext
from deepinsight_core.services.sql_generation_service import SQLGenerationService
from deepinsight_core.services.sql_service import SQLService


def classify_sql_error(error: str) -> str:
    lowered = (error or "").lower()
    if "only read-only" in lowered or "forbidden sql keyword" in lowered:
        return "security"
    if "no such table" in lowered or "doesn't exist" in lowered or "unknown table" in lowered:
        return "unknown_table"
    if "no such column" in lowered or "unknown column" in lowered:
        return "unknown_column"
    if "syntax" in lowered:
        return "syntax"
    if "empty" in lowered:
        return "empty_result"
    return "other"


def retrieve_context_node(
    state: DeepInsightState,
    rag_adapter: LegacyRAGAdapter,
    config: Optional[Dict[str, Any]] = None,
    llm_client: Any = None,
    model_name: Optional[str] = None,
    db_engine: Any = None,
    enable_pruning: bool = True,
) -> DeepInsightState:
    query = state.get("query", "")
    try:
        context = rag_adapter.retrieve(
            query=query,
            config=config,
            llm_client=llm_client,
            model_name=model_name,
            db_engine=db_engine,
            enable_pruning=enable_pruning,
        )
    except Exception as exc:
        next_state = append_events(state, [error_event(str(exc))])
        next_state["error"] = str(exc)
        return next_state

    next_state = append_events(
        state,
        [
            step_event(
                f"Retrieved context: {len(context.core_tables)} core tables",
                status="complete",
            )
        ],
    )
    next_state["retrieval_result"] = context.to_legacy_dict()
    return next_state


def generate_sql_node(state: DeepInsightState, generation_service: SQLGenerationService) -> DeepInsightState:
    if state.get("error"):
        return state

    result = generation_service.generate(
        query=state.get("query", ""),
        retrieval_result=state.get("retrieval_result", {}),
    )
    if not result.ok:
        error = result.error or "SQL generation returned empty SQL"
        next_state = append_events(state, [error_event(error)])
        next_state["error"] = error
        return next_state

    events = [
        step_event("SQL generated", status="complete"),
        code_start_event(),
        code_chunk_event(result.sql),
    ]
    if result.token_usage:
        events.append(token_usage_event(result.token_usage))

    next_state = append_events(state, events)
    next_state["sql"] = result.sql
    return next_state


def heal_sql_node(state: DeepInsightState, generation_service: SQLGenerationService) -> DeepInsightState:
    error = state.get("error", "")
    category = classify_sql_error(error)
    if category == "security":
        next_state = append_events(state, [step_event("SQL healing skipped for security error", status="complete")])
        next_state["error_category"] = category
        return next_state

    previous_sql = state.get("sql", "")
    result = generation_service.repair(
        query=state.get("query", ""),
        previous_sql=previous_sql,
        error=error,
        retrieval_result=state.get("retrieval_result", {}),
    )
    if not result.ok:
        next_state = append_events(state, [error_event(result.error or "SQL healing returned empty SQL")])
        next_state["error_category"] = category
        return next_state

    events = [
        step_event(f"SQL healing applied ({category})", status="complete"),
        code_start_event("Repaired SQL"),
        code_chunk_event(result.sql),
    ]
    if result.token_usage:
        events.append(token_usage_event(result.token_usage))

    next_state = append_events(state, events)
    next_state["sql"] = result.sql
    next_state["error"] = ""
    next_state["error_category"] = category
    next_state["healing_attempts"] = int(state.get("healing_attempts", 0)) + 1
    return next_state


def validate_sql_node(state: DeepInsightState) -> DeepInsightState:
    sql = state.get("sql", "")
    is_valid, error = SQLService.validate_read_only(sql)
    if is_valid:
        return append_events(
            state,
            [step_event("SQL validation passed", status="complete")],
        )

    next_state = append_events(state, [error_event(error)])
    next_state["error"] = error
    return next_state


def execute_sql_node(state: DeepInsightState, sql_service: SQLService) -> DeepInsightState:
    if state.get("error"):
        return state

    sql = state.get("sql", "")
    result = sql_service.execute(sql)
    if not result.ok:
        next_state = append_events(state, [error_event(result.error)])
        next_state["error"] = result.error
        return next_state

    dataframe = result.dataframe
    assert dataframe is not None
    next_state = append_events(
        state,
        [
            step_event(f"SQL executed successfully, returned {len(dataframe)} rows", status="complete"),
            result_event(sql=result.sql, dataframe=dataframe),
        ],
    )
    next_state["result"] = {
        "sql": result.sql,
        "rows": result.to_records(),
        "from_cache": False,
    }
    return next_state


def run_sql_execution_path(state: DeepInsightState, sql_service: SQLService) -> DeepInsightState:
    validated = validate_sql_node(state)
    return execute_sql_node(validated, sql_service)
