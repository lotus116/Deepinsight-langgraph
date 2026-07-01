"""Composable graph nodes for the refactored workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import time

from deepinsight_core.graph.events import (
    append_events,
    code_chunk_event,
    code_start_event,
    error_event,
    result_event,
    step_event,
    thought_chunk_event,
    thought_start_event,
    token_usage_event,
)
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.nl2sql.models import AgentTraceStep, QueryUnderstanding, SQLPlan
from deepinsight_core.nl2sql.sql_safety import SQLSafetyChecker
from deepinsight_core.services.rag_service import LegacyRAGAdapter, RetrievedContext
from deepinsight_core.services.sql_generation_service import SQLGenerationService
from deepinsight_core.services.sql_service import SQLService


def append_trace(state: DeepInsightState, node: str, status: str, summary: str, start_time: float, details: Optional[Dict[str, Any]] = None) -> DeepInsightState:
    next_state = dict(state)
    step = AgentTraceStep(
        node=node,
        status=status,
        summary=summary,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        details=details or {},
    ).to_dict()
    next_state["trace"] = list(state.get("trace", [])) + [step]
    return next_state


def _schema_columns(retrieval: Dict[str, Any]) -> Dict[str, list[str]]:
    columns: Dict[str, list[str]] = {}
    for table_name, table_columns in (retrieval.get("schema_catalog") or {}).items():
        if table_name and table_columns:
            columns[table_name] = list(table_columns)
    for detail in retrieval.get("core_table_details", []) or []:
        table_name = detail.get("table_name")
        if not table_name:
            continue
        table_columns = []
        for column in detail.get("columns", []) or []:
            if isinstance(column, dict):
                name = column.get("col") or column.get("name")
            else:
                name = str(column)
            if name:
                table_columns.append(name)
        if table_columns:
            columns[table_name] = table_columns
    return columns


def _load_schema_catalog(config: Optional[Dict[str, Any]]) -> Dict[str, list[str]]:
    config = config or {}
    schema_paths = []
    if config.get("schema_path"):
        schema_paths.append(config["schema_path"])
    schema_paths.extend(config.get("kb_paths_list") or config.get("kb_paths") or [])

    catalog: Dict[str, list[str]] = {}
    for raw_path in schema_paths:
        path = Path(str(raw_path))
        if not path.exists() or path.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for table in data:
            if not isinstance(table, dict):
                continue
            table_name = table.get("table_name")
            if not table_name:
                continue
            table_columns = []
            for column in table.get("columns", []) or []:
                if isinstance(column, dict):
                    name = column.get("col") or column.get("name")
                    if name:
                        table_columns.append(name)
            if table_columns:
                catalog[table_name] = table_columns
    return catalog


def _augment_retrieval_schema(retrieval: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    catalog = _load_schema_catalog(config)
    if not catalog:
        return retrieval

    augmented = dict(retrieval)
    augmented["schema_catalog"] = catalog
    details = list(augmented.get("core_table_details", []) or [])
    detail_by_table = {
        detail.get("table_name", "").lower(): detail
        for detail in details
        if isinstance(detail, dict) and detail.get("table_name")
    }
    for table_name in augmented.get("core_tables", []) or []:
        columns = catalog.get(table_name)
        if not columns:
            columns = next((cols for name, cols in catalog.items() if name.lower() == table_name.lower()), [])
        if not columns:
            continue
        key = table_name.lower()
        if key in detail_by_table:
            detail = detail_by_table[key]
            if not detail.get("columns"):
                detail["columns"] = [{"col": column} for column in columns]
        else:
            details.append({"table_name": table_name, "columns": [{"col": column} for column in columns]})
    augmented["core_table_details"] = details
    return augmented


def _schema_summary(retrieval: Dict[str, Any]) -> str:
    schema_columns = _schema_columns(retrieval)
    lines = ["Schema retrieval summary:"]
    core_tables = retrieval.get("core_tables", []) or []
    lines.append(f"- selected tables: {', '.join(core_tables) if core_tables else 'none'}")
    for table_name in core_tables:
        columns = schema_columns.get(table_name, [])
        if columns:
            lines.append(f"- {table_name}: {', '.join(columns)}")
    matched_terms = retrieval.get("matched_terms", []) or []
    if matched_terms:
        terms = []
        for item in matched_terms[:5]:
            term = item.get("term", "")
            explanation = item.get("explanation", "")
            terms.append(f"{term} ({explanation})" if explanation else term)
        lines.append("- matched business terms: " + "; ".join(terms))
    matched_examples = retrieval.get("matched_examples", []) or []
    if matched_examples:
        examples = [item.get("query", str(item)) for item in matched_examples[:3]]
        lines.append("- matched examples: " + "; ".join(examples))
    return "\n".join(lines)


def understand_query_node(state: DeepInsightState) -> DeepInsightState:
    start = time.perf_counter()
    query = state.get("query", "")
    lowered = query.lower()
    metrics = []
    dimensions = []
    filters = []
    limit = None

    metric_markers = {
        "count": ["多少", "数量", "count", "num", "number"],
        "sum": ["总", "销售额", "金额", "sum", "revenue", "sales"],
        "avg": ["平均", "avg", "average"],
        "rank": ["排名", "top", "最高", "最低", "前"],
    }
    for metric, markers in metric_markers.items():
        if any(marker in lowered or marker in query for marker in markers):
            metrics.append(metric)
    if any(marker in lowered or marker in query for marker in ["每个", "按", "各", "分布", "by", "per"]):
        dimensions.append("grouping_dimension")
    if any(marker in lowered or marker in query for marker in ["where", "大于", "小于", "超过", "低于", "在", "from", "to"]):
        filters.append("possible_filter")
    if any(marker in lowered or marker in query for marker in ["top", "前"]):
        limit = 10

    ambiguities = []
    if not metrics:
        ambiguities.append("metric is implicit; infer it from the selected table and question wording")
    if not dimensions and any(marker in lowered or marker in query for marker in ["by", "per", "每个", "按"]):
        ambiguities.append("grouping dimension may need schema context")
    if any(marker in lowered or marker in query for marker in ["top", "最高", "最低", "前"]):
        ambiguities.append("ranking direction depends on the business metric")

    understanding = QueryUnderstanding(
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        limit=limit,
        ambiguities=ambiguities,
    ).to_dict()
    selected_possibility = {
        "rank": 1,
        "natural_description": "Answer with read-only SQL using the most relevant retrieved schema.",
        "confidence": 0.8 if metrics or dimensions or filters or limit else 0.65,
        "key_interpretations": {
            "intent": {"desc": understanding["intent"]},
            "metrics": {"desc": ", ".join(metrics) if metrics else "implicit"},
            "dimensions": {"desc": ", ".join(dimensions) if dimensions else "schema-dependent"},
            "filters": {"desc": ", ".join(filters) if filters else "none detected"},
        },
        "ambiguity_resolutions": {"ambiguities": ambiguities},
    }
    alternatives = []
    if not metrics:
        alternatives.append(
            {
                "rank": 2,
                "natural_description": "Interpret the request as a count-style query if no explicit measure is given.",
                "confidence": 0.45,
                "key_interpretations": {"metric": {"desc": "count"}},
                "ambiguity_resolutions": {"metric": "count"},
            }
        )
    if limit:
        alternatives.append(
            {
                "rank": len(alternatives) + 2,
                "natural_description": "Interpret TOP ranking as descending by the generated aggregate metric.",
                "confidence": 0.55,
                "key_interpretations": {"rank": {"desc": "descending"}},
                "ambiguity_resolutions": {"rank_direction": "descending"},
            }
        )
    thought_lines = [
        "Query understanding summary:",
        f"- intent: {understanding['intent']}",
        f"- metrics: {', '.join(metrics) if metrics else 'implicit'}",
        f"- dimensions: {', '.join(dimensions) if dimensions else 'schema-dependent'}",
        f"- filters: {', '.join(filters) if filters else 'none detected'}",
        f"- limit: {limit if limit else 'not specified'}",
    ]
    if ambiguities:
        thought_lines.append("- possible ambiguities: " + "; ".join(ambiguities))

    next_state = dict(state)
    next_state["query_understanding"] = understanding
    next_state["selected_possibility"] = selected_possibility
    next_state["alternatives"] = alternatives
    next_state = append_events(
        next_state,
        [
            thought_start_event(),
            thought_chunk_event("\n".join(thought_lines)),
            step_event("Query understood", status="complete"),
        ],
    )
    return append_trace(next_state, "understand_query", "success", f"metrics={metrics or ['implicit']}", start, understanding)

def plan_sql_node(state: DeepInsightState) -> DeepInsightState:
    start = time.perf_counter()
    retrieval = state.get("retrieval_result", {})
    understanding = state.get("query_understanding", {})
    plan = SQLPlan(
        required_tables=list(retrieval.get("core_tables", [])),
        metrics=list(understanding.get("metrics", [])),
        filters=list(understanding.get("filters", [])),
        limit=understanding.get("limit"),
        assumptions=[
            "Use only retrieved schema fields",
            "Prefer explicit columns and read-only SQL",
        ],
    ).to_dict()
    next_state = dict(state)
    next_state["sql_plan"] = plan
    plan_lines = [
        "SQL planning summary:",
        f"- required tables: {', '.join(plan['required_tables']) if plan['required_tables'] else 'none'}",
        f"- metrics: {', '.join(plan['metrics']) if plan['metrics'] else 'implicit'}",
        f"- filters: {', '.join(plan['filters']) if plan['filters'] else 'none detected'}",
        f"- assumptions: {'; '.join(plan['assumptions'])}",
    ]
    next_state = append_events(
        next_state,
        [
            thought_start_event(),
            thought_chunk_event("\n".join(plan_lines)),
            step_event("SQL plan prepared", status="complete"),
        ],
    )
    return append_trace(next_state, "plan_sql", "success", f"tables={len(plan['required_tables'])}", start, plan)


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
    start = time.perf_counter()
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
        return append_trace(next_state, "retrieve_schema", "error", str(exc), start)

    next_state = append_events(
        state,
        [
            step_event(
                f"Retrieved context: {len(context.core_tables)} core tables",
                status="complete",
            )
        ],
    )
    next_state["retrieval_result"] = _augment_retrieval_schema(context.to_legacy_dict(), config)
    next_state = append_events(
        next_state,
        [
            thought_start_event(),
            thought_chunk_event(_schema_summary(next_state["retrieval_result"])),
        ],
    )
    return append_trace(
        next_state,
        "retrieve_schema",
        "success",
        f"selected {len(context.core_tables)} table(s)",
        start,
        {
            "core_tables": context.core_tables,
            "rough_candidates": context.rough_candidates[:5],
            "matched_examples": context.matched_examples[:3],
            "matched_terms": context.matched_terms[:5],
        },
    )


def generate_sql_node(state: DeepInsightState, generation_service: SQLGenerationService) -> DeepInsightState:
    start = time.perf_counter()
    if state.get("error"):
        return state

    retrieval_payload = dict(state.get("retrieval_result", {}))
    retrieval_payload["sql_plan"] = state.get("sql_plan", {})
    result = generation_service.generate(
        query=state.get("query", ""),
        retrieval_result=retrieval_payload,
    )
    if not result.ok:
        error = result.error or "SQL generation returned empty SQL"
        next_state = append_events(state, [error_event(error)])
        next_state["error"] = error
        return append_trace(next_state, "generate_sql", "error", error, start)

    events = [
        step_event("SQL generated", status="complete"),
        code_start_event(),
        code_chunk_event(result.sql),
    ]
    if result.token_usage:
        events.append(token_usage_event(result.token_usage))

    next_state = append_events(state, events)
    next_state["sql"] = result.sql
    next_state["sql_history"] = list(state.get("sql_history", [])) + [result.sql]
    return append_trace(next_state, "generate_sql", "success", "SQL generated", start, {"sql": result.sql})


def heal_sql_node(state: DeepInsightState, generation_service: SQLGenerationService) -> DeepInsightState:
    start = time.perf_counter()
    error = state.get("error", "")
    category = classify_sql_error(error)
    if category == "security":
        next_state = append_events(state, [step_event("SQL healing skipped for security error", status="complete")])
        next_state["error_category"] = category
        return append_trace(next_state, "repair_sql", "skipped", "security error is not repaired", start)

    previous_sql = state.get("sql", "")
    retrieval_payload = dict(state.get("retrieval_result", {}))
    retrieval_payload["sql_plan"] = state.get("sql_plan", {})
    result = generation_service.repair(
        query=state.get("query", ""),
        previous_sql=previous_sql,
        error=error,
        retrieval_result=retrieval_payload,
    )
    if not result.ok:
        next_state = append_events(state, [error_event(result.error or "SQL healing returned empty SQL")])
        next_state["error_category"] = category
        return append_trace(next_state, "repair_sql", "error", result.error or "empty repaired SQL", start)

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
    next_state["sql_history"] = list(state.get("sql_history", [])) + [result.sql]
    next_state["error_history"] = list(state.get("error_history", [])) + [
        {"category": category, "error": error, "previous_sql": previous_sql}
    ]
    return append_trace(next_state, "repair_sql", "success", f"repaired {category}", start, {"sql": result.sql})


def validate_sql_node(state: DeepInsightState) -> DeepInsightState:
    start = time.perf_counter()
    sql = state.get("sql", "")
    retrieval = state.get("retrieval_result", {})
    tables = retrieval.get("database_index", [])
    safety = SQLSafetyChecker(
        known_tables=tables,
        known_columns=_schema_columns(retrieval),
    ).validate(sql)
    next_state = dict(state)
    next_state["safety_result"] = safety.to_dict()
    if safety.safe:
        next_state["sql"] = safety.normalized_sql
        next_state["final_sql"] = safety.normalized_sql
        next_state = append_events(next_state, [step_event("SQL validation passed", status="complete")])
        return append_trace(next_state, "safety_check", "success", "SQL is read-only and safe", start, safety.to_dict())

    next_state = append_events(next_state, [error_event(safety.reason)])
    next_state["error"] = safety.reason
    next_state["error_category"] = classify_sql_error(safety.reason)
    return append_trace(next_state, "safety_check", "error", safety.reason, start, safety.to_dict())


def execute_sql_node(state: DeepInsightState, sql_service: SQLService) -> DeepInsightState:
    start = time.perf_counter()
    if state.get("error"):
        return state

    sql = state.get("sql", "")
    result = sql_service.execute(sql)
    if not result.ok:
        next_state = append_events(state, [error_event(result.error)])
        next_state["error"] = result.error
        return append_trace(next_state, "execute_sql", "error", result.error, start)

    dataframe = result.dataframe
    assert dataframe is not None
    next_state = append_events(
        state,
        [
            step_event(f"SQL executed successfully, returned {len(dataframe)} rows", status="complete"),
            result_event(sql=result.sql, dataframe=dataframe, selected_possibility=state.get("selected_possibility"), alternatives=state.get("alternatives", [])),
        ],
    )
    next_state["result"] = {
        "sql": result.sql,
        "rows": result.to_records(),
        "from_cache": False,
    }
    return append_trace(next_state, "execute_sql", "success", f"returned {len(dataframe)} row(s)", start, {"row_count": len(dataframe)})


def validate_result_node(state: DeepInsightState) -> DeepInsightState:
    start = time.perf_counter()
    result = state.get("result") or {}
    rows = result.get("rows", [])
    validation = {
        "valid": True,
        "warnings": [],
        "row_count": len(rows),
    }
    if len(rows) == 0:
        validation["warnings"].append("empty_result")
    next_state = dict(state)
    next_state["result_validation"] = validation
    return append_trace(next_state, "validate_result", "success", f"row_count={len(rows)}", start, validation)


def run_sql_execution_path(state: DeepInsightState, sql_service: SQLService) -> DeepInsightState:
    validated = validate_sql_node(state)
    return execute_sql_node(validated, sql_service)

