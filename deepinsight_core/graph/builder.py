"""Graph runner for the refactored Text2SQL workflow.

The first implementation deliberately delegates to the current agent so the
project can gain FastAPI and LangGraph boundaries before the internal nodes are
split apart.  If LangGraph is installed, ``build_graph`` returns a compiled
StateGraph; otherwise ``run_legacy_graph`` remains usable in tests and local
development.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.legacy_agent_service import LegacyAgentService


def run_legacy_graph(agent_service: LegacyAgentService, state: DeepInsightState) -> DeepInsightState:
    events: List[Dict[str, Any]] = []
    result = None
    error = ""

    for event in agent_service.stream_query(
        query=state["query"],
        history_context=state.get("history_context", []),
    ):
        events.append(event)
        event_type = event.get("type")
        if event_type == "result":
            result = {
                "sql": event.get("sql", ""),
                "from_cache": bool(event.get("from_cache", False)),
                "rows": event.get("df").to_dict("records") if event.get("df") is not None else [],
            }
        elif event_type == "error":
            error = event.get("msg", "")

    next_state: DeepInsightState = dict(state)
    next_state["events"] = events
    next_state["result"] = result
    next_state["error"] = error
    return next_state


def iter_legacy_events(agent_service: LegacyAgentService, state: DeepInsightState) -> Iterable[Dict[str, Any]]:
    yield from agent_service.stream_query(
        query=state["query"],
        history_context=state.get("history_context", []),
    )


def build_graph(agent_service: LegacyAgentService) -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise RuntimeError("langgraph is required to build the compiled graph") from exc

    def legacy_agent_node(state: DeepInsightState) -> DeepInsightState:
        return run_legacy_graph(agent_service, state)

    graph = StateGraph(DeepInsightState)
    graph.add_node("legacy_agent", legacy_agent_node)
    graph.set_entry_point("legacy_agent")
    graph.add_edge("legacy_agent", END)
    return graph.compile()
