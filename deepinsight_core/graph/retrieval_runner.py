"""Standalone retrieval runner for graph-shaped RAG execution."""

from __future__ import annotations

from typing import Any, Dict, Optional

from deepinsight_core.graph.nodes import retrieve_context_node
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.rag_service import LegacyRAGAdapter


class RetrievalGraphRunner:
    def __init__(
        self,
        rag_adapter: LegacyRAGAdapter,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Any = None,
        model_name: Optional[str] = None,
        db_engine: Any = None,
        enable_pruning: bool = True,
    ):
        self.rag_adapter = rag_adapter
        self.config = config
        self.llm_client = llm_client
        self.model_name = model_name
        self.db_engine = db_engine
        self.enable_pruning = enable_pruning

    def invoke(self, state: DeepInsightState) -> DeepInsightState:
        return retrieve_context_node(
            state=state,
            rag_adapter=self.rag_adapter,
            config=self.config,
            llm_client=self.llm_client,
            model_name=self.model_name,
            db_engine=self.db_engine,
            enable_pruning=self.enable_pruning,
        )

    def invoke_query(self, query: str, session_id: str = "default") -> DeepInsightState:
        state: DeepInsightState = {"session_id": session_id, "query": query, "events": []}
        return self.invoke(state)

    def build_langgraph(self) -> Any:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph is required to build RetrievalGraphRunner graph") from exc

        def retrieve(state: DeepInsightState) -> DeepInsightState:
            return self.invoke(state)

        graph = StateGraph(DeepInsightState)
        graph.add_node("retrieve_context", retrieve)
        graph.set_entry_point("retrieve_context")
        graph.add_edge("retrieve_context", END)
        return graph.compile()
