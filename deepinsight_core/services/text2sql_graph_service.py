"""Factory for the graph-native Text2SQL path."""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterable, Optional

from openai import OpenAI

from deepinsight_core.config import DeepInsightSettings
from deepinsight_core.graph.text2sql_runner import Text2SQLGraphRunner
from deepinsight_core.services.graph_agent_adapter import GraphAgentAdapter
from deepinsight_core.services.rag_service import ChromaKnowledgeStore, ChromaRAGBridge, LegacyRAGAdapter
from deepinsight_core.services.sql_generation_service import SQLGenerationService
from deepinsight_core.services.sql_service import SQLService
from sqlalchemy import create_engine


def create_openai_client(api_key: str, base_url: str, timeout: float = 60.0) -> OpenAI:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url.endswith("/v1"):
        clean_url += "/v1"
    return OpenAI(api_key=api_key, base_url=clean_url, timeout=timeout, max_retries=1)


class Text2SQLGraphService:
    _rag_lock = threading.Lock()
    _shared_rag = None
    _shared_rag_refcount = 0

    def __init__(self, settings: DeepInsightSettings):
        self.settings = settings
        self._runner: Optional[Text2SQLGraphRunner] = None
        self._lock = threading.Lock()

    @property
    def runner(self) -> Text2SQLGraphRunner:
        if self._runner is None:
            with self._lock:
                if self._runner is None:
                    self._runner = self._build_runner()
        return self._runner

    def _get_or_create_rag(self):
        with self._rag_lock:
            if self._shared_rag is None:
                from rag_engine import IntelRAG

                rag = IntelRAG(
                    model_path=self.settings.model_path,
                    db_uris=self.settings.db_uris,
                    kb_paths=self.settings.kb_paths,
                )
                if self.settings.use_chroma_rag:
                    store = ChromaKnowledgeStore(
                        persist_path=self.settings.chroma_path,
                        collection_name=f"{self.settings.chroma_collection_prefix}_rag",
                    )
                    inserted = ChromaRAGBridge(
                        store=store,
                        namespace=self.settings.chroma_collection_prefix,
                        source="graph_intel_rag",
                    ).attach(rag)
                    if inserted > 0:
                        print(f"[Chroma] Attached {inserted} documents to collection "
                              f"'{self.settings.chroma_collection_prefix}_rag'")
                self._shared_rag = rag
            return self._shared_rag

    def _build_runner(self) -> Text2SQLGraphRunner:
        rag = self._get_or_create_rag()

        llm_client = create_openai_client(
            api_key=self.settings.api_key,
            base_url=self.settings.api_base,
            timeout=float(self.settings.raw.get("llm_timeout", 45.0)),
        )
        print(f"[DeepInsight] Using DB URI: {self.settings.first_db_uri!r}")
        print(f"[DeepInsight] All DB URIs: {self.settings.db_uris!r}")
        db_engine = create_engine(self.settings.first_db_uri) if self.settings.first_db_uri else None

        from deepinsight_core.tracing import install_langsmith_callback
        install_langsmith_callback()

        return Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(rag),
            generation_service=SQLGenerationService(
                llm_client=llm_client,
                model_name=self.settings.model_name,
                temperature=0.0,
            ),
            sql_service=SQLService(self.settings.first_db_uri, engine=db_engine),
            rag_config=self.settings.raw,
            rag_llm_client=llm_client,
            rag_model_name=self.settings.model_name,
            rag_db_engine=db_engine,
            trace_metadata={
                "model_name": self.settings.model_name,
                "db_uri": self.settings.first_db_uri.split("://")[0] if "://" in self.settings.first_db_uri else "unknown",
                "use_chroma_rag": self.settings.use_chroma_rag,
            },
        )

    def stream_query(self, query: str, session_id: str = "default") -> Iterable[Dict[str, Any]]:
        state = self.runner.invoke_query(query=query, session_id=session_id)
        yield from state.get("events", [])

    def as_agent_adapter(self) -> GraphAgentAdapter:
        return GraphAgentAdapter(self.runner)
