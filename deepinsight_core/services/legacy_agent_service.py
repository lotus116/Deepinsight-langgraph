"""Compatibility wrapper around the current Text2SQLAgent."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from deepinsight_core.config import DeepInsightSettings
from deepinsight_core.services.rag_service import ChromaKnowledgeStore, ChromaRAGBridge


class LegacyAgentService:
    """Builds and exposes the legacy agent through a small service facade."""

    def __init__(self, settings: DeepInsightSettings):
        self.settings = settings
        self._agent = None

    @property
    def agent(self) -> Any:
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def _build_agent(self) -> Any:
        from agent_core import Text2SQLAgent
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
            bridge = ChromaRAGBridge(
                store=store,
                namespace=self.settings.chroma_collection_prefix,
                source="legacy_intel_rag",
            )
            bridge.attach(rag)

        return Text2SQLAgent(
            api_key=self.settings.api_key,
            base_url=self.settings.api_base,
            model_name=self.settings.model_name,
            db_uris=self.settings.db_uris,
            rag_engine=rag,
            max_retries=self.settings.max_retries,
            max_candidates=self.settings.max_candidates,
            log_file=self.settings.log_file,
            config=self.settings.raw,
            reasoner_model=self.settings.reasoner_model,
            use_reasoner_for_healing=self.settings.use_reasoner_for_healing,
        )

    def stream_query(
        self,
        query: str,
        history_context: Optional[List[Dict[str, Any]]] = None,
        cache_query_key: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        yield from self.agent.generate_and_execute_stream(
            query=query,
            history_context=history_context or [],
            cache_query_key=cache_query_key,
        )
