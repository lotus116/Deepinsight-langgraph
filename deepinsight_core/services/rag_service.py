"""RAG adapters for the refactored runtime.

``LegacyRAGAdapter`` keeps the existing ``IntelRAG`` behavior available.
``ChromaKnowledgeStore`` is a lightweight persistent store that accepts
precomputed embeddings, so the project can continue using the OpenVINO BGE
embedding model instead of giving Chroma responsibility for model inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


TABLE_MARKER = "\u3010\u8868\u540d\u3011"
DESCRIPTION_MARKER = "\u3010\u63cf\u8ff0\u3011"
QUERY_PATTERN_MARKER = "\u3010\u67e5\u8be2\u6a21\u5f0f\u3011"
QUERY_EXAMPLE_MARKER = "\u3010\u67e5\u8be2\u793a\u4f8b\u3011"
TERM_MARKER = "\u3010\u672f\u8bed\u3011"
BUSINESS_RULE_MARKER = "\u3010\u4e1a\u52a1\u89c4\u5219\u3011"
BUSINESS_METRIC_MARKER = "\u3010\u4e1a\u52a1\u6307\u6807\u3011"
BUSINESS_CONCEPT_MARKER = "\u3010\u4e1a\u52a1\u6982\u5ff5\u3011"


EmbeddingFn = Callable[[str], Sequence[float]]


@dataclass
class RetrievedContext:
    database_index: List[str] = field(default_factory=list)
    rough_candidates: List[Dict[str, Any]] = field(default_factory=list)
    core_tables: List[str] = field(default_factory=list)
    core_table_details: List[Dict[str, Any]] = field(default_factory=list)
    matched_examples: List[Dict[str, Any]] = field(default_factory=list)
    matched_terms: List[Dict[str, Any]] = field(default_factory=list)
    sample_data: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]]) -> "RetrievedContext":
        payload = payload or {}
        return cls(
            database_index=list(payload.get("database_index", [])),
            rough_candidates=list(payload.get("rough_candidates", [])),
            core_tables=list(payload.get("core_tables", [])),
            core_table_details=list(payload.get("core_table_details", [])),
            matched_examples=list(payload.get("matched_examples", [])),
            matched_terms=list(payload.get("matched_terms", [])),
            sample_data=dict(payload.get("sample_data", {})),
            metrics=dict(payload.get("metrics", {})),
        )

    def to_legacy_dict(self) -> Dict[str, Any]:
        return {
            "database_index": self.database_index,
            "rough_candidates": self.rough_candidates,
            "core_tables": self.core_tables,
            "core_table_details": self.core_table_details,
            "matched_examples": self.matched_examples,
            "matched_terms": self.matched_terms,
            "sample_data": self.sample_data,
            "metrics": self.metrics,
        }


class LegacyRAGAdapter:
    def __init__(self, legacy_rag: Any):
        self.legacy_rag = legacy_rag

    def retrieve(
        self,
        query: str,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Any = None,
        model_name: Optional[str] = None,
        db_engine: Any = None,
        enable_pruning: bool = True,
    ) -> RetrievedContext:
        if not hasattr(self.legacy_rag, "retrieve_context"):
            raise TypeError("legacy_rag must provide retrieve_context()")

        payload = self.legacy_rag.retrieve_context(
            query=query,
            config=config,
            llm_client=llm_client,
            model_name=model_name,
            db_engine=db_engine,
            enable_pruning=enable_pruning,
        )
        return RetrievedContext.from_mapping(payload)


class ChromaKnowledgeStore:
    def __init__(self, persist_path: Optional[str], collection_name: str, client: Any = None):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for ChromaKnowledgeStore") from exc

        self.client = client or chromadb.PersistentClient(path=persist_path or "data/chroma")
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_documents(
        self,
        ids: Iterable[str],
        documents: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        embeddings: Optional[Iterable[List[float]]] = None,
    ) -> None:
        id_list = list(ids)
        doc_list = list(documents)
        metadata_list = list(metadatas) if metadatas is not None else None
        embedding_list = list(embeddings) if embeddings is not None else None

        self.collection.upsert(
            ids=id_list,
            documents=doc_list,
            metadatas=metadata_list,
            embeddings=embedding_list,
        )

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 8,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )

    def count(self) -> int:
        return int(self.collection.count())


def infer_document_metadata(document: str, position: int, source: str = "legacy") -> Dict[str, Any]:
    doc_type = "document"
    if document.startswith(TABLE_MARKER):
        doc_type = "table_schema"
    elif document.startswith(QUERY_PATTERN_MARKER):
        doc_type = "query_pattern"
    elif document.startswith(QUERY_EXAMPLE_MARKER):
        doc_type = "few_shot"
    elif document.startswith(TERM_MARKER):
        doc_type = "business_term"
    elif document.startswith(BUSINESS_RULE_MARKER):
        doc_type = "business_rule"
    elif document.startswith(BUSINESS_METRIC_MARKER):
        doc_type = "business_metric"
    elif document.startswith(BUSINESS_CONCEPT_MARKER):
        doc_type = "business_concept"

    return {
        "doc_type": doc_type,
        "table_name": extract_table_name(document) or "",
        "position": position,
        "source": source,
    }


def extract_table_name(document: str) -> str:
    first_line = document.splitlines()[0].strip() if document else ""
    if first_line.startswith(TABLE_MARKER):
        return first_line.replace(TABLE_MARKER, "", 1).strip()
    return ""


def extract_description(document: str) -> str:
    for line in document.splitlines():
        if line.startswith(DESCRIPTION_MARKER):
            return line.replace(DESCRIPTION_MARKER, "", 1).strip()
    return ""


class ChromaRAGIndexer:
    def __init__(self, store: ChromaKnowledgeStore):
        self.store = store

    def upsert_legacy_documents(
        self,
        documents: Sequence[str],
        embedding_fn: EmbeddingFn,
        source: str = "legacy",
        namespace: str = "default",
    ) -> int:
        if not documents:
            return 0

        ids = [f"{namespace}:{idx}" for idx in range(len(documents))]
        embeddings = [list(map(float, embedding_fn(document))) for document in documents]
        metadatas = [
            infer_document_metadata(document=document, position=idx, source=source)
            for idx, document in enumerate(documents)
        ]

        self.store.upsert_documents(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(documents)


class ChromaRAGRoughSearcher:
    def __init__(self, store: ChromaKnowledgeStore):
        self.store = store

    def rough_search(self, query: str, embedding_fn: EmbeddingFn, top_k: int = 12) -> List[Dict[str, Any]]:
        query_embedding = [list(map(float, embedding_fn(query)))]
        raw = self.store.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where={"doc_type": "table_schema"},
        )

        documents = raw.get("documents", [[]])[0] if raw.get("documents") else []
        metadatas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
        distances = raw.get("distances", [[]])[0] if raw.get("distances") else []

        candidates: List[Dict[str, Any]] = []
        for idx, document in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            distance = float(distances[idx]) if idx < len(distances) else 1.0
            table_name = metadata.get("table_name") or extract_table_name(document)
            candidates.append(
                {
                    "table_name": table_name,
                    "description": extract_description(document),
                    "score": 1.0 - distance,
                    "rank": idx + 1,
                    "document": document,
                }
            )

        return candidates


class ChromaRAGBridge:
    """Attach Chroma rough search to a legacy IntelRAG-like object.

    The legacy object keeps ownership of document parsing, embeddings, table
    detail extraction, term matching, LLM pruning, and dependency completion.
    This bridge only replaces `_rough_search()` after indexing the already-built
    `documents` list.  It is opt-in and reversible by recreating the legacy RAG
    object.
    """

    def __init__(
        self,
        store: ChromaKnowledgeStore,
        namespace: str = "default",
        source: str = "legacy",
    ):
        self.store = store
        self.namespace = namespace
        self.source = source
        self.indexer = ChromaRAGIndexer(store)
        self.searcher = ChromaRAGRoughSearcher(store)

    def attach(self, legacy_rag: Any) -> int:
        documents = list(getattr(legacy_rag, "documents", []) or [])
        if not documents:
            return 0
        if not hasattr(legacy_rag, "_get_embedding"):
            raise TypeError("legacy_rag must provide _get_embedding()")

        inserted = self.indexer.upsert_legacy_documents(
            documents=documents,
            embedding_fn=legacy_rag._get_embedding,
            source=self.source,
            namespace=self.namespace,
        )

        def chroma_rough_search(rag_self: Any, query: str, top_k: int = 12) -> List[Dict[str, Any]]:
            start_time = time.perf_counter()
            candidates = self.searcher.rough_search(
                query=query,
                embedding_fn=rag_self._get_embedding,
                top_k=top_k,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            print(f"[Chroma] Rough search completed: {len(candidates)} candidates ({latency_ms:.2f}ms)")
            return candidates

        legacy_rag._rough_search = MethodType(chroma_rough_search, legacy_rag)
        legacy_rag.chroma_store = self.store
        legacy_rag.chroma_enabled = True
        return inserted
