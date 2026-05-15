"""FastAPI entry point for the refactored DeepInsight service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from deepinsight_core.config import DeepInsightSettings
from deepinsight_core.graph.sql_runner import SQLGraphRunner
from deepinsight_core.graph.state import DeepInsightState
from deepinsight_core.services.rag_service import ChromaKnowledgeStore, ChromaRAGIndexer
from deepinsight_core.services.sql_service import SQLService
from deepinsight_core.services.text2sql_graph_service import Text2SQLGraphService, create_openai_client


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return str(value)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str = "default"
    run_id: Optional[str] = None
    history_context: List[Dict[str, Any]] = Field(default_factory=list)


class SQLExecuteRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    query: str = ""
    session_id: str = "default"


class InsightRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: str = "default"


class SessionCreateRequest(BaseModel):
    title: str = "New session"


class IndexRebuildRequest(BaseModel):
    namespace: Optional[str] = None
    source: str = "api_rebuild"


def create_app(
    settings: Optional[DeepInsightSettings] = None,
    graph_service_override: Optional[Any] = None,
    sql_runner_override: Optional[Any] = None,
):
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("fastapi and pydantic are required to create the API app") from exc

    if settings is None:
        from utils import load_config

        settings = DeepInsightSettings.from_mapping(load_config())
    settings.ensure_storage_dirs()

    # Pre-initialize services to avoid lazy-init race conditions under concurrent requests
    graph_service = graph_service_override or Text2SQLGraphService(settings)
    sql_runner = sql_runner_override or SQLGraphRunner(SQLService(settings.first_db_uri))

    app = FastAPI(title="DeepInsight API", version="0.2.0")
    sessions: Dict[str, Dict[str, Any]] = {}
    runs: Dict[str, Dict[str, Any]] = {}

    CHROMA_COLLECTION = f"{settings.chroma_collection_prefix}_rag"

    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_run(session_id: str, query: str, requested_run_id: Optional[str] = None) -> str:
        run_id = requested_run_id or str(uuid4())
        runs[run_id] = {
            "run_id": run_id,
            "session_id": session_id,
            "query": query,
            "status": "running",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "error": "",
        }
        return run_id

    def finish_run(run_id: str, status: str, error: str = "") -> None:
        if run_id in runs:
            runs[run_id]["status"] = status
            runs[run_id]["error"] = error
            runs[run_id]["updated_at"] = now_iso()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": settings.model_name, "db_configured": bool(settings.first_db_uri)}

    @app.post("/v1/sessions")
    def create_session(request: SessionCreateRequest) -> Dict[str, Any]:
        session_id = str(uuid4())
        sessions[session_id] = {
            "session_id": session_id,
            "title": request.title,
            "created_at": now_iso(),
        }
        return sessions[session_id]

    @app.get("/v1/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        if run_id not in runs:
            raise HTTPException(status_code=404, detail="run not found")
        return runs[run_id]

    @app.post("/v1/query/graph/stream")
    def query_graph_stream(request: QueryRequest) -> StreamingResponse:
        if not settings.api_key:
            raise HTTPException(status_code=400, detail="api_key is not configured")
        run_id = create_run(request.session_id, request.query, request.run_id)

        def event_source():
            yield f"data: {json.dumps({'type': 'run', 'run_id': run_id}, ensure_ascii=False)}\n\n"
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "step",
                        "icon": "📡",
                        "msg": "FastAPI 后端已接收请求，正在调用 LangGraph...",
                        "status": "running",
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )
            try:
                if hasattr(graph_service, "as_agent_adapter"):
                    stream = graph_service.as_agent_adapter().generate_and_execute_stream(
                        query=request.query,
                        history_context=request.history_context,
                    )
                else:
                    stream = graph_service.stream_query(query=request.query, session_id=request.session_id)
                for event in stream:
                    if event.get("type") == "result":
                        finish_run(run_id, "completed")
                    elif event.get("type") == "error":
                        finish_run(run_id, "failed", event.get("msg", ""))
                    payload = json.dumps(event, ensure_ascii=False, default=_json_default)
                    yield f"data: {payload}\n\n"
                if runs.get(run_id, {}).get("status") == "running":
                    finish_run(run_id, "completed")
            except Exception as exc:
                finish_run(run_id, "failed", str(exc))
                payload = json.dumps({"type": "error", "msg": str(exc)}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.post("/v1/query/stream")
    def query_stream(request: QueryRequest = Body(...)) -> StreamingResponse:
        """Backward-compatible alias for /v1/query/graph/stream."""
        return query_graph_stream(request)

    @app.post("/v1/sql/execute")
    def execute_sql(request: SQLExecuteRequest) -> Dict[str, Any]:
        state = sql_runner.invoke_sql(
            query=request.query,
            sql=request.sql,
            session_id=request.session_id,
        )
        if state.get("error"):
            raise HTTPException(status_code=400, detail=state["error"])
        return state["result"] or {"rows": [], "sql": request.sql, "from_cache": False}

    @app.post("/v1/insights/stream")
    def insight_stream(request: InsightRequest) -> StreamingResponse:
        if not settings.api_key:
            raise HTTPException(status_code=400, detail="api_key is not configured")

        def text_source():
            rows = request.rows[:10]
            if not rows:
                yield "data: " + json.dumps({"content": "未查询到有效数据，无法生成商业洞察。"}, ensure_ascii=False) + "\n\n"
                return

            columns = list(rows[0].keys()) if rows else []
            preview_lines = [" | ".join(columns)]
            preview_lines.append(" | ".join(["---"] * len(columns)))
            for row in rows:
                preview_lines.append(" | ".join(str(row.get(column, "")) for column in columns))
            data_preview = "\n".join(preview_lines)
            prompt = f"""
你是 DeepInsight 的商业数据分析助手。请直接面向业务用户输出最终答案。

用户问题：{request.query}

查询结果（前10行）：
{data_preview}

输出要求：
1. 只输出最终答案，不要写推理过程、任务复述、提示词分析或"我们被要求"等措辞。
2. 第一句话直接回答用户问题；如果是 TOP/N 排名，请列出名称即可，不要编造数据中没有的字段。
3. 第二句话给出一句业务洞察或建议。
4. 总长度控制在 120 个中文字符以内。
5. 如果数据中的名称包含乱码或特殊字符，按原样保留。
"""
            try:
                client = create_openai_client(
                    api_key=settings.api_key,
                    base_url=settings.api_base,
                    timeout=float(settings.raw.get("llm_timeout", 45.0)),
                )
                stream = client.chat.completions.create(
                    model=settings.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.1,
                )
                for chunk in stream:
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield "data: " + json.dumps({"content": content}, ensure_ascii=False) + "\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"content": f"生成洞察时发生错误: {exc}"}, ensure_ascii=False) + "\n\n"

        return StreamingResponse(text_source(), media_type="text/event-stream")

    @app.post("/v1/index/rebuild")
    def rebuild_index(request: IndexRebuildRequest) -> Dict[str, Any]:
        if not settings.api_key:
            raise HTTPException(status_code=401, detail="api_key is required for index rebuild")

        from rag_engine import IntelRAG

        rag = IntelRAG(
            model_path=settings.model_path,
            db_uris=settings.db_uris,
            kb_paths=settings.kb_paths,
        )
        documents = list(getattr(rag, "documents", []) or [])
        if not hasattr(rag, "_get_embedding"):
            raise HTTPException(status_code=500, detail="embedding function is not available")

        store = ChromaKnowledgeStore(
            persist_path=settings.chroma_path,
            collection_name=CHROMA_COLLECTION,
        )
        count = ChromaRAGIndexer(store).upsert_legacy_documents(
            documents=documents,
            embedding_fn=rag._get_embedding,
            source=request.source,
            namespace=request.namespace or f"{settings.chroma_collection_prefix}:graph",
        )
        return {"indexed_documents": count, "collection": CHROMA_COLLECTION}

    return app


app = create_app()
