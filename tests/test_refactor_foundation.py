import os
import unittest

import pandas as pd
from sqlalchemy import create_engine, text

from deepinsight_api.main import create_app
from deepinsight_core.config import DeepInsightSettings
from deepinsight_core.graph.events import token_usage_zero_event
from deepinsight_core.graph.generation_runner import SQLGenerationGraphRunner
from deepinsight_core.graph.nodes import (
    classify_sql_error,
    generate_sql_node,
    heal_sql_node,
    retrieve_context_node,
    run_sql_execution_path,
    validate_sql_node,
)
from deepinsight_core.graph.retrieval_runner import RetrievalGraphRunner
from deepinsight_core.graph.sql_runner import SQLGraphRunner
from deepinsight_core.graph.text2sql_runner import Text2SQLGraphRunner
from deepinsight_core.services.rag_service import (
    ChromaRAGBridge,
    ChromaKnowledgeStore,
    ChromaRAGIndexer,
    ChromaRAGRoughSearcher,
    RetrievedContext,
    infer_document_metadata,
    LegacyRAGAdapter,
)
from deepinsight_core.services.api_client import DeepInsightAPIClient, parse_sse_events
from deepinsight_core.services.graph_agent_adapter import GraphAgentAdapter
from deepinsight_core.services.sql_generation_service import SQLGenerationService, extract_sql_from_response
from deepinsight_core.services.sql_service import SQLService
from deepinsight_core.services.text2sql_graph_service import Text2SQLGraphService
from utils import apply_env_overrides, load_config, save_config


class FakeGraphService:
    def stream_query(self, query, session_id="default"):
        yield {"type": "step", "msg": f"graph {query}", "status": "complete", "icon": ""}
        yield {"type": "code_chunk", "content": "SELECT 1"}


class FakeLegacyRAG:
    def __init__(self):
        self.documents = [
            "【表名】orders\n【描述】订单主表，包含订单日期和客户编号",
            "【表名】customers\n【描述】客户维度表，包含客户名称和地区",
        ]
        self.chroma_enabled = False

    def _get_embedding(self, text):
        return fake_embedding(text)

    def retrieve_context(self, query):
        candidates = self._rough_search(query, top_k=1)
        return {"rough_candidates": candidates, "core_tables": [candidates[0]["table_name"]]}

    def _rough_search(self, query, top_k=12):
        return [
            {
                "table_name": "legacy",
                "description": "legacy path",
                "score": 0.0,
                "rank": 1,
                "document": "legacy",
            }
        ]


class FakeRetrievalRAG:
    def retrieve_context(
        self,
        query,
        config=None,
        llm_client=None,
        model_name=None,
        db_engine=None,
        enable_pruning=True,
    ):
        return {
            "database_index": ["orders", "customers"],
            "rough_candidates": [{"table_name": "orders", "rank": 1, "score": 1.0, "document": "doc"}],
            "core_tables": ["orders"],
            "core_table_details": [{"table_name": "orders", "columns": []}],
            "matched_examples": [],
            "matched_terms": [],
            "sample_data": {},
            "metrics": {"total_latency_ms": 1.5},
        }


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 4
    total_tokens = 14


class FakeMessage:
    content = "```sql\nSELECT COUNT(*) AS total_orders FROM orders;\n```"


class FakeChoice:
    message = FakeMessage()


class FakeCompletionResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()


class FakeCompletions:
    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeCompletionResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeLLMClient:
    def __init__(self):
        self.chat = FakeChat()


class SequenceFakeMessage:
    def __init__(self, content):
        self.content = content


class SequenceFakeChoice:
    def __init__(self, content):
        self.message = SequenceFakeMessage(content)


class SequenceFakeResponse:
    def __init__(self, content):
        self.choices = [SequenceFakeChoice(content)]
        self.usage = FakeUsage()


class SequenceFakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        content = self.responses.pop(0)
        return SequenceFakeResponse(content)


class SequenceFakeChat:
    def __init__(self, responses):
        self.completions = SequenceFakeCompletions(responses)


class SequenceFakeLLMClient:
    def __init__(self, responses):
        self.chat = SequenceFakeChat(responses)


class FakeInsightDelta:
    reasoning_content = "我们被要求输出分析过程。"
    content = "库存积压TOP5为A、B。建议优先促销。"


class FakeInsightChunk:
    choices = [type("Choice", (), {"delta": FakeInsightDelta()})()]


class FakeInsightCompletions:
    def create(self, **kwargs):
        return iter([FakeInsightChunk()])


class FakeInsightClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeInsightCompletions()})()


def fake_embedding(text):
    lowered = text.lower()
    if "order" in lowered or "订单" in text:
        return [1.0, 0.0, 0.0]
    if "customer" in lowered or "客户" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class RefactorFoundationTests(unittest.TestCase):
    def test_legacy_config_defaults_include_backend_switches(self):
        config = load_config()
        self.assertIn("use_chroma_rag", config)
        self.assertIn(config.get("query_runtime"), {"api", "auto", "local"})
        self.assertIn("api_server_url", config)

    def test_env_overrides_can_enable_graph_backend(self):
        old_chroma = os.environ.get("DEEPINSIGHT_USE_CHROMA_RAG")
        try:
            os.environ["DEEPINSIGHT_USE_CHROMA_RAG"] = "true"
            config = apply_env_overrides({"use_chroma_rag": False})
            self.assertTrue(config["use_chroma_rag"])
        finally:
            if old_chroma is None:
                os.environ.pop("DEEPINSIGHT_USE_CHROMA_RAG", None)
            else:
                os.environ["DEEPINSIGHT_USE_CHROMA_RAG"] = old_chroma

    def test_env_overrides_can_select_api_runtime(self):
        old_runtime = os.environ.get("DEEPINSIGHT_QUERY_RUNTIME")
        old_url = os.environ.get("DEEPINSIGHT_API_SERVER_URL")
        try:
            os.environ["DEEPINSIGHT_QUERY_RUNTIME"] = "api"
            os.environ["DEEPINSIGHT_API_SERVER_URL"] = "http://localhost:9000"
            config = apply_env_overrides({})
            self.assertEqual(config["query_runtime"], "api")
            self.assertEqual(config["api_server_url"], "http://localhost:9000")
        finally:
            if old_runtime is None:
                os.environ.pop("DEEPINSIGHT_QUERY_RUNTIME", None)
            else:
                os.environ["DEEPINSIGHT_QUERY_RUNTIME"] = old_runtime
            if old_url is None:
                os.environ.pop("DEEPINSIGHT_API_SERVER_URL", None)
            else:
                os.environ["DEEPINSIGHT_API_SERVER_URL"] = old_url

    def test_save_config_does_not_persist_env_secrets(self):
        old_api_key = os.environ.get("DEEPINSIGHT_API_KEY")
        old_rec_key = os.environ.get("DEEPINSIGHT_RECOMMENDATION_API_KEY")
        import utils

        old_config_file = utils.CONFIG_FILE
        try:
            os.environ["DEEPINSIGHT_API_KEY"] = "secret-api"
            os.environ["DEEPINSIGHT_RECOMMENDATION_API_KEY"] = "secret-rec"
            tmp_dir = "tests_runtime"
            os.makedirs(tmp_dir, exist_ok=True)
            utils.CONFIG_FILE = os.path.join(tmp_dir, "config_secret_guard.json")
            save_config({"api_key": "secret-api", "recommendation_api_key": "secret-rec"})

            with open(utils.CONFIG_FILE, "r", encoding="utf-8") as handle:
                saved = handle.read()
            self.assertNotIn("secret-api", saved)
            self.assertNotIn("secret-rec", saved)
        finally:
            utils.CONFIG_FILE = old_config_file
            if old_api_key is None:
                os.environ.pop("DEEPINSIGHT_API_KEY", None)
            else:
                os.environ["DEEPINSIGHT_API_KEY"] = old_api_key
            if old_rec_key is None:
                os.environ.pop("DEEPINSIGHT_RECOMMENDATION_API_KEY", None)
            else:
                os.environ["DEEPINSIGHT_RECOMMENDATION_API_KEY"] = old_rec_key

    def test_settings_from_legacy_config(self):
        settings = DeepInsightSettings.from_mapping(
            {
                "api_key": "k",
                "api_base": "https://example.com",
                "model_name": "model",
                "db_uris": ["sqlite:///example.db"],
                "kb_paths_list": ["data/schema.json"],
                "max_retries": "2",
            }
        )
        self.assertEqual(settings.api_key, "k")
        self.assertEqual(settings.api_base, "https://example.com")
        self.assertEqual(settings.first_db_uri, "sqlite:///example.db")
        self.assertEqual(settings.kb_paths, ["data/schema.json"])
        self.assertEqual(settings.max_retries, 2)

    def test_sql_service_rejects_write_sql(self):
        valid, error = SQLService.validate_read_only("DELETE FROM orders")
        self.assertFalse(valid)
        self.assertIn("Only read-only", error)

        valid, error = SQLService.validate_read_only("SELECT * FROM orders; DROP TABLE orders")
        self.assertFalse(valid)
        self.assertIn("Forbidden", error)

    def test_sql_service_executes_read_only_query(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE metrics (name TEXT, value INTEGER)"))
            conn.execute(text("INSERT INTO metrics VALUES ('sales', 42)"))

        service = SQLService("sqlite:///:memory:", engine=engine)
        result = service.execute("SELECT name, value FROM metrics")

        self.assertTrue(result.ok)
        self.assertEqual(result.to_records(), [{"name": "sales", "value": 42}])

    def test_retrieved_context_round_trip(self):
        payload = {"database_index": ["orders"], "core_tables": ["orders"], "metrics": {"latency": 1}}
        context = RetrievedContext.from_mapping(payload)
        self.assertEqual(context.database_index, ["orders"])
        self.assertEqual(context.core_tables, ["orders"])
        self.assertEqual(context.to_legacy_dict()["metrics"], {"latency": 1})

    def test_chroma_metadata_inference_for_table_document(self):
        metadata = infer_document_metadata("【表名】orders\n【描述】订单主表", position=3)
        self.assertEqual(metadata["doc_type"], "table_schema")
        self.assertEqual(metadata["table_name"], "orders")
        self.assertEqual(metadata["position"], 3)

    def test_chroma_rough_search_returns_legacy_candidate_shape(self):
        import chromadb

        client = chromadb.EphemeralClient()
        store = ChromaKnowledgeStore(
            persist_path=None,
            collection_name="test_deepinsight_tables",
            client=client,
        )
        indexer = ChromaRAGIndexer(store)
        searcher = ChromaRAGRoughSearcher(store)

        documents = [
            "【表名】orders\n【描述】订单主表，包含订单日期和客户编号",
            "【表名】customers\n【描述】客户维度表，包含客户名称和地区",
            "【术语】销售额\n解释: 订单金额汇总",
        ]
        inserted = indexer.upsert_legacy_documents(documents, fake_embedding, namespace="unit")
        candidates = searcher.rough_search("查询订单趋势", fake_embedding, top_k=2)

        self.assertEqual(inserted, 3)
        self.assertEqual(store.count(), 3)
        self.assertEqual(candidates[0]["table_name"], "orders")
        self.assertEqual(candidates[0]["rank"], 1)
        self.assertIn("description", candidates[0])
        self.assertIn("document", candidates[0])

    def test_chroma_bridge_replaces_legacy_rough_search_opt_in(self):
        import chromadb

        legacy_rag = FakeLegacyRAG()
        store = ChromaKnowledgeStore(
            persist_path=None,
            collection_name="test_deepinsight_bridge",
            client=chromadb.EphemeralClient(),
        )
        bridge = ChromaRAGBridge(store, namespace="bridge")

        inserted = bridge.attach(legacy_rag)
        payload = legacy_rag.retrieve_context("订单分析")

        self.assertEqual(inserted, 2)
        self.assertTrue(legacy_rag.chroma_enabled)
        self.assertEqual(payload["core_tables"], ["orders"])
        self.assertEqual(payload["rough_candidates"][0]["table_name"], "orders")

    def test_chroma_bridge_guards_against_duplicate_attach(self):
        import chromadb

        legacy_rag = FakeLegacyRAG()
        store = ChromaKnowledgeStore(
            persist_path=None,
            collection_name="test_deepinsight_bridge_dup",
            client=chromadb.EphemeralClient(),
        )
        bridge = ChromaRAGBridge(store, namespace="dup")
        first = bridge.attach(legacy_rag)
        second = bridge.attach(legacy_rag)

        self.assertEqual(first, 2)
        self.assertEqual(second, 2)  # returns existing count, doesn't re-insert

    def test_token_usage_zero_event_keeps_legacy_shape(self):
        event = token_usage_zero_event()
        self.assertEqual(event["type"], "cumulative_token_usage")
        self.assertEqual(event["usage"]["total_tokens"], 0)
        self.assertEqual(event["usage"]["breakdown"], [])

    def test_validate_sql_node_rejects_write_sql(self):
        state = validate_sql_node({"query": "bad", "sql": "DROP TABLE orders", "events": []})
        self.assertIn("error", state)
        self.assertEqual(state["events"][0]["type"], "error")

    def test_sql_execution_path_emits_legacy_result_event(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE metrics (name TEXT, value INTEGER)"))
            conn.execute(text("INSERT INTO metrics VALUES ('profit', 7)"))

        service = SQLService("sqlite:///:memory:", engine=engine)
        state = run_sql_execution_path(
            {"query": "profit", "sql": "SELECT name, value FROM metrics", "events": []},
            service,
        )

        self.assertEqual(state["error"] if "error" in state else "", "")
        self.assertEqual(state["result"]["rows"], [{"name": "profit", "value": 7}])
        self.assertEqual([event["type"] for event in state["events"]], ["step", "step", "result"])
        self.assertEqual(state["events"][-1]["sql"], "SELECT name, value FROM metrics")

    def test_sql_graph_runner_invokes_composable_path(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE metrics (name TEXT, value INTEGER)"))
            conn.execute(text("INSERT INTO metrics VALUES ('margin', 9)"))

        runner = SQLGraphRunner(SQLService("sqlite:///:memory:", engine=engine))
        state = runner.invoke_sql("margin", "SELECT value FROM metrics WHERE name = 'margin'")

        self.assertEqual(state["result"]["rows"], [{"value": 9}])

    def test_sql_graph_runner_can_build_langgraph(self):
        runner = SQLGraphRunner(SQLService("sqlite:///:memory:", engine=create_engine("sqlite:///:memory:")))
        graph = runner.build_langgraph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_retrieve_context_node_populates_state(self):
        adapter = LegacyRAGAdapter(FakeRetrievalRAG())
        state = retrieve_context_node({"query": "orders", "events": []}, adapter)
        self.assertEqual(state["retrieval_result"]["core_tables"], ["orders"])
        self.assertEqual(state["events"][0]["type"], "step")

    def test_retrieval_graph_runner_invokes_and_builds_langgraph(self):
        runner = RetrievalGraphRunner(LegacyRAGAdapter(FakeRetrievalRAG()), enable_pruning=False)
        state = runner.invoke_query("orders")
        graph = runner.build_langgraph()
        self.assertEqual(state["retrieval_result"]["database_index"], ["orders", "customers"])
        self.assertTrue(hasattr(graph, "invoke"))

    def test_extract_sql_from_markdown_response(self):
        sql = extract_sql_from_response("```sql\nSELECT * FROM orders;\n```")
        self.assertEqual(sql, "SELECT * FROM orders")

    def test_generate_sql_node_emits_legacy_code_events(self):
        service = SQLGenerationService(FakeLLMClient(), model_name="fake-model")
        state = generate_sql_node(
            {
                "query": "how many orders",
                "retrieval_result": {"database_index": ["orders"]},
                "events": [],
            },
            service,
        )
        self.assertEqual(state["sql"], "SELECT COUNT(*) AS total_orders FROM orders")
        self.assertEqual(
            [event["type"] for event in state["events"]],
            ["step", "code_start", "code_chunk", "token_usage"],
        )
        self.assertEqual(state["events"][-1]["usage"]["total_tokens"], 14)

    def test_generation_runner_can_build_langgraph(self):
        runner = SQLGenerationGraphRunner(SQLGenerationService(FakeLLMClient(), model_name="fake-model"))
        graph = runner.build_langgraph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_text2sql_runner_executes_end_to_end_with_fakes(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER)"))
            conn.execute(text("INSERT INTO orders VALUES (1)"))
            conn.execute(text("INSERT INTO orders VALUES (2)"))

        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(FakeLLMClient(), model_name="fake-model"),
            sql_service=SQLService("sqlite:///:memory:", engine=engine),
            enable_pruning=False,
        )
        state = runner.invoke_query("how many orders")

        self.assertEqual(state["result"]["rows"], [{"total_orders": 2}])
        self.assertEqual(state["retrieval_result"]["core_tables"], ["orders"])
        self.assertIn("SELECT COUNT(*) AS total_orders FROM orders", state["sql"])

    def test_text2sql_runner_can_build_langgraph(self):
        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(FakeLLMClient(), model_name="fake-model"),
            sql_service=SQLService("sqlite:///:memory:", engine=create_engine("sqlite:///:memory:")),
        )
        graph = runner.build_langgraph()
        self.assertTrue(hasattr(graph, "invoke"))

    def test_text2sql_compiled_langgraph_includes_healing_loop(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER)"))
            conn.execute(text("INSERT INTO orders VALUES (1)"))

        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(
                SequenceFakeLLMClient(
                    [
                        "SELECT missing_col FROM orders",
                        "SELECT COUNT(*) AS total_orders FROM orders",
                    ]
                ),
                model_name="fake-model",
            ),
            sql_service=SQLService("sqlite:///:memory:", engine=engine),
            enable_pruning=False,
            max_healing_attempts=1,
        )
        graph = runner.build_langgraph()
        state = graph.invoke({"query": "how many orders", "events": []})

        self.assertEqual(state["error"], "")
        self.assertEqual(state["healing_attempts"], 1)
        self.assertEqual(state["result"]["rows"], [{"total_orders": 1}])

    def test_classify_sql_error_security(self):
        self.assertEqual(classify_sql_error("Forbidden SQL keyword detected: drop"), "security")

    def test_heal_sql_node_repairs_non_security_error(self):
        service = SQLGenerationService(
            SequenceFakeLLMClient(["SELECT COUNT(*) AS total_orders FROM orders"]),
            model_name="fake-model",
        )
        state = heal_sql_node(
            {
                "query": "how many orders",
                "sql": "SELECT missing_col FROM orders",
                "error": "no such column: missing_col",
                "retrieval_result": {"database_index": ["orders"]},
                "events": [],
            },
            service,
        )
        self.assertEqual(state["error"], "")
        self.assertEqual(state["error_category"], "unknown_column")
        self.assertEqual(state["sql"], "SELECT COUNT(*) AS total_orders FROM orders")

    def test_text2sql_runner_applies_one_healing_attempt(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER)"))
            conn.execute(text("INSERT INTO orders VALUES (1)"))
            conn.execute(text("INSERT INTO orders VALUES (2)"))

        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(
                SequenceFakeLLMClient(
                    [
                        "SELECT missing_col FROM orders",
                        "SELECT COUNT(*) AS total_orders FROM orders",
                    ]
                ),
                model_name="fake-model",
            ),
            sql_service=SQLService("sqlite:///:memory:", engine=engine),
            enable_pruning=False,
            max_healing_attempts=1,
        )
        state = runner.invoke_query("how many orders")

        self.assertEqual(state["error"], "")
        self.assertEqual(state["error_category"], "unknown_column")
        self.assertEqual(state["result"]["rows"], [{"total_orders": 2}])

    def test_fastapi_health_and_missing_key_guard(self):
        from fastapi.testclient import TestClient

        app = create_app(DeepInsightSettings.from_mapping({"api_key": ""}))
        client = TestClient(app)

        health = client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        response = client.post("/v1/query/graph/stream", json={"query": "hello"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("api_key", response.json()["detail"])

    def test_fastapi_sql_execute_endpoint_uses_graph_sql_runner(self):
        from fastapi.testclient import TestClient

        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE metrics (value INTEGER)"))
            conn.execute(text("INSERT INTO metrics VALUES (5)"))

        app = create_app(DeepInsightSettings.from_mapping({"db_uris": []}))
        client = TestClient(app)
        response = client.post("/v1/sql/execute", json={"sql": "DROP TABLE metrics"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Only read-only", response.json()["detail"])

    def test_fastapi_graph_query_stream_requires_api_key(self):
        from fastapi.testclient import TestClient

        app = create_app(DeepInsightSettings.from_mapping({"api_key": ""}))
        client = TestClient(app)
        response = client.post("/v1/query/graph/stream", json={"query": "hello"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("api_key", response.json()["detail"])

    def test_fastapi_graph_query_stream_can_emit_sse_with_override(self):
        from fastapi.testclient import TestClient

        app = create_app(
            DeepInsightSettings.from_mapping({"api_key": "fake"}),
            graph_service_override=FakeGraphService(),
        )
        client = TestClient(app)
        response = client.post("/v1/query/graph/stream", json={"query": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("data:", response.text)
        self.assertIn('"type": "run"', response.text)
        self.assertIn("SELECT 1", response.text)

    def test_fastapi_legacy_query_stream_alias_works(self):
        """Verify /v1/query/stream is an alias for /v1/query/graph/stream."""
        from fastapi.testclient import TestClient

        app = create_app(
            DeepInsightSettings.from_mapping({"api_key": "fake"}),
            graph_service_override=FakeGraphService(),
        )
        client = TestClient(app)
        response = client.post("/v1/query/stream", json={"query": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertIn("SELECT 1", response.text)

    def test_fastapi_session_and_run_status_endpoints(self):
        from fastapi.testclient import TestClient

        app = create_app(
            DeepInsightSettings.from_mapping({"api_key": "fake"}),
            graph_service_override=FakeGraphService(),
        )
        client = TestClient(app)

        session_response = client.post("/v1/sessions", json={"title": "demo"})
        self.assertEqual(session_response.status_code, 200)
        session_id = session_response.json()["session_id"]

        run_id = "test-run-1"
        response = client.post(
            "/v1/query/graph/stream",
            json={"query": "hello", "session_id": session_id, "run_id": run_id},
        )
        self.assertEqual(response.status_code, 200)

        run_response = client.get(f"/v1/runs/{run_id}")
        self.assertEqual(run_response.status_code, 200)
        self.assertEqual(run_response.json()["status"], "completed")

    def test_sse_parser_and_api_client_restore_dataframe(self):
        lines = [
            'data: {"type": "retrieval_display", "display": {"core_tables_display": "orders"}}',
            "",
            'data: {"type": "result", "df": [{"answer": 1}], "sql": "SELECT 1"}',
            "",
        ]
        client = DeepInsightAPIClient()

        events = []
        for event in parse_sse_events(lines):
            normalized = client._normalize_event(event)
            if normalized is not None:
                events.append(normalized)

        self.assertEqual(client.get_retrieval_display_info()["core_tables_display"], "orders")
        self.assertEqual(events[0]["df"].to_dict("records"), [{"answer": 1}])

    def test_api_client_exposes_insight_stream_compat_method(self):
        client = DeepInsightAPIClient()
        chunks = list(client.generate_insight_stream("hello", pd.DataFrame()))
        self.assertTrue(chunks)
        self.assertIn("无法生成商业洞察", chunks[0])

    def test_fastapi_insight_stream_handles_empty_rows(self):
        from fastapi.testclient import TestClient

        app = create_app(DeepInsightSettings.from_mapping({"api_key": "fake"}))
        client = TestClient(app)
        response = client.post("/v1/insights/stream", json={"query": "hello", "rows": []})

        self.assertEqual(response.status_code, 200)
        self.assertIn("data:", response.text)
        self.assertIn("无法生成商业洞察", response.text)

    def test_fastapi_insight_stream_does_not_emit_reasoning_content(self):
        from fastapi.testclient import TestClient
        import deepinsight_api.main as api_main

        original_create_openai_client = api_main.create_openai_client
        try:
            api_main.create_openai_client = lambda **_kwargs: FakeInsightClient()
            app = create_app(DeepInsightSettings.from_mapping({"api_key": "fake"}))
            client = TestClient(app)

            response = client.post(
                "/v1/insights/stream",
                json={"query": "库存积压最严重的TOP5产品是？", "rows": [{"product": "A"}]},
            )
        finally:
            api_main.create_openai_client = original_create_openai_client

        self.assertEqual(response.status_code, 200)
        self.assertIn("库存积压TOP5", response.text)
        self.assertNotIn("我们被要求", response.text)

    def test_graph_agent_adapter_exposes_legacy_stream_protocol(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER)"))
            conn.execute(text("INSERT INTO orders VALUES (1)"))
            conn.execute(text("INSERT INTO orders VALUES (2)"))

        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(FakeLLMClient(), model_name="fake-model"),
            sql_service=SQLService("sqlite:///:memory:", engine=engine),
            enable_pruning=False,
        )
        adapter = GraphAgentAdapter(runner)
        events = list(adapter.generate_and_execute_stream("how many orders", []))

        self.assertIn("result", [event["type"] for event in events])
        result = next(event for event in events if event["type"] == "result")
        self.assertEqual(result["df"].to_dict("records"), [{"total_orders": 2}])
        cumulative = next(event for event in events if event["type"] == "cumulative_token_usage")
        self.assertEqual(cumulative["type"], "cumulative_token_usage")
        self.assertGreaterEqual(cumulative["usage"]["total_tokens"], 14)
        self.assertEqual(events[-1]["type"], "result")
        display = adapter.get_retrieval_display_info()
        self.assertIsNotNone(display)
        self.assertIn("orders", display["core_tables_display"])

    def test_text2sql_graph_service_can_export_agent_adapter(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE orders (id INTEGER)"))
            conn.execute(text("INSERT INTO orders VALUES (1)"))

        runner = Text2SQLGraphRunner(
            rag_adapter=LegacyRAGAdapter(FakeRetrievalRAG()),
            generation_service=SQLGenerationService(FakeLLMClient(), model_name="fake-model"),
            sql_service=SQLService("sqlite:///:memory:", engine=engine),
            enable_pruning=False,
        )
        service = Text2SQLGraphService.__new__(Text2SQLGraphService)
        service._runner = runner

        adapter = service.as_agent_adapter()
        events = list(adapter.generate_and_execute_stream("how many orders", []))

        self.assertIn("result", [event["type"] for event in events])


if __name__ == "__main__":
    unittest.main()
