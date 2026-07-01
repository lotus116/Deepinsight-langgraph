"""Legacy Agent-compatible adapter for graph runners."""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional

from deepinsight_core.graph.text2sql_runner import Text2SQLGraphRunner


class GraphAgentAdapter:
    """Expose `generate_and_execute_stream()` for existing UI/eval consumers."""

    def __init__(self, runner: Text2SQLGraphRunner):
        self.runner = runner
        self.last_retrieval_result: Optional[Dict[str, Any]] = None

    def reset_state(self) -> None:
        self.last_retrieval_result = None

    def get_retrieval_display_info(self) -> Optional[Dict[str, str]]:
        if not self.last_retrieval_result:
            return None
        core_tables = self.last_retrieval_result.get("core_tables", [])
        rough = self.last_retrieval_result.get("rough_candidates", [])
        return {
            "rough_candidates_display": "\n".join(
                f"{item.get('rank', idx + 1)}. {item.get('table_name', '')}"
                for idx, item in enumerate(rough)
            ),
            "core_tables_display": ", ".join(core_tables),
            "matched_terms_display": "\n".join(
                f"- {item.get('term', '')}: {item.get('explanation', '')}"
                for item in self.last_retrieval_result.get("matched_terms", [])
            ) or "无匹配术语",
            "matched_examples_display": "\n".join(
                f"- {item.get('query', '')}"
                for item in self.last_retrieval_result.get("matched_examples", [])
            ) or "无匹配示例",
            "metrics_display": str(self.last_retrieval_result.get("metrics", {})),
        }

    def generate_and_execute_stream(
        self,
        query: str,
        history_context: List[Dict[str, Any]],
        cache_query_key: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        yield {
            "type": "step",
            "icon": "🧭",
            "msg": "正在进入 LangGraph Text2SQL 流程...",
            "status": "running",
        }
        cumulative = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "breakdown": [],
        }

        current_call_type = "generator"
        terminal_event = None
        if hasattr(self.runner, "stream_query"):
            event_stream = self.runner.stream_query(query=query)
        else:
            event_stream = self.runner.invoke_query(query=query).get("events", [])

        for event in event_stream:
            if event.get("type") == "_retrieval_result":
                self.last_retrieval_result = event.get("retrieval_result")
                display_info = self.get_retrieval_display_info()
                if display_info:
                    yield {"type": "retrieval_display", "display": display_info}

                selector_usage = (self.last_retrieval_result or {}).get("metrics", {}).get("selector_token_usage")
                if selector_usage:
                    selector_payload = {
                        "call_type": "selector",
                        "prompt_tokens": selector_usage.get("prompt_tokens", 0),
                        "completion_tokens": selector_usage.get("completion_tokens", 0),
                        "total_tokens": selector_usage.get("total_tokens", 0),
                    }
                    cumulative["breakdown"].append(selector_payload)
                    cumulative["prompt_tokens"] += selector_payload["prompt_tokens"]
                    cumulative["completion_tokens"] += selector_payload["completion_tokens"]
                    cumulative["total_tokens"] += selector_payload["total_tokens"]
                    cumulative["call_count"] += 1
                continue

            if event.get("type") == "token_usage":
                usage = event.get("usage", {})
                payload = {
                    "call_type": current_call_type,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                cumulative["breakdown"].append(payload)
                cumulative["prompt_tokens"] += payload["prompt_tokens"]
                cumulative["completion_tokens"] += payload["completion_tokens"]
                cumulative["total_tokens"] += payload["total_tokens"]
                cumulative["call_count"] += 1
            elif event.get("type") == "code_start" and "repair" in event.get("label", "").lower():
                current_call_type = "retry"
            elif event.get("type") in {"result", "error"}:
                terminal_event = event
                continue
            yield event

        yield {"type": "cumulative_token_usage", "usage": cumulative}
        if terminal_event is not None:
            yield terminal_event

    def chat_stream(self, query: str, history_context: List[Dict[str, Any]]) -> Generator[str, None, None]:
        yield "GraphAgentAdapter only supports Text2SQL queries."

    def generate_insight_stream(self, query: str, df: Any) -> Generator[str, None, None]:
        if df is None or getattr(df, "empty", False):
            yield "未查询到有效数据，无法生成商业洞察。"
            return
        yield "查询已完成，业务洞察生成请使用 FastAPI API 模式或 legacy agent。"
