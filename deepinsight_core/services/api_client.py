"""HTTP/SSE client used by UI code to call the FastAPI backend."""

from __future__ import annotations

import json
from typing import Any, Dict, Generator, Iterable, List, Optional

import httpx
import pandas as pd


class DeepInsightAPIError(RuntimeError):
    """Raised when the API backend cannot complete a request."""


def _stream_error_text(response: httpx.Response) -> str:
    """Read a streaming response before accessing its body text."""
    response.read()
    return response.text


def parse_sse_events(lines: Iterable[str]) -> Generator[Dict[str, Any], None, None]:
    """Parse a minimal Server-Sent Events stream containing JSON data fields."""
    data_lines: List[str] = []
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                yield json.loads(payload)
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield json.loads("\n".join(data_lines))


class DeepInsightAPIClient:
    """Expose the legacy agent stream protocol over FastAPI SSE."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        endpoint: str = "/v1/query/graph/stream",
        timeout: float = 300.0,
        session_id: str = "default",
    ):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint
        self.timeout = timeout
        self.session_id = session_id
        self.last_retrieval_display: Optional[Dict[str, str]] = None

    def reset_state(self) -> None:
        self.last_retrieval_display = None

    def get_retrieval_display_info(self) -> Optional[Dict[str, str]]:
        return self.last_retrieval_display

    def _normalize_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event_type = event.get("type")
        if event_type == "retrieval_display":
            display = event.get("display")
            if isinstance(display, dict):
                self.last_retrieval_display = display
            return None
        if event_type == "result" and isinstance(event.get("df"), list):
            event = dict(event)
            event["df"] = pd.DataFrame(event["df"])
        return event

    def generate_and_execute_stream(
        self,
        query: str,
        history_context: List[Dict[str, Any]],
        cache_query_key: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        self.reset_state()
        payload = {
            "query": query,
            "session_id": self.session_id,
            "history_context": history_context or [],
        }
        url = f"{self.base_url}{self.endpoint}"

        try:
            with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    raise DeepInsightAPIError(_stream_error_text(response))
                for event in parse_sse_events(response.iter_lines()):
                    normalized = self._normalize_event(event)
                    if normalized is not None:
                        yield normalized
        except httpx.HTTPError as exc:
            raise DeepInsightAPIError(f"FastAPI backend request failed: {exc}") from exc

    def chat_stream(self, query: str, history_context: List[Dict[str, Any]]) -> Generator[str, None, None]:
        yield "当前 API 模式仅支持 Text2SQL 查询。"

    def generate_insight_stream(self, query: str, df: pd.DataFrame) -> Generator[str, None, None]:
        if df is None or df.empty:
            yield "未查询到有效数据，无法生成商业洞察。"
            return

        payload = {
            "query": query,
            "session_id": self.session_id,
            "rows": df.head(10).to_dict(orient="records"),
        }
        url = f"{self.base_url}/v1/insights/stream"

        try:
            with httpx.stream("POST", url, json=payload, timeout=self.timeout) as response:
                if response.status_code >= 400:
                    raise DeepInsightAPIError(_stream_error_text(response))
                for event in parse_sse_events(response.iter_lines()):
                    content = event.get("content")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise DeepInsightAPIError(f"FastAPI insight request failed: {exc}") from exc
