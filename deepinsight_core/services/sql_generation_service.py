"""SQL generation service for graph nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SQLGenerationResult:
    sql: str
    raw_response: str = ""
    error: str = ""
    token_usage: Dict[str, int] | None = None

    @property
    def ok(self) -> bool:
        return bool(self.sql) and not self.error


def extract_sql_from_response(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""

    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(("sql:", "sql\uff1a")):
            stripped = stripped.split(":", 1)[-1].strip()
        lines.append(stripped)

    sql = "\n".join(lines).strip().rstrip(";").strip()
    return sql


class SQLGenerationService:
    def __init__(self, llm_client: Any, model_name: str, temperature: float = 0.0):
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature

    def build_prompt(self, query: str, retrieval_result: Optional[Dict[str, Any]] = None) -> str:
        retrieval_result = retrieval_result or {}
        database_index = ", ".join(retrieval_result.get("database_index", []))
        table_details = retrieval_result.get("core_table_details", [])
        terms = retrieval_result.get("matched_terms", [])
        examples = retrieval_result.get("matched_examples", [])

        sections: List[str] = [
            "You are a careful Text-to-SQL engineer.",
            "Generate one read-only SQL query that answers the user question.",
            "Return SQL only. Do not include markdown or explanation.",
            f"User question: {query}",
        ]
        if database_index:
            sections.append(f"Available tables: {database_index}")
        if table_details:
            schema_text = "\n".join(
                detail.get("document", str(detail)) for detail in table_details
            )
            sections.append(f"Relevant schema:\n{schema_text}")
        if terms:
            sections.append(f"Matched business terms:\n{terms}")
        if examples:
            sections.append(f"Relevant examples:\n{examples}")
        return "\n\n".join(sections)

    def build_repair_prompt(
        self,
        query: str,
        previous_sql: str,
        error: str,
        retrieval_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        base_prompt = self.build_prompt(query=query, retrieval_result=retrieval_result)
        return "\n\n".join(
            [
                base_prompt,
                "The previous SQL failed. Repair it.",
                f"Previous SQL:\n{previous_sql}",
                f"Database error:\n{error}",
                "Return the repaired read-only SQL only.",
            ]
        )

    def generate(self, query: str, retrieval_result: Optional[Dict[str, Any]] = None) -> SQLGenerationResult:
        return self._complete(self.build_prompt(query=query, retrieval_result=retrieval_result))

    def repair(
        self,
        query: str,
        previous_sql: str,
        error: str,
        retrieval_result: Optional[Dict[str, Any]] = None,
    ) -> SQLGenerationResult:
        return self._complete(
            self.build_repair_prompt(
                query=query,
                previous_sql=previous_sql,
                error=error,
                retrieval_result=retrieval_result,
            )
        )

    def _complete(self, prompt: str) -> SQLGenerationResult:
        if self.llm_client is None:
            return SQLGenerationResult(sql="", error="LLM client is not configured")
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            raw = response.choices[0].message.content.strip()
            usage = None
            if getattr(response, "usage", None):
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
            return SQLGenerationResult(
                sql=extract_sql_from_response(raw),
                raw_response=raw,
                token_usage=usage,
            )
        except Exception as exc:
            return SQLGenerationResult(sql="", error=str(exc))
