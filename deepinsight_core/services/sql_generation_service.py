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
        sql_plan = retrieval_result.get("sql_plan") or retrieval_result.get("plan") or {}

        sections: List[str] = [
            "You are a careful Text-to-SQL engineer.",
            "Generate one read-only MySQL-compatible SQL query that answers the user question.",
            "Return SQL only. Do not include markdown or explanation.",
            "Use only the provided tables and columns.",
            "Do not invent table names or column names.",
            "Respect exact column names from the schema. For Northwind, products uses Price; orderdetails uses UnitPrice.",
            "If the user asks for profit, margin, or cost but no cost column exists, do not invent cost fields. Use available sales/discount metrics or return a safe query over available fields.",
            "Do not use SELECT *.",
            "Do not generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, CALL, EXEC, or multiple statements.",
            f"User question: {query}",
        ]
        if database_index:
            sections.append(f"Available tables: {database_index}")
        if table_details:
            schema_text = "\n".join(
                detail.get("document", str(detail)) for detail in table_details
            )
            sections.append(f"Relevant schema:\n{schema_text}")
            sections.append(
                "Allowed columns by table:\n"
                + self._format_allowed_columns(
                    table_details,
                    retrieval_result.get("schema_catalog") or {},
                )
            )
        if sql_plan:
            sections.append(f"SQL plan:\n{sql_plan}")
        if terms:
            sections.append(f"Matched business terms:\n{terms}")
        if examples:
            sections.append(f"Relevant examples:\n{examples}")
        return "\n\n".join(sections)

    @staticmethod
    def _format_allowed_columns(
        table_details: List[Dict[str, Any]],
        schema_catalog: Optional[Dict[str, Any]] = None,
    ) -> str:
        lines = []
        seen_tables = set()
        for detail in table_details:
            table_name = detail.get("table_name", "")
            columns = []
            for column in detail.get("columns", []) or []:
                if isinstance(column, dict):
                    name = column.get("col") or column.get("name")
                    col_type = column.get("type")
                    columns.append(f"{name} ({col_type})" if name and col_type else name)
                else:
                    columns.append(str(column))
            if table_name and columns:
                seen_tables.add(table_name.lower())
                lines.append(f"- {table_name}: {', '.join(col for col in columns if col)}")
        for table_name, columns in (schema_catalog or {}).items():
            if not table_name or table_name.lower() in seen_tables:
                continue
            if columns:
                lines.append(f"- {table_name}: {', '.join(str(column) for column in columns if column)}")
        return "\n".join(lines) if lines else "No column list available."

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
