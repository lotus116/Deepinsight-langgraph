"""Read-only SQL execution service.

This service is intentionally stricter than the legacy ``execute_sql`` method:
it validates the statement before opening a connection and returns structured
results that FastAPI and LangGraph nodes can pass around without depending on
Streamlit-specific event dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


READ_ONLY_PREFIXES = ("select", "with", "show", "describe", "desc", "explain")
FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "replace",
    "merge",
    "grant",
    "revoke",
    "call",
    "exec",
    "execute",
)


@dataclass
class SQLExecutionResult:
    dataframe: Optional[pd.DataFrame]
    error: str = ""
    sql: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.dataframe is not None

    def to_records(self) -> List[Dict[str, Any]]:
        if self.dataframe is None:
            return []
        return self.dataframe.to_dict("records")


class SQLService:
    def __init__(self, db_uri: str, engine: Optional[Engine] = None):
        self.db_uri = db_uri
        self.engine = engine or (create_engine(db_uri) if db_uri else None)

    @staticmethod
    def normalize_sql(sql: str) -> str:
        return (sql or "").strip().rstrip(";").strip()

    @classmethod
    def validate_read_only(cls, sql: str) -> Tuple[bool, str]:
        clean_sql = cls.normalize_sql(sql)
        if not clean_sql:
            return False, "SQL is empty"

        lowered = " ".join(clean_sql.lower().split())
        first_token = lowered.split(" ", 1)[0]
        if first_token not in READ_ONLY_PREFIXES:
            return False, f"Only read-only SQL is allowed, got '{first_token}'"

        padded = f" {lowered} "
        for keyword in FORBIDDEN_KEYWORDS:
            if f" {keyword} " in padded:
                return False, f"Forbidden SQL keyword detected: {keyword}"

        return True, ""

    def execute(self, sql: str) -> SQLExecutionResult:
        clean_sql = self.normalize_sql(sql)
        is_valid, error = self.validate_read_only(clean_sql)
        if not is_valid:
            return SQLExecutionResult(dataframe=None, error=error, sql=clean_sql)

        if self.engine is None:
            return SQLExecutionResult(dataframe=None, error="Database engine is not configured", sql=clean_sql)

        try:
            with self.engine.connect() as conn:
                dataframe = pd.read_sql_query(text(clean_sql), conn)
            return SQLExecutionResult(dataframe=dataframe, sql=clean_sql)
        except Exception as exc:
            return SQLExecutionResult(dataframe=None, error=str(exc), sql=clean_sql)
