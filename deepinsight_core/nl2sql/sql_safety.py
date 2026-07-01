"""SQL safety checks for generated analytics queries.

The checker prefers sqlglot when available and falls back to conservative
string checks. It intentionally returns structured details so API/UI/eval code
can explain why a query was rejected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


FORBIDDEN_KEYWORDS = {
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
}


@dataclass
class SQLSafetyResult:
    safe: bool
    reason: str = ""
    normalized_sql: str = ""
    used_tables: List[str] = field(default_factory=list)
    used_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "reason": self.reason,
            "normalized_sql": self.normalized_sql,
            "used_tables": self.used_tables,
            "used_columns": self.used_columns,
            "warnings": self.warnings,
        }


class SQLSafetyChecker:
    def __init__(
        self,
        dialect: str = "mysql",
        default_limit: Optional[int] = None,
        allow_select_star: bool = False,
        known_tables: Optional[Iterable[str]] = None,
        known_columns: Optional[Dict[str, Iterable[str]]] = None,
    ):
        self.dialect = dialect
        self.default_limit = default_limit
        self.allow_select_star = allow_select_star
        self.known_tables = {t.lower() for t in known_tables or [] if t}
        self.known_columns = {
            table.lower(): {column.lower() for column in columns if column}
            for table, columns in (known_columns or {}).items()
            if table
        }

    @staticmethod
    def normalize(sql: str) -> str:
        return (sql or "").strip().rstrip(";").strip()

    def validate(self, sql: str) -> SQLSafetyResult:
        clean = self.normalize(sql)
        if not clean:
            return SQLSafetyResult(False, "SQL is empty", clean)

        if self._has_multiple_statements(clean):
            return SQLSafetyResult(False, "Multiple SQL statements are not allowed", clean)

        lowered = " ".join(clean.lower().split())
        first = lowered.split(" ", 1)[0]
        if first not in {"select", "with"}:
            return SQLSafetyResult(False, f"Only read-only SELECT/WITH SQL is allowed, got '{first}'", clean)

        for keyword in FORBIDDEN_KEYWORDS:
            if f" {keyword} " in f" {lowered} ":
                return SQLSafetyResult(False, f"Forbidden SQL keyword detected: {keyword}", clean)

        parsed = self._parse(clean)
        if parsed is None:
            used_tables = sorted(self._extract_tables_fallback(clean))
            if self.known_tables:
                missing = [table for table in used_tables if table.lower() not in self.known_tables]
                if missing:
                    return SQLSafetyResult(False, f"Unknown table(s): {', '.join(missing)}", clean, used_tables)
            if not self.allow_select_star and self._uses_select_star_fallback(clean):
                return SQLSafetyResult(False, "SELECT * is not allowed", clean, used_tables)
            column_error = self._validate_known_columns_fallback(clean)
            if column_error:
                return SQLSafetyResult(False, column_error, clean, used_tables)
            return SQLSafetyResult(True, "", self._ensure_limit(clean), used_tables=used_tables, warnings=["sqlglot unavailable or parse failed"])

        used_tables = sorted(self._extract_tables(parsed))
        used_columns = sorted(self._extract_columns(parsed))
        if self.known_tables:
            missing = [table for table in used_tables if table.lower() not in self.known_tables]
            if missing:
                return SQLSafetyResult(False, f"Unknown table(s): {', '.join(missing)}", clean, used_tables, used_columns)

        if not self.allow_select_star and self._uses_select_star(parsed):
            return SQLSafetyResult(False, "SELECT * is not allowed", clean, used_tables, used_columns)
        column_error = self._validate_known_columns(parsed)
        if column_error:
            return SQLSafetyResult(False, column_error, clean, used_tables, used_columns)

        normalized = self._ensure_limit(clean)
        return SQLSafetyResult(True, "", normalized, used_tables, used_columns)

    def _parse(self, sql: str) -> Any:
        try:
            import sqlglot

            expressions = sqlglot.parse(sql, read=self.dialect)
            if len(expressions) != 1:
                return None
            return expressions[0]
        except Exception:
            return None

    @staticmethod
    def _has_multiple_statements(sql: str) -> bool:
        in_single = False
        in_double = False
        for char in sql:
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == ";" and not in_single and not in_double:
                return True
        return False

    @staticmethod
    def _extract_tables(parsed: Any) -> set[str]:
        try:
            from sqlglot import exp

            return {table.name for table in parsed.find_all(exp.Table) if table.name}
        except Exception:
            return set()

    @staticmethod
    def _extract_columns(parsed: Any) -> set[str]:
        try:
            from sqlglot import exp

            return {column.name for column in parsed.find_all(exp.Column) if column.name}
        except Exception:
            return set()

    @staticmethod
    def _uses_select_star(parsed: Any) -> bool:
        try:
            from sqlglot import exp

            for select in parsed.find_all(exp.Select):
                for expression in select.expressions:
                    if isinstance(expression, exp.Star):
                        return True
            return False
        except Exception:
            return False

    def _validate_known_columns(self, parsed: Any) -> str:
        if not self.known_columns:
            return ""
        try:
            from sqlglot import exp

            table_aliases: Dict[str, str] = {}
            for table in parsed.find_all(exp.Table):
                table_name = table.name
                if not table_name:
                    continue
                table_aliases[table_name.lower()] = table_name.lower()
                alias = table.alias
                if alias:
                    table_aliases[alias.lower()] = table_name.lower()

            for column in parsed.find_all(exp.Column):
                column_name = column.name
                if not column_name:
                    continue
                table_ref = column.table
                if table_ref:
                    table_name = table_aliases.get(table_ref.lower(), table_ref.lower())
                    known = self.known_columns.get(table_name)
                    if known is not None and column_name.lower() not in known:
                        return f"Unknown column '{table_ref}.{column_name}' for table '{table_name}'"
                    continue

                if not any(column_name.lower() in columns for columns in self.known_columns.values()):
                    return f"Unknown column '{column_name}'"
        except Exception:
            return ""
        return ""

    def _validate_known_columns_fallback(self, sql: str) -> str:
        if not self.known_columns:
            return ""
        table_aliases: Dict[str, str] = {}
        for match in re.finditer(
            r"\b(?:from|join)\s+`?([A-Za-z_][\w]*)`?(?:\s+(?:as\s+)?`?([A-Za-z_][\w]*)`?)?",
            sql,
            flags=re.IGNORECASE,
        ):
            table_name = match.group(1)
            alias = match.group(2)
            if not table_name:
                continue
            table_key = table_name.lower()
            table_aliases[table_key] = table_key
            if alias and alias.lower() not in {"on", "where", "join", "left", "right", "inner", "outer", "group", "order"}:
                table_aliases[alias.lower()] = table_key

        for match in re.finditer(r"\b`?([A-Za-z_][\w]*)`?\.`?([A-Za-z_][\w]*)`?\b", sql):
            table_ref, column_name = match.group(1), match.group(2)
            table_name = table_aliases.get(table_ref.lower(), table_ref.lower())
            known = self.known_columns.get(table_name)
            if known is not None and column_name.lower() not in known:
                return f"Unknown column '{table_ref}.{column_name}' for table '{table_name}'"
        return ""

    @staticmethod
    def _extract_tables_fallback(sql: str) -> set[str]:
        tables: set[str] = set()
        for match in re.finditer(
            r"\b(?:from|join)\s+`?([A-Za-z_][\w]*)`?",
            sql,
            flags=re.IGNORECASE,
        ):
            table_name = match.group(1)
            if table_name:
                tables.add(table_name)
        return tables

    @staticmethod
    def _uses_select_star_fallback(sql: str) -> bool:
        match = re.search(r"\bselect\b(?P<select_list>.*?)\bfrom\b", sql, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return False
        select_list = match.group("select_list")
        if re.search(r"(^|,)\s*\*\s*(,|$)", select_list):
            return True
        if re.search(r"\b[A-Za-z_][\w]*\s*\.\s*\*", select_list):
            return True
        return False

    def _ensure_limit(self, sql: str) -> str:
        if not self.default_limit:
            return sql
        try:
            import sqlglot
            from sqlglot import exp

            parsed = sqlglot.parse_one(sql, read=self.dialect)
            if parsed.args.get("limit") is None:
                parsed.set("limit", exp.Limit(expression=exp.Literal.number(self.default_limit)))
            return parsed.sql(dialect=self.dialect)
        except Exception:
            lowered = sql.lower()
            if " limit " in f" {lowered} ":
                return sql
            return f"{sql} LIMIT {self.default_limit}"
