"""Typed payload helpers for the NL2SQL agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentTraceStep:
    node: str
    status: str
    summary: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "status": self.status,
            "summary": self.summary,
            "latency_ms": round(float(self.latency_ms), 2),
            "details": self.details,
        }


@dataclass
class QueryUnderstanding:
    intent: str = "data_query"
    metrics: List[str] = field(default_factory=list)
    dimensions: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    time_range: Optional[str] = None
    limit: Optional[int] = None
    ambiguities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "filters": self.filters,
            "time_range": self.time_range,
            "limit": self.limit,
            "ambiguities": self.ambiguities,
        }


@dataclass
class SQLPlan:
    required_tables: List[str] = field(default_factory=list)
    joins: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_tables": self.required_tables,
            "joins": self.joins,
            "metrics": self.metrics,
            "filters": self.filters,
            "group_by": self.group_by,
            "order_by": self.order_by,
            "limit": self.limit,
            "assumptions": self.assumptions,
        }
