"""Event helpers that preserve the legacy Streamlit protocol."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import pandas as pd


def step_event(message: str, status: str = "running", icon: str = "") -> Dict[str, Any]:
    return {"type": "step", "icon": icon, "msg": message, "status": status}


def error_event(message: str) -> Dict[str, Any]:
    return {"type": "error", "msg": message}


def code_start_event(label: str = "Generated SQL") -> Dict[str, Any]:
    return {"type": "code_start", "label": label}


def code_chunk_event(content: str) -> Dict[str, Any]:
    return {"type": "code_chunk", "content": content}


def token_usage_event(usage: Dict[str, int]) -> Dict[str, Any]:
    return {"type": "token_usage", "usage": usage}


def result_event(sql: str, dataframe: pd.DataFrame, from_cache: bool = False) -> Dict[str, Any]:
    return {"type": "result", "df": dataframe, "sql": sql, "from_cache": from_cache}


def token_usage_zero_event() -> Dict[str, Any]:
    return {
        "type": "cumulative_token_usage",
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
            "breakdown": [],
        },
    }


def append_events(state: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    next_state = dict(state)
    next_state["events"] = list(state.get("events", [])) + list(events)
    return next_state
