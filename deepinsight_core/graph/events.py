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


def thought_start_event() -> Dict[str, Any]:
    return {"type": "thought_start"}


def thought_chunk_event(content: str) -> Dict[str, Any]:
    return {"type": "thought_chunk", "content": content}


def token_usage_event(usage: Dict[str, int]) -> Dict[str, Any]:
    return {"type": "token_usage", "usage": usage}


def result_event(
    sql: str,
    dataframe: pd.DataFrame,
    from_cache: bool = False,
    **extra: Any,
) -> Dict[str, Any]:
    payload = {"type": "result", "df": dataframe, "sql": sql, "from_cache": from_cache}
    payload.update({key: value for key, value in extra.items() if value is not None})
    return payload


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


def trace_event(trace: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "agent_trace", "trace": list(trace)}


def append_events(state: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    next_state = dict(state)
    next_state["events"] = list(state.get("events", [])) + list(events)
    return next_state
