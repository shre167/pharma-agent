"""LangSmith tracing helpers. Everything degrades to a no-op when no key is configured."""
from __future__ import annotations

import os
import time
from typing import Any, Optional

try:
    from langsmith import Client
except Exception:  # pragma: no cover
    Client = None  # type: ignore

try:
    from langsmith.run_helpers import get_current_run_tree
except Exception:  # pragma: no cover
    get_current_run_tree = None  # type: ignore

# Objects that must never be serialised into a trace payload.
_HIDDEN_INPUTS = {"runtime", "services", "settings", "on_event", "reviewer_fn", "llm_chat", "client", "kb", "self"}

_client: Optional[Any] = None


def trace_safe(value: Any, _seen: set | None = None, _depth: int = 0) -> Any:
    """Bounded, JSON-safe conversion that breaks cyclic references."""
    if _seen is None:
        _seen = set()
    if _depth > 8:
        return "<max-depth>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    oid = id(value)
    if oid in _seen:
        return "<recursive-reference>"
    _seen.add(oid)
    try:
        if isinstance(value, dict):
            return {str(k): trace_safe(v, _seen, _depth + 1) for k, v in list(value.items())[:100]}
        if isinstance(value, (list, tuple, set)):
            return [trace_safe(v, _seen, _depth + 1) for v in list(value)[:100]]
        if hasattr(value, "model_dump"):
            return trace_safe(value.model_dump(), _seen, _depth + 1)
        return str(value)[:2000]
    finally:
        _seen.discard(oid)


def trace_inputs(inputs: dict) -> dict:
    return {k: trace_safe(v) for k, v in inputs.items() if k not in _HIDDEN_INPUTS and not callable(v)}


def trace_output(output: Any) -> dict:
    return {"value": trace_safe(output)}


def configure_langsmith(settings) -> bool:
    """Enable tracing if a key is present. Safe to call repeatedly."""
    global _client
    if not settings.langsmith_api_key or Client is None:
        return False
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    _client = Client()
    return True


def langsmith_enabled() -> bool:
    return _client is not None


def current_trace_id() -> Optional[str]:
    if get_current_run_tree is None:
        return None
    try:
        tree = get_current_run_tree()
        return str(tree.id) if tree is not None else None
    except Exception:
        return None


def get_trace_url(run_id: Optional[str]) -> Optional[str]:
    """Resolve a completed run's URL; retries briefly because ingestion is asynchronous."""
    if _client is None or not run_id:
        return None
    try:
        _client.flush()
    except Exception:
        pass
    for attempt in range(3):
        try:
            run_obj = _client.read_run(run_id)
            return _client.get_run_url(run=run_obj)
        except Exception:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    return None
