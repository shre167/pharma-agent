"""LLM-based input guard. Fails closed (escalate) if the classifier misbehaves."""
from __future__ import annotations

import json
from typing import Callable

from langsmith import traceable

from ..agent.prompts import INPUT_GUARD_PROMPT
from ..observability import trace_inputs, trace_output
from ..utils import parse_json


@traceable(name="input_safety_check", run_type="chain", process_inputs=trace_inputs, process_outputs=trace_output)
def check_input_with_llm(llm_chat: Callable, query: str) -> dict:
    raw = llm_chat([
        {"role": "system", "content": INPUT_GUARD_PROMPT},
        {"role": "user", "content": json.dumps({"request": query})},
    ])
    try:
        data = parse_json(raw)
        action = data.get("action", "escalate")
        if action not in {"allow", "refuse", "escalate"}:
            action = "escalate"
        return {"action": action, "reason": str(data.get("reason", "No reason returned."))[:1000]}
    except Exception:
        return {"action": "escalate", "reason": "The LLM input check returned an invalid response."}
