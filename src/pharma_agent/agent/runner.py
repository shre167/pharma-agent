"""Run the LangChain agent behind the medicine gate."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from .prompts import AGENT_SYSTEM_PROMPT, OUT_OF_SCOPE_MSG
from .scope import check_scope
from .tools import make_tools

EventFn = Optional[Callable[[str, dict], None]]


def _msg_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else b.get("text", "")
                       for b in content if isinstance(b, str) or (isinstance(b, dict) and b.get("type") in (None, "text")))
    return str(content)


def _summarize_observation(text: str) -> dict:
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError
    except Exception:
        return {"result_count": None, "status": None, "path": None, "error": str(text)[:200]}
    summary = {"result_count": data.get("result_count"), "status": data.get("status"),
               "path": data.get("path"), "error": data.get("error")}
    if data.get("note"):
        summary["note"] = data["note"]
    return summary


_OFFER_START = re.compile(
    r"^\s*(?:[-*]\s*)?(?:if you(?:'d| would)? (?:want|like)|would you like|do you want|should i|shall i|let me know|"
    r"i can (?:also |next |further )?(?:extract|produce|prepare|help|provide|generate|create|compare|expand|draft|build|run|search)|"
    r"happy to|feel free)", re.I)


def strip_followups(text: str) -> str:
    """Safety net: remove trailing 'If you want, I can...' offers and closing questions."""
    blocks = re.split(r"\n\s*\n", (text or "").strip())
    for i, block in enumerate(blocks):
        if _OFFER_START.match(block):
            blocks = blocks[:i]
            break
    lines = "\n\n".join(blocks).rstrip().split("\n")
    while lines and lines[-1].strip().endswith("?"):
        lines.pop()
    return "\n".join(lines).rstrip()


def _run_agent(runtime, session, query: str, max_iterations: int, on_event: EventFn) -> dict:
    """Stream the agent. `max_iterations` is a cost safeguard, not a search plan."""
    agent = create_agent(model=runtime.services.chat_model, tools=make_tools(session),
                         system_prompt=AGENT_SYSTEM_PROMPT)
    max_iterations = max(1, min(int(max_iterations), 12))
    config = {"recursion_limit": 2 * max_iterations + 1}  # one model step + one tool step per iteration
    pending: dict[str, dict] = {}
    tool_calls: list[dict] = []
    final_text, hit_limit = "", False
    try:
        for update in agent.stream({"messages": [{"role": "user", "content": query.strip()}]},
                                   config=config, stream_mode="updates"):
            for node_output in update.values():
                if not isinstance(node_output, dict):
                    continue
                for msg in node_output.get("messages", []) or []:
                    if isinstance(msg, AIMessage):
                        for tc in msg.tool_calls or []:
                            pending[tc["id"]] = {"tool": tc["name"], "arguments": tc["args"]}
                        if not msg.tool_calls:
                            final_text = _msg_text(msg)
                    elif isinstance(msg, ToolMessage):
                        info = pending.pop(msg.tool_call_id, {"tool": msg.name, "arguments": {}})
                        summary = _summarize_observation(_msg_text(msg))
                        entry = {"tool": info["tool"], "arguments": info["arguments"], "result_summary": summary}
                        tool_calls.append(entry)
                        if on_event:
                            on_event("tool_call", {"index": len(tool_calls), "max": max_iterations, **entry})
    except GraphRecursionError:
        hit_limit = True

    report_path = next((c["result_summary"]["path"] for c in reversed(tool_calls)
                        if c["tool"] == "build_report" and c["result_summary"].get("path")), None)
    answer = strip_followups(final_text)
    if not answer:
        answer = ("The agent reached its step budget before writing a final answer. "
                  f"{len(tool_calls)} tool call(s) completed; see the tool trace.") if hit_limit \
            else "The agent returned no final answer."
    if report_path:
        answer += f'\n\n(Report file created: "{report_path}")'
    return {
        "status": "STEP_LIMIT_REACHED" if hit_limit else "COMPLETED",
        "answer": answer, "tool_calls": tool_calls, "report_path": report_path,
        "evidence_count": len(session.evidence),
    }


def ask_agent(runtime, query: str, max_iterations: Optional[int] = None,
              on_event: EventFn = None, scope: Optional[dict] = None) -> dict:
    """Fail-closed intake gate -> verified handbook identity -> agent.

    The returned dict includes `capture` ({"text","path","selected"}) when a report was built.
    """
    if not isinstance(query, str) or not query.strip():
        return {"status": "INVALID_INPUT", "answer": "Please enter a question or report request.", "tool_calls": []}
    if len(query) > 10000:
        return {"status": "INVALID_INPUT", "answer": "The request is too long. Please shorten it and try again.", "tool_calls": []}

    max_iterations = max_iterations or runtime.settings.max_iterations
    passed_scope = scope is not None
    scope = scope if scope is not None else check_scope(runtime.kb.vocab, runtime.services.llm_chat, query)
    if on_event and not passed_scope:
        on_event("scope", scope)

    if not scope.get("is_medicine_request"):
        return {"status": "OUT_OF_SCOPE", "scope": scope, "tool_calls": [],
                "answer": OUT_OF_SCOPE_MSG + f"\n\n_Reason: {scope.get('reason', '')}_"}

    generic_names = scope.get("generic_names", [])
    if generic_names:
        matches = runtime.kb.handbook_matches(generic_names)
        mention = ", ".join(scope.get("medicine_mentions", [])) or ", ".join(generic_names)
        if not matches:
            ids = scope.get("active_ingredients_identified", [])
            if ids:
                return {"status": "NOT_IN_HANDBOOK", "scope": scope, "tool_calls": [],
                        "answer": (f"'{mention}' was recognised as a medicine (active ingredient(s): {', '.join(ids)}), "
                                   "but the handbook has no matching formulation. No retrieval or report generation was run.")}
            return {"status": "UNRESOLVED_MEDICINE", "scope": scope, "tool_calls": [],
                    "answer": f"The medicine '{mention}' could not be resolved to a handbook ingredient. "
                              "No retrieval or report generation was run."}
    else:
        matches = []

    session = runtime.new_session()
    session.on_event = on_event
    session.allowed = True          # tools are authorised only after identity validation
    session.generic_names = generic_names
    context = (
        "\n\n[RESOLVED CONTEXT - validated by the intake gate]\n"
        f"Medicine identity: {', '.join(scope.get('medicine_mentions', [])) or 'General pharmaceutical request'}\n"
        f"Identification method: {scope.get('identification_method', 'unknown')}\n"
        f"Identification confidence: {scope.get('confidence', 'unknown')}\n"
        f"Resolved handbook ingredient names: {', '.join(generic_names) if generic_names else 'N/A (general pharma manufacturing)'}\n"
        + (("Matching handbook formulation headings:\n- " + "\n- ".join(matches[:25])) if matches else "General manufacturing inquiry; search handbook documents as needed.")
        + "\nIf the variant the user wants is ambiguous, cover every matching variant separately "
        "and record the ambiguity under evidence gaps. Do not ask the user questions."
    )
    result = _run_agent(runtime, session, query.strip() + context, max_iterations, on_event)
    result["scope"] = scope
    result["capture"] = session.capture
    session.allowed = False         # revoke authorisation once the request finishes
    return result
