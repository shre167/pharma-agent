"""Integrated run: scope gate -> input guard -> Wave 1 agent -> verification -> persisted draft.

The human Approve/Reject decision is a separate step (storage.submit_review) so it works in a
web UI. Automated evaluation never simulates a human decision.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from typing import Callable, Optional

from langsmith import traceable

from ..agent.runner import ask_agent
from ..agent.scope import check_scope
from ..observability import current_trace_id, trace_inputs, trace_output
from ..utils import append_jsonl
from .guard import check_input_with_llm
from .storage import now_iso, refresh_trace_link, report_folder, save_manifest
from .verify import verify_report

EventFn = Optional[Callable[[str, dict], None]]

_REFUSAL_STATUSES = {"OUT_OF_SCOPE", "NOT_IN_HANDBOOK", "UNRESOLVED_MEDICINE", "NEEDS_MEDICINE_NAME", "INVALID_INPUT"}
_RUN_LOG_EXCLUDE = {"report", "checks", "llm_review", "evidence"}


def _log_run(settings, result: dict) -> None:
    append_jsonl(settings.runs_path, {k: v for k, v in result.items() if k not in _RUN_LOG_EXCLUDE})


@traceable(name="pharma_design_report_run", run_type="chain", process_inputs=trace_inputs, process_outputs=trace_output)
def _generate(runtime, query: str, run_id: str, origin: str, on_event: EventFn) -> dict:
    settings = runtime.settings
    started = time.time()
    emit = on_event or (lambda *_a, **_k: None)
    trace_id = current_trace_id()
    base = {"run_id": run_id, "origin": origin, "query": query, "trace_run_id": trace_id,
            "created_at": now_iso()}

    def early(status: str, **extra) -> dict:
        result = {**base, "status": status, "report": None, "tool_calls": [],
                  "latency_s": round(time.time() - started, 3), **extra}
        _log_run(settings, result)
        return result

    # 1. Medicine identity (explicit name OR description) validated against handbook vocabulary.
    scope = check_scope(runtime.kb.vocab, runtime.services.llm_chat, query)
    emit("scope", scope)
    if not scope.get("generic_names"):
        return early("REFUSED", scope=scope,
                     reason=scope.get("reason", "No sufficiently supported medicine identity was found."))

    # 2. Input guard (prompt injection, fabrication, GxP falsification, ...).
    guard = check_input_with_llm(runtime.services.llm_chat, query)
    emit("guard", guard)
    if guard["action"] != "allow":
        return early("REFUSED" if guard["action"] == "refuse" else "ESCALATED", reason=guard["reason"], scope=scope)

    # 3. Wave 1 agent.
    agent_result = ask_agent(runtime, query, on_event=on_event, scope=scope)
    capture = agent_result.get("capture")
    if capture is None:
        status = "REFUSED" if agent_result.get("status") in _REFUSAL_STATUSES else "NO_REPORT"
        return early(status, scope=scope, answer=agent_result.get("answer", ""),
                     wave1_status=agent_result.get("status"), tool_calls=agent_result.get("tool_calls", []))

    # 4. Verification of the exact draft.
    report_text, selected = capture["text"], capture["selected"]
    emit("verify", {"stage": "start"})
    verdict, checks, ai_review = verify_report(runtime.services.llm_chat, report_text, selected)
    emit("verify", {"stage": "done", "verdict": verdict})
    status = "REPORT_GENERATED_PENDING_REVIEW" if verdict == "PASS" else "REPORT_GENERATED_NEEDS_REVIEW"

    # 5. Persist report + evidence manifest.
    report_id = run_id
    folder = report_folder(settings, report_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(report_text, encoding="utf-8")
    evidence_manifest = []
    for i, item in enumerate(selected, 1):
        meta = item.get("metadata", {})
        evidence_manifest.append({
            "evidence_id": item.get("evidence_id"), "tag": f"E{i}",
            "formulation_id": meta.get("formulation_id"), "formulation_name": meta.get("formulation_name"),
            "chunk_type": meta.get("chunk_type"), "source": meta.get("source"),
            "page": meta.get("page"), "pages": meta.get("pages"), "text": item.get("text", ""),
        })
    latency = round(time.time() - started, 3)
    manifest = {
        "report_id": report_id, "run_id": run_id, "origin": origin, "query": query,
        "status": status, "finalised": False, "output_verdict": verdict,
        "mandatory_checks_pass": all(c["passed"] for c in checks),
        "checks": checks, "llm_review": ai_review,
        "review_decision": {"review_type": "pending", "decision": "pending"},
        "scope": scope, "tool_calls": agent_result.get("tool_calls", []),
        "trace_run_id": trace_id, "trace_url": None, "latency_s": latency,
        "wave1_report_path": str(capture["path"]), "evidence": evidence_manifest,
        "report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
        "created_at": base["created_at"], "reviewed_at": None,
    }
    save_manifest(settings, manifest)

    result = {
        **base, "report_id": report_id, "status": status, "output_verdict": verdict,
        "report_path": str(folder / "report.md"), "manifest_path": str(folder / "manifest.json"),
        "report": report_text, "checks": checks, "llm_review": ai_review,
        "evidence_count": len(selected), "tool_calls": agent_result.get("tool_calls", []),
        "answer": agent_result.get("answer", ""), "latency_s": latency,
        "evidence": [{"formulation_name": e.get("metadata", {}).get("formulation_name"),
                      "formulation_id": e.get("metadata", {}).get("formulation_id"),
                      "text": e.get("text", "")} for e in selected],
    }
    _log_run(settings, result)
    return result


def run_generation(runtime, query: str, run_id: Optional[str] = None, origin: str = "live",
                   on_event: EventFn = None) -> dict:
    """Public entry point: run the whole pipeline, then resolve the LangSmith trace link."""
    run_id = run_id or str(uuid.uuid4())
    result = _generate(runtime, query, run_id, origin, on_event)
    return refresh_trace_link(runtime.settings, result)
