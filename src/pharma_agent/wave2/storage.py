"""Persistence for reports, run records and reviewer decisions."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..observability import get_trace_url
from ..utils import append_jsonl, read_jsonl

PENDING_STATUSES = {"REPORT_GENERATED_PENDING_REVIEW", "REPORT_GENERATED_NEEDS_REVIEW"}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def report_folder(settings: Settings, report_id: str) -> Path:
    return settings.reports_dir / str(report_id)


def load_report(settings: Settings, report_id: str) -> tuple[dict, str]:
    folder = report_folder(settings, report_id)
    manifest_file, report_file = folder / "manifest.json", folder / "report.md"
    if not manifest_file.exists() or not report_file.exists():
        raise FileNotFoundError(f"No generated report found for report ID {report_id}")
    return (json.loads(manifest_file.read_text(encoding="utf-8")),
            report_file.read_text(encoding="utf-8"))


def save_manifest(settings: Settings, manifest: dict) -> None:
    folder = report_folder(settings, manifest["report_id"])
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def list_reports(settings: Settings, origin: Optional[str] = "live", pending_only: bool = False) -> list[dict]:
    rows = []
    for mf in settings.reports_dir.glob("*/manifest.json"):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if origin and m.get("origin", "live") != origin:
            continue
        if pending_only and m.get("status") not in PENDING_STATUSES:
            continue
        rows.append(m)
    rows.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return rows


def submit_review(settings: Settings, report_id: str, decision: str, reviewer: str = "", comment: str = "") -> dict:
    """Record a real human Approve/Reject decision.

    A report is FINALISED only if the reviewer approves AND every mandatory check passed AND the
    automatic verdict is PASS. Approval of a flagged/failed report is recorded but blocked.
    """
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be 'approve' or 'reject'")
    manifest, report_text = load_report(settings, report_id)
    if manifest.get("status") not in PENDING_STATUSES:
        raise ValueError(f"Report {report_id} has already been reviewed (status: {manifest.get('status')}).")

    checks = manifest.get("checks", [])
    mandatory_ok = bool(checks) and all(c.get("passed", False) for c in checks)
    verdict = manifest.get("output_verdict")
    finalised = decision == "approve" and mandatory_ok and verdict == "PASS"
    status = "FINALISED" if finalised else ("REJECTED" if decision == "reject" else "BLOCKED_NOT_FINALISED")

    review = {"review_type": "human", "decision": decision, "reviewer": reviewer.strip() or "reviewer",
              "comment": comment.strip(), "decision_recorded": True}
    manifest.update({
        "status": status, "finalised": finalised, "mandatory_checks_pass": mandatory_ok,
        "review_decision": review, "reviewed_at": now_iso(),
    })
    if finalised:
        settings.approved_dir.mkdir(parents=True, exist_ok=True)
        approved = settings.approved_dir / f"{report_id}.md"
        approved.write_text(report_text, encoding="utf-8")
        manifest["approved_path"] = str(approved)
    save_manifest(settings, manifest)
    append_jsonl(settings.reviews_path, {
        "report_id": report_id, **review, "output_verdict": verdict,
        "finalised": finalised, "at": manifest["reviewed_at"],
    })
    return manifest


def latest_runs(settings: Settings) -> list[dict]:
    latest: dict[str, dict] = {}
    for row in read_jsonl(settings.runs_path):
        latest[row.get("run_id")] = row
    return list(latest.values())


def refresh_trace_link(settings: Settings, result: dict) -> dict:
    """Resolve the completed root run's LangSmith URL and write it back to the logs/manifest."""
    trace_id = result.get("trace_run_id")
    if not trace_id:
        return result
    url = get_trace_url(trace_id)
    result["trace_url"] = url
    if url is None:
        return result
    append_jsonl(settings.runs_path, {k: v for k, v in result.items()
                                      if k not in {"report", "checks", "llm_review", "evidence"}})
    manifest_path = result.get("manifest_path")
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest["trace_url"] = url
        Path(manifest_path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
