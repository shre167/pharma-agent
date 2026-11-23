"""Wave 2 final deliverable: a Markdown evidence report built from real run/review/eval records."""
from __future__ import annotations

import pandas as pd

from ..utils import read_jsonl
from .evaluation import EVAL_TARGETS, load_evaluations, summarize
from .storage import latest_runs, now_iso


def build_wave2_summary(settings) -> str:
    lines = [
        "# Wave 2 - Evaluation, Observability, Human-in-the-Loop and Guardrails", "",
        f"Generated: {now_iso()}", "",
        "## 1. Purpose",
        "This document records the implemented Wave 2 controls and the execution evidence available.",
        "It does not claim acceptance where a required run or test has not been performed.", "",
        "## 2. Evaluation method and predefined targets", "",
    ]
    lines += [f"- **{m}:** target >= {t}" for m, t in EVAL_TARGETS.items()]

    lines += ["", "## 3. Test-set evaluation runs", ""]
    evaluations = load_evaluations(settings)
    if not evaluations:
        lines.append("No test-set evaluation has been executed yet.")
    for name, df in evaluations.items():
        lines += [f"### {name}", "", df.drop(columns=["deliverable_path", "trace_run_id"], errors="ignore")
                  .to_markdown(index=False), ""]
        observed = summarize(df)
        if len(observed):
            lines += ["**Measured metrics (errors excluded):**", "", observed.to_frame("observed").to_markdown(), ""]
        failed = df[df["outcome_accuracy"] != 1]
        lines.append(f"Cases not meeting expected outcome: {len(failed)}")
        if len(failed):
            lines += ["", failed[["case_id", "expected", "status", "failure_notes"]].to_markdown(index=False)]
        lines.append("")

    lines += ["## 4. Evaluation errors", ""]
    errors = read_jsonl(settings.outputs_dir / "evaluation_errors.jsonl")
    if errors:
        lines += ["| Evaluation | Case | Error type | Error |", "|---|---|---|---|"]
        lines += [f"| {e.get('evaluation')} | {e.get('case_id')} | {e.get('error_type')} | {str(e.get('error'))[:300]} |"
                  for e in errors]
    else:
        lines.append("No evaluation errors have been recorded.")

    lines += ["", "## 5. Live human-review evidence", ""]
    human = [r for r in read_jsonl(settings.reviews_path) if r.get("review_type") == "human"]
    if human:
        lines += ["| Report ID | Decision | Output verdict | Finalised | Reviewer |", "|---|---|---|---|---|"]
        lines += [f"| {r.get('report_id')} | {r.get('decision')} | {r.get('output_verdict')} | "
                  f"{r.get('finalised')} | {r.get('reviewer')} |" for r in human]
    else:
        lines.append("No real human Approve/Reject decision has been recorded yet.")

    lines += ["", "## 6. Observability and trace links", ""]
    runs = latest_runs(settings)
    if runs:
        lines += ["| Run ID | Status | Latency (s) | LangSmith trace |", "|---|---|---:|---|"]
        for r in runs:
            url = r.get("trace_url")
            cell = f"[Open trace]({url})" if isinstance(url, str) and url.startswith("http") else "Unavailable/not captured"
            lines.append(f"| {r.get('run_id')} | {r.get('status')} | {r.get('latency_s')} | {cell} |")
    else:
        lines.append("No run records are available yet.")

    lines += [
        "", "## 7. Guardrail evidence", "",
        "- Intake gate: medicine identity must resolve to an exact handbook ingredient (fails closed).",
        "- Input guard: records allow/refuse/escalate decisions for executed requests.",
        "- Retrieved content is passed as reference material, never as instructions.",
        "- Output checks: citation-page existence, evidence-tag validity, LLM claim/evidence review.",
        "- Finalisation gate: finalised only when the reviewer approves AND mandatory checks pass AND the verdict is PASS.",
        "", "## 8. Failures, changes and their effects", "",
        "Record any prompt, retrieval, guardrail or code change here. A before/after comparison is only "
        "meaningful after a real documented change; do not claim improvement without measured evidence.",
        "", "## 9. Acceptance status", "",
        "- [ ] Supported product/variant reports have source-level traceability.",
        "- [ ] A live LangSmith trace link is captured for representative report runs.",
        "- [ ] A refusal trace is captured.",
        "- [ ] A rejected-report trace and recorded reviewer rejection are captured.",
        "- [ ] A real human Approve/Reject decision is persisted.",
        "- [ ] No report is finalised when mandatory output checks fail.",
        "- [ ] The labelled test set has been evaluated and per-query deliverables inspected.",
        "- [ ] Guardrail tests and their outcomes are documented.",
        "", "## 10. Remaining work",
        "Any unchecked acceptance item above remains open. Regenerate after completing the outstanding runs.", "",
    ]
    return "\n".join(lines)


def write_wave2_summary(settings):
    text = build_wave2_summary(settings)
    path = settings.outputs_dir / "WAVE2_FINAL_DELIVERABLE.md"
    path.write_text(text, encoding="utf-8")
    return path, text
