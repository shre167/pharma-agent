"""Labelled test-set evaluation (DeepEval + objective metrics). No simulated human approval.

Metrics are fixed in advance and each measures one required dimension:
- outcome_accuracy      expected result vs observed status AND verification verdict (refusal
                        behaviour included: refused/escalated cases are scored here)
- retrieval_hit         gold ingredient/formulation term found in retrieved formulation names
- citation_integrity    deterministic citation checks (pages exist, evidence tags valid)
- faithfulness          DeepEval: report claims supported by retrieved context
- retrieval_relevancy   DeepEval: retrieved context relevant to the query
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..utils import append_jsonl, strip_json_fences
from .pipeline import run_generation
from .storage import latest_runs, now_iso

EVAL_TARGETS = {
    "outcome_accuracy": 1.0, "retrieval_hit_rate": 0.75, "citation_integrity": 1.0,
    "faithfulness": 0.80, "retrieval_relevancy": 0.50,
}

METRIC_COLUMNS = ["outcome_accuracy", "retrieval_hit", "citation_integrity",
                  "faithfulness", "retrieval_relevancy"]


def case_set_hash(cases: list[dict]) -> str:
    """Identity of the evaluated case set; evaluations are comparable only when it matches."""
    material = json.dumps([[c.get("id"), c.get("query"), c.get("expected")] for c in cases],
                          sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:8]


def load_eval_cases(settings) -> list[dict]:
    return json.loads(Path(settings.eval_cases_path).read_text(encoding="utf-8"))


def _make_judge(llm_chat: Callable):
    """DeepEval judge backed by the same Capgemini chat endpoint (imported lazily: heavy)."""
    from deepeval.models import DeepEvalBaseLLM

    class CapgeminiDeepEvalJudge(DeepEvalBaseLLM):
        def __init__(self):
            self._model_name = "capgemini-chat"

        def load_model(self):
            return None

        def get_model_name(self):
            return self._model_name

        def generate(self, prompt, schema=None):
            raw = llm_chat([{"role": "user", "content": prompt + "\n\nReturn only valid JSON if a schema is requested."}])
            if schema is None:
                return raw
            return schema(**json.loads(strip_json_fences(raw)))

        async def a_generate(self, prompt, schema=None):
            return self.generate(prompt, schema)

    return CapgeminiDeepEvalJudge()


def write_case_deliverable(case: dict, result: dict, metrics: dict, failures: list[str], output_dir: Path) -> str:
    """Per-query Markdown deliverable: the report, or a failure analysis when none was produced."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Query deliverable - {case['id']}", "",
             f"**Query:** {case['query']}", "",
             f"**Expected outcome:** {case.get('expected')}",
             f"**Actual status:** {result.get('status')}",
             f"**Run ID:** {result.get('run_id')}",
             f"**Trace URL:** {result.get('trace_url') or 'Not available'}", ""]
    if result.get("report"):
        lines += ["## Generated report", "", result["report"], ""]
    else:
        lines += ["## Deliverable status", "", "No manufacturing report was generated for this query.",
                  f"**Reason / failure:** {result.get('reason') or result.get('answer') or result.get('wave1_status') or 'No report output captured.'}", ""]
    lines += ["## Evaluation and diagnostics", ""]
    lines += [f"- **{k}:** {v}" for k, v in (metrics or {}).items()]
    if failures:
        lines += ["", "### Failures / items requiring attention", ""] + [f"- {x}" for x in failures]
    if result.get("tool_calls") is not None:
        lines += ["", f"**Tool calls recorded:** {len(result.get('tool_calls') or [])}"]
    lines += ["", "## Interpretation", "",
              "This is an educational evidence-assistant output. It is not validated executable "
              "manufacturing instruction or regulatory approval."]
    path = output_dir / "deliverable.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _evaluate_case(runtime, case: dict, name: str, judge) -> tuple[dict, list[str]]:
    from deepeval.metrics import ContextualRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    run_id = f"{name}_{case['id']}_{uuid.uuid4().hex[:6]}"
    failures: list[str] = []
    result = run_generation(runtime, case["query"], run_id=run_id, origin="evaluation")
    has_report = result.get("report") is not None

    verdict = result.get("output_verdict")
    if case["expected"] == "report":
        # Consistent with the review gate: a report the verifier flagged is not a success.
        outcome_ok = (has_report and result.get("status") in
                      {"REPORT_GENERATED_PENDING_REVIEW", "REPORT_GENERATED_NEEDS_REVIEW"}
                      and verdict == "PASS")
        if has_report and verdict != "PASS":
            failures.append(f"Report was produced but verification flagged it (verdict {verdict}).")
    elif case["expected"] == "refuse":
        outcome_ok = result.get("status") == "REFUSED"
    else:
        outcome_ok = result.get("status") in {"REFUSED", "ESCALATED"}
    if (not outcome_ok) and not (case["expected"] == "report" and has_report and verdict != "PASS"):
        failures.append(f"Expected {case['expected']}; observed {result.get('status')}.")

    names = [str(e.get("formulation_name", "")).casefold() for e in result.get("evidence", [])]
    retrieval_hit = (any(t.casefold() in n for t in case.get("gold_terms", []) for n in names)
                     if case.get("gold_terms") and has_report else None)
    if case.get("gold_terms") and has_report and not retrieval_hit:
        failures.append("Expected ingredient/formulation term was not found in retrieved formulation names.")

    citation_ok = faithfulness = relevancy = None
    if has_report:
        cites = [c for c in result.get("checks", []) if c["name"] in {
            "citation_pages_exist_in_selected_evidence", "evidence_tags_match_selected_evidence"}]
        citation_ok = bool(cites) and all(c["passed"] for c in cites)
        if not citation_ok:
            failures.append("Citation integrity checks failed or were unavailable.")
        contexts = [e["text"] for e in result.get("evidence", [])]
        if contexts:
            tc = LLMTestCase(input=case["query"], actual_output=result["report"], retrieval_context=contexts)
            fm = FaithfulnessMetric(threshold=EVAL_TARGETS["faithfulness"], model=judge, async_mode=False)
            fm.measure(tc)
            faithfulness = float(fm.score)
            rm = ContextualRelevancyMetric(threshold=EVAL_TARGETS["retrieval_relevancy"], model=judge, async_mode=False)
            rm.measure(tc)
            relevancy = float(rm.score)
            if faithfulness < EVAL_TARGETS["faithfulness"]:
                failures.append(f"Faithfulness {faithfulness:.3f} below target.")
            if relevancy < EVAL_TARGETS["retrieval_relevancy"]:
                failures.append(f"Contextual relevancy {relevancy:.3f} below target.")
        else:
            failures.append("No retrieved context available for DeepEval metrics.")

    metrics = {
        "outcome_accuracy": int(outcome_ok), "retrieval_hit": retrieval_hit,
        "citation_integrity": citation_ok, "faithfulness": faithfulness,
        "retrieval_relevancy": relevancy,
    }
    deliverable = write_case_deliverable(case, result, metrics, failures,
                                         runtime.settings.eval_dir / name / case["id"])
    row = {"evaluation": name, "case_id": case["id"], "expected": case["expected"],
           "status": result.get("status"), **metrics, "failure_count": len(failures),
           "failure_notes": " | ".join(failures), "deliverable_path": deliverable,
           "trace_run_id": result.get("trace_run_id"), "trace_url": result.get("trace_url"),
           "latency_s": result.get("latency_s"), "error": None}
    return row, failures


def run_evaluation(runtime, name: str = "baseline", cases: Optional[list[dict]] = None,
                   on_case: Optional[Callable[[int, int, dict], None]] = None) -> pd.DataFrame:
    """Run the labelled test set against the current system. Each case gets its own deliverable.md."""
    settings = runtime.settings
    cases = cases or load_eval_cases(settings)
    case_set = case_set_hash(cases)
    judge = _make_judge(runtime.services.llm_chat)
    rows = []
    for i, case in enumerate(cases, 1):
        if on_case:
            on_case(i, len(cases), case)
        try:
            row, _ = _evaluate_case(runtime, case, name, judge)
        except Exception as exc:  # noqa: BLE001
            failure = f"Evaluation/execution error: {type(exc).__name__}: {exc}"
            error_result = {"run_id": f"{name}_{case['id']}", "status": "ERROR", "reason": repr(exc), "tool_calls": []}
            deliverable = write_case_deliverable(case, error_result, {}, [failure],
                                                 settings.eval_dir / name / case["id"])
            append_jsonl(settings.outputs_dir / "evaluation_errors.jsonl", {
                "evaluation": name, "case_id": case["id"], "error_type": type(exc).__name__,
                "error": repr(exc), "at": now_iso()})
            row = {"evaluation": name, "case_id": case["id"], "expected": case["expected"],
                   "status": "ERROR", **{k: None for k in METRIC_COLUMNS}, "failure_count": 1,
                   "failure_notes": failure, "deliverable_path": deliverable,
                   "trace_run_id": None, "trace_url": None, "latency_s": None, "error": repr(exc)}
        row["case_set"] = case_set
        rows.append(row)

    path = settings.eval_dir / f"{name}_evaluation.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return pd.DataFrame(rows)


def load_evaluations(settings) -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(settings.eval_dir.glob("*_evaluation.json")):
        out[f.stem.removesuffix("_evaluation")] = pd.DataFrame(json.loads(f.read_text(encoding="utf-8")))
    return out


def summarize(frame: pd.DataFrame) -> pd.Series:
    ok = frame[frame["status"] != "ERROR"]
    cols = [c for c in METRIC_COLUMNS if c in ok.columns]
    return ok[cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)


def compare_evaluations(settings) -> pd.DataFrame:
    """Before/after comparison. A 'case_set' column marks which case set each run used;
    metrics across runs with different case_set values are not comparable."""
    frames = load_evaluations(settings)
    if not frames:
        return pd.DataFrame()
    cols = {}
    for name, df in frames.items():
        metrics = summarize(df)
        metrics["case_set"] = (df["case_set"].iloc[0]
                               if "case_set" in df.columns and len(df) else "?")
        cols[name] = metrics
    return pd.DataFrame(cols).T
