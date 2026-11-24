"""Objective citation checks plus one LLM claim-vs-evidence review of the actual draft."""
from __future__ import annotations

import json
import re
from typing import Callable

from langsmith import traceable

from ..agent.prompts import VERIFY_PROMPT
from ..observability import trace_inputs, trace_output
from ..utils import parse_json

HARD_CHECKS = {"citation_pages_exist_in_selected_evidence", "evidence_tags_match_selected_evidence"}


def expand_pages(text: str, max_span: int = 100) -> set[int]:
    pages: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*[-–]\s*(\d+)", text):
        if int(b) - int(a) <= max_span:
            pages.update(range(int(a), int(b) + 1))
    pages.update(int(x) for x in re.findall(r"\d+", text))
    return pages


def citation_checks(report_text: str, selected_evidence: list[dict]) -> list[dict]:
    checks = []
    evidence_pages: set[int] = set()
    for e in selected_evidence:
        m = e.get("metadata", {})
        for value in (m.get("page"), m.get("pages")):
            if value is not None:
                evidence_pages |= expand_pages(str(value))

    cited_pages: set[int] = set()
    # [ \t] (not \s) so a page number on the next line is never swallowed.
    for match in re.finditer(r"PDF pages?[ \t]*([\d,\t –-]+)", report_text, re.I):
        cited_pages |= expand_pages(match.group(1))
    bad_pages = sorted(cited_pages - evidence_pages)
    checks.append({
        "name": "citation_pages_exist_in_selected_evidence",
        "passed": bool(cited_pages) and not bad_pages,
        "detail": (f"Unsupported cited pages: {bad_pages}" if bad_pages else
                   "All cited pages occur in selected evidence." if cited_pages else
                   "No PDF page citations were found."),
    })

    referenced = {int(x) for x in re.findall(r"\[E(\d+)\]", report_text)}
    invalid = sorted(referenced - set(range(1, len(selected_evidence) + 1)))
    checks.append({
        "name": "evidence_tags_match_selected_evidence",
        "passed": bool(referenced) and not invalid,
        "detail": (f"Invalid evidence tags: {invalid}" if invalid else
                   "All evidence tags map to selected chunks." if referenced else
                   "No evidence tags were found."),
    })
    checks.append({
        "name": "evidence_available", "passed": bool(selected_evidence),
        "detail": ("No evidence was supplied to the report generator." if not selected_evidence
                   else f"{len(selected_evidence)} evidence chunks captured."),
    })
    return checks


@traceable(name="verify_report", run_type="chain", process_inputs=trace_inputs, process_outputs=trace_output)
def verify_report(llm_chat: Callable, report_text: str, selected_evidence: list[dict]):
    """Returns (verdict PASS|FLAG|FAIL, checks, llm_review)."""
    checks = citation_checks(report_text, selected_evidence)
    payload = {
        "evidence": [{"source": e.get("metadata", {}).get("source"),
                      "page": e.get("metadata", {}).get("page"),
                      "formulation": e.get("metadata", {}).get("formulation_name"),
                      "text": e.get("text", "")[:1500]} for e in selected_evidence[:8]],
        "draft_report": report_text[:7000],
    }
    try:
        raw = llm_chat([{"role": "system", "content": VERIFY_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        review = parse_json(raw)
        if review.get("verdict") not in {"pass", "flag"}:
            review["verdict"] = "flag"
    except Exception as exc:  # noqa: BLE001
        review = {"verdict": "flag", "issues": [], "reason": f"LLM verification unavailable: {exc!r}"}
    checks.append({"name": "llm_claim_evidence_review", "passed": review["verdict"] == "pass",
                   "detail": review.get("reason", "")})

    hard_fail = any(not c["passed"] for c in checks if c["name"] in HARD_CHECKS)
    verdict = "FAIL" if hard_fail else ("PASS" if review["verdict"] == "pass" else "FLAG")
    return verdict, checks, review
