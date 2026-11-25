import json

import pytest

from pharma_agent.wave2.guard import check_input_with_llm
from pharma_agent.wave2.storage import list_reports, load_report, save_manifest, submit_review
from pharma_agent.wave2.verify import citation_checks, verify_report

EVIDENCE = [
    {"text": "bom", "metadata": {"page": 232, "pages": "232, 233", "formulation_name": "ACETAMINOPHEN TABLETS", "source": "h.pdf"}},
    {"text": "process", "metadata": {"page": 240, "pages": "240-241", "formulation_name": "ACETAMINOPHEN TABLETS", "source": "h.pdf"}},
]
GOOD = "Ingredients listed [E1] [SOURCE: h.pdf, PDF page 232]. Steps [E2] [SOURCE: h.pdf, PDF page 241]."


def by_name(checks):
    return {c["name"]: c for c in checks}


def test_citation_checks_pass_for_supported_citations():
    c = by_name(citation_checks(GOOD, EVIDENCE))
    assert all(v["passed"] for v in c.values())


def test_fabricated_page_and_tag_are_caught():
    bad = "Claim [E7] [SOURCE: h.pdf, PDF page 999]."
    c = by_name(citation_checks(bad, EVIDENCE))
    assert not c["citation_pages_exist_in_selected_evidence"]["passed"]
    assert "999" in c["citation_pages_exist_in_selected_evidence"]["detail"]
    assert not c["evidence_tags_match_selected_evidence"]["passed"]


def test_page_number_on_next_line_is_not_misread_as_citation():
    text = "[E1] [SOURCE: h.pdf, PDF page 232]\n3. Next step text [E2]"
    c = by_name(citation_checks(text, EVIDENCE))
    assert c["citation_pages_exist_in_selected_evidence"]["passed"]


def test_verdicts():
    ok = lambda m: json.dumps({"verdict": "pass", "reason": "fine", "issues": []})
    flag = lambda m: json.dumps({"verdict": "flag", "reason": "unsupported", "issues": [{"claim": "x", "reason": "y"}]})
    assert verify_report(ok, GOOD, EVIDENCE)[0] == "PASS"
    assert verify_report(flag, GOOD, EVIDENCE)[0] == "FLAG"
    assert verify_report(ok, "no citations at all", EVIDENCE)[0] == "FAIL"
    assert verify_report(lambda m: "garbage", GOOD, EVIDENCE)[0] == "FLAG"   # LLM outage never silently passes


def test_pp_style_citations_are_recognised():
    # The agent prompt teaches "[Handbook, pp. X-Y]" - the verifier must read that format too.
    report = "Acetaminophen 500 mg [E1] [Handbook, pp. 232-233]"
    c = by_name(citation_checks(report, EVIDENCE))
    assert c["citation_pages_exist_in_selected_evidence"]["passed"]
    assert c["evidence_tags_match_selected_evidence"]["passed"]


def test_input_guard_fails_closed():
    assert check_input_with_llm(lambda m: json.dumps({"action": "allow", "reason": "ok"}), "q")["action"] == "allow"
    assert check_input_with_llm(lambda m: "nonsense", "q")["action"] == "escalate"
    assert check_input_with_llm(lambda m: json.dumps({"action": "maybe"}), "q")["action"] == "escalate"


def _seed(settings, report_id, verdict, checks_pass=True):
    folder = settings.reports_dir / report_id
    folder.mkdir(parents=True)
    (folder / "report.md").write_text("# report", encoding="utf-8")
    save_manifest(settings, {
        "report_id": report_id, "origin": "live", "query": "q", "created_at": "2026-01-01T00:00:00",
        "status": "REPORT_GENERATED_PENDING_REVIEW" if verdict == "PASS" else "REPORT_GENERATED_NEEDS_REVIEW",
        "output_verdict": verdict, "finalised": False,
        "checks": [{"name": "a", "passed": checks_pass, "detail": ""}], "evidence": []})


def test_approve_finalises_only_when_everything_passed(settings):
    _seed(settings, "r1", "PASS")
    m = submit_review(settings, "r1", "approve", "Dr A", "looks right")
    assert m["status"] == "FINALISED" and m["finalised"]
    assert (settings.approved_dir / "r1.md").read_text() == "# report"
    assert json.loads(settings.reviews_path.read_text().splitlines()[-1])["reviewer"] == "Dr A"


def test_approve_is_blocked_when_verification_failed(settings):
    _seed(settings, "r2", "FAIL", checks_pass=False)
    m = submit_review(settings, "r2", "approve", "Dr A")
    assert m["status"] == "BLOCKED_NOT_FINALISED" and not m["finalised"]
    assert not (settings.approved_dir / "r2.md").exists()


def test_reject_and_no_double_review(settings):
    _seed(settings, "r3", "PASS")
    assert submit_review(settings, "r3", "reject", "Dr B")["status"] == "REJECTED"
    with pytest.raises(ValueError):
        submit_review(settings, "r3", "approve", "Dr B")
    with pytest.raises(ValueError):
        submit_review(settings, "r3", "maybe", "Dr B")


def test_list_reports_filters(settings):
    _seed(settings, "p1", "PASS")
    _seed(settings, "p2", "PASS")
    submit_review(settings, "p2", "reject", "x")
    assert [m["report_id"] for m in list_reports(settings, pending_only=True)] == ["p1"]
    assert len(list_reports(settings)) == 2
    assert load_report(settings, "p1")[1] == "# report"


def test_case_set_hash_changes_when_the_set_changes():
    from pharma_agent.wave2.evaluation import case_set_hash

    base = [{"id": "P01", "query": "q", "expected": "report"}]
    changed = [{"id": "P01", "query": "different query", "expected": "report"}]
    assert case_set_hash(base) != case_set_hash(changed)
    assert len(case_set_hash(base)) == 8


def test_compare_evaluations_marks_the_case_set(settings):
    from pharma_agent.wave2.evaluation import case_set_hash, compare_evaluations

    rows = [{"evaluation": "e1", "case_id": "P01", "status": "REPORT_GENERATED_PENDING_REVIEW",
             "outcome_accuracy": 1, "retrieval_hit": 1, "citation_integrity": 1,
             "faithfulness": 0.9, "retrieval_relevancy": 0.6, "failure_count": 0,
             "failure_notes": "", "latency_s": 1.0, "error": None,
             "case_set": case_set_hash([{"id": "P01"}])}]
    (settings.eval_dir / "e1_evaluation.json").write_text(json.dumps(rows), encoding="utf-8")
    frame = compare_evaluations(settings)
    assert frame.loc["e1", "case_set"] == case_set_hash([{"id": "P01"}])
    assert frame.loc["e1", "outcome_accuracy"] == 1


def test_measure_report_quality_degrades_on_judge_failure():
    from pharma_agent.wave2.evaluation import measure_report_quality

    empty = measure_report_quality(lambda m: json.dumps({}), "q", "", ["ctx"])
    assert empty["ok"] is False and empty["faithfulness"] is None
    bad = measure_report_quality(lambda m: "garbage from the judge", "q", "report text", ["ctx"])
    assert bad["ok"] is False and bad["faithfulness"] is None and "DeepEval failed" in bad["note"]


def test_pp_style_citations_are_recognised():
    # The agent prompt teaches "[Handbook, pp. X-Y]" - the verifier must read that format too.
    report = ("# Report\n\n## 2. Bill of materials\nAcetaminophen 500 mg [E1] [Handbook, pp. 74-75]")
    evidence = [_evidence("ACETAMINOPHEN TABLETS", "BOM", page=74, pages="74-75")]
    checks = citation_checks(report, evidence)
    by_name = {c["name"]: c for c in checks}
    assert by_name["citation_pages_exist_in_selected_evidence"]["passed"]
    assert by_name["evidence_tags_match_selected_evidence"]["passed"]
