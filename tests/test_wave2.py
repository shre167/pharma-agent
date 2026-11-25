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

