"""Staged per-section report generation: routing, stable tags, assembly, failure resilience."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharma_agent.agent import prompts  # noqa: E402
from pharma_agent.agent.report import generate_report  # noqa: E402


def make_evidence(ctype: str, text: str, page: int = 2) -> dict:
    return {"metadata": {"formulation_name": "ACETAMINOPHEN TABLETS", "chunk_type": ctype,
                         "formulation_id": "fid_acetaminophen", "source": "handbook.pdf",
                         "page": page, "pages": str(page)},
            "citation": f"[SOURCE: handbook.pdf, PDF page {page}]", "text": text}


BOM_TEXT = "BILL OF MATERIALS Acetaminophen 500 mg, starch 50 mg"
PROCESS_TEXT = "MANUFACTURING DIRECTIONS 1. Sift. 2. Blend 10 minutes. 3. Compress."
SELECTED = [make_evidence("BOM", BOM_TEXT, page=1), make_evidence("PROCESS", PROCESS_TEXT, page=2)]
CLASSIFICATION = {"report_title": "Test Report", "known_evidence_gaps": ["strength not stated"]}


def recording_llm(calls: list, reply: str):
    def llm_chat(messages):
        calls.append((messages[0]["content"], messages[1]["content"]))
        return reply
    return llm_chat


def test_sections_use_separate_calls_with_routed_evidence(tmp_path):
    calls: list = []
    reply = "Cited content [E1] [SOURCE: handbook.pdf, PDF page 1]"
    report, path = generate_report(recording_llm(calls, reply), "design report", SELECTED,
                                   CLASSIFICATION, tmp_path / "out.md")

    assert len(calls) == 6, "five content sections + one evidence-gap call"
    assert all(system == prompts.REPORT_SYSTEM_PROMPT for system, _ in calls)

    bom_msgs = [user for _, user in calls if "section 2. Bill of materials" in user]
    assert bom_msgs and BOM_TEXT in bom_msgs[0] and "Compress" not in bom_msgs[0]
    process_msgs = [user for _, user in calls if "section 3. Manufacturing process" in user]
    assert process_msgs and "[E2] ACETAMINOPHEN TABLETS | PROCESS" in process_msgs[0]

    assert Path(path).read_text(encoding="utf-8") == report
    for fragment in ("# Test Report", "## 1. Product & formulation identity",
                     "## 2. Bill of materials", "## 3. Manufacturing process",
                     "## 4. Equipment & process parameters", "## 5. Quality attributes",
                     "## 6. Evidence gaps", "## 7. Disclaimer", "educational draft"):
        assert fragment in report, fragment


def test_evidence_tags_keep_global_numbering(tmp_path):
    calls: list = []
    generate_report(recording_llm(calls, "body [E1]"), "design report", SELECTED,
                    CLASSIFICATION, tmp_path / "out.md")
    process_msgs = [user for _, user in calls if "section 3. Manufacturing process" in user]
    assert "[E1] ACETAMINOPHEN TABLETS | BOM" not in process_msgs[0], \
        "process section must not reuse tag [E1] for a different chunk"


def test_section_failure_keeps_the_report(tmp_path):
    def failing_llm(messages):
        raise RuntimeError("gateway timeout")

    report, path = generate_report(failing_llm, "design report", SELECTED,
                                   CLASSIFICATION, tmp_path / "out.md")
    assert Path(path).exists()
    assert "## 7. Disclaimer" in report
    assert "could not be generated" in report
    assert "strength not stated" in report
