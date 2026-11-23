from pharma_agent.ingestion.chunking import build_chunks, numbered_steps, split_labeled_sections
from pharma_agent.ingestion.pdf_extract import extract_pdf_pages
from pharma_agent.ingestion.pipeline import validate_chunks
from pharma_agent.ingestion.sections import reconstruct_sections
from pharma_agent.ingestion.terms import detect_dosage_form, ingredient_phrase


def test_ingredient_phrase_and_form():
    assert ingredient_phrase("ACETAMINOPHEN CHEWABLE TABLETS") == "acetaminophen"
    assert ingredient_phrase("IBUPROFEN SUSTAINED RELEASE TABLETS") == "ibuprofen"
    assert detect_dosage_form("PARACETAMOL SYRUP") == "syrup"
    assert detect_dosage_form("MYSTERY STUFF") == "UNKNOWN"


def test_numbered_steps_keeps_wrapped_lines_together():
    text = "1. Sift the powder\nthrough 40 mesh.\n2. Blend 10 min.\n(a) Dry at 60 C"
    steps = numbered_steps(text)
    assert steps == ["1. Sift the powder through 40 mesh.", "2. Blend 10 min.", "(a) Dry at 60 C"]


def test_split_labeled_sections():
    bom, proc = split_labeled_sections("Bill of Materials\nA 1 mg\nManufacturing Directions\n1. Mix")
    assert bom == "A 1 mg" and proc == "1. Mix"
    assert split_labeled_sections("no markers here") == ("", "")


def test_two_column_extraction_end_to_end(synthetic_pdf, tmp_path):
    cache = tmp_path / "cache"
    pages, hit = extract_pdf_pages(synthetic_pdf, cache)
    assert not hit and len(pages) == 2
    _, hit_again = extract_pdf_pages(synthetic_pdf, cache)
    assert hit_again, "second extraction must come from the content-hash cache"

    sections, unassigned = reconstruct_sections(pages, "handbook.pdf")
    assert [s["heading"] for s in sections] == ["ACETAMINOPHEN TABLETS", "IBUPROFEN TABLETS"]
    assert unassigned == []
    # Ibuprofen continues onto page 2 (no new heading) and must keep its own provenance.
    assert sections[1]["pages"] == [1, 2]
    assert "Compress into tablets" in sections[1]["text"]
    # Columns never mix: no ibuprofen text inside acetaminophen.
    assert "Ibuprofen" not in sections[0]["text"]

    chunks, audit = build_chunks(sections, "handbook.pdf")
    by_type = {}
    for c in chunks:
        by_type.setdefault(c["type"], []).append(c)
    assert len(by_type["FORMULATION"]) == 2 and len(by_type["BOM"]) == 2 and len(by_type["PROCESS"]) == 2
    apap_bom = next(c for c in by_type["BOM"] if "ACETAMINOPHEN" in c["metadata"]["formulation_name"])
    assert "Acetaminophen 500 mg" in apap_bom["text"] and "Ibuprofen" not in apap_bom["text"]
    assert all(a["has_bom"] and a["has_process"] for a in audit)
    assert validate_chunks(chunks, audit)["formulations"] == 2


def test_repeated_headings_get_distinct_ids():
    pages = []
    for p in (1, 2):
        pages.append({"page": p, "columns": [{
            "column": "left", "text": "",
            "lines": [{"y": 10, "x": 0, "text": "ACETAMINOPHEN TABLETS", "max_size": 11, "bold": True},
                      {"y": 30, "x": 0, "text": "Bill of Materials", "max_size": 9, "bold": False}],
            "headings": [{"y": 10, "title": "ACETAMINOPHEN TABLETS"}]},
            {"column": "right", "lines": [], "headings": [], "text": ""}]})
    sections, _ = reconstruct_sections(pages, "h.pdf")
    assert len(sections) == 2 and sections[0]["id"] != sections[1]["id"]
