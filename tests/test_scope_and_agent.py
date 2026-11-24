import json

from pharma_agent.agent.runner import strip_followups
from pharma_agent.agent.scope import check_scope
from pharma_agent.agent.session import AgentSession
from pharma_agent.agent.tools import make_tools

VOCAB = {"acetaminophen": {"ACETAMINOPHEN TABLETS"}, "ibuprofen": {"IBUPROFEN TABLETS"}}


def fake_llm(payload):
    return lambda messages: json.dumps(payload)


def test_brand_name_resolves_to_handbook_generic():
    llm = fake_llm({"is_medicine_request": True, "medicine_mentions": ["Crocin"],
                    "matched_handbook_ingredients": ["acetaminophen"], "identification_method": "explicit_name",
                    "confidence": "high"})
    scope = check_scope(VOCAB, llm, "manufacturing process for crocin tablets")
    assert scope["generic_names"] == ["acetaminophen"] and scope["is_medicine_request"]


def test_brand_mapped_by_llm_uses_handbook_spelling_and_fails_closed_otherwise():
    PARACETAMOL_VOCAB = {"paracetamol": {"PARACETAMOL TABLETS"}, "ibuprofen": {"IBUPROFEN TABLETS"}}
    copies_handbook = fake_llm({"medicine_mentions": ["Dolo"],
                                "matched_handbook_ingredients": ["paracetamol"],
                                "active_ingredients_identified": ["paracetamol"],
                                "identification_method": "explicit_name", "confidence": "high"})
    scope = check_scope(PARACETAMOL_VOCAB, copies_handbook, "report for dolo 650")
    assert scope["generic_names"] == ["paracetamol"] and scope["is_medicine_request"]

    echoes_wrong_spelling = fake_llm({"medicine_mentions": ["Dolo"],
                                      "matched_handbook_ingredients": ["acetaminophen"],
                                      "identification_method": "explicit_name", "confidence": "high"})
    scope = check_scope(PARACETAMOL_VOCAB, echoes_wrong_spelling, "report for dolo 650")
    assert scope["generic_names"] == [] and not scope["is_medicine_request"]


def test_generic_term_mapped_by_llm_to_handbook_chemical():
    llm = fake_llm({"medicine_mentions": ["paracetamol"],
                    "matched_handbook_ingredients": ["acetaminophen"], "identification_method": "explicit_name",
                    "confidence": "high"})
    scope = check_scope(VOCAB, llm, "design report for paracetamol 500 mg immediate-release tablets")
    assert scope["generic_names"] == ["acetaminophen"] and scope["is_medicine_request"]


def test_invented_ingredient_never_opens_the_gate():
    llm = fake_llm({"medicine_mentions": ["Zorbex"], "matched_handbook_ingredients": ["zorbexinol"],
                    "identification_method": "explicit_name", "confidence": "high"})
    scope = check_scope(VOCAB, llm, "process for zorbex")
    assert scope["generic_names"] == [] and not scope["is_medicine_request"]


def test_explicit_name_must_appear_in_the_query():
    # LLM "hallucinates" a mention the user never wrote -> must not authorise retrieval.
    llm = fake_llm({"medicine_mentions": ["ibuprofen"], "matched_handbook_ingredients": ["ibuprofen"],
                    "identification_method": "explicit_name", "confidence": "high"})
    assert check_scope(VOCAB, llm, "make me a pain tablet report")["generic_names"] == []


def test_description_supports_matches_and_candidates():
    ok = fake_llm({"matched_handbook_ingredients": ["acetaminophen"],
                   "identification_method": "description", "confidence": "high"})
    scope = check_scope(VOCAB, ok, "non-opioid analgesic antipyretic tablet")
    assert scope["generic_names"] == ["acetaminophen"] and scope["is_medicine_request"]


def test_substantive_descriptions_run_even_when_several_variants_match():
    medium = fake_llm({"matched_handbook_ingredients": ["acetaminophen"],
                       "identification_method": "description", "confidence": "medium"})
    scope = check_scope(VOCAB, medium, "report for a wet-granulated immediate-release fever tablet")
    assert scope["generic_names"] == ["acetaminophen"] and scope["is_medicine_request"]

    multi = fake_llm({"matched_handbook_ingredients": ["acetaminophen", "ibuprofen"],
                      "identification_method": "description", "confidence": "medium"})
    scope = check_scope(VOCAB, multi, "report for an immediate-release analgesic tablet")
    assert scope["generic_names"] == ["acetaminophen", "ibuprofen"] and scope["is_medicine_request"]


def test_vague_descriptions_fail_closed_and_report_candidates():
    vague = fake_llm({"matched_handbook_ingredients": [],
                      "candidate_ingredients": ["acetaminophen", "ibuprofen"],
                      "identification_method": "description", "confidence": "low"})
    scope = check_scope(VOCAB, vague, "prepare a report for a white round fever tablet")
    assert scope["generic_names"] == [] and not scope["is_medicine_request"]
    assert "too vague" in scope["reason"] and "ibuprofen" in scope["reason"]

    low = fake_llm({"matched_handbook_ingredients": ["acetaminophen"],
                    "identification_method": "description", "confidence": "low"})
    scope = check_scope(VOCAB, low, "pain tablet")
    assert scope["generic_names"] == [] and not scope["is_medicine_request"]


def test_invalid_classifier_json_fails_closed():
    scope = check_scope(VOCAB, lambda m: "not json at all", "acetaminophen tablets")
    assert scope["generic_names"] == [] and "invalid JSON" in scope["reason"]


def test_strip_followups_removes_offers_and_questions():
    text = "Report created.\n\n- Item\n\nIf you want, I can also compare variants.\n\nAnything else?"
    assert strip_followups(text) == "Report created.\n\n- Item"


def _session(small_kb, allowed=True):
    calls = []

    def writer(request, selected, classification, path, progress=None):
        calls.append((request, len(selected)))
        return "REPORT [E1] [SOURCE: handbook.pdf, PDF page 1]", str(path)

    s = AgentSession(kb=small_kb, report_writer=writer, wave1_dir=small_kb and __import__("pathlib").Path("/tmp"))
    s.allowed = allowed
    return s, calls


def test_tools_blocked_until_gate_authorises(small_kb):
    s, _ = _session(small_kb, allowed=False)
    assert "not authorised" in s.find_documents("acetaminophen")["error"]
    assert "not authorised" in s.build_report("r", ["x"])["error"]


def test_find_then_build_uses_only_retrieved_evidence_ids(small_kb):
    s, calls = _session(small_kb)
    found = s.find_documents("acetaminophen bom", top_k=3, alternate_names=["paracetamol"])
    assert found["result_count"] == 3
    ids = [r["evidence_id"] for r in found["results"]]

    bad = s.build_report("report", ["fabricated-id"])
    assert "error" in bad and s.capture is None

    ok = s.build_report("report", ids[:2] + ["fabricated-id"], report_title="My Report", known_gaps=["strength"])
    assert ok["status"] == "REPORT_GENERATED" and ok["unknown_evidence_ids"] == ["fabricated-id"]
    assert ok["selected_evidence_count"] == 2 and calls == [("report", 2)]
    assert s.capture["text"].startswith("REPORT")


def test_unknown_focus_does_not_blind_the_search(small_kb):
    s, _ = _session(small_kb)
    found = s.find_documents("acetaminophen", focus="formulation chart")
    assert found["result_count"] > 0
    assert "not a valid evidence type" in found["note"]


def test_focus_is_case_insensitive_and_falls_back_when_empty(small_kb):
    s, _ = _session(small_kb)
    lower = s.find_documents("acetaminophen", focus="bom")
    assert lower["result_count"] == 2
    assert {r["chunk_type"] for r in lower["results"]} == {"BOM"}
    assert "note" not in lower

    empty = s.find_documents("acetaminophen", focus="FORMULATION")
    assert empty["result_count"] > 0
    assert "no FORMULATION chunks matched" in empty["note"]


def test_langchain_tools_are_exactly_two_and_return_json(small_kb):
    s, _ = _session(small_kb)
    tools = make_tools(s)
    assert [t.name for t in tools] == ["find_documents", "build_report"]
    out = json.loads(tools[0].invoke({"search_query": "acetaminophen process", "top_k": 2}))
    assert out["result_count"] == 2
