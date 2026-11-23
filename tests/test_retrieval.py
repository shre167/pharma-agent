def test_rrf_fuses_dense_and_bm25(small_kb):
    # Dense ranking puts ibuprofen first; BM25 for "acetaminophen" favours c1/c2.
    results = small_kb.retrieve("acetaminophen bill of materials", top_k=4)
    ids = [r["metadata"]["formulation_name"] + "|" + r["metadata"]["chunk_type"] for r in results]
    assert ids[0] == "ACETAMINOPHEN TABLETS|BOM"      # agreed by both retrievers
    assert len(results) == 4 and all(r["rrf_score"] > 0 for r in results)
    assert results[0]["citation"] == "[SOURCE: handbook.pdf, PDF page 1]"


def test_chunk_type_and_formulation_filters(small_kb):
    only_bom = small_kb.retrieve("tablets", top_k=10, chunk_types=["BOM"])
    assert {r["metadata"]["chunk_type"] for r in only_bom} == {"BOM"}
    only_ibu = small_kb.retrieve("tablets", top_k=10, formulation_hint="ibuprofen")
    assert {r["metadata"]["formulation_name"] for r in only_ibu} == {"IBUPROFEN TABLETS"}


def test_vocab_is_built_from_headings(small_kb):
    assert set(small_kb.vocab) == {"acetaminophen", "ibuprofen"}
    assert small_kb.handbook_matches(["acetaminophen"]) == ["ACETAMINOPHEN TABLETS"]
    assert small_kb.handbook_matches(["nonexistent"]) == []


def test_manifest_frame(small_kb):
    frame = small_kb.manifest_frame()
    assert len(frame) == 2 and frame["has_bom"].all() and frame["has_process"].all()
