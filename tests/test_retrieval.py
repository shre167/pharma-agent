import json

from pharma_agent.retrieval.knowledge_base import KnowledgeBase

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


def test_retrieve_works_without_a_vectorstore():
    chunks = [
        {"id": "c1", "text": "BILL OF MATERIALS acetaminophen 500 mg starch", "type": "BOM",
         "metadata": {"source": "h.pdf", "page": 1, "pages": "1", "formulation_id": "f1",
                      "formulation_name": "ACETAMINOPHEN TABLETS", "chunk_type": "BOM",
                      "active_ingredient": "x", "dosage_form": "tablets"}},
    ]
    kb = KnowledgeBase(chunks, None)
    assert kb.index_error is None
    results = kb.retrieve("acetaminophen bill of materials", top_k=2)
    assert len(results) == 1 and results[0]["rrf_score"] > 0


def test_load_falls_back_to_bm25_when_the_index_is_stale(settings, small_kb, monkeypatch):
    import pharma_agent.retrieval.knowledge_base as kbmod

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(small_kb.chunks[0], ensure_ascii=False) + "\n")

    def stale(_settings, _chunks, _embeddings):
        raise kbmod.IndexOutOfDate("Index was built with x but EMBEDDING_MODEL is y")

    monkeypatch.setattr(kbmod, "load_index", stale)
    kb = kbmod.KnowledgeBase.load(settings, None)
    assert kb.vectorstore is None and "Index was built with x" in kb.index_error
    assert kb.retrieve("acetaminophen", top_k=2)
