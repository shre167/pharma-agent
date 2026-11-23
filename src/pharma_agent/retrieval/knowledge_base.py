"""KnowledgeBase: hybrid FAISS + BM25 retrieval fused by RRF, plus the ingredient vocabulary."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from rank_bm25 import BM25Okapi

from ..config import Settings
from ..ingestion.terms import ingredient_phrase
from .index import load_index


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(path: Path) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/ingest.py")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class KnowledgeBase:
    def __init__(self, chunks: list[dict], vectorstore: Any, rrf_k: int = 60):
        self.chunks = chunks
        self.vectorstore = vectorstore
        self.rrf_k = rrf_k
        self.chunk_by_id = {c["id"]: c for c in chunks}
        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.vocab = self._build_vocab()

    @classmethod
    def load(cls, settings: Settings, embeddings) -> "KnowledgeBase":
        chunks = load_chunks(settings.chunks_path)
        return cls(chunks, load_index(settings, chunks, embeddings))

    # ---- vocabulary: built from handbook headings, no manual dictionary -------
    def _build_vocab(self) -> dict[str, set[str]]:
        vocab: dict[str, set[str]] = {}
        for c in self.chunks:
            heading = c["metadata"]["formulation_name"]
            phrase = ingredient_phrase(heading)
            if phrase:
                vocab.setdefault(phrase, set()).add(heading)
        return vocab

    def handbook_matches(self, ingredients: list[str]) -> list[str]:
        out: set[str] = set()
        for ing in ingredients:
            out |= self.vocab.get(ing, set())
        return sorted(out)

    # ---- retrieval -----------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 12, formulation_hint: Optional[str] = None,
                 chunk_types: Optional[list[str]] = None) -> list[dict]:
        """Hybrid retrieval. Fuses ranks (not raw scores) so FAISS and BM25 stay comparable."""
        candidate_k = min(max(top_k * 3, top_k), len(self.chunks))
        dense_docs = self.vectorstore.similarity_search(query, k=candidate_k)
        dense_ids = [d.metadata.get("chunk_id") for d in dense_docs
                     if d.metadata.get("chunk_id") in self.chunk_by_id]

        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_k]
        bm25_ids = [self.chunks[i]["id"] for i in order]

        rrf: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(dense_ids, start=1):
            rrf[cid] += 1.0 / (self.rrf_k + rank)
        for rank, cid in enumerate(bm25_ids, start=1):
            rrf[cid] += 1.0 / (self.rrf_k + rank)

        evidence = []
        for cid in sorted(rrf, key=rrf.get, reverse=True):
            c = self.chunk_by_id[cid]
            meta = c["metadata"]
            if chunk_types and meta.get("chunk_type") not in chunk_types:
                continue
            if formulation_hint and formulation_hint.lower() not in meta.get("formulation_name", "").lower():
                continue
            evidence.append({
                "text": c["text"], "metadata": meta, "rrf_score": float(rrf[cid]),
                "citation": f"[SOURCE: {meta.get('source')}, PDF page {meta.get('page')}]",
            })
            if len(evidence) >= top_k:
                break
        return evidence

    # ---- inspection helpers (used by the Knowledge Base page) ----------------
    def manifest_frame(self) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for c in self.chunks:
            m = c["metadata"]
            fid = m["formulation_id"]
            row = rows.setdefault(fid, {
                "formulation_id": fid, "source": m.get("source"),
                "formulation_name": m.get("formulation_name"),
                "active_ingredient": m.get("active_ingredient"),
                "dosage_form": m.get("dosage_form"), "pages": m.get("pages"),
                "has_bom": False, "has_process": False,
            })
            if m.get("chunk_type") == "BOM":
                row["has_bom"] = True
            if m.get("chunk_type") == "PROCESS":
                row["has_process"] = True
        return pd.DataFrame(rows.values())

    def inspect(self, chunk_type: Optional[str] = None, keyword: Optional[str] = None, limit: int = 10) -> list[dict]:
        selected = self.chunks
        if chunk_type:
            selected = [c for c in selected if c["type"] == chunk_type]
        if keyword:
            selected = [c for c in selected if keyword.lower() in c["text"].lower()]
        return selected[:limit]
