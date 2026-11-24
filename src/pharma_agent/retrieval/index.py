"""Build / load the persistent FAISS index with a content fingerprint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from ..config import CHUNKING_VERSION, INDEX_VERSION, MAX_PROCESS_STEPS_PER_CHUNK, Settings
from ..utils import sha256_file

_META_NAME = "index_metadata.json"


class IndexOutOfDate(RuntimeError):
    """The saved FAISS index does not match the current chunks / embedding model."""


def fingerprint(settings: Settings, chunks: list[dict]) -> dict:
    pdfs = sorted(settings.docs_dir.glob("*.pdf"))
    chunk_hash = hashlib.sha256(json.dumps(
        [{"id": c["id"], "text": c["text"], "metadata": c["metadata"]} for c in chunks],
        sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    return {
        "pdf_sha256": {p.name: sha256_file(p) for p in pdfs},
        "chunks_sha256": chunk_hash,
        "embedding_model": settings.embedding_model,
        "embedding_endpoint": settings.embedding_base_url,
        "chunk_count": len(chunks),
        "chunking_config": {"version": CHUNKING_VERSION,
                            "max_process_steps_per_chunk": MAX_PROCESS_STEPS_PER_CHUNK},
        "index_version": INDEX_VERSION,
    }


def _faiss_metadatas(chunks: list[dict]) -> list[dict]:
    out = []
    for c in chunks:
        m = dict(c["metadata"])
        m["chunk_id"] = c["id"]
        m["chunk_type"] = c["metadata"].get("chunk_type", c.get("type", "UNKNOWN"))
        out.append(m)
    return out


def build_or_load_index(settings: Settings, chunks: list[dict], embeddings, force: bool = False):
    """Reuse the saved index if the fingerprint matches; otherwise (re)build and save.

    Returns (vectorstore, rebuilt: bool).
    """
    if not chunks:
        raise ValueError("No chunks were created. Inspect PDF extraction before indexing.")
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_path = index_dir / _META_NAME
    current = fingerprint(settings, chunks)

    files_exist = (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()
    saved = None
    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            saved = None

    if files_exist and saved == current and not force:
        # The FAISS docstore is a pickle: only load indexes you created and trust.
        store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
        return store, False

    store = FAISS.from_texts(texts=[c["text"] for c in chunks], embedding=embeddings,
                             metadatas=_faiss_metadatas(chunks), ids=[c["id"] for c in chunks])
    store.save_local(str(index_dir))
    meta_path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    return store, True


def load_index(settings: Settings, chunks: list[dict], embeddings):
    """Runtime loader: never embeds. Raises IndexOutOfDate with a clear fix."""
    index_dir = settings.index_dir
    meta_path = index_dir / _META_NAME
    if not ((index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists() and meta_path.exists()):
        raise IndexOutOfDate("FAISS index not found. Run: python scripts/ingest.py")
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    if saved.get("embedding_model") != settings.embedding_model:
        raise IndexOutOfDate(
            f"Index was built with {saved.get('embedding_model')} but EMBEDDING_MODEL is "
            f"{settings.embedding_model}. Run: python scripts/ingest.py --rebuild")
    if saved.get("chunk_count") != len(chunks):
        raise IndexOutOfDate("Index and chunks.jsonl disagree. Run: python scripts/ingest.py --rebuild")
    store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    if store.index.ntotal != len(chunks):
        raise IndexOutOfDate("FAISS vector count does not match chunk count. Run: python scripts/ingest.py --rebuild")
    return store
