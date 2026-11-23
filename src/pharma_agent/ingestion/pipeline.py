"""End-to-end ingestion: docs/*.pdf -> chunks.jsonl + manifest.csv (+ FAISS index)."""
from __future__ import annotations

import json
from typing import Callable, Optional

import pandas as pd

from ..config import Settings
from .chunking import build_chunks
from .pdf_extract import extract_pdf_pages
from .sections import reconstruct_sections

Log = Callable[[str], None]


def build_all_chunks(settings: Settings, log: Log = print) -> tuple[list[dict], list[dict]]:
    """Extract every PDF in docs/ (API-free). Returns (chunks, audit rows)."""
    pdfs = sorted(settings.docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {settings.docs_dir}. Copy the handbook PDF there.")
    all_chunks: list[dict] = []
    all_audit: list[dict] = []
    for pdf in pdfs:
        log(f"Extracting {pdf.name} ...")
        def progress(i: int, n: int) -> None:
            if i % 25 == 0 or i == n:
                log(f"  page {i}/{n}")

        pages, hit = extract_pdf_pages(pdf, settings.cache_dir, progress=progress)
        log(f"  {'loaded cached extraction' if hit else 'extracted'}: {len(pages)} pages")
        sections, unassigned = reconstruct_sections(pages, pdf.name)
        chunks, audit = build_chunks(sections, pdf.name)
        log(f"  {len(sections)} formulation instances, {len(chunks)} chunks, "
            f"{len(unassigned)} unassigned blocks (not guessed)")
        all_chunks.extend(chunks)
        all_audit.extend(audit)
    return all_chunks, all_audit


def validate_chunks(chunks: list[dict], audit: list[dict]) -> dict:
    """Mandatory sanity checks before embedding. Raises AssertionError on a hard failure."""
    frame = pd.DataFrame(audit)
    summary = {
        "formulations": int(len(frame)),
        "with_bom": int(frame["has_bom"].sum()) if len(frame) else 0,
        "with_process": int(frame["has_process"].sum()) if len(frame) else 0,
        "missing_bom": int((~frame["has_bom"]).sum()) if len(frame) else 0,
        "missing_process": int((~frame["has_process"]).sum()) if len(frame) else 0,
        "repeated_headings": int(frame.duplicated("formulation_name", keep=False).sum()) if len(frame) else 0,
        "chunks": len(chunks),
    }
    bad = [c for c in chunks if c["type"] == "BOM"
           and "syrup" in c["metadata"]["formulation_name"].lower()
           and "tablets" in c["metadata"]["formulation_name"].lower()]
    assert not bad, "Potential cross-formulation contamination detected."
    assert len({c["id"] for c in chunks}) == len(chunks), "Chunk IDs are not unique."
    return summary


def write_artifacts(settings: Settings, chunks: list[dict], audit: list[dict]) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    pd.DataFrame(audit).to_csv(settings.manifest_path, index=False)


def run_ingestion(settings: Settings, services=None, rebuild: bool = False,
                  chunks_only: bool = False, log: Log = print) -> dict:
    """Full ingestion. `chunks_only=True` skips embedding (no API key needed)."""
    chunks, audit = build_all_chunks(settings, log)
    summary = validate_chunks(chunks, audit)
    write_artifacts(settings, chunks, audit)
    log(f"Wrote {settings.chunks_path.name} and {settings.manifest_path.name}")
    summary["indexed"] = False
    if chunks_only:
        return summary
    if services is None:
        raise ValueError("An embeddings-capable Services object is required unless chunks_only=True.")
    from ..retrieval.index import build_or_load_index
    log("Building / loading FAISS index ...")
    _, rebuilt = build_or_load_index(settings, chunks, services.embeddings, force=rebuild)
    log("FAISS index rebuilt and saved." if rebuilt else "Existing FAISS index is up to date; nothing re-embedded.")
    summary["indexed"] = True
    summary["rebuilt"] = rebuilt
    return summary
