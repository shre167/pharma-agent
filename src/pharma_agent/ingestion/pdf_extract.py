"""Two-column-aware PDF extraction using PyMuPDF word/span coordinates and fonts.

The handbook is a two-column PDF with bold ~11pt uppercase formulation titles.
Each column is extracted independently (left, then right) so columns never mix.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import pymupdf

from ..config import EXTRACTION_VERSION
from ..utils import normalize_space, sha256_file

ProgressFn = Optional[Callable[[int, int], None]]

_EXCLUDED_TITLE_FRAGMENTS = ("HANDBOOK OF PHARMACEUTICAL", "MANUFACTURING FORMULATIONS", "CONTENTS")


def extract_column_lines(page, x0: float, x1: float) -> list[dict]:
    """Extract text lines from one physical column, retaining coordinates/font metadata."""
    clip = pymupdf.Rect(x0, 0, x1, page.rect.height)
    blocks = page.get_text("dict", clip=clip).get("blocks", [])
    lines = []
    for block in blocks:
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            lines.append({
                "y": min(span["bbox"][1] for span in spans),
                "x": min(span["bbox"][0] for span in spans),
                "text": text,
                "max_size": max(span.get("size", 0) for span in spans),
                "bold": any("Bold" in span.get("font", "") for span in spans),
            })
    return sorted(lines, key=lambda z: (z["y"], z["x"]))


def _upper_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def is_formulation_title_line(line: dict) -> bool:
    text = normalize_space(line["text"])
    letters = [c for c in text if c.isalpha()]
    if not letters or len(text) < 7 or len(text) > 95:
        return False
    if not (line["bold"] and line["max_size"] >= 10.5 and _upper_ratio(text) >= 0.72):
        return False
    return not any(x in text.upper() for x in _EXCLUDED_TITLE_FRAGMENTS)


def detect_column_headings(lines: list[dict]) -> list[dict]:
    """Join consecutive bold uppercase title lines into one formulation title."""
    found, i = [], 0
    while i < len(lines):
        if not is_formulation_title_line(lines[i]):
            i += 1
            continue
        parts = [normalize_space(lines[i]["text"])]
        start_y = lines[i]["y"]
        last_y = start_y
        j = i + 1
        while j < len(lines):
            cand = lines[j]
            if cand["y"] - last_y > 19 or not cand["bold"] or cand["max_size"] < 10:
                break
            if _upper_ratio(cand["text"]) < 0.70:
                break
            parts.append(normalize_space(cand["text"]))
            last_y = cand["y"]
            j += 1
        found.append({"y": start_y, "title": normalize_space(" ".join(parts))})
        i = j
    return found


def extract_pdf_pages(pdf_path: Path, cache_dir: Path, progress: ProgressFn = None) -> tuple[list[dict], bool]:
    """Return (page_records, cache_hit). Cached on disk, keyed by PDF content hash."""
    pdf_path = Path(pdf_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = sha256_file(pdf_path)[:24]
    cache_path = cache_dir / f"pages_{key}_v{EXTRACTION_VERSION}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f), True

    page_records: list[dict] = []
    with pymupdf.open(pdf_path) as doc:
        total = len(doc)
        for idx, page in enumerate(doc):
            midpoint = page.rect.width / 2
            columns = []
            for name, x0, x1 in (("left", 0, midpoint), ("right", midpoint, page.rect.width)):
                lines = extract_column_lines(page, x0, x1)
                columns.append({
                    "column": name,
                    "lines": lines,
                    "headings": detect_column_headings(lines),
                    "text": "\n".join(line["text"] for line in lines),
                })
            page_records.append({"page": idx + 1, "columns": columns})
            if progress:
                progress(idx + 1, total)
    with open(cache_path, "wb") as f:
        pickle.dump(page_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    return page_records, False
