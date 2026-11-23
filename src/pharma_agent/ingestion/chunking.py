"""Split each formulation into identity, BOM and process evidence chunks.

The BOM and process text are preserved verbatim. Nothing is normalised, calculated
or invented. Every chunk carries formulation instance ID, heading and page provenance.
"""
from __future__ import annotations

import re

from ..config import MAX_PROCESS_STEPS_PER_CHUNK
from ..utils import normalize_space, stable_id
from .terms import detect_dosage_form, ingredient_phrase


def split_labeled_sections(text: str) -> tuple[str, str]:
    lines = text.splitlines()
    bom_start = next((i for i, x in enumerate(lines) if re.search(r"^\s*Bill of Materials\s*$", x, re.I)), None)
    proc_start = next((i for i, x in enumerate(lines) if re.search(r"^\s*Manufacturing Directions\s*$", x, re.I)), None)
    bom_text = process_text = ""
    if bom_start is not None:
        end = proc_start if proc_start is not None and proc_start > bom_start else len(lines)
        bom_text = "\n".join(lines[bom_start + 1:end]).strip()
    if proc_start is not None:
        process_text = "\n".join(lines[proc_start + 1:]).strip()
    return bom_text, process_text


_STEP_START = re.compile(r"^(?:\d{1,2}[.)]|\([a-z]\)[.)]?)\s*", re.I)


def numbered_steps(process_text: str) -> list[str]:
    """Reconstruct numbered steps; wrapped lines stay with their step."""
    steps: list[str] = []
    current: list[str] = []
    for raw in process_text.splitlines():
        line = normalize_space(raw)
        if not line:
            continue
        if _STEP_START.match(line):
            if current:
                steps.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
        else:
            current = [line]  # preserve preamble / cross-reference text
    if current:
        steps.append(" ".join(current))
    return steps


def build_chunks(sections: list[dict], source: str) -> tuple[list[dict], list[dict]]:
    """Return (chunks, audit_rows). audit_rows double as the manifest.csv rows."""
    chunks: list[dict] = []
    audit: list[dict] = []

    for sec in sections:
        heading = sec["heading"]
        bom_text, process_text = split_labeled_sections(sec["text"])
        pages_text = ", ".join(map(str, sec["pages"]))
        dosage_form = detect_dosage_form(heading)
        active = ingredient_phrase(heading) or "NOT_NORMALIZED"
        meta = {
            "source": source,
            "page": int(sec["start_page"]),
            "pages": pages_text,
            "page_columns": " | ".join(sec["page_columns"]),
            "formulation_id": sec["id"],
            "formulation_name": heading,
            "active_ingredient": active,
            "strength": "UNKNOWN_UNLESS_EXPLICIT_IN_HEADING",
            "dosage_form": dosage_form,
            "release_type": "UNKNOWN_UNLESS_EXPLICIT_IN_HEADING",
        }
        identity_text = (
            f"FORMULATION INSTANCE: {sec['id']}\nFORMULATION HEADING: {heading}\n"
            f"ACTIVE INGREDIENT TERMS IN HEADING: {active}\n"
            f"DOSAGE FORM TERM IN HEADING: {dosage_form}\nSOURCE PAGES: {pages_text}\n"
            f"SOURCE PAGE/COLUMNS: {meta['page_columns']}\n\n{sec['text']}"
        )
        chunks.append({"id": stable_id(sec["id"], "IDENTITY"), "text": identity_text,
                       "type": "FORMULATION", "metadata": {**meta, "chunk_type": "FORMULATION"}})

        if bom_text:
            chunks.append({
                "id": stable_id(sec["id"], "BOM"),
                "text": (f"BILL OF MATERIALS (verbatim extracted text)\nFORMULATION INSTANCE: {sec['id']}\n"
                         f"FORMULATION: {heading}\nSOURCE PAGES: {pages_text}\n\n{bom_text}"),
                "type": "BOM",
                "metadata": {**meta, "chunk_type": "BOM", "bom_extraction": "RAW_TEXT_REQUIRES_VISUAL_QA"},
            })

        if process_text:
            steps = numbered_steps(process_text) or [process_text]
            n = MAX_PROCESS_STEPS_PER_CHUNK
            for start in range(0, len(steps), n):
                group = steps[start:start + n]
                chunks.append({
                    "id": stable_id(sec["id"], "PROCESS", start + 1),
                    "text": (f"MANUFACTURING PROCESS EVIDENCE\nFORMULATION INSTANCE: {sec['id']}\n"
                             f"FORMULATION: {heading}\nSOURCE PAGES: {pages_text}\n"
                             f"STEP GROUP: {start + 1}-{start + len(group)}\n\n" + "\n".join(group)),
                    "type": "PROCESS",
                    "metadata": {**meta, "chunk_type": "PROCESS", "step_group_start": int(start + 1)},
                })

        audit.append({
            "formulation_id": sec["id"], "source": source, "formulation_name": heading,
            "active_ingredient": active, "dosage_form": dosage_form, "pages": pages_text,
            "has_bom": bool(bom_text), "has_process": bool(process_text),
            "bom_chars": len(bom_text), "process_chars": len(process_text),
        })
    return chunks, audit
