"""Reconstruct formulation sections across columns and pages.

Every heading occurrence gets a unique formulation instance ID, even when the printed
heading repeats. A new heading closes the previous section. Text before the first
heading is kept as unassigned (never guessed).
"""
from __future__ import annotations

from ..utils import stable_id


def reconstruct_sections(page_records: list[dict], source: str) -> tuple[list[dict], list[dict]]:
    sections: list[dict] = []
    unassigned: list[dict] = []
    active: dict | None = None

    def close_active() -> None:
        nonlocal active
        if active is None:
            return
        active["text"] = "\n".join(active.pop("_text_parts"))
        active["pages"] = sorted(set(active["pages"]))
        active["id"] = stable_id(source, active["heading"], active["start_page"],
                                 active["start_column"], len(sections))
        sections.append(active)
        active = None

    def attach(text: str, page: int, column: str) -> None:
        if active is not None:
            active["_text_parts"].append(text)
            active["pages"].append(page)
            active["page_columns"].append(f"{page}:{column}")
        else:
            unassigned.append({"page": page, "column": column, "text": text})

    for page in page_records:
        for col in page["columns"]:
            lines = col["lines"]
            positions = []
            for h in col["headings"]:
                idx = min(range(len(lines)), key=lambda k: abs(lines[k]["y"] - h["y"]))
                positions.append((idx, h["title"]))
            positions.sort()

            if not positions:
                block = "\n".join(x["text"] for x in lines).strip()
                if block:
                    attach(block, page["page"], col["column"])
                continue

            prefix = "\n".join(x["text"] for x in lines[:positions[0][0]]).strip()
            if prefix:
                attach(prefix, page["page"], col["column"])

            for pos, (line_idx, heading) in enumerate(positions):
                end_idx = positions[pos + 1][0] if pos + 1 < len(positions) else len(lines)
                body = "\n".join(x["text"] for x in lines[line_idx + 1:end_idx]).strip()
                close_active()
                active = {
                    "heading": heading,
                    "start_page": page["page"],
                    "start_column": col["column"],
                    "pages": [page["page"]],
                    "page_columns": [f"{page['page']}:{col['column']}"],
                    "_text_parts": [body] if body else [],
                }
    close_active()
    return sections, unassigned
