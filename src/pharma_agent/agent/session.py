"""Per-request agent state. Replaces the notebook's module-level globals so concurrent
Streamlit sessions cannot see each other's evidence, gate state or captured report."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from ..retrieval.knowledge_base import KnowledgeBase
from ..utils import stable_id

ReportWriter = Callable[[str, list, dict, Path], tuple]


@dataclass
class AgentSession:
    kb: KnowledgeBase
    report_writer: ReportWriter
    wave1_dir: Path
    allowed: bool = False
    generic_names: list[str] = field(default_factory=list)
    evidence: dict[str, dict] = field(default_factory=dict)
    capture: Optional[dict] = None   # {"text", "path", "selected"} of the generated report
    on_event: Optional[Callable[[str, dict], None]] = None

    # ---- find_documents ------------------------------------------------
    def find_documents(self, search_query: str, top_k: int = 5, focus: str = "all",
                       alternate_names: Optional[List[str]] = None) -> dict:
        if not self.allowed:
            return {"error": "Medicine scope gate has not authorised retrieval.", "results": []}
        if not isinstance(search_query, str) or not search_query.strip():
            return {"error": "search_query must be a non-empty string", "results": []}
        top_k = max(1, min(int(top_k or 5), 10))
        chunk_types = None if focus in (None, "all") else [focus]
        alt = [n.strip() for n in (alternate_names or []) if isinstance(n, str) and n.strip()]
        found = self.kb.retrieve(" ".join([search_query.strip()] + alt), top_k=top_k, chunk_types=chunk_types)
        results = []
        for item in found:
            m = item["metadata"]
            evidence_id = stable_id(m.get("formulation_id", ""), m.get("chunk_type", ""),
                                    m.get("source", ""), m.get("page", ""), item["text"][:300])
            stored = dict(item)
            stored["evidence_id"] = evidence_id
            self.evidence[evidence_id] = stored
            results.append({
                "evidence_id": evidence_id,
                "formulation_id": m.get("formulation_id"),
                "formulation_name": m.get("formulation_name"),
                "chunk_type": m.get("chunk_type"),
                "active_ingredient": m.get("active_ingredient"),
                "strength": m.get("strength"),
                "dosage_form": m.get("dosage_form"),
                "release_type": m.get("release_type"),
                "source": m.get("source"), "page": m.get("page"), "pages": m.get("pages"),
                "rrf_score": round(float(item.get("rrf_score", 0.0)), 6),
                "text": item["text"][:1800],
            })
        return {"query": search_query, "alternate_names": alt, "result_count": len(results), "results": results}

    # ---- build_report --------------------------------------------------
    def build_report(self, report_request: str, evidence_ids: List[str],
                     report_title: Optional[str] = None, known_gaps: Optional[List[str]] = None) -> dict:
        if not self.allowed:
            return {"error": "Medicine scope gate has not authorised report generation."}
        selected, unknown, seen = [], [], set()
        for evidence_id in evidence_ids or []:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            item = self.evidence.get(evidence_id)
            if item is None:
                unknown.append(evidence_id)
            else:
                selected.append(item)
        if not selected:
            return {"error": "No valid retrieved evidence IDs were supplied. Search first; no report was generated."}

        # Keep the most relevant chunks (max 5) to prevent context explosion and API timeouts
        if len(selected) > 5:
            selected = sorted(selected, key=lambda e: float(e.get("rrf_score", 0.0)), reverse=True)[:5]

        formulation_ids = {e["metadata"].get("formulation_id", "") for e in selected}
        classification = {
            "agent_interpretation": report_request,
            "known_evidence_gaps": known_gaps or [],
            "report_title": report_title or "Pharmaceutical Manufacturing Design Report",
        }
        if self.on_event:
            self.on_event("report_start", {
                "evidence_count": len(selected),
                "title": classification["report_title"],
            })
        stub = re.sub(r"[^A-Za-z0-9_-]+", "_", report_title or "pharma_design_report").strip("_")[:60] or "report"
        output_path = self.wave1_dir / f"{stub}_{time.strftime('%Y%m%d_%H%M%S')}.md"
        report, saved_path = self.report_writer(report_request, selected, classification, output_path)
        self.capture = {"text": report, "path": saved_path, "selected": selected}
        return {
            "status": "REPORT_GENERATED", "path": saved_path,
            "selected_evidence_count": len(selected),
            "formulation_ids": sorted(formulation_ids), "unknown_evidence_ids": unknown,
            "report_preview": report[:10000],
        }
