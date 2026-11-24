"""Cited Markdown report generation from selected evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from langsmith import traceable

from ..observability import trace_inputs, trace_output
from .prompts import REPORT_SYSTEM_PROMPT


@traceable(name="wave1_report_generation", run_type="chain",
           process_inputs=trace_inputs, process_outputs=trace_output)
def generate_report(llm_chat: Callable, report_request: str, selected: list[dict],
                    classification: dict, output_file: Path) -> tuple[str, str]:
    """Build a cited Markdown report from selected evidence; returns (report_text, saved_path)."""
    blocks = []
    for i, e in enumerate(selected, 1):
        m = e["metadata"]
        blocks.append(f"[E{i}] {m.get('formulation_name')} | {m.get('chunk_type')} | "
                      f"formulation_id={m.get('formulation_id')} | {e.get('citation')}\n{e['text'][:2500]}")
    user_msg = (
        f"REPORT REQUEST:\n{report_request}\n\n"
        f"TITLE: {classification.get('report_title')}\n"
        f"KNOWN EVIDENCE GAPS FROM AGENT: {json.dumps(classification.get('known_evidence_gaps', []))}\n\n"
        "EVIDENCE:\n" + "\n\n".join(blocks)
    )
    report = llm_chat([{"role": "system", "content": REPORT_SYSTEM_PROMPT},
                       {"role": "user", "content": user_msg}])
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report, encoding="utf-8")
    return report, str(output_file.resolve())
