"""The agent's exactly-two LangChain tools, bound to a per-request AgentSession."""
from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.tools import tool

from .session import AgentSession


def make_tools(session: AgentSession) -> list:
    @tool
    def find_documents(search_query: str, top_k: int = 5, focus: str = "all",
                       alternate_names: Optional[List[str]] = None) -> str:
        """Search the pharmaceutical handbook for matching medicine/formulation evidence. Pass generic/chemical or alternate names in alternate_names. Repeat with refined queries when evidence is incomplete or ambiguous."""
        return json.dumps(session.find_documents(search_query, top_k, focus, alternate_names), ensure_ascii=False)

    @tool
    def build_report(report_request: str, evidence_ids: List[str],
                     report_title: str = "Pharmaceutical Manufacturing Design Report",
                     known_gaps: Optional[List[str]] = None) -> str:
        """Build and save a cited Markdown design report from evidence IDs returned by find_documents. Use only when the user requests a report."""
        return json.dumps(session.build_report(report_request, evidence_ids, report_title, known_gaps),
                          ensure_ascii=False)

    return [find_documents, build_report]
