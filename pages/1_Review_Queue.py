"""Human-in-the-loop: pending drafts and the decision history."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pharma_agent.ui.components import get_runtime, render_report_detail, setup_page  # noqa: E402
from pharma_agent.wave2.storage import list_reports  # noqa: E402

setup_page("Review queue", "📝")
runtime = get_runtime()
settings = runtime.settings

st.title("📝 Review queue")
pending = list_reports(settings, pending_only=True)
history = [m for m in list_reports(settings) if m["status"] not in {"REPORT_GENERATED_PENDING_REVIEW",
                                                                    "REPORT_GENERATED_NEEDS_REVIEW"}]

tab_pending, tab_history = st.tabs([f"Pending ({len(pending)})", f"Decided ({len(history)})"])

with tab_pending:
    if not pending:
        st.info("No drafts are waiting for review. Generate one on the main page.")
    else:
        labels = {m["report_id"]: f"{m['created_at']} · {m['output_verdict']} · {m['query'][:70]}" for m in pending}
        chosen = st.selectbox("Draft", list(labels), format_func=labels.get)
        render_report_detail(settings, chosen, key="queue")

with tab_history:
    if not history:
        st.info("No decisions recorded yet.")
    else:
        frame = pd.DataFrame([{
            "report_id": m["report_id"], "status": m["status"], "verdict": m["output_verdict"],
            "decision": m.get("review_decision", {}).get("decision"),
            "reviewer": m.get("review_decision", {}).get("reviewer"),
            "reviewed_at": m.get("reviewed_at"), "query": m["query"][:90],
        } for m in history])
        st.dataframe(frame, hide_index=True, use_container_width=True)
        chosen = st.selectbox("Open a decided report", [m["report_id"] for m in history],
                              format_func=lambda i: next(m["query"][:70] for m in history if m["report_id"] == i))
        render_report_detail(settings, chosen, key="history")
