"""Streamlit entry point: request -> agent -> verification -> human review.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from pharma_agent.ui.components import (  # noqa: E402
    get_runtime, render_report_detail, render_tool_trace, setup_page, status_label)
from pharma_agent.wave2.pipeline import run_generation  # noqa: E402

setup_page("Generate report")
runtime = get_runtime()

st.title("💊 Pharma Manufacturing Design Report Agent")
st.caption("Evidence-backed drafts from the handbook. Medicine + variant is identified first; no match means decline, "
           "missing data becomes a stated gap. Every draft needs human review - this is not validated manufacturing software.")

st.session_state.setdefault("query_text", "")

query = st.text_area("Your request", key="query_text", height=130,
                     placeholder="Enter any pharmaceutical manufacturing question, describe a medicine (by indication/purpose), or name a formulation...")

if st.button("Generate report", type="primary", disabled=not query.strip()):
    with st.status("Running agent...", expanded=True) as status:
        def on_event(stage: str, payload: dict) -> None:
            if stage == "scope":
                st.write(f"**Scope gate** → medicine: `{payload.get('medicine_mentions')}` · handbook ingredient: "
                         f"`{payload.get('generic_names')}` · confidence: `{payload.get('confidence')}`")
            elif stage == "guard":
                st.write(f"**Input guard** → `{payload.get('action')}` — {payload.get('reason')}")
            elif stage == "tool_call":
                args = payload["arguments"]
                label = args.get("search_query") or args.get("report_request", "")
                st.write(f"**[{payload['index']}/{payload['max']}] {payload['tool']}** → {str(label)[:120]} "
                         f"· `{payload['result_summary']}`")
            elif stage == "verify" and payload.get("stage") == "done":
                st.write(f"**Verification** → `{payload.get('verdict')}`")

        try:
            result = run_generation(runtime, query, on_event=on_event)
            status.update(label=f"Finished: {result['status']}", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            status.update(label="Run failed", state="error")
            st.error(f"Run failed: {exc!r}")
            st.stop()
    st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    st.divider()
    if result.get("report_id"):
        render_report_detail(runtime.settings, result["report_id"], key="main")
    else:
        st.markdown(f"**Outcome:** {status_label(result['status'])}")
        st.warning(result.get("reason") or result.get("answer") or "No report was generated.")
        if result.get("scope", {}).get("candidate_ingredients"):
            st.caption(f"Candidate ingredients considered: {result['scope']['candidate_ingredients']}")
        render_tool_trace(result.get("tool_calls", []))
        if result.get("trace_url"):
            st.markdown(f"[Open LangSmith trace]({result['trace_url']})")
