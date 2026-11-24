"""Streamlit entry point: request -> agent -> verification -> human review.

Run:  streamlit run app.py
"""
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from pharma_agent.ui.components import (  # noqa: E402
    get_runtime, render_report_detail, render_tool_trace, setup_page, status_label)
from pharma_agent.wave2.pipeline import run_generation  # noqa: E402

setup_page("Generate report")
runtime = get_runtime()

st.title("Pharma Manufacturing Design Report Agent")
st.caption("Wave 1 (agent, RAG, citations):  scope gate → input guard → retrieval agent → sectioned report  ·  "
           "Wave 2 (verification, evaluation, safety): verification → DeepEval → human review.  "
           "Evidence-backed drafts from the handbook; every draft needs human review.")

st.session_state.setdefault("query_text", "")

query = st.text_area("Your request", key="query_text", height=130,
                     placeholder="Enter any pharmaceutical manufacturing question, describe a medicine (by indication/purpose), or name a formulation...")

if st.button("Generate report", type="primary", disabled=not query.strip()):
    with st.status("Running agent...", expanded=True) as status:
        # Pipeline callbacks fire from LangGraph worker threads as well as this one;
        # Streamlit commands must stay on the script thread, so events travel via a
        # queue and are rendered from the poll loop below.
        events: queue.Queue = queue.Queue()

        def on_event(stage: str, payload: dict) -> None:
            events.put((stage, payload))

        def render_event(stage: str, payload: dict) -> None:
            if stage == "scope":
                st.write(f"**Wave 1 · Scope gate** → medicine: `{payload.get('medicine_mentions')}` · handbook ingredient: "
                         f"`{payload.get('generic_names')}` · confidence: `{payload.get('confidence')}`")
            elif stage == "guard":
                st.write(f"**Wave 1 · Input guard** → `{payload.get('action')}` — {payload.get('reason')}")
            elif stage == "tool_call":
                args = payload["arguments"]
                label = args.get("search_query") or args.get("report_request", "")
                focus = args.get("focus")
                focus_txt = "" if focus in (None, "", "all") else f" focus=`{focus}`"
                st.write(f"**Wave 1 · [{payload['index']}/{payload['max']}] {payload['tool']}** → {str(label)[:120]}"
                         f"{focus_txt} · `{payload['result_summary']}`")
            elif stage == "report_start":
                st.write(f"**Wave 1 · Drafting report** → Generating sections in parallel from `{payload.get('evidence_count')}` evidence chunks...")
            elif stage == "report_section":
                st.write(f"**Wave 1 · Section complete** → `{payload.get('title')}`")
            elif stage == "verify":
                if payload.get("stage") == "start":
                    st.write("**Wave 2 · Verification** → Checking citations, PDF page numbers, and process claims...")
                elif payload.get("stage") == "done":
                    st.write(f"**Wave 2 · Verification complete** → Verdict: `{payload.get('verdict')}`")
            elif stage == "deepeval":
                if payload.get("stage") == "start":
                    st.write("**Wave 2 · DeepEval** → Scoring faithfulness and retrieval relevancy...")
                elif payload.get("stage") == "done":
                    mark = "meets targets" if payload.get("ok") else "below targets — see the report notes"
                    st.write(f"**Wave 2 · DeepEval complete** → {mark}" + (f" — {payload.get('note')}" if payload.get("note") else ""))

        run_result: dict = {}
        run_error: list = []

        def run_pipeline() -> None:
            try:
                run_result["value"] = run_generation(runtime, query, on_event=on_event)
            except Exception as exc:  # noqa: BLE001
                run_error.append(exc)

        worker = threading.Thread(target=run_pipeline, name="pharma-run", daemon=True)
        worker.start()
        while worker.is_alive():
            try:
                stage, payload = events.get(timeout=0.25)
            except queue.Empty:
                continue
            render_event(stage, payload)
        while not events.empty():
            stage, payload = events.get_nowait()
            render_event(stage, payload)

        if run_error:
            status.update(label="Run failed", state="error")
            st.error(f"Run failed: {run_error[0]!r}")
            st.stop()
        result = run_result["value"]
        status.update(label=f"Finished: {result['status']}", state="complete", expanded=False)
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
