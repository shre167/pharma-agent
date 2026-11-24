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
    get_runtime, render_report_detail, render_tool_trace, setup_page, stage_line, status_label)
from pharma_agent.wave2.pipeline import run_generation  # noqa: E402

setup_page("Generate report")
runtime = get_runtime()

EXAMPLES = [
    ("Named medicine", "Guide me through the design of the manufacturing process for acetaminophen 500 mg immediate-release tablets."),
    ("Brand name", "Prepare a manufacturing design report for Dolo 650."),
    ("Description", "Prepare a report for a white round fever tablet."),
    ("General question", "How is a wet-granulation tablet line designed and validated?"),
]

st.markdown("## Pharma Manufacturing Design Report Agent")
st.caption("Turns a curated pharmaceutical handbook into evidence-backed design drafts. Wave 1 runs the gated "
           "retrieval agent and writes the sectioned, cited report; Wave 2 verifies every claim, scores it "
           "with DeepEval, and holds the draft for human approval.")

st.session_state.setdefault("query_text", "")
chip_cols = st.columns(len(EXAMPLES))
for (label, text), col in zip(EXAMPLES, chip_cols):
    if col.button(label, key=f"example_{label}", use_container_width=True):
        st.session_state["query_text"] = text
        st.rerun()

st.divider()

query = st.text_area("Report request", key="query_text", height=110,
                     placeholder="Name a medicine (brand or generic), describe one, or ask a general "
                                 "pharmaceutical manufacturing question...")

if st.button("Generate draft report", type="primary", disabled=not query.strip()):
    with st.status("Running pipeline...", expanded=True) as status:
        # Pipeline callbacks fire from LangGraph worker threads as well as this one;
        # Streamlit commands must stay on the script thread, so events travel via a
        # queue and are rendered from the poll loop below.
        events: queue.Queue = queue.Queue()

        def on_event(stage: str, payload: dict) -> None:
            events.put((stage, payload))

        def render_event(stage: str, payload: dict) -> None:
            if stage == "scope":
                stage_line("1", "Scope gate",
                           f"medicine {payload.get('medicine_mentions')} · handbook ingredient "
                           f"{payload.get('generic_names')} · confidence {payload.get('confidence')}")
            elif stage == "guard":
                stage_line("1", "Input guard",
                           f"{payload.get('action')} — {payload.get('reason')}")
            elif stage == "tool_call":
                args = payload["arguments"]
                label = str(args.get("search_query") or args.get("report_request") or "")[:120]
                focus = args.get("focus")
                focus_txt = f" · focus {focus}" if focus not in (None, "", "all") else ""
                stage_line("1", f"[{payload['index']}/{payload['max']}] {payload['tool']}",
                           f"{label}{focus_txt} · {payload['result_summary']}")
            elif stage == "report_start":
                stage_line("1", "Drafting report",
                           f"{payload.get('evidence_count')} evidence chunks · sections run in parallel")
            elif stage == "report_section":
                stage_line("1", f"Section {payload.get('number')} complete", payload.get("title"))
            elif stage == "verify":
                if payload.get("stage") == "start":
                    stage_line("2", "Verification", "citations, PDF pages, process claims, LLM claim review")
                else:
                    stage_line("2", "Verification complete", f"verdict {payload.get('verdict')}")
            elif stage == "deepeval":
                if payload.get("stage") == "start":
                    stage_line("2", "DeepEval", "scoring faithfulness and retrieval relevancy")
                else:
                    mark = "meets targets" if payload.get("ok") else "below targets"
                    note = f" — {payload.get('note')}" if payload.get("note") else ""
                    stage_line("2", "DeepEval complete", f"{mark}{note}")

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
