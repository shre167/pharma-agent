"""Streamlit entry point: request -> agent -> verification -> human review.

The generation thread lives in st.session_state, so starting a run and coming back
after navigating to another app page re-attaches to the same run instead of dropping it.

Run:  streamlit run app.py
"""
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

from pharma_agent.ui.components import (  # noqa: E402
    get_runtime, render_report_detail, render_tool_trace, setup_page, stage_line, status_label)
from pharma_agent.wave2.pipeline import run_generation  # noqa: E402

setup_page("Generate report")
runtime = get_runtime()

st.markdown("## Pharma Manufacturing Design Report Agent")
st.caption("Turns a curated pharmaceutical handbook into evidence-backed design drafts. Wave 1 runs the gated "
           "retrieval agent and writes the sectioned, cited report; Wave 2 verifies every claim, scores it "
           "with DeepEval, and holds the draft for human approval. Press Enter to generate.")


def elapsed(run_state: dict) -> str:
    if run_state and run_state.get("t0") is not None:
        return f" · {time.monotonic() - run_state['t0']:.0f}s in"
    return ""


def render_event(stage: str, payload: dict, run_state: dict | None = None) -> None:
    when = elapsed(run_state)
    if stage == "scope":
        text = (f"medicine {payload.get('medicine_mentions')} · handbook ingredient "
                f"{payload.get('generic_names')} · confidence {payload.get('confidence')}{when}")
        stage_line("1", "Scope gate", text)
    elif stage == "guard":
        stage_line("1", "Input guard", f"{payload.get('action')} — {payload.get('reason')}{when}")
    elif stage == "tool_call":
        args = payload["arguments"]
        label = str(args.get("search_query") or args.get("report_request") or "")[:120]
        focus = args.get("focus")
        focus_txt = f" · focus {focus}" if focus not in (None, "", "all") else ""
        stage_line("1", f"[{payload['index']}/{payload['max']}] {payload['tool']}",
                   f"{label}{focus_txt} · {payload['result_summary']}{when}")
    elif stage == "report_start":
        stage_line("1", "Drafting report",
                   f"{payload.get('evidence_count')} evidence chunks · sections run in parallel{when}")
    elif stage == "report_section":
        stage_line("1", f"Section {payload.get('number')} complete", f"{payload.get('title')}{when}")
    elif stage == "verify":
        if payload.get("stage") == "start":
            stage_line("2", "Verification", f"citations, PDF pages, process claims, LLM claim review{when}")
        else:
            stage_line("2", "Verification complete", f"verdict {payload.get('verdict')}{when}")
    elif stage == "deepeval":
        if payload.get("stage") == "start":
            stage_line("2", "DeepEval", f"scoring faithfulness and retrieval relevancy{when}")
        else:
            mark = "meets targets" if payload.get("ok") else "below targets"
            note = f" — {payload.get('note')}" if payload.get("note") else ""
            stage_line("2", "DeepEval complete", f"{mark}{note}{when}")


def drain_events(run_state) -> None:
    """Render events buffered while the run finished on its own (e.g. another page was open)."""
    events = run_state["events"]
    while not events.empty():
        stage, payload = events.get_nowait()
        render_event(stage, payload, run_state)


def finalize(run_state, status=None) -> None:
    if run_state.get("error"):
        if status is not None:
            status.update(label="Run failed", state="error")
        st.error(f"Run failed: {run_state['error']!r}")
        return
    if status is not None:
        status.update(label=f"Finished: {run_state['result']['status']}", state="complete", expanded=False)
    st.session_state["last_result"] = run_state["result"]
    run_state["pending_render"] = False


def poll_until_done(run_state, status) -> None:
    events = run_state["events"]
    while run_state["thread"].is_alive():
        try:
            stage, payload = events.get(timeout=0.25)
        except queue.Empty:
            continue
        render_event(stage, payload, run_state)
    drain_events(run_state)
    finalize(run_state, status)


run_state = st.session_state.get("run_state")

if run_state and run_state["thread"].is_alive():
    with st.status("Pipeline running for a previous request — you can visit other pages and come back.",
                   expanded=True) as status:
        st.caption(run_state["query"])
        poll_until_done(run_state, status)
elif run_state and run_state.get("pending_render") and run_state["result"] is not None:
    with st.status(f"Finished: {run_state['result'].get('status')}", expanded=False):
        st.caption(run_state["query"])
    drain_events(run_state)
    finalize(run_state)
elif run_state and run_state.get("pending_render") and run_state.get("error"):
    st.error(f"Run failed: {run_state['error']!r}")
    run_state["pending_render"] = False
    run_state = None

running = bool(run_state and run_state["thread"].is_alive())

with st.form("request_form"):
    query = st.text_input(
        "Report request", key="query_text",
        placeholder="Name a medicine (brand or generic), describe one, or ask a general pharmaceutical "
                    "manufacturing question — press Enter to generate")
    submitted = st.form_submit_button("Generate draft report", type="primary", disabled=running)

if submitted and query.strip():
    run_state = {"query": query, "events": queue.Queue(), "result": None, "error": None,
                 "pending_render": False, "thread": None, "t0": time.monotonic()}
    st.session_state["run_state"] = run_state

    def worker() -> None:
        try:
            run_state["result"] = run_generation(runtime, run_state["query"])
            run_state["pending_render"] = True
        except Exception as exc:  # noqa: BLE001
            run_state["error"] = exc
            run_state["pending_render"] = True

    run_state["thread"] = threading.Thread(target=worker, name="pharma-run", daemon=True)
    run_state["thread"].start()

    with st.status("Running pipeline...", expanded=True) as status:
        poll_until_done(run_state, status)

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
