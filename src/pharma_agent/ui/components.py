"""Shared Streamlit components: runtime loading, report detail view and the review form."""
from __future__ import annotations

import html
from dataclasses import replace

import pandas as pd
import streamlit as st

from ..config import load_settings
from ..observability import langsmith_enabled
from ..retrieval.index import IndexOutOfDate
from ..runtime import Runtime, build_runtime
from ..wave2.storage import PENDING_STATUSES, load_report, submit_review

STATUS_COLOURS = {
    "FINALISED": "green", "REJECTED": "red", "BLOCKED_NOT_FINALISED": "orange",
    "REPORT_GENERATED_PENDING_REVIEW": "blue", "REPORT_GENERATED_NEEDS_REVIEW": "orange",
    "REFUSED": "red", "ESCALATED": "orange", "NO_REPORT": "gray", "ERROR": "red",
}


_PAGE_CSS = """
<style>
    .block-container {padding-top: 1.1rem; max-width: 1080px;}
    h1, h2 {letter-spacing: -0.02em;}
    .stage-line {display:flex; align-items:baseline; gap:0.6rem; padding:0.3rem 0.75rem;
                 margin:0.15rem 0; border-left:3px solid transparent;
                 border-radius:0 0.35rem 0.35rem 0; font-size:0.87rem; line-height:1.5;
                 background:rgba(128,128,128,0.07);}
    .wave-tag {flex:0 0 auto; font-size:0.62rem; font-weight:700; letter-spacing:0.09em;
               padding:0.12rem 0.5rem; border-radius:999px; align-self:center;}
    .wave-1 {border-left-color:#1A56DB;}
    .wave-1 .wave-tag {background:#E8F0FE; color:#1A56DB;}
    .wave-2 {border-left-color:#046C4E;}
    .wave-2 .wave-tag {background:#DEF7EC; color:#046C4E;}
    .stage-title {font-weight:600; flex:0 0 auto;}
    .stage-detail {color:rgba(90,95,110,1); word-break:break-word;}
</style>
"""


def setup_page(title: str, icon: str | None = None) -> None:
    st.set_page_config(page_title=f"{title} · Pharma Design Agent", page_icon=icon, layout="wide")
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)


def stage_line(wave: str, title: str, detail: str = "") -> None:
    """One pipeline progress row; escapes untrusted detail text."""
    detail_html = html.escape(str(detail)) if detail else ""
    st.markdown(
        f'<div class="stage-line wave-{wave}"><span class="wave-tag">WAVE {wave}</span>'
        f'<span class="stage-title">{html.escape(str(title))}</span>'
        + (f'<span class="stage-detail">{detail_html}</span>' if detail_html else "")
        + "</div>", unsafe_allow_html=True)


def status_label(status: str) -> str:
    return f":{STATUS_COLOURS.get(status, 'gray')}[**{status}**]"


@st.cache_resource(show_spinner="Loading knowledge base and models...")
def _cached_runtime(api_key: str, langsmith_key: str) -> Runtime:
    settings = replace(load_settings(), ge_api_key=api_key, langsmith_api_key=langsmith_key)
    return build_runtime(settings)


def get_runtime() -> Runtime:
    """Return the shared Runtime, or render a helpful message and stop the page."""
    settings = load_settings()
    configured_key = settings.ge_api_key if settings.has_api_key else ""
    session_key = st.session_state.get("ge_api_key", "")
    api_key = session_key or configured_key

    with st.sidebar:
        st.header("Connection")
        if not api_key:
            key = st.text_input("LLM API key (GE_API_KEY)", type="password",
                                help="Key for LLM_BASE_URL, e.g. a Mistral or Capgemini gateway key; "
                                     "or put GE_API_KEY in .env (see .env.example).")
            if key:
                st.session_state["ge_api_key"] = key.strip()
                st.rerun()
        else:
            masked = ("•" * (len(api_key) - 4) + api_key[-4:]) if len(api_key) > 4 else "••••"
            st.caption(f"API key configured: `{masked}`")
            if st.button("Change API Key", key="btn_change_api_key"):
                st.session_state["ge_api_key"] = ""
                st.rerun()

        if not (st.session_state.get("ls_key") or settings.langsmith_api_key):
            ls = st.text_input("LangSmith API key (optional)", type="password")
            if ls:
                st.session_state["ls_key"] = ls.strip()
                st.rerun()

    if not api_key:
        st.info("Enter your `LLM_BASE_URL` API key in the sidebar (or set `GE_API_KEY` in `.env`) to begin.")
        st.stop()
    ls_key = st.session_state.get("ls_key") or settings.langsmith_api_key
    try:
        runtime = _cached_runtime(api_key, ls_key)
    except (FileNotFoundError, IndexOutOfDate) as exc:
        st.error(f"The knowledge base is not ready: {exc}")
        st.markdown("Build the knowledge base using your configured API key:")
        if st.button("Build Knowledge Base Now", type="primary"):
            with st.status("Building knowledge base (embedding chunks)...", expanded=True) as status:
                from pharma_agent.clients import Services
                from pharma_agent.ingestion.pipeline import run_ingestion
                try:
                    settings_with_key = replace(settings, ge_api_key=api_key, langsmith_api_key=ls_key)
                    services = Services(settings_with_key, show_embedding_progress=False)
                    summary = run_ingestion(settings_with_key, services)
                    status.update(label=f"Index built! {summary.get('chunks', 0)} chunks indexed.", state="complete")
                    st.success("Knowledge base built successfully! Reloading...")
                    st.rerun()
                except Exception as e:
                    status.update(label="Build failed", state="error")
                    st.error(f"Failed to build index: {e}")
        st.caption("Or run in terminal:")
        st.code("python scripts/ingest.py", language="bash")
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not start the agent runtime: {exc!r}")
        st.stop()

    with st.sidebar:
        from pharma_agent import __version__
        st.caption(f"Build v{__version__}")
        st.caption(f"LLM: `{runtime.settings.llm_model}`")
        st.caption(f"Embeddings: `{runtime.settings.embedding_model}`")
        st.caption(f"Formulations indexed: **{len({c['metadata']['formulation_id'] for c in runtime.kb.chunks})}** "
                   f"· chunks: **{len(runtime.kb.chunks)}**")
        st.caption("LangSmith tracing: " + ("**ON**" if langsmith_enabled() else "off"))
    return runtime


def render_checks(checks: list[dict], llm_review: dict) -> None:
    for c in checks:
        mark = ":green[PASS]" if c["passed"] else ":orange[CHECK]"
        st.markdown(f"{mark} **{c['name']}** — {c['detail']}")
    issues = (llm_review or {}).get("issues") or []
    if issues:
        st.markdown("**LLM-flagged issues**")
        for issue in issues[:15]:
            st.markdown(f"- **{issue.get('claim', '')}** — {issue.get('reason', '')}")


def _deepeval_mark(value: float, target: float) -> str:
    return ":green[meets target]" if value >= target else ":orange[below target]"


def render_deepeval(deepeval: dict | None) -> None:
    """Per-run DeepEval scores (faithfulness / retrieval relevancy) with targets."""
    if not deepeval:
        return
    st.markdown("**DeepEval quality scores**")
    if deepeval.get("faithfulness") is None:
        st.caption(deepeval.get("note") or "DeepEval scores were not produced for this run.")
        return
    f_mark = _deepeval_mark(deepeval["faithfulness"], deepeval["target_faithfulness"])
    r_mark = _deepeval_mark(deepeval["retrieval_relevancy"], deepeval["target_retrieval_relevancy"])
    st.markdown(
        f"- **Faithfulness:** {deepeval['faithfulness']:.3f} (target ≥ {deepeval['target_faithfulness']}) — {f_mark}\n"
        f"- **Retrieval relevancy:** {deepeval['retrieval_relevancy']:.3f} "
        f"(target ≥ {deepeval['target_retrieval_relevancy']}) — {r_mark}")


def render_tool_trace(tool_calls: list[dict]) -> None:
    if not tool_calls:
        st.caption("No tools were called.")
        return
    rows = [{"#": i, "tool": c["tool"],
             "query / request": str(c["arguments"].get("search_query") or c["arguments"].get("report_request") or "")[:160],
             "focus": c["arguments"].get("focus", ""),
             "results": c["result_summary"].get("result_count"),
             "status": c["result_summary"].get("status") or c["result_summary"].get("error")
             or c["result_summary"].get("note") or ""}
            for i, c in enumerate(tool_calls, 1)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_report_detail(settings, report_id: str, key: str) -> None:
    """Report, verification, evidence, tool trace and the human review form."""
    manifest, report_text = load_report(settings, report_id)
    status = manifest["status"]
    checks = manifest.get("checks", [])
    passed = sum(1 for c in checks if c["passed"])
    issues = (manifest.get("llm_review") or {}).get("issues") or []
    deepeval = manifest.get("deepeval") or {}

    c1, c2, c3, c4, c5, c6 = st.columns([2.4, 1, 1, 1, 1, 1.4])
    c1.markdown(f"{status_label(status)}\n\nID `{report_id[:8]}`")
    c2.metric("Checks", f"{passed}/{len(checks)}")
    c3.metric("Flagged claims", len(issues))
    c4.metric("Evidence", len(manifest.get("evidence", [])))
    c5.metric("Latency", f"{manifest.get('latency_s')} s")
    c6.metric("DeepEval faithfulness",
              "—" if deepeval.get("faithfulness") is None else f"{deepeval['faithfulness']:.3f}")
    if manifest.get("trace_url"):
        st.markdown(f"[Open LangSmith trace]({manifest['trace_url']})")
    st.caption("Pipeline · Wave 1: scope gate → input guard → retrieval agent → sectioned report"
               "  |  Wave 2: verification → DeepEval → human review")

    tabs = st.tabs(["Report", "Verification · Wave 2", "Evidence · Wave 1", "Tool trace · Wave 1", "Review · Wave 2"])
    with tabs[0]:
        st.markdown(report_text)
        st.download_button("Download report.md", report_text, file_name=f"report_{report_id[:8]}.md",
                           mime="text/markdown", key=f"{key}_dl")
    with tabs[1]:
        render_checks(manifest["checks"], manifest.get("llm_review", {}))
        render_deepeval(manifest.get("deepeval"))
    with tabs[2]:
        for e in manifest["evidence"]:
            with st.expander(f"[{e['tag']}] {e['formulation_name']} · {e['chunk_type']} · p.{e['page']}"):
                st.caption(f"formulation `{e['formulation_id']}` · {e['source']} · pages {e['pages']}")
                st.text(e["text"])
    with tabs[3]:
        render_tool_trace(manifest.get("tool_calls", []))
    with tabs[4]:
        _review_panel(settings, manifest, key)


def _review_panel(settings, manifest: dict, key: str) -> None:
    report_id = manifest["report_id"]
    if manifest["status"] not in PENDING_STATUSES:
        d = manifest.get("review_decision", {})
        st.markdown(f"**Decision:** `{d.get('decision')}` by **{d.get('reviewer')}** at {manifest.get('reviewed_at')}")
        if d.get("comment"):
            st.markdown(f"> {d['comment']}")
        if manifest.get("finalised"):
            st.success(f"Finalised. Approved copy: `{manifest.get('approved_path')}`")
        elif manifest["status"] == "BLOCKED_NOT_FINALISED":
            st.warning("Approval was recorded but the report was NOT finalised because mandatory checks "
                       "or the automatic verdict did not pass.")
        return

    if manifest["output_verdict"] != "PASS":
        st.warning("Automatic verification did not fully pass. You may still record a decision, but an approval "
                   "cannot finalise this report.")

    review = manifest.get("llm_review", {})
    issues = (review or {}).get("issues") or []
    passed_checks = sum(1 for c in manifest.get("checks", []) if c.get("passed"))
    total_checks = len(manifest.get("checks", []))
    evidence_count = len(manifest.get("evidence", []))
    st.markdown(f"#### Decision summary — everything you need in one glance")
    st.markdown(
        f"- **Auto-verdict:** `{manifest['output_verdict']}` · **Checks:** {passed_checks}/{total_checks} passed · "
        f"**LLM review issues:** {len(issues)} · **Evidence chunks:** {evidence_count}\n"
        f"- **Query:** {manifest.get('query', '')[:200]}\n"
        f"- **Status:** `{manifest['status']}` — only approve + all checks + verdict PASS finalises this report.")
    deepeval = manifest.get("deepeval") or {}
    if deepeval.get("faithfulness") is not None:
        f_mark = _deepeval_mark(deepeval["faithfulness"], deepeval["target_faithfulness"])
        r_mark = _deepeval_mark(deepeval["retrieval_relevancy"], deepeval["target_retrieval_relevancy"])
        st.markdown(f"- **DeepEval:** faithfulness {deepeval['faithfulness']:.3f} ({f_mark}) · "
                    f"relevancy {deepeval['retrieval_relevancy']:.3f} ({r_mark})")
    elif deepeval.get("note"):
        st.markdown(f"- **DeepEval:** {deepeval['note']}")
    if issues:
        with st.expander(f"Flagged claims ({len(issues)})"):
            render_checks([], review)
    with st.expander("System-level evaluation snapshot (latest labelled run)"):
        from pharma_agent.wave2.evaluation import EVAL_TARGETS, load_evaluations, summarize
        evaluations = load_evaluations(settings)
        if evaluations:
            latest = sorted(evaluations)[-1]
            observed = summarize(evaluations[latest])
            rows = {k: {"observed": observed.get(k), "target": t}
                    for k, t in [("outcome_accuracy", EVAL_TARGETS["outcome_accuracy"]),
                                 ("retrieval_hit", EVAL_TARGETS["retrieval_hit_rate"]),
                                 ("citation_integrity", EVAL_TARGETS["citation_integrity"]),
                                 ("faithfulness", EVAL_TARGETS["faithfulness"]),
                                 ("retrieval_relevancy", EVAL_TARGETS["retrieval_relevancy"])]}
            st.dataframe(pd.DataFrame(rows).T, use_container_width=True)
            st.caption(f"Latest labelled evaluation: “{latest}”. System quality context for your review — "
                       "the decision below applies to this specific report.")
        else:
            st.caption("No labelled evaluation has been run yet (Evaluation & traces page).")
    with st.form(f"{key}_review_form"):
        reviewer = st.text_input("Reviewer name")
        comment = st.text_area("Comment (optional)")
        a, r = st.columns(2)
        approve = a.form_submit_button("Approve", use_container_width=True, type="primary")
        reject = r.form_submit_button("Reject", use_container_width=True)
    if approve or reject:
        if not reviewer.strip():
            st.error("Please enter your name so the decision is attributable.")
            return
        submit_review(settings, report_id, "approve" if approve else "reject", reviewer, comment)
        st.rerun()
