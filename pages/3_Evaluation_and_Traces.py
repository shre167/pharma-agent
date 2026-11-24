"""Wave 2: DeepEval test-set evaluation, run log with LangSmith traces, final summary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from pharma_agent.ui.components import get_runtime, setup_page  # noqa: E402
from pharma_agent.wave2.evaluation import (  # noqa: E402
    EVAL_TARGETS, compare_evaluations, load_eval_cases, load_evaluations, run_evaluation, summarize)
from pharma_agent.wave2.storage import latest_runs  # noqa: E402
from pharma_agent.wave2.summary import write_wave2_summary  # noqa: E402

setup_page("Evaluation & traces")
runtime = get_runtime()
settings = runtime.settings

st.title("Evaluation & traces")
tab_run, tab_results, tab_runs, tab_summary = st.tabs(
    ["Run evaluation", "Results", "Run log & traces", "Wave 2 summary"])

with tab_run:
    cases = load_eval_cases(settings)
    st.caption(f"{len(cases)} labelled cases from `evaluation/eval_cases.json`. Runs the *current* system; "
               "no human approval is simulated. Costs real API calls.")
    st.dataframe(pd.DataFrame(cases)[["id", "expected", "gold_terms", "query"]], hide_index=True, use_container_width=True)
    name = st.text_input("Evaluation label", value="baseline",
                         help="Re-run the same set under a new label after a real system change, then compare.")
    if st.button("Run evaluation", type="primary", disabled=not name.strip()):
        bar = st.progress(0.0, text="Starting...")
        frame = run_evaluation(
            runtime, name.strip(),
            on_case=lambda i, n, c: bar.progress((i - 1) / n, text=f"Case {i}/{n}: {c['id']}"))
        bar.progress(1.0, text="Done")
        st.success(f"Saved evaluation '{name}'. See the Results tab.")

with tab_results:
    evaluations = load_evaluations(settings)
    if not evaluations:
        st.info("No evaluation has been run yet.")
    else:
        chosen = st.selectbox("Evaluation", list(evaluations))
        df = evaluations[chosen]
        observed = summarize(df)
        st.subheader("Measured metrics vs targets")
        targets = {"outcome_accuracy": EVAL_TARGETS["outcome_accuracy"], "retrieval_hit": EVAL_TARGETS["retrieval_hit_rate"],
                   "citation_integrity": EVAL_TARGETS["citation_integrity"], "faithfulness": EVAL_TARGETS["faithfulness"],
                   "retrieval_relevancy": EVAL_TARGETS["retrieval_relevancy"]}
        st.dataframe(pd.DataFrame({"observed": observed, "target": pd.Series(targets)}), use_container_width=True)
        st.subheader("Per-case results")
        st.dataframe(df.drop(columns=["trace_run_id", "deliverable_path"], errors="ignore"),
                     hide_index=True, use_container_width=True,
                     column_config={"trace_url": st.column_config.LinkColumn("trace", display_text="open")})
        for _, row in df.iterrows():
            path = Path(row["deliverable_path"])
            if path.exists():
                with st.expander(f"{row['case_id']} · {row['status']} · deliverable.md"):
                    st.markdown(path.read_text(encoding="utf-8"))
        if len(evaluations) > 1:
            st.subheader("Comparison across evaluations")
            st.dataframe(compare_evaluations(settings), use_container_width=True)

with tab_runs:
    runs = latest_runs(settings)
    if not runs:
        st.info("No runs recorded yet.")
    else:
        frame = pd.DataFrame(runs)
        cols = [c for c in ["created_at", "origin", "run_id", "status", "latency_s", "trace_url", "query"] if c in frame]
        st.dataframe(frame[cols].sort_values("created_at", ascending=False) if "created_at" in cols else frame[cols],
                     hide_index=True, use_container_width=True,
                     column_config={"trace_url": st.column_config.LinkColumn("LangSmith", display_text="open")})

with tab_summary:
    st.caption("Built only from real run, review and evaluation records; unchecked acceptance items stay open.")
    if st.button("Generate Wave 2 summary"):
        path, text = write_wave2_summary(settings)
        st.success(f"Saved to `{path}`")
        st.markdown(text)
        st.download_button("Download WAVE2_FINAL_DELIVERABLE.md", text, "WAVE2_FINAL_DELIVERABLE.md", "text/markdown")
