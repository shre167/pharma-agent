"""Knowledge base: manifest, chunk inspector and retrieval playground."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from pharma_agent.ui.components import get_runtime, setup_page  # noqa: E402

setup_page("Knowledge base", "📚")
runtime = get_runtime()
kb = runtime.kb

st.title("📚 Knowledge base")
manifest = kb.manifest_frame()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Formulations", len(manifest))
c2.metric("Chunks", len(kb.chunks))
c3.metric("Missing BOM", int((~manifest["has_bom"]).sum()))
c4.metric("Missing process", int((~manifest["has_process"]).sum()))
st.caption("Extraction is layout-aware text reconstruction, not proof every value was read correctly. "
           "Spot-check BOM chunks against the rendered PDF pages.")

tab_manifest, tab_inspect, tab_search, tab_vocab = st.tabs(
    ["Manifest", "Chunk inspector", "Retrieval playground", "Ingredient vocabulary"])

with tab_manifest:
    text = st.text_input("Filter by formulation name")
    only_gaps = st.checkbox("Only formulations missing a BOM or process")
    view = manifest
    if text:
        view = view[view["formulation_name"].str.contains(text, case=False, na=False)]
    if only_gaps:
        view = view[~view["has_bom"] | ~view["has_process"]]
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("Download manifest.csv", manifest.to_csv(index=False), "manifest.csv", "text/csv")

with tab_inspect:
    a, b, c = st.columns(3)
    ctype = a.selectbox("Chunk type", ["(any)", "FORMULATION", "BOM", "PROCESS"])
    keyword = b.text_input("Keyword", value="acetaminophen")
    limit = c.number_input("Max results", 1, 50, 8)
    for chunk in kb.inspect(None if ctype == "(any)" else ctype, keyword or None, int(limit)):
        m = chunk["metadata"]
        with st.expander(f"{m['formulation_name']} · {chunk['type']} · pages {m['pages']}"):
            st.caption(f"formulation `{m['formulation_id']}` · chunk `{chunk['id']}`")
            st.text(chunk["text"][:3500])

with tab_search:
    q = st.text_input("Query", value="acetaminophen tablets manufacturing directions")
    a, b = st.columns(2)
    k = a.slider("top_k", 1, 15, 8)
    focus = b.selectbox("Focus", ["all", "FORMULATION", "BOM", "PROCESS"])
    if st.button("Search", disabled=not q.strip()):
        for i, e in enumerate(kb.retrieve(q, top_k=k, chunk_types=None if focus == "all" else [focus]), 1):
            m = e["metadata"]
            with st.expander(f"{i}. {m['formulation_name']} · {m['chunk_type']} · p.{m['page']} · RRF {e['rrf_score']:.4f}"):
                st.text(e["text"][:3000])

with tab_vocab:
    st.caption("Built automatically from handbook headings; the scope gate can only match these exact entries.")
    st.dataframe([{"ingredient": k, "headings": len(v), "examples": "; ".join(sorted(v)[:3])}
                  for k, v in sorted(kb.vocab.items())], hide_index=True, use_container_width=True)
