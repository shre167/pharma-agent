# Pharma Manufacturing Design Report Agent

Streamlit app that turns a medicine request (brand name, generic name, or a detailed description) into a
**cited Markdown design report** drawn only from the handbook, then puts a human reviewer in the loop.
Ported from `Pharma_Manufacturing_Design_Report_Agent_EVALUATION_READY.ipynb`.

> Educational evidence assistant. Reports are drafts for qualified review, not validated manufacturing instructions.

## Architecture (maps to the project diagram)

| Diagram block | Where it lives |
|---|---|
| User request · Streamlit page | `app.py`, `pages/` |
| Agent: identify medicine (product + variant) | `src/pharma_agent/agent/scope.py` (fail-closed gate) |
| Agent: generate cited report | `agent/runner.py`, `agent/tools.py`, `agent/session.py`, `agent/report.py` |
| "No match = decline. Missing data = gap." | scope gate + `AGENT_SYSTEM_PROMPT` + report "Evidence gaps" section |
| Knowledge base: manifest | `data/manifest.csv` (written by ingestion) |
| Knowledge base: index + embeddings | `retrieval/` (FAISS + BM25/RRF), Titan via `clients.py` |
| Ingestion (`ingest.py`) | `scripts/ingest.py` → `src/pharma_agent/ingestion/` |
| Wave 2: human review | `wave2/storage.py::submit_review`, `pages/1_Review_Queue.py` |
| Wave 2: LangSmith | `observability.py` (+ `@traceable` throughout) |
| Wave 2: DeepEval | `wave2/evaluation.py`, `pages/3_Evaluation_and_Traces.py` |
| Final design report (approved) | `outputs/approved/<report_id>.md` |

```
User request ─► scope gate ─► input guard ─► agent (find_documents ⇄ … ─► build_report)
                                                         │
                         outputs/reports/<id>/{report.md, manifest.json}
                                                         ▼
              verification (citation checks + LLM claim review) ─► human Approve/Reject
                                                         ▼
                      FINALISED only if approved AND all checks pass AND verdict = PASS
```

## Quick start

```bash
# 1. environment (Windows: .venv\Scripts\activate)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                     # optional; the entry points also add src/ to the path

# 2. configuration
cp .env.example .env                 # then fill in GE_API_KEY (and optionally LANGSMITH_API_KEY)

# 3. put the handbook PDF in docs/
cp "/path/to/Handbook_of_Pharmaceutical_Manufacturing_Formulations__Third_Edition1-232-445.pdf" docs/

# 4. verify Capgemini chat + Titan embeddings + tool calling
python scripts/check_setup.py

# 5. build the knowledge base (extract → chunks.jsonl + manifest.csv → FAISS)
python scripts/ingest.py             # add --chunks-only to inspect extraction with no API calls

# 6. run
streamlit run app.py
```

In VS Code: open the folder, select the `.venv` interpreter, then use **Run and Debug** (Streamlit, Ingest, Check setup, Pytest)
or **Terminal → Run Task**.

Before trusting results, open the **Knowledge Base** page and spot-check BOM chunks against the rendered PDF pages
(extraction is layout-aware text reconstruction, not proof every value was read correctly).

## Using the app

| Page | Purpose |
|---|---|
| **Generate report** (`app.py`) | Enter a request, watch the gate / guard / tool calls live, read the report, verification, evidence and tool trace, then Approve or Reject. |
| **Review queue** | All drafts awaiting a decision, plus decision history. |
| **Knowledge base** | Manifest, chunk inspector, retrieval playground, ingredient vocabulary. |
| **Evaluation & traces** | Run the labelled test set (DeepEval), view per-case deliverables, run log with LangSmith links, generate the Wave 2 summary. |

Evaluation cases live in `evaluation/eval_cases.json`; edit or add cases there.

### Review rules

* A draft is **never** auto-approved; automated evaluation records `pending`, not a fake human decision.
* `FINALISED` requires: reviewer approves **and** every check passes **and** auto-verdict is `PASS`.
  Approving a flagged/failed draft is recorded as `BLOCKED_NOT_FINALISED`.
* Decisions are appended to `outputs/reviews.jsonl`; approved copies go to `outputs/approved/`.

## Project layout

```
app.py                      Streamlit entry (generate + review)
pages/                      Review queue · Knowledge base · Evaluation & traces
scripts/ingest.py           CLI ingestion          scripts/check_setup.py   API checks
evaluation/eval_cases.json  Labelled test set
docs/                       Source PDFs (git-ignored)
data/                       chunks.jsonl, manifest.csv, faiss_index/, cache/   (generated)
outputs/                    reports/, approved/, evaluation/, runs.jsonl, reviews.jsonl   (generated)
src/pharma_agent/
  config.py  clients.py  observability.py  runtime.py  utils.py
  ingestion/   pdf_extract · sections · chunking · terms · pipeline
  retrieval/   index (FAISS build/load) · knowledge_base (BM25 + RRF)
  agent/       prompts · scope · session · tools · report · runner
  wave2/       guard · verify · pipeline · storage · evaluation · summary
  ui/          components
tests/                      29 tests (synthetic two-column PDF, RRF, gate, tools, scripted-LLM end-to-end, review rules)
```

## Differences from the notebook and the diagram

**Stack.** The diagram shows ChromaDB, local MiniLM and pypdf; the notebook uses FAISS, Capgemini-served Titan and PyMuPDF
(the two-column extraction depends on PyMuPDF font/coordinate data). This project follows the **notebook**. To switch stores or
embeddings, change `retrieval/index.py` and `clients.py`; nothing else touches them.

**Variants.** The diagram says "one report per variant"; the notebook builds one report that covers every matching variant,
kept separate with an evidence-gap note. The notebook behaviour is kept. If you want strictly one report per variant, that is a change
to `AGENT_SYSTEM_PROMPT` (call `build_report` once per formulation ID).

**Structural changes**
* Module globals (`_AGENT_EVIDENCE`, `_REQUEST_GATE`, `_W2_CAPTURE`) became a per-request `AgentSession`, so concurrent Streamlit sessions cannot leak evidence or gate state; the monkey-patched report capture is now an explicit field.
* `input()`-based review became `submit_review()` behind a form.
* Ingestion is a separate step that writes `chunks.jsonl` + `manifest.csv`; the app only loads them and refuses to embed at runtime.
* `docs/` may hold several PDFs; formulation IDs include the source file name.

**Bug fixes found while porting**
* Steps written `(a) text` never matched the step regex (it needed a second `.`/`)`).
* The page-citation check could read the next line's step number (e.g. `3.`) as a cited page.
* The notebook told the agent both "never ask the user" and "ask for clarification before report generation"; now consistently "cover each variant, record the ambiguity as a gap".
* `active_ingredient` metadata came from a hard-coded 7-name list; it now comes from the heading vocabulary, like the scope gate.
* The DeepEval judge and Wave 2 summary used literal `\\n` strings; fixed.

## Tests

```bash
python -m pytest -q
```

No API calls are made: the LLM is scripted, FAISS is replaced by a fake dense ranking, and the PDF fixture is generated on the fly.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "The knowledge base is not ready" | Run `python scripts/ingest.py` |
| Index built with a different embedding model | `python scripts/ingest.py --rebuild` |
| Tool-calling check fails | The Capgemini model/endpoint must support OpenAI-style tool calls; do not run the agent until `check_setup.py` passes |
| 0 formulations detected | PDF layout differs from the handbook; see `docs/README.md` |
| No LangSmith links | Set `LANGSMITH_API_KEY`; links resolve a moment after each run |
