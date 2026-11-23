# docs/

Put the source PDF(s) here, e.g. `Handbook_of_Pharmaceutical_Manufacturing_Formulations__Third_Edition1-232-445.pdf`.

`python scripts/ingest.py` ingests every `*.pdf` in this folder. PDFs are git-ignored (copyrighted).

The extractor is tuned for the handbook's layout: two columns, bold ~11 pt UPPERCASE formulation titles,
and the labels `Bill of Materials` / `Manufacturing Directions`. A PDF with a different layout needs
changes in `src/pharma_agent/ingestion/pdf_extract.py` (title detection) and `chunking.py` (section labels).
