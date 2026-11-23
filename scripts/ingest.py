"""Ingest docs/*.pdf into chunks.jsonl, manifest.csv and the FAISS index.

Usage:
    python scripts/ingest.py                 # extract + embed (reuses index if unchanged)
    python scripts/ingest.py --rebuild       # force re-embedding
    python scripts/ingest.py --chunks-only   # extraction/validation only, no API key needed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharma_agent.clients import Services  # noqa: E402
from pharma_agent.config import load_settings  # noqa: E402
from pharma_agent.ingestion.pipeline import run_ingestion  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="force re-embedding even if the index is current")
    parser.add_argument("--chunks-only", action="store_true", help="skip embeddings (no API calls)")
    args = parser.parse_args()

    settings = load_settings()
    services = None
    if not args.chunks_only:
        services = Services(settings, show_embedding_progress=True)
        dims = services.test_embeddings()
        print(f"Titan embedding test OK | dimensions = {dims}")

    summary = run_ingestion(settings, services, rebuild=args.rebuild, chunks_only=args.chunks_only)
    print("\nIngestion summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["missing_bom"] or summary["missing_process"]:
        print("\nNOTE: some formulations lack a BOM/process marker - review data/manifest.csv "
              "and compare representative entries against the rendered PDF pages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
