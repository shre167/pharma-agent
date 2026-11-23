"""Verify Capgemini chat, Titan embeddings and chat tool-calling before running the app."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharma_agent.clients import Services  # noqa: E402
from pharma_agent.config import load_settings  # noqa: E402


def main() -> int:
    settings = load_settings()
    print("LLM endpoint:", settings.llm_base_url)
    print("LLM model:", settings.llm_model)
    print("Embedding model:", settings.embedding_model)
    print("Docs dir:", settings.docs_dir, "| PDFs:", [p.name for p in settings.docs_dir.glob('*.pdf')])
    if not settings.has_api_key:
        print("FAIL: GE_API_KEY is not set (.env).")
        return 1
    services = Services(settings)
    ok = True
    for name, fn in [("chat", services.test_chat), ("embeddings", services.test_embeddings),
                     ("tool calling", services.test_tool_calling)]:
        try:
            print(f"{name}: OK ->", fn())
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"{name}: FAILED -> {exc!r}")
    if not ok:
        print("\nDo not run the agent until all three checks pass.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
