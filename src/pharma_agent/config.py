"""Central configuration: environment variables, paths and version constants."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Bump deliberately to invalidate caches / rebuild the FAISS index.
EXTRACTION_VERSION = 1
CHUNKING_VERSION = 1
INDEX_VERSION = 3
MAX_PROCESS_STEPS_PER_CHUNK = 4


@dataclass(frozen=True)
class Settings:
    ge_api_key: str
    llm_base_url: str
    llm_model: str
    embedding_model: str
    llm_temperature: float
    max_iterations: int
    langsmith_api_key: str
    langsmith_project: str
    project_root: Path

    # ---- directories -------------------------------------------------
    @property
    def docs_dir(self) -> Path:
        return self.project_root / "docs"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    # ---- knowledge base artifacts (written by scripts/ingest.py) -----
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "faiss_index"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.csv"

    # ---- outputs -----------------------------------------------------
    @property
    def wave1_dir(self) -> Path:
        return self.outputs_dir / "wave1_reports"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def approved_dir(self) -> Path:
        return self.outputs_dir / "approved"

    @property
    def eval_dir(self) -> Path:
        return self.outputs_dir / "evaluation"

    @property
    def runs_path(self) -> Path:
        return self.outputs_dir / "runs.jsonl"

    @property
    def reviews_path(self) -> Path:
        return self.outputs_dir / "reviews.jsonl"

    @property
    def eval_cases_path(self) -> Path:
        return self.project_root / "evaluation" / "eval_cases.json"

    def ensure_dirs(self) -> None:
        for d in (self.docs_dir, self.cache_dir, self.wave1_dir, self.reports_dir,
                  self.approved_dir, self.eval_dir):
            d.mkdir(parents=True, exist_ok=True)
        for f in (self.runs_path, self.reviews_path):
            f.touch(exist_ok=True)

    @property
    def has_api_key(self) -> bool:
        return bool(self.ge_api_key and self.ge_api_key != "YOUR_CAPGEMINI_GENERATIVE_ENGINE_KEY")


def load_settings(project_root: Path | None = None) -> Settings:
    root = Path(project_root) if project_root else PROJECT_ROOT
    load_dotenv(root / ".env", override=False)
    settings = Settings(
        ge_api_key=os.getenv("GE_API_KEY", "").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://openai.generative.engine.capgemini.com/v1").strip(),
        llm_model=os.getenv("LLM_MODEL", "openai.gpt-5-mini").strip(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0").strip(),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", "").strip(),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "pharma-design-report-agent").strip(),
        project_root=root,
    )
    settings.ensure_dirs()
    return settings
