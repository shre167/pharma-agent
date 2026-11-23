"""Small shared helpers."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def normalize_space(s: str | None) -> str:
    return re.sub(r"[ \t]+", " ", (s or "")).strip()


def stable_id(*parts: Any) -> str:
    raw = "||".join(map(str, parts))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strip_json_fences(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip(), flags=re.I)


def parse_json(raw: str) -> Any:
    return json.loads(strip_json_fences(raw))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows
