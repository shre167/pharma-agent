"""Fail-closed intake gate: resolve the medicine (by name OR description) against handbook vocabulary."""
from __future__ import annotations

import re
from typing import Callable

from ..utils import parse_json
from .prompts import SCOPE_PROMPT

_NON_MEDICINE_MENTIONS = {
    "medicine", "medicines", "tablet", "tablets", "capsule", "capsules", "drug", "drugs",
    "pharmaceutical", "pharmaceuticals", "formulation", "formulations", "drug product",
    "manufacturing process", "process", "report", "medicine tablet", "tablet variant",
}

LLMChat = Callable[[list[dict]], str]


def _blocked(reason: str) -> dict:
    return {
        "is_medicine_request": False, "reason": reason, "medicine_mentions": [],
        "matched_handbook_ingredients": [], "active_ingredients_identified": [],
        "generic_names": [], "identification_method": "unknown", "confidence": "low",
        "candidate_ingredients": [],
    }


def check_scope(vocab: dict, llm_chat: LLMChat, query: str) -> dict:
    """Identify explicit or description-implied medicine identity, validated against `vocab`.

    Only an exact handbook-vocabulary entry can open the gate. Descriptions need a single
    high/medium-confidence ingredient; explicit names must literally occur in the query.
    """
    vocab_list = "\n".join(sorted(vocab))
    raw = llm_chat([
        {"role": "system", "content": SCOPE_PROMPT},
        {"role": "user", "content": f"HANDBOOK_INGREDIENTS:\n{vocab_list}\n\nUSER REQUEST:\n{query[:4000]}"},
    ])
    try:
        scope = parse_json(raw)
        if not isinstance(scope, dict):
            raise ValueError
    except Exception:
        return _blocked("Scope classifier returned invalid JSON; retrieval was blocked.")

    def exact(values) -> list[str]:
        return sorted({str(x).strip().casefold() for x in (values or [])
                       if isinstance(x, str) and str(x).strip().casefold() in vocab})

    matched = exact(scope.get("matched_handbook_ingredients"))
    active = exact(scope.get("active_ingredients_identified"))

    q = re.sub(r"\s+", " ", str(query or "")).strip().casefold()
    explicit = []
    for value in scope.get("medicine_mentions", []) or []:
        if isinstance(value, str):
            name = re.sub(r"\s+", " ", value).strip().strip(".,;:!?()[]{}\"'").casefold()
            if name and name not in _NON_MEDICINE_MENTIONS and name in q:
                explicit.append(name)

    method = str(scope.get("identification_method") or ("explicit_name" if explicit else "description")).strip().casefold()
    if method not in {"explicit_name", "description"}:
        method = "explicit_name" if explicit else "description"
    confidence = str(scope.get("confidence", "low")).strip().casefold()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    if not matched and active:
        matched = active
    if method == "description":
        verified = matched if len(matched) == 1 and confidence in {"high", "medium"} else []
    else:
        verified = matched if explicit and matched else []

    scope["identification_method"] = method
    scope["confidence"] = confidence
    scope["medicine_mentions"] = sorted(set(explicit + verified))
    scope["matched_handbook_ingredients"] = verified
    scope["active_ingredients_identified"] = active or verified
    scope["generic_names"] = verified
    scope["is_medicine_request"] = bool(verified)
    scope["candidate_ingredients"] = scope.get("candidate_ingredients", []) or []
    if not verified:
        scope["reason"] = scope.get("reason") or (
            "The description did not produce one sufficiently confident active ingredient "
            "that exactly matches the handbook vocabulary.")
    return scope
