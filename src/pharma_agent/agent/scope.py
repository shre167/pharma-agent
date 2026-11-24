"""Intake gate: resolve medicine identity (by explicit name or description) or general pharma query against handbook vocabulary."""
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
    """Identify explicit or description-implied medicine identity or general pharma query validated against `vocab`."""
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
        """Only vocabulary-attested names pass; any LLM-invented name is dropped here."""
        out = set()
        for x in (values or []):
            if not isinstance(x, str):
                continue
            cleaned = str(x).strip().casefold()
            if cleaned in vocab:
                out.add(cleaned)
        return sorted(out)

    matched = exact(scope.get("matched_handbook_ingredients"))
    active = exact(scope.get("active_ingredients_identified"))
    candidates = exact(scope.get("candidate_ingredients"))

    q = re.sub(r"\s+", " ", str(query or "")).strip().casefold()
    explicit = []
    for value in scope.get("medicine_mentions", []) or []:
        if isinstance(value, str):
            name = re.sub(r"\s+", " ", value).strip().strip(".,;:!?()[]{}\"'").casefold()
            if name and name not in _NON_MEDICINE_MENTIONS and name in q:
                explicit.append(name)

    method = str(scope.get("identification_method") or ("explicit_name" if explicit else "description")).strip().casefold()
    if method not in {"explicit_name", "description", "general_pharma"}:
        method = "explicit_name" if explicit else "description"
    confidence = str(scope.get("confidence", "low")).strip().casefold()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    # Fallbacks for ingredients
    if not matched and active:
        matched = active
    if not matched and candidates:
        matched = candidates

    is_explicit_refusal = scope.get("is_medicine_request") is False
    inconclusive = ""

    if is_explicit_refusal:
        verified = []
        is_medicine = False
        inconclusive = "The description was too vague to identify a medicine."
    elif method == "description":
        # A description that maps to at least one handbook ingredient is processed: one match
        # resolves it, several matches let the agent cover every variant with the ambiguity
        # recorded. Refusal only when nothing maps to the handbook vocabulary.
        if matched:
            verified = matched
        else:
            verified = []
            inconclusive = ("The description could not be matched to any handbook ingredient. "
                            "Candidates considered: " + ", ".join(candidates or matched)
                            + ". Please describe the medicine more specifically or name it.")
        is_medicine = bool(verified)
    elif method == "general_pharma":
        verified = matched
        is_medicine = True
    else:  # explicit_name
        verified = matched if explicit and matched else []
        is_medicine = bool(verified)

    scope["identification_method"] = method
    scope["confidence"] = confidence
    scope["medicine_mentions"] = sorted(set(explicit + verified))
    scope["matched_handbook_ingredients"] = verified
    scope["active_ingredients_identified"] = active or verified
    scope["generic_names"] = verified
    scope["is_medicine_request"] = is_medicine
    scope["candidate_ingredients"] = scope.get("candidate_ingredients", []) or []
    if not is_medicine:
        if method == "description" and inconclusive:
            scope["reason"] = scope.get("reason") or inconclusive
        else:
            scope["reason"] = scope.get("reason") or (
                "The request does not appear to be an in-scope pharmaceutical medicine or formulation request.")
    return scope
