"""Vocabulary helpers derived from handbook headings (no manual medicine dictionaries)."""
from __future__ import annotations

import re

FORM_WORDS = [
    "tablet", "tablets", "caplet", "caplets", "capsule", "capsules",
    "sachet", "sachets", "granules", "suspension", "syrup", "drops",
    "suppository", "suppositories", "cream", "gel", "solution", "powder",
]

STRIP_WORDS = set(FORM_WORDS) | {
    "and", "with", "for", "oral", "film", "coated", "chewable", "sustained", "extended",
    "release", "immediate", "delayed", "effervescent", "dispersible", "enteric", "mg",
    "mcg", "ml", "g", "usp", "tablet", "tablets", "capsule", "capsules", "injection",
    "ointment", "lotion", "suspension",
}


def ingredient_phrase(heading: str) -> str:
    """'ACETAMINOPHEN CHEWABLE TABLETS' -> 'acetaminophen'."""
    words = re.findall(r"[a-z][a-z\-']*", heading.lower())
    return " ".join(w for w in words if w not in STRIP_WORDS).strip()


def detect_dosage_form(heading: str) -> str:
    low = heading.lower()
    for word in FORM_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", low):
            return word
    return "UNKNOWN"
