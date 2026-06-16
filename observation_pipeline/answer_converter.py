"""
observation_pipeline/answer_converter.py
----------------------------------------
Normalize human answers into recipe-step style scene sentences so they fit
the same language as the recipe database.
"""

from __future__ import annotations

import re


_LEADING_PATTERNS = [
    re.compile(r"^(?:i am|i'm|im|we are|we're)\s+", re.IGNORECASE),
    re.compile(r"^(?:the cook is)\s+", re.IGNORECASE),
]

_ING_OVERRIDES = {
    "adding": "adds",
    "boiling": "boils",
    "chopping": "chops",
    "cooking": "cooks",
    "cracking": "cracks",
    "cutting": "cuts",
    "dicing": "dices",
    "draining": "drains",
    "dropping": "drops",
    "grating": "grates",
    "heating": "heats",
    "mixing": "mixes",
    "pouring": "pours",
    "putting": "puts",
    "rendering": "renders",
    "serving": "serves",
    "simmering": "simmers",
    "stirring": "stirs",
    "tossing": "tosses",
    "transferring": "transfers",
    "using": "uses",
    "whisking": "whisks",
}


def _ensure_period(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?")) else text + "."


def _convert_gerund(word: str) -> str:
    lower = word.lower()
    if lower in _ING_OVERRIDES:
        return _ING_OVERRIDES[lower]
    if lower.endswith("ying") and len(lower) > 5:
        return lower[:-4] + "ies"
    if lower.endswith("ing") and len(lower) > 4:
        stem = lower[:-3]
        if stem.endswith(("tt", "pp", "gg", "nn")):
            stem = stem[:-1]
        if stem.endswith("at") or stem.endswith("it"):
            return stem + "es"
        return stem + "s"
    return lower


def convert_answer_to_recipe_step(answer: str) -> str:
    """
    Convert a free-form typed answer into the recipe-step sentence style
    used throughout the pipeline.
    """
    text = answer.strip().strip('"').strip("'").strip()
    if not text:
        return ""

    if text.lower().startswith("a cook "):
        return _ensure_period(text[0].upper() + text[1:])

    for pattern in _LEADING_PATTERNS:
        text = pattern.sub("", text)

    text = text.strip()
    if not text:
        return ""

    negation_markers = ("don't", "do not", "not using", "no ", "doesn't")
    if text.lower().startswith(negation_markers):
        return _ensure_period(answer.strip())

    match = re.match(r"^([A-Za-z]+)\b(.*)$", text)
    if not match:
        return _ensure_period(answer.strip())

    first_word = match.group(1)
    remainder = match.group(2).strip()
    verb = _convert_gerund(first_word)

    normalized = f"A cook {verb}"
    if remainder:
        normalized += f" {remainder}"

    normalized = normalized[0].upper() + normalized[1:]
    return _ensure_period(normalized)
