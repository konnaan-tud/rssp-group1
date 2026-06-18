"""
dialogue/
---------
Human-answer parsing and normalisation utilities.

This package is separate from `observation_pipeline/` because it concerns
free-form human text answers to clarification questions, not VLM video
observations.

Main entry point:
    from dialogue.answer_normalizer import normalize_answer
"""

from .answer_normalizer import (
    AnswerOutcome,
    AnswerType,
    classify_polar_answer,
    normalize_answer,
    to_recipe_sentence,
)

__all__ = [
    "AnswerOutcome",
    "AnswerType",
    "classify_polar_answer",
    "normalize_answer",
    "to_recipe_sentence",
]
