"""
utils/sessions.py
-----------------
Resolve a CLI dish name (e.g. "carbonara", "pesto") into:

  - the ordered list of clip files for that dish
  - the recipe label used in `data/italian_recipe_scenes.json`
    (the ground truth the belief updater will be scored against)
  - the clip directory itself (useful for diagnostic messages)

Replaces the three identical hardcoded `WINDOWS` + `GROUND_TRUTH` blocks
at the top of each orchestrator. Adding a new dish now means adding a
clip folder under `data/clips/<dish>/` and one entry in
`_GROUND_TRUTH_MAP` below — orchestrators don't change at all.

Layout convention
-----------------
`data/clips/<dish>/*.mp4` — clip files are loaded in lexicographic order,
so name them with a leading number (`01_...`, `02_...`, `video_01.mp4`,
etc.) so the ordering matches the recipe's intended sequence.

Usage
-----
    from utils.sessions import load_session
    session = load_session("pesto")
    for clip in session.windows:
        ...
    bu = BeliefUpdaterV3()
    accuracy = (bu.top_recipe()[0] == session.ground_truth)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────
# Dish → ground-truth recipe-name mapping
#
# The left key is the CLI argument name and the clip-folder name. The
# right value is the recipe name as it appears in
# `data/italian_recipe_scenes.json` — what `bu.top_recipe()` returns.
# Keep these two consistent: a typo here makes the accuracy report wrong.
# ─────────────────────────────────────────────────────────────────────────

_GROUND_TRUTH_MAP: dict[str, str] = {
    "carbonara":     "carbonara",
    "pesto":         "pesto pasta",
}


CLIPS_ROOT = Path("data") / "clips"


@dataclass(frozen=True)
class SessionConfig:
    """One cooking session's clips and ground-truth label."""

    dish: str               # CLI alias and clip-folder name
    ground_truth: str       # recipe label in italian_recipe_scenes.json
    clips_dir: Path         # data/clips/<dish>/
    windows: tuple[Path, ...]  # ordered .mp4 paths


def available_dishes() -> list[str]:
    """
    Dishes that have BOTH a ground-truth mapping AND a clip directory
    on disk. Used to populate the `--dish` argparse choices.
    """
    return sorted(
        dish for dish in _GROUND_TRUTH_MAP
        if (CLIPS_ROOT / dish).is_dir()
    )


def load_session(
    dish: str,
    *,
    clips_root: Path = CLIPS_ROOT,
) -> SessionConfig:
    """
    Resolve `dish` to a `SessionConfig`. Raises if either the dish is
    unknown or its clip directory is missing/empty — the orchestrators
    catch the error and print a useful message.
    """
    if dish not in _GROUND_TRUTH_MAP:
        choices = ", ".join(sorted(_GROUND_TRUTH_MAP))
        raise ValueError(
            f"Unknown dish {dish!r}. Available: {choices}. "
            f"Add a new dish by extending _GROUND_TRUTH_MAP in utils/sessions.py."
        )

    clips_dir = clips_root / dish
    if not clips_dir.is_dir():
        raise FileNotFoundError(
            f"Clip directory not found for dish {dish!r}: {clips_dir}. "
            f"Either run scripts/prepare_clips.py, or create the folder "
            f"and drop ordered .mp4 files in it."
        )

    windows = tuple(sorted(clips_dir.glob("*.mp4")))
    if not windows:
        raise FileNotFoundError(
            f"No .mp4 clips found in {clips_dir}. "
            f"Drop ordered clip files in the folder and re-run."
        )

    return SessionConfig(
        dish=dish,
        ground_truth=_GROUND_TRUTH_MAP[dish],
        clips_dir=clips_dir,
        windows=windows,
    )


def describe(session: SessionConfig) -> str:
    """Human-readable one-liner for orchestrator headers."""
    return (
        f"dish={session.dish!r}, ground_truth={session.ground_truth!r}, "
        f"{len(session.windows)} clips from {session.clips_dir}"
    )
