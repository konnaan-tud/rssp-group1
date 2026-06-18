"""
utils/clip_loader.py
--------------------
Auto-discover the per-window video clips for an orchestrator session.

All three orchestrators (obs_only, wh, polar) previously hardcoded their
WINDOWS list to nine carbonara clips. To run a different recipe — pesto,
amatriciana, anything else — you had to edit the orchestrator source.

This module centralises that:

    >>> from utils.clip_loader import discover_clips
    >>> clips, ground_truth = discover_clips("carbonara")
    >>> clips
    [PosixPath('data/clips/carbonara/01_pour_water.mp4'), ...]

Layout convention
-----------------
data/clips/<recipe_name>/<NN_step_label>.mp4

Files are returned **sorted alphabetically** so the orchestrator iterates
them in recipe order. We rely on the user numbering the clip files (e.g.
`01_`, `02_`, ...) or naming them lexicographically (`video_01.mp4`,
`video_02.mp4`, ...). Any file in the directory that isn't a video is
ignored.

The `ground_truth` returned is just the recipe name passed in — the
orchestrators previously had a separate `GROUND_TRUTH = "carbonara"`
constant for the human stub. With auto-discovery the two are the same
string.
"""

from __future__ import annotations

from pathlib import Path


# File extensions we treat as video clips. Add more if you ever feed in
# non-mp4 sources.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# Where the recipe-specific subfolders live by convention. Override with
# an absolute path if the user keeps clips elsewhere.
DEFAULT_CLIPS_ROOT = Path("data/clips")


# ─────────────────────────────────────────────────────────────────────────
# Ground-truth aliases
#
# Some recipes in `data/italian_recipe_scenes.json` have multi-word names
# ("pesto pasta") while the user's clip folder is a single short word
# ("data/clips/pesto/"). The folder name is what the user types on the
# command line (`--recipe pesto`) and what's easy to organise on disk.
# The recipe name in the JSON is what `BeliefUpdaterV3` returns from
# `bu.top_recipe()` and what the accuracy check compares against.
#
# This dict maps the CLI/folder name → the recipe-name string. The
# entries below cover every dish that has a clip folder; add a new
# entry whenever you add a new clip folder whose name doesn't match
# the recipe label.
# ─────────────────────────────────────────────────────────────────────────

_GROUND_TRUTH_ALIASES: dict[str, str] = {
    "carbonara": "carbonara",     # folder name == recipe name (no-op)
    "pesto":     "pesto pasta",   # folder is short; recipe label is two words
}


def discover_clips(
    recipe: str,
    clips_root: Path | str = DEFAULT_CLIPS_ROOT,
) -> tuple[list[Path], str]:
    """
    Return (sorted clip paths, ground-truth recipe name) for a recipe.

    Parameters
    ----------
    recipe
        Recipe name. This is both the subfolder under `clips_root` and
        the ground-truth label the orchestrator uses for the human stub
        and the session summary. Whitespace and case are preserved
        because `BeliefUpdaterV3` looks up recipes by their exact name in
        `recipe_sequences.json`.
    clips_root
        Parent directory containing per-recipe subfolders. Defaults to
        `data/clips`. Pass an absolute path if you keep clips elsewhere.

    Raises
    ------
    FileNotFoundError
        If `clips_root / recipe` doesn't exist or contains no recognised
        video files. The error message names every searched path so the
        user can fix the layout without digging.
    """
    root = Path(clips_root)
    session_dir = root / recipe

    if not session_dir.is_dir():
        # Help the user with a list of what IS available so they can fix
        # the typo in one shot.
        available = (
            sorted(p.name for p in root.iterdir() if p.is_dir())
            if root.exists() else []
        )
        hint = (
            f"\nAvailable recipes in {root}/: " + ", ".join(available)
            if available
            else f"\n{root} is empty or missing — "
                 f"build clips first with: python scripts/prepare_clips.py"
        )
        raise FileNotFoundError(
            f"No clips directory for recipe {recipe!r} at {session_dir}.{hint}"
        )

    clips = sorted(
        p for p in session_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )

    if not clips:
        raise FileNotFoundError(
            f"Clips directory {session_dir} exists but contains no "
            f"video files ({', '.join(VIDEO_EXTS)})."
        )

    # Translate the folder/CLI name to the recipe label in
    # italian_recipe_scenes.json. If the user typed an unknown name we
    # echo it back unchanged so the accuracy check fails loudly with the
    # missing-recipe message instead of silently scoring against the
    # wrong label.
    ground_truth = _GROUND_TRUTH_ALIASES.get(recipe, recipe)
    return clips, ground_truth


def available_recipes(
    clips_root: Path | str = DEFAULT_CLIPS_ROOT,
) -> list[str]:
    """
    Recipe names that have a clip directory on disk. Used to populate
    the `--recipe` argparse choices and to print a useful hint when the
    user passes an unknown name.
    """
    root = Path(clips_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def discover_clips_from_dir(
    clips_dir: Path | str,
    ground_truth: str | None = None,
) -> tuple[list[Path], str]:
    """
    Alternative entry point: skip the `data/clips/<recipe>/` convention
    and point directly at an arbitrary directory of clips.

    The ground-truth recipe name is taken from `ground_truth` if given,
    otherwise it defaults to the directory's own name. Useful for ad-hoc
    sessions where the clips live outside the repo.
    """
    clips_dir = Path(clips_dir)
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    clips = sorted(
        p for p in clips_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not clips:
        raise FileNotFoundError(
            f"Clips directory {clips_dir} contains no video files."
        )

    return clips, ground_truth or clips_dir.name
