"""
utils/clip_loader.py
--------------------
Auto-discover the per-window video clips for an orchestrator session.

All three orchestrators (obs_only, wh, polar) previously hardcoded their
WINDOWS list to one specific recipe. This module centralises clip
discovery so the team can drop in any new recipe folder under
`data/clips/<name>/` and the orchestrators pick it up via `--recipe <name>`
— no code changes needed.

    >>> from utils.clip_loader import discover_clips
    >>> clips, ground_truth = discover_clips("arrabbiata")
    >>> clips
    [PosixPath('data/clips/arrabbiata/01_pour_water.mp4'), ...]
    >>> ground_truth
    'arrabbiata pasta'

═══════════════════════════════════════════════════════════════════════════
FOLDER NAMING — three valid options
═══════════════════════════════════════════════════════════════════════════

1. EXACT recipe name (with spaces → underscores).
       Recipe "carbonara bianca" → folder  data/clips/carbonara_bianca/
       Recipe "cacio e pepe"     → folder  data/clips/cacio_e_pepe/

2. PREFIX SHORTHAND — the unique recipe in italian_recipe_scenes.json
   whose name starts with `<folder_name> ` will be resolved automatically.
       Folder  data/clips/arrabbiata/  → recipe "arrabbiata pasta"
       Folder  data/clips/pesto/       → recipe "pesto pasta"
       Folder  data/clips/clam/        → recipe "clam pasta"
       Folder  data/clips/carbonara/   → recipe "carbonara" (exact)

3. MANIFEST FILE — drop a `recipe.txt` in the folder whose first line is
   the canonical recipe name. Use this when the folder name is ambiguous
   (e.g. "pea" matches both "pea pesto pasta" and "pea butter pasta") or
   when you want a memorable short folder name.
       data/clips/my_run/recipe.txt   →  "carbonara"

═══════════════════════════════════════════════════════════════════════════
CLIP FILE NAMING — sorted lexicographically
═══════════════════════════════════════════════════════════════════════════

The orchestrators iterate clips in alphabetical order, so use leading
zeros to control the sequence:

    01_pour_water.mp4
    02_crack_egg.mp4
    03_grate_pecorino.mp4
    ...

Supported extensions: .mp4, .mov, .mkv, .webm, .avi

═══════════════════════════════════════════════════════════════════════════
ADDING A NEW RECIPE THAT'S NOT IN THE JSON YET
═══════════════════════════════════════════════════════════════════════════

The belief updater only knows recipes that are listed in
`data/italian_recipe_scenes.json` (and pre-embedded into
`data/recipe_sequences.json`). If your test recipe isn't in there:

    1. Add the recipe's ordered scenes to italian_recipe_scenes.json
    2. Re-embed:  python main.py --mode bootstrap
    3. Create your clip folder under data/clips/<name>/
    4. Run:       python orchestrator_v2_vlm.py --recipe <name> --answer-mode text

The clip_loader refuses to start a session if the resolved recipe name
isn't found in italian_recipe_scenes.json — the error message names the
available recipes so you can fix the folder name in one shot.
"""

from __future__ import annotations

import json
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_CLIPS_ROOT = Path("data/clips")
DEFAULT_RECIPES_JSON = Path("data/italian_recipe_scenes.json")

# File extensions we treat as video clips. Add more if you ever feed in
# non-mp4 sources.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# Optional per-folder manifest file. First line = canonical recipe name.
# Used to override auto-resolution when the folder name is ambiguous.
MANIFEST_FILENAME = "recipe.txt"


# ─────────────────────────────────────────────────────────────────────────
# Recipe-name resolution
# ─────────────────────────────────────────────────────────────────────────

def _load_recipe_names(recipes_path: Path = DEFAULT_RECIPES_JSON) -> list[str]:
    """Return the list of recipe `dish` names from italian_recipe_scenes.json."""
    if not recipes_path.exists():
        return []
    with open(recipes_path, encoding="utf-8") as f:
        data = json.load(f)
    return [r["dish"] for r in data]


def _normalize_folder_name(folder_name: str) -> str:
    """Convert underscores to spaces (folder convention → readable name)."""
    return folder_name.replace("_", " ").strip()


def _resolve_ground_truth(
    folder_name: str,
    folder_path: Path,
    recipes_path: Path = DEFAULT_RECIPES_JSON,
) -> str:
    """
    Resolve a clip-folder name to the canonical recipe name in
    italian_recipe_scenes.json. Resolution order:

      1. Manifest:    `<folder>/recipe.txt` first line, if present
      2. Exact match: folder name (underscores→spaces) IS a recipe name
      3. Prefix:      exactly one recipe starts with `<folder_name> `
      4. Fail:        clear error listing available recipes

    Raises ValueError when the folder cannot be resolved. The caller
    should let the exception propagate up to main() — printing a useful
    error is better than silently scoring against the wrong recipe.
    """
    available = _load_recipe_names(recipes_path)

    # 1. Manifest override
    manifest = folder_path / MANIFEST_FILENAME
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8").strip()
        if text:
            recipe = text.splitlines()[0].strip()
            if available and recipe not in available:
                raise ValueError(
                    f"{manifest} declares recipe {recipe!r}, but that "
                    f"recipe is not in {recipes_path}. Available recipes: "
                    f"{', '.join(sorted(available))}"
                )
            return recipe

    normalized = _normalize_folder_name(folder_name)
    if not available:
        # JSON missing — degrade gracefully and use folder name as-is.
        return normalized

    # 2. Exact match
    if normalized in available:
        return normalized

    # 3. Unique prefix match — recipe starts with "<folder_name> "
    prefix_matches = [
        r for r in available
        if r == normalized or r.startswith(normalized + " ")
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ValueError(
            f"Folder name {folder_name!r} is ambiguous — matches multiple "
            f"recipes: {', '.join(prefix_matches)}.\n"
            f"Either rename the folder to be more specific (e.g. "
            f"'{prefix_matches[0].replace(' ', '_')}'), or add a "
            f"{MANIFEST_FILENAME} file in the folder containing the exact "
            f"recipe name."
        )

    # 4. Nothing matched
    raise ValueError(
        f"Folder name {folder_name!r} does not match any recipe in "
        f"{recipes_path}.\n"
        f"Either rename the folder to match an existing recipe "
        f"(e.g. 'arrabbiata' or 'cacio_e_pepe'), add a {MANIFEST_FILENAME} "
        f"file in the folder with the exact recipe name, or add the recipe "
        f"to italian_recipe_scenes.json and re-bootstrap.\n"
        f"Available recipes: {', '.join(sorted(available))}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def discover_clips(
    recipe: str,
    clips_root: Path | str = DEFAULT_CLIPS_ROOT,
    recipes_path: Path = DEFAULT_RECIPES_JSON,
) -> tuple[list[Path], str]:
    """
    Return `(sorted clip paths, canonical recipe name)` for `recipe`.

    `recipe` is the **folder name** under `clips_root` (NOT necessarily
    the recipe name in the JSON — auto-resolution handles the mapping).
    """
    root = Path(clips_root)
    session_dir = root / recipe

    if not session_dir.is_dir():
        available = available_recipes(root)
        hint = (
            f"\nAvailable folders in {root}/: {', '.join(available)}"
            if available
            else f"\n{root} is empty or missing — "
                 f"build clips first with: python scripts/prepare_clips.py"
        )
        raise FileNotFoundError(
            f"No clips folder for {recipe!r} at {session_dir}.{hint}"
        )

    clips = sorted(
        p for p in session_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTS
    )
    if not clips:
        raise FileNotFoundError(
            f"Folder {session_dir} exists but contains no video files "
            f"({', '.join(VIDEO_EXTS)})."
        )

    ground_truth = _resolve_ground_truth(recipe, session_dir, recipes_path)
    return clips, ground_truth


def discover_clips_from_dir(
    clips_dir: Path | str,
    ground_truth: str | None = None,
) -> tuple[list[Path], str]:
    """
    Alternative entry point: skip the `data/clips/<recipe>/` convention
    and point directly at an arbitrary directory of clips.

    `ground_truth` falls back to the directory's own name if not given.
    Useful for ad-hoc sessions where the clips live outside the repo.
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


def available_recipes(
    clips_root: Path | str = DEFAULT_CLIPS_ROOT,
) -> list[str]:
    """
    Folder names under `clips_root` that contain clip files. Used to
    populate the `--recipe` argparse choices and to print a useful hint
    when the user passes an unknown name.

    Folders whose name starts with `_` or `.` are IGNORED — use this
    convention to mark deprecated, backup, or work-in-progress folders
    that you don't want appearing in the orchestrator's CLI choices.
    Examples that get ignored:  `_old_pesto/`, `.scratch/`, `_backup/`
    """
    root = Path(clips_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )
