"""
scripts/generate_results_table.py
---------------------------------
Build the per-recipe / per-condition results table for the evaluation,
using exactly the metrics defined in the proposal (Section 3.3):

  - Final accuracy        Acc = 1[argmax p_T == ground truth]
  - Final entropy         H(I_T)
  - Total information gain IG_total = H(I_0) - H(I_T),  H(I_0) = log2(N)
  - Clarification IG       IG_Q (the primary measure) and mean IG_Q per turn
  - Number of questions    #Q
  - "Speed to correct"     first window where the true recipe became rank 1
  - Final rank of the true recipe

It auto-discovers every session by scanning the folders under
outputs/evaluation/<recipe>/<session_id>/.  The recipe comes from the
<recipe> folder name and the condition from the <session_id> folder name
(which ends in _obs_only / _wh / _polar), so there is no hardcoded
run -> session-id mapping and no reliance on the (sometimes wrong) fields
inside session.json: drop a session folder under outputs/evaluation/ and it
is picked up automatically.

The true recipe is resolved from the folder name against the candidate
recipe list in each session's belief history, and correctness is computed as
predicted == resolved recipe (the stored `accuracy` field is ignored).

Outputs:
  outputs/results_per_recipe.csv   — one row per (recipe, condition)
  outputs/results_per_recipe.md    — same table in Markdown (for the report)
Also prints the table to the terminal.

Run:
    python scripts/generate_results_table.py
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
# Evaluation sessions live under outputs/evaluation/<recipe>/<session_id>/.
EVAL_DIR = ROOT / "outputs" / "evaluation"
OUT_CSV = EVAL_DIR / "results_per_recipe.csv"
OUT_MD = EVAL_DIR / "results_per_recipe.md"

# Number of candidate recipes -> uniform prior entropy H(I_0) = log2(N).
N_RECIPES = 29

# Longest condition tokens first so "obs_only" is matched before any short
# substring could interfere.
CONDITION_ORDER = ["obs_only", "wh", "polar"]

# Generic words that carry no recipe identity, plus version suffixes like v2.
_GENERIC_TOKENS = {"pasta", "sauce_dish"}


def _tokens(name: str) -> set[str]:
    """Lowercase word tokens of a name, dropping version suffixes (v2, v3...)."""
    parts = name.lower().replace("-", "_").replace(" ", "_").split("_")
    return {p for p in parts if p and not re.fullmatch(r"v\d+", p)}


def _condition_from_name(session_folder: str) -> str | None:
    """Parse obs_only / wh / polar out of a session folder name."""
    for cond in CONDITION_ORDER:
        if f"_{cond}" in f"_{session_folder}":
            return cond
    return None


def discover_runs() -> dict[str, dict[str, str]]:
    """Scan EVAL_DIR folders and build recipe_folder -> {condition: session_id}.

    Recipe = the <recipe> folder name; condition = parsed from the session
    folder name.  Nothing inside session.json is consulted here, so the
    discovery is driven purely by the folders that are always present.
    Recipes are returned in sorted order for stable output.
    """
    runs: dict[str, dict[str, str]] = {}
    for recipe_dir in sorted(p for p in EVAL_DIR.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in recipe_dir.iterdir() if p.is_dir()):
            if not (session_dir / "session.json").exists():
                continue
            cond = _condition_from_name(session_dir.name)
            if cond is None:
                print(f"  ! skipping {session_dir}: no condition in folder name")
                continue
            runs.setdefault(recipe_dir.name, {})[cond] = session_dir.name
    return dict(sorted(runs.items()))


def resolve_recipe(recipe_folder: str, candidates: list[str]) -> str:
    """Map a recipe folder name to its canonical recipe name.

    The canonical names are the candidate recipes seen in the belief history
    (e.g. "amatriciana pasta").  A folder like "amatriciana_v2" matches the
    candidate whose tokens are a superset of the folder's tokens.  If nothing
    matches, fall back to the cleaned folder name.
    """
    folder_tokens = _tokens(recipe_folder) - _GENERIC_TOKENS
    for cand in candidates:
        if folder_tokens and folder_tokens <= _tokens(cand):
            return cand
    return " ".join(sorted(_tokens(recipe_folder)))


# recipe_folder -> {condition: session_id}, discovered from the folders.
RUNS: dict[str, dict[str, str]] = discover_runs()

# Column order for the CSV / Markdown / console table.
COLUMNS = [
    "recipe",
    "condition",
    "correct",
    "predicted",
    "final_rank",
    "first_top_window",   # speed to correct
    "final_entropy",
    "ig_total",
    "n_questions",
    "ig_q",
    "mean_ig_q_per_turn",
]

UNIFORM_H = math.log2(N_RECIPES)


# ─────────────────────────────────────────────────────────────────────────
# Per-session metric extraction
# ─────────────────────────────────────────────────────────────────────────

def analyse_session(session_id: str, recipe_folder: str) -> dict:
    """Load one session.json and compute the proposal's metrics for it.

    `recipe_folder` is the evaluation sub-folder name; the canonical recipe
    (ground truth) is resolved from it against the belief-history candidates,
    and correctness is `predicted == resolved recipe` (the stored `accuracy`
    field is ignored because it can be wrong).
    """
    # Sessions are nested one level deep under EVAL_DIR (per recipe), so
    # locate the session folder by id regardless of which recipe folder
    # it sits in.
    matches = list(EVAL_DIR.glob(f"*/{session_id}/session.json"))
    if not matches:
        raise FileNotFoundError(
            f"No session.json for {session_id!r} under {EVAL_DIR}"
        )
    session_path = matches[0]

    data = json.loads(session_path.read_text(encoding="utf-8"))
    result = data["result"]
    history = data.get("belief_history", [])

    # Candidate recipe names appearing in the belief history, used to resolve
    # the folder name to a canonical recipe (e.g. amatriciana_v2 -> amatriciana pasta).
    candidates: list[str] = []
    seen: set[str] = set()
    for row in history:
        for k in row:
            if k.startswith("p::"):
                name = k[len("p::"):]
                if name not in seen:
                    seen.add(name)
                    candidates.append(name)
    ground_truth = resolve_recipe(recipe_folder, candidates)

    # Walk the belief history to find (a) the first window where the true
    # recipe is ranked #1, and (b) its final rank.
    first_top_window: int | None = None
    final_rank: int | None = None

    for row in history:
        probs = {
            k.replace("p::", ""): v
            for k, v in row.items()
            if k.startswith("p::")
        }
        if not probs:
            continue
        step = int(row.get("step", 0))
        ordered = sorted(probs, key=probs.get, reverse=True)
        if first_top_window is None and ordered and ordered[0] == ground_truth:
            first_top_window = step
        final_rank = (ordered.index(ground_truth) + 1) if ground_truth in ordered else None

    predicted = result.get("predicted", "?")
    final_entropy = float(result["final_entropy"])
    n_questions = int(result["questions_asked"])
    ig_q = float(result["total_ig_q"] or 0.0)
    mean_ig_q = (ig_q / n_questions) if n_questions else None

    return {
        "recipe": ground_truth,
        "condition": data.get("condition") or _condition_from_name(session_id) or "?",
        "correct": predicted == ground_truth,
        "predicted": predicted,
        "final_rank": final_rank,
        "first_top_window": first_top_window,
        "final_entropy": round(final_entropy, 3),
        "ig_total": round(UNIFORM_H - final_entropy, 3),
        "n_questions": n_questions,
        "ig_q": round(ig_q, 3),
        "mean_ig_q_per_turn": round(mean_ig_q, 3) if mean_ig_q is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────
# Table assembly + writers
# ─────────────────────────────────────────────────────────────────────────

def build_rows() -> list[dict]:
    rows: list[dict] = []
    for recipe, conditions in RUNS.items():
        for cond in CONDITION_ORDER:
            session_id = conditions.get(cond)
            if not session_id:
                continue
            rows.append(analyse_session(session_id, recipe))
    return rows


def _fmt(value) -> str:
    return "-" if value is None else str(value)


def write_csv(rows: list[dict]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _fmt(row[c]) for c in COLUMNS})


def write_markdown(rows: list[dict]) -> None:
    lines = ["| " + " | ".join(COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[c]) for c in COLUMNS) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    prev_recipe = None
    for row in rows:
        if prev_recipe is not None and row["recipe"] != prev_recipe:
            print()  # blank line between recipes
        prev_recipe = row["recipe"]
        print("  ".join(_fmt(row[c]).ljust(widths[c]) for c in COLUMNS))


def main() -> None:
    rows = build_rows()
    print_table(rows)
    write_csv(rows)
    write_markdown(rows)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}  and  {OUT_MD.relative_to(ROOT)}")
    print(f"(H_0 uniform prior = log2({N_RECIPES}) = {UNIFORM_H:.3f} bits)")


if __name__ == "__main__":
    main()
