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

It reads each session's `session.json` from outputs/sessions/<id>/ using the
run -> session-id mapping below (taken from "RSSP Group 1.pdf"). Edit RUNS to
add or swap sessions.

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
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = ROOT / "outputs" / "sessions"
OUT_CSV = ROOT / "outputs" / "results_per_recipe.csv"
OUT_MD = ROOT / "outputs" / "results_per_recipe.md"

# Number of candidate recipes -> uniform prior entropy H(I_0) = log2(N).
N_RECIPES = 29

# recipe -> {condition: session_id}.  Edit this when you run more recipes.
RUNS: dict[str, dict[str, str]] = {
    "arrabbiata pasta": {
        "obs_only": "20260620_182126_obs_only",
        "wh":       "20260620_182928_wh",
        "polar":    "20260620_185202_polar",
    },
    "vodka sauce pasta": {
        "obs_only": "20260620_190720_obs_only",
        "wh":       "20260620_190939_wh",
        "polar":    "20260620_192006_polar",
    },
    "shrimp garlic pasta": {
        "obs_only": "20260620_194850_obs_only",
        "wh":       "20260620_195046_wh",
        "polar":    "20260620_195613_polar",
    },
    "alfredo pasta": {
        "obs_only": "20260620_200008_obs_only",
        "wh":       "20260620_200254_wh",
        "polar":    "20260620_200655_polar",
    },
    "amatriciana pasta": {
        "obs_only": "20260620_200956_obs_only",
        "wh":       "20260620_201155_wh",
        "polar":    "20260620_201527_polar",
    },
    "clam pasta": {
        "obs_only": "20260620_201904_obs_only",
        "wh":       "20260620_202028_wh",
        "polar":    "20260620_202447_polar",
    },
}

CONDITION_ORDER = ["obs_only", "wh", "polar"]

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

def analyse_session(session_id: str, ground_truth: str) -> dict:
    """Load one session.json and compute the proposal's metrics for it."""
    session_path = SESSIONS_DIR / session_id / "session.json"
    if not session_path.exists():
        raise FileNotFoundError(f"Missing session file: {session_path}")

    data = json.loads(session_path.read_text(encoding="utf-8"))
    result = data["result"]
    history = data.get("belief_history", [])

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

    final_entropy = float(result["final_entropy"])
    n_questions = int(result["questions_asked"])
    ig_q = float(result["total_ig_q"] or 0.0)
    mean_ig_q = (ig_q / n_questions) if n_questions else None

    return {
        "recipe": ground_truth,
        "condition": data.get("condition", "?"),
        "correct": bool(result["accuracy"]),
        "predicted": result.get("predicted", "?"),
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
