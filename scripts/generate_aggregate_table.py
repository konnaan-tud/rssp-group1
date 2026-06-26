"""
scripts/generate_aggregate_table.py
-----------------------------------
Aggregate the per-session results into the condition-level table that
answers the two research questions / tests H1.

It reuses the run->session mapping and the per-session metric extraction
from `generate_results_table.py`, so there is a single source of truth for
which sessions belong to which recipe/condition.

Per condition (obs_only / wh / polar) it reports, across all recipes:

  - n_sessions
  - accuracy                 e.g. "5/6"  (Acc averaged over recipes)
  - total_questions          sum of #Q over all sessions
  - pooled_mean_ig_q         sum(IG_Q) / sum(#Q)   <- bits per turn, pooled
  - sess_mean_ig_q           mean over sessions of (IG_Q / #Q per session)
  - mean_ig_total            mean session IG_total = H_0 - H_T
  - mean_final_entropy       mean final entropy H_T

`pooled_mean_ig_q` (and `sess_mean_ig_q`) are the quantities H1 compares
between polar and wh. The script also prints a per-recipe wh-vs-polar
winner tally and a one-line H1 verdict.

Outputs:
  outputs/results_aggregate.csv
  outputs/results_aggregate.md
Also prints to the terminal.

Run:
    python scripts/generate_aggregate_table.py
"""

from __future__ import annotations

import csv
from pathlib import Path

# Reuse the mapping + per-session analysis from the per-recipe script.
# (When run as `python scripts/generate_aggregate_table.py`, the script's
#  own directory is on sys.path, so this import resolves.)
from generate_results_table import (
    RUNS,
    CONDITION_ORDER,
    N_RECIPES,
    UNIFORM_H,
    analyse_session,
    ROOT,
    EVAL_DIR,
)

OUT_CSV = EVAL_DIR / "results_aggregate.csv"
OUT_MD = EVAL_DIR / "results_aggregate.md"

COLUMNS = [
    "condition",
    "n_sessions",
    "accuracy",
    "total_questions",
    "pooled_mean_ig_q",   # H1 metric: sum(IG_Q) / sum(#Q)
    "sess_mean_ig_q",     # H1 metric: mean of per-session (IG_Q/#Q)
    "mean_ig_total",
    "mean_final_entropy",
]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate() -> dict[str, dict]:
    """Collect per-session metrics grouped by condition."""
    stats = {c: {
        "n": 0, "correct": 0, "nq": 0, "igq": 0.0,
        "sess_means": [], "ig_total": [], "final_h": [],
    } for c in CONDITION_ORDER}

    for recipe, conditions in RUNS.items():
        for cond in CONDITION_ORDER:
            session_id = conditions.get(cond)
            if not session_id:
                continue
            r = analyse_session(session_id, recipe)
            s = stats[cond]
            s["n"] += 1
            s["correct"] += int(r["correct"])
            s["nq"] += r["n_questions"]
            s["igq"] += r["ig_q"]
            s["ig_total"].append(r["ig_total"])
            s["final_h"].append(r["final_entropy"])
            if r["mean_ig_q_per_turn"] is not None:
                s["sess_means"].append(r["mean_ig_q_per_turn"])
    return stats


def build_rows(stats: dict[str, dict]) -> list[dict]:
    rows = []
    for cond in CONDITION_ORDER:
        s = stats[cond]
        pooled = (s["igq"] / s["nq"]) if s["nq"] else None
        rows.append({
            "condition": cond,
            "n_sessions": s["n"],
            "accuracy": f"{s['correct']}/{s['n']}",
            "total_questions": s["nq"],
            "pooled_mean_ig_q": round(pooled, 3) if pooled is not None else "-",
            "sess_mean_ig_q": round(_mean(s["sess_means"]), 3) if s["sess_means"] else "-",
            "mean_ig_total": round(_mean(s["ig_total"]), 3),
            "mean_final_entropy": round(_mean(s["final_h"]), 3),
        })
    return rows


def per_recipe_h1() -> list[tuple[str, float, float, str]]:
    """For each recipe, mean IG_Q/turn for wh vs polar, and the winner."""
    out = []
    for recipe, conditions in RUNS.items():
        vals = {}
        label = recipe  # falls back to folder name if no wh/polar session
        for cond in ("wh", "polar"):
            sid = conditions.get(cond)
            if sid:
                r = analyse_session(sid, recipe)
                vals[cond] = r["mean_ig_q_per_turn"]
                label = r["recipe"]  # canonical resolved recipe name
            else:
                vals[cond] = None
        if vals["wh"] is None or vals["polar"] is None:
            winner = "n/a"
        else:
            winner = "polar" if vals["polar"] > vals["wh"] else "wh"
        out.append((label, vals["wh"], vals["polar"], winner))
    return out


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def write_csv(rows: list[dict]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    lines = ["| " + " | ".join(COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in COLUMNS) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows: list[dict]) -> None:
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in COLUMNS}
    header = "  ".join(c.ljust(widths[c]) for c in COLUMNS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r[c]).ljust(widths[c]) for c in COLUMNS))


def main() -> None:
    stats = aggregate()
    rows = build_rows(stats)

    print("AGGREGATE (per condition, across all recipes)")
    print(f"H_0 uniform prior = log2({N_RECIPES}) = {UNIFORM_H:.3f} bits\n")
    print_table(rows)

    print("\nH1 check — mean IG_Q per turn, wh vs polar (per recipe):")
    tally = {"wh": 0, "polar": 0}
    for recipe, wh, polar, winner in per_recipe_h1():
        if winner in tally:
            tally[winner] += 1
        print(f"  {recipe:20} wh={_fmt(wh):>6}  polar={_fmt(polar):>6}  -> {winner}")
    print(f"\n  per-recipe wins: polar {tally['polar']}  /  wh {tally['wh']}")

    wh_row = next(r for r in rows if r["condition"] == "wh")
    polar_row = next(r for r in rows if r["condition"] == "polar")
    p, w = polar_row["pooled_mean_ig_q"], wh_row["pooled_mean_ig_q"]
    verdict = "supported (polar > wh)" if (isinstance(p, float) and isinstance(w, float) and p > w) \
        else "not supported"
    print(f"\n  H1 (polar > wh on mean IG_Q/turn): {verdict}  "
          f"[pooled polar={p} vs wh={w} bits/turn]")

    write_csv(rows)
    write_markdown(rows)
    print(f"\nWrote {OUT_CSV.relative_to(ROOT)}  and  {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
