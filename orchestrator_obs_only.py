"""
orchestrator_obs_only.py
------------------------
Observations-only condition (Condition 1 baseline).

Walks through the cooking clips in order, runs each through Qwen2.5-VL
(via Ollama) to produce a scene sentence, and updates the belief
distribution. No clarification questions are asked at any point.

IG_obs per window is the only signal — this is the baseline against which
the wh and polar conditions are compared.

Run:
    python orchestrator_obs_only.py
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import argparse

from probability.belief_updater_v3 import BeliefUpdaterV3
from observation_pipeline.video_observer import load_qwen_vlm, describe_clip
from utils.session_logger import SessionLogger
from utils.clip_loader import available_recipes, discover_clips


# ─────────────────────────────────────────────────────────────────────────
# Observation trajectory
#
# WINDOWS and GROUND_TRUTH are no longer hardcoded — they're resolved at
# main() call time from data/clips/<recipe>/ via discover_clips.
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def main(recipe: str = "carbonara"):
    # 0. Resolve clips + ground-truth label for this dish.
    try:
        windows_list, ground_truth = discover_clips(recipe)
    except (FileNotFoundError, ValueError) as e:
        print(f"[clips] {e}", file=sys.stderr)
        return
    WINDOWS = windows_list
    GROUND_TRUTH = ground_truth
    print(f"[session] recipe={recipe!r}  ground_truth={GROUND_TRUTH!r}  "
          f"{len(WINDOWS)} clips")

    missing = [p for p in WINDOWS if not p.exists()]
    if missing:
        print("Missing clip files:")
        for p in missing:
            print(f"  - {p}")
        print("\nRun: python scripts/prepare_clips.py")
        return

    bu = BeliefUpdaterV3()

    print()
    model, processor, device = load_qwen_vlm()
    logger = SessionLogger(condition="obs_only", ground_truth=GROUND_TRUTH)

    history_rows: list[dict] = []

    def snapshot(step_num: int, event_type: str, text: str,
                 entropy_before: float) -> None:
        ent = bu.entropy()
        top, top_p = bu.top_recipe()
        row = {
            "step":       step_num,
            "type":       event_type,
            "text":       text,
            "entropy":    round(ent, 4),
            "ig":         round(entropy_before - ent, 4),
            "top_recipe": top,
            "top_prob":   round(top_p, 4),
        }
        for name, prob in bu.belief.items():
            row[f"p::{name}"] = round(prob, 6)
        history_rows.append(row)

    # Step 0 — uniform prior
    snapshot(step_num=0, event_type="initial", text="",
             entropy_before=bu.entropy())

    print(f"\nMax entropy: {bu.max_entropy():.3f} bits")

    for i, clip_path in enumerate(WINDOWS, start=1):
        print(f"\n── Window {i}: {clip_path.name} ──────────────────────")

        sentence = describe_clip(clip_path, model, processor, device)
        print(f"  observation : {sentence}")
        if not sentence:
            print("  [warn] VLM returned empty observation; skipping update.")
            continue

        entropy_before = bu.entropy()
        bu.update(sentence)
        ig_obs = entropy_before - bu.entropy()

        snapshot(step_num=i, event_type="observation",
                 text=sentence, entropy_before=entropy_before)
        
        logger.log_observation(
            window=i, clip=clip_path.name, sentence=sentence,
            entropy_before=entropy_before, entropy_after=bu.entropy(),
        )

        top, p = bu.top_recipe()
        print(f"  entropy : {bu.entropy():.4f} bits  (IG_obs {ig_obs:+.4f})")
        print(f"  top     : {top} ({p:.3f})")

    # ── Final report ──────────────────────────────────────────────────────
    top, p = bu.top_recipe()
    is_correct = (top == GROUND_TRUTH)
    accuracy   = 1 if is_correct else 0

    print("\n" + "═" * 60)
    print(f"Ground truth     : {GROUND_TRUTH}")
    print(f"Final prediction : {top} (p={p:.4f})")
    print(f"Accuracy (Acc)   : {accuracy}  ({'CORRECT' if is_correct else 'WRONG'})")
    print(f"Final entropy    : {bu.entropy():.4f} bits")

    total_ig_obs = sum(
        row["ig"] for row in history_rows if row["type"] == "observation"
    )
    print(f"Total IG_obs     : {total_ig_obs:.4f} bits")

    # ── Write CSVs ────────────────────────────────────────────────────────
    # Per-session belief_history.csv goes into the SessionLogger's
    # session_dir; cross-session session_summary.csv stays at the top-level
    # outputs/ so it accumulates across runs.
    outputs_root = Path("outputs")
    outputs_root.mkdir(exist_ok=True)
    outputs_dir = logger.session_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    import csv as _csv
    if history_rows:
        # Same canonical filename across all three conditions — the session
        # folder name (with the condition suffix) is what disambiguates.
        csv_path = outputs_dir / "belief_history.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"\nWrote belief history → {csv_path}")

    import datetime as _dt
    summary_csv = outputs_root / "session_summary.csv"
    summary_row = {
        "timestamp":      _dt.datetime.now().isoformat(timespec="seconds"),
        "condition":      "obs_only",
        "ground_truth":   GROUND_TRUTH,
        "predicted":      top,
        "accuracy":       accuracy,
        "predicted_prob": round(p, 4),
        "final_entropy":  round(bu.entropy(), 4),
        "questions_asked": 0,
        "n_recipes":      bu.N,
        "n_windows":      len(WINDOWS),
        "total_ig_q":     0.0,
        "total_unique_value": 0.0,
    }
    summary_fields = list(summary_row.keys())
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)
    print(f"Appended session summary → {summary_csv}")

    print(f"\nPlot with: python scripts/plot_belief.py --session-dir {outputs_dir}")
    logger.save(
        predicted=top, accuracy=accuracy, final_prob=p,
        final_entropy=bu.entropy(), questions_asked=0,
        belief_history=history_rows, n_recipes=bu.N,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the observations-only (no questioning) orchestrator."
    )
    recipes = available_recipes() or ["carbonara"]
    parser.add_argument(
        "--recipe",
        choices=recipes,
        default="carbonara" if "carbonara" in recipes else recipes[0],
        help=(
            "Dish to run. Clips live in data/clips/<recipe>/. "
            f"Detected recipes: {', '.join(recipes)}."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(recipe=args.recipe)