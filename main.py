"""
main.py — single dispatcher for the V3 pipeline.

Usage:
    python main.py --mode bootstrap   # embed recipes into data/recipe_sequences.json
    python main.py --mode cluster     # cluster recipes + write similarity outputs
    python main.py --mode test        # run the carbonara test session (no VLM)
    python main.py --mode vlm         # run the full VLM orchestrator (stub answers)
    python main.py --mode vlm --answer-mode text   # typed human answers
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_bootstrap() -> None:
    """
    Re-embed every scene in data/italian_recipe_scenes.json and write the
    per-recipe vector lists to data/recipe_sequences.json. Safe to run
    multiple times — the embedder cache avoids redundant work.
    """
    from recipe_pipeline.recipe_sequencer import build_recipe_sequences

    scenes_path = ROOT / "data" / "italian_recipe_scenes.json"
    out_path = ROOT / "data" / "recipe_sequences.json"
    print(f"Embedding scenes from {scenes_path} → {out_path}")
    build_recipe_sequences(str(scenes_path), str(out_path))
    print("Bootstrap complete.")


def run_test() -> None:
    """Run the standalone carbonara belief-update test (no VLM, no orchestrator)."""
    from probability.test_belief_v3 import run_test as _run_test
    _run_test()


def run_cluster(cluster_count: int | None) -> None:
    """
    Compute recipe similarity artefacts and clusters.

    If `cluster_count` is None, the script picks k by silhouette score
    over k ∈ [2, 8]. Pass an int to force a specific cluster count.
    """
    from scripts.cluster_recipes import run_recipe_clustering

    result = run_recipe_clustering(n_clusters=cluster_count)
    print(
        f"Clustered {result['recipe_count']} recipes into "
        f"{result['cluster_count']} groups."
    )
    s = result["silhouette"]
    print(
        f"Silhouette: best k = {s['best_k']} (score {s['best_score']:.3f}), "
        f"chosen k = {s['chosen_k']}"
    )
    print(f"Salads baseline included: {result['salads_baseline']}")
    print(f"Similarity matrix → {result['matrix_csv']}")
    print(f"Cluster report    → {result['clusters_md']}")
    print(f"Heatmap           → {result['heatmap_png']}")
    print(f"Similarity curve  → {result['curve_png']}")


def run_vlm(answer_mode: str, recipe: str) -> None:
    """Run the VLM orchestrator end-to-end."""
    from orchestrator_v2_vlm import main as _main
    _main(answer_mode=answer_mode, recipe=recipe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSSP Group 1 entry point.")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "cluster", "test", "vlm"],
        default="test",
        help=(
            "bootstrap: re-embed recipe scenes into recipe_sequences.json. "
            "cluster: compute recipe similarity outputs and clusters. "
            "test: run the carbonara belief-update test (no VLM). "
            "vlm: run the full VLM-driven orchestrator."
        ),
    )
    parser.add_argument(
        "--answer-mode",
        choices=["stub", "text"],
        default="stub",
        help=(
            "Human-answer source for --mode vlm. "
            "'stub' uses the built-in carbonara simulator; "
            "'text' prompts for a typed answer in the terminal."
        ),
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=None,
        help=(
            "Target number of recipe clusters for --mode cluster. "
            "Default: pick automatically by silhouette score."
        ),
    )
    parser.add_argument(
        "--recipe",
        default="carbonara",
        help=(
            "Dish to run for --mode vlm. Clips live in data/clips/<recipe>/. "
            "Default: carbonara."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "bootstrap":
        run_bootstrap()
    elif args.mode == "cluster":
        run_cluster(args.cluster_count)
    elif args.mode == "test":
        run_test()
    elif args.mode == "vlm":
        run_vlm(args.answer_mode, args.recipe)


if __name__ == "__main__":
    main()
