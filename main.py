"""
main.py — single dispatcher for the V3 pipeline.

Usage:
    python main.py --mode bootstrap   # embed recipes into data/recipe_sequences.json
    python main.py --mode test        # run the carbonara test session (no VLM)
    python main.py --mode vlm         # run the full VLM orchestrator
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


def run_vlm() -> None:
    """Run the VLM orchestrator end-to-end."""
    from orchestrator_v2_vlm import main as _main
    _main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSSP Group 1 entry point.")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "test", "vlm"],
        default="test",
        help=(
            "bootstrap: re-embed recipe scenes into recipe_sequences.json. "
            "test: run the carbonara belief-update test (no VLM). "
            "vlm: run the full VLM-driven orchestrator."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "bootstrap":
        run_bootstrap()
    elif args.mode == "test":
        run_test()
    elif args.mode == "vlm":
        run_vlm()


if __name__ == "__main__":
    main()
