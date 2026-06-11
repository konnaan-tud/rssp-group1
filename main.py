from __future__ import annotations

import argparse
import json
from pathlib import Path

from observation_pipeline.dynamic_scene import ClipObservation, DynamicSceneBuilder
from probability.belief_updater_v3 import BeliefUpdaterV3
from recipe_pipeline.generate_recipe_scenes import generate_recipe_scene_graphs
from recipe_pipeline.recipe_sequencer import build_recipe_sequences


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "data" / "dataset.csv"
GRAPH_DIR = ROOT / "recipe_graphs" / "v3"
SEQUENCE_PATH = GRAPH_DIR / "recipe_sequences.json"


def bootstrap() -> None:
    """Generate recipe graphs and recipe sequences from the dataset."""
    generate_recipe_scene_graphs(DATASET_PATH, GRAPH_DIR)
    build_recipe_sequences(GRAPH_DIR, SEQUENCE_PATH)


def demo_belief_update() -> dict[str, float]:
    """Run a tiny end-to-end demo using the new package layout."""
    bootstrap()
    builder = DynamicSceneBuilder()
    updater = BeliefUpdaterV3.from_recipe_directory(GRAPH_DIR, SEQUENCE_PATH)

    observation = builder.build(
        ClipObservation(
            clip_id="demo-clip-001",
            ingredients=["cucumber", "onion", "sour cream"],
            actions=["slice", "mix", "serve"],
            notes="Cook assembles a cold salad and folds in a creamy dressing.",
        )
    )
    updater.update(observation.sentences)
    return updater.belief


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RSSP Group 1 entry point.")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "demo"],
        default="demo",
        help="bootstrap generates graph artifacts; demo also runs a belief update.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "bootstrap":
        bootstrap()
        print(f"Generated recipe artifacts under {GRAPH_DIR}")
        return

    belief = demo_belief_update()
    print(json.dumps(belief, indent=2))


if __name__ == "__main__":
    main()
