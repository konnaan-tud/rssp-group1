"""
recipe_pipeline/recipe_sequencer.py
-------------------------------------
Converts italian_recipe_scenes.json into embedded vector sequences.

For each recipe:
  - Takes the ordered list of scene sentences
  - Embeds each sentence using all-mpnet-base-v2
  - Saves as an ordered list of vectors

Output: data/recipe_sequences.json
  {
    "carbonara": [[v1], [v2], ...],
    "pesto pasta": [[v1], [v2], ...],
    ...
  }

Each vector is a list of 768 floats. Order is preserved —
first scene sentence → first vector, etc.

Run:
    python -m recipe_pipeline.recipe_sequencer
    or
    python recipe_pipeline/recipe_sequencer.py
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observation_pipeline.embedder import Embedder

SCENES_PATH = os.path.join("data", "italian_recipe_scenes.json")
OUTPUT_PATH = os.path.join("data", "recipe_sequences.json")


def build_recipe_sequences(
    scenes_path: str = SCENES_PATH,
    output_path: str = OUTPUT_PATH,
) -> dict[str, list[list[float]]]:
    """
    Load recipe scenes, embed each sentence, save sequences.

    Parameters
    ----------
    scenes_path : path to italian_recipe_scenes.json
    output_path : path to save recipe_sequences.json

    Returns
    -------
    dict mapping dish name → ordered list of 768-dim vectors (as lists)
    """
    # Load recipe scenes
    with open(scenes_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    print(f"Loaded {len(recipes)} recipes from {scenes_path}")

    embedder = Embedder()

    # Collect all unique sentences across all recipes for batch embedding
    all_sentences: list[str] = []
    for recipe in recipes:
        all_sentences.extend(recipe["scenes"])

    unique_sentences = list(set(all_sentences))
    print(f"\nFound {len(unique_sentences)} unique scene sentences across all recipes.")
    print("Embedding all sentences (cached ones will be fast)...")

    # Embed all unique sentences in one batch
    vectors = embedder.embed_batch(unique_sentences)
    sentence_to_vector = {
        s: v.tolist()
        for s, v in zip(unique_sentences, vectors)
    }

    # Build per-recipe ordered sequence
    recipe_sequences: dict[str, list[list[float]]] = {}

    print("\n=== Recipe sequences ===")
    for recipe in recipes:
        dish = recipe["dish"]
        scenes = recipe["scenes"]
        sequence = [sentence_to_vector[s] for s in scenes]
        recipe_sequences[dish] = sequence
        print(f"  [{dish}]: {len(scenes)} scenes → {len(sequence)} vectors")

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipe_sequences, f)

    print(f"\nSaved recipe sequences → {output_path}")
    print(f"Embedder cache size: {embedder.cache_size}")

    return recipe_sequences


if __name__ == "__main__":
    sequences = build_recipe_sequences()

    print("\n=== Verification ===")
    for dish, vecs in list(sequences.items())[:3]:
        print(f"  {dish}: {len(vecs)} vectors, "
              f"each shape ({len(vecs[0])},)")