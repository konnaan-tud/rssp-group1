"""
recipe_vectoriser.py
--------------------
Converts recipe JSON graphs into embedding vectors and saves them to disk.

Run once at setup time. Results are reused every session.

Outputs:
    recipe_vectors.npy  - numpy array of shape (N, 768)
    recipe_index.json   - mapping from index to recipe name

Usage:
    python recipe_vectoriser.py --recipe_dir recipe_graphs/instruct
"""

from __future__ import annotations

import argparse
import json
import os
import glob
import numpy as np
from embedder import Embedder


VECTORS_PATH = "recipe_vectors.npy"
INDEX_PATH = "recipe_index.json"


def recipe_to_text(recipe: dict) -> str:
    """
    Convert a recipe dict to structured natural text for embedding.

    Format:
        "<name>. ingredients: <ing1>, <ing2>, ... 
         <stage_name>: <action> <ingredient>, <action> <ingredient>. 
         <object> is <state>. ..."
    """
    parts = []

    # Recipe name
    name = recipe.get("name", "").replace("_", " ")
    parts.append(name)

    # All ingredients
    ingredients = recipe.get("all_ingredients", [])
    if ingredients:
        parts.append(f"ingredients: {', '.join(ingredients)}")

    # Stages — connect actions to ingredients, include object states
    for stage in recipe.get("stages", []):
        stage_name = stage.get("name", "").replace("_", " ")
        stage_parts = []

        for step in stage.get("steps", []):
            actions = step.get("actions", [])
            step_ingredients = step.get("ingredients", [])
            object_states = step.get("object_states", {})

            # Connect actions to ingredients where possible
            if actions and step_ingredients:
                action_text = ", ".join(
                    f"{a} {step_ingredients[0]}" if i == 0 else a
                    for i, a in enumerate(actions)
                )
                stage_parts.append(action_text)
            elif actions:
                stage_parts.append(", ".join(actions))

            # Add object states
            for obj, state in object_states.items():
                obj = obj.replace("_", " ")
                state = state.replace("_", " ")
                stage_parts.append(f"{obj} is {state}")

        if stage_parts:
            parts.append(f"{stage_name}: {'. '.join(stage_parts)}")

    return ". ".join(parts) + "."


def build_recipe_vectors(
    recipe_dir: str,
    vectors_path: str = VECTORS_PATH,
    index_path: str = INDEX_PATH,
) -> tuple[np.ndarray, dict]:
    """
    Load all recipe JSONs, convert to text, embed, save to disk.

    Returns
    -------
    vectors : np.ndarray of shape (N, 768)
    index   : dict mapping recipe name → row index
    """
    embedder = Embedder()

    # Load all recipe JSON files
    json_files = sorted(glob.glob(os.path.join(recipe_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {recipe_dir}")

    print(f"\nFound {len(json_files)} recipe files in {recipe_dir}")

    recipes = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            recipes.append(json.load(f))

    # Convert each recipe to text
    recipe_texts = []
    recipe_names = []

    print("\n=== Recipe texts ===")
    for recipe in recipes:
        text = recipe_to_text(recipe)
        name = recipe.get("name", "").replace("_", " ")
        recipe_texts.append(text)
        recipe_names.append(name)
        print(f"\n[{name}]\n  {text[:120]}...")

    # Embed all recipes
    print(f"\n\nEmbedding {len(recipe_texts)} recipes...")
    vectors_list = embedder.embed_batch(recipe_texts)
    vectors = np.stack(vectors_list)  # shape (N, 768)

    # Build index: recipe name → row index
    index = {name: i for i, name in enumerate(recipe_names)}

    # Save to disk
    np.save(vectors_path, vectors)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nSaved {vectors.shape} vectors → {vectors_path}")
    print(f"Saved index ({len(index)} recipes) → {index_path}")
    print(f"Embedder cache size: {embedder.cache_size}")

    return vectors, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe_dir",
        default="recipe_graphs/instruct",
        help="Directory containing recipe JSON files",
    )
    args = parser.parse_args()

    vectors, index = build_recipe_vectors(args.recipe_dir)

    print("\n=== Index ===")
    for name, idx in list(index.items())[:5]:
        print(f"  [{idx}] {name}")
    print(f"  ... ({len(index)} total)")