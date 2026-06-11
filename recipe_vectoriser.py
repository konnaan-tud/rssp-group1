"""
recipe_vectoriser.py
--------------------
Converts recipe JSON graphs into per-term embedding vectors and saves to disk.

Instead of one vector per recipe, saves individual vectors for each
ingredient and action. This preserves discriminating signal for soft matching.

Run once at setup time. Results reused every session.

Output:
    recipe_terms.json — {recipe_name: {term: [vector]}} for all recipes

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


RECIPE_TERMS_PATH = "recipe_terms.json"


def get_recipe_terms(recipe: dict) -> list[str]:
    """
    Extract all meaningful terms from a recipe.
    Returns a flat list of ingredients and actions.
    """
    terms = []

    # All ingredients
    for ing in recipe.get("all_ingredients", []):
        ing = ing.replace("_", " ").strip()
        if ing:
            terms.append(ing)

    # All actions across all stages and steps
    for stage in recipe.get("stages", []):
        for step in stage.get("steps", []):
            for action in step.get("actions", []):
                action = action.replace("_", " ").strip()
                if action:
                    terms.append(action)

    return terms


def build_recipe_terms(
    recipe_dir: str,
    output_path: str = RECIPE_TERMS_PATH,
) -> dict:
    """
    Load all recipe JSONs, extract terms, embed each term,
    save to disk as recipe_terms.json.

    Returns
    -------
    recipe_terms : dict
        {recipe_name: {term: vector_list}}
    """
    embedder = Embedder()

    # Load all recipe JSON files
    json_files = sorted(glob.glob(os.path.join(recipe_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {recipe_dir}")

    print(f"\nFound {len(json_files)} recipe files in {recipe_dir}")

    # Collect all unique terms across all recipes for batch embedding
    all_terms: set[str] = set()
    recipes = []

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            recipe = json.load(f)
        recipes.append(recipe)
        terms = get_recipe_terms(recipe)
        all_terms.update(terms)

    print(f"Found {len(all_terms)} unique terms across all recipes.")
    print("Embedding all terms (cached terms will be fast)...")

    # Embed all unique terms in one batch
    term_list = sorted(all_terms)
    vectors = embedder.embed_batch(term_list)
    term_to_vector = {
        term: vector.tolist()
        for term, vector in zip(term_list, vectors)
    }

    # Build per-recipe term dict
    recipe_terms: dict = {}
    for recipe in recipes:
        name = recipe.get("name", "").replace("_", " ").strip()
        terms = get_recipe_terms(recipe)

        # Deduplicate terms while preserving all
        recipe_terms[name] = {
            term: term_to_vector[term]
            for term in set(terms)
            if term in term_to_vector
        }

        print(f"  [{name}]: {len(recipe_terms[name])} terms")

    # Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipe_terms, f)

    print(f"\nSaved recipe terms → {output_path}")
    print(f"Embedder cache size: {embedder.cache_size}")

    return recipe_terms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe_dir",
        default="recipe_graphs/instruct",
        help="Directory containing recipe JSON files",
    )
    args = parser.parse_args()

    recipe_terms = build_recipe_terms(args.recipe_dir)

    print("\n=== Sample terms per recipe ===")
    for name, terms in list(recipe_terms.items())[:3]:
        print(f"  [{name}]: {list(terms.keys())}")