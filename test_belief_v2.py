"""
test_belief_v2.py
-----------------
Tests belief_updater_v2.py using 5 mock recipes.

Simulates a creamy cucumber salad session:
  Window 1: slice cucumber         → ambiguous (all recipes start here)
  Window 2: chop onion             → still ambiguous
  Answer:   "I am adding sour cream" → should spike creamy cucumber
  Window 3: whisk sour cream       → confirms creamy cucumber
  Window 4: toss, cover            → finishing steps

Compares IG from passive observation vs clarification answer.
"""

from __future__ import annotations

import json
import math
import numpy as np
import os

from embedder import Embedder
from belief_updater_v2 import BeliefUpdaterV2, f1_match

MOCK_TERMS_PATH = "test_recipe_terms.json"

# ── Mock recipes ───────────────────────────────────────────────────────────

MOCK_RECIPES = [
    {
        "name": "classic vinegar cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar",
                            "white sugar", "water", "dill"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"],
                 "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["boil", "combine"],
                 "ingredients": ["white vinegar", "white sugar", "water"],
                 "object_states": {"dressing": "boiling"}},
                {"actions": ["stir", "pour"],
                 "ingredients": ["dill"],
                 "object_states": {"dressing": "mixed with dill"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["cover", "refrigerate"],
                 "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "creamy cucumber salad",
        "all_ingredients": ["cucumber", "onion", "sour cream",
                            "white sugar", "white vinegar", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"],
                 "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["whisk", "mix"],
                 "ingredients": ["sour cream", "white sugar", "white vinegar", "salt"],
                 "object_states": {"dressing": "creamy and mixed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"],
                 "ingredients": [],
                 "object_states": {"salad": "chilling"}}
            ]}
        ]
    },
    {
        "name": "spicy cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar",
                            "white sugar", "jalapeno", "chilli flakes", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop", "dice"],
                 "ingredients": ["cucumber", "onion", "jalapeno"],
                 "object_states": {"cucumber": "sliced", "jalapeno": "diced"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["mix", "combine"],
                 "ingredients": ["white vinegar", "white sugar",
                                 "chilli flakes", "salt"],
                 "object_states": {"dressing": "spicy and mixed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"],
                 "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "celery seed cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar",
                            "white sugar", "water", "celery seed", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"],
                 "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["combine", "boil"],
                 "ingredients": ["white vinegar", "white sugar", "water", "salt"],
                 "object_states": {"dressing": "boiling"}},
                {"actions": ["add", "stir"],
                 "ingredients": ["celery seed"],
                 "object_states": {"dressing": "mixed with celery seed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["pour", "toss", "cover", "refrigerate"],
                 "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "cucumber tomato salad",
        "all_ingredients": ["cucumber", "onion", "tomato", "olive oil",
                            "red wine vinegar", "salt", "black pepper"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop", "dice"],
                 "ingredients": ["cucumber", "onion", "tomato"],
                 "object_states": {"cucumber": "sliced", "tomato": "diced"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["whisk", "combine"],
                 "ingredients": ["olive oil", "red wine vinegar",
                                 "salt", "black pepper"],
                 "object_states": {"dressing": "whisked"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"],
                 "ingredients": [],
                 "object_states": {"salad": "chilling"}}
            ]}
        ]
    },
]


# ── Build mock term vectors ────────────────────────────────────────────────

def get_recipe_terms(recipe: dict) -> list[str]:
    terms = []
    for ing in recipe.get("all_ingredients", []):
        terms.append(ing.replace("_", " ").strip())
    for stage in recipe.get("stages", []):
        for step in stage.get("steps", []):
            for action in step.get("actions", []):
                terms.append(action.replace("_", " ").strip())
    return list(set(terms))


def build_mock_terms():
    """Build and save mock recipe term vectors if not already present."""
    if os.path.exists(MOCK_TERMS_PATH):
        print("[test] Loading existing mock term vectors.")
        return

    print("[test] Building mock recipe term vectors...")
    embedder = Embedder()

    # Collect all unique terms
    all_terms: set[str] = set()
    for recipe in MOCK_RECIPES:
        all_terms.update(get_recipe_terms(recipe))

    # Embed all unique terms in one batch
    term_list = sorted(all_terms)
    vectors = embedder.embed_batch(term_list)
    term_to_vector = {
        term: vec.tolist()
        for term, vec in zip(term_list, vectors)
    }

    # Build per-recipe term dict
    recipe_terms = {}
    for recipe in MOCK_RECIPES:
        name = recipe["name"].replace("_", " ")
        terms = get_recipe_terms(recipe)
        recipe_terms[name] = {
            term: term_to_vector[term]
            for term in terms
            if term in term_to_vector
        }

    with open(MOCK_TERMS_PATH, "w") as f:
        json.dump(recipe_terms, f)

    print(f"[test] Saved mock term vectors → {MOCK_TERMS_PATH}")


# ── Session simulation ─────────────────────────────────────────────────────

def run_test():
    build_mock_terms()

    updater = BeliefUpdaterV2(
        recipe_terms_path=MOCK_TERMS_PATH,
        temperature=0.1,
    )

    print(f"\n=== Belief Updater V2 Test (Soft Matching) ===")
    print(f"Recipes: {updater.recipe_names}")
    print(f"Max entropy: {updater.max_entropy():.3f} bits\n")

    windows = [
        {
            "label": "Window 1: slice cucumber",
            "ingredients": ["cucumber"],
            "actions": ["slice"],
            "answer": None,
        },
        {
            "label": "Window 2: chop onion",
            "ingredients": ["cucumber", "onion"],
            "actions": ["slice", "chop"],
            "answer": None,
        },
        {
            "label": "Window 3: whisk sour cream",
            "ingredients": ["cucumber", "onion", "sour cream",
                           "white sugar", "white vinegar"],
            "actions": ["slice", "chop", "whisk"],
            "answer": "I am adding sour cream to make a creamy dressing",
        },
        {
            "label": "Window 4: toss and cover",
            "ingredients": ["cucumber", "onion", "sour cream",
                           "white sugar", "white vinegar"],
            "actions": ["slice", "chop", "whisk", "toss", "cover", "refrigerate"],
            "answer": None,
        },
    ]

    for i, window in enumerate(windows, start=1):
        entropy_before = updater.entropy()

        # Incorporate clarification answer before this window if present
        if window["answer"]:
            print(f"── Clarification Answer ──────────────────────────")
            print(f"  Answer: \"{window['answer']}\"")
            h_before_answer = updater.entropy()
            updater.incorporate_answer(window["answer"])
            ig_q = updater.ig_q()
            print(f"  H before : {h_before_answer:.4f} bits")
            print(f"  H after  : {updater.entropy():.4f} bits")
            print(f"  IG_Q     : {ig_q:.4f} bits  ← clarification gain")
            print()
            entropy_before = updater.entropy()

        # Update from observation
        belief = updater.update(
            window["ingredients"],
            window["actions"],
        )

        s = updater.summary()
        ig_obs = entropy_before - updater.entropy()

        print(f"── {window['label']} ──────────────────────────────────")
        print(f"  ingredients : {window['ingredients']}")
        print(f"  actions     : {window['actions']}")
        print(f"  obs terms   : {s['obs_terms']}")
        print(f"  entropy     : {s['entropy']:.4f} / {s['max_entropy']:.4f} bits")
        print(f"  IG_obs      : {ig_obs:.4f} bits")
        print(f"  top recipe  : {s['top_recipe']} ({s['top_prob']:.4f})")
        print()
        print("  F1 similarity scores:")
        for name, sim in sorted(s['similarities'].items(), key=lambda x: -x[1]):
            bar = "█" * int(sim * 40)
            print(f"    {name:40s} {sim:.4f}  {bar}")
        print()
        print("  Belief distribution:")
        for name, prob in sorted(belief.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"    {name:40s} {prob:.4f}  {bar}")
        print()

    print("═" * 60)
    print(f"Final prediction : {updater.top_recipe()[0]}")
    print(f"Correct answer   : creamy cucumber salad")
    print(f"Correct?         : {updater.top_recipe()[0] == 'creamy cucumber salad'}")
    print(f"\nIG_Q (from answer) : {updater.ig_q():.4f} bits")
    print(f"Final entropy      : {updater.entropy():.4f} bits")


if __name__ == "__main__":
    run_test()