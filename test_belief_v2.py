"""
test_belief_v2.py
-----------------
Tests belief_updater_v2.py using the 5 mock recipes.

Simulates a creamy cucumber salad session:
  Window 1: slice cucumber         → ambiguous
  Window 2: chop onion             → still ambiguous
  Answer:   "I am adding sour cream" → should spike creamy cucumber
  Window 3: whisk sour cream       → confirms creamy cucumber
  Window 4: toss, cover            → finishing

Compares IG from passive observation vs clarification answer.
"""

import json
import math
import numpy as np
import os
import sys

# ── Build mock recipe vectors from test_recipes ────────────────────────────
# We build vectors directly here so the test is self-contained
# and doesn't depend on recipe_vectoriser.py having been run first.

from embedder import Embedder
from belief_updater_v2 import BeliefUpdaterV2, build_observation_text, combined_similarity

MOCK_VECTORS_PATH = "test_recipe_vectors.npy"
MOCK_INDEX_PATH = "test_recipe_index.json"

MOCK_RECIPES = [
    {
        "name": "classic vinegar cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar", "white sugar", "water", "dill"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"], "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["boil", "combine"], "ingredients": ["white vinegar", "white sugar", "water"],
                 "object_states": {"dressing": "boiling"}},
                {"actions": ["stir", "pour"], "ingredients": ["dill"],
                 "object_states": {"dressing": "mixed with dill"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["cover", "refrigerate"], "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "creamy cucumber salad",
        "all_ingredients": ["cucumber", "onion", "sour cream", "white sugar", "white vinegar", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"], "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["whisk", "mix"], "ingredients": ["sour cream", "white sugar", "white vinegar", "salt"],
                 "object_states": {"dressing": "creamy and mixed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"], "ingredients": [],
                 "object_states": {"salad": "chilling"}}
            ]}
        ]
    },
    {
        "name": "spicy cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar", "white sugar", "jalapeno", "chilli flakes", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop", "dice"], "ingredients": ["cucumber", "onion", "jalapeno"],
                 "object_states": {"cucumber": "sliced", "jalapeno": "diced"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["mix", "combine"], "ingredients": ["white vinegar", "white sugar", "chilli flakes", "salt"],
                 "object_states": {"dressing": "spicy and mixed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"], "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "celery seed cucumber salad",
        "all_ingredients": ["cucumber", "onion", "white vinegar", "white sugar", "water", "celery seed", "salt"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop"], "ingredients": ["cucumber", "onion"],
                 "object_states": {"cucumber": "sliced", "onion": "chopped"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["combine", "boil"], "ingredients": ["white vinegar", "white sugar", "water", "salt"],
                 "object_states": {"dressing": "boiling"}},
                {"actions": ["add", "stir"], "ingredients": ["celery seed"],
                 "object_states": {"dressing": "mixed with celery seed"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["pour", "toss", "cover", "refrigerate"], "ingredients": [],
                 "object_states": {"salad": "marinating"}}
            ]}
        ]
    },
    {
        "name": "cucumber tomato salad",
        "all_ingredients": ["cucumber", "onion", "tomato", "olive oil", "red wine vinegar", "salt", "black pepper"],
        "stages": [
            {"stage_id": 1, "name": "prep", "steps": [
                {"actions": ["slice", "chop", "dice"], "ingredients": ["cucumber", "onion", "tomato"],
                 "object_states": {"cucumber": "sliced", "tomato": "diced"}}
            ]},
            {"stage_id": 2, "name": "make dressing", "steps": [
                {"actions": ["whisk", "combine"], "ingredients": ["olive oil", "red wine vinegar", "salt", "black pepper"],
                 "object_states": {"dressing": "whisked"}}
            ]},
            {"stage_id": 3, "name": "finish", "steps": [
                {"actions": ["toss", "cover", "refrigerate"], "ingredients": [],
                 "object_states": {"salad": "chilling"}}
            ]}
        ]
    },
]


def recipe_to_text(recipe: dict) -> str:
    """Convert recipe dict to structured text for embedding."""
    from recipe_vectoriser import recipe_to_text as _r2t
    return _r2t(recipe)


def build_mock_vectors():
    """Build and save mock recipe vectors if not already present."""
    if os.path.exists(MOCK_VECTORS_PATH) and os.path.exists(MOCK_INDEX_PATH):
        print("[test] Loading existing mock vectors.")
        return

    print("[test] Building mock recipe vectors...")
    embedder = Embedder()
    texts = [recipe_to_text(r) for r in MOCK_RECIPES]
    vectors = np.stack(embedder.embed_batch(texts))
    index = {r["name"]: i for i, r in enumerate(MOCK_RECIPES)}

    np.save(MOCK_VECTORS_PATH, vectors)
    with open(MOCK_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[test] Saved {vectors.shape} mock vectors.")


# ── Session simulation ─────────────────────────────────────────────────────

def run_test():
    build_mock_vectors()

    updater = BeliefUpdaterV2(
        vectors_path=MOCK_VECTORS_PATH,
        index_path=MOCK_INDEX_PATH,
        alpha=0.7,
        temperature=0.5,
    )

    print(f"\n=== Belief Updater V2 Test ===")
    print(f"Recipes: {updater.recipe_names}")
    print(f"Max entropy: {updater.max_entropy():.3f} bits\n")

    # ── Simulated session: creamy cucumber salad ───────────────────────────
    windows = [
        {
            "label": "Window 1: slice cucumber",
            "ingredients": ["cucumber"],
            "actions": ["slice"],
            "states": {"cucumber": "sliced"},
            "answer": None,
        },
        {
            "label": "Window 2: chop onion",
            "ingredients": ["cucumber", "onion"],
            "actions": ["slice", "chop"],
            "states": {"cucumber": "sliced", "onion": "chopped"},
            "answer": None,
        },
        {
            "label": "Window 3: whisk sour cream (after answer)",
            "ingredients": ["cucumber", "onion", "sour cream", "white sugar", "white vinegar"],
            "actions": ["slice", "chop", "whisk"],
            "states": {"cucumber": "sliced", "onion": "chopped", "dressing": "creamy"},
            "answer": "I am adding sour cream to make a creamy dressing",
        },
        {
            "label": "Window 4: toss and cover",
            "ingredients": ["cucumber", "onion", "sour cream", "white sugar", "white vinegar"],
            "actions": ["slice", "chop", "whisk", "toss", "cover", "refrigerate"],
            "states": {"salad": "chilling"},
            "answer": None,
        },
    ]

    for i, window in enumerate(windows, start=1):
        entropy_before = updater.entropy()

        # If this window has a clarification answer, incorporate it first
        if window["answer"]:
            print(f"── Clarification Answer ──────────────────────────────────")
            print(f"  Answer: \"{window['answer']}\"")
            entropy_before_answer = updater.entropy()
            updater.incorporate_answer(window["answer"])
            ig_q = updater.ig_q()
            print(f"  H before answer : {entropy_before_answer:.4f} bits")
            print(f"  H after answer  : {updater.entropy():.4f} bits")
            print(f"  IG_Q            : {ig_q:.4f} bits")
            print()

        # Update from observation
        belief = updater.update(
            window["ingredients"],
            window["actions"],
            window["states"],
        )

        s = updater.summary()
        ig_obs = entropy_before - updater.entropy()

        print(f"── {window['label']} ──────────────────────────────────")
        print(f"  ingredients : {window['ingredients']}")
        print(f"  actions     : {window['actions']}")
        print(f"  entropy     : {s['entropy']:.4f} / {s['max_entropy']:.4f} bits")
        print(f"  IG_obs      : {ig_obs:.4f} bits")
        print(f"  top recipe  : {s['top_recipe']} ({s['top_prob']:.4f})")
        print()
        print("  Belief distribution:")
        for name, prob in sorted(belief.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 40)
            print(f"    {name:40s} {prob:.4f}  {bar}")
        print()

    # ── Final summary ──────────────────────────────────────────────────────
    print("═" * 60)
    print(f"Final prediction : {updater.top_recipe()[0]}")
    print(f"Correct answer   : creamy cucumber salad")
    print(f"Correct?         : {updater.top_recipe()[0] == 'creamy cucumber salad'}")
    print(f"\nIG_Q (from answer)    : {updater.ig_q():.4f} bits")
    print(f"Final entropy         : {updater.entropy():.4f} bits")


if __name__ == "__main__":
    run_test()