"""
test_recipes.py
---------------
5 hand-crafted cucumber salad recipe graphs for testing the belief updater.
All recipes share Stage 1 (prep) and Stage 3 (finish), diverging only in
Stage 2 where the discriminating ingredient appears.

Stored as plain dicts matching the same JSON structure as recipe_graphs/instruct/.
"""

RECIPES = [
    {
        "name": "classic vinegar cucumber salad",
        "category": "salad",
        "all_ingredients": [
            "cucumber", "onion", "white vinegar", "white sugar", "water", "dill"
        ],
        "stages": [
            {
                "stage_id": 1,
                "name": "prep",
                "steps": [
                    {
                        "step_id": "s1_1",
                        "actions": ["slice", "chop"],
                        "ingredients": ["cucumber", "onion"],
                        "object_states": {"cucumber": "sliced", "onion": "chopped"}
                    }
                ]
            },
            {
                "stage_id": 2,
                "name": "make dressing",
                "steps": [
                    {
                        "step_id": "s2_1",
                        "actions": ["boil", "combine"],
                        "ingredients": ["white vinegar", "white sugar", "water"],
                        "object_states": {"dressing": "boiling"}
                    },
                    {
                        "step_id": "s2_2",
                        "actions": ["stir", "pour"],
                        "ingredients": ["dill"],
                        "object_states": {"dressing": "mixed with dill"}
                    }
                ]
            },
            {
                "stage_id": 3,
                "name": "finish",
                "steps": [
                    {
                        "step_id": "s3_1",
                        "actions": ["cover", "refrigerate"],
                        "ingredients": [],
                        "object_states": {"salad": "marinating"}
                    }
                ]
            }
        ]
    },

    {
        "name": "creamy cucumber salad",
        "category": "salad",
        "all_ingredients": [
            "cucumber", "onion", "sour cream", "white sugar", "white vinegar", "salt"
        ],
        "stages": [
            {
                "stage_id": 1,
                "name": "prep",
                "steps": [
                    {
                        "step_id": "s1_1",
                        "actions": ["slice", "chop"],
                        "ingredients": ["cucumber", "onion"],
                        "object_states": {"cucumber": "sliced", "onion": "chopped"}
                    }
                ]
            },
            {
                "stage_id": 2,
                "name": "make dressing",
                "steps": [
                    {
                        "step_id": "s2_1",
                        "actions": ["whisk", "mix"],
                        "ingredients": ["sour cream", "white sugar", "white vinegar", "salt"],
                        "object_states": {"dressing": "creamy and mixed"}
                    }
                ]
            },
            {
                "stage_id": 3,
                "name": "finish",
                "steps": [
                    {
                        "step_id": "s3_1",
                        "actions": ["toss", "cover", "refrigerate"],
                        "ingredients": [],
                        "object_states": {"salad": "chilling"}
                    }
                ]
            }
        ]
    },

    {
        "name": "spicy cucumber salad",
        "category": "salad",
        "all_ingredients": [
            "cucumber", "onion", "white vinegar", "white sugar", "jalapeno", "chilli flakes", "salt"
        ],
        "stages": [
            {
                "stage_id": 1,
                "name": "prep",
                "steps": [
                    {
                        "step_id": "s1_1",
                        "actions": ["slice", "chop", "dice"],
                        "ingredients": ["cucumber", "onion", "jalapeno"],
                        "object_states": {
                            "cucumber": "sliced",
                            "onion": "chopped",
                            "jalapeno": "diced"
                        }
                    }
                ]
            },
            {
                "stage_id": 2,
                "name": "make dressing",
                "steps": [
                    {
                        "step_id": "s2_1",
                        "actions": ["mix", "combine"],
                        "ingredients": ["white vinegar", "white sugar", "chilli flakes", "salt"],
                        "object_states": {"dressing": "spicy and mixed"}
                    }
                ]
            },
            {
                "stage_id": 3,
                "name": "finish",
                "steps": [
                    {
                        "step_id": "s3_1",
                        "actions": ["toss", "cover", "refrigerate"],
                        "ingredients": [],
                        "object_states": {"salad": "marinating"}
                    }
                ]
            }
        ]
    },

    {
        "name": "celery seed cucumber salad",
        "category": "salad",
        "all_ingredients": [
            "cucumber", "onion", "white vinegar", "white sugar", "water", "celery seed", "salt"
        ],
        "stages": [
            {
                "stage_id": 1,
                "name": "prep",
                "steps": [
                    {
                        "step_id": "s1_1",
                        "actions": ["slice", "chop"],
                        "ingredients": ["cucumber", "onion"],
                        "object_states": {"cucumber": "sliced", "onion": "chopped"}
                    }
                ]
            },
            {
                "stage_id": 2,
                "name": "make dressing",
                "steps": [
                    {
                        "step_id": "s2_1",
                        "actions": ["combine", "boil"],
                        "ingredients": ["white vinegar", "white sugar", "water", "salt"],
                        "object_states": {"dressing": "boiling"}
                    },
                    {
                        "step_id": "s2_2",
                        "actions": ["add", "stir"],
                        "ingredients": ["celery seed"],
                        "object_states": {"dressing": "mixed with celery seed"}
                    }
                ]
            },
            {
                "stage_id": 3,
                "name": "finish",
                "steps": [
                    {
                        "step_id": "s3_1",
                        "actions": ["pour", "toss", "cover", "refrigerate"],
                        "ingredients": [],
                        "object_states": {"salad": "marinating"}
                    }
                ]
            }
        ]
    },

    {
        "name": "cucumber tomato salad",
        "category": "salad",
        "all_ingredients": [
            "cucumber", "onion", "tomato", "olive oil", "red wine vinegar", "salt", "black pepper"
        ],
        "stages": [
            {
                "stage_id": 1,
                "name": "prep",
                "steps": [
                    {
                        "step_id": "s1_1",
                        "actions": ["slice", "chop", "dice"],
                        "ingredients": ["cucumber", "onion", "tomato"],
                        "object_states": {
                            "cucumber": "sliced",
                            "onion": "chopped",
                            "tomato": "diced"
                        }
                    }
                ]
            },
            {
                "stage_id": 2,
                "name": "make dressing",
                "steps": [
                    {
                        "step_id": "s2_1",
                        "actions": ["whisk", "combine"],
                        "ingredients": ["olive oil", "red wine vinegar", "salt", "black pepper"],
                        "object_states": {"dressing": "whisked"}
                    }
                ]
            },
            {
                "stage_id": 3,
                "name": "finish",
                "steps": [
                    {
                        "step_id": "s3_1",
                        "actions": ["toss", "cover", "refrigerate"],
                        "ingredients": [],
                        "object_states": {"salad": "chilling"}
                    }
                ]
            }
        ]
    }
]

# Quick check
if __name__ == "__main__":
    for r in RECIPES:
        print(f"{r['name']}: {len(r['stages'])} stages | ingredients: {r['all_ingredients']}")
