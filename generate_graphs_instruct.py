import os
import json
import csv
import requests
import time

# ── Settings ───────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "qwen2.5-7b-instruct"
OUTPUT_DIR = "recipe_graphs/instruct"
CSV_PATH = "dataset.csv"

# ── Prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a recipe analysis assistant. Your job is to convert a recipe into a "
    "structured JSON graph. You must return only a valid JSON object — no extra text, "
    "no explanations, no markdown code fences."
)

def build_user_prompt(name, category, ingredients, directions):
    return (
        f"Convert the following recipe into a structured JSON graph.\n\n"
        f"Recipe Name: {name}\n"
        f"Category: {category}\n"
        f"Ingredients: {ingredients}\n"
        f"Directions: {directions}\n\n"
        f"Return only a JSON object following this exact template:\n"
        f"{{\n"
        f'  "name": "<recipe name lowercase>",\n'
        f'  "category": "<category lowercase>",\n'
        f'  "all_ingredients": ["<ingredient1>", "<ingredient2>", ...],\n'
        f'  "stages": [\n'
        f'    {{\n'
        f'      "stage_id": 1,\n'
        f'      "name": "<stage name e.g. Prep, Make Dressing, Combine and Finish>",\n'
        f'      "steps": [\n'
        f'        {{\n'
        f'          "step_id": "s1_1",\n'
        f'          "actions": ["<action1>", "<action2>"],\n'
        f'          "ingredients": ["<ingredient1>", "<ingredient2>"],\n'
        f'          "object_states": {{"<ingredient>": "<state after this step>"}}\n'
        f'        }}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ]\n'
        f'}}\n\n"'
        f"Rules:\n"
        f"- Group steps into logical stages. Within a stage, steps can happen in any order. "
        f"Stages must happen in sequence.\n"
        f"- Actions must be short verb phrases of 1-5 words (e.g. 'slice cucumbers', 'mix dressing', 'cover and refrigerate'). "
        f"Never copy full sentences from the directions.\n"
        f"- Always produce at least 3 stages per recipe.\n"
        f"- Use lowercase for all values including stage names and ingredient names. No underscores in names.\n"
        f"- step_id format: s<stage_id>_<step_number> e.g. s1_1, s1_2, s2_1\n"
        f"- object_states should reflect what the ingredient looks like after this step\n"
        f"- Return only the JSON object, nothing else"
    )

# ── API call ───────────────────────────────────────────────────────────────
def call_llm(name, category, ingredients, directions):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(name, category, ingredients, directions)}
    ]
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2048,
    }
    response = requests.post(LM_STUDIO_URL, json=payload)
    if not response.ok:
        print(f"  Error {response.status_code}: {response.text}")
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# ── JSON parser ────────────────────────────────────────────────────────────
def parse_json(raw_text, recipe_name):
    clean = raw_text.strip()
    # Strip markdown fences if model ignores instructions
    if clean.startswith("```"):
        lines = clean.split("\n")
        clean = "\n".join(lines[1:-1])
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"  Warning: could not parse JSON for '{recipe_name}': {e}")
        return {"raw_output": raw_text}

# ── Main ───────────────────────────────────────────────────────────────────
def generate_graphs(target_recipes=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        recipes = list(reader)

    # Filter to target recipes if specified
    if target_recipes:
        recipes = [r for r in recipes if r["recipe_title"] in target_recipes]

    print(f"Generating graphs for {len(recipes)} recipes using {MODEL_ID}\n")

    for i, row in enumerate(recipes):
        name = row["recipe_title"]
        category = row["category"]
        ingredients = row["clean_ingredients"]
        directions = row["direction_text"]

        print(f"[{i+1}/{len(recipes)}] {name}")

        start = time.time()
        try:
            raw = call_llm(name, category, ingredients, directions)
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")

            parsed = parse_json(raw, name)

            # Save as JSON file
            filename = name.lower().replace(" ", "_").replace("'", "").replace("(", "").replace(")", "") + ".json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as out:
                json.dump(parsed, out, indent=2)
            print(f"  Saved → {filepath}")

        except Exception as e:
            print(f"  Failed: {e}")

        print()

if __name__ == "__main__":
    # Start with just the three target recipes
    target = [
        "Cucumber Salad",
        "Mom's Marinated Cucumbers",
        "Mizeria (Polish Cucumber Salad)"
    ]
    generate_graphs(target_recipes=target)