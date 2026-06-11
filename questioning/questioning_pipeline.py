from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_static_graphs(graph_folder: str | Path) -> list[dict]:
    """
    Load all recipe graph JSON files from a folder.
    """
    graph_folder = Path(graph_folder)

    if not graph_folder.exists():
        raise FileNotFoundError(f"Graph folder not found: {graph_folder}")

    graphs = []
    for json_file in sorted(graph_folder.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            graphs.append(json.load(f))

    if not graphs:
        raise FileNotFoundError(f"No JSON graph files found in: {graph_folder}")

    return graphs


def build_question_prompt(
    static_graphs: list[dict],
    belief_state: dict[str, float],
) -> str:
    """
    Build the prompt that asks the VLM to generate clarification questions.
    """

    static_graphs_json = json.dumps(static_graphs, indent=2, ensure_ascii=False)
    belief_state_json = json.dumps(belief_state, indent=2, ensure_ascii=False)

    prompt = f"""
You are a clarification-question planner for a VLM-based cooking observer.

The observer has watched a short cooking video clip and currently has uncertainty
over which recipe is being prepared. The candidate recipes are listed below.

YOUR TASK
Generate exactly 3 open wh-questions whose answers would reduce uncertainty
over the current recipe belief state as much as possible.

═══ HOW TO FILL "targets" — READ CAREFULLY ═══

The "targets" field is the most important part of each question. It is the
list of CONCRETE WORDS OR PHRASES that the human's answer might contain —
actual ingredient names or action names drawn from the recipe vocabulary.

A downstream planner uses these targets to test whether the question would
actually distinguish between recipes. Abstract placeholder names CANNOT be
matched against any recipe and make the question useless.

GOOD targets (concrete ingredients or actions from the recipes below):
    "targets": ["sour cream", "white vinegar", "mayonnaise"]
    "targets": ["whisk", "boil", "stir"]
    "targets": ["dill", "celery seed", "mint"]

BAD targets (NEVER produce these — they cannot be matched):
    "targets": ["main_ingredient"]
    "targets": ["stage_id"]
    "targets": ["next_action"]
    "targets": ["first_step"]
    "targets": ["ingredient or action being tested"]

Every string in "targets" must appear — in the same or very similar form —
in the "all_ingredients" or "actions" fields of one of the recipe graphs
shown below.

═══ FORMAT ═══

For each question return:
  - "question":          the natural-language clarification question
  - "question_form":     "wh"
  - "targets":           list of concrete ingredient/action strings
  - "distinguishes":     list of recipe names this question helps separate
  - "expected_information_gain_reason": one-sentence justification

Do not ask:
  - "What recipe are you making?"
  - "Which recipe is this?"

Each question must begin with one of: what, which, where, when, why, how.

Rank from best to worst by expected information gain.

═══ CURRENT BELIEF STATE ═══
{belief_state_json}

═══ STATIC RECIPE GRAPHS ═══
(These are the only candidate recipes. Draw your targets from the
ingredients and actions listed here.)

{static_graphs_json}

═══ RESPONSE ═══

Return ONLY valid JSON in the format below. The examples use concrete
vocabulary — replace them with concrete vocabulary drawn from the recipes
above. DO NOT keep the schema-style placeholders like "stage_id" or
"main_ingredient".

{{
  "questions": [
    {{
      "rank": 1,
      "question": "Which ingredient are you adding to the dressing?",
      "question_form": "wh",
      "targets": ["sour cream", "white vinegar", "mayonnaise"],
      "distinguishes": ["cucumber salad with sour cream", "mizeria"],
      "expected_information_gain_reason": "These dressings appear in different recipes; naming the ingredient identifies the recipe family."
    }},
    {{
      "rank": 2,
      "question": "What action are you about to perform next?",
      "question_form": "wh",
      "targets": ["whisk", "boil", "marinate"],
      "distinguishes": ["best-ever-cucumber-dill-salad", "moms marinated cucumbers"],
      "expected_information_gain_reason": "Different recipes use different dressing preparations; the next action narrows the candidates."
    }},
    {{
      "rank": 3,
      "question": "Which herb or spice will you add?",
      "question_form": "wh",
      "targets": ["dill", "mint", "celery seed"],
      "distinguishes": ["best-ever-cucumber-dill-salad", "tomato cucumber salad with mint"],
      "expected_information_gain_reason": "Each recipe uses a distinctive herb so naming it disambiguates."
    }}
  ]
}}
"""

    return prompt.strip()


def run_qwen_prompt(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 768,
) -> str:
    """
    Load Qwen2.5-VL and run a simple text prompt.

    Returns the generated response as a string.
    """

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading {model_name} on {device}...", flush=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    print("Model ready.", flush=True)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
    ).to(device)

    print("Generating...", flush=True)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


if __name__ == "__main__":
    graph_folder = "recipe_graphs/instruct"

    # replace with actual belief state json

    belief_state = {
        "cucumber salad": 0.34,
        "mizeria": 0.33,
        "mom_s_marinated_cucumbers": 0.33,
    }

    # Replace with actual path to recipe graph JSON files

    static_graphs = load_static_graphs(graph_folder)

    prompt = build_question_prompt(
        static_graphs=static_graphs,
        belief_state=belief_state,
    )

    response = run_qwen_prompt(prompt)

    print("\n----- Qwen response -----\n")
    print(response)