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
over which recipe is being prepared.

Your task:
Generate exactly 3 open wh-questions that would reduce uncertainty over the
current recipe belief state as much as possible.

The questions should be useful for distinguishing between the currently plausible
recipes. The questions should also not 

Do not ask directly:
- "What recipe are you making?"
- "Which recipe is this?"

Only ask wh-questions.
Each question must begin with one of:
what, which, where, when, who, why, or how.

Rank the questions from best to worst by expected information gain.

CURRENT BELIEF STATE:
{belief_state_json}

STATIC RECIPE GRAPHS:
{static_graphs_json}

Return only valid JSON in this exact format:

{{
  "questions": [
    {{
      "rank": 1,
      "question": "Which ingredient are you adding next?",
      "question_form": "wh",
      "targets": ["ingredient or action being tested"],
      "distinguishes": ["recipe_a", "recipe_b"],
      "expected_information_gain_reason": "Brief reason why this question should reduce uncertainty."
    }},
    {{
      "rank": 2,
      "question": "...",
      "question_form": "wh",
      "targets": ["..."],
      "distinguishes": ["..."],
      "expected_information_gain_reason": "..."
    }},
    {{
      "rank": 3,
      "question": "...",
      "question_form": "wh",
      "targets": ["..."],
      "distinguishes": ["..."],
      "expected_information_gain_reason": "..."
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