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


def build_recipe_context(
    belief_updater,                       # BeliefUpdaterV2 (duck-typed to avoid circular import)
    active_threshold: float = 0.02,
    top_k: int = 8,
) -> str:
    """
    Build a compact text summary of session state for the VLM prompt.

    Replaces the old "dump every recipe graph JSON" approach. In V2 the
    canonical recipe representation is the per-term vocabulary inside
    BeliefUpdaterV2, so we draw the context directly from there.

    The returned block contains three sections:
      1. ALREADY OBSERVED — terms the VLM should not ask about
      2. ACTIVE CANDIDATES — top recipes with their REMAINING vocabulary
         (these are the strings the VLM should draw targets from)
      3. VOCABULARY POOL — flat list of every remaining term across the
         active candidates, as a final guard for the targets rule.

    Parameters
    ----------
    belief_updater   : a BeliefUpdaterV2 instance
    active_threshold : recipes with P(r) below this are dropped from context
    top_k            : hard upper bound on number of candidates listed
    """
    bel = belief_updater.belief

    # 1. Pick the candidates to surface
    active = [(name, p) for name, p in bel.items() if p >= active_threshold]
    active.sort(key=lambda x: -x[1])
    if len(active) > top_k:
        active = active[:top_k]
    if not active:                                # safety net
        active = sorted(bel.items(), key=lambda x: -x[1])[:top_k]

    # 2. Already-observed terms (drawn from the belief updater directly)
    obs_terms = sorted(belief_updater._obs_terms.keys())
    obs_set = set(obs_terms)

    # 3. Per-recipe remaining vocabulary
    unseen_by_recipe: dict[str, list[str]] = {}
    for name, _p in active:
        rec_terms = belief_updater.recipe_term_vectors[name]
        unseen_by_recipe[name] = sorted(
            t for t in rec_terms if t not in obs_set
        )

    # 4. Compose the text block
    lines: list[str] = []

    # 4a. Belief distribution up front — this is what the VLM should reason
    # about. High-entropy belief = ask broad questions; sharp belief = ask
    # questions that distinguish the top recipes only.
    import math
    H = -sum(p * math.log2(p) for _, p in active if p > 0)
    Hmax = math.log2(len(bel)) if len(bel) > 1 else 0.0
    lines.append("CURRENT BELIEF DISTRIBUTION (sorted by probability):")
    for name, p in active:
        bar = "█" * int(round(p * 30))
        lines.append(f"  {p:.3f}  {bar:<30}  {name}")
    lines.append(f"  entropy: {H:.3f} bits over {len(active)} active recipes "
                 f"(max possible: {Hmax:.3f} bits)")
    lines.append("")

    lines.append("ALREADY OBSERVED (do not ask about these):")
    if obs_terms:
        lines.append("  " + ", ".join(obs_terms))
    else:
        lines.append("  (nothing observed yet)")
    lines.append("")

    lines.append("ACTIVE CANDIDATE RECIPES")
    lines.append("(Each recipe lists its current probability and its "
                 "REMAINING vocabulary — the ingredients and actions still")
    lines.append("unseen for that recipe. These remaining terms are the "
                 "strings you should draw your 'targets' from.)")
    lines.append("")
    for name, p in active:
        lines.append(f"- {name}  (p={p:.3f})")
        unseen = unseen_by_recipe[name]
        if unseen:
            lines.append(f"    remaining: {', '.join(unseen)}")
        else:
            lines.append(f"    remaining: (all terms already observed)")
    lines.append("")

    # 5. Flat pool — every term you may use as a target
    pool = sorted({t for terms in unseen_by_recipe.values() for t in terms})
    lines.append("VOCABULARY POOL (every concrete string you may use as a "
                 "target — do NOT invent anything outside this list):")
    lines.append("  " + (", ".join(pool) if pool else "(empty)"))

    return "\n".join(lines)


def build_question_prompt(recipe_context: str) -> str:
    """
    Build the prompt that asks the VLM to generate clarification questions.

    The `recipe_context` argument is a text block produced by
    build_recipe_context() — it contains the active candidates, their
    remaining vocabulary, and the already-observed terms. This replaces the
    earlier "dump full static_graphs JSON" approach, which was a V1 leftover
    that didn't match V2's bag-of-terms recipe representation.
    """

    prompt = f"""
You are a clarification-question planner for a VLM-based cooking observer.

The observer has watched a short cooking video clip and currently has uncertainty
over which recipe is being prepared. The candidate recipes are listed below.

YOUR TASK
Generate exactly 3 open wh-questions whose answers would reduce uncertainty
over the current recipe belief state as much as possible.

═══ HOW TO FILL "targets" — READ CAREFULLY ═══

The "targets" field is the most important part of each question. It is the
list of CONCRETE possible answers the human might give — actual ingredient
names or action names that distinguish the candidate recipes from each
other.

A downstream planner uses these targets to test whether the question would
actually distinguish between recipes. Abstract placeholder names and
generic cooking terms CANNOT be matched against any recipe and make the
question useless.

GOOD targets — concrete strings drawn from the VOCABULARY POOL of the
current session (each example below uses a different food domain to show
the general shape):
    "targets": ["miso", "soy sauce", "fish sauce"]      ← Asian sauces
    "targets": ["bake", "fry", "grill"]                  ← cooking methods
    "targets": ["basil", "oregano", "rosemary"]          ← Mediterranean herbs

BAD targets (NEVER produce these — they cannot be matched):
    "targets": ["main_ingredient"]                       ← abstract placeholder
    "targets": ["stage_id"]                              ← schema field name
    "targets": ["next_action"]                           ← abstract placeholder
    "targets": ["first_step"]                            ← abstract placeholder
    "targets": ["ingredient or action being tested"]     ← description, not a value

Every string in "targets" must appear — in the same or very similar form —
in the VOCABULARY POOL listed in the SESSION CONTEXT below. Do NOT make up
new strings, do NOT copy example targets verbatim.

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

═══ SESSION CONTEXT ═══
{recipe_context}

═══ RESPONSE FORMAT ═══

Return ONLY valid JSON in the structure shown below.

★ CRITICAL ★
The examples below are from a DIFFERENT cooking domain (Asian stir-fry).
They illustrate the *structure* only — they are NOT valid answers for the
current session. You MUST:
  - Rewrite every question so it fits the recipes in the SESSION CONTEXT.
  - Replace every target with concrete vocabulary from the VOCABULARY POOL.
  - Replace every "distinguishes" entry with names of ACTIVE CANDIDATE
    recipes from the SESSION CONTEXT.

DO NOT copy the example targets ("tofu", "ginger", "soy sauce", etc.). They
do not appear in the current recipes and will be discarded by the planner.

EXAMPLE STRUCTURE (Asian stir-fry domain — do NOT reuse these values):
{{
  "questions": [
    {{
      "rank": 1,
      "question": "Which protein are you adding to the wok?",
      "question_form": "wh",
      "targets": ["tofu", "chicken", "shrimp"],
      "distinguishes": ["vegetarian pad thai", "kung pao chicken"],
      "expected_information_gain_reason": "Different stir-fries use different proteins; naming it identifies the dish family."
    }},
    {{
      "rank": 2,
      "question": "Which aromatic are you frying first?",
      "question_form": "wh",
      "targets": ["ginger", "garlic", "lemongrass"],
      "distinguishes": ["thai green curry", "vietnamese pho"],
      "expected_information_gain_reason": "Different cuisines start with different aromatics; this disambiguates the recipe family."
    }},
    {{
      "rank": 3,
      "question": "What sauce base will you use?",
      "question_form": "wh",
      "targets": ["soy sauce", "fish sauce", "oyster sauce"],
      "distinguishes": ["thai basil chicken", "general tso's chicken"],
      "expected_information_gain_reason": "Each sauce maps to a different dish family."
    }}
  ]
}}

Now produce YOUR JSON for the current SESSION CONTEXT. Remember: targets
MUST be drawn from the VOCABULARY POOL above. Do not invent new strings,
do not reuse the example targets.
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
    # Smoke test: build a context from a real BeliefUpdaterV2 instance and
    # show the prompt that would be sent. We don't actually call Qwen here
    # (that takes ~30 s); pipe through orchestrator_v2_vlm.py for an end-to-end run.
    from belief_updater_v2 import BeliefUpdaterV2

    bu = BeliefUpdaterV2(recipe_terms_path="recipe_terms.json")
    bu.update(ingredients_seen=["cucumber", "onion"],
              actions_seen=["slice", "chop"])

    context = build_recipe_context(bu)
    print("\n----- Recipe context -----\n")
    print(context)

    prompt = build_question_prompt(recipe_context=context)
    print("\n----- Full prompt -----\n")
    print(prompt)