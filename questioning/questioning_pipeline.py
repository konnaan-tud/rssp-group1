"""
questioning/questioning_pipeline.py
-----------------------------------
Builds the clarification-question prompt for the VLM and runs it.

V3 port of the V2 pipeline. The architectural shift:

V2 surfaced each recipe's REMAINING TERM VOCABULARY (e.g. "sour cream,
white vinegar, dill") plus a flat "vocabulary pool" the VLM had to draw
targets from.

V3 surfaces each recipe's REMAINING SCENE SENTENCES — the next likely
clip-level scenes the cook would produce if making that recipe. The VLM's
"targets" field now holds plausible scene-style answers rather than bare
ingredient words, because that's what the planner's simulator will compare
against.

Public functions:
  build_recipe_context(bu)         — text block of session state for the prompt
  build_question_prompt(context)   — full prompt string sent to Qwen
  run_qwen_prompt(prompt)          — load Qwen, generate, return string
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


# ═══════════════════════════════════════════════════════════════════════════
# Session context builder
# ═══════════════════════════════════════════════════════════════════════════

def build_recipe_context(
    belief_updater,                       # BeliefUpdaterV3 (duck-typed)
    active_threshold: float = 0.02,
    top_k: int = 6,
    future_scenes: int = 4,
) -> str:
    """
    Compose a compact text summary of session state for the VLM prompt.

    Sections, in order:
      1. CURRENT BELIEF DISTRIBUTION   — top recipes by probability, with bars
      2. ALREADY OBSERVED              — full sentences seen so far, in order
      3. ACTIVE CANDIDATE RECIPES      — for each, the next `future_scenes`
                                         unseen scene sentences from the recipe

    Parameters
    ----------
    belief_updater  : a BeliefUpdaterV3 instance
    active_threshold: recipes with P(r) below this are dropped from context
    top_k           : hard upper bound on number of candidates listed
    future_scenes   : per-recipe limit on how many remaining scenes to show
    """
    bel = belief_updater.belief

    # 1. Pick active candidates
    active = [(name, p) for name, p in bel.items() if p >= active_threshold]
    active.sort(key=lambda x: -x[1])
    if len(active) > top_k:
        active = active[:top_k]
    if not active:                                # safety net
        active = sorted(bel.items(), key=lambda x: -x[1])[:top_k]

    # 2. Already-observed sentences (clips + answers, in temporal order)
    observed = belief_updater.observed_sentences()

    # 3. Compose
    lines: list[str] = []

    # 3a. Belief distribution
    H = -sum(p * math.log2(p) for _, p in active if p > 0)
    Hmax = math.log2(len(bel)) if len(bel) > 1 else 0.0
    lines.append("CURRENT BELIEF DISTRIBUTION (sorted by probability):")
    for name, p in active:
        bar = "█" * int(round(p * 30))
        lines.append(f"  {p:.3f}  {bar:<30}  {name}")
    lines.append(
        f"  entropy: {H:.3f} bits over {len(active)} active recipes "
        f"(max possible: {Hmax:.3f} bits)"
    )
    lines.append("")

    # 3b. Observation trail
    lines.append("ALREADY OBSERVED (in temporal order — do not ask about these):")
    if observed:
        for i, s in enumerate(observed, start=1):
            lines.append(f"  {i}. {s}")
    else:
        lines.append("  (nothing observed yet)")
    lines.append("")

    # 3c. Per-recipe remaining scenes — this is the discriminating signal
    lines.append("ACTIVE CANDIDATE RECIPES — next likely scenes per recipe:")
    lines.append("(Each list shows the recipe's REMAINING scene sentences.")
    lines.append("Use them to identify what each recipe would distinctively")
    lines.append("do next; draw your candidate answers from these scenes.)")
    lines.append("")
    for name, p in active:
        unseen = belief_updater.unseen_recipe_scenes(name)[:future_scenes]
        lines.append(f"- {name}  (p={p:.3f})")
        if unseen:
            for s in unseen:
                lines.append(f"      • {s}")
        else:
            lines.append("      (all scenes already observed)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt template
# ═══════════════════════════════════════════════════════════════════════════

def build_question_prompt(recipe_context: str) -> str:
    """
    Build the full prompt sent to Qwen. `recipe_context` is the block
    produced by build_recipe_context(); it carries the belief distribution,
    observation trail, and per-recipe remaining scenes.
    """

    prompt = f"""
You generate clarification questions for a cooking observer that is
uncertain which recipe the cook is making. The candidate recipes and
their remaining (unseen) scene sentences are below.

TASK
Generate exactly 3 wh-questions whose answers would reduce uncertainty
over the current recipe belief.

For each question return:
  - "question":      the wh-question (starts with what/which/where/when/why/how)
  - "question_form": "wh"
  - "targets":       2-4 scene-style sentences a cook MIGHT say as an answer.
                     Each must be drawn from the REMAINING SCENES of one
                     active candidate, or a close paraphrase of one. They
                     should focus on what's about to happen NEXT, not far
                     in the future.
  - "distinguishes": names of recipes this question helps separate (from
                     the ACTIVE CANDIDATES below)
  - "expected_information_gain_reason": one sentence

Do not ask "What recipe are you making?" or "Which recipe is this?".

═══ SESSION CONTEXT ═══
{recipe_context}

═══ EXAMPLE STRUCTURE (placeholder values — do NOT copy verbatim) ═══
{{
  "questions": [
    {{
      "rank": 1,
      "question": "<a wh-question about an upcoming step>",
      "question_form": "wh",
      "targets": [
        "<scene sentence drawn from candidate A's remaining scenes>",
        "<scene sentence drawn from candidate B's remaining scenes>",
        "<scene sentence drawn from candidate C's remaining scenes>"
      ],
      "distinguishes": ["<candidate A>", "<candidate B>", "<candidate C>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }},
    {{ "rank": 2, "question": "...", ... }},
    {{ "rank": 3, "question": "...", ... }}
  ]
}}

★ Rules ★
  - Replace EVERY <placeholder> with concrete content drawn from the
    SESSION CONTEXT above.
  - Targets must be scene sentences from the REMAINING SCENES list — do
    not invent ingredients or dishes that aren't in the candidates.
  - Prefer questions about the NEXT step, not steps far in the future.
  - Return ONLY the JSON object, no commentary.
"""

    return prompt.strip()


# ═══════════════════════════════════════════════════════════════════════════
# Qwen runner
# ═══════════════════════════════════════════════════════════════════════════

def generate_text_with_model(
    prompt: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 768,
) -> str:
    """
    Text-only generation against an already-loaded Qwen2.5-VL model.
    Used by the orchestrator when it has loaded the VLM once (via
    observation_pipeline.video_observer.load_qwen_vlm) and wants to reuse
    that load for question generation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], return_tensors="pt", padding=True,
    ).to(device)

    print("Generating...", flush=True)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return response


def run_qwen_prompt(
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 768,
) -> str:
    """
    Backwards-compatible wrapper: loads Qwen2.5-VL fresh and runs a text
    prompt. Prefer load_qwen_vlm() + generate_text_with_model() for any
    workflow that runs more than one VLM call per session.
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

    response = generate_text_with_model(
        prompt, model, processor, device, max_new_tokens=max_new_tokens,
    )
    return response


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print the context + prompt without calling Qwen. Useful for fast
    # iteration on the prompt wording.
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from probability.belief_updater_v3 import BeliefUpdaterV3

    bu = BeliefUpdaterV3()
    bu.update("A cook pours water into a large pot.")
    bu.update("A cook cracks eggs into a mixing bowl.")

    context = build_recipe_context(bu)
    print("\n----- Recipe context -----\n")
    print(context)

    prompt = build_question_prompt(recipe_context=context)
    print("\n----- Full prompt -----\n")
    print(prompt)
