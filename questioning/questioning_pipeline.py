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
    asked_questions: list[str] | None = None,
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

    # Show previously-asked questions so the VLM knows not to paraphrase
    # them. Display-only — no hard filter, just transparency.
    if asked_questions:
        lines.append("")
        lines.append("ALREADY ASKED THIS SESSION (do NOT repeat or paraphrase):")
        for i, q in enumerate(asked_questions, start=1):
            lines.append(f"  {i}. {q}")

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

═══ THE 3 QUESTIONS MUST COVER DIFFERENT ASPECTS ═══

Do not produce three questions of the same kind (e.g. three ingredient
questions). Pick ONE question from each of these three categories:

  1. INGREDIENT — what the cook is adding or using.
        e.g. "Which ___ are you adding next?"
        e.g. "What ___ is going into the bowl?"

  2. METHOD or TECHNIQUE — how the cook is preparing or handling something.
        e.g. "How are you preparing the ___?"
        e.g. "Are you whisking, scrambling, or cooking the eggs?"
        e.g. "Which technique are you using for the ___?"

  3. SEQUENCE or ORDER — when something happens in the process.
        e.g. "When in the process do you add the ___?"
        e.g. "What did you do BEFORE the ___?"
        e.g. "Will you ___ before or after ___?"

When the candidate recipes SHARE INGREDIENTS but differ in METHOD or
ORDER, ingredient questions cannot tell them apart — only method and
sequence questions can. Use this as guidance, not a hard rule:

  - If top recipes' next scenes diverge in INGREDIENT, an ingredient
    question is usually the best rank-1.
  - If top recipes' next scenes share the same ingredient but differ in
    HOW it's prepared, a method question is the best rank-1.
  - If top recipes' steps come in different ORDER, a sequence question
    is the best rank-1.

You may produce three questions all of one category if that's what the
session actually calls for — but try to make the three questions
discriminative, not redundant. Two near-identical questions waste a
slot.

For each question return:
  - "question":      the wh-question (starts with what/which/where/when/why/how)
  - "question_form": "wh"
  - "category":      one of "ingredient", "method", or "sequence" — try
                     to cover different categories across the 3 questions
                     unless the session genuinely calls for the same
                     category (see guidance above)
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
      "question": "<INGREDIENT question — about which item is being added>",
      "question_form": "wh",
      "category": "ingredient",
      "targets": [
        "<scene sentence drawn from candidate A's remaining scenes>",
        "<scene sentence drawn from candidate B's remaining scenes>"
      ],
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }},
    {{
      "rank": 2,
      "question": "<METHOD question — about how the cook is handling/preparing something>",
      "question_form": "wh",
      "category": "method",
      "targets": [
        "<scene sentence describing the technique candidate A uses>",
        "<scene sentence describing the technique candidate B uses>"
      ],
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }},
    {{
      "rank": 3,
      "question": "<SEQUENCE question — about when or in what order something happens>",
      "question_form": "wh",
      "category": "sequence",
      "targets": [
        "<scene sentence at a specific point in candidate A's order>",
        "<scene sentence at a different point in candidate B's order>"
      ],
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }}
  ]
}}

★ Rules ★
  - Each of the 3 questions must use a DIFFERENT category value
    (one "ingredient", one "method", one "sequence").
  - Replace EVERY <placeholder> with concrete content drawn from the
    SESSION CONTEXT above.
  - Targets must be scene sentences from the REMAINING SCENES list — do
    not invent ingredients or dishes that aren't in the candidates.
  - Prefer questions about the NEXT step, not steps far in the future.
  - Return ONLY the JSON object, no commentary.
"""

    return prompt.strip()


def build_question_prompt_polar(recipe_context: str) -> str:
    """
    Polar (yes/no) variant of build_question_prompt. Each question asserts a
    SINGLE proposition the cook can confirm or deny. The 'proposition' field
    holds the scene-style statement being asserted — on a "yes" it is
    incorporated as an observation, on a "no" it is negated via
    incorporate_negative_answer. Categories mirror the wh prompt so the two
    conditions are matched on what they probe.
    """
    prompt = f"""
You generate yes/no (polar) clarification questions for a cooking observer
that is uncertain which recipe the cook is making. The candidate recipes and
their remaining (unseen) scene sentences are below.

TASK
Generate exactly 3 polar (yes/no) questions whose answers would reduce
uncertainty over the current recipe belief. Each question must be answerable
with a plain "yes" or "no" and must assert ONE concrete proposition.

═══ THE 3 QUESTIONS MUST COVER DIFFERENT ASPECTS ═══

Pick ONE question from each of these three categories:

  1. INGREDIENT — whether the cook is adding or using a specific item.
        e.g. "Are you adding pecorino next?"

  2. METHOD or TECHNIQUE — whether the cook prepares something a specific way.
        e.g. "Are you whisking the eggs rather than scrambling them?"

  3. SEQUENCE or ORDER — whether something happens at a specific point.
        e.g. "Will you drain the pasta before adding the egg mixture?"

A good polar question SPLITS the active candidates: ideally about half of the
top recipes' near-future would answer "yes" and half "no". A question that
every candidate answers the same way gains nothing — avoid those.

For each question return:
  - "question":      the yes/no question (answerable with plain yes/no)
  - "question_form": "polar"
  - "category":      one of "ingredient", "method", or "sequence" — cover a
                     different category across the 3 questions unless the
                     session genuinely calls for the same one
  - "proposition":   the scene-style statement the question asserts, drawn
                     from the REMAINING SCENES of one active candidate (e.g.
                     "A cook adds pecorino to the bowl."). This is the
                     statement a "yes" confirms and a "no" denies.
  - "distinguishes": names of recipes this question helps separate (from the
                     ACTIVE CANDIDATES below)
  - "expected_information_gain_reason": one sentence

Do not ask "Are you making carbonara?" or name a recipe directly.

═══ SESSION CONTEXT ═══
{recipe_context}

═══ EXAMPLE STRUCTURE (placeholder values — do NOT copy verbatim) ═══
{{
  "questions": [
    {{
      "rank": 1,
      "question": "<INGREDIENT yes/no question>",
      "question_form": "polar",
      "category": "ingredient",
      "proposition": "<scene sentence the question asserts, from a candidate>",
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }},
    {{
      "rank": 2,
      "question": "<METHOD yes/no question>",
      "question_form": "polar",
      "category": "method",
      "proposition": "<scene sentence the question asserts, from a candidate>",
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }},
    {{
      "rank": 3,
      "question": "<SEQUENCE yes/no question>",
      "question_form": "polar",
      "category": "sequence",
      "proposition": "<scene sentence the question asserts, from a candidate>",
      "distinguishes": ["<candidate A>", "<candidate B>"],
      "expected_information_gain_reason": "<one-sentence reason>"
    }}
  ]
}}

★ Rules ★
  - Each question must be answerable yes/no and assert exactly ONE proposition.
  - Each of the 3 questions must use a DIFFERENT category value
    (one "ingredient", one "method", one "sequence").
  - The proposition must be a scene sentence from the REMAINING SCENES list —
    do not invent ingredients or dishes that aren't in the candidates.
  - Prefer questions about the NEXT step, not steps far in the future.
  - Return ONLY the JSON object, no commentary.
"""

    return prompt.strip()
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