"""
orchestrator_v2_vlm.py
----------------------
End-to-end demo of the V3 pipeline with VLM-generated clarification
questions.

What it does:
  1. Walks through a simulated carbonara session (one scene sentence per
     window) and feeds each clip to BeliefUpdaterV3.
  2. After every window the cheap gate (entropy threshold) decides whether
     it's worth calling the VLM at all.
  3. If yes, generates candidate clarification questions with Qwen2.5-VL,
     scores each by expected information gain on a *copy* of the belief,
     and applies the EIG gate.
  4. If a question fires, the hard-coded human (pick_human_answer) gives a
     scene-style answer that gets appended to the observation sequence via
     bu.incorporate_answer(...).
  5. Predicted EIG vs realised IG_Q is logged per question.

Negation handling is intentionally NOT implemented here. In V3 the belief
is recomputed from scratch each step, so the V2-style direct down-weight
would be wiped out on the next observation. If the embedder mishandles
negation answers (likely for the "I am not using X" case), that's a known
limitation noted in the docs.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 python orchestrator_v2_vlm.py
"""

from __future__ import annotations

import json
import re
import sys
import os
from pathlib import Path

# Make package imports work when run from the repo root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from probability.belief_updater_v3 import BeliefUpdaterV3
from questioning.questioning_pipeline import (
    build_recipe_context,
    build_question_prompt,
    run_qwen_prompt,
)
from questioning.questioning_planner import should_ask, simulate_answer


# ─────────────────────────────────────────────────────────────────────────
# Mock observation trajectory — carbonara session.
# Each entry is one scene sentence (one VLM-observed clip in the real
# pipeline). In production these come from run_dynamic_loop.py fed into
# bu.update() one window at a time.
# ─────────────────────────────────────────────────────────────────────────

WINDOWS = [
    "A cook pours water into a large pot.",
    "A cook cracks eggs into a mixing bowl.",
    "A cook grates pecorino into the mixing bowl.",
    "A cook dices pancetta on a cutting board.",
    "A cook cooks pancetta in a skillet.",
]


# ─────────────────────────────────────────────────────────────────────────
# Human-answer stub
#
# The cook is making carbonara. pick_human_answer returns what the cook
# would plausibly say for the given question:
#   - Keyword overrides for questions whose natural answer is a negation
#     or summary that isn't a single recipe scene
#     (e.g. "Are you using butter?" → "no, just pancetta and eggs").
#   - Default: ask the planner's own simulator what scene the ground-truth
#     recipe (carbonara) would name for this question. That is the same
#     mechanism the planner uses to predict per-recipe answers, applied
#     to the actual recipe being cooked. This is a much better demo stub
#     than a hardcoded default — the human stays "in character" with the
#     recipe they're cooking.
#
# In real experiments this whole function is replaced with input() (typed
# human) or audio capture.
# ─────────────────────────────────────────────────────────────────────────

GROUND_TRUTH = "carbonara"


def pick_human_answer(question_text: str, bu: BeliefUpdaterV3) -> str:
    q = question_text.lower()

    # Overrides: questions whose true answer is a negation/summary, not a
    # single recipe scene.
    if any(w in q for w in ["butter", "cream"]) and any(w in q for w in ["fat", "base", "using"]):
        return ("No butter and no cream; the fat in carbonara comes from "
                "rendered pancetta and egg yolks.")
    if "herb" in q:
        return "Carbonara does not use fresh herbs, only black pepper."

    # Default: ask the simulator what the carbonara cook would say.
    simulated = simulate_answer(
        {"question": question_text}, GROUND_TRUTH, bu,
    )
    if simulated:
        # Convert the recipe-style sentence into a first-person human answer.
        # "A cook cracks eggs into a mixing bowl." → "I am cracking eggs into a mixing bowl."
        return simulated.replace("A cook ", "I am ").replace(
            "cracks", "cracking").replace("grates", "grating").replace(
            "dices", "dicing").replace("cooks", "cooking").replace(
            "pours", "pouring").replace("drops", "dropping").replace(
            "drains", "draining").replace("transfers", "transferring").replace(
            "stirs", "stirring").replace("tosses", "tossing")

    return "I am cracking eggs into a mixing bowl."  # safety fallback


# ─────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────

# Gate thresholds. Either condition being satisfied skips the question:
#   - entropy already sharp (low residual uncertainty)
#   - top recipe already dominant (one clear leader)
# Picked so the carbonara session asks at windows 1 and 2 (uncertain) but
# stops by window 3 once carbonara is at 80%+.
ENTROPY_THRESHOLD = 1.5      # bits; below this skip
TOP_PROB_CEILING = 0.75      # if top recipe at or above this, skip
EIG_THRESHOLD = 0.10         # bits; below this no question is worth asking

# Question budget — disabled for now (kept for easy re-enable).
# MAX_QUESTIONS = 2
MAX_QUESTIONS = 9_999

# Context prompt limits.
PROMPT_TOP_K = 6
PROMPT_ACTIVE_THRESHOLD = 0.02


# ─────────────────────────────────────────────────────────────────────────
# VLM output parsing
# ─────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_vlm_questions(raw_response: str) -> list[dict]:
    """
    Parse Qwen's JSON output into the dict shape that the planner expects.

    Tolerates code fences, leading commentary, missing fields, wrong types.
    Malformed entries are silently dropped.
    """
    text = _FENCE_RE.sub("", raw_response).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        print("  [parse] No JSON object found in VLM response.")
        return []

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        print(f"  [parse] JSON decode failed: {e}")
        return []

    raw_questions = parsed.get("questions", [])
    if not isinstance(raw_questions, list):
        print("  [parse] 'questions' field is missing or not a list.")
        return []

    clean: list[dict] = []
    for i, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            continue
        question_text = (q.get("question") or "").strip()
        if not question_text:
            continue
        targets = q.get("targets", [])
        distinguishes = q.get("distinguishes", [])
        clean.append({
            "rank":          q.get("rank", i + 1),
            "question":      question_text,
            "question_form": q.get("question_form", "wh"),
            "targets":       targets if isinstance(targets, list) else [],
            "distinguishes": distinguishes if isinstance(distinguishes, list) else [],
        })
    return clean


def generate_vlm_questions(belief_updater: BeliefUpdaterV3) -> list[dict]:
    """
    Build the prompt, call Qwen, parse the JSON response.
    """
    context = build_recipe_context(
        belief_updater,
        active_threshold=PROMPT_ACTIVE_THRESHOLD,
        top_k=PROMPT_TOP_K,
    )
    n_active = sum(1 for p in belief_updater.belief.values()
                   if p >= PROMPT_ACTIVE_THRESHOLD)
    print(f"  [VLM] Active recipes in context: {min(n_active, PROMPT_TOP_K)}")

    prompt = build_question_prompt(recipe_context=context)
    print("  [VLM] Calling Qwen (this loads the model on first call)...")
    raw_response = run_qwen_prompt(prompt)

    print("\n  [VLM raw response]")
    print("  " + raw_response.replace("\n", "\n  "))

    questions = parse_vlm_questions(raw_response)
    print(f"\n  [VLM] Parsed {len(questions)} candidate question(s).")
    return questions


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def main():
    bu = BeliefUpdaterV3()

    questions_asked = 0
    print(f"\nMax entropy: {bu.max_entropy():.3f} bits")

    for i, sentence in enumerate(WINDOWS, start=1):
        print(f"\n── Window {i} ──────────────────────────────────")
        print(f"  observation : {sentence}")

        # 1. Observation update
        bu.update(sentence)
        top, p = bu.top_recipe()
        print(f"  entropy : {bu.entropy():.4f} bits")
        print(f"  top     : {top} ({p:.3f})")

        # 2. CHEAP gate first — skip the VLM call when the answer is already
        # obvious. Budget gate is disabled for now (uncomment to restore).
        # if questions_asked >= MAX_QUESTIONS:
        #     print("  [gate] Question budget exhausted — skipping.")
        #     continue
        if bu.entropy() < ENTROPY_THRESHOLD:
            print(f"  [gate] Entropy {bu.entropy():.3f} < {ENTROPY_THRESHOLD} — skipping.")
            continue
        if p >= TOP_PROB_CEILING:
            print(f"  [gate] Top recipe at {p:.3f} ≥ {TOP_PROB_CEILING} — skipping.")
            continue

        # 3. EXPENSIVE step — generate candidate questions with the VLM.
        questions = generate_vlm_questions(bu)
        if not questions:
            print("  [gate] No usable questions from VLM — skipping.")
            continue

        # 4. Score with the planner. entropy_threshold=0.0 here because we
        # already vetted entropy above; only the EIG and budget checks remain.
        ask, best_q, ranked = should_ask(
            belief_updater=bu,
            questions=questions,
            entropy_threshold=0.0,
            eig_threshold=EIG_THRESHOLD,
            questions_asked=questions_asked,
            max_questions=MAX_QUESTIONS,
        )

        # Per-branch EIG breakdown — for each candidate question, show what
        # each top-k recipe is hypothesised to answer and how much entropy
        # would drop in that branch.
        #   p          : current belief probability for that recipe
        #   sim answer : scene the recipe would name if it were the truth
        #   drop       : entropy drop in that branch (H_before − H_after)
        #   weighted   : p × drop, the branch's share of the total EIG
        # Total EIG = sum of weighted contributions across the top-k branches.
        print("\n  candidate questions (with EIG breakdown):")
        for q in ranked:
            print(f"    EIG {q['measured_eig']:+.3f} bits  |  {q['question']}")
            for b in q["branches"]:
                if b["answer"] is None:
                    print(f"        - {b['recipe']:<28} p={b['p']:.3f}  "
                          f"(no simulated answer)")
                    continue
                weighted = b["p"] * b["delta_h"]
                ans = b["answer"]
                if len(ans) > 50:
                    ans = ans[:47] + "..."
                print(f"        - {b['recipe']:<28} p={b['p']:.3f}  "
                      f"sim → {ans!r:<55}  "
                      f"drop={b['delta_h']:+.3f}  weighted={weighted:+.3f}")

        if not ask:
            print("  [gate] Best EIG below threshold — skipping.")
            continue

        # 5. Fire the chosen question. The human stub answers using the
        # planner's simulator pointed at the ground-truth recipe — so it
        # gives a scene-style answer consistent with the recipe being
        # cooked.
        answer = pick_human_answer(best_q["question"], bu)

        print(f"\n  >> Asking : {best_q['question']}")
        print(f"     human   : {answer}")
        print(f"     predicted EIG : {best_q['measured_eig']:+.4f} bits")

        h_before = bu.entropy()
        bu.incorporate_answer(answer)
        h_after = bu.entropy()
        realised = h_before - h_after

        print(f"     realised IG_Q : {realised:+.4f} bits "
              f"(Δ vs predicted: {realised - best_q['measured_eig']:+.4f})")

        # NEW: show the belief distribution after the answer so it's clear
        # how the question reshaped uncertainty.
        post_top = sorted(bu.belief.items(), key=lambda x: -x[1])[:3]
        print(f"     belief after Q:")
        for name, prob in post_top:
            bar = "█" * int(round(prob * 30))
            print(f"        {prob:.3f}  {bar:<30}  {name}")

        questions_asked += 1

    # Final report
    top, p = bu.top_recipe()
    print("\n" + "═" * 60)
    print(f"Final prediction : {top} ({p:.4f})")
    print(f"Final entropy    : {bu.entropy():.4f} bits")
    print(f"Questions asked  : {questions_asked}")
    print(f"IG_Q history     : {json.dumps(bu.ig_q_log(), indent=2)}")


if __name__ == "__main__":
    main()
