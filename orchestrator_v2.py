"""
orchestrator_v2.py
------------------
Minimal end-to-end demo of the V2 pipeline with EIG-based clarification.

What it does:
  1. Walks through the same 5-recipe mock session as test_belief_v2.py,
     feeding each window's observations to BeliefUpdaterV2.
  2. After every window, calls questioning_planner.should_ask() to decide
     whether to ask a clarification question. The gate combines:
        - question budget          (don't ask too often)
        - entropy threshold        (don't ask when confident)
        - measured EIG threshold   (don't ask if no Q would help)
  3. If the gate says yes, picks the question with highest measured EIG,
     "asks" it (the real human answer is hard-coded for this demo), and
     feeds the answer back through incorporate_answer.
  4. Logs predicted EIG vs realised IG_Q for the chosen question.

How to wire in the real VLM later:
  Replace MOCK_QUESTIONS with the output of
  `questioning_pipeline.run_qwen_prompt(...)` parsed into the same dict
  shape. Replace REAL_ANSWER with the actual human input.

Run:
  python orchestrator_v2.py
"""

from __future__ import annotations

import json

from belief_updater_v2 import BeliefUpdaterV2
from questioning_planner import should_ask


# ─────────────────────────────────────────────────────────────────────────
# Mock session — same trajectory as test_belief_v2.py
# (creamy cucumber salad: ambiguous early, clear once sour cream is whisked)
# ─────────────────────────────────────────────────────────────────────────

WINDOWS = [
    {"window": 1,
     "ingredients": ["cucumber"],
     "actions":     ["slice"]},
    {"window": 2,
     "ingredients": ["cucumber", "onion"],
     "actions":     ["slice", "chop"]},
    {"window": 3,
     "ingredients": ["cucumber", "onion", "sour cream",
                     "white sugar", "white vinegar"],
     "actions":     ["slice", "chop", "whisk"]},
    {"window": 4,
     "ingredients": ["cucumber", "onion", "sour cream",
                     "white sugar", "white vinegar"],
     "actions":     ["slice", "chop", "whisk", "toss", "cover", "refrigerate"]},
]


# ─────────────────────────────────────────────────────────────────────────
# Mock candidate questions
# In the real pipeline these come from questioning_pipeline.run_qwen_prompt().
# Keep the dict shape identical so the swap is a one-line change.
# ─────────────────────────────────────────────────────────────────────────

MOCK_QUESTIONS = [
    {
        # Off-domain question — recipes don't track tools. Planner should
        # score this near zero EIG (the professor's example).
        "rank": 1,
        "question": "What tool are you using to slice the cucumber?",
        "question_form": "wh",
        "targets": ["knife", "mandoline"],
        "distinguishes": [],
    },
    {
        # Recipe-grounded discriminating question — different top recipes
        # would give different answers (sour cream vs white vinegar vs ...).
        # Should score high.
        "rank": 2,
        "question": "Which ingredient are you adding to the dressing?",
        "question_form": "wh",
        "targets": ["sour cream", "white vinegar", "mayonnaise", "lime"],
        "distinguishes": ["creamy cucumber salad",
                          "classic vinegar cucumber salad"],
    },
    {
        # Recipe-grounded but non-discriminating — every recipe ends in
        # toss/cover/refrigerate. Should score moderate-to-low.
        "rank": 3,
        "question": "What are you doing next with the salad?",
        "question_form": "wh",
        "targets": ["toss", "marinate", "cover", "refrigerate"],
        "distinguishes": [],
    },
]


# ─────────────────────────────────────────────────────────────────────────
# Question-conditional human answers. Same idea as orchestrator_v2_vlm.py:
# route by question keywords so the human's answer matches what was asked,
# letting predicted EIG and realised IG_Q be compared meaningfully.
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_HUMAN_ANSWER = "I am adding sour cream to make a creamy dressing"


def pick_human_answer(question_text: str) -> str:
    q = question_text.lower()
    if any(w in q for w in ["tool", "knife", "utensil"]):
        return "I am using a knife and a cutting board"
    if "next" in q and ("step" in q or "doing" in q or "salad" in q):
        return "I am about to whisk sour cream into the dressing"
    if "dressing" in q or "sauce" in q:
        return "I am whisking sour cream into a creamy dressing"
    if "ingredient" in q:
        return "I am adding sour cream to the dressing"
    return DEFAULT_HUMAN_ANSWER


# ─────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────

def main():
    # Reuse the same mock vector store the test session uses.
    bu = BeliefUpdaterV2(recipe_terms_path="test_recipe_terms.json")

    questions_asked = 0
    print(f"\nMax entropy: {bu.max_entropy():.3f} bits")

    for w in WINDOWS:
        print(f"\n── Window {w['window']} ──────────────────────────────────")

        # 1. Observation update
        bu.update(ingredients_seen=w["ingredients"], actions_seen=w["actions"])
        top, p = bu.top_recipe()
        print(f"  entropy : {bu.entropy():.4f} bits")
        print(f"  top     : {top} ({p:.3f})")

        # 2. Gate: should we ask, and if so which question?
        # Budget is disabled for now (max_questions huge). Restore with 2
        # to bring it back.
        ask, best_q, ranked = should_ask(
            belief_updater=bu,
            questions=MOCK_QUESTIONS,
            entropy_threshold=0.5,   # below this we consider the belief sharp
            eig_threshold=0.10,      # below this no question is worth asking
            questions_asked=questions_asked,
            max_questions=9_999,     # budget disabled
        )

        # Always log the scores so we can see what the planner thought.
        if ranked:
            print("  candidate EIGs:")
            for q in ranked:
                print(f"    [{q['measured_eig']:.3f}]  {q['question']}")

        # 3. If the gate fires, ask + incorporate the question-conditional
        # human answer.
        if ask:
            answer = pick_human_answer(best_q["question"])
            print(f"\n  >> Asking: {best_q['question']}")
            print(f"     human   : {answer}")
            print(f"     predicted EIG: {best_q['measured_eig']:.4f} bits")

            h_before = bu.entropy()
            bu.incorporate_answer(answer)
            h_after = bu.entropy()
            realised = h_before - h_after

            print(f"     realised IG_Q: {realised:.4f} bits "
                  f"(Δ vs predicted: {realised - best_q['measured_eig']:+.4f})")
            questions_asked += 1

    # Final report
    top, p = bu.top_recipe()
    print("\n" + "═" * 60)
    print(f"Final prediction : {top} ({p:.4f})")
    print(f"Final entropy    : {bu.entropy():.4f} bits")
    print(f"Questions asked  : {questions_asked}")
    print(f"IG_Q history     : {json.dumps(bu.ig_q_history(), indent=2)}")


if __name__ == "__main__":
    main()
