"""
probability/test_belief_v3.py
------------------------------
Tests BeliefUpdaterV3 with a simulated carbonara cooking session.

Session:
  Clip 1: "pours water into a large pot"          → shared, ambiguous
  Clip 2: "cracks eggs into a mixing bowl"        → somewhat distinctive
  --- clarification answer after clip 2 ---
  Clip 3: "grates pecorino into the mixing bowl"  → distinctive
  Clip 4: "dices pancetta on a cutting board"     → confirms carbonara
  Clip 5: "cooks pancetta in a skillet"           → strong confirmation

Ground truth: carbonara

Run from project root:
    python probability/test_belief_v3.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probability.belief_updater_v3 import BeliefUpdaterV3


# ── Session definition ─────────────────────────────────────────────────────

GROUND_TRUTH = "carbonara"

CLIPS = [
    "pours water into a large pot",
    "cracks eggs into a mixing bowl",
    "grates pecorino into the mixing bowl",
    "dices pancetta on a cutting board",
    "cooks pancetta in a skillet",
]

CLARIFICATION_ANSWER = "I am adding eggs and pecorino to make a creamy sauce"
ANSWER_AFTER_CLIP = 2  # fire answer after this clip number (1-indexed)


# ── Display helpers ────────────────────────────────────────────────────────

def print_divider(label: str):
    print(f"\n── {label} {'─' * max(1, 54 - len(label))}")


def print_distribution(belief: dict[str, float], similarities: dict[str, float]):
    """Print belief and similarity scores side by side, sorted by belief."""
    print()
    print(f"  {'Recipe':35s}  {'Sim':6s}  {'Prob':6s}  Bar")
    print(f"  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*30}")
    for name, prob in sorted(belief.items(), key=lambda x: -x[1]):
        sim = similarities.get(name, 0.0)
        bar = "█" * int(prob * 30)
        print(f"  {name:35s}  {sim:.4f}  {prob:.4f}  {bar}")


# ── Main test ──────────────────────────────────────────────────────────────

def run_test():
    updater = BeliefUpdaterV3(temperature=0.01)

    print(f"\n{'═'*60}")
    print(f"  Belief Updater V3 — DTW Test")
    print(f"  Ground truth : {GROUND_TRUTH}")
    print(f"  Recipes      : {updater.N}")
    print(f"  Max entropy  : {updater.max_entropy():.3f} bits")
    print(f"{'═'*60}")

    entropy_log = []  # (label, entropy, ig)

    for i, clip in enumerate(CLIPS, start=1):
        entropy_before = updater.entropy()

        # Update from observation
        belief = updater.update(clip)
        entropy_after = updater.entropy()
        ig_obs = entropy_before - entropy_after

        print_divider(f"Clip {i}: \"{clip}\"")
        print(f"  entropy : {entropy_after:.4f} bits  |  IG_obs : {ig_obs:.4f} bits")
        top, prob = updater.top_recipe()
        print(f"  top     : {top} ({prob:.4f})")
        print_distribution(belief, updater.current_similarities())

        entropy_log.append((f"Clip {i}", entropy_after, ig_obs))

        # Fire clarification answer after specified clip
        if i == ANSWER_AFTER_CLIP:
            print_divider(f"Clarification Answer")
            print(f"  \"{CLARIFICATION_ANSWER}\"")

            h_before = updater.entropy()
            belief = updater.incorporate_answer(CLARIFICATION_ANSWER)
            h_after = updater.entropy()
            ig_q = updater.ig_q()

            print(f"\n  H before : {h_before:.4f} bits")
            print(f"  H after  : {h_after:.4f} bits")
            print(f"  IG_Q     : {ig_q:.4f} bits  ← clarification gain")
            top, prob = updater.top_recipe()
            print(f"  top      : {top} ({prob:.4f})")
            print_distribution(belief, updater.current_similarities())

            entropy_log.append(("Answer", h_after, ig_q))

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    top, prob = updater.top_recipe()
    print(f"  Final prediction : {top}")
    print(f"  Correct          : {top == GROUND_TRUTH}")
    print(f"  Final prob       : {prob:.4f}")
    print(f"  Final entropy    : {updater.entropy():.4f} bits")

    print(f"\n  Information gain summary:")
    print(f"  {'Step':25s}  {'Entropy':8s}  {'IG':8s}")
    print(f"  {'─'*25}  {'─'*8}  {'─'*8}")
    for label, ent, ig in entropy_log:
        marker = " ← answer" if label == "Answer" else ""
        print(f"  {label:25s}  {ent:.4f}    {ig:.4f}{marker}")

    print(f"\n  IG_Q log:")
    for entry in updater.ig_q_log():
        print(f"    answer  : {entry['answer']}")
        print(f"    H before: {entry['entropy_before']}")
        print(f"    H after : {entry['entropy_after']}")
        print(f"    IG_Q    : {entry['ig_q']}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run_test()