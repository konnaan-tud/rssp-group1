"""
probability/test_belief_v3.py
------------------------------
Tests BeliefUpdaterV3 with a simulated carbonara cooking session.

Flow:
  Clip 1 → observation → one targeted question → answer
  Clip 2 → observation → one targeted question → answer
  Clip 3 → observation only
  Clip 4 → observation only
  Clip 5 → observation only

Questions chosen to maximise information gain at each point.

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

QUESTIONS = [
    {
        "question": "Are you using butter, cream, or olive oil as the base of this dish?",
        "answer": "No, the fat comes from cured meat and eggs",
    },
    {
        "question": "Are you making a sauce based on eggs?",
        "answer": "Yes, eggs are the primary binding ingredient in my sauce",
    },
    None,
    None,
    None,
]


# ── Display helpers ────────────────────────────────────────────────────────

def print_divider(label: str = "", width: int = 60):
    print(f"\n── {label} {'─' * max(1, width - len(label) - 4)}")


def print_distribution(
    belief: dict[str, float],
    similarities: dict[str, float],
    contrastive: dict[str, float],
    top_n: int = 8,
):
    print()
    print(f"  {'Recipe':35s}  {'Sim':6s}  {'Cont':6s}  {'Prob':6s}  Bar")
    print(f"  {'─'*35}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*20}")
    sorted_belief = sorted(belief.items(), key=lambda x: -x[1])
    for name, prob in sorted_belief[:top_n]:
        sim = similarities.get(name, 0.0)
        cont = contrastive.get(name, 0.0)
        bar = "█" * int(prob * 20)
        print(f"  {name:35s}  {sim:.4f}  {cont:+.4f}  {prob:.4f}  {bar}")
    if len(sorted_belief) > top_n:
        remaining = sum(p for _, p in sorted_belief[top_n:])
        print(f"  {'... other recipes':35s}  {'':6s}  {'':6s}  {remaining:.4f}")


# ── Main test ──────────────────────────────────────────────────────────────

def run_test():
    updater = BeliefUpdaterV3(temperature=0.05)

    print(f"\n{'═'*60}")
    print(f"  Belief Updater V3 — Contrastive Similarity Test")
    print(f"  Ground truth : {GROUND_TRUTH}")
    print(f"  Recipes      : {updater.N}")
    print(f"  Temperature  : {updater.temperature}")
    print(f"  Max entropy  : {updater.max_entropy():.3f} bits")
    print(f"{'═'*60}")

    ig_obs_log = []
    ig_q_log_local = []

    for i, (clip, qa) in enumerate(zip(CLIPS, QUESTIONS), start=1):

        # ── Observation ────────────────────────────────────────────────────
        entropy_before_obs = updater.entropy()
        belief = updater.update(clip)
        entropy_after_obs = updater.entropy()
        ig_obs = entropy_before_obs - entropy_after_obs

        print_divider(f"Clip {i}: \"{clip}\"")
        print(f"  H before obs : {entropy_before_obs:.4f} bits")
        print(f"  H after obs  : {entropy_after_obs:.4f} bits")
        print(f"  IG_obs       : {ig_obs:.4f} bits")
        top, prob = updater.top_recipe()
        print(f"  top recipe   : {top} ({prob:.4f})")
        print_distribution(
            belief,
            updater.current_similarities(),
            updater.current_contrastive(),
        )

        ig_obs_log.append((f"Clip {i}", entropy_after_obs, ig_obs))

        # ── Clarification (only for Clips 1 and 2) ────────────────────────
        if qa is not None:
            print_divider(f"Question after Clip {i}")
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")

            h_before = updater.entropy()
            belief = updater.incorporate_answer(qa["answer"])
            h_after = updater.entropy()
            ig_q = updater.ig_q()

            print(f"\n  H before : {h_before:.4f} bits")
            print(f"  H after  : {h_after:.4f} bits")
            print(f"  IG_Q     : {ig_q:.4f} bits  ← clarification gain")
            top, prob = updater.top_recipe()
            print(f"  top      : {top} ({prob:.4f})")
            print_distribution(
                belief,
                updater.current_similarities(),
                updater.current_contrastive(),
            )

            ig_q_log_local.append((f"Q{i}", h_after, ig_q, ig_obs))

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    top, prob = updater.top_recipe()
    print(f"  Final prediction : {top}")
    print(f"  Correct          : {top == GROUND_TRUTH}")
    print(f"  Final prob       : {prob:.4f}")
    print(f"  Final entropy    : {updater.entropy():.4f} bits")

    print(f"\n  IG comparison — observation vs clarification:")
    print(f"  {'Step':10s}  {'IG_obs':8s}  {'IG_Q':8s}  {'IG_Q > IG_obs':15s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*15}")
    for label, _, ig_q, ig_obs in ig_q_log_local:
        check = "✓" if ig_q > ig_obs else "✗"
        print(f"  {label:10s}  {ig_obs:.4f}    {ig_q:.4f}    {check}")

    print(f"\n  Full IG log:")
    print(f"  {'Step':25s}  {'Type':15s}  {'IG':8s}")
    print(f"  {'─'*25}  {'─'*15}  {'─'*8}")
    for label, _, ig in ig_obs_log:
        print(f"  {label:25s}  {'observation':15s}  {ig:.4f}")
        matching_q = [q for q in ig_q_log_local if q[0] == f"Q{label[-1]}"]
        if matching_q:
            print(f"  {'':25s}  {'clarification':15s}  {matching_q[0][2]:.4f}")

    print(f"{'═'*60}\n")


if __name__ == "__main__":
    run_test()