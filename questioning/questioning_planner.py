"""
questioning/questioning_planner.py
----------------------------------
Scores candidate clarification questions by *measured* Expected Information
Gain (EIG) under the current belief, using a per-recipe answer simulator.

V3 port of the V2 planner. The high-level shape is identical:

    1. questioning_pipeline.py asks the VLM for K candidate questions
    2. questioning_planner.rank_questions(qs, bu) scores each one by EIG
    3. questioning_planner.should_ask(bu, qs) gates the decision and picks best
    4. caller asks the human, then bu.incorporate_answer(real_answer)

EIG formulation (Bayesian experimental design):

    EIG(Q) ≈ Σ_{r in top-k}  P(r) · [ H(B) − H(B | simulated_answer_if_r(Q)) ]

What changed from V2:
- Recipes are no longer bags of terms; they are ordered scene sequences.
- simulate_answer now produces a SCENE-STYLE SENTENCE per candidate recipe,
  chosen as the recipe's unseen scene most semantically related to the
  question. This replaces V2's "match targets against term vocabulary"
  logic.
- The EIG loop, ranking, and gating are unchanged — they only depend on
  bu.copy(), bu.incorporate_answer(), and bu.entropy(), all of which V3
  supports with the same signatures.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from probability.belief_updater_v3 import BeliefUpdaterV3


# ═══════════════════════════════════════════════════════════════════════════
# Per-recipe scene simulation
# ═══════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# Temporal window base. The actual window per recipe grows as that recipe's
# early scenes are observed — see _adaptive_window() below.
NEAR_FUTURE_WINDOW = 3


def _adaptive_window(
    belief_updater: BeliefUpdaterV3,
    recipe_name: str,
    base: int = NEAR_FUTURE_WINDOW,
) -> int:
    """
    Number of recipe scenes to consider for `recipe_name`.

    Grows as the recipe's early scenes get semantically observed. With
    `base=3`, the simulator looks at the next 3 unseen scenes when the
    session has just started, but at scenes 5–10 after several scenes
    have been observed. This unlocks ambiguous-twin handling: when top-k
    recipes share their first few scenes, the simulator now looks past
    those shared scenes to find where the recipes actually diverge.

    Formula: window = base + (number of scenes already observed for this
    recipe), capped at the number of remaining unseen scenes.
    """
    total = len(belief_updater.recipe_sentences.get(recipe_name, []))
    unseen_count = len(belief_updater.unseen_recipe_scenes(recipe_name))
    observed_count = max(0, total - unseen_count)
    return min(base + observed_count, max(unseen_count, 1))


def simulate_answer(
    question: dict,
    recipe_name: str,
    belief_updater: BeliefUpdaterV3,
    near_future_window: int | None = None,
) -> str | None:
    """
    Hypothetical scene-style answer the human would give if `recipe_name`
    were the truth.

    Strategy (V3, scene-aware with adaptive temporal locality):
      1. Take the recipe's UNSEEN scenes (semantic-match — handled by
         BeliefUpdaterV3.unseen_recipe_scenes).
      2. Restrict to the next K scenes, where K = `_adaptive_window(...)`
         if no explicit override is passed. Default behaviour grows K as
         more of the recipe's scenes are observed.
      3. Pick the one whose embedding is most semantically similar to the
         question.
      4. If nothing is unseen, return None.

    The default adaptive K replaces the previous fixed K=3 — it lets the
    planner reach later recipe scenes when the early ones are already
    confirmed, which is what makes "ambiguous twin" questions scorable.
    """
    question_text = (question.get("question") or "").strip()
    if not question_text:
        return None

    unseen = belief_updater.unseen_recipe_scenes(recipe_name)
    if not unseen:
        return None

    if near_future_window is None:
        near_future_window = _adaptive_window(belief_updater, recipe_name)

    # Restrict to the next K unseen scenes (already in recipe order).
    near = unseen[:max(1, near_future_window)]

    # Embed the question and each near-future scene. The embedder is
    # cached, so repeated calls during simulation are cheap.
    q_vec = belief_updater._embedder.embed(question_text)
    scene_vecs = belief_updater._embedder.embed_batch(near)

    cosines = [_cosine(q_vec, v) for v in scene_vecs]
    best_idx = int(max(range(len(cosines)), key=lambda i: cosines[i]))
    return near[best_idx]


# ═══════════════════════════════════════════════════════════════════════════
# Expected Information Gain
# ═══════════════════════════════════════════════════════════════════════════

def expected_information_gain(
    question: dict,
    belief_updater: BeliefUpdaterV3,
    top_k: int = 5,
) -> tuple[float, list[dict]]:
    """
    Measured EIG of `question` under the current belief.

    Iterates over the top-k recipes by current belief, simulates each
    recipe's answer to the question, applies it to a *copy* of the belief
    updater, and averages H(B) − H(B | a) weighted by P(r).

    Returns (eig, per_branch_details). Each branch dict captures the
    simulated answer, predicted posterior entropy, and ΔH — useful for
    debugging and for the figure in the paper.
    """
    h_now = belief_updater.entropy()
    candidates = sorted(
        belief_updater.belief.items(), key=lambda x: -x[1]
    )[:top_k]

    eig = 0.0
    branches: list[dict] = []
    for recipe_name, p_r in candidates:
        ans = simulate_answer(question, recipe_name, belief_updater)
        if ans is None:
            # Nothing to simulate — treat as zero contribution.
            branches.append({
                "recipe": recipe_name,
                "p": round(p_r, 4),
                "answer": None,
                "h_after": round(h_now, 4),
                "delta_h": 0.0,
            })
            continue

        # Simulate on a copy; suppress any prints from incorporate_answer.
        clone = belief_updater.copy()
        with contextlib.redirect_stdout(io.StringIO()):
            clone.incorporate_answer(ans)
        h_after = clone.entropy()
        delta = h_now - h_after

        eig += p_r * delta
        branches.append({
            "recipe": recipe_name,
            "p": round(p_r, 4),
            "answer": ans,
            "h_after": round(h_after, 4),
            "delta_h": round(delta, 4),
        })

    return eig, branches


def rank_questions(
    questions: list[dict],
    belief_updater: BeliefUpdaterV3,
    top_k: int = 5,
) -> list[dict]:
    """
    Annotate each candidate question with measured EIG and per-branch
    detail, sorted by EIG descending.

    Each returned dict has the original question fields plus:
      - "measured_eig": float, in bits
      - "branches":     list of per-recipe simulation results
    """
    ranked = []
    for q in questions:
        eig, branches = expected_information_gain(q, belief_updater, top_k)
        ranked.append({
            **q,
            "measured_eig": round(eig, 4),
            "branches": branches,
        })
    ranked.sort(key=lambda x: -x["measured_eig"])
    return ranked


# ═══════════════════════════════════════════════════════════════════════════
# Asking-gate — should we ask a question right now?
# ═══════════════════════════════════════════════════════════════════════════

def should_ask(
    belief_updater: BeliefUpdaterV3,
    questions: list[dict],
    entropy_threshold: float = 0.5,
    eig_threshold: float = 0.10,
    questions_asked: int = 0,
    max_questions: int = 2,
    top_k: int = 5,
) -> tuple[bool, dict | None, list[dict]]:
    """
    Combined gate. Ask iff *all three* hold:

      (a) budget not burnt:   questions_asked < max_questions
      (b) uncertainty high:   H(belief) > entropy_threshold
      (c) useful Q exists:    max measured EIG > eig_threshold

    Returns (should_ask, best_question_or_None, all_ranked).
    `all_ranked` is returned even when we choose not to ask so the caller
    can log scores for the paper.
    """
    if questions_asked >= max_questions:
        return False, None, []

    if belief_updater.entropy() < entropy_threshold:
        return False, None, []  # already confident — don't bother the cook

    ranked = rank_questions(questions, belief_updater, top_k)
    if not ranked or ranked[0]["measured_eig"] < eig_threshold:
        # Nothing on offer would help. Better to wait for more observations.
        return False, None, ranked

    return True, ranked[0], ranked
