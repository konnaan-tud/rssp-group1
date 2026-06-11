"""
questioning_planner.py
----------------------
Scores candidate clarification questions by *measured* Expected Information
Gain (EIG) under the current belief, using a per-recipe answer simulator.

The pipeline this slots into:

    1. questioning_pipeline.py asks the VLM for K candidate questions
    2. questioning_planner.rank_questions(qs, bu) scores each one by EIG
    3. questioning_planner.should_ask(bu, qs) gates the decision and picks best
    4. caller asks the human, then bu.incorporate_answer(real_answer)

EIG formulation (Bayesian experimental design):

    EIG(Q) ≈ Σ_{r in top-k}  P(r) · [ H(B) − H(B | simulated_answer_if_r(Q)) ]

For each candidate question we ask "if recipe r is the truth, what answer
would the human give?", simulate that answer on a *copy* of the belief, and
average the resulting entropy drops weighted by the current belief over r.

This is option A from the discussion — deterministic, no extra VLM calls
needed during scoring.
"""

from __future__ import annotations

import contextlib
import io

from belief_updater_v2 import BeliefUpdaterV2


# ═══════════════════════════════════════════════════════════════════════════
# Per-recipe answer simulation
# ═══════════════════════════════════════════════════════════════════════════

def _fuzzy_pick(target: str, candidates: set[str]) -> str | None:
    """
    Substring match between a question target and a recipe term.
    Handles "vinegar" → "white vinegar", "sour cream" → "sour cream", etc.
    Returns the matched candidate term, or None.
    """
    t = target.lower().strip()
    if not t:
        return None
    for term in candidates:
        if t in term or term in t:
            return term
    return None


def _term_rarity(term: str, belief_updater: BeliefUpdaterV2) -> float:
    """
    Inverse document frequency over the recipe set.
    Rare term → high value → preferred as a discriminating answer.
    """
    n = sum(
        1 for r_terms in belief_updater.recipe_term_vectors.values()
        if term in r_terms
    )
    return 1.0 / max(n, 1)


def simulate_answer(
    question: dict,
    recipe_name: str,
    belief_updater: BeliefUpdaterV2,
) -> str | None:
    """
    Hypothetical answer the human would give if `recipe_name` were the truth.

    Strategy:
      0. If the question's targets don't appear anywhere in the recipe
         vocabulary (e.g. tool/colour question — recipes don't track those),
         answer literally with targets[0]. All recipes return the same string
         → EIG correctly comes out near zero.
      1. Otherwise, if this recipe's *unseen* vocabulary overlaps with the
         question's targets, return that overlap term (with fuzzy matching).
      2. Otherwise (the recipe is grounded in the question's domain but
         doesn't share a target), return the recipe's rarest unseen term
         — its most discriminating "this is what I'm doing instead" answer.
      3. If nothing is unseen, return None (nothing new to say).

    Returns the simulated answer string, or None.
    """
    recipe_terms = set(belief_updater.recipe_term_vectors[recipe_name].keys())
    obs_terms = set(belief_updater._obs_terms.keys())
    unseen = recipe_terms - obs_terms

    targets = question.get("targets", []) or []

    # Step 0 — grounded check: do any targets touch the recipe vocabulary at all?
    all_recipe_terms = belief_updater.known_recipe_terms
    grounded = any(
        _fuzzy_pick(t, all_recipe_terms) is not None for t in targets
    )

    if not grounded:
        # Off-domain question (tool, colour, etc.) — every recipe gives the
        # same literal answer, so EIG is zero by construction.
        return targets[0] if targets else None

    # Step 1 — target hits this recipe's unseen terms?
    for target in targets:
        match = _fuzzy_pick(target, unseen)
        if match is not None:
            return match

    # Step 2 — this recipe doesn't have any of the target ingredients.
    # Pick its rarest unseen term as a "this is what I'd say instead" answer.
    if not unseen:
        return None
    return max(unseen, key=lambda t: _term_rarity(t, belief_updater))


# ═══════════════════════════════════════════════════════════════════════════
# Expected Information Gain
# ═══════════════════════════════════════════════════════════════════════════

def expected_information_gain(
    question: dict,
    belief_updater: BeliefUpdaterV2,
    top_k: int = 5,
) -> tuple[float, list[dict]]:
    """
    Measured EIG of `question` under the current belief.

    Iterates over the top-k recipes by current belief, simulates each one's
    answer to the question, applies it to a copy of the belief updater, and
    averages H(B) − H(B | a) weighted by P(r).

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

        # Simulate on a copy; suppress the [compound terms extracted] print
        # so the planner doesn't drown the logs.
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
    belief_updater: BeliefUpdaterV2,
    top_k: int = 5,
) -> list[dict]:
    """
    Annotate each candidate question with measured EIG and per-branch detail,
    sorted by EIG descending.

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
    belief_updater: BeliefUpdaterV2,
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

    Rationale:
      - (a) Cooking is interactive; asking every window annoys the human.
      - (b) If the belief is already sharp, an answer adds little.
      - (c) Even with high entropy, if no available question actually
            distinguishes the top recipes, asking is wasted breath
            (the professor's "what tool are you using?" example).

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
