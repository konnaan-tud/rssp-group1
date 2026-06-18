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

# The simulator's APPLICABLE_MIN_COSINE (set below) controls which branches
# return None. The EIG is then scaled by the fraction of top-k belief MASS
# that can actually answer the question — heavy discount if most recipes
# (by belief) can't answer. Set to 1.0 to disable, 0.5 to require half of
# the mass to be answerable for full credit.
MIN_ANSWERABLE_MASS_FRACTION = 0.5

# When the generic-question fallback rescues a wh-question whose embedding
# didn't clear the per-recipe cosine gate, we multiply its EIG by this
# factor. Specific questions whose embeddings DID clear the gate keep
# their full EIG, so they outrank generic-fallback questions in the
# planner's ranking when both are available. Generic-fallback questions
# still produce a non-zero EIG so they survive when nothing specific is
# on offer. 0.5 is a balance: specific wins ~2× margin, generic still
# clears the 0.10 bit EIG threshold most of the time.
GENERIC_FALLBACK_EIG_PENALTY = 0.5

# Minimum cosine between the question and the best near-future scene for
# the planner to consider this recipe-branch "applicable" to the question.
# If the best cosine is below this, simulate_answer returns None, the
# branch contributes zero to EIG, and if every top-k branch returns None
# the question's overall EIG comes out at 0 → gate skips. This is the
# planner-level analogue of the human stub's "doesn't apply" fallback, and
# it prevents questions that don't fit the session state from ever being
# asked.
APPLICABLE_MIN_COSINE = 0.35


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
    min_cosine: float = APPLICABLE_MIN_COSINE,
    mode: str = "future",
) -> str | None:
    """
    Hypothetical scene-style answer the human would give if `recipe_name`
    were the truth.

    Strategy (V3, scene-aware with adaptive temporal locality):
      1. Take the recipe's UNSEEN scenes (semantic-match — handled by
         BeliefUpdaterV3.unseen_recipe_scenes).
      2. Restrict to the next K scenes, where K = `_adaptive_window(...)`
         if no explicit override is passed.
      3. Pick the one whose embedding is most semantically similar to the
         question.
      4. APPLICABILITY CHECK — if that best cosine is below `min_cosine`,
         return None. The question doesn't fit any of this recipe's
         near-future scenes, so the cook wouldn't answer it naturally.
         Returning None propagates "no useful answer" to expected_information_gain
         which gives the branch zero contribution. If every top-k branch
         is None, the question's overall EIG comes out at 0 and the gate
         correctly skips it — no wasted VLM call, no doesn't-apply turn.
    """
    question_text = (question.get("question") or "").strip()
    if not question_text:
        return None

    unseen = belief_updater.unseen_recipe_scenes(recipe_name)
    if not unseen:
        return None

    # SEQUENCE-AWARE FILTER: drop unseen scenes that lie BEFORE the
    # cook's current position in this recipe. The belief updater already
    # cares about order (via the order_consistency bonus), so the
    # simulator should too. Without this filter the simulator can pick
    # a recipe scene from the past — e.g. by Window 6 of pesto pasta,
    # "toasts pine nuts" (scene 2) was returned as the cook's "next"
    # action even though the cook is already at scene 7.
    #
    # In CLARIFICATION mode we intentionally do NOT apply the cursor
    # filter: clarification questions ask about the CURRENT (just-
    # observed) action, which can map to a skipped-earlier scene as
    # easily as to a future one. The cook may have done step 2 off-camera
    # and the ambiguous observation is finally revealing it.
    if mode != "clarification":
        cursor = belief_updater.recipe_cursor(recipe_name)
        if cursor >= 0:
            all_scenes = belief_updater.recipe_sentences.get(recipe_name, [])
            future_unseen = [
                s for i, s in enumerate(all_scenes)
                if i > cursor and s in unseen
            ]
            if future_unseen:
                unseen = future_unseen
            # If no scenes lie strictly after the cursor (cook has reached
            # the recipe's last matched scene with skips earlier), fall back
            # to the original unseen list so we always have something to
            # simulate.

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
    best_cos = cosines[best_idx]

    # Applicability check: low cosine means the question doesn't fit this
    # recipe's near-future. Treat as no answer.
    if best_cos < min_cosine:
        return None

    return near[best_idx]


def simulate_polar_answer(
    question: dict,
    recipe_name: str,
    belief_updater: BeliefUpdaterV3,
    min_cosine: float = APPLICABLE_MIN_COSINE,
    mode: str = "future",
) -> str | None:
    """
    Polar analogue of simulate_answer. Returns "yes" or "no" — what
    `recipe_name` would answer to the question's asserted proposition — or
    None if the proposition is irrelevant to this recipe's near-future (no
    credit, branch contributes zero to EIG).

    "yes" iff the asserted proposition matches one of the recipe's adaptive
    near-future scenes above min_cosine; otherwise "no". The None case mirrors
    simulate_answer's applicability check: if the proposition is far outside
    this recipe's near-future, the cook couldn't meaningfully answer.
    """
    proposition = (question.get("proposition") or "").strip()
    if not proposition:
        return None

    unseen = belief_updater.unseen_recipe_scenes(recipe_name)
    if not unseen:
        return None

    # Same sequence-aware filter as simulate_answer — drop scenes that
    # are before the cook's current recipe-position (skipped on camera).
    # In CLARIFICATION mode we skip this filter (the proposition may
    # refer to a skipped-earlier scene).
    if mode != "clarification":
        cursor = belief_updater.recipe_cursor(recipe_name)
        if cursor >= 0:
            all_scenes = belief_updater.recipe_sentences.get(recipe_name, [])
            future_unseen = [
                s for i, s in enumerate(all_scenes)
                if i > cursor and s in unseen
            ]
            if future_unseen:
                unseen = future_unseen

    window = _adaptive_window(belief_updater, recipe_name)
    near = unseen[:max(1, window)]

    p_vec = belief_updater._embedder.embed(proposition)
    scene_vecs = belief_updater._embedder.embed_batch(near)
    best_cos = max(_cosine(p_vec, v) for v in scene_vecs)

    return "yes" if best_cos >= min_cosine else "no"


# ═══════════════════════════════════════════════════════════════════════════
# Expected Information Gain
# ═══════════════════════════════════════════════════════════════════════════

def expected_information_gain(
    question: dict,
    belief_updater: BeliefUpdaterV3,
    top_k: int = 5,
    polar: bool = False,
) -> tuple[float, list[dict]]:
    """
    Measured EIG of `question` under the current belief.

    Iterates over the top-k recipes by current belief, simulates each
    recipe's answer to the question, applies it to a *copy* of the belief
    updater, and averages H(B) − H(B | a) weighted by P(r).

    When polar=True, each recipe answers "yes"/"no" to the question's
    proposition: "yes" is incorporated as an observation (the affirmed
    proposition), "no" via incorporate_negative_answer (persistent penalty).
    When polar=False, the wh path returns a scene-style answer.

    Returns (eig, per_branch_details). Each branch dict captures the
    simulated answer, predicted posterior entropy, and ΔH — useful for
    debugging and for the figure in the paper.
    """
    h_now = belief_updater.entropy()
    candidates = sorted(
        belief_updater.belief.items(), key=lambda x: -x[1]
    )[:top_k]

    # First pass: try the standard cosine-gated simulator per recipe.
    raw_answers: list[str | None] = []
    for recipe_name, _ in candidates:
        if polar:
            ans = simulate_polar_answer(question, recipe_name, belief_updater)
        else:
            ans = simulate_answer(question, recipe_name, belief_updater)
        raw_answers.append(ans)

    # GENERIC-QUESTION FALLBACK (wh only). If EVERY top-k branch returned
    # None, it usually means the question is too generic to match any
    # specific scene by cosine — e.g. "What are you adding next?" or
    # "What is the next step?". The question's embedding is abstract and
    # the recipe scenes are concrete, so the best cosine for every recipe
    # sits below APPLICABLE_MIN_COSINE = 0.35.
    #
    # For polar questions we don't apply the fallback: a yes/no question
    # whose proposition doesn't match any recipe's near-future genuinely
    # IS unanswerable (the cook can't say "yes" to a scene that isn't
    # coming up). For wh-questions, "what's next?" maps naturally to each
    # recipe's literal first near-future scene — that's what a real cook
    # would answer.
    fallback_applied = False
    if not polar and all(a is None for a in raw_answers):
        has_any_unseen = any(
            belief_updater.unseen_recipe_scenes(r)
            for r, _ in candidates
        )
        if has_any_unseen:
            print("  [planner] generic-question fallback: no scene cleared "
                  f"cosine {APPLICABLE_MIN_COSINE} for any recipe — using "
                  f"each recipe's literal next scene (EIG will be penalised "
                  f"by ×{GENERIC_FALLBACK_EIG_PENALTY} so specific questions "
                  f"outrank generic ones).")
            for i, (recipe_name, _) in enumerate(candidates):
                unseen = belief_updater.unseen_recipe_scenes(recipe_name)
                raw_answers[i] = unseen[0] if unseen else None
            fallback_applied = True

    eig = 0.0
    branches: list[dict] = []
    for (recipe_name, p_r), ans in zip(candidates, raw_answers):
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
            if polar:
                if ans == "yes":
                    clone.incorporate_answer(question["proposition"])
                else:
                    clone.incorporate_negative_answer(question["proposition"])
            else:
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

    # Answerability gate: scale EIG by the fraction of top-k BELIEF MASS
    # that can actually answer this question. If most top-k branches
    # returned None (no applicable scene), the question is mostly
    # off-topic at this point in the session, and the gate should skip it.
    #
    # The scaling is gentle when most mass can answer (close to 1.0) and
    # aggressive when only a minority can (small multiplier). A question
    # with 30% answerable mass keeps 30% of its raw EIG. This naturally
    # ties EIG to the question's practical applicability.
    total_mass = sum(b["p"] for b in branches)
    answerable_mass = sum(
        b["p"] for b in branches if b["answer"] is not None
    )
    if total_mass > 0:
        answerability = answerable_mass / total_mass
    else:
        answerability = 0.0

    # When most mass is answerable (>= MIN_ANSWERABLE_MASS_FRACTION), no
    # penalty. Below that, scale linearly.
    if answerability < MIN_ANSWERABLE_MASS_FRACTION:
        eig *= answerability

    # Generic-question penalty: if the EIG was earned via the fallback path
    # (literal next scene per recipe instead of cosine-matched scene),
    # multiply by GENERIC_FALLBACK_EIG_PENALTY so specific questions whose
    # embeddings cleared the cosine gate outscore generic ones in ranking.
    if fallback_applied:
        eig *= GENERIC_FALLBACK_EIG_PENALTY

    return eig, branches


# Embedding-level dedup threshold. Two questions whose sentence-encoder
# cosine is above this are treated as near-identical paraphrases of each
# other — the later one is dropped from the candidate pool, even if Qwen
# ignored the "ALREADY ASKED" prompt instruction.
#
# Calibration: 0.85 was too aggressive — it killed entire candidate
# batches in late windows once the discriminating-question space was
# exhausted, and the orchestrator skipped asking even when the belief
# was still uncertain (top P around 0.30). 0.92 only fires for
# near-verbatim repeats ("What is being added to the skillet?" vs
# "What is being added to the skillet?") and lets distinct rewordings
# through. When dedup STILL kills everything, rank_questions falls back
# to the un-deduped pool — see _maybe_relax_dedup below.
DEDUP_COSINE_THRESHOLD = 0.92


def _drop_paraphrases_of_asked(
    questions: list[dict],
    asked_questions: list[str],
    belief_updater: BeliefUpdaterV3,
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> list[dict]:
    """
    Drop any candidate whose embedding cosine to a previously-asked
    question exceeds `threshold`. Backstop for when Qwen ignores the
    "ALREADY ASKED" prompt section.
    """
    if not asked_questions:
        return questions

    asked_vecs = belief_updater._embedder.embed_batch(asked_questions)
    kept: list[dict] = []
    for q in questions:
        q_text = (q.get("question") or "").strip()
        if not q_text:
            continue
        q_vec = belief_updater._embedder.embed(q_text)
        too_similar = False
        for a_vec, a_text in zip(asked_vecs, asked_questions):
            if _cosine(q_vec, a_vec) >= threshold:
                print(f"  [dedup] dropped near-duplicate "
                      f"(cos≥{threshold}): {q_text!r}  vs  {a_text!r}")
                too_similar = True
                break
        if not too_similar:
            kept.append(q)
    return kept


def rank_questions(
    questions: list[dict],
    belief_updater: BeliefUpdaterV3,
    top_k: int = 5,
    polar: bool = False,
    asked_questions: list[str] | None = None,
) -> list[dict]:
    """
    Annotate each candidate question with measured EIG and per-branch
    detail, sorted by EIG descending.

    If `asked_questions` is provided, near-paraphrases of previously
    asked questions are dropped BEFORE EIG ranking — see
    `_drop_paraphrases_of_asked`.

    Each returned dict has the original question fields plus:
      - "measured_eig": float, in bits
      - "branches":     list of per-recipe simulation results
    """
    # Dedup the candidate pool against previously-asked questions. If
    # dedup leaves NOTHING, fall back to the original pool so the gate
    # can still find a question to ask. The orchestrator was previously
    # refusing to ask in late windows because dedup zeroed every
    # candidate — but the situation called for asking (top P ≈ 0.30,
    # entropy > 1.5). Better to ask a slight rewording of an earlier
    # question than to ask nothing at all.
    if asked_questions:
        original = questions
        deduped = _drop_paraphrases_of_asked(
            questions, asked_questions, belief_updater,
        )
        if deduped:
            questions = deduped
        else:
            print("  [dedup] all candidates were near-duplicates of prior "
                  "asks; falling back to un-deduped pool so the gate has "
                  "something to score.")
            questions = original

    ranked = []
    for q in questions:
        eig, branches = expected_information_gain(
            q, belief_updater, top_k, polar=polar
        )
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
    polar: bool = False,
    asked_questions: list[str] | None = None,
) -> tuple[bool, dict | None, list[dict]]:
    """
    Combined gate. Ask iff *all three* hold:

      (a) budget not burnt:   questions_asked < max_questions
      (b) uncertainty high:   H(belief) > entropy_threshold
      (c) useful Q exists:    max measured EIG > eig_threshold

    `asked_questions` is the list of question texts already fired this
    session. When provided, near-paraphrases are dropped BEFORE the EIG
    ranking so we never gate on a duplicate.

    Returns (should_ask, best_question_or_None, all_ranked).
    `all_ranked` is returned even when we choose not to ask so the caller
    can log scores for the paper.
    """
    if questions_asked >= max_questions:
        return False, None, []

    if belief_updater.entropy() < entropy_threshold:
        return False, None, []  # already confident — don't bother the cook

    ranked = rank_questions(
        questions, belief_updater, top_k,
        polar=polar, asked_questions=asked_questions,
    )
    if not ranked or ranked[0]["measured_eig"] < eig_threshold:
        # Nothing on offer would help. Better to wait for more observations.
        return False, None, ranked

    return True, ranked[0], ranked