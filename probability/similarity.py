"""
probability/similarity.py
--------------------------
Hybrid similarity between an observation sequence and a recipe sequence.

Two components:
  1. F1 scene matching   — for each observed scene, find best cosine match
                           in recipe. For each recipe scene, find best match
                           in observations. F1 harmonic mean of both.
                           Catches distinctive ingredients with strong signal.

  2. Order consistency   — checks whether matched scene pairs appear in the
                           same relative order (subsequence check).
                           Small bonus multiplier on the F1 score.

Final similarity:
    sim = f1 × (1 + alpha × order_ratio)

Usage:
    from probability.similarity import hybrid_similarity
    sim = hybrid_similarity(obs_sequence, recipe_sequence)
"""

from __future__ import annotations

import numpy as np


# ── Cosine similarity ──────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Higher = more similar."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


# ── F1 scene matching ──────────────────────────────────────────────────────

def f1_scene_match(
    obs_seq: list[np.ndarray],
    recipe_seq: list[np.ndarray],
    threshold: float = 0.3,
) -> tuple[float, list[tuple[int, int]]]:
    """
    Bidirectional F1 scene matching between observation and recipe sequences.

    Precision: for each observed scene, best cosine match in recipe.
               Measures how well observations are explained by the recipe.
    Recall:    for each recipe scene, best cosine match in observations.
               Measures how well the recipe covers what has been observed.
    F1:        harmonic mean of precision and recall.

    Parameters
    ----------
    obs_seq    : ordered list of observation scene vectors
    recipe_seq : ordered list of recipe scene vectors
    threshold  : minimum cosine similarity to count as a match

    Returns
    -------
    f1_score : float in [0, 1]
    matches  : list of (obs_idx, recipe_idx) pairs for order check
    """
    n = len(obs_seq)
    m = len(recipe_seq)

    if n == 0 or m == 0:
        return 0.0, []

    # Build full similarity matrix
    sim_matrix = np.zeros((n, m), dtype=np.float32)
    for i, obs_vec in enumerate(obs_seq):
        for j, rec_vec in enumerate(recipe_seq):
            sim_matrix[i, j] = cosine_similarity(obs_vec, rec_vec)

    # Precision: for each obs scene, best match in recipe
    precision_scores = []
    best_recipe_for_obs = []
    for i in range(n):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        precision_scores.append(best_sim if best_sim >= threshold else 0.0)
        best_recipe_for_obs.append((i, best_j))
    precision = float(np.mean(precision_scores))

    # Recall: for each recipe scene, best match in obs
    recall_scores = []
    for j in range(m):
        best_sim = float(np.max(sim_matrix[:, j]))
        recall_scores.append(best_sim if best_sim >= threshold else 0.0)
    recall = float(np.mean(recall_scores))

    # F1
    if precision + recall == 0:
        return 0.0, best_recipe_for_obs
    f1 = 2.0 * precision * recall / (precision + recall)

    return f1, best_recipe_for_obs


# ── Order consistency ──────────────────────────────────────────────────────

def order_consistency(
    matches: list[tuple[int, int]],
) -> float:
    """
    Check what fraction of matched (obs_idx, recipe_idx) pairs form
    a valid subsequence — appear in the same relative order.

    A pair (i1, j1) and (i2, j2) is consistent if:
        i1 < i2 implies j1 <= j2

    Parameters
    ----------
    matches : list of (obs_idx, recipe_idx) tuples

    Returns
    -------
    float in [0, 1] — fraction of consecutive pairs that are order-consistent
    """
    if len(matches) <= 1:
        return 1.0

    sorted_matches = sorted(matches, key=lambda x: x[0])

    consistent = 0
    total = 0

    for k in range(len(sorted_matches) - 1):
        _, j1 = sorted_matches[k]
        _, j2 = sorted_matches[k + 1]
        total += 1
        if j1 <= j2:
            consistent += 1

    return consistent / total if total > 0 else 1.0


# ── Hybrid similarity ──────────────────────────────────────────────────────

def hybrid_similarity(
    obs_seq: list[np.ndarray],
    recipe_seq: list[np.ndarray],
    threshold: float = 0.3,
    alpha: float = 0.3,
) -> float:
    """
    Hybrid similarity combining F1 scene matching and order consistency.

    sim = f1 × (1 + alpha × order_ratio)

    Parameters
    ----------
    obs_seq    : ordered list of observation scene vectors
    recipe_seq : ordered list of recipe scene vectors
    threshold  : minimum cosine similarity to count as a match (default 0.3)
    alpha      : weight of order consistency bonus (default 0.3)

    Returns
    -------
    float in [0, ~1.3] — higher = better match.
    Values above 1.0 possible due to order bonus — softmax handles this.
    """
    if not obs_seq or not recipe_seq:
        return 0.0

    f1, matches = f1_scene_match(obs_seq, recipe_seq, threshold)
    order_ratio = order_consistency(matches)
    return f1 * (1.0 + alpha * order_ratio)