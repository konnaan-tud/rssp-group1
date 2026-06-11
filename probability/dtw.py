"""
probability/dtw.py
------------------
Dynamic Time Warping (DTW) for comparing two sequences of embedding vectors.

Uses cosine distance as the element-wise cost function.
Normalises total path cost by path length so sequences of different
lengths are comparable.

Usage:
    from probability.dtw import dtw_distance, dtw_similarity

    cost = dtw_distance(obs_sequence, recipe_sequence)
    sim  = dtw_similarity(obs_sequence, recipe_sequence)
"""

from __future__ import annotations

import numpy as np


# ── Cosine distance ────────────────────────────────────────────────────────

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance between two vectors.
    Returns value in [0, 2]. Lower = more similar.
    0 = identical direction, 2 = opposite direction.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b))
    # Clamp to [-1, 1] to handle floating point drift
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


# ── DTW ───────────────────────────────────────────────────────────────────

def dtw_distance(
    seq_a: list[np.ndarray],
    seq_b: list[np.ndarray],
) -> float:
    """
    Compute normalised DTW distance between two sequences of vectors.

    Uses cosine distance as the element-wise cost.
    Normalises by path length (length of longer sequence) so sequences
    of different lengths are comparable.

    Parameters
    ----------
    seq_a : list of np.ndarray — observation sequence
    seq_b : list of np.ndarray — recipe sequence

    Returns
    -------
    float — normalised DTW cost in [0, 2]. Lower = better match.
    """
    n = len(seq_a)
    m = len(seq_b)

    if n == 0 or m == 0:
        return 2.0  # maximum distance if either sequence is empty

    # Build cost matrix
    # dtw[i][j] = minimum cost to align seq_a[:i+1] with seq_b[:j+1]
    dtw = np.full((n, m), np.inf, dtype=np.float32)

    # Initialise first cell
    dtw[0, 0] = cosine_distance(seq_a[0], seq_b[0])

    # First row — can only come from the left
    for j in range(1, m):
        dtw[0, j] = dtw[0, j - 1] + cosine_distance(seq_a[0], seq_b[j])

    # First column — can only come from above
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + cosine_distance(seq_a[i], seq_b[0])

    # Fill remaining cells — each cell takes min of three neighbours
    for i in range(1, n):
        for j in range(1, m):
            cost = cosine_distance(seq_a[i], seq_b[j])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],      # insertion
                dtw[i, j - 1],      # deletion
                dtw[i - 1, j - 1],  # match
            )

    # Normalise by path length (longer sequence)
    normalised = float(dtw[n - 1, m - 1]) / max(n, m)
    return normalised


def dtw_similarity(
    seq_a: list[np.ndarray],
    seq_b: list[np.ndarray],
) -> float:
    """
    Convert DTW distance to similarity in (0, 1].

    similarity = 1 / (1 + dtw_distance)

    Higher = better match.

    Parameters
    ----------
    seq_a : observation sequence
    seq_b : recipe sequence

    Returns
    -------
    float in (0, 1]
    """
    dist = dtw_distance(seq_a, seq_b)
    return 1.0 / (1.0 + dist)


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from observation_pipeline.embedder import Embedder

    embedder = Embedder()

    # Two carbonara scenes — should be similar to each other
    carbonara_scenes = [
        "cracks eggs into a mixing bowl",
        "grates pecorino into the mixing bowl",
        "dices pancetta on a cutting board",
        "cooks pancetta in a skillet",
    ]

    # Two pesto scenes — should be different from carbonara
    pesto_scenes = [
        "toasts pine nuts in a dry pan",
        "spoons store-bought pesto into a mixing bowl",
        "drains pasta over the sink",
        "folds pasta into the pesto",
    ]

    # Observation — partial carbonara session
    obs_scenes = [
        "cracks eggs into a mixing bowl",
        "dices pancetta on a cutting board",
    ]

    carbonara_vecs = embedder.embed_batch(carbonara_scenes)
    pesto_vecs = embedder.embed_batch(pesto_scenes)
    obs_vecs = embedder.embed_batch(obs_scenes)

    print("=== DTW Distance Test ===\n")
    d_carbonara = dtw_distance(obs_vecs, carbonara_vecs)
    d_pesto = dtw_distance(obs_vecs, pesto_vecs)
    s_carbonara = dtw_similarity(obs_vecs, carbonara_vecs)
    s_pesto = dtw_similarity(obs_vecs, pesto_vecs)

    print(f"obs vs carbonara:  distance={d_carbonara:.4f}  similarity={s_carbonara:.4f}")
    print(f"obs vs pesto:      distance={d_pesto:.4f}  similarity={s_pesto:.4f}")
    print(f"\nCorrect ranking: {'✓' if s_carbonara > s_pesto else '✗'} "
          f"(carbonara should be higher)")