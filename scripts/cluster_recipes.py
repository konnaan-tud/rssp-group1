"""
scripts/cluster_recipes.py
--------------------------
Compute pairwise recipe similarity from the embedded scene sequences and
produce clustering / similarity-curve artefacts that motivate the
dataset choice ("why pasta and not salads") for the supervisor.

Outputs written to outputs/:
  - recipe_similarity_matrix.csv           — 29 × 29 normalised matrix
  - recipe_clusters.json                   — cluster assignments + sizes
  - recipe_clusters.md                     — readable cluster report
  - figure_recipe_similarity_heatmap.png   — heatmap with cluster blocks
  - figure_recipe_similarity_curve.png     — sorted-pairs curve with
                                              salads baseline overlay

Run:
    python scripts/cluster_recipes.py
    python main.py --mode cluster

Improvements over the V1 implementation
---------------------------------------
1. Similarity is **normalised to [0, 1]** for the external artefacts —
   the V1 hybrid metric ranges over [0, ~1.3] because of the order
   bonus, which produced cluster averages > 1.0 in JSON and saturated
   the heatmap colourmap. The belief updater is unchanged.
2. The matrix diagonal is computed with the same function as the
   off-diagonal cells (was hardcoded to 1.0).
3. **scipy.cluster.hierarchy** is used for the linkage and leaf
   ordering instead of a hand-rolled agglomerative loop.
4. **silhouette_score** picks the cluster count automatically over
   k ∈ [2, 8]; the chosen k is reported. CLI override is still available.
5. The heatmap annotates cluster boundaries with rectangles.
6. The similarity-curve figure **overlays a salads-baseline curve**
   built from data/salads_scenes.json. This is the comparison the
   supervisor asked for ("why pasta and not salads"): pasta has a
   systematically higher distribution of pairwise similarities,
   making it the harder and more interesting dataset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from observation_pipeline.embedder import Embedder
from probability.similarity import hybrid_similarity


ROOT = Path(__file__).resolve().parents[1]
PASTA_SEQUENCES_PATH = ROOT / "data" / "recipe_sequences.json"
SALADS_SCENES_PATH = ROOT / "data" / "salads_scenes.json"
OUTPUTS_DIR = ROOT / "outputs"

# Hybrid similarity returns values in [0, 1 + alpha]. To present external
# figures and tables in a sensible [0, 1] range we divide by the maximum
# possible value the metric can produce.
_HYBRID_ALPHA = 0.3   # mirrors probability/similarity.py default
HYBRID_MAX = 1.0 + _HYBRID_ALPHA


# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_pasta_sequences(
    path: Path = PASTA_SEQUENCES_PATH,
) -> dict[str, list[np.ndarray]]:
    """Load pre-embedded pasta-recipe scene vectors from data/."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        name: [np.array(vec, dtype=np.float32) for vec in seq]
        for name, seq in raw.items()
    }


def load_salads_sequences(
    path: Path = SALADS_SCENES_PATH,
) -> dict[str, list[np.ndarray]]:
    """
    Build embeddings for the salads baseline on the fly. The salads
    dataset is small and only used here, so it doesn't justify a
    `data/recipe_sequences.json`-style pre-embedded artefact.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    embedder = Embedder()
    sequences: dict[str, list[np.ndarray]] = {}
    for entry in raw:
        name = entry["dish"]
        vecs = embedder.embed_batch(entry["scenes"])
        sequences[name] = [np.asarray(v, dtype=np.float32) for v in vecs]
    return sequences


# ═══════════════════════════════════════════════════════════════════════════
# Similarity
# ═══════════════════════════════════════════════════════════════════════════

def _symmetric_normalised_similarity(
    recipe_a: list[np.ndarray],
    recipe_b: list[np.ndarray],
) -> float:
    """
    Symmetric, [0, 1]-normalised version of the hybrid similarity used by
    the belief updater.

    `hybrid_similarity` is directional (treats argument 1 like an
    observation sequence). We average both directions for a clean
    recipe-to-recipe similarity, then divide by HYBRID_MAX so the result
    sits in [0, 1].
    """
    raw = 0.5 * (
        hybrid_similarity(recipe_a, recipe_b)
        + hybrid_similarity(recipe_b, recipe_a)
    )
    return float(raw / HYBRID_MAX)


def compute_similarity_matrix(
    recipe_sequences: dict[str, list[np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    """
    Symmetric n × n similarity matrix in [0, 1]. The diagonal is computed
    with the same function as the off-diagonal cells — no hardcoded 1.0 —
    so the heatmap colour scale matches what the belief updater
    actually sees.
    """
    names = sorted(recipe_sequences)
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i, left in enumerate(names):
        # Diagonal: same metric, same direction-average. Should come out
        # at the global max (1.0) for any non-trivial sequence.
        matrix[i, i] = _symmetric_normalised_similarity(
            recipe_sequences[left], recipe_sequences[left]
        )
        for j in range(i + 1, n):
            right = names[j]
            sim = _symmetric_normalised_similarity(
                recipe_sequences[left],
                recipe_sequences[right],
            )
            matrix[i, j] = sim
            matrix[j, i] = sim

    return names, matrix


# ═══════════════════════════════════════════════════════════════════════════
# Clustering — scipy linkage with silhouette-picked k
# ═══════════════════════════════════════════════════════════════════════════

def _similarity_to_distance(matrix: np.ndarray) -> np.ndarray:
    """Convert a similarity matrix in [0, 1] to a distance matrix."""
    distance = 1.0 - matrix
    # numerical safety: enforce zero diagonal and non-negative entries
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    # enforce exact symmetry (scipy is fussy about float drift)
    distance = 0.5 * (distance + distance.T)
    return distance


def _linkage(matrix: np.ndarray) -> np.ndarray:
    """
    Average-link linkage matrix. Average link mirrors the V1
    intent ("recipe-to-recipe average similarity") without the
    hand-rolled O(n^3) loop.
    """
    distance = _similarity_to_distance(matrix)
    condensed = squareform(distance, checks=False)
    return hierarchy.linkage(condensed, method="average")


def leaf_order(matrix: np.ndarray) -> list[int]:
    """Dendrogram leaf order for the heatmap — keeps similar recipes adjacent."""
    Z = _linkage(matrix)
    return list(hierarchy.leaves_list(Z))


def pick_cluster_count_by_silhouette(
    matrix: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, float, dict[int, float]]:
    """
    Pick k in [k_min, k_max] that maximises the silhouette score over
    the distance matrix. Returns (best_k, best_score, all_scores).
    """
    Z = _linkage(matrix)
    distance = _similarity_to_distance(matrix)

    scores: dict[int, float] = {}
    best_k, best_score = k_min, -1.0
    for k in range(k_min, min(k_max, matrix.shape[0] - 1) + 1):
        labels = hierarchy.fcluster(Z, t=k, criterion="maxclust")
        if len(set(labels)) < 2:
            continue
        score = float(
            silhouette_score(distance, labels, metric="precomputed")
        )
        scores[k] = round(score, 4)
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score, scores


def assign_clusters(
    matrix: np.ndarray,
    n_clusters: int,
) -> list[list[int]]:
    """
    Cut the linkage tree to `n_clusters` clusters, returning lists of
    member indices ordered by cluster size (large clusters first).
    """
    Z = _linkage(matrix)
    labels = hierarchy.fcluster(Z, t=n_clusters, criterion="maxclust")

    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(idx)

    return sorted(by_label.values(), key=lambda members: (-len(members), members[0]))


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

def write_similarity_csv(
    names: list[str],
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["recipe", *names])
        for name, row in zip(names, matrix):
            writer.writerow([name, *[f"{float(v):.4f}" for v in row]])


def _cluster_internal_similarity(
    matrix: np.ndarray,
    members: list[int],
) -> float:
    if len(members) <= 1:
        return 1.0
    vals = [
        float(matrix[i, j])
        for pos, i in enumerate(members)
        for j in members[pos + 1 :]
    ]
    return float(np.mean(vals)) if vals else 1.0


def write_cluster_report(
    names: list[str],
    matrix: np.ndarray,
    clusters: list[list[int]],
    silhouette_info: dict,
    output_json: Path,
    output_md: Path,
) -> None:
    payload: dict = {
        "silhouette": silhouette_info,
        "clusters": [],
    }
    lines = [
        "# Recipe Clusters",
        "",
        "Similarity is the symmetric hybrid metric used by the belief "
        "updater (F1 + order bonus), normalised to the **[0, 1]** range "
        "by dividing by (1 + alpha). Cluster count was chosen "
        f"automatically by silhouette score (best k = {silhouette_info['best_k']}, "
        f"silhouette = {silhouette_info['best_score']:.3f}).",
        "",
        "## Per-k silhouette scores",
        "",
    ]
    for k, score in sorted(silhouette_info["scores"].items()):
        marker = " ←" if k == silhouette_info["best_k"] else ""
        lines.append(f"- k = {k}: {score:.3f}{marker}")
    lines.append("")

    for idx, members in enumerate(clusters, start=1):
        member_names = [names[i] for i in members]
        avg_sim = _cluster_internal_similarity(matrix, members)
        payload["clusters"].append({
            "cluster_id": idx,
            "size": len(member_names),
            "average_internal_similarity": round(avg_sim, 4),
            "recipes": member_names,
        })
        lines.append(
            f"## Cluster {idx} — {len(member_names)} recipes "
            f"(avg internal similarity {avg_sim:.3f})"
        )
        lines.append("")
        for recipe in member_names:
            lines.append(f"- {recipe}")
        lines.append("")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_similarity_heatmap(
    names: list[str],
    matrix: np.ndarray,
    order: list[int],
    clusters: list[list[int]],
    output_path: Path,
) -> None:
    """
    Heatmap with the leaves ordered by hierarchical-clustering leaf order
    and **cluster blocks outlined with rectangles** so the reader can see
    cluster structure at a glance.
    """
    ordered_names = [names[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]

    # Compute block start/end in the new ordering. Iterate over clusters
    # and for each find the contiguous range of `order` positions that
    # fall inside it (clusters are guaranteed contiguous in leaf order).
    pos_of = {idx: i for i, idx in enumerate(order)}
    blocks: list[tuple[int, int]] = []
    for cluster in clusters:
        positions = sorted(pos_of[idx] for idx in cluster)
        if positions:
            blocks.append((positions[0], positions[-1]))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(ordered_matrix, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(
        "Recipe similarity heatmap (normalised hybrid similarity)\n"
        "Bright = similar. Cluster boundaries outlined in white."
    )
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered_names, fontsize=8)

    for (start, end) in blocks:
        rect = Rectangle(
            (start - 0.5, start - 0.5),
            end - start + 1,
            end - start + 1,
            linewidth=1.8,
            edgecolor="white",
            facecolor="none",
        )
        ax.add_patch(rect)

    fig.colorbar(im, ax=ax, label="Similarity (0 = unrelated, 1 = identical)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _pairwise_similarities(matrix: np.ndarray) -> list[float]:
    """Off-diagonal upper-triangle pairwise similarity values."""
    pairs: list[float] = []
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(matrix[i, j]))
    return pairs


def plot_similarity_curve(
    pasta_names: list[str],
    pasta_matrix: np.ndarray,
    salads_names: list[str],
    salads_matrix: np.ndarray | None,
    output_path: Path,
) -> None:
    """
    Sorted-pair similarity curve with a salads baseline overlay.

    Each curve is a sorted plot of all C(n, 2) pairwise similarities,
    most-similar at the left. The vertical axis is shared. Pasta sitting
    visibly above salads → pasta is a harder dataset than salads for
    the same similarity threshold — directly answers the supervisor's
    "why pasta and not salads" question.
    """
    pasta_vals = sorted(_pairwise_similarities(pasta_matrix), reverse=True)
    pasta_x = np.linspace(0.0, 1.0, len(pasta_vals)) if pasta_vals else np.array([])

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.plot(
        pasta_x,
        pasta_vals,
        linewidth=2.4,
        color="#b33c2f",
        label=f"Italian pasta ({len(pasta_names)} recipes, {len(pasta_vals)} pairs)",
    )

    if salads_matrix is not None and salads_names:
        salads_vals = sorted(
            _pairwise_similarities(salads_matrix), reverse=True
        )
        if salads_vals:
            salads_x = np.linspace(0.0, 1.0, len(salads_vals))
            ax.plot(
                salads_x,
                salads_vals,
                linewidth=2.4,
                color="#3a6b8c",
                label=f"Salads baseline ({len(salads_names)} recipes, {len(salads_vals)} pairs)",
                linestyle="--",
            )
            # Mean horizontal references — easy to point at in slides.
            ax.axhline(
                float(np.mean(pasta_vals)),
                color="#b33c2f",
                linestyle=":",
                alpha=0.45,
                label=f"Pasta mean = {np.mean(pasta_vals):.3f}",
            )
            ax.axhline(
                float(np.mean(salads_vals)),
                color="#3a6b8c",
                linestyle=":",
                alpha=0.45,
                label=f"Salads mean = {np.mean(salads_vals):.3f}",
            )

    ax.set_title(
        "Sorted pairwise recipe similarity — pasta vs. salads baseline"
    )
    ax.set_xlabel(
        "Pair rank (normalised: 0 = most similar pair, 1 = least similar pair)"
    )
    ax.set_ylabel("Normalised hybrid similarity (0–1)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_recipe_clustering(
    n_clusters: int | None = None,
) -> dict[str, object]:
    """
    Run the full clustering pipeline.

    Parameters
    ----------
    n_clusters
        If `None` (default), the cluster count is chosen automatically by
        silhouette score. Pass an int to force a specific k.
    """
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Pasta — the dataset under study
    pasta_seq = load_pasta_sequences()
    pasta_names, pasta_matrix = compute_similarity_matrix(pasta_seq)

    best_k, best_score, scores = pick_cluster_count_by_silhouette(pasta_matrix)
    chosen_k = n_clusters if n_clusters is not None else best_k

    order = leaf_order(pasta_matrix)
    clusters = assign_clusters(pasta_matrix, n_clusters=chosen_k)

    silhouette_info = {
        "best_k": best_k,
        "best_score": round(best_score, 4),
        "chosen_k": chosen_k,
        "scores": scores,
    }

    # Salads — baseline for the similarity curve
    salads_seq = load_salads_sequences()
    if salads_seq:
        salads_names, salads_matrix = compute_similarity_matrix(salads_seq)
    else:
        salads_names, salads_matrix = [], None

    # Files
    matrix_csv = OUTPUTS_DIR / "recipe_similarity_matrix.csv"
    clusters_json = OUTPUTS_DIR / "recipe_clusters.json"
    clusters_md = OUTPUTS_DIR / "recipe_clusters.md"
    heatmap_png = OUTPUTS_DIR / "figure_recipe_similarity_heatmap.png"
    curve_png = OUTPUTS_DIR / "figure_recipe_similarity_curve.png"

    write_similarity_csv(pasta_names, pasta_matrix, matrix_csv)
    write_cluster_report(
        pasta_names, pasta_matrix, clusters, silhouette_info,
        clusters_json, clusters_md,
    )
    plot_similarity_heatmap(pasta_names, pasta_matrix, order, clusters, heatmap_png)
    plot_similarity_curve(
        pasta_names, pasta_matrix,
        salads_names, salads_matrix,
        curve_png,
    )

    return {
        "recipe_count": len(pasta_names),
        "cluster_count": len(clusters),
        "silhouette": silhouette_info,
        "salads_baseline": bool(salads_seq),
        "matrix_csv": str(matrix_csv),
        "clusters_json": str(clusters_json),
        "clusters_md": str(clusters_md),
        "heatmap_png": str(heatmap_png),
        "curve_png": str(curve_png),
    }


def main() -> None:
    result = run_recipe_clustering()
    print(
        f"Clustered {result['recipe_count']} recipes into "
        f"{result['cluster_count']} groups."
    )
    s = result["silhouette"]
    print(
        f"Silhouette: best k = {s['best_k']} (score {s['best_score']:.3f}), "
        f"chosen k = {s['chosen_k']}"
    )
    print(f"Salads baseline included: {result['salads_baseline']}")
    print(f"Similarity matrix → {result['matrix_csv']}")
    print(f"Cluster report    → {result['clusters_md']}")
    print(f"Heatmap           → {result['heatmap_png']}")
    print(f"Similarity curve  → {result['curve_png']}")


if __name__ == "__main__":
    main()
