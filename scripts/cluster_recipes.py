"""
scripts/cluster_recipes.py
--------------------------
Compute pairwise recipe similarity from the embedded scene sequences and
produce lightweight clustering artifacts for analysis/presentation.

Outputs written to outputs/:
  - recipe_similarity_matrix.csv
  - recipe_clusters.json
  - recipe_clusters.md
  - figure_recipe_similarity_heatmap.png
  - figure_recipe_similarity_curve.png

Run:
    python scripts/cluster_recipes.py
    python main.py --mode cluster
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from probability.similarity import hybrid_similarity


ROOT = Path(__file__).resolve().parents[1]
SEQUENCES_PATH = ROOT / "data" / "recipe_sequences.json"
OUTPUTS_DIR = ROOT / "outputs"


def load_recipe_sequences(path: Path = SEQUENCES_PATH) -> dict[str, list[np.ndarray]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        name: [np.array(vec, dtype=np.float32) for vec in seq]
        for name, seq in raw.items()
    }


def symmetric_recipe_similarity(
    recipe_a: list[np.ndarray],
    recipe_b: list[np.ndarray],
) -> float:
    """
    Symmetrise the hybrid similarity used by the belief updater.

    `hybrid_similarity(a, b)` is directional because it treats the first
    argument like an observation sequence. For recipe-to-recipe comparison
    we average both directions so the matrix is symmetric.
    """
    return 0.5 * (
        hybrid_similarity(recipe_a, recipe_b)
        + hybrid_similarity(recipe_b, recipe_a)
    )


def compute_similarity_matrix(
    recipe_sequences: dict[str, list[np.ndarray]],
) -> tuple[list[str], np.ndarray]:
    names = sorted(recipe_sequences)
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i, left in enumerate(names):
        matrix[i, i] = 1.0
        for j in range(i + 1, n):
            right = names[j]
            sim = symmetric_recipe_similarity(
                recipe_sequences[left],
                recipe_sequences[right],
            )
            matrix[i, j] = sim
            matrix[j, i] = sim

    return names, matrix


def _cluster_similarity(
    matrix: np.ndarray,
    cluster_a: list[int],
    cluster_b: list[int],
) -> float:
    vals = [matrix[i, j] for i in cluster_a for j in cluster_b]
    return float(np.mean(vals)) if vals else 0.0


def _best_merge_order(
    matrix: np.ndarray,
    left: list[int],
    right: list[int],
) -> list[int]:
    """
    Pick the concatenation that keeps adjacent leaves as similar as possible.
    """
    candidates = [
        left + right,
        left + list(reversed(right)),
        list(reversed(left)) + right,
        list(reversed(left)) + list(reversed(right)),
    ]
    best = candidates[0]
    best_score = -float("inf")

    for order in candidates:
        score = 0.0
        for i in range(len(order) - 1):
            score += float(matrix[order[i], order[i + 1]])
        if score > best_score:
            best = order
            best_score = score
    return best


def hierarchical_recipe_order(matrix: np.ndarray) -> list[int]:
    """
    Small average-link agglomeration to produce a sensible heatmap ordering
    without introducing a scipy dependency.
    """
    clusters = [[i] for i in range(matrix.shape[0])]

    while len(clusters) > 1:
        best_i = 0
        best_j = 1
        best_sim = -float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = _cluster_similarity(matrix, clusters[i], clusters[j])
                if sim > best_sim:
                    best_i, best_j, best_sim = i, j, sim

        left = clusters.pop(best_j)
        right = clusters.pop(best_i)
        clusters.append(_best_merge_order(matrix, right, left))

    return clusters[0]


def assign_clusters(matrix: np.ndarray, n_clusters: int = 6) -> list[list[int]]:
    """
    Average-link clustering cut to a fixed number of recipe families.
    """
    clusters = [[i] for i in range(matrix.shape[0])]
    n_clusters = max(1, min(n_clusters, len(clusters)))

    while len(clusters) > n_clusters:
        best_i = 0
        best_j = 1
        best_sim = -float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = _cluster_similarity(matrix, clusters[i], clusters[j])
                if sim > best_sim:
                    best_i, best_j, best_sim = i, j, sim

        left = clusters.pop(best_j)
        right = clusters.pop(best_i)
        clusters.append(sorted(right + left))

    return sorted(clusters, key=lambda members: (-len(members), members[0]))


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


def write_cluster_report(
    names: list[str],
    matrix: np.ndarray,
    clusters: list[list[int]],
    output_json: Path,
    output_md: Path,
) -> None:
    payload = []
    lines = [
        "# Recipe Clusters",
        "",
        "Similarity computed from the embedded ordered scene sequences using",
        "the same hybrid similarity family as the belief updater.",
        "",
    ]

    for idx, members in enumerate(clusters, start=1):
        member_names = [names[i] for i in members]
        if len(members) > 1:
            sims = [
                float(matrix[i, j])
                for pos, i in enumerate(members)
                for j in members[pos + 1 :]
            ]
            avg_sim = float(np.mean(sims))
        else:
            avg_sim = 1.0

        payload.append(
            {
                "cluster_id": idx,
                "size": len(member_names),
                "average_internal_similarity": round(avg_sim, 4),
                "recipes": member_names,
            }
        )

        lines.append(
            f"## Cluster {idx} ({len(member_names)} recipes, avg sim {avg_sim:.3f})"
        )
        lines.append("")
        for recipe in member_names:
            lines.append(f"- {recipe}")
        lines.append("")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def plot_similarity_heatmap(
    names: list[str],
    matrix: np.ndarray,
    order: list[int],
    output_path: Path,
) -> None:
    ordered_names = [names[i] for i in order]
    ordered_matrix = matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(ordered_matrix, cmap="magma", vmin=0.0, vmax=1.05)
    ax.set_title("Recipe similarity heatmap")
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered_names, fontsize=8)
    fig.colorbar(im, ax=ax, label="Symmetric hybrid similarity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_similarity_curve(
    names: list[str],
    matrix: np.ndarray,
    output_path: Path,
) -> None:
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((float(matrix[i, j]), names[i], names[j]))
    pairs.sort(reverse=True, key=lambda item: item[0])

    values = [item[0] for item in pairs]
    x = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, values, linewidth=2.2, color="#b33c2f")
    ax.set_title("Sorted pairwise recipe similarity")
    ax.set_xlabel("Pair rank (most similar to least similar)")
    ax.set_ylabel("Symmetric hybrid similarity")
    ax.grid(alpha=0.25, linestyle="--")

    if pairs:
        most_sim = pairs[0]
        least_sim = pairs[-1]
        ax.annotate(
            f"Most similar: {most_sim[1]} / {most_sim[2]} ({most_sim[0]:.3f})",
            xy=(1, most_sim[0]),
            xytext=(10, most_sim[0] + 0.03),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "alpha": 0.4},
        )
        ax.annotate(
            f"Least similar: {least_sim[1]} / {least_sim[2]} ({least_sim[0]:.3f})",
            xy=(len(values), least_sim[0]),
            xytext=(max(1, len(values) - 120), least_sim[0] + 0.05),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "alpha": 0.4},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_recipe_clustering(n_clusters: int = 6) -> dict[str, object]:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    recipe_sequences = load_recipe_sequences()
    names, matrix = compute_similarity_matrix(recipe_sequences)
    order = hierarchical_recipe_order(matrix)
    clusters = assign_clusters(matrix, n_clusters=n_clusters)

    matrix_csv = OUTPUTS_DIR / "recipe_similarity_matrix.csv"
    clusters_json = OUTPUTS_DIR / "recipe_clusters.json"
    clusters_md = OUTPUTS_DIR / "recipe_clusters.md"
    heatmap_png = OUTPUTS_DIR / "figure_recipe_similarity_heatmap.png"
    curve_png = OUTPUTS_DIR / "figure_recipe_similarity_curve.png"

    write_similarity_csv(names, matrix, matrix_csv)
    write_cluster_report(names, matrix, clusters, clusters_json, clusters_md)
    plot_similarity_heatmap(names, matrix, order, heatmap_png)
    plot_similarity_curve(names, matrix, curve_png)

    return {
        "recipe_count": len(names),
        "cluster_count": len(clusters),
        "matrix_csv": str(matrix_csv),
        "clusters_json": str(clusters_json),
        "clusters_md": str(clusters_md),
        "heatmap_png": str(heatmap_png),
        "curve_png": str(curve_png),
    }


def main() -> None:
    result = run_recipe_clustering()
    print(f"Clustered {result['recipe_count']} recipes into {result['cluster_count']} groups.")
    print(f"Similarity matrix → {result['matrix_csv']}")
    print(f"Cluster report    → {result['clusters_md']}")
    print(f"Heatmap           → {result['heatmap_png']}")
    print(f"Similarity curve  → {result['curve_png']}")


if __name__ == "__main__":
    main()
