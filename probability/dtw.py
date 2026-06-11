from __future__ import annotations


def l1_distance(vec_a: list[float], vec_b: list[float]) -> float:
    size = min(len(vec_a), len(vec_b))
    total = sum(abs(vec_a[i] - vec_b[i]) for i in range(size))
    total += sum(abs(v) for v in vec_a[size:])
    total += sum(abs(v) for v in vec_b[size:])
    return total


def dtw_distance(sequence_a: list[list[float]], sequence_b: list[list[float]]) -> float:
    if not sequence_a or not sequence_b:
        return float(max(len(sequence_a), len(sequence_b)))

    rows = len(sequence_a) + 1
    cols = len(sequence_b) + 1
    dp = [[float("inf")] * cols for _ in range(rows)]
    dp[0][0] = 0.0

    for i in range(1, rows):
        for j in range(1, cols):
            cost = l1_distance(sequence_a[i - 1], sequence_b[j - 1])
            dp[i][j] = cost + min(
                dp[i - 1][j],
                dp[i][j - 1],
                dp[i - 1][j - 1],
            )

    return dp[-1][-1]
