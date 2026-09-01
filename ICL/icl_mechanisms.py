"""Small, dependency-free models of three in-context-learning mechanisms.

These routines are executable specifications, not implementations of a language
model.  They make the assumptions in the accompanying notes testable:

* finite latent-concept Bayesian updating;
* one full-batch linear-regression gradient step and its linear-attention form;
* exact-match induction over a token sequence;
* simple KV-cache and attention-score memory accounting.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def _check_rectangular(matrix: Matrix, name: str) -> tuple[int, int]:
    rows = len(matrix)
    if rows == 0:
        raise ValueError(f"{name} must be non-empty")
    cols = len(matrix[0])
    if cols == 0 or any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return rows, cols


def posterior(prior: Vector, likelihood_rows: Iterable[Vector]) -> list[float]:
    """Return p(c | observations) for a finite concept set.

    ``likelihood_rows[t][c]`` is p(observation_t | concept_c).  Computation is
    in log space, so long evidence sequences do not underflow prematurely.
    """
    if not prior or any(not math.isfinite(p) or p <= 0.0 for p in prior):
        raise ValueError("prior entries must be finite and positive")
    total = math.fsum(prior)
    log_weights = [math.log(p / total) for p in prior]
    for row in likelihood_rows:
        if len(row) != len(prior) or any(
            not math.isfinite(value) or value < 0.0 for value in row
        ):
            raise ValueError(
                "each likelihood row must be finite, non-negative, and match prior"
            )
        log_weights = [
            w + (math.log(value) if value > 0.0 else -math.inf)
            for w, value in zip(log_weights, row)
        ]
        if all(weight == -math.inf for weight in log_weights):
            raise ValueError("evidence has zero likelihood under every concept")
    offset = max(log_weights)
    weights = [math.exp(w - offset) for w in log_weights]
    z = math.fsum(weights)
    return [w / z for w in weights]


def posterior_path(prior: Vector, likelihood_rows: Iterable[Vector]) -> list[list[float]]:
    """Return the posterior before evidence and after every observation."""
    rows = [list(row) for row in likelihood_rows]
    path = [posterior(prior, [])]
    for end in range(1, len(rows) + 1):
        path.append(posterior(prior, rows[:end]))
    return path


def _matvec(matrix: Matrix, vector: Vector) -> list[float]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix/vector shape mismatch")
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def linear_gd_update(w0: Matrix, x: Matrix, y: Matrix, eta: float) -> list[list[float]]:
    """One full-batch GD step for L(W)=||X W^T - Y||_F^2/(2k)."""
    k, d = _check_rectangular(x, "x")
    ky, m = _check_rectangular(y, "y")
    mw, dw = _check_rectangular(w0, "w0")
    if ky != k or mw != m or dw != d:
        raise ValueError("expected x:kxd, y:kxm, w0:mxd")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")

    updated = [list(row) for row in w0]
    for i in range(k):
        prediction = _matvec(w0, x[i])
        residual = [y[i][out] - prediction[out] for out in range(m)]
        for out in range(m):
            for feature in range(d):
                updated[out][feature] += eta * residual[out] * x[i][feature] / k
    return updated


def linear_attention_prediction(
    w0: Matrix, x: Matrix, y: Matrix, x_query: Vector, eta: float
) -> list[float]:
    """Compute W0*xq plus the unnormalized linear-attention correction.

    Keys are x_i, queries are x_query, and values are residuals
    y_i - W0*x_i.  The 1/k factor matches the mean squared loss.
    """
    k, d = _check_rectangular(x, "x")
    ky, m = _check_rectangular(y, "y")
    mw, dw = _check_rectangular(w0, "w0")
    if ky != k or mw != m or dw != d or len(x_query) != d:
        raise ValueError("expected x:kxd, y:kxm, w0:mxd, x_query:d")
    if eta < 0.0:
        raise ValueError("eta must be non-negative")

    output = _matvec(w0, x_query)
    for xi, yi in zip(x, y):
        residual = [target - pred for target, pred in zip(yi, _matvec(w0, xi))]
        similarity = sum(a * b for a, b in zip(xi, x_query))
        for out in range(m):
            output[out] += eta * residual[out] * similarity / k
    return output


def induction_distribution(tokens: Sequence[str], query: str) -> Mapping[str, float]:
    """Empirical next-token distribution after earlier exact matches of query."""
    successors = [tokens[i + 1] for i in range(len(tokens) - 1) if tokens[i] == query]
    if not successors:
        return {}
    counts = Counter(successors)
    total = len(successors)
    return {token: count / total for token, count in sorted(counts.items())}


def kv_cache_bytes(
    layers: int, tokens: int, kv_heads: int, head_dim: int, bytes_per_element: int
) -> int:
    """Bytes for dense K and V storage: 2*layers*tokens*kv_heads*head_dim*b."""
    values = (layers, tokens, kv_heads, head_dim, bytes_per_element)
    if any(value < 0 for value in values):
        raise ValueError("cache dimensions must be non-negative")
    return 2 * layers * tokens * kv_heads * head_dim * bytes_per_element


def attention_score_bytes(
    layers: int, heads: int, tokens: int, bytes_per_element: int
) -> int:
    """Naive materialized prefill score bytes: layers*heads*tokens^2*b."""
    values = (layers, heads, tokens, bytes_per_element)
    if any(value < 0 for value in values):
        raise ValueError("attention dimensions must be non-negative")
    return layers * heads * tokens * tokens * bytes_per_element


def demo() -> None:
    evidence = [[0.8, 0.2], [0.7, 0.3], [0.9, 0.1]]
    print("Bayes posterior path:")
    for step, probs in enumerate(posterior_path([0.5, 0.5], evidence)):
        print(f"  k={step}: {[round(p, 4) for p in probs]}")

    w0 = [[0.2, -0.1]]
    x = [[1.0, 2.0], [-1.0, 1.0], [2.0, 0.5]]
    y = [[1.5], [-0.5], [2.0]]
    xq = [0.5, -1.0]
    eta = 0.3
    gd_prediction = _matvec(linear_gd_update(w0, x, y, eta), xq)
    attention_prediction = linear_attention_prediction(w0, x, y, xq, eta)
    print("GD prediction:", gd_prediction)
    print("linear-attention prediction:", attention_prediction)

    sequence = ["def", "name", "(", ")", "def", "other", "(", ")", "def"]
    print("induction distribution after 'def':", induction_distribution(sequence, "def"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    demo()
