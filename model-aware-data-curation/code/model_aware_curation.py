"""Minimal gradient-geometry primitives for model-aware data curation.

This is an educational NumPy implementation. It consumes already-computed
per-sample gradient sketches; it does not extract gradients from an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


def _as_2d(x: Array | Sequence[Sequence[float]]) -> Array:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape={a.shape}")
    return a


def row_normalize(x: Array, eps: float = 1e-12) -> Array:
    """L2-normalize rows, leaving an all-zero row as all zeros."""
    a = _as_2d(x)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, eps)


def cosine_scores(gradients: Array, reference: Array, eps: float = 1e-12) -> Array:
    """Cosine similarity of every row in ``gradients`` to one reference."""
    g = row_normalize(gradients, eps=eps)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if g.shape[1] != r.shape[0]:
        raise ValueError("gradient and reference dimensions do not match")
    r = r / max(float(np.linalg.norm(r)), eps)
    return g @ r


def rademacher_project(
    gradients: Array, projection_dim: int, seed: int = 0
) -> Array:
    """Project gradients with a seeded Rademacher JL transform.

    The returned vector dimension is ``projection_dim``. The 1/sqrt(d) scale
    approximately preserves pairwise inner products in expectation.
    """
    g = _as_2d(gradients)
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    rng = np.random.default_rng(seed)
    r = rng.choice((-1.0, 1.0), size=(g.shape[1], projection_dim))
    return (g @ r) / np.sqrt(projection_dim)


def gradient_vendi(gradients: Array, eps: float = 1e-12) -> float:
    """Exponentiated spectral entropy of normalized gradient directions.

    For n >> d, using G.T @ G / n avoids materializing the n x n Gram matrix;
    its non-zero eigenvalues equal those of G @ G.T / n.
    """
    g = _as_2d(gradients)
    if g.shape[0] == 0:
        return 0.0
    g = row_normalize(g, eps=eps)
    covariance = (g.T @ g) / g.shape[0]
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= eps:
        return 0.0
    p = eigenvalues / total
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


def conflict_scores(gradients: Array, protection_gradient: Array) -> Array:
    """Positive risk for directions opposed to a protected capability."""
    return np.maximum(0.0, -cosine_scores(gradients, protection_gradient))


def sparse_cluster_ids(
    cluster_ids: Sequence[int], sparse_fraction: float = 0.2
) -> set[int]:
    """Return IDs of the least-populated cluster fraction.

    At least one cluster is returned. Ties are deterministic by cluster id.
    """
    if not 0.0 < sparse_fraction <= 1.0:
        raise ValueError("sparse_fraction must be in (0, 1]")
    ids = np.asarray(cluster_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("cluster_ids must be a non-empty 1-D sequence")
    unique, counts = np.unique(ids, return_counts=True)
    k = max(1, int(np.ceil(len(unique) * sparse_fraction)))
    order = np.lexsort((unique, counts))
    return set(int(x) for x in unique[order[:k]])


@dataclass(frozen=True)
class SelectionResult:
    indices: list[int]
    target_scores: Array
    conflict_scores: Array
    vendi: float


def greedy_select(
    gradients: Array,
    target_gradient: Array,
    protection_gradient: Array,
    budget: int,
    quality_mask: Iterable[bool] | None = None,
    max_conflict: float = 0.0,
    target_weight: float = 1.0,
    coverage_weight: float = 1.0,
) -> SelectionResult:
    """Greedy target + marginal log-G-Vendi selection under a conflict gate."""
    g = _as_2d(gradients)
    n = g.shape[0]
    if budget < 0:
        raise ValueError("budget must be non-negative")
    quality = (
        np.ones(n, dtype=bool)
        if quality_mask is None
        else np.asarray(list(quality_mask), dtype=bool)
    )
    if quality.shape != (n,):
        raise ValueError("quality_mask must have one value per sample")

    target = cosine_scores(g, target_gradient)
    conflict = conflict_scores(g, protection_gradient)
    eligible = [i for i in range(n) if quality[i] and conflict[i] <= max_conflict]
    selected: list[int] = []

    while eligible and len(selected) < budget:
        base_vendi = gradient_vendi(g[selected]) if selected else 1.0
        best_index = None
        best_score = -np.inf
        for i in eligible:
            proposed = selected + [i]
            new_vendi = max(gradient_vendi(g[proposed]), 1.0)
            coverage_gain = np.log(new_vendi) - np.log(max(base_vendi, 1.0))
            score = target_weight * target[i] + coverage_weight * coverage_gain
            if score > best_score:
                best_score = float(score)
                best_index = i
        assert best_index is not None
        selected.append(best_index)
        eligible.remove(best_index)

    final_vendi = gradient_vendi(g[selected]) if selected else 0.0
    return SelectionResult(selected, target, conflict, final_vendi)
