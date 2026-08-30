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


def fisher_logdet(gradients: Array, alpha: float = 1.0) -> float:
    """Log determinant of I + alpha times the empirical Fisher matrix."""
    g = _as_2d(gradients)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    information = np.eye(g.shape[1]) + alpha * (g.T @ g)
    sign, value = np.linalg.slogdet(information)
    if sign <= 0.0:
        raise ValueError("information matrix must be positive definite")
    return float(value)


def fisher_marginal_gain(
    selected_gradients: Array, candidate_gradient: Array, alpha: float = 1.0
) -> float:
    """Marginal Fisher/log-det gain from adding one candidate gradient."""
    selected = _as_2d(selected_gradients)
    candidate = np.asarray(candidate_gradient, dtype=np.float64).reshape(-1)
    if candidate.shape[0] != selected.shape[1]:
        raise ValueError("selected and candidate gradient dimensions do not match")
    before = fisher_logdet(selected, alpha=alpha)
    after = fisher_logdet(np.vstack([selected, candidate]), alpha=alpha)
    return after - before


def selected_set_conflict(
    candidate_gradient: Array, selected_gradients: Array, eta: float = 1e-12
) -> float:
    """SPICE's negative-cosine penalty against the selected-set mean."""
    selected = _as_2d(selected_gradients)
    candidate = np.asarray(candidate_gradient, dtype=np.float64).reshape(-1)
    if candidate.shape[0] != selected.shape[1]:
        raise ValueError("selected and candidate gradient dimensions do not match")
    if selected.shape[0] == 0:
        return 0.0
    mean = selected.mean(axis=0)
    denominator = float(np.linalg.norm(candidate) * np.linalg.norm(mean) + eta)
    cosine = float(candidate @ mean) / denominator
    return max(0.0, -cosine)


def spice_score(
    selected_gradients: Array,
    candidate_gradient: Array,
    alpha: float = 1.0,
    conflict_weight: float = 0.1,
    eta: float = 1e-12,
) -> float:
    """Fisher marginal gain minus SPICE's selected-set conflict penalty."""
    if conflict_weight < 0.0:
        raise ValueError("conflict_weight must be non-negative")
    gain = fisher_marginal_gain(selected_gradients, candidate_gradient, alpha)
    conflict = selected_set_conflict(candidate_gradient, selected_gradients, eta)
    return gain - conflict_weight * conflict


def gradient_isolation_scores(gradients: Array, eps: float = 1e-12) -> Array:
    """One minus maximum absolute cosine to any other sample.

    This is a novelty diagnostic, not a usefulness or learnability score.
    """
    g = row_normalize(gradients, eps=eps)
    n = g.shape[0]
    if n < 2:
        return np.ones(n, dtype=np.float64)
    similarities = np.clip(np.abs(g @ g.T), 0.0, 1.0)
    np.fill_diagonal(similarities, -np.inf)
    return 1.0 - np.max(similarities, axis=1)


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


@dataclass(frozen=True)
class SpiceSelectionResult:
    indices: list[int]
    information_gains: list[float]
    conflicts: list[float]
    scores: list[float]
    logdet: float


def spice_greedy_select(
    gradients: Array,
    budget: int,
    alpha: float = 1.0,
    conflict_weight: float = 0.1,
    early_stop_ratio: float | None = None,
    eta: float = 1e-12,
) -> SpiceSelectionResult:
    """Greedily maximize Fisher gain minus selected-set conflict.

    If ``early_stop_ratio`` is set, selection stops before adding a later
    candidate whose Fisher gain is no larger than that fraction of the first
    selected candidate's gain. This mirrors the SPICE+ stopping criterion.
    """
    g = _as_2d(gradients)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if conflict_weight < 0.0:
        raise ValueError("conflict_weight must be non-negative")
    if early_stop_ratio is not None and not 0.0 <= early_stop_ratio <= 1.0:
        raise ValueError("early_stop_ratio must be in [0, 1]")

    selected: list[int] = []
    remaining = list(range(g.shape[0]))
    information_gains: list[float] = []
    conflicts: list[float] = []
    scores: list[float] = []
    first_gain: float | None = None

    while remaining and len(selected) < budget:
        selected_gradients = g[selected]
        best: tuple[float, int, float, float] | None = None
        for index in remaining:
            gain = fisher_marginal_gain(selected_gradients, g[index], alpha)
            conflict = selected_set_conflict(g[index], selected_gradients, eta)
            score = gain - conflict_weight * conflict
            candidate = (float(score), -index, float(gain), float(conflict))
            if best is None or candidate > best:
                best = candidate

        assert best is not None
        score, negative_index, gain, conflict = best
        index = -negative_index
        if (
            first_gain is not None
            and early_stop_ratio is not None
            and gain <= early_stop_ratio * first_gain
        ):
            break

        selected.append(index)
        remaining.remove(index)
        information_gains.append(gain)
        conflicts.append(conflict)
        scores.append(score)
        if first_gain is None:
            first_gain = gain

    final_logdet = fisher_logdet(g[selected], alpha) if selected else 0.0
    return SpiceSelectionResult(
        selected, information_gains, conflicts, scores, final_logdet
    )


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
