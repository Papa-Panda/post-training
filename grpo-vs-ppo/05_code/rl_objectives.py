"""Dependency-free reference math for PPO/GRPO notes.

This is intentionally small: it validates semantics, not distributed training.
All functions return Python lists/floats so the tests run with the standard library.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def _floats(xs: Iterable[float]) -> List[float]:
    return [float(x) for x in xs]


def token_ratios(new_logp: Sequence[float], behavior_logp: Sequence[float]) -> List[float]:
    """rho_t = exp(log pi_theta - log pi_behavior)."""
    if len(new_logp) != len(behavior_logp):
        raise ValueError("log-probability vectors must have equal length")
    return [math.exp(float(n) - float(b)) for n, b in zip(new_logp, behavior_logp)]


def sequence_ratio_from_token_ratios(ratios: Sequence[float]) -> float:
    """Exact trajectory ratio, computed in log space for numerical hygiene."""
    if not ratios:
        raise ValueError("a sequence must contain at least one action token")
    if any(r <= 0 for r in ratios):
        raise ValueError("importance ratios must be positive")
    return math.exp(sum(math.log(float(r)) for r in ratios))


def clipped_surrogate(
    ratios: Sequence[float],
    advantages: Sequence[float],
    epsilon_low: float = 0.2,
    epsilon_high: float | None = None,
) -> List[float]:
    """Elementwise PPO-style pessimistic surrogate to maximize."""
    if len(ratios) != len(advantages):
        raise ValueError("ratios and advantages must have equal length")
    if epsilon_low < 0:
        raise ValueError("epsilon_low must be non-negative")
    epsilon_high = epsilon_low if epsilon_high is None else epsilon_high
    if epsilon_high < 0:
        raise ValueError("epsilon_high must be non-negative")
    low, high = 1.0 - epsilon_low, 1.0 + epsilon_high
    out = []
    for ratio, advantage in zip(ratios, advantages):
        ratio, advantage = float(ratio), float(advantage)
        clipped = min(max(ratio, low), high)
        out.append(min(ratio * advantage, clipped * advantage))
    return out


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    terminated: Sequence[bool],
    gamma: float = 1.0,
    lam: float = 0.95,
) -> List[float]:
    """Finite-horizon GAE.

    values has length T+1. values[-1] is used to bootstrap a truncated final
    transition; it is ignored when terminated[-1] is True.
    """
    rewards = _floats(rewards)
    values = _floats(values)
    if len(values) != len(rewards) + 1 or len(terminated) != len(rewards):
        raise ValueError("need T rewards, T done flags, and T+1 values")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("gamma and lambda must lie in [0, 1]")
    advantages = [0.0] * len(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if terminated[t] else 1.0
        delta = rewards[t] + gamma * continuation * values[t + 1] - values[t]
        running = delta + gamma * lam * continuation * running
        advantages[t] = running
    return advantages


def group_advantages(
    rewards: Sequence[float], eps: float = 1e-8, scale_by_std: bool = True
) -> List[float]:
    """Centered GRPO advantages using population standard deviation.

    Equal-reward groups return zeros. G=1 is rejected because no relative
    comparison exists.
    """
    rewards = _floats(rewards)
    if len(rewards) < 2:
        raise ValueError("group-relative advantage requires G >= 2")
    mean = sum(rewards) / len(rewards)
    centered = [r - mean for r in rewards]
    if not scale_by_std:
        return centered
    variance = sum(x * x for x in centered) / len(centered)
    std = math.sqrt(variance)
    if std <= eps:
        return [0.0] * len(rewards)
    return [x / std for x in centered]


def leave_one_out_advantages(rewards: Sequence[float]) -> List[float]:
    """R_i minus the mean reward of the other G-1 samples."""
    rewards = _floats(rewards)
    if len(rewards) < 2:
        raise ValueError("leave-one-out advantage requires G >= 2")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]


def kl_k1(logp: float, logq: float) -> float:
    """log(p/q): unbiased KL(p||q) value estimate for a sample from p."""
    return float(logp) - float(logq)


def kl_k2(logp: float, logq: float) -> float:
    """Half squared log-ratio: positive local approximation, generally biased."""
    z = kl_k1(logp, logq)
    return 0.5 * z * z


def kl_k3(logp: float, logq: float) -> float:
    """exp(-z)-1+z: non-negative and unbiased in value for a sample from p."""
    z = kl_k1(logp, logq)
    return math.exp(-z) - 1.0 + z


def aggregate_masked(
    per_response_values: Sequence[Sequence[float]],
    masks: Sequence[Sequence[int]],
    mode: str,
) -> float:
    """Aggregate token values as response_mean or token_mean."""
    if len(per_response_values) != len(masks) or not per_response_values:
        raise ValueError("values and masks need the same non-zero group size")
    sums, counts = [], []
    for values, mask in zip(per_response_values, masks):
        if len(values) != len(mask):
            raise ValueError("each response and mask must have equal length")
        selected = [float(v) for v, keep in zip(values, mask) if keep]
        if not selected:
            raise ValueError("each response needs at least one active token")
        sums.append(sum(selected))
        counts.append(len(selected))
    if mode == "response_mean":
        return sum(s / n for s, n in zip(sums, counts)) / len(sums)
    if mode == "token_mean":
        return sum(sums) / sum(counts)
    raise ValueError("mode must be 'response_mean' or 'token_mean'")
