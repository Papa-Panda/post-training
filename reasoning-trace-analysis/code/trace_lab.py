"""Tiny CPU models for reasoning-trace analysis: structure, efficiency, supervision.

This module is a semantics simulator, not a performance predictor. It makes four
ideas from the trace-analysis literature executable on any CPU, with numpy only
(no torch):

(a) deep-thinking ratio toy (Think Deep style):
    layer-wise prediction revision -> settling depth -> deep/shallow token
    classification -> deep-thinking ratio. The toy shows why a ratio of
    "tokens that settled late" can correlate with simulated accuracy while raw
    length does not.

(b) U-shaped entropy + difficulty routing (DiffAdapt style):
    entropy as a function of problem difficulty is U-shaped; a three-tier
    router (Easy / Normal / Hard) spends different token budgets per tier and
    the toy reports token savings vs a uniform budget.

(c) hedging / self-doubt detector on synthetic traces:
    regex scans for hedging markers ("wait", "hold on", ...) and the four
    overthinking patterns (hedging, rechecking, re-derivation, tangents) from
    the batch-prompting literature; a "batched" compressed trace is compared
    against a long single-query trace.

(d) PRM vs ORM toy on step-labeled data (Let's Verify Step by Step style):
    outcome-only scoring vs product-of-step-correctness scoring under
    best-of-N selection, including false positives (right final answer reached
    through a wrong step). A second demo contrasts GRPO-style uniform scalar
    advantage with segment-level (DASH-style) advantage shaping on a drift trace.

All numbers inside are toy-model outputs. The paper numbers they illustrate
are quoted in the docs, not reproduced here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# (a) Deep-thinking ratio toy
# ---------------------------------------------------------------------------
#
# In the paper, for each generation step t and layer l:
#   D_{t,l} = JSD(p_{t,L} || p_{t,l})                      (Jensen-Shannon div.)
#   \bar D_{t,l} = min_{j <= l} D_{t,j}                    (monotone envelope)
#   c_t = min { l : \bar D_{t,l} <= g }                    (settling depth)
# A token is "deep-thinking" when c_t >= ceil(rho * L); the deep-thinking
# ratio is DTR(S) = (1/T) * sum_t 1[c_t in deep regime].
#
# The toy replaces the unembedding projection with an analytic decay curve:
#   D(t, l) = d0 * exp(-l / tau_t),
# where tau_t is the token's "settling time constant". Small tau -> the
# distribution converges in early layers (shallow token); large tau -> it
# keeps being revised in deep layers (deep-thinking token). The classification
# machinery (g, rho, min-envelope) is identical to the paper's.


def jsd_trajectory(tau: float, n_layers: int, d0: float = 1.0) -> np.ndarray:
    """Analytic stand-in for the JSD trajectory D_{t,l}, l = 1..L.

    Larger ``tau`` means the prediction keeps being revised deeper into the
    stack, mirroring tokens whose JSD to the final-layer distribution decays
    slowly.
    """
    layers = np.arange(1, n_layers + 1, dtype=float)
    return d0 * np.exp(-layers / tau)


def settling_depth(traj: np.ndarray, g: float) -> int:
    """First layer where the min-envelope of the JSD trajectory falls <= g."""
    envelope = np.minimum.accumulate(traj)
    hits = np.nonzero(envelope <= g)[0]
    if hits.size == 0:
        return int(len(traj))
    return int(hits[0]) + 1  # 1-based layer index


def deep_thinking_ratio(taus: Sequence[float], n_layers: int,
                        g: float = 0.15, rho: float = 0.8) -> float:
    """Fraction of tokens whose settling depth lies in the deep regime."""
    if len(taus) == 0:
        return 0.0
    cutoff = math.ceil(rho * n_layers)
    deep = sum(1 for tau in taus
               if settling_depth(jsd_trajectory(tau, n_layers), g) >= cutoff)
    return deep / len(taus)


def _sample_taus(rng: np.random.Generator, n_tokens: int,
                 think_prob: float) -> List[float]:
    """Mix of filler tokens (fast settling) and thinking tokens (slow)."""
    is_think = rng.random(n_tokens) < think_prob
    taus = np.where(is_think,
                    rng.uniform(14.0, 24.0, n_tokens),   # late settlers
                    rng.uniform(1.5, 4.0, n_tokens))     # early settlers
    return [float(t) for t in taus]


def run_correlation_experiment(n_layers: int = 36, n_seq: int = 600,
                               seed: int = 7) -> Dict[str, float]:
    """Compare DTR vs raw length as predictors of simulated accuracy.

    The toy data-generating process: the true driver of accuracy is the
    fraction of deep-thinking tokens (genuine internal revision); length is
    noisy and partially driven by filler/hedging tokens, so it correlates
    weakly. Mirrors the paper's Figure 1 (length r ~ -0.5, DTR r ~ +0.8).
    """
    rng = np.random.default_rng(seed)
    dtrs, lens, accs = [], [], []
    for _ in range(n_seq):
        think_prob = float(rng.uniform(0.05, 0.65))
        n_tokens = int(rng.integers(40, 400))
        taus = _sample_taus(rng, n_tokens, think_prob)
        dtr = deep_thinking_ratio(taus, n_layers)
        # Accuracy is driven by DTR plus noise; length adds uninformative bulk.
        logit = 6.0 * dtr - 2.2 + rng.normal(0.0, 0.35)
        acc = 1.0 / (1.0 + math.exp(-logit))
        dtrs.append(dtr)
        lens.append(float(n_tokens))
        accs.append(acc)
    dtrs = np.asarray(dtrs)
    lens = np.asarray(lens)
    accs = np.asarray(accs)
    return {
        "corr_dtr_accuracy": float(np.corrcoef(dtrs, accs)[0, 1]),
        "corr_length_accuracy": float(np.corrcoef(lens, accs)[0, 1]),
        "mean_dtr": float(dtrs.mean()),
    }


# ---------------------------------------------------------------------------
# (b) U-shaped entropy + difficulty routing (DiffAdapt style)
# ---------------------------------------------------------------------------
#
# Paper facts encoded here: generation entropy H(d) is U-shaped in problem
# difficulty d -- high on easy problems despite high accuracy, minimal on
# medium difficulty, high again on hard problems. DiffAdapt routes each
# question to Easy / Normal / Hard inference strategies (different prompt,
# temperature, max tokens) via a small probe on the final hidden state.


def u_entropy(difficulty: np.ndarray, h_min: float = 1.0,
              kappa: float = 0.09, d_star: float = 5.5) -> np.ndarray:
    """U-shaped generation entropy: minimum at medium difficulty."""
    return h_min + kappa * (difficulty - d_star) ** 2


def correctness_prob(difficulty: np.ndarray) -> np.ndarray:
    """Toy: accuracy falls as difficulty rises (paper: easy stays accurate)."""
    return np.clip(1.05 - difficulty / 11.0, 0.05, 0.99)


@dataclass(frozen=True)
class RouteResult:
    strategy: str
    budget_fraction: float
    temperature: float


def route(difficulty: float, max_tokens: int = 1000) -> Tuple[RouteResult, int]:
    """DiffAdapt-style three-tier router. Returns (strategy, token budget)."""
    if difficulty < 4.0:
        # Easy: "direct solving with verification", low temp, small budget.
        return RouteResult("Easy", 0.4, 0.5), int(0.4 * max_tokens)
    if difficulty <= 7.0:
        # Normal: "step-by-step methodical approach", full budget.
        return RouteResult("Normal", 1.0, 0.8), max_tokens
    # Hard: "fail fast" -- strict token limit instead of unproductive churn.
    return RouteResult("Hard", 0.5, 0.4), int(0.5 * max_tokens)


def routing_experiment(n: int = 3000, max_tokens: int = 1000,
                       seed: int = 11) -> Dict[str, float]:
    """Uniform full-budget baseline vs difficulty-routed budgets (toy).

    A question needs ``required(d)`` fraction of the budget; below that it is
    answered wrong. Easy questions need little, hard questions need more than
    the Hard tier gives (fail fast: spend 0.5x, accept the miss).
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(1.0, 10.0, n)
    # Budget fraction needed to solve. Mirrors the paper's three regions:
    # easy problems stay solvable with a small budget (high correctness),
    # normal problems need near-full budget, hard problems exceed the
    # model's capability at any budget ("fail fast": spend 0.5x, cut losses).
    required = np.where(d < 4.0, 0.35, np.where(d <= 7.0, 0.9, 1.6))

    uniform_tokens = float(n * max_tokens)
    uniform_correct = float(np.sum(required <= 1.0))

    routed_tokens, routed_correct = 0.0, 0.0
    for di, ri in zip(d, required):
        _, budget = route(float(di), max_tokens)
        routed_tokens += budget
        routed_correct += 1.0 if budget / max_tokens >= ri else 0.0

    return {
        "uniform_tokens": uniform_tokens,
        "routed_tokens": routed_tokens,
        "token_saving": 1.0 - routed_tokens / uniform_tokens,
        "uniform_accuracy": uniform_correct / n,
        "routed_accuracy": routed_correct / n,
        "mean_entropy_easy": float(u_entropy(np.array([2.0]))[0]),
        "mean_entropy_medium": float(u_entropy(np.array([5.5]))[0]),
        "mean_entropy_hard": float(u_entropy(np.array([9.0]))[0]),
    }


# ---------------------------------------------------------------------------
# (c) Hedging / self-doubt detector on synthetic traces
# ---------------------------------------------------------------------------
#
# Marker vocabularies are adapted from the papers' examples:
#  - hedging: "wait", "hold on", "but maybe", "perhaps", ... (SelfDoubt HVR)
#  - rechecking: "let me double-check", "actually careful", ...
#  - re-derivation: repeated equation statements (toy: repeated '=' lines)
#  - tangents: topic-shift phrases ("by the way", "unrelated", ...)

HEDGE_MARKERS = ["wait", "hold on", "but maybe", "perhaps",
                 "let me reconsider", "not sure", "might be wrong"]
VERIFY_MARKERS = ["verify", "let me check", "substitute back",
                  "check the result"]
RECHECK_MARKERS = ["let me double-check", "actually careful",
                   "recheck", "double check"]
TANGENT_MARKERS = ["by the way", "unrelated", "off topic", "speaking of which"]


def _count_markers(text: str, markers: Sequence[str]) -> int:
    lowered = text.lower()
    total = 0
    for m in markers:
        total += len(re.findall(r"\b" + re.escape(m) + r"\b", lowered))
    return total


def _count_rederivations(text: str) -> int:
    """Toy re-derivation signal: equations restated more than twice.

    Counts '='-containing lines whose left-hand side repeats an earlier one.
    """
    seen = set()
    repeats = 0
    for line in text.splitlines():
        if "=" in line and "==" not in line:
            lhs = line.split("=", 1)[0].strip().lower()
            if lhs and lhs in seen:
                repeats += 1
            seen.add(lhs)
    return repeats


@dataclass
class TraceScan:
    n_tokens: int
    hedge: int
    verify: int
    recheck: int
    rederiv: int
    tangents: int

    @property
    def hvr(self) -> float:
        """Hedge-to-Verify Ratio: h(T) / (v(T) + 1)."""
        return self.hedge / (self.verify + 1)

    def pattern_frequencies(self) -> Dict[str, float]:
        """Normalized frequency per 100 tokens, like the batching paper."""
        scale = 100.0 / max(self.n_tokens, 1)
        return {
            "hedging": self.hedge * scale,
            "rechecking": self.recheck * scale,
            "re-derivation": self.rederiv * scale,
            "tangents": self.tangents * scale,
        }


def scan_trace(text: str) -> TraceScan:
    n_tokens = len(text.split())
    return TraceScan(
        n_tokens=n_tokens,
        hedge=_count_markers(text, HEDGE_MARKERS),
        verify=_count_markers(text, VERIFY_MARKERS),
        recheck=_count_markers(text, RECHECK_MARKERS),
        rederiv=_count_rederivations(text),
        tangents=_count_markers(text, TANGENT_MARKERS),
    )


LONG_TRACE = """\
Let x be the unknown. We have 2x + 3 = 11.
2x = 11 - 3 = 8.
x = 8 / 2 = 4. The answer is 4.
Wait, let me reconsider. But maybe the question is a trick? Hold on.
Let me double-check: 2x + 3 = 11. Actually careful now.
2x = 8 again. x = 4. Verify: 2*4 + 3 = 11. Check the result: yes.
Hmm, not sure though. Perhaps I should re-derive from scratch.
2x + 3 = 11. 2x = 8. x = 4. Same result.
By the way, this reminds me of a similar problem with fractions.
Unrelated thought: linear equations are the simplest kind.
Let me reconsider once more. But maybe I misread the constant?
2x + 3 = 11 -> x = 4. Substitute back: 2*4+3 = 11. Confirmed.
"""

# Same problem, batched-style: the model distributes effort, skips the
# metacognitive loop, and answers directly.
BATCHED_TRACE = """\
2x + 3 = 11 -> 2x = 8 -> x = 4. Verify: 2*4 + 3 = 11. The answer is 4.
"""


def hedging_experiment() -> Dict[str, Dict[str, float]]:
    """Compare overthinking-pattern frequencies: long vs batched trace."""
    long_scan = scan_trace(LONG_TRACE)
    batched_scan = scan_trace(BATCHED_TRACE)
    return {
        "long": long_scan.pattern_frequencies(),
        "batched": batched_scan.pattern_frequencies(),
        "hvr": {"long": long_scan.hvr, "batched": batched_scan.hvr},
    }


# ---------------------------------------------------------------------------
# (d) PRM vs ORM toy on step-labeled data
# ---------------------------------------------------------------------------
#
# ORM: trained to predict final-answer correctness; at test time the score is
# the final-token prediction. False positives -- correct answer via a wrong
# step -- are misgraded as good (Lightman et al. discuss this explicitly).
# PRM: predicts per-step correctness; solution score = product of step
# probabilities, i.e. P(every step correct). Both select best-of-N.


@dataclass(frozen=True)
class Candidate:
    steps_correct: Tuple[bool, ...]
    final_correct: bool


def make_candidates(n: int, seed: int = 5) -> List[Candidate]:
    """Synthetic candidates including ORM false positives.

    30% of candidates reach the right final answer through a wrong step.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        n_steps = int(rng.integers(3, 7))
        steps = tuple(rng.random(n_steps) < 0.85)
        if rng.random() < 0.20:
            # False positive: flip one step wrong, keep the final answer right.
            idx = int(rng.integers(0, n_steps))
            steps = tuple(False if i == idx else s for i, s in enumerate(steps))
            final = True
        else:
            final = all(steps)
        out.append(Candidate(steps, final))
    return out


def orm_score(c: Candidate) -> float:
    """Outcome supervision: only the final answer matters."""
    return 1.0 if c.final_correct else 0.0


def prm_score(c: Candidate) -> float:
    """Process supervision: product of per-step correctness probabilities."""
    return 1.0 if all(c.steps_correct) else 0.0


def best_of_n(candidates: Sequence[Candidate], score_fn, n: int) -> Candidate:
    pool = list(candidates)[:n]
    return max(pool, key=score_fn)


def prm_vs_orm_experiment(n_problems: int = 400, n_candidates: int = 8,
                          seed: int = 21) -> Dict[str, float]:
    """Best-of-N selection: how often does the winner have all steps right?"""
    rng = np.random.default_rng(seed)
    orm_good = prm_good = 0
    problems_with_clean = prm_clean_given_available = 0
    for _ in range(n_problems):
        cands = make_candidates(n_candidates, seed=int(rng.integers(1 << 30)))
        if all(best_of_n(cands, orm_score, n_candidates).steps_correct):
            orm_good += 1
        prm_pick = best_of_n(cands, prm_score, n_candidates)
        if all(prm_pick.steps_correct):
            prm_good += 1
        if any(all(c.steps_correct) for c in cands):
            problems_with_clean += 1
            if all(prm_pick.steps_correct):
                prm_clean_given_available += 1
    return {
        "orm_step_clean_rate": orm_good / n_problems,
        "prm_step_clean_rate": prm_good / n_problems,
        # Whenever a step-clean candidate exists, the PRM rule must find one
        # (its score is 1.0 only for all-steps-correct solutions).
        "prm_clean_given_available": (prm_clean_given_available
                                      / max(problems_with_clean, 1)),
    }


@dataclass(frozen=True)
class DriftTrace:
    """A trace that reaches a correct intermediate answer, then drifts away.

    ``step_correct`` per token block; ``checkpoint_idx`` marks where the model
    first commits to the correct answer; the final block drifts to a wrong
    answer (like the DASH paper's answer-drift traces).
    """
    step_correct: Tuple[bool, ...]
    checkpoint_idx: int


def uniform_advantage(trace: DriftTrace, group_advantage: float) -> List[float]:
    """GRPO-style: one scalar advantage broadcast to every token position."""
    return [group_advantage] * len(trace.step_correct)


def segment_advantage(trace: DriftTrace, group_advantage: float,
                      alpha_pos: float = 1.0, alpha_neg: float = 1.0,
                      gamma: float = 0.5) -> List[float]:
    """DASH-style segment credit: reward the segment that found the correct
    answer, penalize the drift segment with escalating weight."""
    mag = abs(group_advantage)
    adv: List[float] = []
    for i, ok in enumerate(trace.step_correct):
        if i < trace.checkpoint_idx:
            adv.append(group_advantage * 0.2)          # neutral pre-answer
        elif ok:
            adv.append(mag * alpha_pos * gamma)         # productive segment
        else:
            # escalating penalty the further past the checkpoint we drift
            w = 1.0 + (i - trace.checkpoint_idx)
            adv.append(-mag * alpha_neg * w)
    return adv


def credit_assignment_demo() -> Dict[str, List[float]]:
    trace = DriftTrace(step_correct=(True, True, True, False, False),
                       checkpoint_idx=2)
    return {
        "uniform": uniform_advantage(trace, group_advantage=-1.0),
        "segment": segment_advantage(trace, group_advantage=-1.0),
    }


def run_all() -> Dict[str, object]:
    return {
        "deep_thinking": run_correlation_experiment(),
        "routing": routing_experiment(),
        "hedging": hedging_experiment(),
        "prm_vs_orm": prm_vs_orm_experiment(),
        "credit": credit_assignment_demo(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=2, default=float))
