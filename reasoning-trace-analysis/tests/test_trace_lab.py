"""Semantic tests for code/trace_lab.py. CPU-only, deterministic (fixed seeds)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from trace_lab import (  # noqa: E402
    BATCHED_TRACE,
    LONG_TRACE,
    Candidate,
    DriftTrace,
    best_of_n,
    credit_assignment_demo,
    deep_thinking_ratio,
    hedging_experiment,
    jsd_trajectory,
    make_candidates,
    orm_score,
    prm_score,
    prm_vs_orm_experiment,
    route,
    routing_experiment,
    run_correlation_experiment,
    scan_trace,
    segment_advantage,
    settling_depth,
    u_entropy,
    uniform_advantage,
)

# ---------------------------------------------------------------------------
# (a) deep-thinking ratio
# ---------------------------------------------------------------------------


def test_jsd_trajectory_decays_monotonically():
    traj = jsd_trajectory(tau=10.0, n_layers=36)
    assert len(traj) == 36
    assert np.all(np.diff(traj) < 0), "JSD to final-layer dist must shrink"


def test_settling_depth_early_vs_late():
    fast = jsd_trajectory(tau=2.0, n_layers=36)
    slow = jsd_trajectory(tau=20.0, n_layers=36)
    assert settling_depth(fast, g=0.15) < settling_depth(slow, g=0.15)


def test_settling_depth_matches_definition():
    traj = jsd_trajectory(tau=8.0, n_layers=36)
    env = np.minimum.accumulate(traj)
    c = settling_depth(traj, g=0.15)
    assert env[c - 1] <= 0.15
    if c > 1:
        assert env[c - 2] > 0.15


def test_dtr_bounds_and_extremes():
    fast = [2.0] * 50
    slow = [20.0] * 50
    assert deep_thinking_ratio(fast, 36) == pytest.approx(0.0)
    assert deep_thinking_ratio(slow, 36) == pytest.approx(1.0)
    mixed = [2.0] * 50 + [20.0] * 50
    assert 0.0 < deep_thinking_ratio(mixed, 36) < 1.0
    assert deep_thinking_ratio([], 36) == 0.0


def test_dtr_monotone_in_thinking_fraction():
    base = [2.0] * 80
    d1 = deep_thinking_ratio(base + [20.0] * 20, 36)
    d2 = deep_thinking_ratio(base + [20.0] * 60, 36)
    assert d2 > d1, "more slow-settling tokens -> higher DTR"


def test_dtr_beats_length_as_accuracy_predictor():
    stats = run_correlation_experiment()
    assert stats["corr_dtr_accuracy"] > 0.5
    assert stats["corr_dtr_accuracy"] > stats["corr_length_accuracy"]


# ---------------------------------------------------------------------------
# (b) entropy + routing
# ---------------------------------------------------------------------------


def test_entropy_is_u_shaped():
    d = np.array([2.0, 5.5, 9.0])
    h = u_entropy(d)
    assert h[0] > h[1] and h[2] > h[1], "minimum at medium difficulty"
    assert h[0] > 0 and h[2] > 0


def test_router_tiers():
    strat, budget = route(2.0)
    assert strat.strategy == "Easy" and budget == 400
    strat, budget = route(5.5)
    assert strat.strategy == "Normal" and budget == 1000
    strat, budget = route(9.0)
    assert strat.strategy == "Hard" and budget == 500


def test_routing_saves_tokens_without_losing_accuracy():
    stats = routing_experiment()
    assert stats["token_saving"] > 0.1, "routed must spend clearly less"
    assert stats["routed_accuracy"] >= stats["uniform_accuracy"]
    # entropy reported alongside: easy high, medium lowest, hard high
    assert stats["mean_entropy_easy"] > stats["mean_entropy_medium"]
    assert stats["mean_entropy_hard"] > stats["mean_entropy_medium"]


# ---------------------------------------------------------------------------
# (c) hedging detector
# ---------------------------------------------------------------------------


def test_scan_counts_markers():
    scan = scan_trace(LONG_TRACE)
    assert scan.hedge > 0
    assert scan.verify > 0
    assert scan.n_tokens > 0


def test_hvr_formula():
    scan = scan_trace("wait wait. verify.")
    assert scan.hvr == pytest.approx(scan.hedge / (scan.verify + 1))


def test_batched_trace_suppresses_patterns():
    exp = hedging_experiment()
    for pattern in ("hedging", "rechecking", "re-derivation", "tangents"):
        assert exp["batched"][pattern] <= exp["long"][pattern], pattern
    assert exp["hvr"]["batched"] < exp["hvr"]["long"]
    # The long trace really is pathological in the toy: at least one pattern
    # must be strictly reduced by the batched version.
    assert any(exp["batched"][p] < exp["long"][p]
               for p in ("hedging", "rechecking", "re-derivation", "tangents"))


def test_empty_trace_is_safe():
    scan = scan_trace("")
    assert scan.hvr == 0.0
    assert all(v == 0.0 for v in scan.pattern_frequencies().values())


# ---------------------------------------------------------------------------
# (d) PRM vs ORM
# ---------------------------------------------------------------------------


def test_scores_match_definitions():
    clean = Candidate((True, True, True), True)
    false_pos = Candidate((True, False, True), True)  # right answer, wrong step
    wrong = Candidate((True, False), False)
    assert orm_score(clean) == 1.0 and orm_score(false_pos) == 1.0
    assert orm_score(wrong) == 0.0  # ORM misgrades the false positive as good
    assert prm_score(clean) == 1.0
    assert prm_score(false_pos) == 0.0  # PRM catches the bad step
    assert prm_score(wrong) == 0.0


def test_prm_picks_step_clean_solutions_more_often():
    stats = prm_vs_orm_experiment()
    assert stats["prm_step_clean_rate"] >= stats["orm_step_clean_rate"]
    # PRM's scoring rule is step-clean-monotone: whenever a fully clean
    # candidate exists in the pool, the winner must be one.
    assert stats["prm_clean_given_available"] == pytest.approx(1.0)


def test_credit_assignment_signs():
    trace = DriftTrace(step_correct=(True, True, True, False, False),
                       checkpoint_idx=2)
    uni = uniform_advantage(trace, group_advantage=-1.0)
    assert uni == [-1.0] * 5, "scalar advantage hits every token"
    seg = segment_advantage(trace, group_advantage=-1.0)
    # The segment that found the correct answer gets positive credit even
    # though the rollout-level advantage is negative; the drift suffix gets
    # escalating negative credit.
    assert seg[2] > 0
    assert seg[3] < 0 and seg[4] < seg[3]
    demo = credit_assignment_demo()
    assert demo["uniform"] == uni and demo["segment"] == seg
