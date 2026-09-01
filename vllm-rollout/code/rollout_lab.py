#!/usr/bin/env python3
"""CPU-only capacity calculator and rollout queue simulator.

This is a teaching and experiment-design model, not a vLLM performance model.
It makes queue, scheduler-token, KV-capacity, preemption, and policy-age
assumptions explicit so that a real benchmark can replace each parameter.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

GIB = 1024**3


@dataclass(frozen=True)
class ModelShape:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: float = 2.0


@dataclass
class SimConfig:
    duration_s: float = 30.0
    drain_s: float = 30.0
    step_s: float = 0.01
    arrival_rate_rps: float = 4.0
    prompt_tokens_mean: int = 256
    output_tokens_mean: int = 128
    length_cv: float = 0.25
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    prefill_tokens_per_s: float = 20000.0
    decode_tokens_per_s: float = 4000.0
    kv_capacity_tokens: int = 8192
    prefill_chunk_tokens: int = 512
    request_timeout_s: float = 20.0
    max_preemptions_per_request: int = 2
    policy_update_interval_s: float = 5.0
    accept_probability: float = 1.0
    seed: int = 7


@dataclass
class Request:
    request_id: int
    arrival_s: float
    prompt_tokens: int
    target_output_tokens: int
    behavior_version: int
    prompt_remaining: int
    generated_tokens: int = 0
    resident_tokens: int = 0
    preemptions: int = 0
    first_admit_s: Optional[float] = None
    first_token_s: Optional[float] = None
    last_token_s: Optional[float] = None
    finish_s: Optional[float] = None
    accepted: bool = False

    @property
    def stage(self) -> str:
        return "prefill" if self.prompt_remaining > 0 else "decode"


def kv_bytes_per_token(shape: ModelShape) -> float:
    """Logical KV bytes for one sequence token across all layers."""
    if min(shape.layers, shape.kv_heads, shape.head_dim) <= 0 or shape.dtype_bytes <= 0:
        raise ValueError("model dimensions and dtype_bytes must be positive")
    return 2.0 * shape.layers * shape.kv_heads * shape.head_dim * shape.dtype_bytes


def estimate_kv_capacity_tokens(
    total_memory_gib: float,
    gpu_memory_utilization: float,
    weights_gib: float,
    non_kv_gib: float,
    shape: ModelShape,
) -> Dict[str, float]:
    if total_memory_gib <= 0 or weights_gib < 0 or non_kv_gib < 0:
        raise ValueError("memory values must be non-negative and total must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    budget_gib = total_memory_gib * gpu_memory_utilization - weights_gib - non_kv_gib
    per_token = kv_bytes_per_token(shape)
    tokens = max(0, math.floor(budget_gib * GIB / per_token))
    return {
        "kv_bytes_per_token": per_token,
        "kv_budget_gib": max(0.0, budget_gib),
        "estimated_kv_tokens": tokens,
    }


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    xs = sorted(values)
    if not xs:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _sample_length(rng: random.Random, mean: int, cv: float) -> int:
    if mean <= 0:
        raise ValueError("token means must be positive")
    if cv <= 0:
        return mean
    sigma2 = math.log1p(cv * cv)
    mu = math.log(mean) - sigma2 / 2
    return max(1, int(round(rng.lognormvariate(mu, math.sqrt(sigma2)))))


def _validate_config(c: SimConfig) -> None:
    positive = (
        c.duration_s,
        c.step_s,
        c.prompt_tokens_mean,
        c.output_tokens_mean,
        c.max_num_seqs,
        c.max_num_batched_tokens,
        c.prefill_tokens_per_s,
        c.decode_tokens_per_s,
        c.kv_capacity_tokens,
        c.prefill_chunk_tokens,
        c.request_timeout_s,
        c.policy_update_interval_s,
    )
    if any(x <= 0 for x in positive) or c.arrival_rate_rps < 0 or c.drain_s < 0:
        raise ValueError("rates, capacities, token counts, and time limits must be positive")
    if not 0 <= c.accept_probability <= 1:
        raise ValueError("accept_probability must be in [0, 1]")


def generate_requests(c: SimConfig, rng: random.Random) -> List[Request]:
    requests: List[Request] = []
    if c.arrival_rate_rps == 0:
        return requests
    t = rng.expovariate(c.arrival_rate_rps)
    while t < c.duration_s:
        prompt = _sample_length(rng, c.prompt_tokens_mean, c.length_cv)
        output = _sample_length(rng, c.output_tokens_mean, c.length_cv)
        requests.append(
            Request(
                request_id=len(requests),
                arrival_s=t,
                prompt_tokens=prompt,
                target_output_tokens=output,
                behavior_version=int(t // c.policy_update_interval_s),
                prompt_remaining=prompt,
            )
        )
        t += rng.expovariate(c.arrival_rate_rps)
    return requests


def simulate(c: SimConfig) -> Dict[str, Any]:
    """Run a deterministic-seed, finite-capacity, continuous-batching model.

    Each step has a scheduler token budget. Decode is served first, one token per
    sequence per round, then chunked prefill uses the remainder. KV is allocated
    as tokens are processed. When full, the affected request is recomputed from
    scratch, matching the *shape* of recompute preemption without claiming to
    reproduce a particular vLLM release's scheduler.
    """
    _validate_config(c)
    rng = random.Random(c.seed)
    future = generate_requests(c, rng)
    waiting: List[Request] = []
    active: List[Request] = []
    completed: List[Request] = []
    timed_out: List[Request] = []
    capacity_failed: List[Request] = []
    next_arrival = 0
    processed_prefill = 0
    generated_attempted = 0
    preemptions = 0
    max_queue = 0
    max_active = 0
    max_kv = 0
    decode_cursor = 0
    prefill_cursor = 0

    def release(req: Request) -> None:
        req.resident_tokens = 0

    def preempt(req: Request) -> None:
        nonlocal preemptions
        preemptions += 1
        req.preemptions += 1
        release(req)
        req.prompt_remaining = req.prompt_tokens
        req.generated_tokens = 0
        req.first_token_s = None
        req.last_token_s = None
        active.remove(req)
        if req.preemptions > c.max_preemptions_per_request:
            capacity_failed.append(req)
        else:
            waiting.append(req)

    end_s = c.duration_s + c.drain_s
    steps = math.ceil(end_s / c.step_s)
    for step in range(steps + 1):
        now = step * c.step_s
        while next_arrival < len(future) and future[next_arrival].arrival_s <= now:
            waiting.append(future[next_arrival])
            next_arrival += 1

        for pool, sink in ((waiting, timed_out), (active, timed_out)):
            expired = [r for r in pool if now - r.arrival_s >= c.request_timeout_s]
            for req in expired:
                pool.remove(req)
                release(req)
                sink.append(req)

        while waiting and len(active) < c.max_num_seqs:
            req = waiting.pop(0)
            if req.prompt_tokens > c.kv_capacity_tokens:
                capacity_failed.append(req)
                continue
            if req.first_admit_s is None:
                req.first_admit_s = now
            active.append(req)

        scheduler_budget = c.max_num_batched_tokens
        decode_budget = min(scheduler_budget, int(c.decode_tokens_per_s * c.step_s))
        decode_reqs = [r for r in active if r.stage == "decode"]
        if decode_reqs:
            offset = decode_cursor % len(decode_reqs)
            ordered_decode = decode_reqs[offset:] + decode_reqs[:offset]
        else:
            ordered_decode = []
        served = 0
        # Autoregressive decode advances each selected sequence by one token per
        # scheduler step; it never emits many sequential tokens at one timestamp.
        for req in ordered_decode[:decode_budget]:
            kv_used = sum(r.resident_tokens for r in active)
            if kv_used + 1 > c.kv_capacity_tokens:
                preempt(req)
                continue
            req.generated_tokens += 1
            req.resident_tokens += 1
            generated_attempted += 1
            served += 1
            scheduler_budget -= 1
            if req.first_token_s is None:
                req.first_token_s = now
            req.last_token_s = now
            if req.generated_tokens >= req.target_output_tokens:
                req.finish_s = now
                req.accepted = rng.random() < c.accept_probability
                active.remove(req)
                release(req)
                completed.append(req)
        decode_cursor += served

        prefill_budget = min(
            scheduler_budget,
            int(c.prefill_tokens_per_s * c.step_s),
        )
        prefill_reqs = [r for r in active if r.stage == "prefill"]
        while prefill_reqs and prefill_budget > 0:
            req = prefill_reqs[prefill_cursor % len(prefill_reqs)]
            prefill_cursor += 1
            kv_used = sum(r.resident_tokens for r in active)
            room = c.kv_capacity_tokens - kv_used
            take = min(req.prompt_remaining, c.prefill_chunk_tokens, prefill_budget, room)
            if take <= 0:
                preempt(req)
                prefill_reqs = [r for r in active if r.stage == "prefill"]
                continue
            req.prompt_remaining -= take
            req.resident_tokens += take
            processed_prefill += take
            prefill_budget -= take
            if req.prompt_remaining == 0:
                prefill_reqs = [r for r in active if r.stage == "prefill"]

        max_queue = max(max_queue, len(waiting))
        max_active = max(max_active, len(active))
        max_kv = max(max_kv, sum(r.resident_tokens for r in active))
        if now >= c.duration_s and next_arrival == len(future) and not waiting and not active:
            break

    unfinished = waiting + active + future[next_arrival:]
    for req in active:
        release(req)

    ttft = [r.first_token_s - r.arrival_s for r in completed if r.first_token_s is not None]
    e2e = [r.finish_s - r.arrival_s for r in completed if r.finish_s is not None]
    queue_wait = [r.first_admit_s - r.arrival_s for r in completed if r.first_admit_s is not None]
    tpot = [
        (r.last_token_s - r.first_token_s) / (r.generated_tokens - 1)
        for r in completed
        if r.first_token_s is not None and r.last_token_s is not None and r.generated_tokens > 1
    ]
    completion_lag = [
        int((r.finish_s or 0) // c.policy_update_interval_s) - r.behavior_version
        for r in completed
    ]
    completed_tokens = sum(r.target_output_tokens for r in completed)
    accepted_tokens = sum(r.target_output_tokens for r in completed if r.accepted)
    observed_s = min(end_s, (step + 1) * c.step_s)
    return {
        "config": asdict(c),
        "counts": {
            "arrived": len(future),
            "completed": len(completed),
            "accepted": sum(r.accepted for r in completed),
            "timed_out": len(timed_out),
            "capacity_failed": len(capacity_failed),
            "unfinished": len(unfinished),
            "preemptions": preemptions,
        },
        "tokens": {
            "processed_prefill": processed_prefill,
            "generated_attempted": generated_attempted,
            "generated_completed": completed_tokens,
            "generated_accepted": accepted_tokens,
            "wasted_generated": generated_attempted - accepted_tokens,
        },
        "rates": {
            "request_throughput_rps": len(completed) / observed_s,
            "output_throughput_tps": completed_tokens / observed_s,
            "useful_goodput_tps": accepted_tokens / observed_s,
            "offered_output_tps": c.arrival_rate_rps * c.output_tokens_mean,
        },
        "latency_s": {
            "queue_p50": percentile(queue_wait, 0.50),
            "queue_p95": percentile(queue_wait, 0.95),
            "ttft_p50": percentile(ttft, 0.50),
            "ttft_p95": percentile(ttft, 0.95),
            "tpot_p50": percentile(tpot, 0.50),
            "tpot_p95": percentile(tpot, 0.95),
            "e2e_p50": percentile(e2e, 0.50),
            "e2e_p95": percentile(e2e, 0.95),
        },
        "policy_lag_versions": {
            "p50": percentile(completion_lag, 0.50),
            "p95": percentile(completion_lag, 0.95),
            "max": max(completion_lag) if completion_lag else None,
        },
        "peaks": {
            "queue_depth": max_queue,
            "active_sequences": max_active,
            "kv_tokens": max_kv,
            "kv_utilization": max_kv / c.kv_capacity_tokens,
        },
        "model_notice": (
            "CPU abstraction only: rates and capacities are inputs. "
            "Calibrate them from a version-pinned benchmark before decisions."
        ),
    }


def _load_config(path: Optional[str]) -> SimConfig:
    if not path:
        return SimConfig()
    data = json.loads(Path(path).read_text())
    unknown = set(data) - set(SimConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return SimConfig(**data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    kv = sub.add_parser("kv", help="estimate logical KV bytes/token and token capacity")
    kv.add_argument("--layers", type=int, required=True)
    kv.add_argument("--kv-heads", type=int, required=True)
    kv.add_argument("--head-dim", type=int, required=True)
    kv.add_argument("--dtype-bytes", type=float, default=2.0)
    kv.add_argument("--total-memory-gib", type=float, required=True)
    kv.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    kv.add_argument("--weights-gib", type=float, required=True)
    kv.add_argument("--non-kv-gib", type=float, required=True)

    sim = sub.add_parser("simulate", help="run the finite-capacity rollout simulation")
    sim.add_argument("--config", help="JSON file containing SimConfig fields")
    sim.add_argument("--out", help="optional JSON output path")
    sim.add_argument("--arrival-rate-rps", type=float)
    sim.add_argument("--kv-capacity-tokens", type=int)
    sim.add_argument("--max-num-seqs", type=int)
    sim.add_argument("--seed", type=int)

    args = parser.parse_args()
    if args.command == "kv":
        result = estimate_kv_capacity_tokens(
            args.total_memory_gib,
            args.gpu_memory_utilization,
            args.weights_gib,
            args.non_kv_gib,
            ModelShape(args.layers, args.kv_heads, args.head_dim, args.dtype_bytes),
        )
    else:
        config = _load_config(args.config)
        for name in ("arrival_rate_rps", "kv_capacity_tokens", "max_num_seqs", "seed"):
            value = getattr(args, name)
            if value is not None:
                setattr(config, name, value)
        result = simulate(config)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
