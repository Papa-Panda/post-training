"""r2-Day14: FSDP/ZeRO memory ledger + collective byte accounting (pure Python).

Every function here actually runs on CPU. There is no torch/CUDA in this
environment, so nothing claims a real FSDP run -- real sharded training is
marked "execution not validated / 待H100验证".

Conventions:
  P : number of model parameters (elements)
  N : number of ranks (data-parallel world size)
  M : bytes participating in one collective
  Stage 0 = plain DDP (replicate everything), 1/2/3 = ZeRO stages.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------
# 1. Model-state ledger: bytes per parameter.
# --------------------------------------------------------------------------

def model_state_bytes_per_param(precision: str = "mixed_adam") -> dict:
    """Bytes/param for params, grads, optimizer states under one precision regime.

    mixed_adam : fp16 params (2B) + fp16 grads (2B) + fp32 master params (4B)
                 + fp32 Adam m (4B) + fp32 Adam v (4B) = 16 B/param.
    fp32_adam  : everything fp32 = 4+4+4+4 = 16 B/param.
    fp32_sgd   : fp32 params + grads + momentum = 4+4+4 = 12 B/param.
    roadmap    : the ROADMAP_45D ledger used for the 7B check --
                 fp16 params (2B) + fp16 grads (2B) + fp32 Adam m+v (8B),
                 i.e. the fp32 master copy is folded away = 12 B/param.
    """
    table = {
        "mixed_adam": {"params": 2, "grads": 2, "master": 4, "opt": 8},
        "fp32_adam": {"params": 4, "grads": 4, "master": 0, "opt": 8},
        "fp32_sgd": {"params": 4, "grads": 4, "master": 0, "opt": 4},
        "roadmap": {"params": 2, "grads": 2, "master": 0, "opt": 8},
    }
    if precision not in table:
        raise ValueError(f"unknown precision regime: {precision}")
    return dict(table[precision])


def sharded_state_bytes(P: int, stage: int, N: int,
                        precision: str = "mixed_adam") -> dict:
    """Per-rank model-state bytes for DDP (stage 0) and ZeRO stages 1/2/3.

    ZeRO-1 shards optimizer states only.
    ZeRO-2 shards optimizer states + gradients.
    ZeRO-3 shards optimizer states + gradients + parameters.
    Everything not sharded is replicated on every rank.
    """
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0 (DDP), 1, 2 or 3")
    if N < 1:
        raise ValueError("N must be >= 1")
    bpp = model_state_bytes_per_param(precision)
    # Which buckets get divided by N at each stage.
    shard = {
        0: set(),
        1: {"opt"},
        2: {"opt", "grads"},
        3: {"opt", "grads", "params", "master"},
    }[stage]
    out = {}
    for bucket, b in bpp.items():
        if b == 0:
            out[bucket] = 0
        elif bucket in shard:
            out[bucket] = P * b / N
        else:
            out[bucket] = P * b
    out["total"] = sum(out.values())
    return out


# --------------------------------------------------------------------------
# 2. Collective byte accounting (ring formulas, consistent with r2-Day03).
# --------------------------------------------------------------------------

def collective_bytes(M: float, N: int, kind: str) -> float:
    """Bytes moved per rank for one collective over M payload bytes.

    Ring all-reduce  : 2(N-1)/N * M   (r2-Day03)
    All-gather       : (N-1)/N * M
    Reduce-scatter   : (N-1)/N * M
    """
    if N < 2:
        return 0.0
    if kind == "allreduce":
        return 2.0 * (N - 1) / N * M
    if kind in ("allgather", "reducescatter"):
        return (N - 1) / N * M
    raise ValueError(f"unknown collective: {kind}")


def ddp_step_comm_bytes(grad_bytes: float, N: int) -> float:
    """Per-rank comm bytes of one DDP optimizer step: one all-reduce of grads."""
    return collective_bytes(grad_bytes, N, "allreduce")


def zero3_step_comm_bytes(param_bytes: float, grad_bytes: float, N: int) -> float:
    """Per-rank comm bytes of one ZeRO-3 step.

    all-gather params before forward  + all-gather params before backward
    + reduce-scatter grads after backward.  ZeRO paper: 1.5x of DDP.
    """
    return (collective_bytes(param_bytes, N, "allgather")
            + collective_bytes(param_bytes, N, "allgather")
            + collective_bytes(grad_bytes, N, "reducescatter"))


# --------------------------------------------------------------------------
# 3. Toy sharding semantics (plain Python lists; byte-exact bookkeeping).
# --------------------------------------------------------------------------

def shard_vector(vec: list, N: int, rank: int) -> list:
    """Rank `rank`'s ZeRO-style shard of `vec` (contiguous split, exact)."""
    n = len(vec)
    if n % N != 0:
        raise ValueError("toy requires len(vec) divisible by N")
    chunk = n // N
    return vec[rank * chunk:(rank + 1) * chunk]


def all_gather_sim(shards: list) -> list:
    """Reassemble the full vector from every rank's shard (order preserved)."""
    out = []
    for s in shards:
        out.extend(s)
    return out


def reduce_scatter_sim(full_vecs: list, N: int) -> list:
    """Sum `full_vecs[r]` elementwise across ranks, then shard the sum.

    Returns the list of per-rank result shards.  Mirrors what a gradient
    reduce-scatter does before the sharded optimizer step.
    """
    n = len(full_vecs[0])
    if any(len(v) != n for v in full_vecs):
        raise ValueError("all rank vectors must have equal length")
    total = [0] * n
    for v in full_vecs:
        for i, x in enumerate(v):
            total[i] += x
    return [shard_vector(total, N, r) for r in range(N)]


# --------------------------------------------------------------------------
# 4. FSDP per-block peak memory.
# --------------------------------------------------------------------------

def fsdp_peak_bytes(P: int, block_params: int, N: int,
                    precision: str = "mixed_adam") -> float:
    """Per-rank peak model-state bytes for per-block FULL_SHARD (ZeRO-3 style).

    Steady state holds only the 1/N shard of every block, but before each
    block's forward the full block params are all-gathered transiently.
    peak = sharded_total + one full block of params (fp16 here).
    This is the FSDP analogue of r2-Day02's (P-b)/G + b DDP peak formula.
    """
    bpp = model_state_bytes_per_param(precision)
    sharded = sharded_state_bytes(P, 3, N, precision)["total"]
    return sharded + block_params * bpp["params"]


def fmt_gb(x: float) -> str:
    """Decimal GB (1e9 bytes), matching vendor specs: H100 80GB, roadmap 14GB."""
    return f"{x / 1e9:.1f} GB"


# --------------------------------------------------------------------------
# 5. Hand-checkable demos (values asserted in test_zero_memory.py).
# --------------------------------------------------------------------------

def demo_7b_ledger() -> None:
    P = 7_000_000_000
    road = sharded_state_bytes(P, 0, 1, "roadmap")
    print("7B roadmap ledger (fp16 P+G + fp32 Adam m+v):")
    for k in ("params", "grads", "opt", "total"):
        print(f"  {k:6s} {fmt_gb(road[k])}")
    fits = road['total'] < 80e9  # decimal GB, like the vendor spec
    print(f"  single 80GB H100 fits? {fits}  (84GB > 80GB: NO)")
    full = sharded_state_bytes(P, 0, 1, "mixed_adam")
    print(f"  with fp32 master copy: {fmt_gb(full['total'])} (16 B/param)")
    z3 = sharded_state_bytes(P, 3, 8, "mixed_adam")
    print(f"  ZeRO-3 over 8 ranks: {fmt_gb(z3['total'])}/rank model states")


def demo_toy_stages() -> None:
    print("Toy P=100 fp32-Adam, N=4  (bytes/rank):")
    for stage in (0, 1, 2, 3):
        s = sharded_state_bytes(100, stage, 4, "fp32_adam")
        print(f"  stage {stage}: total={s['total']:.0f} "
              f"(P={s['params']:.0f} G={s['grads']:.0f} O={s['opt']:.0f})")
    grad_b = 100 * 4
    ddp = ddp_step_comm_bytes(grad_b, 4)
    z3 = zero3_step_comm_bytes(100 * 4, grad_b, 4)
    print(f"  comm/step: DDP={ddp:.0f}B  ZeRO-3={z3:.0f}B  ratio={z3 / ddp:.2f}x")


def demo_shard_semantics() -> None:
    vec = list(range(8))                      # toy "params"
    shards = [shard_vector(vec, 4, r) for r in range(4)]
    assert all_gather_sim(shards) == vec       # gather inverts shard
    grads = [[1] * 8, [2] * 8, [3] * 8, [4] * 8]
    rs = reduce_scatter_sim(grads, 4)
    assert rs[0] == [10, 10]                  # summed then sharded
    print("shard/all-gather/reduce-scatter roundtrips OK")


def demo_fsdp_peak() -> None:
    P, L, N = 7_000_000_000, 32, 8
    peak = fsdp_peak_bytes(P, P // L, N, "mixed_adam")
    print(f"7B, 32 blocks, N=8 per-block FSDP peak: {fmt_gb(peak)}/rank "
          f"(sharded {fmt_gb(sharded_state_bytes(P, 3, N, 'mixed_adam')['total'])} "
          f"+ one full fp16 block)")


if __name__ == "__main__":
    demo_7b_ledger()
    print()
    demo_toy_stages()
    print()
    demo_shard_semantics()
    print()
    demo_fsdp_peak()
