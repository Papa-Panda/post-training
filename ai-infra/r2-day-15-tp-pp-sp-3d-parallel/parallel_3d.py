"""r2-Day15: 3D parallelism (TP/PP/SP) topology + traffic model.

Pure-Python model of Megatron-style 3D parallelism. Every function below is
exercised by test_parallel_3d.py; nothing here shells out to torch/NCCL/CUDA.

Conventions
-----------
- rank = (d * P + p) * T + t  (Megatron-LM parallel_state.py ordering:
  the T tensor-parallel ranks are contiguous).
- Bandwidth numbers are theoretical (vendor specs), NOT measured:
  NVLink 900 GB/s per GPU (H100 SXM, bidirectional aggregate),
  IB NDR 400 Gb/s = 50 GB/s per NIC, PCIe Gen5 x16 64 GB/s.
- All traffic formulas are ring-collective models from r2-Day03,
  labelled theoretical estimates, not benchmarks.
"""
from __future__ import annotations

NVLINK_GBS = 900.0   # H100 SXM per-GPU bidirectional aggregate (vendor spec)
IB_GBS = 50.0        # NDR 400 Gb/s per NIC (theoretical)
PCIE_GBS = 64.0      # PCIe Gen5 x16 (theoretical)


def check_topology(world_size: int, t: int, p: int, d: int) -> dict:
    """Validate T*P*D == world_size; return the parsed topology."""
    if t <= 0 or p <= 0 or d <= 0:
        raise ValueError("T, P, D must be positive")
    if t * p * d != world_size:
        raise ValueError(f"T*P*D={t * p * d} != world_size={world_size}")
    return {"world_size": world_size, "T": t, "P": p, "D": d}


def rank_to_coords(rank: int, t: int, p: int, d: int) -> tuple[int, int, int]:
    """rank -> (d_rank, p_rank, t_rank) under rank=(d*P+p)*T+t."""
    check_topology(t * p * d, t, p, d)
    if not 0 <= rank < t * p * d:
        raise ValueError(f"rank {rank} out of range")
    t_rank = rank % t
    p_rank = (rank // t) % p
    d_rank = rank // (t * p)
    return (d_rank, p_rank, t_rank)


def coords_to_rank(d_rank: int, p_rank: int, t_rank: int, t: int, p: int, d: int) -> int:
    """(d_rank, p_rank, t_rank) -> rank."""
    check_topology(t * p * d, t, p, d)
    if not (0 <= d_rank < d and 0 <= p_rank < p and 0 <= t_rank < t):
        raise ValueError("coordinate out of range")
    return (d_rank * p + p_rank) * t + t_rank


def tp_group(rank: int, t: int, p: int, d: int) -> list[int]:
    """All ranks sharing (d, p): the tensor-parallel all-reduce group."""
    d_rank, p_rank, _ = rank_to_coords(rank, t, p, d)
    return [coords_to_rank(d_rank, p_rank, tr, t, p, d) for tr in range(t)]


def pp_group(rank: int, t: int, p: int, d: int) -> list[int]:
    """All ranks sharing (d, t): the pipeline stage group (point-to-point)."""
    d_rank, _, t_rank = rank_to_coords(rank, t, p, d)
    return [coords_to_rank(d_rank, pr, t_rank, t, p, d) for pr in range(p)]


def dp_group(rank: int, t: int, p: int, d: int) -> list[int]:
    """All ranks sharing (t, p): the data-parallel gradient-sync group."""
    _, p_rank, t_rank = rank_to_coords(rank, t, p, d)
    return [coords_to_rank(dr, p_rank, t_rank, t, p, d) for dr in range(d)]


def node_of_rank(rank: int, gpus_per_node: int = 8) -> int:
    """Node index assuming ranks fill nodes densely in rank order."""
    return rank // gpus_per_node


def allreduce_ring_bytes(payload_bytes: float, n: int) -> float:
    """Per-rank ring all-reduce traffic: 2*(n-1)/n * payload (r2-Day03)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    return 2.0 * (n - 1) / n * payload_bytes


def tp_comm_bytes_per_rank_per_layer_microbatch(
    b: int, s: int, h: int, t: int, bytes_per_elem: int = 2
) -> float:
    """TP traffic per rank, per layer, per microbatch.

    4 all-reduces/layer (2 fwd + 2 bwd), payload = b*s*h elements each.
    """
    payload = b * s * h * bytes_per_elem
    return 4.0 * allreduce_ring_bytes(payload, t)


def tp_step_bytes_per_rank(b: int, s: int, h: int, n_layers: int, t: int,
                           bytes_per_elem: int = 2) -> float:
    """Aggregate TP wire traffic per rank for one optimizer step (1 microbatch)."""
    return n_layers * tp_comm_bytes_per_rank_per_layer_microbatch(
        b, s, h, t, bytes_per_elem)


def tp_step_seconds_theoretical(total_bytes: float, bandwidth_gbs: float) -> float:
    """Idealized time = bytes / bandwidth. Theoretical only, not a benchmark."""
    return total_bytes / (bandwidth_gbs * 1e9)


def pp_bubble_fraction(p: int, m: int) -> float:
    """GPipe pipeline bubble fraction: (p-1)/(m+p-1)."""
    if p < 1 or m < 1:
        raise ValueError("p and m must be >= 1")
    return (p - 1) / (m + p - 1)


def pp_p2p_bytes_per_boundary(b: int, s: int, h: int,
                              bytes_per_elem: int = 2) -> float:
    """Point-to-point bytes per stage boundary per microbatch.

    forward activations (b*s*h) + backward grads (b*s*h).
    """
    return 2.0 * b * s * h * bytes_per_elem


def sp_activation_saving_factor(t: int) -> int:
    """Sequence parallelism shrinks sharded-region activation memory by T."""
    return t


def tp_params_per_rank(total_params: int, t: int) -> float:
    """Weight params per rank under tensor parallelism (weights split by T)."""
    return total_params / t
