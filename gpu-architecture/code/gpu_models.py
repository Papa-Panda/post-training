"""CPU-only analytical models for GPU architecture study.

The functions model mechanisms, not measured device performance.  They use no
CUDA runtime and therefore remain runnable in CPU-only CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Iterable, Sequence


@dataclass(frozen=True)
class CoalescingReport:
    transactions: int
    transferred_bytes: int
    requested_bytes: int
    efficiency: float
    segments: tuple[int, ...]


def warp_addresses(
    *, base: int = 0, lanes: int = 32, element_bytes: int = 4, stride: int = 1
) -> list[int]:
    """Byte address used by each lane for a scalar strided access."""
    if lanes <= 0 or element_bytes <= 0 or stride <= 0:
        raise ValueError("lanes, element_bytes, and stride must be positive")
    return [base + lane * stride * element_bytes for lane in range(lanes)]


def coalescing_report(
    addresses: Sequence[int], *, access_bytes: int = 4, segment_bytes: int = 32
) -> CoalescingReport:
    """Count aligned memory segments touched by fixed-width lane accesses.

    This is the common CC 6.0+ teaching model based on 32-byte segments.  It is
    deliberately not a cache or DRAM-transaction simulator.
    """
    if access_bytes <= 0 or segment_bytes <= 0:
        raise ValueError("access_bytes and segment_bytes must be positive")
    touched: set[int] = set()
    for address in addresses:
        if address < 0:
            raise ValueError("addresses must be non-negative")
        first = address // segment_bytes
        last = (address + access_bytes - 1) // segment_bytes
        touched.update(range(first, last + 1))
    segments = tuple(sorted(index * segment_bytes for index in touched))
    transferred = len(segments) * segment_bytes
    requested = len(addresses) * access_bytes
    efficiency = requested / transferred if transferred else 1.0
    return CoalescingReport(len(segments), transferred, requested, efficiency, segments)


@dataclass(frozen=True)
class SMResourceLimits:
    max_threads: int
    max_warps: int
    max_blocks: int
    registers: int
    shared_memory_bytes: int
    warp_size: int = 32


@dataclass(frozen=True)
class OccupancyReport:
    active_blocks: int
    active_warps: int
    occupancy: float
    limiting_resources: tuple[str, ...]


def occupancy(
    *,
    block_threads: int,
    registers_per_thread: int,
    shared_memory_per_block: int,
    limits: SMResourceLimits,
) -> OccupancyReport:
    """Resource-ceiling occupancy model for one kernel on one SM.

    Allocation granularities and architecture-specific reserved resources are
    intentionally omitted; use the CUDA occupancy API/Nsight for real launch
    decisions.
    """
    if block_threads <= 0 or block_threads > limits.max_threads:
        raise ValueError("block_threads is outside the SM thread limit")
    if registers_per_thread < 0 or shared_memory_per_block < 0:
        raise ValueError("resource use cannot be negative")
    warps_per_block = ceil(block_threads / limits.warp_size)
    ceilings = {
        "threads": limits.max_threads // block_threads,
        "warps": limits.max_warps // warps_per_block,
        "blocks": limits.max_blocks,
        "registers": (
            limits.registers // (block_threads * registers_per_thread)
            if registers_per_thread
            else limits.max_blocks
        ),
        "shared_memory": (
            limits.shared_memory_bytes // shared_memory_per_block
            if shared_memory_per_block
            else limits.max_blocks
        ),
    }
    active_blocks = max(0, min(ceilings.values()))
    active_warps = active_blocks * warps_per_block
    value = min(1.0, active_warps / limits.max_warps)
    binding = tuple(name for name, ceiling in ceilings.items() if ceiling == active_blocks)
    return OccupancyReport(active_blocks, active_warps, value, binding)


@dataclass(frozen=True)
class RooflineReport:
    arithmetic_intensity: float
    bandwidth_ceiling_flops_per_s: float
    attainable_flops_per_s: float
    ridge_point_flops_per_byte: float
    bound: str
    ideal_time_s: float


def roofline(
    *, flops: float, bytes_moved: float, peak_flops_per_s: float, bandwidth_bytes_per_s: float
) -> RooflineReport:
    """Single-level Roofline upper bound."""
    if min(flops, bytes_moved, peak_flops_per_s, bandwidth_bytes_per_s) <= 0:
        raise ValueError("all inputs must be positive")
    intensity = flops / bytes_moved
    bandwidth_ceiling = bandwidth_bytes_per_s * intensity
    attainable = min(peak_flops_per_s, bandwidth_ceiling)
    ridge = peak_flops_per_s / bandwidth_bytes_per_s
    bound = "memory" if bandwidth_ceiling < peak_flops_per_s else "compute"
    return RooflineReport(
        intensity,
        bandwidth_ceiling,
        attainable,
        ridge,
        bound,
        flops / attainable,
    )


@dataclass(frozen=True)
class CollectiveCost:
    algorithm: str
    steps: int
    latency_s: float
    wire_bytes_per_rank: float
    total_s: float


def collective_cost(
    *, kind: str, algorithm: str, ranks: int, payload_bytes: int, alpha_s: float, beta_s_per_byte: float
) -> CollectiveCost:
    """Alpha-beta teaching models for ring/tree collectives.

    ``payload_bytes`` is the logical tensor size per rank.  The formulas assume
    homogeneous full-duplex links and omit topology contention and protocol
    effects; measured NCCL performance can differ substantially.
    """
    if ranks < 2 or payload_bytes < 0 or alpha_s < 0 or beta_s_per_byte < 0:
        raise ValueError("invalid collective model input")
    if kind not in {"all_reduce", "all_gather", "reduce_scatter"}:
        raise ValueError("unsupported collective kind")
    if algorithm == "ring":
        phases = 2 if kind == "all_reduce" else 1
        steps = phases * (ranks - 1)
        wire_bytes = phases * (ranks - 1) / ranks * payload_bytes
    elif algorithm == "tree":
        depth = ceil(log2(ranks))
        phases = 2 if kind == "all_reduce" else 1
        steps = phases * depth
        # Simplified tree model: a rank may send one full payload per phase level.
        wire_bytes = phases * depth * payload_bytes
    else:
        raise ValueError("algorithm must be 'ring' or 'tree'")
    latency = steps * alpha_s
    return CollectiveCost(algorithm, steps, latency, wire_bytes, latency + wire_bytes * beta_s_per_byte)


@dataclass(frozen=True)
class GemmTraffic:
    flops: int
    naive_bytes: int
    tiled_bytes: int
    naive_intensity: float
    tiled_intensity: float
    reduction: float


def gemm_traffic(
    m: int, n: int, k: int, *, tile_m: int, tile_n: int, element_bytes: int = 2
) -> GemmTraffic:
    """Idealized HBM traffic for C[M,N] = A[M,K] @ B[K,N].

    Naive traffic rereads one A and one B scalar for every FMA.  The tiled
    model loads each A row once per N tile and each B column once per M tile,
    then writes C once.  It excludes cache effects, partial-sum spills, and
    read-modify-write of an existing C.
    """
    if min(m, n, k, tile_m, tile_n, element_bytes) <= 0:
        raise ValueError("dimensions and element size must be positive")
    flops = 2 * m * n * k
    c_write = m * n * element_bytes
    naive = 2 * m * n * k * element_bytes + c_write
    tiled_reads = (ceil(n / tile_n) * m * k + ceil(m / tile_m) * k * n) * element_bytes
    tiled = tiled_reads + c_write
    return GemmTraffic(
        flops,
        naive,
        tiled,
        flops / naive,
        flops / tiled,
        naive / tiled,
    )


def shared_bank_conflict_degree(
    word_indices: Iterable[int], *, banks: int = 32
) -> int:
    """Maximum distinct-address multiplicity on one bank.

    This simple model treats identical-address accesses as broadcast rather than
    a conflict and assumes 32-bit bank words.
    """
    if banks <= 0:
        raise ValueError("banks must be positive")
    per_bank: dict[int, set[int]] = {}
    for word in word_indices:
        if word < 0:
            raise ValueError("word index must be non-negative")
        per_bank.setdefault(word % banks, set()).add(word)
    return max((len(words) for words in per_bank.values()), default=0)
