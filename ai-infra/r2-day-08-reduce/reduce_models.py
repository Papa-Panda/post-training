#!/usr/bin/env python3
"""Executable CPU semantics for three CUDA reduction organizations.

This models data flow and synchronization/atomic counts. It is not a GPU
benchmark and makes no latency, bandwidth, or throughput claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


WARP_SIZE = 32


@dataclass(frozen=True)
class ReductionStats:
    value: float
    input_elements: int
    blocks: int
    global_atomic_updates: int
    block_barriers: int
    shared_writes: int
    warp_shuffle_instructions: int


def _validate_block_size(block_size: int) -> None:
    if block_size < WARP_SIZE or block_size > 1024:
        raise ValueError("block_size must be in [32, 1024]")
    if block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if block_size % WARP_SIZE:
        raise ValueError("block_size must be a multiple of 32")


def atomic_reduce(values: Sequence[float], block_size: int = 256) -> ReductionStats:
    """Model one global atomicAdd for every input element."""
    _validate_block_size(block_size)
    out = 0.0
    for value in values:
        out += float(value)  # the modeled global atomic update is actually used
    blocks = math.ceil(len(values) / block_size) if values else 0
    return ReductionStats(out, len(values), blocks, len(values), 0, 0, 0)


def _shared_tree(tile: list[float]) -> float:
    """Execute an in-place power-of-two binary reduction tree."""
    stride = len(tile) // 2
    while stride:
        for lane in range(stride):
            tile[lane] += tile[lane + stride]
        stride //= 2
    return tile[0]


def shared_tree_reduce(
    values: Sequence[float], block_size: int = 256
) -> ReductionStats:
    """Model one shared-memory tree and one global atomicAdd per block."""
    _validate_block_size(block_size)
    out = 0.0
    blocks = math.ceil(len(values) / block_size) if values else 0
    for block in range(blocks):
        start = block * block_size
        tile = [float(v) for v in values[start : start + block_size]]
        tile.extend([0.0] * (block_size - len(tile)))
        block_sum = _shared_tree(tile)
        out += block_sum  # the modeled block-leader atomic update is used
    barriers_per_block = 1 + int(math.log2(block_size))
    return ReductionStats(
        out,
        len(values),
        blocks,
        blocks,
        blocks * barriers_per_block,
        blocks * block_size,
        0,
    )


def _warp_shuffle_sum(lanes: Sequence[float]) -> float:
    """Execute lane exchange semantics for a full 32-lane down-shuffle tree."""
    if len(lanes) != WARP_SIZE:
        raise ValueError("a modeled warp must contain exactly 32 lanes")
    state = [float(v) for v in lanes]
    for offset in (16, 8, 4, 2, 1):
        previous = state.copy()
        for lane in range(WARP_SIZE - offset):
            state[lane] = previous[lane] + previous[lane + offset]
    return state[0]


def warp_shuffle_reduce(
    values: Sequence[float], block_size: int = 256
) -> ReductionStats:
    """Model warp reductions, one shared warp-sum array, and one block atomic."""
    _validate_block_size(block_size)
    blocks = math.ceil(len(values) / block_size) if values else 0
    warps_per_block = block_size // WARP_SIZE
    out = 0.0

    for block in range(blocks):
        start = block * block_size
        tile = [float(v) for v in values[start : start + block_size]]
        tile.extend([0.0] * (block_size - len(tile)))

        warp_sums = []
        for warp in range(warps_per_block):
            lo = warp * WARP_SIZE
            warp_sums.append(_warp_shuffle_sum(tile[lo : lo + WARP_SIZE]))

        first_warp = warp_sums + [0.0] * (WARP_SIZE - warps_per_block)
        block_sum = _warp_shuffle_sum(first_warp)
        out += block_sum  # the modeled block-leader atomic update is used

    shuffle_instructions = blocks * (warps_per_block + 1) * 5
    return ReductionStats(
        out,
        len(values),
        blocks,
        blocks,
        blocks,  # one block barrier after warp leaders publish partial sums
        blocks * warps_per_block,
        shuffle_instructions,
    )


def build_report(values: Iterable[float], block_size: int) -> dict[str, object]:
    materialized = list(values)
    variants = {
        "global_atomic_per_element": atomic_reduce(materialized, block_size),
        "shared_tree_per_block": shared_tree_reduce(materialized, block_size),
        "warp_shuffle_per_block": warp_shuffle_reduce(materialized, block_size),
    }
    return {
        "status": "CPU semantic model only; CUDA/H100 execution not validated",
        "block_size": block_size,
        "variants": {name: asdict(stats) for name, stats in variants.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    args = parser.parse_args()
    if args.elements < 0:
        parser.error("--elements must be non-negative")
    values = [float(i + 1) for i in range(args.elements)]
    print(json.dumps(build_report(values, args.block_size), indent=2))


if __name__ == "__main__":
    main()
