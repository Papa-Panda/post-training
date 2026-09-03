#!/usr/bin/env python3
"""Executable CPU semantics and traffic accounting for naive and tiled GEMM.

The traffic counts model scalar payload transfers implied by the two algorithms;
they are theoretical estimates, not measured cache/DRAM transactions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

Matrix = list[list[float]]


@dataclass(frozen=True)
class GemmStats:
    result: Matrix
    m: int
    n: int
    k: int
    tile: int | None
    multiply_adds: int
    conventional_flops: int
    modeled_global_loads: int
    modeled_global_stores: int
    modeled_payload_bytes_fp32: int
    arithmetic_intensity_flop_per_byte: float


def _shape(matrix: Sequence[Sequence[float]], name: str) -> tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return rows, cols


def _validate_inputs(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> tuple[int, int, int]:
    m, k = _shape(a, "A")
    k_b, n = _shape(b, "B")
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("A and B must have non-zero dimensions")
    if k != k_b:
        raise ValueError(f"inner dimensions disagree: A is {m}x{k}, B is {k_b}x{n}")
    return m, n, k


def _stats(result: Matrix, m: int, n: int, k: int, tile: int | None, loads: int) -> GemmStats:
    multiply_adds = m * n * k
    flops = 2 * multiply_adds  # conventional GEMM count: one multiply + one add
    stores = m * n
    payload_bytes = 4 * (loads + stores)
    return GemmStats(
        result=result,
        m=m,
        n=n,
        k=k,
        tile=tile,
        multiply_adds=multiply_adds,
        conventional_flops=flops,
        modeled_global_loads=loads,
        modeled_global_stores=stores,
        modeled_payload_bytes_fp32=payload_bytes,
        arithmetic_intensity_flop_per_byte=flops / payload_bytes,
    )


def naive_gemm(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> GemmStats:
    """Compute C=A@B with one conceptual A/B global load per inner-loop use."""
    m, n, k = _validate_inputs(a, b)
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    loads = 0
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for inner in range(k):
                left = float(a[row][inner])
                right = float(b[inner][col])
                loads += 2
                acc += left * right
            c[row][col] = acc
    return _stats(c, m, n, k, None, loads)


def tiled_gemm(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], tile: int) -> GemmStats:
    """Compute C=A@B by loading each A/B tile once per output tile.

    Edge tiles load only in-bounds values and use zero for padding. The returned
    load count therefore counts useful scalar payload, not issued memory
    transactions or cache-line bytes.
    """
    m, n, k = _validate_inputs(a, b)
    if tile <= 0:
        raise ValueError("tile must be positive")
    c = [[0.0 for _ in range(n)] for _ in range(m)]
    loads = 0

    for row0 in range(0, m, tile):
        rows = min(tile, m - row0)
        for col0 in range(0, n, tile):
            cols = min(tile, n - col0)
            accum = [[0.0 for _ in range(cols)] for _ in range(rows)]
            for inner0 in range(0, k, tile):
                depth = min(tile, k - inner0)
                a_tile = [
                    [float(a[row0 + i][inner0 + q]) for q in range(depth)]
                    for i in range(rows)
                ]
                b_tile = [
                    [float(b[inner0 + q][col0 + j]) for j in range(cols)]
                    for q in range(depth)
                ]
                loads += rows * depth + depth * cols
                for i in range(rows):
                    for j in range(cols):
                        for q in range(depth):
                            accum[i][j] += a_tile[i][q] * b_tile[q][j]
            for i in range(rows):
                for j in range(cols):
                    c[row0 + i][col0 + j] = accum[i][j]
    return _stats(c, m, n, k, tile, loads)


def build_report(size: int, tile: int) -> dict[str, object]:
    if size <= 0:
        raise ValueError("size must be positive")
    a = [[float((i + q) % 7 - 3) for q in range(size)] for i in range(size)]
    b = [[float((2 * q + j) % 5 - 2) for j in range(size)] for q in range(size)]
    naive = naive_gemm(a, b)
    tiled = tiled_gemm(a, b, tile)
    if naive.result != tiled.result:
        raise AssertionError("naive and tiled results differ")
    return {
        "status": "CPU semantic model only; CUDA/H100 execution not validated",
        "naive": asdict(naive),
        "tiled": asdict(tiled),
        "modeled_load_reduction": naive.modeled_global_loads / tiled.modeled_global_loads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--tile", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(build_report(args.size, args.tile), indent=2))


if __name__ == "__main__":
    main()
