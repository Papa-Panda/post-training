#!/usr/bin/env python3
"""Executable CPU semantics and HBM-payload accounting for standard attention
vs tiled attention with online softmax (FlashAttention-style).

Conventions:
- Single head, batch 1: Q, K, V are N x d row-major nested lists.
- `standard_attention` materializes the full N x N score matrix S (the
  baseline FlashAttention replaces).
- `flash_attention` runs the paper's loop structure: outer loop over K/V
  blocks, inner loop over Q blocks, with running (m, l) statistics so S is
  never materialized. Numerically exact, not an approximation.
- Payload counts model scalar fp32 transfers implied by each algorithm's
  traffic pattern. They are THEORETICAL ESTIMATES, not measured
  cache/DRAM transactions. No GPU is touched.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Sequence

Matrix = list[list[float]]
NEG_INF = float("-inf")


@dataclass(frozen=True)
class AttnStats:
    o: Matrix
    n: int
    d: int
    br: int | None
    bc: int | None
    conventional_flops: int
    modeled_hbm_loads: int
    modeled_hbm_stores: int
    modeled_payload_bytes_fp32: int
    logsumexp: tuple[float, ...] | None


def _shape(matrix: Sequence[Sequence[float]], name: str) -> tuple[int, int]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} must be rectangular")
    return rows, cols


def _validate(q: Sequence[Sequence[float]], k: Sequence[Sequence[float]],
              v: Sequence[Sequence[float]]) -> tuple[int, int]:
    nq, d = _shape(q, "Q")
    nk, dk = _shape(k, "K")
    nv, dv = _shape(v, "V")
    if nq <= 0 or d <= 0:
        raise ValueError("Q must have non-zero dimensions")
    if (nk, dk) != (nq, d) or (nv, dv) != (nq, d):
        raise ValueError(
            f"Q/K/V shapes disagree: Q={nq}x{d}, K={nk}x{dk}, V={nv}x{dv}")
    return nq, d


def _conventional_flops(n: int, d: int) -> int:
    # S = QK^T: 2N^2d; rowwise softmax ~ N^2 exp + 2N^2; O = PV: 2N^2d.
    return 4 * n * n * d + 3 * n * n


def standard_attention(q: Sequence[Sequence[float]], k: Sequence[Sequence[float]],
                       v: Sequence[Sequence[float]], scale: float = 1.0) -> AttnStats:
    """Baseline: materialize S (N x N), softmax rows, then O = P V."""
    n, d = _validate(q, k, v)
    s = [[scale * sum(float(q[i][t]) * float(k[j][t]) for t in range(d))
          for j in range(n)] for i in range(n)]
    o: Matrix = [[0.0] * d for _ in range(n)]
    lse: list[float] = []
    for i in range(n):
        m = max(s[i])
        exps = [math.exp(x - m) for x in s[i]]
        total = sum(exps)
        lse.append(m + math.log(total))
        for j in range(n):
            pij = exps[j] / total
            for c in range(d):
                o[i][c] += pij * float(v[j][c])
    # HBM model: read Q,K (2Nd) to form S; write S (N^2); read S (N^2) for
    # softmax; write P (N^2); read P (N^2) and V (Nd) for O; write O (Nd).
    loads = 3 * n * d + 2 * n * n
    stores = 2 * n * n + n * d
    return AttnStats(o=o, n=n, d=d, br=None, bc=None,
                     conventional_flops=_conventional_flops(n, d),
                     modeled_hbm_loads=loads, modeled_hbm_stores=stores,
                     modeled_payload_bytes_fp32=4 * (loads + stores),
                     logsumexp=tuple(lse))


def flash_attention(q: Sequence[Sequence[float]], k: Sequence[Sequence[float]],
                    v: Sequence[Sequence[float]],
                    br: int, bc: int, scale: float = 1.0) -> AttnStats:
    """Tiled exact attention with online softmax.

    Outer loop over K/V blocks (cols of S), inner loop over Q blocks (rows).
    Running per-row statistics m (running max) and l (running normalizer)
    absorb each new block with the rescale factors exp(m_old - m_new) and
    exp(m_block - m_new). S is never materialized.
    """
    n, d = _validate(q, k, v)
    if br < 1 or bc < 1:
        raise ValueError(f"block sizes must be >= 1, got br={br}, bc={bc}")
    t_r = (n + br - 1) // br
    t_c = (n + bc - 1) // bc
    m = [NEG_INF] * n
    l = [0.0] * n
    o: Matrix = [[0.0] * d for _ in range(n)]
    loads = 0
    stores = 0
    for j in range(t_c):
        c0 = j * bc
        c1 = min(c0 + bc, n)
        for i in range(t_r):
            r0 = i * br
            r1 = min(r0 + br, n)
            rlen, clen = r1 - r0, c1 - c0
            # Per paper Algorithm 1 HBM ops: load Q_i, O_i, m_i, l_i, K_j, V_j;
            # store O_i, m_i, l_i. Edge tiles counted at actual size.
            loads += 2 * rlen * d + 2 * clen * d + 2 * rlen
            stores += rlen * d + 2 * rlen
            for ri in range(r0, r1):
                srow = [scale * sum(float(q[ri][t]) * float(k[c][t])
                                    for t in range(d))
                        for c in range(c0, c1)]
                m_block = max(srow)
                p_tilde = [math.exp(x - m_block) for x in srow]
                l_tilde = sum(p_tilde)
                m_new = m_block if m_block > m[ri] else m[ri]
                # m[ri] is -inf on the first visit -> exp(-inf) == 0.0, so the
                # old accumulator is correctly discarded.
                resc_old = math.exp(m[ri] - m_new)
                resc_new = math.exp(m_block - m_new)
                l[ri] = resc_old * l[ri] + resc_new * l_tilde
                for c_ in range(d):
                    pv = sum(p_tilde[kk] * float(v[c0 + kk][c_])
                             for kk in range(clen))
                    o[ri][c_] = resc_old * o[ri][c_] + resc_new * pv
                m[ri] = m_new
    for ri in range(n):
        inv = 1.0 / l[ri]
        for c_ in range(d):
            o[ri][c_] *= inv
    lse = tuple(m[ri] + math.log(l[ri]) for ri in range(n))
    return AttnStats(o=o, n=n, d=d, br=br, bc=bc,
                     conventional_flops=_conventional_flops(n, d),
                     modeled_hbm_loads=loads, modeled_hbm_stores=stores,
                     modeled_payload_bytes_fp32=4 * (loads + stores),
                     logsumexp=lse)


def hand_example() -> tuple[Matrix, Matrix, Matrix]:
    """The hand-computable lesson example: N=4, d=2, scale=1 (stated)."""
    q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    k = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 2.0]]
    v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    return q, k, v


def build_report(br: int = 2, bc: int = 2, scale: float = 1.0) -> dict:
    q, k, v = hand_example()
    std = standard_attention(q, k, v, scale)
    tiled = flash_attention(q, k, v, br, bc, scale)
    err = max(abs(std.o[i][c] - tiled.o[i][c])
              for i in range(std.n) for c in range(std.d))
    return {
        "standard": asdict(std),
        "tiled": asdict(tiled),
        "max_abs_error_tiled_vs_standard": err,
        "note": ("payload counts are a theoretical source-level model, "
                 "not measured DRAM traffic; CUDA/H100 not validated"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--br", type=int, default=2)
    parser.add_argument("--bc", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = build_report(args.br, args.bc, args.scale)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return
    std, tiled = report["standard"], report["tiled"]
    print(f"N={std['n']} d={std['d']} scale={args.scale} "
          f"blocks Br={args.br} Bc={args.bc}")
    print(f"O[0] standard = {[round(x, 4) for x in std['o'][0]]}")
    print(f"O[0] tiled    = {[round(x, 4) for x in tiled['o'][0]]}")
    print(f"max |tiled - standard| = {report['max_abs_error_tiled_vs_standard']:.2e}")
    print(f"logsumexp row0: standard={std['logsumexp'][0]:.4f} "
          f"tiled={tiled['logsumexp'][0]:.4f}")
    print(f"modeled HBM payload (fp32 elements): "
          f"standard={std['modeled_hbm_loads'] + std['modeled_hbm_stores']} "
          f"tiled={tiled['modeled_hbm_loads'] + tiled['modeled_hbm_stores']}")
    print("NOTE: payload counts are a theoretical model, not measured traffic.")


if __name__ == "__main__":
    main()
