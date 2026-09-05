#!/usr/bin/env python3
"""Executable CPU semantics for r2-Day11: Triton block-level programming and
torch.compile graph splitting.

- `eager_softmax` runs the multi-pass PyTorch-eager-style softmax: row max,
  sub+exp, row sum, div are separate passes over the data, exactly what
  `torch.softmax` lowers to without fusion. Element traffic is counted per pass.
- `fused_softmax` runs the single-pass fused version: one load, the whole
  max/exp/sum/div pipeline happens on-"chip", one store. Numerically identical
  to the eager path (this is the CPU mirror of `softmax_triton_kernel.py`).
- `program_grid` computes the Triton program_id -> element mapping, i.e. what
  `pid = tl.program_id(0)`, `offs = pid * BLOCK + tl.arange(0, BLOCK)` and
  `mask = offs < n` do for a 1-D launch.
- `split_graphs` models torch.compile / Dynamo graph breaking: a linear op trace
  containing break markers (data-dependent control flow, `.item()`, `print`, ...)
  is split into graphs; fusion can only happen *inside* one graph.

Traffic counts are THEORETICAL source-level models, not measured DRAM traffic.
No GPU is touched; triton / torch / CUDA are not installed in this environment.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SoftmaxStats:
    o: list[float]
    n: int
    passes: int
    modeled_loads: int
    modeled_stores: int


@dataclass(frozen=True)
class Program:
    pid: int
    offs: list[int]
    mask: list[bool]


def _check_1d(x: list[float], name: str) -> int:
    if not isinstance(x, list) or any(not isinstance(v, (int, float)) for v in x):
        raise ValueError(f"{name} must be a list of numbers")
    n = len(x)
    if n == 0:
        raise ValueError(f"{name} must be non-empty")
    return n


def eager_softmax(x: list[float]) -> SoftmaxStats:
    """Multi-pass eager softmax, one Python pass per would-be kernel.

    Pass 1 (rowmax):  load N.  Pass 2 (sub+exp): load N, store N.
    Pass 3 (rowsum):  load N.  Pass 4 (div):     load N, store N.
    Total: loads 4N, stores 2N, payload 6N elements.
    """
    n = _check_1d(x, "x")
    vals = [float(v) for v in x]
    loads = 0
    stores = 0
    m = max(vals)                      # pass 1: rowmax, loads N
    loads += n
    e = [math.exp(v - m) for v in vals]  # pass 2: sub+exp, loads N, stores N
    loads += n
    stores += n
    s = sum(e)                         # pass 3: rowsum, loads N
    loads += n
    o = [v / s for v in e]             # pass 4: div, loads N, stores N
    loads += n
    stores += n
    return SoftmaxStats(o=o, n=n, passes=4,
                        modeled_loads=loads, modeled_stores=stores)


def fused_softmax(x: list[float]) -> SoftmaxStats:
    """Single-pass fused softmax: the whole pipeline runs on-"chip".

    One load of the row, max/exp/sum/div all happen before the single store.
    Total: loads N, stores N, payload 2N elements. Numerically identical to
    `eager_softmax`; this mirrors `softmax_triton_kernel.fused_softmax_kernel`.
    """
    n = _check_1d(x, "x")
    vals = [float(v) for v in x]
    loads = n                          # the single tl.load of the row
    m = max(vals)                      # on-chip: tl.max
    e = [math.exp(v - m) for v in vals]  # on-chip: tl.exp
    s = sum(e)                         # on-chip: tl.sum
    o = [v / s for v in e]             # on-chip div, then the single tl.store
    stores = n
    return SoftmaxStats(o=o, n=n, passes=1,
                        modeled_loads=loads, modeled_stores=stores)


def program_grid(n: int, block: int) -> list[Program]:
    """Triton 1-D launch mapping: program pid handles BLOCK elements.

    Returns one Program per launched program: the element offsets
    `pid * BLOCK + arange(BLOCK)` and the boundary mask `offs < n`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    programs = []
    for pid in range((n + block - 1) // block):
        offs = [pid * block + t for t in range(block)]
        mask = [o < n for o in offs]
        programs.append(Program(pid=pid, offs=offs, mask=mask))
    return programs


def split_graphs(trace: list[tuple[str, str]]) -> list[list[str]]:
    """Split a linear op trace at graph-break markers.

    `trace` entries are ("op", name) or ("break", reason). Returns the list of
    graphs (each a list of op names); break reasons are recorded in the report
    but fusion stops at every break.
    """
    graphs: list[list[str]] = [[]]
    for kind, payload in trace:
        if kind == "break":
            graphs.append([])
        elif kind == "op":
            graphs[-1].append(payload)
        else:
            raise ValueError(f"trace entry kind must be 'op' or 'break', "
                             f"got {kind!r}")
    return [g for g in graphs if g]


def hand_example() -> list[float]:
    """The hand-computable lesson example: N=8, x = [1..8]."""
    return [float(i) for i in range(1, 9)]


def build_report(n: int = 8, block: int = 4) -> dict:
    x = hand_example()
    assert len(x) == n
    eager = eager_softmax(x)
    fused = fused_softmax(x)
    err = max(abs(a - b) for a, b in zip(eager.o, fused.o))
    grid = program_grid(n, block)
    return {
        "eager": asdict(eager),
        "fused": asdict(fused),
        "max_abs_error_fused_vs_eager": err,
        "program_grid": [asdict(p) for p in grid],
        "note": ("traffic counts are a theoretical source-level model, "
                 "not measured DRAM traffic; Triton/CUDA/H100 not validated"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=int, default=4)
    args = parser.parse_args()
    x = hand_example()
    n = len(x)
    report = build_report(n, args.block)
    eager, fused = report["eager"], report["fused"]
    print(f"N={n} x=[1..8] BLOCK={args.block}")
    print(f"programs launched: {len(report['program_grid'])} "
          f"(ceil({n}/{args.block}))")
    for p in report["program_grid"]:
        print(f"  pid={p['pid']} offs={p['offs']} mask={p['mask']}")
    print(f"max(x) = {max(x):.1f}")
    print(f"fused out[0] = {fused['o'][0]:.6f}  out[7] = {fused['o'][7]:.6f}")
    print(f"max |fused - eager| = {report['max_abs_error_fused_vs_eager']:.2e}")
    print(f"modeled traffic (elements): eager loads={eager['modeled_loads']} "
          f"stores={eager['modeled_stores']} (payload "
          f"{eager['modeled_loads'] + eager['modeled_stores']}); "
          f"fused loads={fused['modeled_loads']} stores={fused['modeled_stores']} "
          f"(payload {fused['modeled_loads'] + fused['modeled_stores']})")
    print("NOTE: traffic counts are a theoretical model, not measured traffic.")


if __name__ == "__main__":
    main()
