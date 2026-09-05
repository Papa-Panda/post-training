#!/usr/bin/env python3
"""Real Triton fused-softmax kernel source for r2-Day11.

This is genuine, complete Triton code -- the same kernel whose semantics and
traffic model `triton_models.fused_softmax` mirrors on CPU. It is shipped as
guarded source, NOT as an executed artifact: triton / torch / CUDA are not
installed in this environment, so `launch()` refuses to run and the lesson
stays `blocked` / 待H100验证 per the quality gate (no fabricated timings).

Local validation of this file is limited to a Python syntax check
(`ast.parse` in the tests); kernel correctness and any speedup claim are
execution not validated.
"""

try:
    import triton
    import triton.language as tl
    import torch

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - expected in this environment
    _HAS_TRITON = False


if _HAS_TRITON:  # pragma: no cover - needs triton + CUDA GPU

    @triton.jit
    def fused_softmax_kernel(
        x_ptr,          # *fp32, input row(s), strided
        out_ptr,        # *fp32, output, same shape as input
        n_cols,         # int32, row length N
        BLOCK: tl.constexpr,  # compile-time block size (power of 2)
    ):
        """One Triton program fuses a whole softmax row.

        Grid: (triton.cdiv(n_cols, BLOCK),). Program pid handles elements
        [pid*BLOCK, (pid+1)*BLOCK): a *block* of data per program, versus one
        element per thread in the CUDA thread-level model (r2-Day07/08).
        """
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)   # block-level addressing
        mask = offs < n_cols                        # boundary mask, not a branch

        x = tl.load(x_ptr + offs, mask=mask)        # the single load of the row
        m = tl.max(x, axis=0)                       # on-chip reduction
        e = tl.exp(x - m)                           # on-chip, numerically stable
        s = tl.sum(e, axis=0)                       # on-chip reduction
        tl.store(out_ptr + offs, e / s, mask=mask)  # the single store


def launch(x):
    """Launch the fused softmax kernel on a torch tensor.

    Raises RuntimeError in this environment: execution not validated
    (no triton/torch/CUDA here). On a CUDA machine:

        grid = (triton.cdiv(n, BLOCK),)
        fused_softmax_kernel[grid](x, out, n, BLOCK=1024)
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "execution not validated: triton/torch/CUDA are not installed "
            "in this environment; kernel shipped as source only (待H100验证)")
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n)
    grid = (triton.cdiv(n, BLOCK),)
    fused_softmax_kernel[grid](x, out, n, BLOCK)  # type: ignore[name-defined]
    return out
