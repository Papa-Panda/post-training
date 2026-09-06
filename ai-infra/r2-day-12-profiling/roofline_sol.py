"""Roofline + Speed-of-Light (SOL) semantic model.

Pure-Python, CPU-only. Every number produced here is a *theoretical*
roofline quantity computed from vendor peak specs -- it is NOT a
measurement. The corresponding measurement is Nsight Compute's
SpeedOfLight section (``ncu --section SpeedOfLight``), which is
execution-not-validated in this environment (no CUDA GPU / ncu here).

Conventions (same as r2-day-06): decimal SI prefixes everywhere,
H100 SXM FP32 peak 67 TFLOP/s, HBM peak 3.35 TB/s.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops: float      # FLOP/s, vendor theoretical peak (decimal SI)
    peak_bandwidth: float  # byte/s, vendor theoretical peak (decimal SI)


@dataclass(frozen=True)
class KernelTraffic:
    flops: float  # floating-point ops performed by the kernel
    bytes: float  # HBM payload in bytes (read + write)


# H100 SXM, FP32 CUDA-core peak (matches r2-day-06).
H100_SXM = Hardware("H100 SXM (FP32)", peak_flops=67e12, peak_bandwidth=3.35e12)
# H100 SXM, FP16/BF16 Tensor-Core dense peak: the ridge point moves ~15x up,
# which is why feeding Tensor Cores demands much more data reuse.
H100_SXM_FP16 = Hardware("H100 SXM (FP16 tensor, dense)",
                         peak_flops=989e12, peak_bandwidth=3.35e12)


def arithmetic_intensity(kt: KernelTraffic) -> float:
    """FLOP per byte. Scale-free: N=8 and N=2**20 vector-add share it."""
    return kt.flops / kt.bytes


def ridge_point(hw: Hardware) -> float:
    """I*: AI above this is compute-bound, below is memory-bound."""
    return hw.peak_flops / hw.peak_bandwidth


def bound_class(hw: Hardware, kt: KernelTraffic) -> str:
    """'memory' or 'compute'. A model verdict, not a profiler conclusion."""
    return "compute" if arithmetic_intensity(kt) >= ridge_point(hw) else "memory"


def achievable_flops(hw: Hardware, kt: KernelTraffic) -> float:
    """Roofline ceiling: min(peak FLOPs, AI * bandwidth)."""
    return min(hw.peak_flops, arithmetic_intensity(kt) * hw.peak_bandwidth)


def sol_report(hw: Hardware, kt: KernelTraffic) -> dict:
    """Mimic what ncu's SpeedOfLight section answers.

    For each GPU unit, achieved throughput as % of its theoretical peak.
    A memory-bound kernel is *expected* to show DRAM ~100% and SM ~0.4%:
    that shape is the signature of the bound, not a bug.
    """
    ai = arithmetic_intensity(kt)
    ridge = ridge_point(hw)
    bound = "compute" if ai >= ridge else "memory"
    aflops = min(hw.peak_flops, ai * hw.peak_bandwidth)
    byte_rate = aflops / ai  # byte/s demanded at the roofline point
    return {
        "hardware": hw.name,
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
        "bound": bound,
        "achievable_flops": aflops,
        "sol_flops_pct": 100.0 * aflops / hw.peak_flops,
        "sol_bandwidth_pct": 100.0 * byte_rate / hw.peak_bandwidth,
        # Lower bound on wall time; real kernels are slower (launch, tail
        # effects, imperfect coalescing...). Theoretical estimate only.
        "t_min_s": max(kt.bytes / hw.peak_bandwidth, kt.flops / hw.peak_flops),
    }


def vecadd_traffic(n: int, dtype_bytes: int = 4) -> KernelTraffic:
    """FP32 vector add z = x + y: 1 FLOP/element, 3 elements of traffic."""
    return KernelTraffic(flops=float(n), bytes=float(3 * n * dtype_bytes))


if __name__ == "__main__":
    # Hand-check example from the lesson: N=8 FP32 vector add on H100 SXM.
    kt = vecadd_traffic(8)
    r = sol_report(H100_SXM, kt)
    print(f"N=8 vector add: AI={r['arithmetic_intensity']:.6f} FLOP/byte, "
          f"ridge={r['ridge_point']:.1f}, bound={r['bound']}")
    print(f"  achievable={r['achievable_flops'] / 1e12:.4f} TFLOP/s, "
          f"SOL_flops={r['sol_flops_pct']:.4f}%, "
          f"SOL_bw={r['sol_bandwidth_pct']:.2f}%")
    print(f"  t_min={r['t_min_s'] * 1e12:.2f} ps (theoretical, not measured)")
    print(f"FP16-tensor ridge point: {ridge_point(H100_SXM_FP16):.1f} FLOP/byte")
