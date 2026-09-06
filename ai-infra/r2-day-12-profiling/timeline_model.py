"""Nsight-Systems-style timeline model (illustrative, CPU-only).

A serialized event trace: events occupy [t, t+duration) back to back.
The GPU is 'busy' only during 'kernel' events; every non-kernel event is
an idle gap attributed to the host-side event that caused it.

IMPORTANT: launch overheads and kernel times here are model *parameters*,
not measurements. Kernel times below are roofline lower bounds
(bytes / 3.35 TB/s); the launch overhead (default 5 us) is an illustrative
parameter. Real numbers come from `nsys profile` -- execution not
validated in this environment.
"""

from dataclasses import dataclass

from roofline_sol import H100_SXM


@dataclass(frozen=True)
class Event:
    kind: str        # 'host_launch' | 'kernel' | 'memcpy' | 'host_other'
    duration_us: float
    label: str


def simulate(events) -> dict:
    """Walk the trace once, accumulating span / busy / gaps.

    Every event is actually consumed: kernel events add to gpu_busy_us,
    everything else becomes an attributed idle gap. Nothing is print-only.
    Serialized-execution assumption is documented (no overlap modeling).
    """
    t_us = 0.0
    busy_us = 0.0
    gaps = []
    for e in events:
        if e.kind == "kernel":
            busy_us += e.duration_us
        else:
            gaps.append({"start_us": t_us,
                         "duration_us": e.duration_us,
                         "attributed_to": e.label,
                         "kind": e.kind})
        t_us += e.duration_us
    span_us = t_us
    return {
        "span_us": span_us,
        "gpu_busy_us": busy_us,
        "gpu_idle_us": span_us - busy_us,
        "busy_fraction": busy_us / span_us if span_us > 0 else 0.0,
        "gaps": gaps,
    }


def softmax_demo(n: int = 2 ** 20, launch_overhead_us: float = 5.0) -> dict:
    """Day11's eager-4-pass vs fused-1-pass softmax as Systems traces.

    Payloads come from r2-day-11 (eager 6N elements, fused 2N elements,
    FP32); kernel times are the roofline lower bound payload / 3.35 TB/s.
    """
    bw = H100_SXM.peak_bandwidth  # byte/s, theoretical peak
    eager_bytes = 6 * n * 4
    fused_bytes = 2 * n * 4
    eager_kernel_us = eager_bytes / bw * 1e6
    fused_kernel_us = fused_bytes / bw * 1e6
    launch = "cudaLaunchKernel (illustrative overhead)"

    eager_trace = []
    for label in ("rowmax", "sub+exp", "rowsum", "div"):
        eager_trace.append(Event("host_launch", launch_overhead_us, launch))
        eager_trace.append(Event("kernel", eager_kernel_us / 4.0,
                                 f"softmax/{label}"))
    fused_trace = [Event("host_launch", launch_overhead_us, launch),
                   Event("kernel", fused_kernel_us, "softmax/fused")]

    eager = simulate(eager_trace)
    fused = simulate(fused_trace)
    return {
        "n": n,
        "launch_overhead_us": launch_overhead_us,
        "eager_kernel_us": eager_kernel_us,
        "fused_kernel_us": fused_kernel_us,
        "eager": eager,
        "fused": fused,
        # Model speedup, NOT a benchmark: roofline lower bounds + illustrative
        # launch overhead. The real ratio comes from nsys.
        "model_speedup": eager["span_us"] / fused["span_us"],
    }


if __name__ == "__main__":
    d = softmax_demo()
    for name in ("eager", "fused"):
        r = d[name]
        print(f"{name}: span={r['span_us']:.3f} us, "
              f"gpu_busy={r['gpu_busy_us']:.3f} us "
              f"({100 * r['busy_fraction']:.1f}%), "
              f"gaps={len(r['gaps'])}")
    print(f"model speedup (eager/fused): {d['model_speedup']:.4f}x "
          f"(theoretical, not measured)")
    print("gap attribution (eager):",
          sorted({g["attributed_to"] for g in d["eager"]["gaps"]}))
