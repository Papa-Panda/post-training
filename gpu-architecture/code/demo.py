#!/usr/bin/env python3
"""Run the CPU-only GPU architecture teaching models."""
from gpu_models import (
    SMResourceLimits,
    coalescing_report,
    collective_cost,
    gemm_traffic,
    occupancy,
    roofline,
    warp_addresses,
)


def main() -> None:
    aligned = coalescing_report(warp_addresses())
    strided = coalescing_report(warp_addresses(stride=2))
    print("Coalescing (32 FP32 lanes, 32-byte teaching segments)")
    print(f"  aligned: {aligned.transactions} transactions, efficiency={aligned.efficiency:.0%}")
    print(f"  stride2: {strided.transactions} transactions, efficiency={strided.efficiency:.0%}")

    limits = SMResourceLimits(
        max_threads=2048,
        max_warps=64,
        max_blocks=32,
        registers=65536,
        shared_memory_bytes=96 * 1024,
    )
    occ = occupancy(
        block_threads=256,
        registers_per_thread=64,
        shared_memory_per_block=32 * 1024,
        limits=limits,
    )
    print("\nIllustrative occupancy (not tied to a named GPU)")
    print(f"  active_blocks={occ.active_blocks}, occupancy={occ.occupancy:.1%}, limit={occ.limiting_resources}")

    roof = roofline(
        flops=1e12,
        bytes_moved=100e9,
        peak_flops_per_s=60e12,
        bandwidth_bytes_per_s=3e12,
    )
    print("\nRoofline")
    print(f"  intensity={roof.arithmetic_intensity:.1f} FLOP/B, ridge={roof.ridge_point_flops_per_byte:.1f} FLOP/B")
    print(f"  predicted bound={roof.bound}, ideal time={roof.ideal_time_s*1e3:.3f} ms")

    gemm = gemm_traffic(1024, 1024, 1024, tile_m=128, tile_n=128)
    print("\nIdealized GEMM data reuse")
    print(f"  naive intensity={gemm.naive_intensity:.2f} FLOP/B")
    print(f"  tiled intensity={gemm.tiled_intensity:.2f} FLOP/B, traffic reduction={gemm.reduction:.1f}x")

    for algorithm in ("ring", "tree"):
        cost = collective_cost(
            kind="all_reduce",
            algorithm=algorithm,
            ranks=8,
            payload_bytes=1024**3,
            alpha_s=2e-6,
            beta_s_per_byte=1 / 100e9,
        )
        print(f"\n{algorithm} all-reduce teaching model")
        print(f"  steps={cost.steps}, per-rank wire={cost.wire_bytes_per_rank/1e9:.3f} GB, total={cost.total_s*1e3:.3f} ms")


if __name__ == "__main__":
    main()
