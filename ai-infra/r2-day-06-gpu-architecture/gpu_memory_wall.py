#!/usr/bin/env python3
"""Pure-Python roofline and working-set calculations for r2-Day06.

This script makes theoretical estimates only. It does not benchmark a GPU.
"""

from dataclasses import asdict, dataclass
import argparse
import json


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    hbm_bandwidth_tb_s: float
    fp32_tflop_s: float
    l2_mb: float
    shared_kib_per_sm: float
    max_shared_kib_per_block: float


H100_SXM = GPU(
    name="H100 SXM",
    memory_gb=80.0,
    hbm_bandwidth_tb_s=3.35,
    fp32_tflop_s=67.0,
    l2_mb=50.0,
    shared_kib_per_sm=228.0,
    max_shared_kib_per_block=227.0,
)


def vector_add_traffic(num_elements: int, bytes_per_element: int = 4) -> tuple[int, int]:
    """Return (FLOPs, bytes) for C=A+B: two reads, one write, one add."""
    if num_elements <= 0 or bytes_per_element <= 0:
        raise ValueError("num_elements and bytes_per_element must be positive")
    flops = num_elements
    bytes_moved = 3 * num_elements * bytes_per_element
    return flops, bytes_moved


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    if flops < 0 or bytes_moved <= 0:
        raise ValueError("flops must be non-negative and bytes_moved positive")
    return flops / bytes_moved


def ridge_point_flop_per_byte(gpu: GPU) -> float:
    """Roofline knee = peak arithmetic rate / peak memory bandwidth."""
    return (gpu.fp32_tflop_s * 1e12) / (gpu.hbm_bandwidth_tb_s * 1e12)


def bandwidth_roof_tflop_s(gpu: GPU, intensity_flop_per_byte: float) -> float:
    if intensity_flop_per_byte < 0:
        raise ValueError("intensity must be non-negative")
    return gpu.hbm_bandwidth_tb_s * intensity_flop_per_byte


def theoretical_hbm_time_seconds(gpu: GPU, bytes_moved: int) -> float:
    if bytes_moved < 0:
        raise ValueError("bytes_moved must be non-negative")
    return bytes_moved / (gpu.hbm_bandwidth_tb_s * 1e12)


def bf16_tile_bytes(edge: int, tile_count: int = 2) -> int:
    """Bytes for tile_count square BF16 tiles, e.g. one A and one B tile."""
    if edge <= 0 or tile_count <= 0:
        raise ValueError("edge and tile_count must be positive")
    return tile_count * edge * edge * 2


def build_report(num_elements: int, tile_edge: int) -> dict:
    # Step 1: use the declared device specification in all later calculations.
    gpu = H100_SXM

    # Step 2: account for every algorithmic byte and FLOP of vector add.
    flops, bytes_moved = vector_add_traffic(num_elements)
    intensity = arithmetic_intensity(flops, bytes_moved)

    # Step 3: compare arithmetic intensity with the FP32 roofline knee.
    ridge = ridge_point_flop_per_byte(gpu)
    bound = "memory" if intensity < ridge else "compute"
    bandwidth_roof = bandwidth_roof_tflop_s(gpu, intensity)
    minimum_time = theoretical_hbm_time_seconds(gpu, bytes_moved)

    # Step 4: test whether two BF16 input tiles fit in one block's opt-in shared memory.
    tile_bytes = bf16_tile_bytes(tile_edge)
    shared_limit_bytes = int(gpu.max_shared_kib_per_block * 1024)
    tile_fits = tile_bytes <= shared_limit_bytes

    return {
        "status": "theoretical estimate; execution not validated on H100",
        "gpu": asdict(gpu),
        "vector_add": {
            "elements": num_elements,
            "flops": flops,
            "bytes_moved": bytes_moved,
            "arithmetic_intensity_flop_per_byte": intensity,
            "fp32_ridge_point_flop_per_byte": ridge,
            "classification": f"{bound}-bound",
            "bandwidth_roof_tflop_s": bandwidth_roof,
            "theoretical_minimum_hbm_time_us": minimum_time * 1e6,
        },
        "shared_memory_check": {
            "tile_count": 2,
            "bf16_tile_shape": [tile_edge, tile_edge],
            "working_set_bytes": tile_bytes,
            "max_shared_bytes_per_block": shared_limit_bytes,
            "fits": tile_fits,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--tile-edge", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(build_report(args.elements, args.tile_edge), indent=2))


if __name__ == "__main__":
    main()
