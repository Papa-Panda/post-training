#!/usr/bin/env python3
"""CPU-only model of CUDA launch geometry and memory-access mappings.

This computes addresses, 32-byte global-memory segments, and 32-bank shared-
memory mappings. It is not a CUDA benchmark and makes no latency claim.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

WARP_SIZE = 32
GLOBAL_SEGMENT_BYTES = 32
FLOAT32_BYTES = 4
SHARED_BANKS = 32
SHARED_BANK_WIDTH_BYTES = 4


@dataclass(frozen=True)
class LaunchGeometry:
    elements: int
    threads_per_block: int
    blocks: int
    launched_threads: int
    inactive_threads: int


def ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ValueError("numerator must be non-negative and denominator positive")
    return (numerator + denominator - 1) // denominator


def launch_geometry(elements: int, threads_per_block: int) -> LaunchGeometry:
    if elements <= 0 or threads_per_block <= 0:
        raise ValueError("elements and threads_per_block must be positive")
    blocks = ceil_div(elements, threads_per_block)
    launched = blocks * threads_per_block
    return LaunchGeometry(
        elements=elements,
        threads_per_block=threads_per_block,
        blocks=blocks,
        launched_threads=launched,
        inactive_threads=launched - elements,
    )


def warp_float_addresses(*, stride: int = 1, offset_words: int = 0) -> list[int]:
    """Byte addresses requested by one warp of lanes 0..31."""
    if stride <= 0 or offset_words < 0:
        raise ValueError("stride must be positive and offset_words non-negative")
    return [(offset_words + lane * stride) * FLOAT32_BYTES for lane in range(WARP_SIZE)]


def global_segments(addresses: list[int]) -> list[int]:
    """Aligned 32-byte segment bases touched by the supplied byte addresses."""
    if not addresses or any(address < 0 for address in addresses):
        raise ValueError("addresses must be a non-empty list of non-negative integers")
    return sorted({(address // GLOBAL_SEGMENT_BYTES) * GLOBAL_SEGMENT_BYTES for address in addresses})


def global_access_report(*, stride: int = 1, offset_words: int = 0) -> dict:
    addresses = warp_float_addresses(stride=stride, offset_words=offset_words)
    segments = global_segments(addresses)
    requested = len(addresses) * FLOAT32_BYTES
    modeled_transfer = len(segments) * GLOBAL_SEGMENT_BYTES
    return {
        "stride_words": stride,
        "offset_words": offset_words,
        "lane_addresses_bytes": addresses,
        "segment_bases_bytes": segments,
        "segment_count": len(segments),
        "requested_bytes": requested,
        "modeled_transaction_bytes": modeled_transfer,
        "modeled_load_efficiency": requested / modeled_transfer,
    }


def shared_bank_report(*, row_stride_words: int, column: int = 0) -> dict:
    """Map lane x to bank for tile[x][column] in row-major shared memory."""
    if row_stride_words <= 0 or column < 0:
        raise ValueError("row_stride_words must be positive and column non-negative")
    word_indices = [lane * row_stride_words + column for lane in range(WARP_SIZE)]
    banks = [word_index % SHARED_BANKS for word_index in word_indices]
    counts = {bank: banks.count(bank) for bank in sorted(set(banks))}
    return {
        "row_stride_words": row_stride_words,
        "column": column,
        "bank_by_lane": banks,
        "distinct_banks": len(counts),
        "max_lanes_per_bank": max(counts.values()),
    }


def build_report(elements: int, threads_per_block: int) -> dict:
    geometry = launch_geometry(elements, threads_per_block)
    aligned = global_access_report(stride=1, offset_words=0)
    misaligned = global_access_report(stride=1, offset_words=1)
    stride_two = global_access_report(stride=2, offset_words=0)
    bank_conflict = shared_bank_report(row_stride_words=32)
    bank_padded = shared_bank_report(row_stride_words=33)
    return {
        "status": "analytical model only; CUDA execution not validated",
        "launch": asdict(geometry),
        "global_memory": {
            "aligned_unit_stride": aligned,
            "misaligned_unit_stride": misaligned,
            "stride_two": stride_two,
        },
        "shared_memory": {
            "tile_32_columns": bank_conflict,
            "tile_33_columns": bank_padded,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1000)
    parser.add_argument("--threads-per-block", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build_report(args.elements, args.threads_per_block), indent=2))


if __name__ == "__main__":
    main()
