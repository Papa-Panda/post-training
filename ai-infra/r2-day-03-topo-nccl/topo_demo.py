"""Unit-aware analytical models for ring collectives and topology labels.

This module does not call NCCL and does not predict benchmark performance.  It
makes the payload, units, startup latency, and assumed effective one-way
bandwidth explicit so that estimates can be checked before measurement.
"""

from dataclasses import dataclass


_TOPOLOGY = {
    "X": "same device",
    "PIX": "path crosses at most one PCIe switch",
    "PXB": "path crosses multiple PCIe switches without a PCIe host bridge",
    "PHB": "path crosses a PCIe host bridge",
    "NODE": "path crosses PCIe host bridges within one NUMA node",
    "SYS": "path crosses PCIe and the interconnect between NUMA nodes",
}


@dataclass(frozen=True)
class RingEstimate:
    collective: str
    ranks: int
    payload_bytes: int
    steps: int
    bytes_sent_per_rank: float
    bandwidth_time_s: float
    startup_time_s: float

    @property
    def total_time_s(self) -> float:
        return self.bandwidth_time_s + self.startup_time_s



def gbps_to_bytes_per_second(gigabits_per_second: float) -> float:
    """Convert decimal Gb/s to decimal bytes/s (400 Gb/s -> 50 GB/s)."""
    if gigabits_per_second <= 0:
        raise ValueError("gigabits_per_second must be positive")
    return gigabits_per_second * 1e9 / 8



def topology_description(label: str) -> str:
    """Explain one nvidia-smi topo -m path label.

    NV# is accepted as a family (for example NV4 or NV12).  The topology matrix
    describes devices in one system; NODE is a NUMA-local PCIe path, not a
    cross-host network path.
    """
    normalized = label.upper()
    if normalized.startswith("NV") and normalized[2:].isdigit():
        return f"bonded set of {int(normalized[2:])} NVLinks"
    try:
        return _TOPOLOGY[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown topology label: {label}") from exc



def ring_estimate(
    collective: str,
    *,
    payload_bytes: int,
    ranks: int,
    effective_bandwidth_bytes_per_s: float,
    step_latency_s: float = 0.0,
) -> RingEstimate:
    """Estimate idealized ring time for one rank.

    ``payload_bytes`` is the logical tensor size before partitioning.
    ``effective_bandwidth_bytes_per_s`` is a measured or explicitly assumed
    one-way payload bandwidth, not a vendor's bidirectional aggregate headline.
    """
    if payload_bytes < 0:
        raise ValueError("payload_bytes must be non-negative")
    if ranks < 2:
        raise ValueError("ring collectives require at least two ranks")
    if effective_bandwidth_bytes_per_s <= 0:
        raise ValueError("effective bandwidth must be positive")
    if step_latency_s < 0:
        raise ValueError("step latency must be non-negative")

    phase_factor = {
        "reduce_scatter": 1,
        "all_gather": 1,
        "all_reduce": 2,
    }
    try:
        phases = phase_factor[collective]
    except KeyError as exc:
        raise ValueError(f"unsupported ring collective: {collective}") from exc

    steps = phases * (ranks - 1)
    bytes_sent = phases * (ranks - 1) / ranks * payload_bytes
    return RingEstimate(
        collective=collective,
        ranks=ranks,
        payload_bytes=payload_bytes,
        steps=steps,
        bytes_sent_per_rank=bytes_sent,
        bandwidth_time_s=bytes_sent / effective_bandwidth_bytes_per_s,
        startup_time_s=steps * step_latency_s,
    )



def main() -> None:
    payload = 1_000_000_000
    assumptions = {
        "NVLink example (450 GB/s one-way assumption)": 450e9,
        "PCIe 5.0 x16 example (64 GB/s one-way nominal)": 64e9,
        "NDR 400 line rate before overhead": gbps_to_bytes_per_second(400),
    }
    for name, bandwidth in assumptions.items():
        estimate = ring_estimate(
            "all_reduce",
            payload_bytes=payload,
            ranks=8,
            effective_bandwidth_bytes_per_s=bandwidth,
            step_latency_s=2e-6,
        )
        print(
            f"{name}: bytes/rank={estimate.bytes_sent_per_rank / 1e9:.3f} GB, "
            f"steps={estimate.steps}, idealized={estimate.total_time_s * 1e3:.3f} ms"
        )
    print("NODE:", topology_description("NODE"))
    print("These are analytical estimates, not NCCL measurements.")


if __name__ == "__main__":
    main()
