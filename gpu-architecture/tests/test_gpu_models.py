import math
import sys
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from gpu_models import (  # noqa: E402
    SMResourceLimits,
    coalescing_report,
    collective_cost,
    gemm_traffic,
    occupancy,
    roofline,
    shared_bank_conflict_degree,
    warp_addresses,
)


class GPUModelsTest(unittest.TestCase):
    def test_aligned_fp32_warp_uses_four_segments(self):
        report = coalescing_report(warp_addresses())
        self.assertEqual(report.transactions, 4)
        self.assertEqual(report.transferred_bytes, 128)
        self.assertAlmostEqual(report.efficiency, 1.0)

    def test_misaligned_fp32_warp_uses_five_segments(self):
        report = coalescing_report(warp_addresses(base=4))
        self.assertEqual(report.transactions, 5)
        self.assertAlmostEqual(report.efficiency, 0.8)

    def test_stride_two_has_half_segment_efficiency(self):
        report = coalescing_report(warp_addresses(stride=2))
        self.assertEqual(report.transactions, 8)
        self.assertAlmostEqual(report.efficiency, 0.5)

    def test_access_crossing_segment_boundary_counts_both(self):
        report = coalescing_report([31], access_bytes=4)
        self.assertEqual(report.transactions, 2)
        self.assertEqual(report.segments, (0, 32))

    def test_occupancy_reports_resource_limit(self):
        limits = SMResourceLimits(2048, 64, 32, 65536, 96 * 1024)
        report = occupancy(
            block_threads=256,
            registers_per_thread=64,
            shared_memory_per_block=32 * 1024,
            limits=limits,
        )
        self.assertEqual(report.active_blocks, 3)
        self.assertEqual(report.active_warps, 24)
        self.assertAlmostEqual(report.occupancy, 24 / 64)
        self.assertEqual(report.limiting_resources, ("shared_memory",))

    def test_more_registers_can_reduce_occupancy(self):
        limits = SMResourceLimits(2048, 64, 32, 65536, 128 * 1024)
        low = occupancy(block_threads=256, registers_per_thread=32, shared_memory_per_block=0, limits=limits)
        high = occupancy(block_threads=256, registers_per_thread=128, shared_memory_per_block=0, limits=limits)
        self.assertGreater(low.occupancy, high.occupancy)

    def test_roofline_classifies_memory_bound(self):
        report = roofline(flops=1e12, bytes_moved=100e9, peak_flops_per_s=60e12, bandwidth_bytes_per_s=3e12)
        self.assertEqual(report.bound, "memory")
        self.assertAlmostEqual(report.arithmetic_intensity, 10.0)
        self.assertAlmostEqual(report.ridge_point_flops_per_byte, 20.0)
        self.assertAlmostEqual(report.attainable_flops_per_s, 30e12)

    def test_roofline_classifies_compute_bound(self):
        report = roofline(flops=1e12, bytes_moved=10e9, peak_flops_per_s=60e12, bandwidth_bytes_per_s=3e12)
        self.assertEqual(report.bound, "compute")
        self.assertAlmostEqual(report.attainable_flops_per_s, 60e12)

    def test_ring_allreduce_volume(self):
        report = collective_cost(kind="all_reduce", algorithm="ring", ranks=8, payload_bytes=800, alpha_s=0, beta_s_per_byte=1)
        self.assertEqual(report.steps, 14)
        self.assertEqual(report.wire_bytes_per_rank, 1400)
        self.assertEqual(report.total_s, 1400)

    def test_tree_has_logarithmic_steps(self):
        report = collective_cost(kind="all_reduce", algorithm="tree", ranks=8, payload_bytes=1, alpha_s=1, beta_s_per_byte=0)
        self.assertEqual(report.steps, 6)
        self.assertEqual(report.total_s, 6)

    def test_tiling_reduces_idealized_gemm_traffic(self):
        report = gemm_traffic(256, 256, 256, tile_m=64, tile_n=64, element_bytes=2)
        self.assertGreater(report.tiled_intensity, report.naive_intensity)
        self.assertGreater(report.reduction, 1)
        self.assertEqual(report.flops, 2 * 256**3)

    def test_shared_memory_padding_removes_column_conflict(self):
        unpadded = shared_bank_conflict_degree(32 * lane for lane in range(32))
        padded = shared_bank_conflict_degree(33 * lane for lane in range(32))
        self.assertEqual(unpadded, 32)
        self.assertEqual(padded, 1)

    def test_broadcast_is_not_counted_as_conflict(self):
        self.assertEqual(shared_bank_conflict_degree([7] * 32), 1)


if __name__ == "__main__":
    unittest.main()
