import unittest

from gpu_memory_wall import (
    H100_SXM,
    arithmetic_intensity,
    bf16_tile_bytes,
    build_report,
    ridge_point_flop_per_byte,
    theoretical_hbm_time_seconds,
    vector_add_traffic,
)


class MemoryWallTests(unittest.TestCase):
    def test_vector_add_accounting(self):
        flops, bytes_moved = vector_add_traffic(1 << 20)
        self.assertEqual(flops, 1_048_576)
        self.assertEqual(bytes_moved, 12_582_912)
        self.assertAlmostEqual(arithmetic_intensity(flops, bytes_moved), 1 / 12)

    def test_h100_fp32_ridge_point(self):
        self.assertAlmostEqual(ridge_point_flop_per_byte(H100_SXM), 20.0)

    def test_theoretical_minimum_time(self):
        _, bytes_moved = vector_add_traffic(1 << 20)
        self.assertAlmostEqual(
            theoretical_hbm_time_seconds(H100_SXM, bytes_moved) * 1e6,
            3.756092,
            places=5,
        )

    def test_two_bf16_tiles_fit_or_do_not_fit(self):
        self.assertEqual(bf16_tile_bytes(128), 65_536)
        self.assertLessEqual(bf16_tile_bytes(128), 227 * 1024)
        self.assertGreater(bf16_tile_bytes(256), 227 * 1024)

    def test_report_uses_calculated_values(self):
        report = build_report(1 << 20, 128)
        self.assertEqual(report["vector_add"]["classification"], "memory-bound")
        self.assertTrue(report["shared_memory_check"]["fits"])
        self.assertIn("execution not validated", report["status"])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            vector_add_traffic(0)
        with self.assertRaises(ValueError):
            bf16_tile_bytes(-1)


if __name__ == "__main__":
    unittest.main()
