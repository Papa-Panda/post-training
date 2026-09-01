import unittest

from fsdp_h100_profiler_beyond7b import StatePrecision, estimate_capacity


class CapacityPlannerTest(unittest.TestCase):
    def test_common_mixed_precision_state_is_explicitly_16_bytes(self):
        self.assertEqual(StatePrecision().bytes_per_parameter, 16)

    def test_unsharded_7b_state_is_112_decimal_gb(self):
        report = estimate_capacity(parameters=7_000_000_000, ranks=1)
        self.assertEqual(report.unsharded_model_state_gb, 112.0)
        self.assertEqual(report.fsdp_resident_model_state_gb_per_rank, 112.0)

    def test_fully_sharded_resident_state_scales_with_world_size(self):
        two = estimate_capacity(parameters=7_000_000_000, ranks=2)
        eight = estimate_capacity(parameters=7_000_000_000, ranks=8)
        self.assertEqual(two.fsdp_resident_model_state_gb_per_rank, 56.0)
        self.assertEqual(eight.fsdp_resident_model_state_gb_per_rank, 14.0)

    def test_largest_unit_adds_only_missing_parameter_shards(self):
        report = estimate_capacity(
            parameters=1_000,
            ranks=4,
            largest_layer_parameters=100,
        )
        expected_extra_bytes = 100 * 2 * 3 / 4
        self.assertAlmostEqual(
            report.largest_layer_materialization_extra_gb,
            expected_extra_bytes / 1e9,
        )

    def test_precision_policy_changes_accounting(self):
        without_master = StatePrecision(master_parameter_bytes=0)
        report = estimate_capacity(
            parameters=7_000_000_000,
            ranks=2,
            precision=without_master,
        )
        self.assertEqual(report.bytes_per_parameter, 12)
        self.assertEqual(report.fsdp_resident_model_state_gb_per_rank, 42.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            estimate_capacity(parameters=0, ranks=1)
        with self.assertRaises(ValueError):
            estimate_capacity(parameters=1, ranks=0)
        with self.assertRaises(ValueError):
            estimate_capacity(parameters=10, ranks=1, largest_layer_parameters=11)


if __name__ == "__main__":
    unittest.main()
