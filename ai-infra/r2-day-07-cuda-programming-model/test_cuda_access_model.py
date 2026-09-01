import unittest

from cuda_access_model import (
    ceil_div,
    global_access_report,
    launch_geometry,
    shared_bank_report,
)


class TestCudaAccessModel(unittest.TestCase):
    def test_ceil_div(self):
        self.assertEqual(ceil_div(1000, 256), 4)
        self.assertEqual(ceil_div(1024, 256), 4)

    def test_launch_geometry_uses_guarded_tail(self):
        report = launch_geometry(1000, 256)
        self.assertEqual(report.blocks, 4)
        self.assertEqual(report.launched_threads, 1024)
        self.assertEqual(report.inactive_threads, 24)

    def test_aligned_contiguous_float_warp_uses_four_segments(self):
        report = global_access_report(stride=1, offset_words=0)
        self.assertEqual(report["segment_count"], 4)
        self.assertEqual(report["modeled_load_efficiency"], 1.0)

    def test_one_float_offset_uses_five_segments(self):
        report = global_access_report(stride=1, offset_words=1)
        self.assertEqual(report["segment_count"], 5)
        self.assertAlmostEqual(report["modeled_load_efficiency"], 0.8)

    def test_stride_two_uses_eight_segments(self):
        report = global_access_report(stride=2, offset_words=0)
        self.assertEqual(report["segment_count"], 8)
        self.assertAlmostEqual(report["modeled_load_efficiency"], 0.5)

    def test_unpadded_column_has_32_way_bank_conflict(self):
        report = shared_bank_report(row_stride_words=32)
        self.assertEqual(report["distinct_banks"], 1)
        self.assertEqual(report["max_lanes_per_bank"], 32)

    def test_padding_breaks_column_bank_conflict(self):
        report = shared_bank_report(row_stride_words=33)
        self.assertEqual(report["distinct_banks"], 32)
        self.assertEqual(report["max_lanes_per_bank"], 1)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            launch_geometry(0, 256)
        with self.assertRaises(ValueError):
            global_access_report(stride=0)
        with self.assertRaises(ValueError):
            shared_bank_report(row_stride_words=0)


if __name__ == "__main__":
    unittest.main()
