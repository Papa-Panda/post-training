import unittest

from gemm_models import build_report, naive_gemm, tiled_gemm


class TestGemmModels(unittest.TestCase):
    def test_hand_computable_two_by_two(self):
        a = [[1.0, 2.0], [3.0, 4.0]]
        b = [[5.0, 6.0], [7.0, 8.0]]
        expected = [[19.0, 22.0], [43.0, 50.0]]
        self.assertEqual(naive_gemm(a, b).result, expected)
        self.assertEqual(tiled_gemm(a, b, 1).result, expected)
        self.assertEqual(tiled_gemm(a, b, 2).result, expected)

    def test_four_by_four_tile_two_traffic(self):
        a = [[float(i * 4 + j + 1) for j in range(4)] for i in range(4)]
        b = [[float(i == j) for j in range(4)] for i in range(4)]
        naive = naive_gemm(a, b)
        tiled = tiled_gemm(a, b, 2)

        self.assertEqual(tiled.result, a)
        self.assertEqual(naive.result, tiled.result)
        self.assertEqual(naive.multiply_adds, 64)
        self.assertEqual(naive.conventional_flops, 128)
        self.assertEqual(naive.modeled_global_loads, 128)
        self.assertEqual(tiled.modeled_global_loads, 64)
        self.assertEqual(naive.modeled_global_stores, 16)
        self.assertEqual(tiled.modeled_global_stores, 16)
        self.assertAlmostEqual(naive.arithmetic_intensity_flop_per_byte, 128 / 576)
        self.assertAlmostEqual(tiled.arithmetic_intensity_flop_per_byte, 128 / 320)

    def test_rectangular_and_edge_tiles(self):
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        expected = [[58.0, 64.0], [139.0, 154.0]]
        self.assertEqual(naive_gemm(a, b).result, expected)
        self.assertEqual(tiled_gemm(a, b, 2).result, expected)

    def test_tile_larger_than_problem(self):
        a = [[2.0, -1.0]]
        b = [[3.0], [4.0]]
        tiled = tiled_gemm(a, b, 16)
        self.assertEqual(tiled.result, [[2.0]])
        self.assertEqual(tiled.modeled_global_loads, 4)

    def test_bad_inner_dimension_rejected(self):
        with self.assertRaises(ValueError):
            naive_gemm([[1.0, 2.0]], [[1.0, 2.0]])

    def test_ragged_and_empty_inputs_rejected(self):
        with self.assertRaises(ValueError):
            tiled_gemm([[1.0], [2.0, 3.0]], [[1.0]], 1)
        with self.assertRaises(ValueError):
            naive_gemm([], [])

    def test_non_positive_tile_rejected(self):
        with self.assertRaises(ValueError):
            tiled_gemm([[1.0]], [[1.0]], 0)

    def test_report_executes_both_algorithms(self):
        report = build_report(4, 2)
        self.assertEqual(report["naive"]["result"], report["tiled"]["result"])
        self.assertEqual(report["modeled_load_reduction"], 2.0)
        self.assertIn("not validated", report["status"])


if __name__ == "__main__":
    unittest.main()
