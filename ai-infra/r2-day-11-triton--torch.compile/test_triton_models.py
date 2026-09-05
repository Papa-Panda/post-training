import ast
import math
import os
import unittest

from triton_models import (
    eager_softmax,
    fused_softmax,
    hand_example,
    program_grid,
    split_graphs,
)

KERNEL_PATH = os.path.join(os.path.dirname(__file__),
                           "softmax_triton_kernel.py")


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


class TestSoftmaxSemantics(unittest.TestCase):
    def test_hand_example_matches_hand_computation(self):
        # Hand-derived in README: x=[1..8], m=8, S=sum(exp(i-8)),
        # out[0]=exp(-7)/S, out[7]=1/S.
        x = hand_example()
        fused = fused_softmax(x)
        s = sum(math.exp(v - 8.0) for v in x)
        self.assertAlmostEqual(s, 1.5814460128059595, places=12)
        self.assertAlmostEqual(fused.o[0], math.exp(-7.0) / s, places=12)
        self.assertAlmostEqual(fused.o[0], 0.0005766127696870058, places=12)
        self.assertAlmostEqual(fused.o[7], 1.0 / s, places=12)
        self.assertAlmostEqual(fused.o[7], 0.6323326828120425, places=12)

    def test_fused_exactly_matches_eager(self):
        x = hand_example()
        eager = eager_softmax(x)
        fused = fused_softmax(x)
        self.assertLess(_max_abs_diff(eager.o, fused.o), 1e-12)
        # Probabilities sum to 1 on both paths.
        self.assertAlmostEqual(sum(eager.o), 1.0, places=12)
        self.assertAlmostEqual(sum(fused.o), 1.0, places=12)

    def test_traffic_counts_match_hand_model(self):
        # N=8. Eager: 4 passes -> loads 4N=32, stores 2N=16, payload 48 = 6N.
        # Fused: single pass -> loads N=8, stores N=8, payload 16 = 2N.
        x = hand_example()
        eager = eager_softmax(x)
        fused = fused_softmax(x)
        self.assertEqual((eager.modeled_loads, eager.modeled_stores), (32, 16))
        self.assertEqual((fused.modeled_loads, fused.modeled_stores), (8, 8))
        self.assertEqual(eager.passes, 4)
        self.assertEqual(fused.passes, 1)

    def test_larger_deterministic_exact(self):
        x = [float((i * 13 + 7) % 17) - 8.0 for i in range(37)]
        eager = eager_softmax(x)
        fused = fused_softmax(x)
        self.assertLess(_max_abs_diff(eager.o, fused.o), 1e-12)
        self.assertEqual((fused.modeled_loads, fused.modeled_stores), (37, 37))

    def test_single_element(self):
        self.assertEqual(fused_softmax([3.0]).o, [1.0])
        self.assertEqual(eager_softmax([3.0]).o, [1.0])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            eager_softmax([])
        with self.assertRaises(ValueError):
            fused_softmax([])


class TestProgramGrid(unittest.TestCase):
    def test_hand_grid_two_programs(self):
        grid = program_grid(8, 4)
        self.assertEqual(len(grid), 2)
        self.assertEqual(grid[0].offs, [0, 1, 2, 3])
        self.assertEqual(grid[1].offs, [4, 5, 6, 7])
        self.assertTrue(all(grid[0].mask) and all(grid[1].mask))

    def test_edge_program_mask(self):
        # N=10, BLOCK=4 -> 3 programs; last program masks 2 of 4 lanes.
        grid = program_grid(10, 4)
        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[2].pid, 2)
        self.assertEqual(grid[2].offs, [8, 9, 10, 11])
        self.assertEqual(grid[2].mask, [True, True, False, False])

    def test_bad_args_rejected(self):
        with self.assertRaises(ValueError):
            program_grid(0, 4)
        with self.assertRaises(ValueError):
            program_grid(8, 0)


class TestGraphSplit(unittest.TestCase):
    def test_two_breaks_make_three_graphs(self):
        trace = [("op", "matmul"), ("op", "add"),
                 ("break", "tensor.item() forces sync"),
                 ("op", "relu"), ("op", "mul"),
                 ("break", "data-dependent python if"),
                 ("op", "softmax")]
        graphs = split_graphs(trace)
        self.assertEqual(graphs, [["matmul", "add"], ["relu", "mul"],
                                  ["softmax"]])

    def test_no_break_single_graph(self):
        trace = [("op", "matmul"), ("op", "add"), ("op", "relu")]
        self.assertEqual(split_graphs(trace), [["matmul", "add", "relu"]])

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            split_graphs([("kernel", "matmul")])


class TestKernelSource(unittest.TestCase):
    def test_kernel_file_parses(self):
        with open(KERNEL_PATH, encoding="utf-8") as f:
            src = f.read()
        ast.parse(src)  # syntax check only; execution needs triton + CUDA

    def test_kernel_has_triton_structure(self):
        with open(KERNEL_PATH, encoding="utf-8") as f:
            src = f.read()
        for token in ("@triton.jit", "tl.program_id", "tl.arange",
                      "tl.load", "tl.store", "mask", "BLOCK: tl.constexpr"):
            self.assertIn(token, src)

    def test_launch_refuses_without_triton(self):
        import softmax_triton_kernel as k
        self.assertFalse(k._HAS_TRITON)
        with self.assertRaises(RuntimeError):
            k.launch([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
