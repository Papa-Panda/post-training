import math
import unittest

from flash_models import flash_attention, hand_example, standard_attention


def _max_abs_diff(a, b):
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


class TestFlashModels(unittest.TestCase):
    def test_hand_example_exactness(self):
        q, k, v = hand_example()
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 2, 2)
        self.assertLess(_max_abs_diff(std.o, tiled.o), 1e-9)

    def test_hand_example_row0_matches_hand_computation(self):
        # Hand-derived in README: S row0 = [1,0,1,0] (scale=1),
        # softmax = [e,1,e,1]/(2e+2), O_0 = (3.5379, 4.5379).
        q, k, v = hand_example()
        o0 = flash_attention(q, k, v, 2, 2).o[0]
        self.assertAlmostEqual(o0[0], 3.5379, places=3)
        self.assertAlmostEqual(o0[1], 4.5379, places=3)

    def test_logsumexp_matches_stable_reference(self):
        q, k, v = hand_example()
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 2, 2)
        for a, b in zip(std.logsumexp, tiled.logsumexp):
            self.assertAlmostEqual(a, b, places=12)
        # Direct stable logsumexp of the materialized S row 0 = log(2e+2).
        self.assertAlmostEqual(tiled.logsumexp[0], math.log(2 * math.e + 2),
                               places=12)

    def test_toy_payload_counts_match_hand_model(self):
        # N=4, d=2, Br=Bc=2. Standard: loads 3Nd+2N^2=56, stores 2N^2+Nd=40.
        # Tiled: 4 iters x (loads 20, stores 8) = 80/32. The toy sits near
        # break-even, which is the lesson's "when it does not pay" point.
        q, k, v = hand_example()
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 2, 2)
        self.assertEqual((std.modeled_hbm_loads, std.modeled_hbm_stores),
                         (56, 40))
        self.assertEqual((tiled.modeled_hbm_loads, tiled.modeled_hbm_stores),
                         (80, 32))
        self.assertEqual(std.conventional_flops, tiled.conventional_flops)
        self.assertEqual(std.conventional_flops, 4 * 16 * 2 + 3 * 16)

    def test_nondvisible_tiles_exact(self):
        n, d = 5, 3
        q = [[float(i * d + t + 1) for t in range(d)] for i in range(n)]
        k = [[float((i + t) % 4) for t in range(d)] for i in range(n)]
        v = [[float(i - t) for t in range(d)] for i in range(n)]
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 2, 3)
        self.assertLess(_max_abs_diff(std.o, tiled.o), 1e-9)

    def test_larger_deterministic_exact(self):
        n, d = 8, 4
        q = [[float((i * 7 + t * 3) % 5) for t in range(d)] for i in range(n)]
        k = [[float((i * 5 - t * 2) % 5) for t in range(d)] for i in range(n)]
        v = [[float(i + t) for t in range(d)] for i in range(n)]
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 4, 2)
        self.assertLess(_max_abs_diff(std.o, tiled.o), 1e-9)

    def test_single_block_exact(self):
        q, k, v = hand_example()
        std = standard_attention(q, k, v)
        tiled = flash_attention(q, k, v, 8, 8)
        self.assertLess(_max_abs_diff(std.o, tiled.o), 1e-9)

    def test_scale_consistent_across_paths(self):
        q, k, v = hand_example()
        std = standard_attention(q, k, v, scale=0.5)
        tiled = flash_attention(q, k, v, 2, 2, scale=0.5)
        self.assertLess(_max_abs_diff(std.o, tiled.o), 1e-9)

    def test_degenerate_1x1(self):
        std = standard_attention([[2.0]], [[3.0]], [[5.0]])
        tiled = flash_attention([[2.0]], [[3.0]], [[5.0]], 1, 1)
        self.assertEqual(std.o, [[5.0]])
        self.assertEqual(tiled.o, [[5.0]])

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            standard_attention([[1.0, 2.0]], [[1.0]], [[1.0]])
        with self.assertRaises(ValueError):
            flash_attention([[1.0]], [[1.0]], [[1.0, 2.0]], 1, 1)
        with self.assertRaises(ValueError):
            flash_attention([[1.0], [2.0, 3.0]], [[1.0], [1.0]],
                            [[1.0], [1.0]], 1, 1)
        with self.assertRaises(ValueError):
            flash_attention([[1.0]], [[1.0]], [[1.0]], 0, 1)
        with self.assertRaises(ValueError):
            standard_attention([], [], [])


if __name__ == "__main__":
    unittest.main()
