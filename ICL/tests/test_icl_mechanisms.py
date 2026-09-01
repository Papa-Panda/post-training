import math
import pathlib
import sys
import unittest

ICL_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ICL_ROOT))

from icl_mechanisms import (  # noqa: E402
    attention_score_bytes,
    induction_distribution,
    kv_cache_bytes,
    linear_attention_prediction,
    linear_gd_update,
    posterior,
    posterior_path,
)


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


class BayesTests(unittest.TestCase):
    def test_repeated_evidence_updates_odds_multiplicatively(self):
        path = posterior_path([0.5, 0.5], [[0.8, 0.2]] * 3)
        expected = [0.5, 0.8, 16 / 17, 64 / 65]
        self.assertEqual(len(path), len(expected))
        for actual, target in zip(path, expected):
            self.assertAlmostEqual(actual[0], target)
            self.assertAlmostEqual(sum(actual), 1.0)

    def test_log_space_stays_finite(self):
        result = posterior([0.5, 0.5], [[1e-20, 1e-21]] * 100)
        self.assertTrue(all(math.isfinite(value) for value in result))
        self.assertAlmostEqual(sum(result), 1.0)
        self.assertGreater(result[0], 1.0 - 1e-12)

    def test_zero_likelihood_excludes_a_concept(self):
        self.assertEqual(posterior([0.5, 0.5], [[1.0, 0.0]]), [1.0, 0.0])

    def test_invalid_likelihood_rejected(self):
        with self.assertRaises(ValueError):
            posterior([0.5, 0.5], [[1.0]])
        with self.assertRaises(ValueError):
            posterior([0.5, 0.5], [[-0.1, 1.0]])
        with self.assertRaises(ValueError):
            posterior([0.5, 0.5], [[0.0, 0.0]])


class GradientDescentTests(unittest.TestCase):
    def test_linear_attention_equals_one_gd_step_at_query(self):
        w0 = [[0.2, -0.1], [0.0, 0.4]]
        x = [[1.0, 2.0], [-1.0, 1.0], [2.0, 0.5]]
        y = [[1.5, -0.2], [-0.5, 0.7], [2.0, 1.1]]
        xq = [0.5, -1.0]
        eta = 0.3
        expected = matvec(linear_gd_update(w0, x, y, eta), xq)
        actual = linear_attention_prediction(w0, x, y, xq, eta)
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def test_full_batch_update_is_order_invariant(self):
        w0 = [[0.0, 0.0]]
        x = [[1.0, 2.0], [-1.0, 3.0], [0.5, 1.0]]
        y = [[2.0], [-1.0], [0.7]]
        forward = linear_gd_update(w0, x, y, 0.2)
        reverse = linear_gd_update(w0, list(reversed(x)), list(reversed(y)), 0.2)
        self.assertEqual(forward, reverse)

    def test_mean_loss_is_invariant_to_full_dataset_duplication(self):
        w0 = [[0.1, -0.2]]
        x = [[1.0, 2.0], [-1.0, 3.0]]
        y = [[2.0], [-1.0]]
        once = linear_gd_update(w0, x, y, 0.2)
        twice = linear_gd_update(w0, x + x, y + y, 0.2)
        for once_row, twice_row in zip(once, twice):
            for once_value, twice_value in zip(once_row, twice_row):
                self.assertAlmostEqual(once_value, twice_value)

    def test_shape_error_is_explicit(self):
        with self.assertRaises(ValueError):
            linear_gd_update([[0.0, 0.0]], [[1.0]], [[1.0]], 0.1)


class CircuitAndSystemsTests(unittest.TestCase):
    def test_induction_aggregates_all_matching_successors(self):
        result = induction_distribution(["A", "B", "A", "C", "A", "B", "A"], "A")
        self.assertEqual(result, {"B": 2 / 3, "C": 1 / 3})
        self.assertEqual(induction_distribution(["A"], "A"), {})

    def test_kv_cache_counts_keys_and_values(self):
        self.assertEqual(kv_cache_bytes(2, 10, 4, 8, 2), 2560)
        self.assertEqual(kv_cache_bytes(2, 20, 4, 8, 2), 5120)

    def test_naive_attention_score_storage_is_quadratic(self):
        small = attention_score_bytes(2, 4, 10, 2)
        large = attention_score_bytes(2, 4, 20, 2)
        self.assertEqual(large, 4 * small)


if __name__ == "__main__":
    unittest.main()
