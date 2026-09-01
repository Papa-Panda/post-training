import math
import unittest

from rl_objectives import (
    aggregate_masked,
    clipped_surrogate,
    compute_gae,
    group_advantages,
    kl_k1,
    kl_k2,
    kl_k3,
    leave_one_out_advantages,
    sequence_ratio_from_token_ratios,
    token_ratios,
)


class TestClipping(unittest.TestCase):
    def test_clipping_is_sign_asymmetric(self):
        ratios = [1.5, 0.5]
        self.assertEqual(clipped_surrogate(ratios, [1.0, 1.0]), [1.2, 0.5])
        self.assertEqual(clipped_surrogate(ratios, [-1.0, -1.0]), [-1.5, -0.8])

    def test_decoupled_clip(self):
        self.assertEqual(clipped_surrogate([1.4], [1.0], 0.2, 0.28), [1.28])


class TestGAE(unittest.TestCase):
    def test_lambda_one_matches_return_minus_value(self):
        rewards = [1.0, 2.0]
        values = [0.4, 0.5, 0.0]
        got = compute_gae(rewards, values, [False, True], gamma=0.9, lam=1.0)
        expected = [1.0 + 0.9 * 2.0 - 0.4, 2.0 - 0.5]
        for actual, want in zip(got, expected):
            self.assertAlmostEqual(actual, want)

    def test_truncation_bootstraps_but_termination_does_not(self):
        truncated = compute_gae([1.0], [0.2, 0.7], [False], gamma=0.9, lam=1.0)
        terminal = compute_gae([1.0], [0.2, 0.7], [True], gamma=0.9, lam=1.0)
        self.assertAlmostEqual(truncated[0], 1.0 + 0.9 * 0.7 - 0.2)
        self.assertAlmostEqual(terminal[0], 1.0 - 0.2)


class TestGroupAdvantages(unittest.TestCase):
    def test_population_zscore(self):
        got = group_advantages([1.0, 0.0, 1.0, 0.0])
        self.assertEqual(got, [1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(sum(got), 0.0)
        self.assertAlmostEqual(sum(x * x for x in got) / len(got), 1.0)

    def test_equal_rewards_zero_and_singleton_rejected(self):
        self.assertEqual(group_advantages([2.0, 2.0]), [0.0, 0.0])
        with self.assertRaises(ValueError):
            group_advantages([1.0])

    def test_leave_one_out_scale_identity(self):
        rewards = [0.0, 1.0, 3.0, 4.0]
        centered = group_advantages(rewards, scale_by_std=False)
        loo = leave_one_out_advantages(rewards)
        for actual, base in zip(loo, centered):
            self.assertAlmostEqual(actual, len(rewards) / (len(rewards) - 1) * base)


class TestRatiosAndKL(unittest.TestCase):
    def test_sequence_ratio_is_token_product(self):
        ratios = token_ratios([math.log(0.4), math.log(0.3)], [math.log(0.2), math.log(0.6)])
        self.assertAlmostEqual(ratios[0], 2.0)
        self.assertAlmostEqual(ratios[1], 0.5)
        self.assertAlmostEqual(sequence_ratio_from_token_ratios(ratios), 1.0)

    def test_k3_is_nonnegative(self):
        for z in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            self.assertGreaterEqual(kl_k3(z, 0.0), -1e-14)
        self.assertEqual(kl_k2(0.0, 0.0), 0.0)

    def test_k1_and_k3_expectations_equal_exact_forward_kl(self):
        p, q = [0.75, 0.25], [0.5, 0.5]
        exact = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
        e_k1 = sum(pi * kl_k1(math.log(pi), math.log(qi)) for pi, qi in zip(p, q))
        e_k3 = sum(pi * kl_k3(math.log(pi), math.log(qi)) for pi, qi in zip(p, q))
        self.assertAlmostEqual(e_k1, exact)
        self.assertAlmostEqual(e_k3, exact)


class TestAggregation(unittest.TestCase):
    def test_response_and_token_mean_encode_different_weighting(self):
        values = [[3.0], [0.0, 0.0, 0.0]]
        masks = [[1], [1, 1, 1]]
        self.assertAlmostEqual(aggregate_masked(values, masks, "response_mean"), 1.5)
        self.assertAlmostEqual(aggregate_masked(values, masks, "token_mean"), 0.75)

    def test_mask_excludes_observation_tokens(self):
        values = [[1.0, 100.0, 3.0], [2.0]]
        masks = [[1, 0, 1], [1]]
        self.assertAlmostEqual(aggregate_masked(values, masks, "token_mean"), 2.0)


if __name__ == "__main__":
    unittest.main()
