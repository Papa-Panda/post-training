import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "code"))

from model_aware_curation import (  # noqa: E402
    conflict_scores,
    cosine_scores,
    fisher_logdet,
    fisher_marginal_gain,
    gradient_isolation_scores,
    gradient_vendi,
    greedy_select,
    rademacher_project,
    selected_set_conflict,
    sparse_cluster_ids,
    spice_greedy_select,
    spice_score,
)


class ModelAwareCurationTest(unittest.TestCase):
    def test_vendi_identical_is_one(self):
        g = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.assertAlmostEqual(gradient_vendi(g), 1.0, places=10)

    def test_vendi_orthogonal_is_dimension(self):
        self.assertAlmostEqual(gradient_vendi(np.eye(3)), 3.0, places=10)

    def test_cosine_and_conflict(self):
        g = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(cosine_scores(g, [1.0, 0.0]), [1.0, -1.0, 0.0])
        np.testing.assert_allclose(conflict_scores(g, [1.0, 0.0]), [0.0, 1.0, 0.0])

    def test_fisher_logdet_is_sign_blind(self):
        selected = np.array([[1.0, 0.0]])
        positive = np.array([1.0, 0.0])
        negative = -positive
        self.assertAlmostEqual(
            fisher_marginal_gain(selected, positive),
            fisher_marginal_gain(selected, negative),
            places=12,
        )
        self.assertAlmostEqual(
            fisher_logdet(np.vstack([selected, positive])),
            fisher_logdet(np.vstack([selected, negative])),
            places=12,
        )

    def test_spice_conflict_distinguishes_gradient_sign(self):
        selected = np.array([[1.0, 0.0]])
        positive = np.array([1.0, 0.0])
        negative = -positive
        self.assertAlmostEqual(selected_set_conflict(positive, selected), 0.0)
        self.assertAlmostEqual(selected_set_conflict(negative, selected), 1.0)
        self.assertGreater(
            spice_score(selected, positive, conflict_weight=0.1),
            spice_score(selected, negative, conflict_weight=0.1),
        )

    def test_selected_set_conflict_is_not_retention_risk(self):
        candidate = np.array([[1.0, 0.0]])
        selected = np.array([[1.0, 0.0]])
        protected = np.array([-1.0, 0.0])
        set_conflict = selected_set_conflict(candidate[0], selected)
        retention_risk = conflict_scores(candidate, protected)[0]
        self.assertAlmostEqual(set_conflict, 0.0)
        self.assertAlmostEqual(retention_risk, 1.0)

    def test_isolation_does_not_imply_target_value(self):
        gradients = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.1, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        isolation = gradient_isolation_scores(gradients)
        target_value = cosine_scores(gradients, [1.0, 0.0, 0.0])
        self.assertGreater(isolation[2], isolation[0])
        self.assertAlmostEqual(target_value[2], 0.0)

    def test_spice_greedy_prefers_coherent_sign(self):
        gradients = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ]
        )
        result = spice_greedy_select(
            gradients, budget=2, alpha=1.0, conflict_weight=0.5
        )
        self.assertEqual(result.indices, [0, 1])
        self.assertAlmostEqual(result.conflicts[1], 0.0)

    def test_rademacher_projection_is_seeded(self):
        g = np.arange(20, dtype=float).reshape(4, 5)
        a = rademacher_project(g, 7, seed=9)
        b = rademacher_project(g, 7, seed=9)
        self.assertEqual(a.shape, (4, 7))
        np.testing.assert_array_equal(a, b)

    def test_sparse_cluster_selection(self):
        self.assertEqual(sparse_cluster_ids([0, 0, 0, 1, 1, 2], 0.34), {1, 2})

    def test_greedy_respects_quality_and_conflict(self):
        g = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.5, 0.8, 0.0],
                [0.8, 0.0, -0.6],  # conflicts with protection
                [0.7, 0.0, 0.7],   # marked low quality
            ]
        )
        result = greedy_select(
            g,
            target_gradient=[1.0, 0.0, 0.0],
            protection_gradient=[0.0, 0.0, 1.0],
            budget=3,
            quality_mask=[True, True, True, True, False],
            max_conflict=0.01,
            coverage_weight=1.0,
        )
        self.assertNotIn(3, result.indices)
        self.assertNotIn(4, result.indices)
        self.assertEqual(len(result.indices), 3)
        self.assertGreater(result.vendi, 1.0)


if __name__ == "__main__":
    unittest.main()
