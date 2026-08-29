import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "code"))

from model_aware_curation import (  # noqa: E402
    conflict_scores,
    cosine_scores,
    gradient_vendi,
    greedy_select,
    rademacher_project,
    sparse_cluster_ids,
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
