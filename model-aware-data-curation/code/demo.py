#!/usr/bin/env python3
"""Run a tiny, deterministic model-aware curation example."""

import numpy as np

from model_aware_curation import (
    gradient_vendi,
    greedy_select,
    sparse_cluster_ids,
)


# Rows stand in for already-projected per-sample loss gradients.
# 0/1 are target-aligned duplicates; 2 adds coverage; 3 conflicts with safety;
# 4 is low quality; 5 is a second safe direction.
GRADIENTS = np.array(
    [
        [1.00, 0.00, 0.00],
        [0.98, 0.02, 0.00],
        [0.55, 0.83, 0.00],
        [0.70, 0.00, -0.70],
        [0.60, 0.00, 0.80],
        [0.50, -0.86, 0.00],
    ]
)
TARGET = np.array([1.0, 0.0, 0.0])
PROTECT = np.array([0.0, 0.0, 1.0])
QUALITY = [True, True, True, True, False, True]


if __name__ == "__main__":
    result = greedy_select(
        GRADIENTS,
        target_gradient=TARGET,
        protection_gradient=PROTECT,
        budget=3,
        quality_mask=QUALITY,
        max_conflict=0.05,
        target_weight=0.5,
        coverage_weight=2.0,
    )
    print("selected indices:", result.indices)
    print("selected G-Vendi:", round(result.vendi, 4))
    print("full-pool G-Vendi:", round(gradient_vendi(GRADIENTS), 4))

    # Cluster IDs could come from k-means over a real gradient datastore.
    print("sparse cluster IDs:", sparse_cluster_ids([0, 0, 0, 1, 1, 2], 0.34))
