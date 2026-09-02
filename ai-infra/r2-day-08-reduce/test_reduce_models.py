import math
import unittest

from reduce_models import (
    atomic_reduce,
    build_report,
    shared_tree_reduce,
    warp_shuffle_reduce,
)


class TestReductionModels(unittest.TestCase):
    def test_all_variants_compute_same_sum(self):
        values = [float((i % 11) - 5) for i in range(1000)]
        expected = math.fsum(values)
        for reduce_fn in (atomic_reduce, shared_tree_reduce, warp_shuffle_reduce):
            self.assertEqual(reduce_fn(values, 256).value, expected)

    def test_hand_sum_one_through_eight(self):
        values = [float(i) for i in range(1, 9)]
        self.assertEqual(warp_shuffle_reduce(values, 32).value, 36.0)

    def test_sixty_four_element_resource_counts(self):
        values = [1.0] * 64
        atomic = atomic_reduce(values, 64)
        shared = shared_tree_reduce(values, 64)
        shuffle = warp_shuffle_reduce(values, 64)

        self.assertEqual(atomic.global_atomic_updates, 64)
        self.assertEqual(shared.global_atomic_updates, 1)
        self.assertEqual(shared.block_barriers, 7)  # load + log2(64) stages
        self.assertEqual(shared.shared_writes, 64)
        self.assertEqual(shuffle.global_atomic_updates, 1)
        self.assertEqual(shuffle.block_barriers, 1)
        self.assertEqual(shuffle.shared_writes, 2)  # one partial per warp
        self.assertEqual(shuffle.warp_shuffle_instructions, 15)  # 3 warps x 5

    def test_partial_last_block_is_zero_padded(self):
        values = [1.0] * 70
        for reduce_fn in (shared_tree_reduce, warp_shuffle_reduce):
            result = reduce_fn(values, 64)
            self.assertEqual(result.value, 70.0)
            self.assertEqual(result.blocks, 2)
            self.assertEqual(result.global_atomic_updates, 2)

    def test_empty_input(self):
        for reduce_fn in (atomic_reduce, shared_tree_reduce, warp_shuffle_reduce):
            result = reduce_fn([], 64)
            self.assertEqual(result.value, 0.0)
            self.assertEqual(result.blocks, 0)

    def test_invalid_block_sizes_rejected(self):
        for block_size in (0, 16, 48, 2048):
            with self.assertRaises(ValueError):
                warp_shuffle_reduce([1.0], block_size)

    def test_report_executes_and_uses_all_variants(self):
        report = build_report(range(1, 65), 64)
        variants = report["variants"]
        self.assertEqual(set(variants), {
            "global_atomic_per_element",
            "shared_tree_per_block",
            "warp_shuffle_per_block",
        })
        for stats in variants.values():
            self.assertEqual(stats["value"], 2080.0)


if __name__ == "__main__":
    unittest.main()
