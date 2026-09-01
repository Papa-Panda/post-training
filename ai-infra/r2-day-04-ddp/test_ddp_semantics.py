import importlib.util
import tempfile
import unittest
from pathlib import Path

from ddp_demo import (
    _run_torch_demo,
    average_rank_gradients,
    distributed_indices,
    replicas_equal,
    ring_allreduce_bytes,
    sgd_step,
)


class DDPSemanticsTest(unittest.TestCase):
    def test_divisible_sampler_is_disjoint_and_complete(self):
        shards = [distributed_indices(12, 3, rank) for rank in range(3)]
        self.assertEqual([len(shard) for shard in shards], [4, 4, 4])
        self.assertEqual(set().union(*map(set, shards)), set(range(12)))
        self.assertTrue(all(set(shards[i]).isdisjoint(shards[j]) for i in range(3) for j in range(i)))

    def test_nondivisible_sampler_padding_is_explicit(self):
        shards = [distributed_indices(5, 2, rank) for rank in range(2)]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(len(flattened), 6)
        self.assertEqual(set(flattened), set(range(5)))
        self.assertEqual(len(flattened) - len(set(flattened)), 1)

    def test_drop_last_is_disjoint(self):
        shards = [distributed_indices(5, 2, rank, drop_last=True) for rank in range(2)]
        self.assertEqual(sum(map(len, shards)), 4)
        self.assertTrue(set(shards[0]).isdisjoint(shards[1]))

    def test_ddp_mean_gradient_matches_equal_global_batch(self):
        local_gradients = [[2.0, -2.0], [4.0, 6.0]]
        self.assertEqual(average_rank_gradients(local_gradients), [3.0, 2.0])

    def test_equal_initial_parameters_and_mean_gradient_preserve_replicas(self):
        initial = [1.0, -1.0]
        mean_gradient = average_rank_gradients([[2.0, 4.0], [4.0, 2.0]])
        replicas = [sgd_step(initial, mean_gradient, 0.1) for _ in range(2)]
        self.assertTrue(replicas_equal(replicas))
        replicas[1][0] += 0.01
        self.assertFalse(replicas_equal(replicas, tolerance=1e-6))

    def test_ring_volume_is_per_rank(self):
        self.assertEqual(ring_allreduce_bytes(800, 8), 1400)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            distributed_indices(10, 0, 0)
        with self.assertRaises(ValueError):
            average_rank_gradients([[1.0], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            ring_allreduce_bytes(1, 1)


class OptionalPyTorchIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_single_process_training_writes_rank_zero_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            _run_torch_demo(str(checkpoint))
            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
