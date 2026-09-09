"""CPU tests for r2-Day14. All hand-computed numbers are asserted exactly."""

import math
import unittest

from zero_memory import (
    all_gather_sim,
    collective_bytes,
    ddp_step_comm_bytes,
    fsdp_peak_bytes,
    fmt_gb,
    model_state_bytes_per_param,
    reduce_scatter_sim,
    shard_vector,
    sharded_state_bytes,
    zero3_step_comm_bytes,
)

GB = 10 ** 9  # decimal GB, matching vendor specs (H100 80GB, roadmap 14GB)


class TestLedgers(unittest.TestCase):
    def test_7b_roadmap_ledger(self):
        P = 7_000_000_000
        s = sharded_state_bytes(P, 0, 1, "roadmap")
        self.assertEqual(s["params"], 14 * GB)   # fp16 params
        self.assertEqual(s["grads"], 14 * GB)    # fp16 grads
        self.assertEqual(s["opt"], 56 * GB)      # fp32 Adam m+v
        self.assertEqual(s["total"], 84 * GB)
        self.assertGreater(s["total"], 80 * GB)  # does NOT fit one 80GB card

    def test_7b_mixed_adam_16_bytes_per_param(self):
        s = sharded_state_bytes(7_000_000_000, 0, 1, "mixed_adam")
        self.assertEqual(s["total"], 7_000_000_000 * 16)
        self.assertEqual(s["total"], 112 * GB)

    def test_toy_stage_bytes(self):
        # P=100, fp32 Adam (16 B/param -> 1600 B), N=4.
        expect = {0: 1600.0, 1: 1000.0, 2: 700.0, 3: 400.0}
        for stage, total in expect.items():
            with self.subTest(stage=stage):
                s = sharded_state_bytes(100, stage, 4, "fp32_adam")
                self.assertAlmostEqual(s["total"], total)

    def test_toy_stage3_breakdown(self):
        s = sharded_state_bytes(100, 3, 4, "fp32_adam")
        self.assertAlmostEqual(s["params"], 100.0)  # 400/4
        self.assertAlmostEqual(s["grads"], 100.0)    # 400/4
        self.assertAlmostEqual(s["opt"], 200.0)      # 800/4

    def test_stage_monotone_and_ddp_equals_replicate(self):
        totals = [sharded_state_bytes(1000, st, 8, "mixed_adam")["total"]
                  for st in range(4)]
        self.assertEqual(totals[0], 1000 * 16)
        self.assertTrue(totals[0] > totals[1] > totals[2] > totals[3])
        self.assertAlmostEqual(totals[3], 1000 * 16 / 8)


class TestCollectives(unittest.TestCase):
    def test_ring_formulas(self):
        M, N = 400.0, 4
        self.assertAlmostEqual(collective_bytes(M, N, "allreduce"), 600.0)
        self.assertAlmostEqual(collective_bytes(M, N, "allgather"), 300.0)
        self.assertAlmostEqual(collective_bytes(M, N, "reducescatter"), 300.0)

    def test_zero3_is_1p5x_ddp(self):
        M, N = 400.0, 4  # toy param/grad bytes
        ddp = ddp_step_comm_bytes(M, N)
        z3 = zero3_step_comm_bytes(M, M, N)
        self.assertAlmostEqual(ddp, 600.0)
        self.assertAlmostEqual(z3, 900.0)
        self.assertAlmostEqual(z3 / ddp, 1.5)

    def test_single_rank_no_comm(self):
        self.assertEqual(collective_bytes(1000.0, 1, "allreduce"), 0.0)
        self.assertEqual(ddp_step_comm_bytes(1000.0, 1), 0.0)


class TestShardSemantics(unittest.TestCase):
    def test_shard_allgather_roundtrip(self):
        vec = list(range(12))
        shards = [shard_vector(vec, 3, r) for r in range(3)]
        self.assertEqual(shards[1], [4, 5, 6, 7])
        self.assertEqual(all_gather_sim(shards), vec)

    def test_reduce_scatter_sums_then_shards(self):
        grads = [[1.0] * 8, [2.0] * 8, [3.0] * 8, [4.0] * 8]
        rs = reduce_scatter_sim(grads, 4)
        self.assertEqual(len(rs), 4)
        self.assertEqual(rs[0], [10.0, 10.0])
        self.assertEqual(rs[3], [10.0, 10.0])
        # reassembled sum equals the true total gradient
        self.assertEqual(all_gather_sim(rs), [10.0] * 8)

    def test_shard_requires_divisible(self):
        with self.assertRaises(ValueError):
            shard_vector([1, 2, 3], 2, 0)


class TestFsdpPeak(unittest.TestCase):
    def test_7b_8rank_per_block_peak(self):
        P, L, N = 7_000_000_000, 32, 8
        peak = fsdp_peak_bytes(P, P // L, N, "mixed_adam")
        sharded = 7_000_000_000 * 16 / 8          # 14 GiB
        one_block = (P // L) * 2                   # fp16 block params
        self.assertAlmostEqual(peak, sharded + one_block)
        self.assertLess(peak, 15 * GB)            # fits comfortably per rank
        self.assertGreater(peak, 14 * GB)

    def test_peak_matches_ddp_form_intuition(self):
        # r2-Day02's (P-b)/G + b intuition: peak = sharded part + one full block.
        P, b, N = 1000, 250, 4
        peak = fsdp_peak_bytes(P, b, N, "fp32_adam")
        self.assertAlmostEqual(peak, 1000 * 16 / 4 + b * 4)


if __name__ == "__main__":
    unittest.main()
