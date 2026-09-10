"""CPU tests for r2-Day15 3D parallelism model.

All numbers are hand-computable; bandwidth figures are theoretical
(theoretical estimate), never claimed as measured benchmarks.
"""
import unittest

from parallel_3d import (
    IB_GBS,
    NVLINK_GBS,
    allreduce_ring_bytes,
    check_topology,
    coords_to_rank,
    dp_group,
    node_of_rank,
    pp_bubble_fraction,
    pp_group,
    pp_p2p_bytes_per_boundary,
    rank_to_coords,
    sp_activation_saving_factor,
    tp_comm_bytes_per_rank_per_layer_microbatch,
    tp_group,
    tp_params_per_rank,
    tp_step_bytes_per_rank,
    tp_step_seconds_theoretical,
)

T, P, D = 8, 4, 2
WORLD = 64


class TestTopology(unittest.TestCase):
    def test_check_topology_ok(self):
        topo = check_topology(WORLD, T, P, D)
        self.assertEqual(topo, {"world_size": 64, "T": 8, "P": 4, "D": 2})

    def test_check_topology_mismatch_raises(self):
        with self.assertRaises(ValueError):
            check_topology(64, 8, 4, 3)

    def test_rank_coords_roundtrip_all_64(self):
        for rank in range(WORLD):
            d_rank, p_rank, t_rank = rank_to_coords(rank, T, P, D)
            self.assertEqual(coords_to_rank(d_rank, p_rank, t_rank, T, P, D), rank)

    def test_rank_formula_spot(self):
        # rank = (d*P + p)*T + t
        self.assertEqual(rank_to_coords(0, T, P, D), (0, 0, 0))
        self.assertEqual(rank_to_coords(7, T, P, D), (0, 0, 7))
        self.assertEqual(rank_to_coords(8, T, P, D), (0, 1, 0))
        self.assertEqual(rank_to_coords(32, T, P, D), (1, 0, 0))
        self.assertEqual(rank_to_coords(63, T, P, D), (1, 3, 7))


class TestGroups(unittest.TestCase):
    def test_tp_group_contiguous(self):
        g = tp_group(3, T, P, D)
        self.assertEqual(g, list(range(8)))  # ranks 0..7

    def test_tp_group_other_stage(self):
        g = tp_group(20, T, P, D)  # rank 20 -> (d0, p2, t4)
        self.assertEqual(g, list(range(16, 24)))

    def test_pp_group_spans_stages(self):
        g = pp_group(5, T, P, D)  # (d0, t5) across p=0..3
        self.assertEqual(g, [5, 13, 21, 29])

    def test_dp_group_two_ranks(self):
        g = dp_group(0, T, P, D)  # (t0, p0) across d=0,1
        self.assertEqual(g, [0, 32])

    def test_groups_partition_world(self):
        seen = set()
        for rank in range(WORLD):
            seen.update(tp_group(rank, T, P, D))
        self.assertEqual(len(seen), WORLD)


class TestNodeMapping(unittest.TestCase):
    def test_tp_group_intra_node(self):
        # (d0, p0) TP group ranks 0-7 all on node 0
        nodes = {node_of_rank(r) for r in tp_group(0, T, P, D)}
        self.assertEqual(nodes, {0})

    def test_pp_group_crosses_nodes(self):
        nodes = [node_of_rank(r) for r in pp_group(0, T, P, D)]
        self.assertEqual(nodes, [0, 1, 2, 3])  # point-to-point across nodes

    def test_64_rank_node_table(self):
        self.assertEqual(node_of_rank(0), 0)
        self.assertEqual(node_of_rank(7), 0)
        self.assertEqual(node_of_rank(8), 1)
        self.assertEqual(node_of_rank(31), 3)
        self.assertEqual(node_of_rank(32), 4)
        self.assertEqual(node_of_rank(63), 7)


class TestTraffic(unittest.TestCase):
    def test_ring_formula_matches_day03(self):
        # 2*(N-1)/N * M with M = 67,108,864, N = 8
        m = 4096 * 8192 * 2
        self.assertEqual(m, 67108864)
        got = allreduce_ring_bytes(m, 8)
        self.assertAlmostEqual(got, 2 * 7 / 8 * m, places=3)
        self.assertAlmostEqual(got / 1e6, 117.4, places=1)  # ~117.4 MB

    def test_tp_traffic_handcalc(self):
        # 4 all-reduces/layer/microbatch, b=1 S=4096 H=8192 T=8
        got = tp_comm_bytes_per_rank_per_layer_microbatch(1, 4096, 8192, 8)
        self.assertAlmostEqual(got / 1e6, 469.8, places=1)  # ~469.8 MB

    def test_tp_step_80_layers(self):
        total = tp_step_bytes_per_rank(1, 4096, 8192, 80, 8)
        self.assertAlmostEqual(total / 1e9, 37.6, places=1)  # ~37.6 GB

    def test_tp_nvlink_vs_ib_theoretical(self):
        total = tp_step_bytes_per_rank(1, 4096, 8192, 80, 8)
        t_nv = tp_step_seconds_theoretical(total, NVLINK_GBS)
        t_ib = tp_step_seconds_theoretical(total, IB_GBS)
        self.assertAlmostEqual(t_nv, 0.0418, places=3)   # ~41.8 ms
        self.assertAlmostEqual(t_ib, 0.752, places=2)    # ~0.75 s
        self.assertAlmostEqual(t_ib / t_nv, 18.0, places=0)  # 18x slower

    def test_pp_bubble(self):
        self.assertAlmostEqual(pp_bubble_fraction(4, 8), 3 / 11, places=6)
        self.assertAlmostEqual(pp_bubble_fraction(4, 32), 3 / 35, places=6)
        # more microbatches -> smaller bubble
        self.assertLess(pp_bubble_fraction(4, 32), pp_bubble_fraction(4, 8))

    def test_pp_p2p_boundary(self):
        got = pp_p2p_bytes_per_boundary(1, 4096, 8192)
        self.assertAlmostEqual(got / 1e6, 134.2, places=1)  # ~134.2 MB

    def test_sp_saving_factor(self):
        self.assertEqual(sp_activation_saving_factor(8), 8)

    def test_tp_params_per_rank(self):
        self.assertAlmostEqual(tp_params_per_rank(70e9, 8), 8.75e9)


if __name__ == "__main__":
    unittest.main()
