import unittest

from topo_demo import (
    gbps_to_bytes_per_second,
    ring_estimate,
    topology_description,
)


class TopologyAndCollectiveTest(unittest.TestCase):
    def test_line_rate_unit_conversion(self):
        self.assertEqual(gbps_to_bytes_per_second(400), 50e9)

    def test_ring_allreduce_volume_and_steps(self):
        report = ring_estimate(
            "all_reduce",
            payload_bytes=800,
            ranks=8,
            effective_bandwidth_bytes_per_s=100,
        )
        self.assertEqual(report.steps, 14)
        self.assertEqual(report.bytes_sent_per_rank, 1400)
        self.assertEqual(report.bandwidth_time_s, 14)

    def test_reduce_scatter_and_allgather_compose_allreduce(self):
        kwargs = dict(
            payload_bytes=1_000_000,
            ranks=4,
            effective_bandwidth_bytes_per_s=1e9,
            step_latency_s=3e-6,
        )
        reduce_scatter = ring_estimate("reduce_scatter", **kwargs)
        all_gather = ring_estimate("all_gather", **kwargs)
        all_reduce = ring_estimate("all_reduce", **kwargs)
        self.assertEqual(
            reduce_scatter.bytes_sent_per_rank + all_gather.bytes_sent_per_rank,
            all_reduce.bytes_sent_per_rank,
        )
        self.assertAlmostEqual(
            reduce_scatter.total_time_s + all_gather.total_time_s,
            all_reduce.total_time_s,
        )

    def test_startup_cost_counts_ring_steps(self):
        report = ring_estimate(
            "all_reduce",
            payload_bytes=0,
            ranks=8,
            effective_bandwidth_bytes_per_s=1e9,
            step_latency_s=2e-6,
        )
        self.assertAlmostEqual(report.startup_time_s, 28e-6)

    def test_node_is_not_a_network_or_cross_host_label(self):
        description = topology_description("NODE")
        self.assertIn("one NUMA node", description)
        self.assertNotIn("cross-host", description)
        self.assertNotIn("network", description)

    def test_nvlink_family_and_invalid_label(self):
        self.assertEqual(topology_description("NV12"), "bonded set of 12 NVLinks")
        with self.assertRaises(ValueError):
            topology_description("IB")

    def test_invalid_collective_inputs(self):
        with self.assertRaises(ValueError):
            ring_estimate(
                "broadcast",
                payload_bytes=1,
                ranks=2,
                effective_bandwidth_bytes_per_s=1,
            )
        with self.assertRaises(ValueError):
            ring_estimate(
                "all_reduce",
                payload_bytes=1,
                ranks=1,
                effective_bandwidth_bytes_per_s=1,
            )


if __name__ == "__main__":
    unittest.main()
