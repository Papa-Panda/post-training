"""CPU tests for r2-day-12 profiling models.

All expectations are hand-computed theoretical values (see README/NOTES);
no CUDA GPU, nsys, or ncu is needed or used.
"""

import unittest

import roofline_sol as rs
import timeline_model as tm
import nsys_ncu_recipes as recipes


class TestRooflineSol(unittest.TestCase):
    def test_ridge_fp32_h100(self):
        # 67 TFLOP/s / 3.35 TB/s = 20 FLOP/byte (r2-day-06 value).
        self.assertAlmostEqual(rs.ridge_point(rs.H100_SXM), 20.0, places=9)

    def test_ridge_fp16_tensor(self):
        # 989 / 3.35: Tensor Cores push the ridge ~15x higher.
        self.assertAlmostEqual(rs.ridge_point(rs.H100_SXM_FP16),
                               989.0 / 3.35, places=6)

    def test_ai_vecadd_n8(self):
        kt = rs.vecadd_traffic(8)
        self.assertEqual((kt.flops, kt.bytes), (8.0, 96.0))
        self.assertAlmostEqual(rs.arithmetic_intensity(kt), 1.0 / 12.0,
                               places=12)

    def test_ai_is_scale_free(self):
        # Same operator at N=2**20 (r2-day-06's N) shares the AI.
        self.assertAlmostEqual(
            rs.arithmetic_intensity(rs.vecadd_traffic(2 ** 20)),
            1.0 / 12.0, places=12)

    def test_bound_memory(self):
        self.assertEqual(rs.bound_class(rs.H100_SXM, rs.vecadd_traffic(8)),
                         "memory")

    def test_sol_report_vecadd_n8(self):
        r = rs.sol_report(rs.H100_SXM, rs.vecadd_traffic(8))
        self.assertEqual(r["bound"], "memory")
        # 0.2792 TFLOP/s achievable vs 67 TFLOP/s peak.
        self.assertAlmostEqual(r["achievable_flops"], 3.35e12 / 12.0,
                               delta=100.0)
        self.assertAlmostEqual(r["sol_flops_pct"], 0.4166667, places=4)
        # Memory-bound: byte rate saturates the HBM roof.
        self.assertAlmostEqual(r["sol_bandwidth_pct"], 100.0, places=6)
        # 96 B / 3.35 TB/s.
        self.assertAlmostEqual(r["t_min_s"], 96.0 / 3.35e12, places=22)

    def test_compute_bound_contrast(self):
        # AI=100 > ridge=20: the opposite SOL shape.
        r = rs.sol_report(rs.H100_SXM, rs.KernelTraffic(800.0, 8.0))
        self.assertEqual(r["bound"], "compute")
        self.assertAlmostEqual(r["sol_flops_pct"], 100.0, places=9)
        self.assertAlmostEqual(r["sol_bandwidth_pct"], 20.0, places=9)
        # Compute-bound t_min comes from the FLOP roof, not the byte roof.
        self.assertAlmostEqual(r["t_min_s"], 800.0 / 67e12, places=20)

    def test_t_min_picks_slower_roof(self):
        mem = rs.sol_report(rs.H100_SXM, rs.vecadd_traffic(8))
        self.assertAlmostEqual(mem["t_min_s"], 96.0 / 3.35e12, places=22)
        cmp_ = rs.sol_report(rs.H100_SXM, rs.KernelTraffic(800.0, 8.0))
        self.assertAlmostEqual(cmp_["t_min_s"], 800.0 / 67e12, places=20)


class TestTimelineModel(unittest.TestCase):
    def setUp(self):
        self.d = tm.softmax_demo()  # n=2**20, launch=5us (illustrative)

    def test_eager_span(self):
        # 4*5 us launch + 25165824 B / 3.35 TB/s of kernel (roofline floor).
        self.assertAlmostEqual(self.d["eager"]["span_us"], 27.512186, places=4)

    def test_eager_busy_fraction(self):
        self.assertAlmostEqual(self.d["eager"]["busy_fraction"], 0.273049,
                               places=4)

    def test_fused_span(self):
        # 1*5 us launch + 8388608 B / 3.35 TB/s of kernel (roofline floor).
        self.assertAlmostEqual(self.d["fused"]["span_us"], 7.504062, places=4)

    def test_fused_busy_fraction(self):
        self.assertAlmostEqual(self.d["fused"]["busy_fraction"], 0.333694,
                               places=4)

    def test_model_speedup(self):
        # 27.512186 / 7.504062: traffic saved 3x plus 3 launches saved.
        # Theoretical model only -- NOT a benchmark.
        self.assertAlmostEqual(self.d["model_speedup"], 3.666306, places=4)

    def test_gap_attribution(self):
        eager_gaps = self.d["eager"]["gaps"]
        self.assertEqual(len(eager_gaps), 4)
        for g in eager_gaps:
            self.assertIn("cudaLaunchKernel", g["attributed_to"])
            self.assertAlmostEqual(g["duration_us"], 5.0, places=9)
        fused_gaps = self.d["fused"]["gaps"]
        self.assertEqual(len(fused_gaps), 1)

    def test_span_is_sum_of_durations(self):
        for name in ("eager", "fused"):
            r = self.d[name]
            self.assertAlmostEqual(
                r["span_us"], r["gpu_busy_us"] + r["gpu_idle_us"], places=9)

    def test_custom_launch_overhead_propagates(self):
        d2 = tm.softmax_demo(launch_overhead_us=10.0)
        self.assertAlmostEqual(d2["eager"]["span_us"],
                               4 * 10.0 + d2["eager_kernel_us"], places=6)
        self.assertAlmostEqual(d2["fused"]["span_us"],
                               10.0 + d2["fused_kernel_us"], places=6)


class TestRecipes(unittest.TestCase):
    def test_nsys_recipe_argv(self):
        argv = recipes.get_recipe("systems_train_iter")["argv"]
        self.assertEqual(argv[0], "nsys")
        self.assertIn("profile", argv)
        self.assertIn("--trace=cuda,nvtx,osrt", argv)

    def test_ncu_sol_recipe_sections(self):
        argv = recipes.get_recipe("compute_sol")["argv"]
        self.assertEqual(argv[0], "ncu")
        self.assertIn("SpeedOfLight", argv)
        self.assertIn("SpeedOfLight_RooflineChart", argv)

    def test_unknown_recipe_raises(self):
        with self.assertRaises(KeyError):
            recipes.get_recipe("nope")

    def test_tools_probe_shape(self):
        present = recipes.tools_present()
        self.assertEqual(set(present), {"nsys", "ncu"})
        self.assertTrue(all(isinstance(v, bool) for v in present.values()))
        # This sandbox has no GPU tooling; both must be absent here.
        self.assertFalse(present["nsys"])
        self.assertFalse(present["ncu"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
