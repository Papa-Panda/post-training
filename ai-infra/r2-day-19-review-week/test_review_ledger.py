"""r2-Day19 复盘测试：断言第二层各课手算锚点在总账里自洽。

每个断言的期望值都直接来自对应 r2 day 的 README/NOTES 手算，
不是本课新编的数字；本课只验证"收拢公式"没有抄错。
纯 CPU。
"""

import math
import unittest

from review_ledger import (
    GB,
    amp_state_ledger,
    checkpoint_ledger,
    checkpoint_worth_opening,
    comm_time_s,
    dcp_sharded_vs_gather,
    end_to_end_7b,
    fsdp_per_rank_gb,
    kv_bytes_per_token_per_layer,
    model_state_ledger,
    pp_bubble,
    ring_allreduce_traffic,
    tp_traffic,
    young_daly,
    zero3_comm_gb,
)

P7B = 7e9


class TestDay14Ledger(unittest.TestCase):
    def test_7b_model_state(self):
        r = model_state_ledger(P7B)
        self.assertAlmostEqual(r["params_gb"], 14.0)   # Day14：fp16
        self.assertAlmostEqual(r["grads_gb"], 14.0)    # Day14：fp16
        self.assertAlmostEqual(r["opt_gb"], 56.0)      # Day14：Adam fp32 m+v
        self.assertAlmostEqual(r["total_gb"], 84.0)
        self.assertEqual(r["bytes_per_param"], 12)

    def test_single_80gb_does_not_fit(self):
        # Day14 结论：84 > 80，单卡装不下
        self.assertGreater(model_state_ledger(P7B)["total_gb"], 80.0)


class TestDay16Amp(unittest.TestCase):
    def test_7b_amp_ledger(self):
        r = amp_state_ledger(P7B)
        self.assertAlmostEqual(r["total_gb"], 112.0)  # Day16：14+14+28+56
        self.assertEqual(r["bytes_per_param"], 16)

    def test_amp_equals_pure_fp32(self):
        # Day16：AMP 16 B/param = 纯 fp32 训练 4+4+8=16，BF16 省范围不省字节
        pure_fp32 = model_state_ledger(P7B, bytes_param=4, bytes_grad=4,
                                       bytes_opt=8)
        self.assertAlmostEqual(amp_state_ledger(P7B)["total_gb"],
                               pure_fp32["total_gb"])


class TestDay17Sharding(unittest.TestCase):
    def test_fsdp_per_rank(self):
        # Day17：7B/N=8 -> 10.5 GB/rank
        self.assertAlmostEqual(fsdp_per_rank_gb(84.0, 8), 10.5)

    def test_zero3_1p5x_ddp(self):
        # Day17：DDP 24.5 GB/step（7B fp16 grads 14GB ring N=8）
        # -> ZeRO-3 36.75 GB/step
        ddp = ring_allreduce_traffic(14 * GB, 8) / GB
        self.assertAlmostEqual(ddp, 24.5, places=1)
        self.assertAlmostEqual(zero3_comm_gb(ddp), 36.75, places=1)


class TestDay15Comm(unittest.TestCase):
    def test_ring_formula_anchor(self):
        # Day15 手算锚点：M=67,108,864 B，N=8 -> ~117.4 MB
        got = ring_allreduce_traffic(67108864, 8) / 1e6
        self.assertAlmostEqual(got, 117.4, places=1)

    def test_tp_traffic_anchor(self):
        per_ar, per_layer, total = tp_traffic(1, 4096, 8192, 8, 80)
        self.assertAlmostEqual(per_ar / 1e6, 117.4, places=1)    # Day15
        self.assertAlmostEqual(per_layer / 1e6, 469.8, places=1)  # 4x
        self.assertAlmostEqual(total / 1e9, 37.6, places=1)       # 80 层

    def test_tp_time_ratio(self):
        # Day15：NVLink 41.8ms vs IB 0.75s，约 18x（theoretical）
        _, _, total = tp_traffic(1, 4096, 8192, 8, 80)
        nv = comm_time_s(total, 900)
        ib = comm_time_s(total, 50)
        self.assertAlmostEqual(nv * 1e3, 41.8, places=0)
        self.assertAlmostEqual(ib, 0.75, places=1)
        self.assertAlmostEqual(ib / nv, 18.0, places=0)

    def test_pp_bubble(self):
        self.assertAlmostEqual(pp_bubble(4, 8), 3 / 11)   # Day15：27.3%
        self.assertAlmostEqual(pp_bubble(4, 32), 3 / 35)  # Day15：8.6%


class TestDay18Checkpoint(unittest.TestCase):
    def test_7b_checkpoint_ledger(self):
        r = checkpoint_ledger(P7B)
        self.assertAlmostEqual(r["total_gb"], 70.0)  # Day18：14+56
        self.assertEqual(r["bytes_per_param"], 10)

    def test_ckpt_diff_is_grads(self):
        # Day18 vs Day14 口径差：84 - 70 = 14 = grads（不存）
        diff = (model_state_ledger(P7B)["total_gb"]
                - checkpoint_ledger(P7B)["total_gb"])
        self.assertAlmostEqual(diff, model_state_ledger(P7B)["grads_gb"])

    def test_dcp_vs_gather(self):
        r = dcp_sharded_vs_gather(P7B, 64)
        self.assertAlmostEqual(r["per_rank_gb"], 70 / 64)        # 1.09375
        self.assertAlmostEqual(r["gather_extra_gb"], 63 / 64 * 70)  # 68.90625

    def test_young_daly_toy(self):
        r = young_daly(10, 1000)
        self.assertAlmostEqual(r["tau_s"], math.sqrt(20000), places=1)
        self.assertAlmostEqual(r["waste"], 0.1414, places=3)  # Day18：14.14%

    def test_young_daly_real(self):
        r = young_daly(60, 10 * 3600)
        self.assertAlmostEqual(r["tau_s"] / 60, 34.6, places=0)  # ~34.6 min
        self.assertAlmostEqual(r["waste"], 0.0577, places=3)     # ~5.8%

    def test_young_daly_async(self):
        r = young_daly(0.055, 10 * 3600)
        self.assertAlmostEqual(r["tau_s"], 62.9, places=0)  # ~63 s
        self.assertAlmostEqual(r["waste"], 0.0017, places=3)  # ~0.17%

    def test_short_train_not_worth(self):
        # Day18 讨论题：T_train < tau* -> 不赚
        r = checkpoint_worth_opening(60, 10 * 3600, 1000)
        self.assertFalse(r["worth"])
        r2 = checkpoint_worth_opening(60, 10 * 3600, 2 * 3600)
        self.assertTrue(r2["worth"])


class TestDay13Kv(unittest.TestCase):
    def test_kv_table(self):
        self.assertEqual(kv_bytes_per_token_per_layer("mha"), 16384)
        self.assertEqual(kv_bytes_per_token_per_layer("gqa8"), 4096)
        self.assertEqual(kv_bytes_per_token_per_layer("mla"), 1152)
        self.assertEqual(kv_bytes_per_token_per_layer("mqa"), 512)

    def test_mla_ratio(self):
        ratio = (kv_bytes_per_token_per_layer("mha")
                 / kv_bytes_per_token_per_layer("mla"))
        self.assertAlmostEqual(ratio, 14.2, places=1)  # Day13：14.2x


class TestEndToEnd(unittest.TestCase):
    def test_end_to_end_7b(self):
        r = end_to_end_7b()
        self.assertAlmostEqual(r["per_rank_state_gb"], 84 / 64)   # 1.3125
        self.assertAlmostEqual(r["per_rank_ckpt_gb"], 70 / 64)    # 1.09375
        self.assertAlmostEqual(r["tp_total_gb"], 37.6, places=1)
        self.assertAlmostEqual(r["tp_nvlink_ms"], 41.8, places=0)
        self.assertAlmostEqual(r["tp_ib_s"], 0.75, places=1)
        self.assertAlmostEqual(r["tau_star_min"], 34.6, places=0)
        self.assertAlmostEqual(r["waste"], 0.0577, places=3)
        self.assertTrue(r["worth_2h"])
        mha_gib, mla_gib = r["mla_kv_gib_vs_mha"]
        self.assertAlmostEqual(mha_gib, 64.0, places=0)  # Day13：64 GiB
        self.assertAlmostEqual(mla_gib, 4.5, places=1)   # Day13：4.5 GiB


if __name__ == "__main__":
    unittest.main()
