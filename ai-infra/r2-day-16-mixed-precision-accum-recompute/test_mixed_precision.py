#!/usr/bin/env python3
"""r2-Day16 测试：断言 README / NOTES 里出现的全部手算数字。

纯 CPU、可执行；不依赖 torch / numpy / CUDA。
"""
import math
import unittest

from mixed_precision import (
    amp_memory_ledger_gb,
    checkpoint_activation_memory_mb,
    ddp_comm_per_sample,
    dtype_bits,
    dtype_max,
    dynamic_loss_scale_update,
    fp16_underflow_demo,
    grad_accum,
    in_fp16_range,
    machine_epsilon,
    min_normal,
    naive_activation_memory_mb,
    optimal_checkpoint_interval,
    pure_fp32_memory_ledger_gb,
    recompute_extra_compute_frac,
    round_to_bf16,
    unit_roundoff,
)


class TestDtypeTable(unittest.TestCase):
    def test_dtype_bit_layout(self):
        # 手算 A：位布局 1+8+23 / 1+8+7 / 1+5+10
        self.assertEqual(dtype_bits("fp32"), (1, 8, 23))
        self.assertEqual(dtype_bits("bf16"), (1, 8, 7))
        self.assertEqual(dtype_bits("fp16"), (1, 5, 10))

    def test_machine_epsilon(self):
        # eps = 2**(-显式尾数位)：0.0078 / 9.77e-4 / 1.19e-7
        self.assertAlmostEqual(machine_epsilon("bf16"), 0.0078125, places=12)
        self.assertAlmostEqual(machine_epsilon("bf16"), 0.0078, places=4)
        self.assertAlmostEqual(machine_epsilon("fp16"), 2.0 ** -10, places=15)
        self.assertAlmostEqual(machine_epsilon("fp16"), 9.77e-4, places=6)
        self.assertAlmostEqual(machine_epsilon("fp32"), 2.0 ** -23, places=20)
        self.assertAlmostEqual(machine_epsilon("fp32"), 1.19e-7, places=9)

    def test_unit_roundoff(self):
        # u = eps/2：BF16 约 0.0039
        self.assertAlmostEqual(unit_roundoff("bf16"), 2.0 ** -8, places=15)
        self.assertAlmostEqual(unit_roundoff("bf16"), 0.0039, places=4)
        self.assertAlmostEqual(unit_roundoff("fp16"), 2.0 ** -11, places=15)

    def test_min_normal(self):
        # 最小正规格数：FP32/BF16 2^-126 ≈ 1.18e-38；FP16 2^-14 ≈ 6.10e-5
        self.assertAlmostEqual(min_normal("fp32"), 2.0 ** -126, places=40 - 30)
        self.assertAlmostEqual(min_normal("bf16"), 2.0 ** -126, places=10)
        self.assertAlmostEqual(min_normal("bf16"), 1.18e-38, places=40)
        self.assertEqual(min_normal("fp16"), 2.0 ** -14)
        self.assertAlmostEqual(min_normal("fp16"), 6.10e-5, places=7)

    def test_dtype_max(self):
        # 最大值：FP32 ≈ 3.40e38；BF16 ≈ 3.39e38；FP16 = 65504
        self.assertEqual(dtype_max("fp16"), 65504.0)
        self.assertAlmostEqual(dtype_max("bf16"), 3.3895313892515355e38, delta=1e30)
        self.assertAlmostEqual(dtype_max("bf16"), 3.39e38, delta=0.005e38)
        self.assertAlmostEqual(dtype_max("fp32"), 3.4028234663852886e38, delta=1e30)
        self.assertAlmostEqual(dtype_max("fp32"), 3.40e38, delta=0.005e38)


class TestUnderflowAndLossScaling(unittest.TestCase):
    def test_fp16_underflow_and_loss_scale(self):
        # 手算 B：g=3.0e-5 < 6.10e-5 下溢为 0；s=2^15=32768 -> 0.98304 可表示
        demo = fp16_underflow_demo(grad=3.0e-5, scale=2 ** 15)
        self.assertEqual(demo["scale"], 32768)
        self.assertFalse(demo["orig_in_fp16_range"])
        self.assertAlmostEqual(demo["scaled_grad"], 0.98304, places=9)
        self.assertTrue(demo["scaled_in_fp16_range"])

    def test_bf16_no_loss_scaling_needed(self):
        # BF16 最小正规格数 2^-126 远小于 3e-5：同一梯度在 BF16 下无需 loss scaling
        self.assertLess(min_normal("bf16"), 3.0e-5)
        self.assertLess(3.0e-5, dtype_max("bf16"))

    def test_dynamic_loss_scale_grow(self):
        # 缩放后仍在 fp16 范围内：scale 翻倍，不跳步
        new_scale, skip = dynamic_loss_scale_update(32768.0, [3.0e-5])
        self.assertEqual(new_scale, 65536.0)
        self.assertFalse(skip)

    def test_dynamic_loss_scale_backoff_skip(self):
        # 3.0*32768=98304 > 65504 上溢：scale 减半并跳过本步
        new_scale, skip = dynamic_loss_scale_update(32768.0, [3.0])
        self.assertEqual(new_scale, 16384.0)
        self.assertTrue(skip)
        self.assertFalse(in_fp16_range(3.0 * 32768.0))


class TestBf16Rounding(unittest.TestCase):
    def test_bf16_rounding_loses_tiny_update(self):
        # 1.0 + 1e-4：更新量 < u/2 (2^-8/2) -> 舍入回 1.0，更新丢失
        self.assertEqual(round_to_bf16(1.0 + 1e-4), 1.0)

    def test_bf16_rounding_keeps_visible_update(self):
        # 1.0 + 0.01：大于半个 spacing -> 落到 1 + 2^-7 = 1.0078125
        self.assertAlmostEqual(round_to_bf16(1.0 + 0.01), 1.0078125, places=12)

    def test_bf16_round_half_to_even(self):
        # 恰为半个 spacing：偶数舍入 -> 1.0；1.5 个 spacing：向偶数 2 进 -> 1.015625
        self.assertEqual(round_to_bf16(1.0 + 2.0 ** -8), 1.0)
        self.assertAlmostEqual(round_to_bf16(1.0 + 3.0 * 2.0 ** -8), 1.015625, places=12)


class TestGradAccum(unittest.TestCase):
    def test_grad_accum_sum_equivalence(self):
        # 手算 C：b=2, m=4 -> B_eff=8；micro 和 [0.3,0.7,1.1,1.5] 总和 3.6
        grads = [0.1 * i for i in range(1, 9)]
        r = grad_accum(grads, m=4)
        self.assertEqual((r["b"], r["m"], r["B_eff"]), (2, 4, 8))
        for got, want in zip(r["micro_sums"], [0.3, 0.7, 1.1, 1.5]):
            self.assertAlmostEqual(got, want, places=10)
        self.assertAlmostEqual(r["total"], 3.6, places=10)
        self.assertAlmostEqual(r["full_sum"], 3.6, places=10)
        self.assertAlmostEqual(r["total"], r["full_sum"], places=12)

    def test_grad_accum_mean_reduction_needs_divide_by_m(self):
        # mean reduction：microbatch 内平均后再累加，必须除以 m=4 才等于全 batch 均值 0.45
        grads = [0.1 * i for i in range(1, 9)]
        r = grad_accum(grads, m=4)
        self.assertAlmostEqual(r["full_mean"], 0.45, places=12)
        self.assertAlmostEqual(r["accum_mean"], 0.45, places=12)
        raw_micro_mean_sum = sum(s / 2 for s in r["micro_sums"])
        self.assertAlmostEqual(raw_micro_mean_sum, 1.8, places=10)  # 不除 m 会大 4 倍

    def test_grad_accum_ddp_comm_per_sample(self):
        # DDP 每 optimizer step 只做一次 all-reduce：每样本分摊通信量 ÷ m
        per_b2 = ddp_comm_per_sample(14.0e9, 2)
        per_b8 = ddp_comm_per_sample(14.0e9, 8)
        self.assertAlmostEqual(per_b8, per_b2 / 4.0, places=6)


class TestCheckpoint(unittest.TestCase):
    def test_optimal_checkpoint_interval(self):
        # 手算 D：L=64 -> 最优 k = sqrt(64) = 8
        self.assertEqual(optimal_checkpoint_interval(64), 8)

    def test_checkpoint_memory_hand_numbers(self):
        # 朴素 64*128MB = 8192MB = 8.192GB；checkpoint (8+8)*128MB = 2048MB = 2.048GB
        self.assertEqual(naive_activation_memory_mb(64, 128), 8192)
        r = checkpoint_activation_memory_mb(64, 128, 8)
        self.assertEqual(r["checkpoints"], 8)
        self.assertEqual(r["recompute_segment"], 8)
        self.assertEqual(r["memory_mb"], 2048)
        self.assertAlmostEqual(naive_activation_memory_mb(64, 128) / r["memory_mb"], 4.0)

    def test_checkpoint_sweep_minimum_at_sqrt(self):
        # 在 k=1..64 扫描：k=8 处内存最小，验证 (L/k + k) 的极小点
        mems = {k: checkpoint_activation_memory_mb(64, 128, k)["memory_mb"]
                for k in (1, 2, 4, 8, 16, 32, 64)}
        self.assertEqual(min(mems, key=mems.get), 8)
        self.assertEqual(mems[8], 2048)

    def test_recompute_extra_compute_one_third(self):
        # forward:backward = 1:2，重算多一次 forward -> 额外计算 1/3 ≈ 33%
        self.assertAlmostEqual(recompute_extra_compute_frac(), 1.0 / 3.0, places=12)


class TestAmpLedger(unittest.TestCase):
    def test_amp_ledger_7b(self):
        # 手算 E：7e9 参数 AMP 台账 14+14+28+56 = 112GB > 80GB
        r = amp_memory_ledger_gb(7e9)
        self.assertEqual(r["params_GB"], 14.0)
        self.assertEqual(r["grads_GB"], 14.0)
        self.assertEqual(r["master_GB"], 28.0)
        self.assertEqual(r["adam_GB"], 56.0)
        self.assertEqual(r["total_GB"], 112.0)
        self.assertGreater(r["total_GB"], 80.0)

    def test_pure_fp32_ledger_7b(self):
        # 纯 fp32 台账 28+28+56 = 112GB（master 即参数本身，无额外 copy）。
        # 注意：与 AMP 的 112GB 完全相同 —— params+grads 省下的 28GB
        # 被 fp32 master copy 的 28GB 精确抵消。AMP 不缩小 model-state 字节数。
        r = pure_fp32_memory_ledger_gb(7e9)
        self.assertEqual(r["params_GB"], 28.0)
        self.assertEqual(r["grads_GB"], 28.0)
        self.assertEqual(r["adam_GB"], 56.0)
        self.assertEqual(r["total_GB"], 112.0)
        self.assertEqual(r["total_GB"], amp_memory_ledger_gb(7e9)["total_GB"])


if __name__ == "__main__":
    unittest.main()
