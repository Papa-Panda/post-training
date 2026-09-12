#!/usr/bin/env python3
"""r2-Day17 测试：断言 README 里出现的全部手算数字。

纯 CPU、可执行；不依赖 torch / numpy / CUDA。
"""
import unittest

from framework_selection import (
    bytes_per_param,
    comparison_table_7b_n8,
    ddp_grad_comm_gb,
    ddp_state_per_rank_gb,
    fsdp_comm_gb,
    fsdp_sharded_state_per_rank_gb,
    megatron_tp_params_per_rank_gb,
    ring_allreduce_bytes,
    select_framework,
    tp_allreduces_per_layer,
)


class TestToyLedger(unittest.TestCase):
    def test_bytes_per_param(self):
        # 默认 fp16/bf16 口径：2（params）+ 2（grads）+ 8（Adam m/v）= 12
        self.assertEqual(bytes_per_param(), 12)

    def test_toy_p100_n4_state(self):
        # 手算：P=100, N=4。DDP 每 rank 100×12=1200B；FSDP 切分 1200/4=300B
        self.assertEqual(ddp_state_per_rank_gb(100) * 1e9, 1200)
        self.assertEqual(fsdp_sharded_state_per_rank_gb(100, 4) * 1e9, 300)

    def test_toy_p100_n4_comm(self):
        # grad 字节 = 100×2=200B；ring all-reduce = 2×3/4×200 = 300B
        self.assertEqual(ring_allreduce_bytes(200, 4), 300)
        self.assertAlmostEqual(ddp_grad_comm_gb(100, 4) * 1e9, 300)
        # FSDP ≈ 1.5×DDP = 450B
        self.assertAlmostEqual(fsdp_comm_gb(100, 4) * 1e9, 450)

    def test_toy_tp_params(self):
        # Megatron TP=4 toy：params 200B / 4 = 50B 每 rank
        self.assertEqual(megatron_tp_params_per_rank_gb(100, 4) * 1e9, 50)


class Test7BLedger(unittest.TestCase):
    def test_7b_ddp_state(self):
        # 7B：14（params）+ 14（grads）+ 56（opt）= 84GB，>80GB 装不下
        # 与 r2-Day14 台账一致
        self.assertAlmostEqual(ddp_state_per_rank_gb(7e9), 84.0)

    def test_7b_fsdp_state(self):
        # FSDP full shard N=8：84/8 = 10.5GB 每 rank
        self.assertAlmostEqual(fsdp_sharded_state_per_rank_gb(7e9, 8), 10.5)

    def test_7b_ddp_comm(self):
        # DDP all-reduce grad = 2×7/8×14 = 24.5GB
        self.assertAlmostEqual(ddp_grad_comm_gb(7e9, 8), 24.5)

    def test_7b_fsdp_comm(self):
        # FSDP ≈ 1.5 × 24.5 = 36.75GB
        self.assertAlmostEqual(fsdp_comm_gb(7e9, 8), 36.75)

    def test_7b_tp8_params(self):
        # Megatron TP=8：params 14GB / 8 = 1.75GB 每 rank
        self.assertAlmostEqual(megatron_tp_params_per_rank_gb(7e9, 8), 1.75)

    def test_tp_allreduces_per_layer(self):
        # 每层固定 4 次（forward 2 + backward 2），与 r2-Day15 口径一致
        self.assertEqual(tp_allreduces_per_layer(), 4)


class TestSelection(unittest.TestCase):
    def test_single_gpu_none(self):
        name, _ = select_framework(1, 7.0, False, False, True)
        self.assertEqual(name, "none")

    def test_small_model_none(self):
        name, _ = select_framework(8, 0.5, False, False, True)
        self.assertEqual(name, "none")

    def test_cross_node_large_megatron(self):
        name, reason = select_framework(64, 70.0, True, False, False)
        self.assertEqual(name, "megatron")
        self.assertIn("3D", reason)

    def test_hf_trainer_deepspeed(self):
        name, reason = select_framework(8, 7.0, False, True, True)
        self.assertEqual(name, "deepspeed")
        self.assertIn("ZeRO-3", reason)

    def test_low_intrusion_fsdp(self):
        name, reason = select_framework(8, 7.0, False, False, True)
        self.assertEqual(name, "fsdp")
        self.assertIn("fully_shard", reason)

    def test_default_fsdp(self):
        name, _ = select_framework(8, 7.0, True, False, False)
        self.assertEqual(name, "fsdp")

    def test_branch_priority_cross_node_over_hf(self):
        # 跨机大模型优先于 HF 生态分支（顺序即优先级）
        name, _ = select_framework(64, 70.0, True, True, True)
        self.assertEqual(name, "megatron")


class TestComparisonTable(unittest.TestCase):
    def test_table_numbers_match_functions(self):
        rows = comparison_table_7b_n8()
        self.assertEqual(len(rows), 3)
        ddp, fsdp, tp = rows
        self.assertAlmostEqual(ddp["state_gb_per_rank"], 84.0)
        self.assertAlmostEqual(ddp["comm_gb_per_step"], 24.5)
        self.assertAlmostEqual(fsdp["state_gb_per_rank"], 10.5)
        self.assertAlmostEqual(fsdp["comm_gb_per_step"], 36.75)
        self.assertAlmostEqual(tp["state_gb_per_rank"], 1.75)
        # 表里的数字必须和函数复算一致（防手抄错位）
        self.assertAlmostEqual(
            ddp["state_gb_per_rank"], ddp_state_per_rank_gb(7e9), places=2
        )
        self.assertAlmostEqual(
            fsdp["comm_gb_per_step"], fsdp_comm_gb(7e9, 8), places=2
        )


if __name__ == "__main__":
    unittest.main()
