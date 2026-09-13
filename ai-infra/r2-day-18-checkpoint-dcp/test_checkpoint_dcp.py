#!/usr/bin/env python3
"""r2-Day18 测试：台账数字、Young/Daly、快照一致性、两槽轮换、重分片、crash 恢复。

纯 CPU（标准库 + unittest），共 19 个测试。
"""
import math
import os
import shutil
import tempfile
import unittest

from checkpoint_dcp import (
    GB,
    ShardedCheckpointer,
    ToyTrainer,
    ckpt_bytes_per_param,
    full_ckpt_gb,
    optimal_interval_s,
    rank0_gather_extra_gb,
    sharded_per_rank_gb,
    waste_fraction,
)

P_7B = 7e9


class TestLedger(unittest.TestCase):
    def test_ckpt_bytes_per_param_default_10(self):
        # params 2B + Adam m/v 8B；grads 是 step 内瞬态，不进 checkpoint
        self.assertEqual(ckpt_bytes_per_param(), 10)
        self.assertEqual(ckpt_bytes_per_param(param_bytes=4, opt_bytes=8), 12)

    def test_full_ckpt_7b_70gb(self):
        # 7B：14GB params(fp16) + 56GB opt(fp32 m+v) = 70GB
        self.assertAlmostEqual(full_ckpt_gb(P_7B), 70.0)

    def test_sharded_per_rank_64(self):
        self.assertAlmostEqual(sharded_per_rank_gb(P_7B, 64), 70.0 / 64)
        self.assertAlmostEqual(sharded_per_rank_gb(P_7B, 64), 1.09375)

    def test_rank0_gather_extra_7b_64(self):
        # 除 rank0 外 63 个 shard 都要搬运：63/64 × 70 = 68.90625GB 纯额外流量
        self.assertAlmostEqual(rank0_gather_extra_gb(70.0, 64), 68.90625)

    def test_rank0_gather_grows_with_n(self):
        # N 越大，gather 越接近"搬运整个 checkpoint"；sharded 则恒为 0
        self.assertLess(rank0_gather_extra_gb(70.0, 2), rank0_gather_extra_gb(70.0, 1024))
        self.assertAlmostEqual(rank0_gather_extra_gb(70.0, 1024), 70.0 * 1023 / 1024)


class TestYoungDaly(unittest.TestCase):
    def test_optimal_interval_toy(self):
        # delta=10s, M=1000s -> tau* = sqrt(20000) ≈ 141.42s
        self.assertAlmostEqual(optimal_interval_s(10.0, 1000.0), math.sqrt(20000.0))

    def test_waste_at_optimum_equal_split(self):
        # 最优点处两项相等：delta/tau = tau/(2M)
        delta, m = 10.0, 1000.0
        tau = optimal_interval_s(delta, m)
        self.assertAlmostEqual(delta / tau, tau / (2.0 * m))
        self.assertAlmostEqual(waste_fraction(delta, m, tau), 2.0 * delta / tau)
        self.assertAlmostEqual(waste_fraction(delta, m, tau), 0.141421356237, places=6)

    def test_waste_minimum(self):
        delta, m = 60.0, 36000.0
        tau_star = optimal_interval_s(delta, m)
        w_star = waste_fraction(delta, m, tau_star)
        self.assertLess(w_star, waste_fraction(delta, m, tau_star / 10))
        self.assertLess(w_star, waste_fraction(delta, m, tau_star * 10))

    def test_optimal_interval_sqrt_scaling(self):
        # delta 翻倍 -> tau* 变为 sqrt(2) 倍（不是翻倍）
        base = optimal_interval_s(60.0, 36000.0)
        self.assertAlmostEqual(optimal_interval_s(120.0, 36000.0), base * math.sqrt(2.0))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            optimal_interval_s(0.0, 1000.0)
        with self.assertRaises(ValueError):
            optimal_interval_s(10.0, -5.0)
        with self.assertRaises(ValueError):
            waste_fraction(10.0, 1000.0, 0.0)


class TestSnapshotAndSlots(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r2d18_")
        self.ckpt = ShardedCheckpointer(os.path.join(self.tmp, "ckpts"), world_size=4)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _trainer_state(self, steps=6):
        t = ToyTrainer(seed=7)
        for _ in range(steps):
            t.train_step()
        return t.state_dict()

    def test_sync_save_roundtrip(self):
        sd = self._trainer_state(6)
        self.ckpt.save(sd, "a")
        back = self.ckpt.load()
        self.assertEqual(back["flat_params"], sd["flat_params"])
        self.assertEqual(back["opt_state"], sd["opt_state"])
        self.assertEqual(back["step"], 6)
        self.assertEqual(back["rng"], sd["rng"])

    def test_async_save_snapshot_isolation(self):
        # 快照在 async_save 返回前已 deepcopy：之后再改 state dict，
        # 落盘的仍是快照时刻的值（一致性点语义）。
        sd = self._trainer_state(6)
        fut = self.ckpt.async_save(sd, "a")
        sd["flat_params"][0] += 999.0  # 快照之后的手改，训练继续的类比
        sd["step"] = 12345
        fut.result(timeout=30)
        back = self.ckpt.load()
        self.assertNotAlmostEqual(back["flat_params"][0], sd["flat_params"][0])
        self.assertEqual(back["step"], 6)

    def test_torn_write_falls_back_to_previous_slot(self):
        self.ckpt.save(self._trainer_state(10), "a")  # step=10，好槽
        self.ckpt.save(self._trainer_state(20), "b")  # step=20，active
        # 模拟 torn write：写坏 active 槽的一个 shard 文件
        slot_b = os.path.join(self.tmp, "ckpts", "slot_b")
        shard = os.path.join(slot_b, sorted(os.listdir(slot_b))[0])
        with open(shard, "w", encoding="utf-8") as f:
            f.write('{"params": [0], "opt": [0]}')  # 内容与 sha256 不符
        back = self.ckpt.load()
        self.assertEqual(back["step"], 10)  # 回退到上一槽

    def test_all_slots_torn_raises(self):
        self.ckpt.save(self._trainer_state(10), "a")
        slot_a = os.path.join(self.tmp, "ckpts", "slot_a")
        shard = os.path.join(slot_a, sorted(os.listdir(slot_a))[0])
        with open(shard, "w", encoding="utf-8") as f:
            f.write("corrupt")
        with self.assertRaises(RuntimeError):
            self.ckpt.load()


class TestReshard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="r2d18r_")
        self.ckpt = ShardedCheckpointer(os.path.join(self.tmp, "ckpts"), world_size=4)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reshard_4_to_2(self):
        t = ToyTrainer(seed=7)
        for _ in range(5):
            t.train_step()
        sd = t.state_dict()
        self.ckpt.save(sd, "a")
        back = self.ckpt.load(world_size=2)  # 换 2 卡重启
        self.assertEqual(back["saved_world_size"], 4)
        self.assertEqual(back["loaded_world_size"], 2)
        self.assertEqual(back["rank_param_shards"][0], sd["flat_params"][:2])
        self.assertEqual(back["rank_param_shards"][1], sd["flat_params"][2:])
        self.assertEqual(back["flat_params"], sd["flat_params"])

    def test_reshard_4_to_8_empty_shards_ok(self):
        # 4 个参数切到 8 卡：一半 rank 空 shard，全局仍能拼回
        t = ToyTrainer(seed=7)
        sd = t.state_dict()
        self.ckpt.save(sd, "a")
        back = self.ckpt.load(world_size=8)
        self.assertEqual(back["flat_params"], sd["flat_params"])
        self.assertEqual(len(back["rank_param_shards"]), 8)
        rebuilt = [x for shard in back["rank_param_shards"] for x in shard]
        self.assertEqual(rebuilt, sd["flat_params"])


class TestCrashRecovery(unittest.TestCase):
    def _run(self, steps, seed=7):
        t = ToyTrainer(seed=seed)
        for _ in range(steps):
            t.train_step()
        return t

    def test_crash_recovery_identical(self):
        # 对照组： uninterrupted 30 步
        ref = self._run(30)
        # 实验组：跑 25 步，每 10 步 checkpoint；第 25 步后"crash"（内存全丢），
        # 从 step=20 的 checkpoint 恢复（rng/opt/step 全在），再跑 10 步
        ckpts = {}
        t = ToyTrainer(seed=7)
        for _ in range(25):
            t.train_step()
            if t.step % 10 == 0:
                ckpts[t.step] = t.state_dict()
        t2 = ToyTrainer(seed=999)  # crash 后是新进程，初始状态无关
        t2.load_state_dict(ckpts[20])
        for _ in range(10):
            t2.train_step()
        self.assertEqual(t2.step, 30)
        self.assertEqual(t2.flat_params, ref.flat_params)
        self.assertEqual(t2.opt_state, ref.opt_state)
        self.assertEqual(t2.rng.getstate(), ref.rng.getstate())

    def test_cold_restart_without_opt_rng_diverges(self):
        # 反例：只恢复 params（momentum 归零、rng 重开），轨迹分叉——
        # 证明 opt state + rng 必须进 checkpoint
        ref = self._run(30)
        t = self._run(20)
        t2 = ToyTrainer(seed=999)
        t2.flat_params = list(t.flat_params)  # 只拷参数
        for _ in range(10):
            t2.train_step()
        self.assertTrue(any(abs(a - b) > 1e-9
                            for a, b in zip(t2.flat_params, ref.flat_params)))

    def test_rework_bounded_by_interval(self):
        # 每 10 步 checkpoint，第 25 步 crash：最多重做 25-20=5 步 ≤ 间隔
        t = ToyTrainer(seed=7)
        last_ckpt_step = 0
        for _ in range(25):
            t.train_step()
            if t.step % 10 == 0:
                last_ckpt_step = t.step
        self.assertEqual(last_ckpt_step, 20)
        self.assertLessEqual(t.step - last_ckpt_step, 10)


if __name__ == "__main__":
    unittest.main()
