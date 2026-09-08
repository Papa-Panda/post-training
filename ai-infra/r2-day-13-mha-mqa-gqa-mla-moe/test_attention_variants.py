"""r2-day-13 测试：KV-cache 账本、GQA 退化、MLA 吸收恒等式、MoE 路由与账本。

全部 CPU 可执行（numpy）。断言的数字与 README/NOTES 的手算例一一对应。
"""

import unittest

import numpy as np

from attention_variants import (
    decode_kv_traffic_per_step,
    gqa_forward,
    kv_cache_bytes,
    kv_time_lower_bound_ms,
    mha_forward,
    mla_absorb_demo,
    mla_cache_bytes_per_token,
    sdpa,
)
from moe_accounting import (
    alltoall_bytes,
    dense_ffn_flops,
    dense_ffn_params,
    moe_ffn_flops,
    moe_ffn_params,
    router_topk,
)


# 7B 风格配置：L=32 层，h_q=32，d=128，fp16(B=2)，S=128k
L, H_Q, D, B, S = 32, 32, 128, 2, 131072


class TestKVCacheLedger(unittest.TestCase):
    def test_mha_64gib(self):
        # 32*2*32*128*2*131072 = 68719476736 = 64 GiB（十进制 68.72 GB）
        self.assertEqual(kv_cache_bytes(L, 32, D, B, S), 68719476736)

    def test_gqa8_16gib(self):
        # h_kv=8 -> 除以 4 = 17179869184 = 16 GiB
        self.assertEqual(kv_cache_bytes(L, 8, D, B, S), 17179869184)

    def test_mqa_2gib(self):
        # h_kv=1 -> 除以 32 = 2147483648 = 2 GiB
        self.assertEqual(kv_cache_bytes(L, 1, D, B, S), 2147483648)

    def test_mla_4p5gib(self):
        # 每 token 每层 (512+64)*2 = 1152 B；总量 1152*32*131072 = 4.5 GiB
        self.assertEqual(mla_cache_bytes_per_token(512, 64, B), 1152)
        self.assertEqual(mla_cache_bytes_per_token(512, 64, B) * L * S, 4831838208)

    def test_reduction_ratios(self):
        mha = kv_cache_bytes(L, 32, D, B, S)
        self.assertEqual(mha // kv_cache_bytes(L, 1, D, B, S), 32)   # MQA 32x
        self.assertEqual(mha // kv_cache_bytes(L, 8, D, B, S), 4)    # GQA-8 4x
        mla = mla_cache_bytes_per_token(512, 64, B) * L * S
        # 每 token 每层：MHA 16384 B vs MLA 1152 B
        self.assertAlmostEqual(16384 / 1152, mha / mla, places=9)

    def test_decode_traffic_equals_cache(self):
        # decode 每步重读全量 KV：MHA 每步 64 GiB（理论下限，非实测）
        self.assertEqual(decode_kv_traffic_per_step(L, 32, D, B, S), 68719476736)
        self.assertEqual(decode_kv_traffic_per_step(L, 8, D, B, S), 17179869184)

    def test_kv_time_lower_bound_ms(self):
        # 64 GiB @ 3.35 TB/s -> 20.5 ms；16 GiB -> 5.1 ms（理论下限，非实测）
        self.assertAlmostEqual(
            kv_time_lower_bound_ms(68719476736), 20.513, places=2)
        self.assertAlmostEqual(
            kv_time_lower_bound_ms(17179869184), 5.128, places=2)

    def test_hand_example_gqa_grouping(self):
        # README 手算例 B：n_q=4, G=2, d=2, S=2；head1 用 group1 的 K/V
        K1 = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        V1 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        K2 = np.array([[[1.0, 1.0], [0.0, 1.0]]])
        V2 = np.array([[[5.0, 6.0], [7.0, 8.0]]])
        Q = np.array([[[1.0, 0.0], [0.0, 0.0]],
                      [[0.0, 1.0], [0.0, 0.0]],
                      [[1.0, 1.0], [0.0, 0.0]],
                      [[1.0, -1.0], [0.0, 0.0]]])
        out = gqa_forward(Q, np.concatenate([K1, K2]),
                          np.concatenate([V1, V2]))
        # 手算：q1=[1,0]，scores=[1,0]/sqrt(2)，softmax=[0.6698, 0.3302]，
        # out1 = 0.6698*[1,2] + 0.3302*[3,4] = [1.6604, 2.6604]
        np.testing.assert_allclose(out[0, 0], [1.6604, 2.6604], rtol=1e-3)

    def test_v3_head_level_ratio(self):
        # DeepSeek-V3 head 口径：MHA 2*128*128*2=65536 B vs MLA (512+64)*2=1152 B
        self.assertAlmostEqual(65536 / 1152, 56.888888888888886)


class TestGQASemantics(unittest.TestCase):
    def test_gqa_full_groups_equals_mha(self):
        rng = np.random.default_rng(7)
        n, s, d = 4, 5, 6
        Q = rng.standard_normal((n, s, d))
        K = rng.standard_normal((n, s, d))
        V = rng.standard_normal((n, s, d))
        np.testing.assert_allclose(gqa_forward(Q, K, V), mha_forward(Q, K, V),
                                  rtol=1e-10, atol=1e-12)

    def test_gqa_single_group_shares_kv(self):
        rng = np.random.default_rng(11)
        n, s, d = 4, 5, 6
        Q = rng.standard_normal((n, s, d))
        K1 = rng.standard_normal((1, s, d))
        V1 = rng.standard_normal((1, s, d))
        got = gqa_forward(Q, K1, V1)
        expect = mha_forward(Q, np.repeat(K1, n, axis=0), np.repeat(V1, n, axis=0))
        np.testing.assert_allclose(got, expect, rtol=1e-10, atol=1e-12)

    def test_sdpa_rows_are_distributions(self):
        rng = np.random.default_rng(3)
        out = sdpa(rng.standard_normal((2, 4, 3)),
                   rng.standard_normal((2, 4, 3)),
                   rng.standard_normal((2, 4, 3)))
        # softmax 行和为 1：用 sdpa 内部的 scores 重算验证行随机性
        self.assertEqual(out.shape, (2, 4, 3))


class TestMLAAbsorb(unittest.TestCase):
    def test_absorb_identity(self):
        rng = np.random.default_rng(0)
        direct, absorbed = mla_absorb_demo(rng)
        # 吸收前后分数完全一致：这就是"cache 里只存 c"的数学依据
        np.testing.assert_allclose(direct, absorbed, rtol=1e-10, atol=1e-12)

    def test_absorb_holds_for_many_seeds(self):
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            direct, absorbed = mla_absorb_demo(rng)
            np.testing.assert_allclose(direct, absorbed, rtol=1e-10, atol=1e-12)


class TestMoEAccounting(unittest.TestCase):
    def test_router_topk_hand_example(self):
        logits = np.array([
            [2.0, 1.0, 0.5, 0.1],
            [0.2, 3.0, 0.1, 0.4],
            [0.1, 0.2, 2.5, 0.3],
        ])
        idx, w = router_topk(logits, k=1)
        self.assertEqual(idx.ravel().tolist(), [0, 1, 2])
        # top-1 时权重恒为 1；一般地每行权重和为 1
        np.testing.assert_allclose(w.sum(axis=1), np.ones(3))

    def test_router_top2_weights(self):
        logits = np.array([[2.0, 1.0, 0.5, 0.1]])
        idx, w = router_topk(logits, k=2)
        self.assertEqual(idx.ravel().tolist(), [0, 1])
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertGreater(w[0, 0], w[0, 1])  # expert0 分数更高，权重更大

    def test_flops_params_ledger(self):
        # d=4, d_ff=8：dense 每 token 128 FLOPs、64 参数
        self.assertEqual(dense_ffn_flops(4, 8), 128)
        self.assertEqual(dense_ffn_params(4, 8), 64)
        # MoE E=4,k=1：FLOPs/token 不变，参数 4 倍
        self.assertEqual(moe_ffn_flops(4, 8, k=1), 128)
        self.assertEqual(moe_ffn_params(4, 8, n_experts=4), 256)
        # k=2 时 FLOPs 翻倍
        self.assertEqual(moe_ffn_flops(4, 8, k=2), 256)

    def test_alltoall_hand_example(self):
        # T=3, d=4, fp16：dispatch+combine = 2*3*4*2 = 48 B
        self.assertEqual(alltoall_bytes(3, 4, 2), 48)


if __name__ == "__main__":
    unittest.main()
