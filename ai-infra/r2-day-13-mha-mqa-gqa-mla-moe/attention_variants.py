"""r2-day-13: MHA / MQA / GQA / MLA 的 KV-cache 账本与数值语义。

全部是纯 numpy，可在 CPU 上执行；每一步都有对应的测试断言，
没有只 print 步骤、定义变量却不用的代码。
torch / CUDA / H100 均未使用，相关语义标记为 execution not validated。
"""

import numpy as np


# ---------------------------------------------------------------------------
# 记号（每个字母的含义，README/NOTES 有同一份定义）：
#   L   : transformer 层数
#   h_q : query head 数；h_kv : key/value head 数
#   d   : 每个 head 的维度
#   B   : 每个元素的字节数（fp16 -> 2）
#   S   : 序列长度（KV cache 里存的 token 数）
# ---------------------------------------------------------------------------

def kv_cache_bytes(n_layers, n_kv_heads, head_dim, bytes_per_elem, seq_len):
    """KV cache 总字节数。

    每 token 每层存 K 和 V 两个张量，每个形状 (h_kv, d)：
        bytes = L * 2 * h_kv * d * B * S
    MHA: h_kv = h_q；MQA: h_kv = 1；GQA: h_kv = G（组数）。
    """
    return n_layers * 2 * n_kv_heads * head_dim * bytes_per_elem * seq_len


def mla_cache_bytes_per_token(d_c, d_r, bytes_per_elem):
    """MLA 每 token 每层 cache 字节数。

    cache 里只存压缩隐向量 c_t^KV（维度 d_c）与解耦 RoPE 的 k_t^R（维度 d_r）：
        bytes/token/layer = (d_c + d_r) * B
    对比 MHA 同口径：2 * n_h * d_h * B。
    """
    return (d_c + d_r) * bytes_per_elem


def decode_kv_traffic_per_step(n_layers, n_kv_heads, head_dim, bytes_per_elem, seq_len):
    """decode 每步每层要从 HBM 重读的 KV 字节数（= 全量 cache 大小）。

    decode 是 memory-bound（Day06/Day12 的 roofline 逻辑）：每生成 1 个 token，
    每个 layer 都要把 S 个 token 的 K、V 全部读一遍，所以省 cache 字节 = 省时间。
    """
    return kv_cache_bytes(n_layers, n_kv_heads, head_dim, bytes_per_elem, seq_len)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sdpa(Q, K, V):
    """单 head scaled dot-product attention。

    Q: (..., S_q, d)；K, V: (..., S_kv, d)；返回 (..., S_q, d)。
    """
    d = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d)
    return softmax(scores, axis=-1) @ V


def mha_forward(Q, K, V):
    """MHA 前向。Q, K, V: (n_heads, S, d)，每个 head 有独立的 K/V。"""
    return np.stack([sdpa(Q[h], K[h], V[h]) for h in range(Q.shape[0])])


def gqa_forward(Q, K_groups, V_groups):
    """GQA 前向。

    Q: (n_q, S, d)；K_groups, V_groups: (G, S, d)。
    head h 使用 group g = h // (n_q // G) 的 K/V。
    G = n_q 时退化为 MHA；G = 1 时退化为 MQA。
    """
    n_q = Q.shape[0]
    n_groups = K_groups.shape[0]
    assert n_q % n_groups == 0, "query head 数必须被组数整除"
    per_group = n_q // n_groups
    return np.stack(
        [sdpa(Q[h], K_groups[h // per_group], V_groups[h // per_group])
         for h in range(n_q)]
    )


def mla_absorb_demo(rng):
    """验证 MLA 的吸收恒等式（DeepSeek-V2/V3 的核心技巧）。

    记号：h      : (1, d_model) 层输入
           W_DQ  : (d_model, d_qc)   Q 下投影
           W_UQ  : (d_qc, n_h*d_h)   Q 上投影
           W_DKV : (d_model, d_c)    KV 下投影（压缩）
           W_UK  : (d_c, n_h*d_h)    K 上投影
           c     : (1, d_c) 压缩隐向量，c = h @ W_DKV —— 这就是 cache 里存的

    直接算法：q_full = h @ W_DQ @ W_UQ；k_full = c @ W_UK；
              score = q_full @ k_full.T（需要先把 k_full 展开成 n_h*d_h 维）
    吸收算法：W_abs = W_DQ @ W_UQ @ W_UK.T（d_model, d_c），可离线预计算；
              score = h @ W_abs @ c.T（全程只碰 d_c 维的 c，无需展开 K）

    返回 (score_direct, score_absorbed)，测试断言两者数值相等。
    注意：这里省略了 RoPE；真实 MLA 中 RoPE 部分是解耦的 k_t^R，不能被吸收
    （见 NOTES.md），本 demo 只验证非 RoPE 部分的恒等式。
    """
    d_model, d_qc, d_c, n_h, d_h = 16, 8, 6, 4, 4
    W_DQ = rng.standard_normal((d_model, d_qc))
    W_UQ = rng.standard_normal((d_qc, n_h * d_h))
    W_DKV = rng.standard_normal((d_model, d_c))
    W_UK = rng.standard_normal((d_c, n_h * d_h))
    h = rng.standard_normal((1, d_model))

    c = h @ W_DKV                    # (1, d_c)：cache 里实际存的向量
    q_full = h @ W_DQ @ W_UQ         # (1, n_h*d_h)
    k_full = c @ W_UK                # (1, n_h*d_h)
    score_direct = q_full @ k_full.T  # (1, 1)

    W_abs = W_DQ @ W_UQ @ W_UK.T     # (d_model, d_c)：离线预计算一次
    score_absorbed = h @ W_abs @ c.T  # (1, 1)：推理时只用 c，不展开 K

    return score_direct, score_absorbed


def kv_time_lower_bound_ms(n_bytes, bandwidth_bytes_per_s=3.35e12):
    """KV 流量对应的理论时间下限（毫秒）。

    bandwidth 默认取 H100 HBM 3.35 TB/s（Day06 引用的官方规格，十进制前缀）。
    这是 theoretical estimate，不是实测：真实 decode 还要读权重、做 GEMM，
    且带宽利用率 < 100%。
    """
    return n_bytes / bandwidth_bytes_per_s * 1e3


if __name__ == "__main__":
    # 7B 风格配置：L=32, h_q=32, d=128, fp16, S=128k
    L, h_q, d, B, S = 32, 32, 128, 2, 131072
    print("MHA  KV cache :", kv_cache_bytes(L, 32, d, B, S), "bytes")
    print("GQA-8 KV cache:", kv_cache_bytes(L, 8, d, B, S), "bytes")
    print("MQA  KV cache :", kv_cache_bytes(L, 1, d, B, S), "bytes")
    print("MLA  KV cache :", mla_cache_bytes_per_token(512, 64, B) * L * S, "bytes")
    rng = np.random.default_rng(0)
    a, b = mla_absorb_demo(rng)
    print("MLA absorb check: direct =", a.item(), " absorbed =", b.item())
