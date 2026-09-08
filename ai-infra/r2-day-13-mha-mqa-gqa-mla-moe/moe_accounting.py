"""r2-day-13: MoE 路由 / FLOPs / 参数 / all-to-all 通信账本。

全部是纯 numpy，可在 CPU 上执行；每一步都有对应的测试断言。
torch / CUDA / H100 均未使用，相关语义标记为 execution not validated。

记号：E : 专家总数；k : 每个 token 激活的 top-k 专家数；
      d : 模型 hidden 维；d_ff : 专家 FFN 中间维；T : token 数。
"""

import numpy as np


def router_topk(logits, k):
    """top-k 路由。

    logits: (T, E)，第 t 个 token 对 E 个专家的亲和分数。
    返回 (indices, weights)：indices (T, k) 为选中的专家编号；
    weights (T, k) 为 top-k logits 做 softmax 重归一后的混合权重（和为 1）。
    """
    assert 1 <= k <= logits.shape[1]
    indices = np.argsort(-logits, axis=1)[:, :k]
    top = np.take_along_axis(logits, indices, axis=1)
    w = np.exp(top - top.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    return indices, w


def dense_ffn_flops(d_model, d_ff):
    """dense FFN 每 token FLOPs：up 投影 2*d*d_ff + down 投影 2*d_ff*d。"""
    return 4 * d_model * d_ff


def moe_ffn_flops(d_model, d_ff, k):
    """MoE 每 token FLOPs：只算被选中的 k 个专家。k=1 时与 dense 持平。"""
    return k * dense_ffn_flops(d_model, d_ff)


def dense_ffn_params(d_model, d_ff):
    """dense FFN 参数量：up (d*d_ff) + down (d_ff*d)。"""
    return 2 * d_model * d_ff


def moe_ffn_params(d_model, d_ff, n_experts):
    """MoE 参数量：E 个专家各一份 dense FFN。FLOPs/token 不涨，参数涨 E 倍。"""
    return n_experts * dense_ffn_params(d_model, d_ff)


def alltoall_bytes(n_tokens, d_model, bytes_per_elem):
    """expert parallelism 的 all-to-all 通信量（dispatch + combine）。

    每个 token 的 hidden 向量（d 个元素）被发往它命中的专家所在卡（dispatch），
    算完再发回来拼（combine）：2 * T * d * B 字节。这是 MoE 为稀疏付的通信税，
    与 Day03 的 collective 主题直接相连。
    """
    return 2 * n_tokens * d_model * bytes_per_elem


if __name__ == "__main__":
    # 手算例：E=4, k=1, d=4, d_ff=8, T=3
    logits = np.array([
        [2.0, 1.0, 0.5, 0.1],
        [0.2, 3.0, 0.1, 0.4],
        [0.1, 0.2, 2.5, 0.3],
    ])
    idx, w = router_topk(logits, k=1)
    print("top-1 experts:", idx.ravel().tolist())
    print("dense FLOPs/token:", dense_ffn_flops(4, 8))
    print("moe   FLOPs/token:", moe_ffn_flops(4, 8, k=1))
    print("moe   params     :", moe_ffn_params(4, 8, n_experts=4))
    print("all-to-all bytes :", alltoall_bytes(3, 4, 2))
