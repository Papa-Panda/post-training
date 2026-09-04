# NOTES — r2-Day10 FlashAttention

## 准确术语

- **Tiling / blocking**：把 $N$ 维按 $B_r$（行，$Q$ 侧）与 $B_c$（列，$K/V$ 侧）切块，使每个 $(i,j)$ 步的工作集塞进 SRAM。
- **SRAM（片上）vs HBM（片外）**：本课沿用论文口径，SRAM 指 SM 可直接用的快速存储（含 shared memory / registers）；HBM 指 GPU 主存。tiling 的目标是最小化 HBM 访存量，不是减少 FLOPs。
- **Online softmax**：Milakov & Gimelshein (2018) 的分块归一化技巧；FlashAttention 把它用在 $QK^T$ 分块上，使 softmax 不需要看完整行。
- **Running 统计量**：$m$（行最大值）、$\\ell$（running 归一化子和）。$m$ 保证数值稳定（等价于 stable softmax 先减 max）；$\\ell$ 替代"分母"。
- **Rescale 因子**：$m$ 更新为 $m'$ 时，旧累加量乘 $e^{m-m'}$、新 block 乘 $e^{m_j-m'}$。这是"分母变了，分子分母同乘"的精确代数，不是近似。
- **Logsumexp $L=m+\\log\\ell$**：每行 softmax 分母的对数；前向结束时写回 HBM（$O(N)$），反向重算 $S_{ij}$ 时用它恢复 $P_{ij}=\\exp(S_{ij}-L_i)$。
- **IO-aware**：block 尺寸由 SRAM 容量 $M$ 反推（$B_r d+2B_c d+B_rB_c+B_r d\\le M$），而非固定超参数。
- **Exact（精确）vs approximate（近似）**：FlashAttention 是精确等价于标准 attention 的实现优化；稀疏 attention（如 Longformer）才是近似。面试高频混淆点。
- **Recomputation（反向重算）**：反向不存 $S/P$，用存下的 $L$ 和 $Q,K,V,O,dO$ 逐 block 重算 $S_{ij}$ 求梯度：省 $O(N^2)$ 显存，多一次正向量级的 HBM 遍历。
- **$B_r/B_c$**：行 block 高 / 列 block 宽。$B_c$ 大 → 外层循环少 → $Q_i/O_i$ 重载次数少；$B_r$ 大 → 内层并行度高。两者都受 SRAM 约束。

## Connection to Prev 的实质

Day09 的 tiled GEMM 解决"输入复用"：$A/B$ tile 进 shared memory 被 $T^2$ 个输出复用。Day10 解决"中间量落地"：

1. 标准 attention 的 $S=QK^T$（$N\\times N$）必须完整写出、读回两次，主导 $4N^2$ HBM payload；
2. Tiling 让 $S_{ij}$ 只在 SRAM 里存在一个 block 的寿命：算出 → softmax 片段 → 乘 $V_j$ 累加进 $O_i$ → 丢弃；
3. Online softmax 的 $(m,\\ell)$ 是"跨 block 的记忆"，让分块 softmax 与整行 softmax 代数等价；
4. 代价从"存 $N^2$"变成"每外层循环重载 $Q_i/O_i$"——IO 复杂度从 $\\Theta(N^2)$ 降到 $\\Theta(N^2d^2/M)$，$M$ 越大赢得越多。

## Online softmax 推导（逐行，省略行下标）

设已处理 blocks $1..j-1$，记 $m=\\max$ 历史块行 max，$\\ell=\\sum_{t<j}e^{m_t-m}\\tilde{\\ell}_t$，$O=\\sum_{t<j}e^{m_t-m}\\tilde{P}_tV_t$，其中 $\\tilde{P}_t=\\exp(S_t-m_t)$、$\\tilde{\\ell}_t=\\mathrm{rowsum}(\\tilde{P}_t)$、$m_t=\\mathrm{rowmax}(S_t)$。

新 block $j$ 到达：$m_j=\\mathrm{rowmax}(S_j)$，$m'=\\max(m,m_j)$。把旧分母 $\\ell e^{m}$ 与新分子的分母统一到 $e^{m'}$ 尺度：

$$\\ell' e^{m'}=\\ell e^{m}+\\tilde{\\ell}_j e^{m_j}=\\sum_{t\\le j}\\sum_{k\\in t}e^{S_k}$$

两边除以 $e^{m'}$ 即得 $\\ell'=e^{m-m'}\\ell+e^{m_j-m'}\\tilde{\\ell}_j$；分子 $O'$ 同理。最终

$$\\frac{O'}{\\ell'}=\\frac{\\sum_k e^{S_k}v_k}{\\sum_k e^{S_k}}=\\mathrm{softmax}(S)V$$

证毕。数值上这与"先减 max 再 exp"的 stable softmax 是同一技巧，只是 max 在线更新。

## 循环伪代码（论文 Algorithm 1 的极简版）

```text
将 K, V 切成 T_c 个 Bc 列块；将 Q, O, l, m 切成 T_r 个 Br 行块
for j = 1..T_c:                       # 外层：K/V block（每块只加载一次）
  从 HBM 加载 K_j, V_j 到 SRAM
  for i = 1..T_r:                     # 内层：Q block
    从 HBM 加载 Q_i, O_i, l_i, m_i 到 SRAM
    S_ij = Q_i K_j^T / sqrt(d)        # SRAM 内，从不落地
    m_ij = rowmax(S_ij); P~_ij = exp(S_ij - m_ij); l~_ij = rowsum(P~_ij)
    m_i_new = max(m_i, m_ij)
    l_i = exp(m_i - m_i_new)*l_i + exp(m_ij - m_i_new)*l~_ij
    O_i = diag(exp(m_i - m_i_new))*O_i + diag(exp(m_ij - m_i_new))*P~_ij V_j
    写回 O_i, l_i, m_i 到 HBM
收尾：O_i = diag(l_i)^{-1} O_i；L_i = m_i + log(l_i)
```

本课 `flash_attention()` 即此循环的可执行版本（纯 Python，语义与计数逐行对应）。

## 反向传播一句话

前向存 $O(N)$ 的 $L$；反向对每个 $(i,j)$ 从 HBM 重载 $Q_i,K_j,V_j,O_i,dO_i,L_i$，在 SRAM 里重算 $S_{ij}$、$P_{ij}=\\exp(S_{ij}-L_i)$，累加 $dQ_i,dK_j,dV_j$。论文给出反向同样 $O(N^2d^2/M)$ HBM 访问、$O(N)$ 额外显存。

## 何时不赚（再强调一次）

1. $N$ 小：tiling 的循环与 bookkeeping 开销超过 $S$ 本身流量（本课玩具 $N=4$ 就是实例：112 vs 96）。
2. $d$ 相对 SRAM 大：$B_r,B_c$ 被迫取小，$Q_i/O_i$ 每外层重载，主导项 $2N^2d(1/B_r+1/B_c)$ 逼近甚至超过 $4N^2$。
3. 非 attention-bound：模型瓶颈在 MLP/通信时，fused attention 省下的 HBM 对端到端无感——这是 Day12 profiling 要验证的。
4. 已有高度优化的 fused kernel（FA2/FA3、FlashInfer）时，手写 FA1 风格 kernel 不赚；本课价值在理解"为什么"，不在产出 kernel。

## 与 Day11/12 的连接

Day11（Triton/`torch.compile`）回答"fused kernel 在工程上怎么写"；Day12（Nsight）回答"怎么证明省下的 HBM 真的变成了 wall-time"。本课只负责把数学与流量账算对。
