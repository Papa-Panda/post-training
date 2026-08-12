# 03 线 II：前向就是梯度下降 — 数学最硬

Papers：von Oswald et al. 2022, Akyürek et al. 2022, Dai et al. 2022

## 核心构造（线性回归情形）

任务：$y = W^* x$，loss $\ell = \frac12 \|W x - y\|^2$。一步 GD：

$$
W_1 = W_0 - \eta \sum_i (W_0 x_i - y_i) x_i^\top = W_0 + \eta \sum_i e_i x_i^\top,\quad e_i = y_i - W_0 x_i
$$

线性注意力（去掉 softmax）：

$$
\text{Attn}(Q,K,V) = V K^\top Q
$$

构造：令 token $e_i = [x_i; y_i]$ 拼进 KV，query $q = [x_q; 0]$，则

$$
V K^\top Q = \sum_i [0; e_i][x_i^\top 0][x_q;0] = [0; (\sum_i e_i x_i^\top) x_q] = \Delta W x_q
$$

加 residual：$out = W_0 x_q + \Delta W x_q = W_1 x_q$。证毕。**单层 Transformer 可精确实现一步 GD**。

## 深层推广

- $L$ 层 = $L$ 步 GD。Garg et al. 2022 在正弦、稀疏线性、决策树上复现：loss 随层数对数下降，与 GD 曲线相关 >0.98。
- Dai Dual Form：

```
h_q' = h_q + Σ_i α_{qi} v_i
若 v_i = η e_i ⊗ x_i，α = <k_i,q>
则 h_q' = h_q - η ∇_{h_q} Σ_i ℓ(x_i,y_i)
```

attention 的 key-query 相似度 = 梯度门控，value = 梯度方向。

## 预言 vs 实测

| 维度 | GD 观点预言 | 实测 |
|---|---|---|
| 顺序 | 求和可交换 → 应顺序不变 | softmax 非线性引入位置偏置 → 半对 |
| scaling | $k$ 越大，$\|\Delta W\|$ 线性增 → 溢出 | many-shot 需 LayerNorm 救场，否则 early stop |
| 非线性 | 多层可拟多步 | 实测比一阶 GD 更快，暗含 preconditioning |

## 对你做 infra / prompt 的用处

- 设计 prompt 格式让 demo 形态利于 $V K^\top Q$ 累加：KV 对齐梯度形式，demo 用等长、显式 `x->y`
- many-shot 需监控 $\|\Delta W\|$ 发散，加 LayerNorm / 按类重排 KV 缓解
