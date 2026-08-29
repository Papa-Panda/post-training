# 02 — 从归因到目标化选择

## 元信息
- 内容类型：跨论文方法综述（归因 → 目标化选择）
- 新增核心论文：[TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186) · [GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning](https://arxiv.org/abs/2602.21492v2)
- 已有 `ai-data` 精读：[Influence Functions](../ai-data/day-02-2017-influence-functions/NOTES.md) · [TracIn](../ai-data/day-03-2020-tracin/NOTES.md) · [LESS](../ai-data/day-04-2024-less/NOTES.md) · [DataInf](../ai-data/day-05-2024-datainf/NOTES.md)


> 本章只建立跨方法坐标系。已有逐篇笔记不复制：
> [Influence Functions](../ai-data/day-02-2017-influence-functions/NOTES.md) ·
> [TracIn](../ai-data/day-03-2020-tracin/NOTES.md) ·
> [LESS](../ai-data/day-04-2024-less/NOTES.md) ·
> [DataInf](../ai-data/day-05-2024-datainf/NOTES.md)

## 1. 一个问题，四种近似

训练样本 $z$ 对目标样本 $v$ 的价值，可看成“训练一步后目标 loss 降多少”。一阶展开：

$$
L_v(\theta-\eta g_z)\approx L_v(\theta)-\eta g_v^\top g_z.
$$

不同方法主要在回答：应该使用哪个梯度、哪个曲率、哪个 checkpoint、如何降维。

| 方法 | 核心估计 | 计算/系统取舍 | 在闭环中的角色 |
|---|---|---|---|
| Influence Functions | $-g_v^\top H^{-1}g_z$ | 曲率更完整，但 $H^{-1}$ 昂贵且深网近似敏感 | 理论基线 |
| TracIn | $\sum_t\eta_t g_v(\theta_t)^\top g_z(\theta_t)$ | 用 checkpoints 避开 Hessian | 训练轨迹归因 |
| TRAK | 随机投影梯度 + after-kernel 线性化 + 少量模型 ensemble | 把高维归因压入可复用矩阵计算 | 大规模行为归因 |
| LESS | optimizer-aware、低秩梯度 datastore 与目标 few-shot 相似度 | 一次建库，多目标复用；SFT 静态选择 | 目标能力选数 |
| DataInf | 针对 LoRA / empirical Fisher 的高效闭式近似 | 适合参数高效微调 | LoRA 场景曲率校正 |
| GradAlign | 当前 policy gradient 与 trusted validation gradient 的 cosine | 周期性重算，适应 RL 非平稳性 | 在线 RL curriculum |

## 2. TRAK：重要的不是“又一个点积”

TRAK（Tracing with the Randomly-projected After Kernel）将每条样本的梯度/输出敏感度投影为低维特征 $\phi(z)$，再通过线性化后的核矩阵求 attribution。抽象写作：

若 $\Phi$ 按行堆叠训练样本特征、$Q=\mathrm{diag}(1-p_i^*)$，一组训练样本对目标 $v$ 的 attribution 向量可抽象写成：

$$
\tau(v,D)=\phi(v)^\top
(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top Q.
$$

并对少量独立训练模型/子集取平均以降低 seed 方差。实际 estimator 还包含论文定义的 soft-thresholding 等细节，应遵循原论文/官方代码，而非把上式当作完整复现。

**已核实 claim**：TRAK 原论文说 *a handful of trained models* 可匹配需要 *thousands of models* 的 attribution 方法。论文摘要没有承诺一个跨任务固定的模型数，所以本专题不写“固定 5 个”或“固定 20 个”。

## 3. Targeted selection：从解释过去到选择未来

归因问“谁造成了预测”；目标化选择把同一几何反向用于候选排序：

$$
s_i=\frac{\langle P g_i,\ P\bar g_V\rangle}
{\|Pg_i\|\,\|P\bar g_V\|},
\qquad P\in\mathbb R^{d\times |\theta|}.
$$

- $P$ 可由 LoRA 梯度、随机投影或选定层组成；
- LESS 强调 Adam-aware 表征和可复用 datastore；
- DataInf 在 LoRA 设置中近似曲率修正；
- 不能只取 top-$k$：相似样本会挤占预算，需与 coverage 联合。

## 4. GradAlign：RL 中分数会过期

对第 $t$ 轮候选问题 $p_i$，由 rollout/reward 得到 policy gradient $g^{RL}_{i,t}$；trusted validation set 给出：

$$
\bar g^{RL}_{V,t}=\frac1{|V|}\sum_{v\in V}g^{RL}_{v,t},
\qquad
a_{i,t}=\cos(g^{RL}_{i,t},\bar g^{RL}_{V,t}).
$$

选择 $a_{i,t}$ 高的题训练，并周期性以新 policy 重算。这和 LESS 最大的系统差异不是余弦公式，而是：

1. rollout 随 policy 变化；
2. reward 可能不可靠；
3. selector 是 adaptive curriculum，不是离线一次性筛选。

GradAlign 是 2026 年预印本，适合作为前沿方向和系统设计参照，不应写成已成熟工业标准。

## 5. 组合方式

实用 pipeline：

```text
quality gate
  -> projected gradient store
  -> target-alignment shortlist
  -> G-Vendi / sparse-cluster coverage
  -> conflict gate
  -> train
  -> refresh on drift
```

归因给“价值”，但不保证“覆盖”；下一章用谱熵补上这一维。
