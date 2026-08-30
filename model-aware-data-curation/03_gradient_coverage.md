# 03 — 多样性覆盖：从语义空间到梯度空间

## 元信息
- 内容类型：跨论文方法综述（embedding diversity → gradient diversity）
- 核心论文：[Prismatic Synthesis: Gradient-based Data Diversification Boosts Generalization in LLM Reasoning](https://arxiv.org/abs/2505.20161) · [SPICE: Submodular Penalized Information-Conflict Selection for Efficient Large Language Model Training](https://arxiv.org/pdf/2601.23155v2)
- 已有 `ai-data` 精读：[Vendi Score](../ai-data/day-19-2023-vendi-score/NOTES.md) · [D4 / SemDeDup](../ai-data/day-24-2023-semdedup-d4/NOTES.md)


> 已有逐篇笔记只复用：
> [Vendi Score](../ai-data/day-19-2023-vendi-score/NOTES.md) ·
> [D4 / SemDeDup](../ai-data/day-24-2023-semdedup-d4/NOTES.md)

## 1. “文本不同”不等于“学到不同”

embedding diversity 衡量表面语义或预训练表示上的差异；model-aware diversity 衡量样本对**当前模型更新方向**的差异。

两个问题可能词面不同，但诱导几乎平行的梯度；另两个词面接近的问题，可能分别需要不同算法，因此梯度方向显著不同。对 reasoning/coding data，后一种区分更接近“学习方向覆盖”。

## 2. Vendi 的共同数学骨架

给定单位化特征 $u_i$，构造核：

$$
K_{ij}=\frac{u_i^\top u_j}{n},\qquad
\sum_j\lambda_j(K)=\mathrm{tr}(K)=1.
$$

Vendi Score 是谱分布的有效秩：

$$
\mathrm{VS}(D)=
\exp\left(-\sum_j\lambda_j\log\lambda_j\right).
$$

- 所有方向完全相同：VS 接近 1；
- $m$ 个正交且均匀方向：VS 接近 $m$；
- 它不只是平均 pairwise distance，而是看整个方向谱是否被少数主成分支配。

## 3. G-Vendi：把特征换成 model-induced gradients

Prismatic Synthesis 定义样本表征：

$$
g_\theta(x,y)=
\frac{-\nabla_\theta\log p_\theta(y\mid x)}
{\lVert -\nabla_\theta\log p_\theta(y\mid x)\rVert_2},
$$

再用 Rademacher 随机矩阵 $\Pi\in\{-1,+1\}^{|\theta|\times d}$ 降维：

$$
\tilde g_\theta(x,y)=\Pi^\top g_\theta(x,y),\qquad d\ll |\theta|.
$$

对 $G=[\tilde g_1;\ldots;\tilde g_n]$，以 $GG^\top/n$（或当 $n\gg d$ 时等价地使用 $G^\top G/n$ 的非零谱）计算 Vendi：

$$
\text{G-Vendi}(D)=
\exp\left(-\sum_j\lambda_j\log\lambda_j\right).
$$

论文实验使用 $d=1024$，并展示无需 in-domain warm-up 的小型 instruction-tuned proxy 也可提供有用梯度几何。

## 4. D4 / SemDeDup 与 G-Vendi 的关系

| 层 | 表征 | 优化对象 | 适合阶段 |
|---|---|---|---|
| exact / MinHash dedup | 字符、shingle | 去复制 | 数据入口 |
| SemDeDup | 预训练 embedding | 去语义近重复 | 大规模 corpus |
| D4 | embedding cluster / prototype | 去重后扩覆盖 | 预训练选择 |
| Vendi | 任意用户给定 kernel | 量化集合有效多样性 | 通用诊断 |
| G-Vendi | 当前/代理模型 loss gradient | 覆盖模型学习方向 | SFT/RL/合成闭环 |

顺序不是“G-Vendi 替代去重”，而是：

```text
exact dedup -> semantic dedup -> quality gate -> gradient-space coverage
```

低成本规则先缩池，昂贵梯度只算在剩余候选上。

## 5. Fisher/log-det：另一种 gradient-space coverage

SPICE 沿用 Fisher/D-optimal design 的集合目标：

$$
F_S=\sum_{i\in S}g_i g_i^\top,
\qquad
U(S)=\log\det(I+\alpha F_S).
$$

其候选边际覆盖增益为：

$$
\Delta_x(S)
=
\log\left(
1+\alpha g_x^\top(I+\alpha F_S)^{-1}g_x
\right).
$$

它与 G-Vendi 都读取梯度谱，但优化语义不同：

| 方法 | 集合量 | 更直接鼓励什么 |
|---|---|---|
| G-Vendi | 归一化谱的指数熵 | 有效方向数与谱均匀度 |
| Fisher/log-det | $\log\det(I+\alpha F_S)$ | information volume 与新方向的边际增益 |

两者还有一个共同边界：基于 $g_i g_i^\top$ 或 Gram spectrum 的 coverage 对 $g_i$ 与 $-g_i$ 不敏感。它们能识别“轴是否新”，不能单独识别“沿该轴更新的符号是否会抵消训练”。SPICE 因此在 Fisher marginal gain 外另加 sign-sensitive conflict penalty；完整推导见 [`09_spice_information_conflict.md`](09_spice_information_conflict.md)。

## 6. Coverage-aware greedy selection

从目标化 shortlist 中迭代加入使谱熵增益最大的样本：

$$
z^*=\arg\max_{z\notin S}
\left[\alpha v(z)+\beta\Delta\log\mathrm{GV}(z\mid S)\right]
\quad\text{s.t.}\ r(z)\le\epsilon.
$$

生产上不必每次完整特征分解：

- 先对低维梯度做 mini-batch k-means；
- 稀疏簇配额上采样，密集簇降采样；
- 周期性精确算 G-Vendi 作为监控，而非每个候选都做 eigendecomposition；
- 记录每个簇的 correctness rate，防止“噪声 = 多样性”。

## 7. 证据边界

Prismatic 报告 G-Vendi 与 OOD performance 的 Spearman $\rho\approx0.9$，且在 NLI 和数学推理上观察到；这是受控数据规模/质量下的秩相关。它不意味着：

- 任意任务都能复现 0.9；
- 最大化 G-Vendi 必然最大化准确率；
- 随机/错误输出带来的梯度熵就是有益覆盖。

因此至少联合三个 dashboard：`correctness × target alignment × G-Vendi`。
