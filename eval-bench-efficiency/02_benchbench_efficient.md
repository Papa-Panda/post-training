# 02 BenchBench / Efficient Benchmarking — 从 IRT 到特征选择

## HELM Efficient Benchmarking (arXiv:2308.11696v5)

- 提出 **DIoR (Decision Impact on Reliability)**：子集排序与全量排序的决策一致性，而非单点 RMSE。问 “如果我用 100 题决定用 model A 还是 B，错判概率多大？”
- 结论：只用 **≈1% 题就能得到可靠排名**，x100 成本节省，但随机子集方差大，seed 不稳定。

> 原文写道：ranking can be obtained by a fraction of examples, but naive random is unreliable.

## 新视角：高效评测 = 特征选择 + 多元回归 (arXiv:2605.25773)

> Title: *Efficient Benchmarking Is Just Feature Selection and Multiple Regression*, 2026.05.

把任务重新表述：

- 每道题 $j$ 是特征 $x_j \in \{0,1\}^n$ (n 个模型的对错)
- 总分 $y \in \mathbb{R}^n$ 是目标
- 选 k 个特征 $S$, $|S|=k$, 回归 $y \approx w^T x_S$

于是：

1. **Kernel Ridge Regression** 代替线性回归：$K(x_i,x_j)=\exp(-\gamma \|x_i-x_j\|^2)$，对非线性交互更强，显著提升现有方法。
2. **mRMR (minimum Redundancy Maximum Relevance)** 信息论选题：

$$J(f) = I(f; y) - \frac1{|S|}\sum_{s\in S} I(f; s)$$

- $I(f;y)$ = 题 f 与总分互信息，越高越 relevant
- 减去与已选题平均冗余，**最大相关最小冗余**

对比结果 (在论文表 2/3)：

- mRMR > IRT/clustering，跨 5 seed **选同一套题**，更稳定，MAE/RMSE/Spearman $\rho$/Kendall $\tau$ 全更好。
- 只有在极穷数据 (<50 models) 时 IRT 略优。
- 快：无需 EM 迭代拟合 IRT，几十秒出子集。

## IRT 数学题策展 (PMLR 273)

另一支工作 *IRT-based math benchmark curation* 用 discrimination 参数 $a_j$ 选高辨识题，发现：

- 高 $a_j$ 题能显著提升排序可靠性（去掉噪音题）
- 与 mRMR 一致：**好题 = 能把模型拉开差距的题**

## BenchBench 自身是什么？

搜索里 BenchBench 也指 **benchmark 的 benchmark**：自动评估 “生成 benchmark 的方法好不好”。不是你评测主线，可当 optional 参考，评估你的子集生成器是否 self-consistent。

### Tutorial links (代码层面)

```bash
# efficiently benchmark tutorial (HELM repo hands-on merged into artifact but concept)
pip install scikit-learn
# see toy_mrmr_select.py next door for 30 LOC demo
```

## 对比表

| 方法 | 选题依据 | 稳定性 | 需大矩阵? | 每日 RL 推荐度 |
|---|---|---|---|---|
| metabench IRT | $I(\theta)$ Fisher | 中 (seed 抖动) | 是 (5k models) | 周检用 (能力监控) |
| mRMR KR | $I(f;y)-redundancy$ | **高** (固定套题) | 否 (50+ models 就行) | **每日用** |
| DIoR random | random | 低 | 否 | 调研 baseline |
| clustering | k-means on embedding | 中低 | 否 | 不推荐 |

> 与 GLM-5.2 的启示呼应：没有 one-size-fits-all，**task-dependent**，但这次选的是评测方法本身。

##，为啥同一套题很重要

- 固定题目 → cache prompt / KV复用，vLLM rollout 压测那套也受益
- 固定题目 → 可做 disjoint repeat 版本 (留一小块只做防过拟核验，不进训练选题)
