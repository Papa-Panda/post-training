# bench-efficiency — 用 1% 的题拿到 99% 的排序信号

> 为什么关心：post-training / agentic RL 每天都要跑 eval，全量 28k 题太贵。**eval 压缩**的另一种含义：不压上下文，压**评测集本身**。1% 子集 ≈ 99% 排序保真，就可以做每日回归。

```
eval/
├── context-compression/  # Factory式压缩失真探测 (之前 eval-compression)
└── bench-efficiency/     # 你现在看的：metabench / mRMR 高效评测
```

## 核心问题

- 全量 eval = 慢 + 贵，挡住 RL 飞轮
- 子集 eval 能否做到 **有效 eval**（effective eval）？
- 代表作：**metabench** (IRT 蒸馏 28k→858, <3%)、**BenchBench / Efficient Benchmarking** (HELM DIoR, mRMR 特征选择, x100 成本节省)

## 三类路线对比

| 路线 | 代表 | 原理 | 优点 | 缺点 |
|---|---|---|---|---|
| **IRT 心理测量** | metabench arXiv:2407.12844 | 用 2PL/3PL IRT $P_{ij}=c+(1-c)/(1+e^{-a(\theta-b)})$，信息量 $I(\theta)$ 选高辨识度题 | 可解释能力 $\theta$，点分+能力分，Spearman r=0.94 与总分 | 需 n>5k 模型 item-response 矩阵 |
| **特征选择回归** | mRMR arXiv:2605.25773 | 把题当特征 $y=w^Tx$, mRMR 最大相关最小冗余选题 + Kernel Ridge | 比 IRT/聚类 稳定跨 seed、同一套题、MAE/RMSE/Spearman/Kendall 更好 | 极穷数据下不如 IRT |
| **DIoR / 聚类** | HELM Efficient Bench 2308.11696 | 决策影响可靠性 DIoR，看子集排序能否保持 | x100 成本，直观 | 方差大，seed 不稳定 |

> 结论：短验 vs 长验同 GLM-5.2 那篇一样，**任务依赖**。daily RL 用 mRMR 1% 子集，每周全量校准 + metabench 潜变量能力监控。

##，怎么用

1. `01_metabench.md` — IRT 蒸馏数学、858 题构造、RM RMSE 1.24% 单科 / 0.58% 总分、HF `HCAI/metabench`
2. `02_benchbench_efficient.md` — DIoR、mRMR、Kernel Ridge、为何同一套题跨 seed
3. `03_practical.md` — 给你 RL loop 的 1% 子集挑选手冊 + 防过拟合护栏

Maintained bilingual, LaTeX-first, no employer IDs. Papers in `papers.md`, toy code in `code/`.
