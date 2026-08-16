# 01 metabench — 用 IRT 把 6 个 benchmark 压到 <3%

> Source: Metabench — Sparse Benchmarking of Large Language Models, arXiv:2407.12844 (ICLR 2025), OpenReview, GitHub adkipnis/metabench, HF HCAI/metabench.

## 问题

6 个常用 benchmark ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande 合计 d=28,632 题，n>5000 LLMs 跑全量太贵。能否用 858 题 (<3%) 还原排序和能力？

## 构造 5 步

1. **Collect**: n≈5000+ Models × d 题的 item-wise accuracy 矩阵 $X \in \{0,1\}^{n\times d}$，来源 OpenLLM Leaderboard
2. **Variance filter**: 去掉极低方差题 (人人都对/错)
3. **CV subsample 350**: 每 benchmark 随机抽 350 题做 cross-validated 预筛，保证 domain 覆盖
4. **IRT fit**: 拟合多种 IRT variants:
   - 2PL: $P_{ij}= \sigma(a_j(\theta_i-b_j))$, 3PL/4PL 加猜测 $c_j$ 和上界 $d_j$
   - a_j = 辨识度 discrimination, b_j = 难度 difficulty
   - Fisher Information $I_j(\theta)=a_j^2 P_j(1-P_j)/(1-c)$ 在 $\theta$ 附近高的题更 informativo
5. **Information filtering + GAM**: 按 $I_j(\hat\theta)$ 选 top，留每 benchmark ~100-200 题，总计 858。用 GAM $y = s(\theta)$ 从潜变量重建原始分数，并做 factor analysis 找 single underlying factor $g$.

## 结果 (median RMSE)

| Benchmark | #items | 2PL RMSE | 3PL RMSE |
|---|---|---|---|
| ARC | ~ | 1.2% | ~1.2% |
| GSM8K | ~ | 1.3% | similar |
| HellaSwag | ~ | 0.9% | ~ |
| MMLU | ~ | 0.8% | ~ |
| 总体单个 benchmark | — | **1.24% 平均** | — |
| 6 科合成总分 | 858 | **0.58% RMSE** | — |
| 潜因子 $g$ vs 总分 | 1 维 | **Spearman r=0.94** | — |

> 点分估计 (point scores) 和能力估计 (ability estimator $\hat\theta$) 都提供，$\hat\theta$ 对短子集更鲁棒。

## Adaptive Testing 第2版

metabench v2 支持自适应选题：先用难易中等的 seed 题估 $\theta$，再按 $I(\theta)$ 顺序挑下一题，≈20 题内收敛到 within 1% error，适合在线持续监控。

## Hands-on

```bash
git clone https://github.com/adkipnis/metabench
# R 重建分数 (论文提供 reconstruct.R)
Rscript reconstruct.R --input your_model_preds.csv --benchmark mmlu
# HF 快速加载 858 题
python -c "from datasets import load_dataset; ds=load_dataset('HCAI/metabench'); print(ds)"
```

> 你的 RL loop：用 858 题做 daily gate，周度再跑全量校准 drift。

## takeaways for post-training

- IRT 不是魔术，是心理学 50 年老工具，好处是 **可解释的 $\theta$ + 信息量理论保证**
- 需要先有大矩阵 X (5000+模型)，冷启动小 lab 用 HF 预训练好的 a_j,b_j 即可
- 与 GLM-5.2 那篇同理：**评价方法要随任务形态共变**
