# GRPO vs PPO — Post-training / Agentic RL 的两条 RL 路线

> PPO 是 RLHF 的经典，GRPO 是 DeepSeekMath/R1 把 critic 干掉、靠组内相对分做 baseline 的新解。  
> This folder is a compact note in the same style as `ICL/` — math first, infra second.

## 为什么关心 Why care (agentic RL)

Post-training / agentic RL 的 rollout 贵、带工具调用、多步长：

- PPO 需要 `actor + critic + ref + reward` 4 个模型常驻，显存高，且价值网络 `V(s)` 在稀疏 reward / code 任务上很难学准。
- GRPO 把 critic 换成 group mean/std baseline，**显存省 ~30-50%，稳定省调参**，天然适配 R1/DeepSeekMath 的 `G 次采样 → 排名` 范式。
- 但 PPO 的 GAE + critic 方差更低，对需要长 horizon credit assignment 的 env 仍有优势。

> 结论：code/math 的 rule-based reward 任务优先 GRPO，密集 reward / 强价值建模场景仍看 PPO。

## 结构 Structure

```
.
├── README.md                # 你现在看的
├── 01_ppo_objective.md      # PPO 目标 / clipping / GAE
├── 02_grpo_objective.md     # GRPO / group baseline / no critic
├── 03_math_comparison.md    # 方差/偏差/内存/计算比分
├── 04_infra_tradeoffs.md    # vLLM rollout / token / multi-step
├── 05_code/                 # 最小复现 + verl/OpenRLHF 片段
└── papers.md                # 关键论文卡
```

## TL;DR 对比

| 维度 | PPO | GRPO |
|---|---|---|
| **Objective** | $E[\min(r_t A_t, \text{clip}(r_t) A_t)]$ | $E[\frac1G\sum_i \min(r_{i,t}\hat A_i, \text{clip} \hat A_i) - \beta KL]$ |
| **Advantage** | GAE: $(γλ)$ 递推，需 $V(s)$ | Group relative: $\hat A_i = \frac{r_i-μ}{σ}$，无 $V$ |
| **Model 数** | 4: policy / old / critic / ref | 2-3: policy / ref / reward (no critic) |
| **显存 peak** | 高，多 1x critic | 省 critic，适合 70B+ RL |
| **Variance** | 低 (GAE smoothing) | 高一些，但组归一化降方差 |
| **Sparse reward** | critic 难学，偏差大 | 更鲁棒，math/code 友好 |
| **多步 agentic** | 需价值传播，调参多 | 每 outcome 一个标量 reward 即可 |

参考交互：看 ICL 专题的数学比分风格，这里延续 scorecard 思路。

## 怎么用落地

1. **math/code RL**: 试 GRPO + rule reward，G=8~64，采样多样性靠温度。
2. **infra**: 用 vLLM 批量出 G 条 rollout，critic 省下来的卡给更大 batch。
3. **稳定化**: KL 0.02-0.08，advantage clip 而非只 clip ratio。
4. **评估**: 对比 PPO critic loss 震荡 vs GRPO reward-group std。

> Maintained bilingual, LaTeX-first, no employer IDs. Source papers in `papers.md`.
