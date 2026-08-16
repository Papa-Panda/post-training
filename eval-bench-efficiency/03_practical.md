# 03 Practical — 给 agentic RL 的 1% 子集手册

## 目标：1% 题 ≈ 99% 排序

你现在每天跑 6 bench 28k 题，日均 2-3h A100。切到 1% (280 题, metabench 858 更宽松) + mRMR 固定套题，可把 **daily eval 5-10min 内**。

## 三档方案

### 1. Daily gate (1% mRMR)
- k=200-350 (每科 ~50)
- 固定 seed，Kernel Ridge 回归重建总分
- CI 关卡：若 rank 相关 <0.98 或 std 过高 → 触发全量复核

### 2. Weekly calibration (metabench 858, <3%)
- 跑完整 858，估 6 科点分 + 潜因子 $\hat\theta_g$
- 对比 daily gate 的重建误差，漂移 >1.5% RMSE → 重新选题
- 能力维度 $\theta$ 更能捕捉 long-horizon 提升 (代码/推理)

### 3. Monthly full (100%)
- 全量 28k，当作 ground-truth，用来更新 $X$ 矩阵，重新算 $I(f;y)$ 和 IRT a_j,b_j
- 同时生成 **disjoint repeat**：留 10% 题完全不进选题池，只做防过拟

##，护栏 Guardrails

1. **Hard subset**: 人工加 20-30 道超难但可验的 coding `hard` (SWE-bench Pro 子集)，mRMR 会倾向简单高方差题，难的要手动保
2. **Rank correlation monitor**: 每天算 Spearman $\rho$(subset vs weekly full-proxy)，<0.95 报警
3. **Disjoint repeat**: metabench 论文也建议留 repeat 版本避免过拟合到公开子集，Factory 那篇压缩评测同理
4. **Per-task tracking**: agentic RL 的长 horizon (SWE-Marathon) 别只看平均分，看 per-trajectory artifact_trail 完整度，那是 GLM-5.2 讨论的领域

## x100 节省怎么来的

HELM 报告：随机 1% 也能在 80% 情况下保序，但 mRMR 让 1% 在 **95%+ 置信**保序，batch 推理 + vLLM prefix-cache → tokens/task 下降同理。

$ Cost_{daily} \propto \frac{k}{d} \cdot \frac{1}{\text{cache hit}} $  k=285,d=28632 → 0.01，但 prefix hit 3-5x, 综合 ≈0.003

## 检表 (RL 飞轮专用)

- [ ] 固定 `bench-efficiency/mrmr_285_seed42.json` 提交到 repo，不随手变
- [ ] cache 去重：跑前 `vllm prefix` 热身
- [ ] fail 开放端：超长 0.5% fail 不是 noise，直接进全量子集复核
- [ ] 与 `vllm-rollout` 联动：goodput 拐点处评测吞吐同样拐，别在 OOM 区评

## 与 context-compression 的关系

- context-compression 评 “模型忘记了啥” (probe 6 维)
- bench-efficiency 评 “题集能不能代表能力”

俩加起来 = 可信 daily eval：**少跑题** + **少丢上下文**，才能做到 post-training 24h 闭环。

> 小专题命名建议见 `../README.md`。
