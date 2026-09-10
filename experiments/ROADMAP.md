# 知识图谱（Roadmap）

> 每条 claim 必须挂到这张图的一个节点上（模板里的"知识图谱节点"字段）。
> 图的用途就两个：看到哪里是**空白**（该去找 claim 的地方），
> 哪里已经有结论（不要重复测）。

## 核心地图：model-in-the-loop 数据策展（主线）

设候选样本为 $z=(x,y)$ ，当前代理模型参数为 $\theta$ ，目标验证集为 $V$ ：

$$g_z=-\nabla_\theta\log p_\theta(y\mid x),\qquad \bar g_V=\frac1{|V|}\sum_{v\in V}g_v.$$

| 线（节点名） | 核心问题 | 代表方法 | repo 章节 |
|---|---|---|---|
| 归因基础 | 谁导致了这个行为？ | Influence Functions, TracIn, TRAK | [02](../model-aware-data-curation/02_attribution_to_targeting.md) |
| 目标化选择 | 哪些样本推动目标能力？ | LESS, DataInf, GradAlign | [02](../model-aware-data-curation/02_attribution_to_targeting.md) |
| 无梯度估值 | 哪条 demonstration 能立刻改善目标探针集？ | RICo, Nuggets / ICP | [10](../model-aware-data-curation/10_rico_icl_valuation.md) |
| 多样性覆盖 | 当前梯度空间还缺哪些方向？ | Vendi, D4/SemDeDup, G-Vendi, FisherSFT | [03](../model-aware-data-curation/03_gradient_coverage.md) |
| 集合协调 | 候选会不会抵消已选集合的平均更新？ | SPICE | [09](../model-aware-data-curation/09_spice_information_conflict.md) |
| 主动生成 | 如何补稀疏区域而非继续堆重复样本？ | Prismatic Synthesis | [04](../model-aware-data-curation/04_prismatic_synthesis.md) |
| 安全/持续学习 | 新能力如何不破坏明确要保护的能力？ | GrADS, OGS | [05](../model-aware-data-curation/05_safety_continual_learning.md) |
| 可学性门 | 孤立方向是稀缺能力，还是当前模型学不进去？ | RLVR unlearnability 分析 | [05](../model-aware-data-curation/05_safety_continual_learning.md) |
| 系统实现 | 梯度存、近似算、调度怎么做？ | gradient datastore, 近似计算 | [06](../model-aware-data-curation/06_system_architecture.md) |
| coding 闭环 | 失败簇 → 生成 → 验证 → 训练 → 回归 | coding data flywheel | [07](../model-aware-data-curation/07_coding_data_flywheel.md) |

## 外围（非主线，claim 优先级低）

- **ai-infra 补充**（面试/系统知识）：`vllm-rollout/`, `grpo-vs-ppo/`,
  `gpu-architecture/`, `harness-engineering/` —— 这些线的 claim 以理解性为主，
  不强制配实验。
- **physical AI 背景音**：`physical-ai/` —— 只收知识结构，不设 claim 指标。

## 怎么用

1. 新 claim 在模板里填"知识图谱节点"（填上表第一列的节点名）。
2. 找下一个 claim 时，先看图找空白节点，别在已有结论的节点上重复测。
3. 每月自测时，对着 `claims/` 扫一遍：哪些节点已有 claim，哪些还是空白，
   空白的就是下个月的 hunting ground。
