# Agent Error Dataset: Scaling 50000 Error–Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training

> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/day-02-aed/

## 元信息

- Title: Agent Error Dataset: Scaling 50000 Error–Diagnosis Pairs for Failure Analysis and Error-Aware Post-Training
- Authors: Zhu 等
- Link / arXiv: https://arxiv.org/abs/2609.40111
- Date of publication: 2026-09-30
- Tags: [failure-data, diagnosis, citation-first]
- 与前篇的关系：Day 01 SPADE 是合成环境（Designer 造题）；Day 02 反向，从真实 rollout 的失败里挖数据，失败样本第一次当成主资产。SPADE 的验证在环境 verifier 里，AED 的验证在 matched replay 里。

## 一句话总结

把 agent 的失败当成主资产而不是废料：50,228 条 error–diagnosis 对，来自 9,961 个任务；五步 AET 流水线用 citation-first 的 judge 做诊断、用 matched replay 验证修复，再分出诊断与恢复两个 SFT 视图。Qwen3-8B 做诊断，teacher-label 的 exact-step 一致率从 47.19% 升到 63.56%。

## 3 个问题

1. 诊断为什么是可密集学习的“结构化判断”任务？
2. 修复监督为什么高度依赖环境？
3. citation-first（每条诊断必须引用原始 trace 里的 token）防住了什么？没有它，judge 会怎么偷懒？

## 核心机制

1. **Motivation**：合成数据都在教模型怎么做对的事，失败样本一直被当成废料丢掉。但失败里藏着“错在哪、为什么错、怎么修”的结构化信息，而且真实 rollout 的失败是自然分布的，不用编。
2. **方法**：五步 AET 流水线：收集真实 rollout 里的自然失败 → citation-first 的 judge 出诊断与修复建议（每条诊断必须引用原始 trace 里的 token）→ 对可复现任务做 matched replay 验证修复动作 → 分别建成“诊断 SFT 视图”和“actor 恢复 SFT 视图”。
3. **关键数字**（论文自报，经本系列 papers.md 注释核实）：50,228 条 error–diagnosis 对，来自 9,961 个任务，33 个环境、19 个 harness、23 个策略模型；Qwen3-8B 做诊断，teacher-label 的 exact-step 一致率从 47.19% 升到 63.56%。警告：只训“出错后怎么修动作”，在 WebShop-lite 上 +6.67 个点，在 TextQuest 上反而 −8.33 个点。

## 边界

- 修复监督高度依赖环境：WebShop-lite +6.67，TextQuest −8.33——“学会修”不等于“学会做”，换个环境可能倒扣。
- 诊断质量挂在 judge 身上：citation-first 要求引用 trace token，但引用不等于因果，judge 可能引对了位置、说错了原因。
- 50,228 对来自 23 个策略模型的 rollout，分布随模型变；新一代模型失败模式不同，数据集会老化。

## 迁移

- “失败当主资产”可搬到 coding agent / 数学 RL：错题本比做对的题更值钱，但要配 citation 式的归因约束，不然就是拿错答案训。
- 与 Day 01 SPADE 构成对照：SPADE 是合成环境、AED 是真实失败；SPADE 的验证在环境 verifier 里，AED 的验证在 matched replay 里——两种“验证”的活法。
- 如果复现，第一步先攒带完整 trace 的真实失败 rollout，再谈 judge——没有 trace，citation-first 无从谈起。

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？（judge 的 citation 是否真指向因果 token；matched replay 的“可复现”筛选是否偏向简单任务）
- 换成今天的模型规模，这个方法还成立吗？
