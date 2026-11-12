# SPADE: Self-Play in Adaptive Synthetic Executable Environments

> 📖 阅读版：https://papa-panda.github.io/post-training/synthetic-data/day-01-spade/

## 元信息

- Title: SPADE: Self-Play in Adaptive Synthetic Executable Environments
- Authors: Liu 等
- Link / arXiv: https://arxiv.org/abs/2608.19197
- Date of publication: 2026-08-19
- Tags: [environment-generation, self-play, hint-regret, verification]
- 与前篇的关系：本系列第一篇，后面 14 篇的对照基线。SPADE 回答“环境从哪长出来”，后面的 ToolLoop、Data Turnstile 回答“生成过程怎么验证、怎么修”。

## 一句话总结

同一个 LLM 分饰 Designer 与 Agent 两角，Designer 按 hint 前后 regret 差获得奖励，持续生成卡在 Agent 能力边界上的可执行环境。30B 模型一轮 400 步生成 3,310 个独立环境，Qwen3-30B-A3B 在 8 个 held-out 评测上 50.2 → 58.3。

## 3 个问题

1. hint-regret 和“学习潜力”度量有什么区别？
2. 为什么无锚定的 Designer 会主动误导 Agent？
3. 环境记忆库在增量演化里到底防住了什么——没有它，环境会怎样退化？

## 核心机制

1. **Motivation**：固定环境集跟不上 Agent 的能力，训练信号会随 Agent 变强而枯竭。SPADE 把环境本身变成可训练、可演化的对象，让环境跟着 Agent 的能力边界一起长。
2. **方法**：Environment Designer 生成完整可执行 Gym 风格环境（reset()/step()、状态转移、奖励、verifier），Reasoning Agent 在里面学。Designer 的奖励不是通过率，而是 hint 前后的 regret 差：r = E[R | 有提示] − E[R | 无提示]。Designer 必须从大规模预训练语料采样文档做 grounding，并维护环境记忆库做增量演化。hint 由 Designer 本人在环境通过校验后另起一个 prompt 写出：每环境一条、几句话，只给关键思路/策略与期望答案格式，明令禁止透露 exact answer，且不做内容校验（论文 §4.2）。
3. **关键数字**（论文自报，经本系列 papers.md 注释核实）：30B 一轮 400 步生成 3,310 个独立环境；Qwen3-30B-A3B 在 8 个 held-out 评测上 50.2 → 58.3；消融：去掉 Designer 训练和记忆直接崩到 40.5，比不训还低 9.7 个点。

## 边界

- Designer 无锚定会主动误导 Agent，消融为证——grounding 与记忆库不是装饰，是方法成立的条件。
- 数字是 Qwen3-30B-A3B 上的自报结果，换基座模型、换评测集，幅度不一定迁移。
- 环境“独立”的计数口径与 8 个 held-out 评测的覆盖面，是读原文时要核对的点，注释未展开。

## 迁移

- hint-regret 这类“能力边界度量”可以搬到别的合成数据任务：只生成刚好学得会、学了有用的样本，而不是越多越好。
- 与本系列后面的 ToolLoop、Data Turnstile 构成对照：SPADE 回答“环境从哪长出来”，后者回答“生成过程怎么验证、怎么修”。
- 如果复现，第一步先把 Designer 的 grounding 与记忆库搭起来，再谈奖励——消融证明这两块掉了，整个方法比不训还差。

## 讨论题

- 这篇的结论里，哪一条最可能被后来的工作推翻？
- 数据 pipeline 中哪个环节最容易藏 bug？（Designer 的 verifier、记忆库去重、hint 的设计）
- 换成今天的模型规模，这个方法还成立吗？
