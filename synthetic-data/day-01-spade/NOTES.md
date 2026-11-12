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

1. 设计一个对照实验，区分“hint-regret 选出了好环境”与“Designer 在用 hint 泄题刷 regret”：固定环境集合，三臂——A 原生 hint；B 同样长度的无关文本替换 hint（打乱 hint 与环境的配对）；C 去掉 hint 只留环境。都训同一个 Agent，看 hint-free 的 held-out 分数。判据：若 A≈C 且都高于 B → regret 信号真在选环境；若 A≫C → 分数里有 hint 泄漏的功劳，Designer 赚的是“钥匙”不是“锁”。再说这个设计本身的漏洞：B 的无关文本会不会本身干扰 Agent（负向控制不干净）？
2. 消融说去 Designer 训练+记忆崩到 40.5，比不训的 50.2 还低 9.7。设计一个 2×2：Designer 训练 ∈ {开，关} × 记忆库 ∈ {开，关}，四个格子各跑一轮 400 步。先预测四个格子的 held-out 排序并写理由；再指出这个实验的漏洞：关掉记忆的同时是不是也关掉了“防重复”的去重（两个变量搅在一起）？怎么拆？
3. 小 backbone 上观测到 regret 为负（hint 误导 Agent）。构造一个具体例子：什么样的 hint 会让有提示的回报低于无提示？再设计一个检验：同一批环境，hint 由“同一模型 Designer 角色”写 vs 由“更强外部模型”写，比较两组 regret 分布——若外部 hint 的负 regret 消失，说明负 regret 是 Designer 能力问题还是 hint 机制本身的问题？

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
