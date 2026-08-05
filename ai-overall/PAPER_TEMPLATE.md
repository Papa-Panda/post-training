# Paper 模板 — ai-overall 通用版

> 复制这个模板到 `papers/{year}_{short-name}/NOTES.md`

## 元信息
- Title:
- Authors / Org:
- Link / arXiv:
- Date read: YYYY-MM-DD
- Tags: [agentic-rl, post-training, infra, ai4ai, rsi, eval, reasoning, system, coding-agent]
- Thread: ai overall

## 一句话总结
这篇 paper 解决了什么 AI 问题，用了什么关键方法，效果怎样。

## 核心

1.  **Motivation / RSI视角**: 为什么要做？baseline 为什么不行？跟 Recursive Self-Improvement 的关系？
2.  **System / Method**: 系统架构 → 训练/推理如何对齐 → 关键 operator / search / feedback 机制
3.  **Training Details**: SFT / RL 怎么做的？数据如何去重/防泄漏？Reward / Verifiable signal 是什么？
4.  **Key Tricks**: 3个最值得抄的细节（operator 设计、异步搜索、experience prior、resource约束）
5.  **Results**: 在什么 bench 上，用什么 budget，相对 base 提升多少？和 SOTA (GPT-5.5/5.6, Kimi K3 等) 对比？

## 泛化 / Transfer

- 方法在 held-out 上是否 transfer？模型 vs 框架 哪个贡献更大？
- 对你 Infra → Post-training 迁移的 1-2 个直接启发：
- Infra 视角：可扩展性 / 成本 / 评测自动化 / 可复现性：

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题
- 如果要复现 / 小规模试，第一个实验做什么？

## 原文金句 (1-2句)

>
