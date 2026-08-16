# 02 Factory Method — Dec 2025 Evaluating Compression

> Source: Factory AI news `evaluating-compression` Dec 2025 — defines agent context compression eval differently from prompt / KV-cache compression.

## Factory 的定义

> *"We don't evaluate the compressor as a summarizer. We evaluate the compressed state as a working memory."*

即：压缩不是摘要任务，是 **working memory 任务**。压缩后 agent 还能不能继续干活？

他们明确区分：

- ❌ **Not prompt compression**: 不是把长 prompt 变短喂给 LLM 降本
- ❌ **Not KV-cache compression**: 不是 TurboQuant / LeanKV 那种显存优化
- ✅ **Agent context compression**: session 级的、带 tool traces 的状态管理

## 6 维度 Dimensions (0-5)

Factory + Hermes harness 统一用 6 维，每个 0-5，judge LLM 打分，平均成 quality score：

| 维度 | 问了什么 | 典型失败 |
|------|----------|----------|
| **accuracy** | 压缩后事实对吗？ | hallucinate 不存在的文件 |
| **context_awareness** | 还记得用户当前阶段目标吗？ | 把 "fix bug" 记成 "add feature" |
| **artifact_trail** | 能追溯自己产生过的代码产物变更吗？ | 丢掉 commit diff 导致 revert |
| **completeness** | 待做事项 / 约束还有多少没做？ | 漏掉 test coverage requirement |
| **continuity** | 对话连贯么？压缩后有时间倒错么？ | 把先发生的认为是后发生的 |
| **instruction_following** | 最终还能遵从 system 指令么？ | system 说 "no push without test" 却 push 了 |

> Scoreboard framing not adopted — 他们刻意不用 leaderboard，因为压缩质量和任务形态强相关，没法跨 task 一表排。

## Probe 设计

Instead of one summary eval, they use **30 probe questions per fixture**:

```
fixture (real 80K agent session)
  → compressor.compress(fixture) → compressed_state (20K)
  → for each probe q in 30:
        answer = agent.ask(compressed_state, q)
        score = judge_llm(q, compressed_state, answer, gold)
```

Probe 分三类：

1. **Retention** (12): 压缩前显式出现过的 fact，能不能召回？
   - e.g. "What was the failing test name in round 2?"
2. **Inference** (10): 需要从多条 trace 中综合
   - e.g. "Why did we switch from approach A to B?"
3. **Continuation** (8): 基于压缩后状态，能否做下一步合理决策？
   - e.g. "What should you do next to satisfy the PR description?"

Cost: 30 probes * judge calls ≈ 60 LLM calls / fixture — cheap enough for CI, expensive enough to catch drift.

## 为什么这比 prior art 好

Prior compression eval (arxiv 2402.00861 "Compression as evaluation") 把 compression = lower description length → lower perplexity → smarter model，这套只适用于 LM 本身，不能评 agent memory。

Factory 指出：

> *"A compressor that achieves 10x ratio but breaks continuity is worse than a 2x compressor that preserves instruction_following."*

所以比例 `ratio = orig / comp` 只是 infra 指标，quality 必须 behavioral。

## 落地 Insight for us

- Long-horizon RL (GLM-5.2) 遇到同样问题：GRPO 要求 group 内样本同分布可比，但长任务压缩后 sub-trajectory 长度方差大 → group 组不齐 → 被迫回 PPO critic token-level advantage【附在 grpo-vs-ppo/track】
- Compression eval 其实在测 **critic 好不好学**：如果压缩后 history 失真，value 估不准，PPO 也救不了
