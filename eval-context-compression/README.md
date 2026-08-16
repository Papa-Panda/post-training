# eval-compression — 评估压缩 Evaluation for Agent Context Compression

> 长程 agent 任务里，上下文一定会爆。压缩是必经之路，但压缩的好坏不能只看 perplexity。

Long-horizon agent sessions inevitably exceed context windows. Compression is mandatory infrastructure, but quality can't be measured by token count or ppl alone.

This track studies **how to evaluate compression**, not how to compress.

## 为什么需要 Why needed

- **Context = memory**: SWE / agentic RL 工程里，session 轻松 200K+ tokens，>1M 也不稀奇。
- **压缩 = infra**: Factory AI, Z.ai SLIME, Hermes-agent 都把 `ContextCompressor` 当作一等公民，和 KV-cache、rollout 并列。
- **评测盲区**: Unit test 全绿，线上 summary 却丢关键 artifact/指令 — 传统 eval 捕捉不到。
- Perplexity 在 50% sparse 下还能 hold，但下游 task 已崩 [arxiv:2409.11233] — 需要 probe-based, judge-based eval。

> Factory Dec 2025: *"Scoreboard framing is not adopted — we need probe questions from compressed state"* 【Factory AI: Evaluating Compression】

## 结构 Structure

```
eval-compression/
├── README.md                    # 你在看的
├── 01_why_compression_eval.md   # 问题：exceeds threshold 后的失真
├── 02_factory_method.md         # Factory 6 维度方法
├── 03_hermes_harness.md         # Hermes offline harness 实操
├── 04_prompt_vs_context_vs_kv.md # 三条压缩路线对比
├── 05_metrics.md               # 从 JS divergence 到 judge 0-5
├── code/
│   └── toy_compressor_eval.py   # 最小可跑 probe 示例
└── papers.md
```

关联 Tracks:
- `grpo-vs-ppo/06_glm52_ppo_comeback.md` — 长程任务为何从 GRPO 回归 PPO (和压缩后变长分布的同源问题)
- `vllm-rollout/` — compression 降低 `max_model_len` 后 throughput 提升、但 probe recall 怎么掉
- `ICL/` — ICL 本质也是压缩： demonstrations → task vector

## 快速判断 Quick heuristic

> 压缩前/后问自己 3 个探针：
> 1. **我还在解哪个任务？** (instruction_following)
> 2. **上一步工具返回了什么关键产物？** (artifact_trail)
> 3. **还有哪些约束没满足？** (completeness / context_awareness)

如果压缩后答不上来，summary 再短也是垃圾。

## 维护 Maintenance

Bilingual concise, LaTeX-friendly, no employer IDs. For new papers, add to `papers.md` and update `reading-log.csv` if needed.
