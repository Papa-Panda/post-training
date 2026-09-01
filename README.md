# post-training

> GitHub: [`Papa-Panda/post-training`](https://github.com/Papa-Panda/post-training)。顶层按轨道拆分，infra 实战统一在 `ai-infra/`。

45-day journey from ML for Infra (data center predictive modeling, autoscaling, $400M savings) to Post-training / Agentic RL Infra.

- **Goal**: 6-12 month transition in coding data + post-training → Staff+ / E6+ at large Bay Area / SF frontier labs
- **Structure**:
  - `ai-infra/` — Infra Systems 实战：DDP/FSDP/pjit/checkpoint/H100 beyond-7B/vLLM（原 `day-01`~`day-10` 已收敛进来）
  - `ICL/` — In-Context Learning 知识脉络与数学（含 trajectory-error prompting 的四种用法、code 版 Socratic-SWE）
  - `grpo-vs-ppo/` — PPO vs GRPO 对比，含 GLM-5.2 回归 PPO 讨论
  - `vllm-rollout/` — vLLM rollout 压测 & 失败 taxonomy
  - `eval-context-compression/` — 评估压缩：Factory 方法 + Hermes probe harness + 三类压缩对比（原 eval-compression）
  - `eval-bench-efficiency/` — 高效评测：metabench IRT 蒸馏 28632→858 (<3%) + mRMR 特征选择 DIoR x100
  - `eval/` — 评估索引（指向上面两个平铺 track）
  - `model-aware-data-curation/` — 梯度驱动的数据价值、覆盖、主动生成与持续学习闭环（跨论文系统专题，不重复 `ai-data`）
  - [`harness-engineering/`](harness-engineering/README.md) — Agent Runtime、context/memory、workflow/subagents、可观测性与可回归的自改进 Harness
  - `ai-data/` — Data-centric papers：coding data / SFT / RL data / curation
  - `ai_daily.csv` — source of truth，45 天 2026-08-02 → 2026-09-15
- **Tracks**: Infra Systems (18d) / RL Training (11d) / Reasoning Data (8d) / Papers (5d) / Reflection (3d)
- **Daily sync**: 06:00 America/Los_Angeles → 每日问题库 app (dual tab: investment / AI learning)

This repo is the public artifact for the transition - each day: code + NOTES with 3 numbers (single-GPU time, 2-GPU time, comm overhead). Infra labs 现在统一在 `ai-infra/day-0x/`.

Daily habit: meditation + 王继武健身十六式打卡 08:30.

> Eye constraint: 20/20/20 reminders explicitly stopped 2026-07-31, only keeping evening winddown 21:00 + nightly log 21:30.
