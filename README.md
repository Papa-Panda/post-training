# post-training-rl-infra

> GitHub: `post-training-rl-infra` → 本地简称 **post-training**，顶层现在按轨道拆分，infra 实战都进 `rl-infra/`。

45-day journey from ML for Infra (data center predictive modeling, autoscaling, $400M savings) to Post-training / Agentic RL Infra.

- **Goal**: 6-12 month transition in coding data + post-training → Staff+ / E6+ at large Bay Area / SF frontier labs
- **Structure**:
  - `rl-infra/` — Infra Systems 实战：DDP/FSDP/pjit/checkpoint/H100 beyond-7B/vLLM（原 `day-01`~`day-10` 已收敛进来）
  - `ICL/` — In-Context Learning 知识脉络与数学（含 trajectory-error prompting 的四种用法、code 版 Socratic-SWE）
  - `ai-data/` — Data-centric papers：coding data / SFT / RL data / curation
  - `ai_daily.csv` — source of truth，45 天 2026-08-02 → 2026-09-15
- **Tracks**: Infra Systems (18d) / RL Training (11d) / Reasoning Data (8d) / Papers (5d) / Reflection (3d)
- **Daily sync**: 06:00 America/Los_Angeles → 每日问题库 app (dual tab: investment / AI learning)

This repo is the public artifact for the transition - each day: code + NOTES with 3 numbers (single-GPU time, 2-GPU time, comm overhead). Infra labs 现在统一在 `rl-infra/day-0x/`.

Daily habit: meditation + 王继武健身十六式打卡 08:30.

> Eye constraint: 20/20/20 reminders explicitly stopped 2026-07-31, only keeping evening winddown 21:00 + nightly log 21:30.
