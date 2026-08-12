# 07 Coding Data 专用版

> 有专门做 coding data 的吗？

## 有，三条线都已切到 code

### 1) Socratic-SWE — 你问的原理的 code 闭环

Paper：*Socratic-SWE: Self-Evolving Coding Agents via Trace-Derived Agent Skills* (arXiv 2606.07412 2025)

Abstract 要点：不把 trace 只当 reward证据，直接蒸馏成 structured skills，总结高频失败和有效 repair 模式，再去定向生成新 repo 任务，execution 验证 + solver-gradient 对齐打分。三轮后 SWE-bench Verified 50.4%。

原文：*reuses agent's historical solving traces as source of training signal... distills them into structured agent skills that summarize recurring failures and effective repair patterns.*

这就是你 06 里的 **2** 的完美 code 版：trace → skill → synthetic task → SFT。

### 2) SWE-Gym / SWE-bench 数据流

- 收集 SWE-Agent / OpenHands 在真实 repo 的成功/失败轨迹（几万条）
- 过滤可执行 patch，配测试
- 正/反例直接当 SFT+DPO 数据，刷榜最稳，当前开源 SOTA 训练集

### 3) CYCLE / Self-Debug / SelfEvolve

逻辑：执行反馈 → 解释错误 → 重写 code，再执行。

- *CYCLE: Learning to Self-Refine* (2023) 是鼻祖：用 `python execution trace` 当反馈训模型
- Self-Debugging (Chen et al. 2024) 用 few-shot 展示如何读报错
- 和你之前讨论的 trajectory-error prompting 打通：把 `common pitfall` 编成 rule 再拿去蒸数据

## 落到你现在的 coding data 工作

- 你现有字段 `ai-data/papers/` 已有 StarCoder2 / Phi-1 / Llama 3 15T / DeepSeek-V3 的 curation，可加一列 `error-principle`
- 下一步：把你自己跑 SWE-bench Lite 的失败 log 聚类，提炼 5 条 principle，丢到 `ICL/06` 然后用它们去洗 `ai-data` 的 synthesis prompt
- 评测：单独测 induction head 是否捕捉 `def test_xxx` → `assert` 模式，比总体 pass@k 更早给信号

## Links

- Socratic-SWE https://arxiv.org/abs/2606.07412
- SWE-Gym https://arxiv.org/abs/2412.21139
- CYCLE https://arxiv.org/abs/2305.02309
- Self-Debugging https://arxiv.org/abs/2304.12243
