# 06 Trajectory Error → 通用 Prompt

> 来自讨论：出于对 trajectory 的 error analysis 加一个一般性的 prompt 提示，有这种做法吗？现在还有效吗？

## 有，常见且仍有效

现在叫法：**error-aware system prompt / principle learning / guideline extraction**

2024-26 长上下文 + 推理模型下依然吃香：many-shot 能塞几百条 principle，推理模型会当 checklist 用，零成本不改权重。

## 代表论文

1. **Reflexion** — Shinn et al. 2023 *Reflexion: Language Agents with Verbal Reinforcement Learning*。失败轨迹 → verbal reflection 塞回 memory，下一次重试。
2. **LEAP** — Zhang et al. 2024 *In-Context Principle Learning from Mistakes*。故意让模型在 few-shot 上犯错，再反思提炼通用 principle，拼回 prompt 去答新题。原文：`LEAP does not require any more examples than standard few-shot`。7.5% ↑ DROP / 3.3% ↑ HotpotQA on GPT-4。
3. **ExpeL** — Zhao et al. 2023 *LLM Agents Are Experiential Learners*。收集成功/失败池，抽象跨任务 insights，评测时召回。流程：(1) collect (2) extract (3) apply。
4. **AutoGuide** — Fu et al. 2024 *Automated Generation and Selection of Context-Aware Guidelines*。从 offline 经验自动生成条件式 guideline，`state-aware`，可无缝接入任何 prompt agent。

## 四种用法（翻译）

1. **System 常驻军规**：直接塞 system，`We avoid ...`，别臆造文件名，先校验工具返回。
2. **反洗成 SFT/RL 数据（最值钱）**：把 principle 去筛/重写你收集的 trajectory，符合的当正例，违反的当反例，凑成 SFT/DPO 对子去训，训完天生懂，省 token。
3. **当 Reward Model Verifier**：principle 编成可执行 check，RL 时当 rule-based reward。
4. **人工写 Agent SOP**：给人+机共用，统一 oncall / 数据清洗规范。

你现在做 coding agent，**2 最值**：提炼完去刷数据比一直修 prompt 稳。1 最快上线，3 最适合 RLVR 约束，4 沉淀成 infra 知识。

## 实践坑

- 原则抽象过宽 → 稀释注意力，过窄 → 过拟合某批 error
- 长 prompt 会推挤 KV，需做 guideline selection（AutoGuide 的 top-k）
- 需做消融：加了后坏 case 真降了没，别凭感觉

检索：Reflexion → ExpeL → LEAP → AutoGuide是一条进化线。
