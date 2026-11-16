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

1. citation-first 的作用到底是"防 judge 胡编"还是"防 judge 说对位置、说错原因"？设计一个实验：固定同一批失败 trace，trace 全程保持完整，三臂——A citation-first 且引用须校验（引用必须是 trace 里的原文子串，无效引用驳回）；B 同样的 judge 但去掉引用要求；C 引用要求保留、但引用不校验（可以现编 token）。评两个指标：diagnosis 的 teacher-label exact-step 一致率（位置），以及在位置答对的条件下、原因类别与 teacher 的一致率。判据：若 A 只在位置上 ≫B、原因给定位置后两边一样 → 引用主要是防胡编；若 A 只在原因上 ≫B、位置两边一样 → 引用主要是改善因果归因；若 A≫C≈B → 起作用的是 grounding（引用必须真指到证据上），引用的形式本身没用；若 A≈C≫B → 起作用的是引用这个动作本身。漏洞：不要拿"打乱 trace"当对照臂——打乱是销毁信息（失败 step 的因果链断了，任务本身变不可解），不是控制变量；而且被迫引用噪声 token 会主动把 judge 往随机位置上锚，C 的低分是构造出来的，什么都证明不了。