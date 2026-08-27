# Magicoder: Empowering Code Generation with OSS-Instruct

## 元信息
- Title: Magicoder: Empowering Code Generation with OSS-Instruct
- Authors / Org: University of Illinois Urbana-Champaign / Tsinghua University
- Link / arXiv: https://arxiv.org/abs/2312.02120
- Date read: 2026-08-27
- Tags: [synthetic-data, coding-data, sft, instruction-tuning, open-source-code, diversity, decontamination, quality]

## 一句话总结
OSS-Instruct 从开源代码中抽取 80K 个短片段作为多样化现实锚点，让教师模型据此合成 coding problem + solution，经去重与 benchmark decontamination 后得到约 75K 条指令数据，减少只靠少量人工种子或固定演化规则带来的分布偏置。

## 和之前工作的关系
- **知识图谱位置**：位于合成指令线的 coding 专用分叉：Day21 Self-Instruct 通用自举 → Day22 Evol-Instruct 显式提升复杂度 → Day27 OSS-Instruct 用真实开源代码片段扩展任务来源；再向下连接 Day06 Phi-1 的合成 code 数据与 Day16 Qwen2.5-Coder 的 execution-filter。
- **接了哪条线**：接 Day21/22 的 synthetic SFT 线，但把生成起点从少量人工 seed instructions / 固定改写 heuristics 换成大规模开源代码片段。
- **补了哪个短板**：Self-Instruct 的 code 版本 Code Alpaca 仅从 21 个 seed tasks 扩展，Code Evol-Instruct 依赖 5 类演化规则；OSS-Instruct 用不同语言、结构与语义的真实代码片段约束生成，提升任务的现实性、多样性与可控性。
- **替代 / 分叉 / 改进**：它不是 Evol-Instruct 的替代，而是正交的数据源分叉；一个改变任务的来源与语境，一个从已有指令继续演化复杂度，两者可以串联或混合。
- **直接对比 Day22**：Day22 从已有 instruction 出发问“怎样把题变难、变广”；Day27 从 source code 出发问“怎样产生更真实、更分散的新题”。前者改写难度分布，后者改写 seed/source 分布。
- **回接 Day16**：Day27 解决上游“题从哪里来”，Day16 的 execution-filter 解决下游“答案是否可执行/正确”；两者合起来才是 source-grounded synthesis → execution validation 的 coding-data 闭环。

## 为什么今天读它
前两天已补齐通用合成指令（Day21）、复杂度演化（Day22）和偏好池构造（Day26），但 coding data 还缺一条“从真实代码资产生成 SFT 任务”的公开配方。OSS-Instruct 正好把开源代码变成可控的合成种子：对 SFT 是更现实的 instruction-response pool；对 RL data 可进一步接 execution tests、难度分层和可验证筛选。这里只研究数据来源、生成、清洗、多样性与防污染，不展开训练算法。

## 今天的 3 问
1. 从 starcoderdata 的 80K 个 code documents 中随机截取 1–15 行、覆盖 9 种语言，经过哪些去重与 benchmark decontamination 才落成约 75K 条数据？哪些噪声被有意保留，为什么？
2. 对比 Day22 Evol-Instruct：真实代码 snippet grounding 与固定 complexity heuristics 分别改变任务分布的哪一维；如何组合两者而不牺牲多样性或放大教师模型偏差？
3. 回接 Day16 Qwen2.5-Coder：若为 OSS-Instruct 增加 parser / compile / unit-test execution filter，应如何记录 provenance、失败类型与难度，才能把 SFT pool 继续转成可验证 RL data？

## 核心
1. **Motivation**: 只靠少量人工 seed tasks 或固定演化规则生成 coding instructions，容易继承教师模型与预定义任务的偏置，覆盖不足。
2. **Data Pipeline**: starcoderdata 开源代码文档 → 每个文档随机抽取 1–15 行 seed snippet → 教师模型生成 coding problem + solution → 去除重复样本 / 重复 seed → benchmark decontamination → 约 75K 条 OSS-Instruct 数据。
3. **Key Tricks**: 待读后重点填写 seed 语言配比、生成 prompt 如何约束“借鉴而非复述”、去重与 contamination 匹配规则。
4. **Results**: 待读后记录数据多样性、相似度与下游消融；只讨论数据配方带来的变化，不展开优化或训练算法。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：用真实 repository snippet 作为任务生成锚点；为每条合成题保留 source language、repo/document provenance、生成 prompt 版本、去重簇和 execution status。
- Infra 视角：将 snippet sampling、generation、dedup、decontamination、execution validation 拆成可重放 stage，按语言/库/任务类型监控覆盖与失败率。

## 疑问 / 下一步
- OSS-Instruct 只移除重复与 benchmark contamination，却有意保留部分不完整 solution；加入 execution filtering 后，怎样避免只留下容易验证的短算法题、反而损失真实长尾任务？

## 原文金句 (1-2句)
> 待读后补充。
