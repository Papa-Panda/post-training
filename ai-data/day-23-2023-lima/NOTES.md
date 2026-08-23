# Paper 模板 - Day 23

> 自动生成骨架 2026-08-23，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: LIMA: Less Is More for Alignment
- Authors / Org: Zhou et al. / Meta AI, Carnegie Mellon University, University of Southern California, Tel Aviv University
- Link / arXiv: https://arxiv.org/abs/2305.11206
- Date read: 2026-08-23
- Tags: [sft, alignment-data, data-quality, diversity, curation, less-is-more, coding-data]
- Folder: day-23-2023-lima
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-23-2023-lima

## 一句话总结
用仅 1,000 条经过来源、风格与多样性精心策展的 SFT 样本验证“对齐数据质量与覆盖比数量更关键”，把 Day17 LIMO / Day18 s1 的少即是多现象追溯到通用对齐数据的早期起点。

## 和之前工作的关系

> 知识图谱位置：对齐极简主线起点，Day21 Self-Instruct（合成扩量）→ Day22 Evol-Instruct（提升复杂度）与 Day23 LIMA（人工精选、质量优先）形成分叉；随后汇入 Day17 LIMO / Day18 s1 的少量高质推理数据。

- **接了哪条线：**接 Day17 LIMO、Day18 s1 的“少即是多”线，但时间上是它们的前身；同时和 Day21/22 的 synthetic scale 线形成“扩量 vs 精选”的正面对照。
- **补了哪个短板：**此前知道如何自举更多指令、如何演化复杂度，却缺少“极少量数据到底需要满足什么质量与覆盖条件”的通用对齐基线。LIMA 把来源策展、回答风格、任务多样性与近重复控制放到中心。
- **替代 / 分叉 / 改进：**它不替代 Self-Instruct / Evol-Instruct，而是分叉出 quality-first 路线；可把合成池先扩量，再用 LIMA 式门槛和 Day20 DEITA 的质量×复杂度×多样性筛成小而强的数据集。
- **对之前 Day X 的直接对比：**vs Day20 DEITA，LIMA 主要依靠人工来源与策展原则构造 1k 高质集，DEITA 则把复杂度、质量与多样性评分自动化后选 6k；今天重点判断 LIMA 的人工标准哪些能转成可规模化的数据门禁。

## 为什么今天读它

前两天沿 Day21→Day22 学了“从少量种子合成更多、更复杂指令”，今天需要补相反但关键的一半：什么时候不该继续加量。对 coding SFT，可把高质答案、任务覆盖、风格一致性和去重做成小型 gold set；对 RL data，可让它作为候选题/轨迹进入可验证池之前的质量锚点。本文只研究数据选择与策展，不展开训练算法。

## 今天的 3 问
1. LIMA 的 1,000 条数据具体由哪些来源和策展标准组成？质量、任务覆盖、回答风格与去重各自如何定义，哪些标准最可能贡献主要增益？
2. 对比 Day21 Self-Instruct / Day22 Evol-Instruct 的“合成扩量与复杂度演化”，LIMA 的 quality-first 路线在哪些任务上更强，在哪些长尾覆盖上会吃亏？能否组合成“先扩池、再精选”的数据流水线？
3. 对比 Day20 DEITA 的自动三因子选数，哪些 LIMA 人工标准能自动化为 coding data 的门禁（可执行性、边界条件覆盖、答案简洁度、近重复），并作为 SFT / RL data 的小型 gold set？

## 核心
1. **Motivation**: [待读后填写] 为什么大规模 instruction-tuning 数据未必必要？高质量少样本对齐要解决什么数据问题？
2. **Data Pipeline**: [待读后填写] 来源选择 → 人工/社区答案策展 → 质量与风格门槛 → 多样性和去重 → 形成 1k 数据集 → 如何评估。
3. **Key Tricks**: [待读后填写] 记录数据来源、筛选准则、任务分布、回答风格、去重与质量审计；不展开训练算法。
4. **Results**: [待读后填写] 只记录数据规模、数据消融、质量/覆盖评估和 downstream 对照，不展开 optimizer / RLHF 等算法细节。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：[待读后填写]
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：[待读后填写]

## 疑问 / 下一步
- 没看懂的 / 想深挖的 1 个问题：[待读后填写]

## 原文金句 (1-2句)
> [阅读后补原文，勿凭记忆引用]

## 今晚产出
- 按模板补齐 Data Pipeline / Key Tricks / Results / 可迁移
- 保留并完善「和之前工作的关系」小节
- 全程只写数据：curation / selection / quality / diversity / complexity / synthetic / execution-filter；算法只一句带过

> 自动化：reading-log 已追加 / commit 由本次自动化推送 / ai data sheet 由本次自动化同步
