# UltraFeedback: Boosting Language Models with Scaled AI Feedback

## 元信息
- Title: UltraFeedback: Boosting Language Models with Scaled AI Feedback
- Authors / Org: OpenBMB / Tsinghua University
- Link / arXiv: https://arxiv.org/abs/2310.01377
- Date read: 2026-08-26
- Tags: [preference-data, ai-feedback, data-quality, diversity, curation, sft, rl-data]

## 一句话总结
UltraFeedback 从 64k 条多来源指令出发，为每条指令采样 4 个不同模型回答，并用 GPT-4 生成细粒度评价与分数，把普通 instruction pool 转成可追溯的偏好数据底座。

## 和之前工作的关系
- **知识图谱位置**：偏好数据构造线的源头层，位于 Day10 Llama 3.1/3.2 后训练数据切分之后、Day13 DPO-Reward-Gap 偏好对选择之前。
- **接了哪条线**：接 Day21 Self-Instruct / Day22 Evol-Instruct 的“先造 instruction pool”，再为每个 prompt 扩展多模型 response，并补上 AI feedback 标注。
- **补了哪个短板**：Day10 展示了后训练数据分阶段但配方闭源；UltraFeedback 给出公开、可复用的 prompt → 多回答 → critique/score → preference pair 数据管线。
- **替代 / 分叉 / 改进**：它不是替代 Day13，而是其上游数据源；UltraFeedback 负责造广覆盖偏好池，Day13 再从池中挑 reward gap 小的难偏好对。
- **直接对比 Day13**：Day13 研究“哪些 pair 值得留下”，本篇研究“pair 从哪里来、如何批量标注”；两者可串成生成 → 打分 → 难例选择。

## 为什么今天读它
前 25 天已覆盖预训练过滤、SFT 合成与少即是多，但偏好数据仍缺一个透明底座。UltraFeedback 把 coding/SFT prompt 扩成多候选回答并附细粒度质量信号，正好补齐从 instruction data 到 RL/preference data 的数据转换层；这里仅研究数据来源、覆盖、标注质量和 pair 构造，不展开 DPO/RLHF 算法。

## 今天的 3 问
1. 64k prompts 来自哪些数据源，怎样控制任务覆盖、难度与重复，避免偏好池只放大原始 instruction distribution？
2. 每个 prompt 的 4 个回答由哪些模型产生；多模型采样如何增加质量跨度与风格多样性，GPT-4 细粒度标注又会引入什么系统性偏差？
3. 对比 Day13 DPO-Reward-Gap：若把 UltraFeedback 的 4-way scores 转成 pairs，再优先保留小 gap 难对，怎样同时守住 prompt 多样性、回答质量和难度分布？

## 核心
1. **Motivation**: 大规模偏好数据昂贵且来源不透明；需要一条可扩展的 AI-feedback 数据生产管线。
2. **Data Pipeline**: 多来源 prompts → 多模型各生成回答 → GPT-4 按多个质量维度给 critique 与 score → 整理为偏好排序 / chosen-rejected pairs → 做数据质检与评估。
3. **Key Tricks**: 多模型响应增加质量跨度；细粒度维度评分保留比单一总分更丰富的监督；同时保留 critique、score 与 response，方便后续重筛和重组 pair。
4. **Results**: 待读后填写；重点记录数据规模、来源占比、标注一致性与下游数据消融，不展开训练算法。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：为同一 coding prompt 采样多个模型答案，保存 execution result、judge critique 与分项分数，再按正确性、难度和多样性组成 pair。
- Infra 视角：把生成、执行验证、AI judge、pair 构造拆成可追踪 stage；每条记录保留 provenance 与版本，支持重放、重标和偏差审计。

## 疑问 / 下一步
- GPT-4 judge 的绝对分与 pairwise preference 在不同 prompt domain 上是否校准一致？coding execution signal 能否作为独立硬门禁纠偏？

## 原文金句 (1-2句)
> 待读后补充。
