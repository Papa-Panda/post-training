# Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models

## 元信息
- Title: Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models
- Authors / Org: Martin Riddell, Ansong Ni, Arman Cohan / Yale University
- Link / arXiv: https://arxiv.org/abs/2403.04811
- Date read: 2026-08-30
- Tags: [coding-data, decontamination, benchmark-leakage, semantic-dedup, data-quality, evaluation]

## 一句话总结
这篇把 code benchmark 防污染从只查文本重合升级为 surface-level 与 semantic-level 双重匹配，在 The Stack / The Pile 中定位 HumanEval、MBPP 的近重复解法，并验证污染子集会显著抬高代码生成成绩。

## 和之前工作的关系
- **知识图谱位置**：这是 30 天 data 主线的质量门收口：Day 24 D4 / SemDeDup 处理训练语料内部的冗余，Day 27 OSS-Instruct 构造开源代码锚定的合成数据，Day 29 SWE-Gym 构造 repo-level 可执行任务，Day 30 则在这些数据进入训练前检查它们是否泄漏 benchmark。
- **接了哪条线**：接 Day 14 StarCoder2 与 Day 16 Qwen2.5-Coder 的代码清洗线，也承接 Day 29 留下的“如何做 train / eval 仓库、时间与语义级去重”问题。
- **补了哪个短板**：此前的 dedup / execution-filter 分别回答“是不是重复”和“能不能运行”，却不能保证训练样本没有换变量名、改格式或保留同一算法语义后泄漏测试集；本篇补上 surface + semantic 两层 contamination gate。
- **替代 / 分叉 / 改进**：它不替代 Day 24 的大规模语料去重，而是在评测边界上增加更严格的定向排除；通用语料去重优化训练分布，benchmark decontamination 保护评测可信度。
- **直接对比 Day 27 / Day 29**：Day 27、29 解决“如何造更真实、更可验证的 coding data”，本篇解决“这些数据是否已经包含评测题或语义近邻”；生成/采集之后必须再过防污染门，才能形成可信闭环。

## 为什么今天读它
前 29 天已经串起 coding data 的合成、筛选、执行验证和 repo-level 轨迹，但没有可靠的 train–eval 隔离，所有提升都可能被 benchmark 泄漏夸大。今天用这篇把数据闭环收口：对 SFT / RL data，不仅要查 prompt 和答案的字面重合，还要查重命名、改写与算法结构近似；这里只讨论污染检测、数据排除和评测切分，不展开模型训练算法。

## 今天的 3 问
1. surface-level Levenshtein 与 semantic-level Dolos / AST k-gram 分别能抓住哪类 code overlap；阈值如何用人工核验校准，避免把常见模板误判成污染？
2. 论文如何把 HumanEval / MBPP 与 The Stack / The Pile 做规模化匹配，并证明被判为污染的子集确实带来更高模型得分，而不只是题目更容易？
3. 对比 Day 24 D4 和 Day 29 SWE-Gym：若要为 repo-level SFT / RL 数据建立防污染门，应该按文件、函数、issue、commit 时间和执行语义分别做哪些 split 与去重？

## 核心
1. **Motivation**: 代码 benchmark 可能进入预训练或微调语料；仅靠精确字符串匹配会漏掉变量重命名、格式变化和语义近似实现，导致评测分数被污染抬高。
2. **Data Pipeline**: HumanEval / MBPP 测试题与参考解法 → 在 The Stack / The Pile 中检索候选 → surface-level 字符串相似度 + semantic-level code matching → 阈值判定与人工检查 → 比较污染 / 干净子集上的模型表现。
3. **Key Tricks**: 同时保留 lexical 与 semantic 信号；对每个 benchmark 样本取最相似训练候选；把污染标签与模型表现、题目难度和长度联合分析，而不是只报 overlap 比例。
4. **Results**: 论文报告常用 code benchmark 与开放训练语料存在大量重合；模型在训练语料中出现相似解法的子集上表现显著更好。待读后核对各语料、阈值与分组统计。

## 可迁移
- 对 coding data 工作：在数据入库时保存 repo、path、commit timestamp 与 provenance；评测前依次跑 exact/hash、token/字符相似、AST/k-gram 或 embedding 检索，并隔离同 repo / fork / commit lineage。
- Infra 视角：建立 versioned contamination registry，让每次新增 SFT / RL 轨迹都能重跑 train–eval overlap；把候选召回、语义复核、排除原因与 clean-set 指标做成可审计流水线。

## 疑问 / 下一步
- 对 SWE-Gym 这类 repo-level agent 任务，函数级语义相似仍可能漏掉 issue/patch 的共同演化链；怎样结合 commit graph 和时间切分，得到更保守且不过度删除的 clean split？

## 原文金句 (1-2句)
> 待读后补充。
