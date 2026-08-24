# Paper 模板 - Day 24

> 自动生成骨架 2026-08-24，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: D4: Improving LLM Pretraining via Document De-Duplication and Diversification
- Authors / Org: Tirumala et al. / Meta AI
- Link / arXiv: https://arxiv.org/abs/2308.12284
- Date read: 2026-08-24
- Tags: [pretraining, data-curation, deduplication, diversity, quality, semantic-dedup, coding-data]
- Folder: day-24-2023-semdedup-d4
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-24-2023-semdedup-d4

## 一句话总结
D4 把语义近重复去除（SemDeDup）与表示空间中的原型式多样化剪枝串成预训练数据管线，在压缩冗余语料的同时保留更广覆盖，补上 Day19 Vendi Score“会量多样性但还不会规模化选数”的工程短板。

## 和之前工作的关系

> 知识图谱位置：预训练去重源头；Day07 Llama3 / Day14 StarCoder2 的大规模过滤与去重实践，向上追溯到可复用的语义去重 + 多样化选择方法；并把 Day19 Vendi Score 的多样性概念落成实际数据剪枝。

- **接了哪条线：**接 Day19 Vendi Score → Day20 DEITA 的 diversity / redundancy 线，也回接 Day07 Llama3、Day14 StarCoder2、Day16 Qwen2.5-Coder 的预训练与 code data 清洗线。
- **补了哪个短板：**此前的管线多写 MinHash、规则过滤或一句“去重”，缺少对语义近重复、簇内保留策略和覆盖损失的系统拆解；D4 把“删重复”和“保多样”明确分成两步。
- **替代 / 分叉 / 改进：**它不是替代 MinHash / exact dedup，而是在词面去重后增加 semantic dedup，再用 diversity-aware pruning 避免只保留高密度主流样本；这是质量过滤之外的正交改进。
- **对之前 Day X 的直接对比：**vs Day19 Vendi Score，Vendi 给出数据集多样性的度量标尺，D4 给出可执行的删样本流程；vs Day20 DEITA，DEITA 在 SFT 池中按质量×复杂度×多样性选 6k，D4 面向预训练规模，重点是近重复消除与覆盖保持。

## 为什么今天读它

Day21→23 已完成“合成扩量 → 复杂度演化 → 少量高质策展”，今天转到数据池进入训练前的第一道规模化门禁：先把重复信息删掉，又不能把长尾覆盖一起删掉。对 coding data，可用文件/函数表示做语义近重复聚类，并在每簇中保留质量更高、许可与测试更完整的代表；对 SFT / RL data，也可把同模板改写和等价题目视为语义簇，控制有效样本数与任务覆盖。本文只研究数据去重、选择与覆盖，不展开训练算法。

## 今天的 3 问
1. D4 如何定义并串联 lexical dedup、SemDeDup 和 prototype-based diversification？每一步删除哪类冗余，簇内代表样本按什么数据标准保留？
2. 对比 Day19 Vendi Score 与 Day20 DEITA：D4 的“多样化剪枝”是在优化一个可量化的 diversity 指标，还是依赖表示空间启发式？三者如何组合成“质量门禁 → 去重 → 覆盖审计”的流水线？
3. 迁移到 coding / SFT / RL data 时，embedding 选型、相似度阈值和簇粒度如何避免误删“表面相似但边界条件不同”的样本？该用哪些覆盖与下游评测验证去重没有伤到长尾？

## 核心
1. **Motivation**: [待读后填写] 大规模预训练语料中的重复与高密度主题如何浪费计算、放大记忆，并挤压长尾覆盖？
2. **Data Pipeline**: [待读后填写] 文档表示 → 聚类/近邻 → 语义近重复删除 → 原型式多样化剪枝 → 数据规模与覆盖审计 → 进入预训练。
3. **Key Tricks**: [待读后填写] 记录 embedding、聚类粒度、相似度阈值、簇内保留规则、删除比例，以及不同语料域是否使用不同门槛；不展开训练算法。
4. **Results**: [待读后填写] 只记录数据压缩率、重复率/多样性变化、token/compute 节省和 downstream 对照，不展开 optimizer 等算法细节。

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
