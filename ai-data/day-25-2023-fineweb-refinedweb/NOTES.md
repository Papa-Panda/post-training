# Paper 模板 - Day 25

> 自动生成骨架 2026-08-25，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: FineWeb: Decanting the Web for the Finest Text Data at Scale（对照 RefinedWeb）
- Authors / Org: Penedo et al. / Hugging Face（RefinedWeb: Penedo et al. / Technology Innovation Institute）
- Link / arXiv: https://arxiv.org/abs/2406.17557
- Supporting paper: https://arxiv.org/abs/2306.01116
- Date read: 2026-08-25
- Tags: [pretraining, web-data, data-curation, filtering, deduplication, quality, scale]
- Folder: day-25-2023-fineweb-refinedweb
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-25-2023-fineweb-refinedweb

## 一句话总结
FineWeb 把 RefinedWeb 开创的“只靠高质量 Common Crawl 也能训练强模型”路线扩展为 15T-token、可复现且经消融验证的网页语料管线，用 URL/文本过滤、语言识别、重复控制和质量评测把“过滤规则清单”升级为可审计的数据配方。

## 和之前工作的关系

> 知识图谱位置：大规模预训练过滤主干；接 Day24 D4 的“去重且保覆盖”，并把 Day07 Llama3、Day14 StarCoder2、Day16 Qwen2.5-Coder 里分散的工业清洗步骤组织成可复现、可消融的数据瀑布。

- **接了哪条线：**接 Day24 D4 / Day19 Vendi 的 dedup-diversity 线，也回接 Day07 Llama3 的网页预训练瀑布与 Day14 StarCoder2 的规则化 code curation。
- **补了哪个短板：**Day24 重点解释“怎样删语义重复并保多样性”，但没有给出从原始 Common Crawl 到训练语料的完整、可审计 web pipeline；FineWeb 补上逐步过滤、去重位置、质量代理和消融评测。
- **替代 / 分叉 / 改进：**它不是替代 D4，而是上层系统化配方；相对 RefinedWeb，它把管线设计选择和过滤步骤做了更细的消融，强调每个数据决策都要由小模型训练评测验证，而不是把规则当常识。
- **对之前 Day X 的直接对比：**vs Day07 Llama3，FineWeb 更像公开可复现的 web-only 对照组，便于看清每级过滤的边际收益；vs Day24 D4，D4 深挖 semantic dedup / diversification，FineWeb 覆盖端到端网页语料生产，两者应组合为“规模过滤瀑布 + 语义去重/覆盖审计”。

## 为什么今天读它

Day21→24 已完成“合成扩量 → 复杂度演化 → 少量高质策展 → 语义去重与多样化剪枝”，今天补齐预训练数据工厂本身：原始网页怎样经过可复现的门禁变成万亿 token 级训练池。对 coding data，可直接借鉴规则分层、去重顺序、每一步保留率与小规模训练消融；对 SFT / RL data，则可迁移“每个过滤器都必须用下游质量曲线验收”的方法，而不是照搬网页规则。本文只讨论数据获取、过滤、去重、质量与评测，不展开训练算法。

## 今天的 3 问
1. FineWeb 从 Common Crawl 到 15T-token 语料依次经过哪些 data gates？URL、语言、文本质量、重复与个人信息等过滤分别改变了什么，应该记录哪些保留率和误杀率？
2. 对比 Day24 D4 与 Day07 Llama3：FineWeb 的去重粒度、去重时机和质量消融，如何与 D4 的语义去重/多样化剪枝组合，哪些步骤与 Llama3 的闭源五级过滤可以一一对应？
3. 迁移到 coding / SFT / RL data 时，怎样建立“规则或模型过滤器 → 小规模训练/评测 → 保留或回滚”的可审计闭环，避免高分过滤器把长尾、难例或新颖解法一起删掉？

## 核心
1. **Motivation**: [待读后填写] 为什么仅报告 token 规模不足以定义高质量 web corpus？RefinedWeb 留下了哪些可复现性与过滤归因问题？
2. **Data Pipeline**: [待读后填写] Common Crawl 获取 → URL/语言/文本质量过滤 → 重复控制 → 数据混合与质量评测 → 发布数据集；记录每级输入、输出、保留率和失败样本。
3. **Key Tricks**: [待读后填写] 只记 data 细节：过滤规则、阈值、去重范围/顺序、质量分类器、抽样与消融设计；不展开 optimizer 或训练算法。
4. **Results**: [待读后填写] 只记录数据规模、保留率、污染/重复变化、与 RefinedWeb/C4 等数据集的控制变量评测及计算成本。

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
