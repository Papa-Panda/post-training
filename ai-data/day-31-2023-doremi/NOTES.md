# DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

## 元信息
- Title: DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining
- Authors / Org: Sang Michael Xie et al. / Google DeepMind & Stanford University
- Link / arXiv: https://arxiv.org/abs/2305.10429
- Date read: 2026-08-31
- Tags: [pretraining, data-mixture, domain-reweighting, curation, quality, proxy-model]

## 一句话总结
DoReMi 用 280M reference/proxy model 的跨域 excess loss 自动学出预训练数据的 domain mixture，再按该配比重采样给 8B 模型训练：在 The Pile 上平均 few-shot 准确率高 6.5 个百分点，并用 2.6× 更少训练步数达到 baseline 水平。

## 和之前工作的关系
- **知识图谱位置**：这是预训练数据线新增的“配比层”。Day 24 D4 决定域内哪些文档该去重/保留，Day 25 FineWeb 决定网页数据如何过滤，Day 31 DoReMi 决定清洗后的 Wikipedia、GitHub、books、web 等数据域各占多少。
- **接了哪条线**：接 Day 07 Llama 3、Day 08 DeepSeek-V3、Day 09 Qwen2.5 中“多域预训练 mixture”这条线，把大模型报告里的经验配比改造成可由小模型学习的数据配方。
- **补了哪个短板**：此前路线覆盖了样本级质量、去重、多样性和 selection，却缺少 domain-level allocation；即使每个域都很干净，错误配比仍会浪费 token budget 或压低长尾域表现。
- **替代 / 分叉 / 改进**：它不替代 Day 25 的过滤，也不替代 Day 04 LESS 的目标任务样本选择；它位于两者之间，是“先把数据分域，再决定各域采样概率”的宏观重加权层。
- **直接对比 Day 04 / Day 25**：Day 04 LESS 用目标任务梯度在样本级选少量 SFT 数据，Day 25 FineWeb 用规则、去重和消融造干净网页池；DoReMi 不依赖下游任务标签，用小代理模型的跨域 excess loss 给整个预训练池配比。

## 为什么今天读它
30 天主干已经闭环了“数据从哪里来、怎么合成、过滤、去重、选择、验证和防漏”，但还少一个关键旋钮：不同来源各喂多少。对 coding data，可以把 GitHub、StackExchange、技术文档、合成题、执行轨迹等看成不同 domain，用小模型先估计 learnable headroom，再决定 SFT / continued-pretrain 数据预算；这里只研究 domain 定义、配比估计与重采样，不展开优化器或训练算法。

## 今天的 3 问
1. DoReMi 为什么用 proxy 相对 reference 的 non-negative excess loss，而不是直接用高 loss 给 domain 加权；它如何避免把纯噪声或天然高熵域误当成高价值数据？
2. 对 coding data，domain 应按来源（GitHub/文档/问答）、语言、难度还是可执行性切分；怎样保留 provenance，才能让学到的 mixture 可审计、可重采样？
3. 对比 Day 04 LESS 与 Day 25 FineWeb：什么时候应做样本级 target-aware selection，什么时候应做规则过滤，什么时候应做 DoReMi 式 domain-level allocation；三层怎样串成一条数据管线？

## 核心
1. **Motivation**: 预训练各数据域的采样比例显著影响效果，但常靠 token 量或人工经验设置；需要一个不依赖下游任务、可由小模型估计并迁移到大模型的数据配方。
2. **Data Pipeline**: 将语料划分为多个 domain → 按 reference mixture 训练小 reference model → 用同规模 proxy model 估计各域相对 reference 的 excess loss → 对 domain 权重做平滑并沿训练轨迹求平均 → 按新权重重采样语料 → 训练 full-size model。
3. **Key Tricks**: 用 excess loss 近似“可学习余量”而非原始 loss；把负 excess loss 截为 0，降低噪声/极易域误导；保留小的 uniform smoothing 权重，避免 domain 被彻底饿死；用时间平均权重提高稳定性。
4. **Results**: 论文用 280M reference/proxy 为 8B 主模型定配比；在 The Pile 上平均 few-shot 准确率提升 6.5 个百分点，并以 2.6× 更少训练步数达到 baseline 准确率，且各域 perplexity 都改善。待读后核对各域权重、消融与跨规模稳定性。

## 可迁移
- 对 coding data 工作：先定义稳定、互斥且有 provenance 的 source × language × verifiability domains；对每个域记录 token 量、excess loss、最终采样权重和下游增益，形成可版本化 mixture recipe。
- Infra 视角：用小 proxy 周期性重估 mixture，materialize 可复现的 weighted sampler；监控 domain starvation、重复采样、有效 token 利用率和跨规模 transfer drift。

## 疑问 / 下一步
- 当 coding domains 重叠且层级化（同一条样本同时属于 Python、GitHub、可执行、repo-level）时，怎样设计 domain taxonomy，才能避免权重不可辨识并保持 mixture 的工程可解释性？

## 原文金句 (1-2句)
> 待读后补充。
