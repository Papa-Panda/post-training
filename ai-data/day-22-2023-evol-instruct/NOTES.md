# Paper 模板 - Day 22

> 自动生成骨架 2026-08-22，基于 PAPER_TEMPLATE.md，纯 Data 视角；算法只一句带过，不在本轨道展开。

## 元信息
- Title: WizardLM: Empowering Large Language Models to Follow Complex Instructions
- Authors / Org: Can Xu et al. / Microsoft Research Asia, Peking University
- Link / arXiv: https://arxiv.org/abs/2304.12244
- Date read: 2026-08-22
- Tags: [synthetic-data, sft, coding-data, instruction-evolution, complexity, curation, quality]
- Folder: day-22-2023-evol-instruct
- GitHub: https://github.com/Papa-Panda/post-training/tree/master/ai-data/day-22-2023-evol-instruct

## 一句话总结
用 Evol-Instruct 的 In-depth / In-breadth 演化算子把简单指令递归改写成约 70k 条更复杂、多样的 SFT 数据，补上 Self-Instruct 会自举但容易停留在简单任务分布的短板。

## 和之前工作的关系

> 知识图谱位置：合成指令主线第二站，Day21 Self-Instruct（从少量种子扩规模）→ Day22 Evol-Instruct（显式提升复杂度）→ Day27 OSS-Instruct（迁移到 code 数据）。

- **接了哪条线：**直接接 Day21 的 synthetic/bootstrap 线。Self-Instruct 解决“没有指令池时怎么从种子造池”，Evol-Instruct 解决“造出的池为什么仍太简单”。
- **补了哪个短板：**把复杂度从事后评分变成生成阶段的可控维度；同时为 Day20 DEITA 的 Evol-Complexity scorer 提供来源，形成“先演化复杂度，再按复杂度×质量×多样性筛选”的流水线。
- **替代 / 分叉 / 改进：**不是替代 Self-Instruct，而是其复杂度升级；相对 Day06 Phi-1 的“合成教科书/代码内容”，本篇改造的是 instruction/task 分布；Day27 OSS-Instruct 再把这套演化迁移到 coding 场景。
- **对之前 Day X 的直接对比：**vs Day21，Self-Instruct 主要靠新任务自举与 ROUGE-L 去重扩大覆盖，Evol-Instruct 通过约束增加、具体化、推理步骤增加与 breadth mutation 主动改变难度；今天重点检查这种“更复杂”是否真的带来新能力，而不只是更长、更啰嗦。

## 为什么今天读它

Self-Instruct 给了 coding SFT / RL 冷启动的“造池”起点，但真实 coding data 需要可控难度梯度：从单函数题演化到边界条件、复杂约束、多文件上下文和可执行验证。Evol-Instruct 正好补“复杂度如何生成”的数据方法，并可在进入 SFT 或可验证 RL 数据池前，串上 Day16 parser/exec 过滤与 Day20 DEITA 三因子筛选。

## 今天的 3 问
1. In-depth 与 In-breadth 的演化算子里，哪些真的增加任务约束或推理结构，哪些只是拉长表述？应如何用数据指标审计“复杂度上升但质量不降”？
2. 对比 Day21 Self-Instruct 的 ROUGE-L 去重与分类/非分类分池，递归演化后还需要哪些门禁来拦截语义漂移、不可解任务、重复约束和答案幻觉？
3. 把 Evol-Instruct 迁移到 coding data 时，如何把“增加约束/具体化/增加推理步骤”改写成边界条件、多文件依赖、性能约束与测试用例，并在进 SFT / RL 数据池前接 Day16 的 parser + execution filter？

## 核心
1. **Motivation**: [待读后填写] 为什么普通 instruction-tuning 数据偏简单？复杂指令覆盖有什么缺口？
2. **Data Pipeline**: [待读后填写] 种子从哪来 → In-depth / In-breadth 怎么演化 → 怎么过滤失败样本 → 如何形成约 70k 数据集 → 如何评估。
3. **Key Tricks**: [待读后填写] 记录演化算子、停止条件、失败过滤、去重和复杂度判断；只记数据构造，不展开模型训练算法。
4. **Results**: [待读后填写] 只记录数据规模、质量/复杂度评估和 downstream 对照，不展开 optimizer / RLHF 等算法细节。

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
- 全程只写数据：synthetic / complexity / curation / quality / diversity / execution-filter；算法只一句带过

> 自动化：reading-log 已追加 / commit 由本次自动化推送 / ai data sheet 由本次自动化同步
