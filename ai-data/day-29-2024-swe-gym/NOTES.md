# Training Software Engineering Agents and Verifiers with SWE-Gym

## 元信息
- Title: Training Software Engineering Agents and Verifiers with SWE-Gym
- Authors / Org: Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang / UC Berkeley, UIUC, CMU, Apple
- Link / arXiv: https://arxiv.org/abs/2412.21139
- Date read: 2026-08-29
- Tags: [coding-data, rl-data, executable-environment, trajectory-data, unit-tests, curation, quality]

## 一句话总结
SWE-Gym 从 11 个开源 Python 仓库整理 2,438 个真实 issue 任务，为每题封装代码库、自然语言需求、可执行环境与单元测试，把静态 coding 样本升级为可验证的仓库级交互数据，并公开训练轨迹供 SFT / RL 数据构造使用。

## 和之前工作的关系
- **知识图谱位置**：Day 27 OSS-Instruct（开源代码锚定合成 instruction）→ Day 16 Qwen2.5-Coder（parser / compiler / execution 过滤）→ **Day 29 SWE-Gym（真实 issue + repo context + executable environment + unit tests + agent trajectories）**；同时把 Day 28 ORZ 的“可验证题池”从数学最终答案扩展到仓库级代码修改。
- **接上的线**：它接 Day 27 的 coding synthetic 线，但把数据单位从“代码片段—合成问答”推进为“issue—完整仓库—交互轨迹—测试结果”；也接 Day 16 的 execution-filter 线，把执行从离线清洗门禁推进为每个任务自带的训练反馈。
- **补的短板**：此前数据大多是静态文本或单轮答案，缺少依赖可复现、跨文件修改、测试反馈与失败轨迹。SWE-Gym 补的是可运行的环境化数据，不是新的优化算法。
- **替代 / 分叉 / 改进**：它不替代 OSS-Instruct 的低成本规模化合成，而是分叉到更贵但真实性和可验证性更高的 repo-level 数据；与 Day 28 相比，verifier 从答案匹配升级为环境内单元测试，数据工程成本也显著上升。

## 为什么今天读它
它是 coding data 从静态 SFT 样本走向 agentic RL data 的关键桥：真实 issue 提供任务分布，仓库快照提供上下文，隔离执行环境与单元测试提供低噪声 verifier，agent rollout 再产出可筛选的成功 / 失败轨迹。今天只关注任务如何采集、封装、去重、验证与形成轨迹池；训练算法一句带过。

## 今天的 3 问
1. 2,438 个任务从 11 个 Python 仓库进入 SWE-Gym 前，issue / PR、仓库快照、依赖和测试分别经过了哪些过滤与可复现性门禁？
2. 论文最终用于训练的 491 条轨迹如何从 rollout 池中选择：只留成功轨迹会不会损失失败诊断信号，又会引入多大的 agent scaffold 偏置？
3. 对比 Day 27 OSS-Instruct 与 Day 16 Qwen2.5-Coder：SWE-Gym 的真实 repo + 单元测试到底补了哪些静态合成 / execution-filter 无法覆盖的数据维度，代价又是什么？

## 核心
1. **Motivation**: 静态 coding 数据缺少真实仓库上下文、依赖环境和可执行验证，无法直接支持仓库级软件工程 agent 的训练数据闭环。
2. **Data Pipeline**: GitHub 真实 issue / 代码库 → 固定仓库状态与依赖 → 封装可执行 runtime → 配套 unit tests → agent rollout → 按测试结果形成可验证轨迹数据。
3. **Key Tricks**: 待读论文后填写，重点记录任务采集规则、环境复现门禁、测试可靠性、轨迹筛选与污染控制。
4. **Results**: 待读论文后核对数据规模、有效环境比例、轨迹数量与下游 resolve-rate 增益；算法细节不展开。

## 可迁移
- 对 coding data 工作：把数据 schema 从 `{prompt, answer}` 扩展为 `{issue, repo_snapshot, environment, tests, trajectory, outcome}`，并显式记录编译、运行、测试与基础设施失败类型。
- Infra 视角：关注环境构建成功率、冷启动成本、可复现率、测试 flakiness、sandbox 隔离、轨迹存储与 verifier 吞吐，而不是展开具体 RL 优化器。

## 疑问 / 下一步
- 如何做 train / eval 的仓库、时间与语义级去重，避免来自同一代码演化链的近重复 issue 造成 benchmark contamination？

## 原文金句 (1-2句)
> We present SWE-Gym, the first environment for training real-world software engineering (SWE) agents.

> SWE-Gym contains 2,438 real-world Python task instances, each comprising a codebase with an executable runtime environment, unit tests, and a task specified in natural language.
