# Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

## 元信息
- Title: Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
- Authors / Org: Open-Reasoner-Zero Team
- Link / arXiv: https://arxiv.org/abs/2503.24290
- Date read: 2026-08-28
- Tags: [rl-data, reasoning-data, verifiable-data, difficulty, curation, quality]

## 一句话总结
从约 40k 可验证数学题构造可持续采样的数据池，并用可验证答案、难度分层和质量门禁支撑推理数据飞轮，把“少量高价值题”从 Day 11 LIMR 的静态选择推进到可验证 RL 数据闭环。

## 和之前工作的关系
- **知识图谱位置**：偏好 / RL 数据线：Day 26 UltraFeedback（造偏好池）→ Day 13 DPO-Gap（选难偏好对）→ Day 11 LIMR（选高价值 RL 题）→ **Day 28 Open-Reasoner-Zero / DeepScaleR（可验证题池与难度分层）** → Day 15 DeepSeek-R1（冷启动 + 可验证推理数据）。
- **接了哪条线**：承接 Day 11 LIMR 的“RL 数据也要少而精”，也接 Day 16 Qwen2.5-Coder 的 execution-filter 思路：数学最终答案校验与代码执行一样，都是把训练样本变成可自动验证的数据。
- **补了哪个短板**：LIMR 更像一次性挑题，Day 15 R1 又没有展开开放数据配方；本篇补上题源清洗、答案可验证性、难度分层与持续采样这几个可复现的数据层环节。
- **替代 / 分叉 / 改进**：不是替代 UltraFeedback 的主观 AI feedback，而是分叉出 objective-verifiable data；相对 Day 13 的偏好对筛选，这里直接围绕“题—最终答案—验证器”组织样本，反馈噪声更低。
- **直接对比 Day 11**：Day 11 问“哪些题最值得留下”，Day 28 进一步问“留下后如何按当前能力持续采样，并用验证结果回流更新题池”。

## 为什么今天读它
- **coding data**：把数学答案验证器映射到 unit test / compiler / sandbox；可直接复用“可验证 + 难度分层 + 失败回流”的数据骨架。
- **SFT data**：区分带完整解答的冷启动样本与仅有可验证答案的题目池，避免把所有推理题都做成昂贵的长 CoT 标注。
- **RL data**：重点不在优化算法，而在题源、去重、答案标准化、可验证覆盖率、难度桶与动态采样；这些决定 RL 是否有稳定、低噪声的数据供给。

## 今天的 3 问
1. 约 40k 题从哪些来源进入数据池？去重、污染检查、答案标准化与可验证性门禁各自淘汰了多少样本？
2. 难度分层应依据静态来源标签，还是依据当前模型 pass rate 动态更新；怎样避免只采“刚好会一点”的题导致覆盖坍缩？
3. 对比 Day 11 LIMR：静态高价值选题与 Day 28 的可验证难度采样，在 coding task 上分别该映射成哪些信号（梯度 / 轨迹价值 vs unit-test pass rate / execution failure）？

## 核心
1. **Motivation**: 为什么通用数学语料不能直接成为低噪声、可持续的推理训练数据？
2. **Data Pipeline**: 题源 → 去重 / 防污染 → 答案标准化 → verifier 检查 → 难度分桶 → 采样 → 失败结果回流。
3. **Key Tricks**: 记录题源与 license；统一答案格式；用当前模型通过率重估难度并保留跨桶覆盖。
4. **Results**: 重点核对数据规模、有效验证率、难度分布，以及不同数据配方对 downstream 的增益。

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：把 unit-test pass rate 作为动态难度标签；按错误类型（compile / runtime / wrong answer / timeout）回流成数据质量与课程采样信号。
- Infra 视角：可扩展性 / 成本 / 评测自动化的启发：把 verifier 结果一次落盘，同时服务过滤、难度估计、训练抽样和 eval，避免重复执行；按题目哈希与测试版本做 lineage。

## 疑问 / 下一步
- 数据池的难度分布会随模型变强而漂移，怎样定义稳定的重采样与退役规则，同时保留长尾能力覆盖？

## 原文金句 (1-2句)
> 待读完原文后补充；只摘与数据构造、可验证性或难度采样相关的句子。
