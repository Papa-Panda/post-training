# Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model

## 元信息
- Title: Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model
- Authors: Jingcheng Hu, Yinmin Zhang, Qi Han, Daxin Jiang, Xiangyu Zhang, Heung-Yeung Shum
- Link / arXiv: https://arxiv.org/abs/2503.24290
- Related: ProRL — https://arxiv.org/abs/2505.24864
- Date read: 2026-08-28
- Tags: [rl-data, reasoning-data, verifiable-data, pass-rate, hard-example-mining, prolonged-rl]

## 一句话总结
Open-Reasoner-Zero 用可验证题目、基于模型通过率的两端过滤，以及 v1 中“129k 全量 RL → 13k 困难尾部继续 RL”的 hard-example mining 支撑约 1,200 步长程强化学习；它是后来 ProRL 系统化“prolonged RL”路线的重要先行证据。

## 版本说明：v1 与 v2
- **v1（2025-03）**：初始数据约 129k；32B 模型先在全量数据上训练 1,100 步，再把 64 次作答中答对少于 4 次的题定义为困难题，得到约 13k，继续训练 100 步。
- **v2（2025-07，当前 arXiv 版本）**：报告的主要训练集为 ORZ 57k；保留“用 LLM 估计通过率并删除极端通过率样本”的描述，但不再完整保留 v1 的 129k → 13k 两阶段细节。
- 因此不能笼统写成“固定 easy / medium / hard 分桶 + 持续动态采样”。更准确的是：**模型通过率过滤；v1 另有一次训练后困难样本挖掘。**

## 和之前工作的关系
- **知识图谱位置**：Day 26 UltraFeedback（偏好池）→ Day 13 DPO-Gap（困难偏好对）→ Day 11 LIMR（高价值 RL 题）→ **Day 28 ORZ（可验证题池 + pass-rate filtering + hard mining）** → Day 15 DeepSeek-R1（大规模可验证推理 RL）。
- **连接 ProRL**：ORZ 已经显示，大规模、多样、可验证数据能让 RL 从通常的几百步延长到约 1,200 步，并持续提升 reward、回答长度与 benchmark 表现。随后 ProRL 把这条路线明确命名并系统化到 2,000+ 步，加入动态采样、KL 控制与 reference-policy reset。可以把关系概括为：**ORZ 提供先行证据，ProRL 提供完整的 prolonged-RL recipe。**
- **连接 LIMR**：LIMR 按学习轨迹与整体曲线的对齐度筛高价值题；ORZ 更直接地用可验证 rollout 的经验通过率刻画“对当前模型有多难”。
- **连接 coding data**：数学答案 verifier 对应 compiler / unit tests / sandbox；`pass rate` 可直接映射为测试通过率，构造当前模型相对难度。
- **区别 UltraFeedback**：UltraFeedback 依赖 AI 对开放回答做多维主观评价；ORZ 围绕“题目—最终答案—确定性 verifier”构造低噪声结果奖励数据。

## 数据与难度处理
1. **题源**：AIME（截至 2023）、MATH、Numina-Math、Tulu3 MATH、OpenR1-Math-220k、AoPS，以及程序化合成的逻辑、多步推理和反事实题。
2. **可验证门禁**：删除难以用规则可靠评分的题，例如证明题；v1 还明确排除了选择题。
3. **初始难度过滤**：用 LLM 多次解题，以经验通过率

   $$p_i=\frac{\text{correct rollouts}}{\text{total rollouts}}$$

   作为模型相对难度；删除通过率过高的题和通过率为 0 的题，避免太简单无学习信号，以及不可解、答案错误或完全超出能力边界的样本。
4. **v1 困难尾部挖掘**：全量训练 1,100 步后，每题采样 64 次；若答对少于 4 次，即 \(p_i<4/64=6.25\%\)，进入约 13k 的 hard set。
5. **应用方式**：从第 1,100 步的同一模型检查点继续，用 hard set 再做 100 步 PPO。两阶段都是 RL，不是 SFT → RL。

## 为什么它对长程 RL 重要
- 短 RL 容易在简单题上迅速饱和；扩大题目数量和能力覆盖，可推迟 plateau。
- 只用最难题也不行：全 0 奖励没有正轨迹，且可能混入坏题或错误答案。
- ORZ 的思路是先保留“可验证且可学习”的主体，再在模型变强后挖出它当时仍薄弱的困难尾部。
- 这解释了它为什么像 ProRL：两者都把**足够大、足够多样、始终能产生有效奖励的数据供给**视为延长 RL 的前提；只是 ProRL 后来增加了更完整的长期稳定机制。

## 关键实验结果
- v2 中 ORZ 57k 相比 MATH train 7.5k，训练 reward 与回答长度继续增长，而小数据较早 plateau。
- ORZ 在 Qwen2.5-{0.5B, 1.5B, 7B, 32B} 上均显示可扩展趋势。
- 相同 Qwen2.5-32B base 下，ORZ 在 AIME2024、MATH500 和 GPQA Diamond 上取得强结果，并报告只需 DeepSeek-R1-Zero pipeline 约十分之一的训练步数。

## 可迁移到 coding data
- 用 `passed_tests / total_tests` 代替数学题通过率，得到模型相对难度。
- 先删除始终全过的题，以及所有 rollout 都编译失败或被测试基础设施误杀的题；后者需先区分“真难”与“坏测试”。
- 全量 RL 后再重跑 rollout，挖出低但非纯噪声的通过率尾部做第二阶段训练。
- 将 compile error、runtime error、wrong answer、timeout 分开记录；同样的 0 分不应被视为同一种数据问题。

## 局限
- v1 的 `<4/64` 是一次 hard mining 阈值，不等于完整的持续动态课程。
- 难度是相对于生成这些 rollout 的模型与采样配置，不是题目的永久属性。
- verifier 只能保证最终答案或测试结果可检查，不能保证中间推理真实、简洁或无投机。
- v1 与 v2 数据规模和训练叙述不同，引用数字时必须注明版本。

## 原文关键句
> We also employ LLM-based filtering to evaluate problem difficulty, removing samples with extreme pass rates to maintain a balanced dataset.

> We initially train the 32B model for 1100 steps with data sampled from the complete 129k-sample dataset. Subsequently, we pinpoint particularly difficult prompts, defined as those where the model achieves fewer than 4 correct answers out of a total of 64 attempts, resulting in approximately 13k challenging prompts.
