# Paper 模板

## 元信息
- Title: Difficulty-Based Preference Data Selection by DPO Implicit Reward Gap
- Authors / Org: Xuan Qi, Rongwu Xu, Zhijing Jin - UW / Tsinghua / MPI
- Link / arXiv: https://arxiv.org/abs/2508.04149
- Date read: 2026-08-13
- Tags: [rl-data, dpo, preference-data, data-selection, alignment, coding-data]

## 一句话总结
做 DPO 对齐别全量上，算 DPO 隐式 reward 的 chosen-rejected gap，gap 小的难例留 10%，对齐效果打赢全量，证明难的偏好对才值钱。

## 核心
1.  **Motivation**: RLHF/DPO 都靠大偏好集，贵，且高质量偏好怎么选没人说清
2.  **Data Pipeline**: 拿 base 模型算 DPO implicit reward gap -> 按 gap 排序 -> gap 小的当难例留 10% -> 训 DPO/Reward Model。理论是 gap 小=模型分不清=学习信号大。
3.  **Key Tricks**: 
    - 隐式 reward gap 不用外部 RM，直接用 DPO 公式算，10% 就超全量【6968311559756508151†L25-L29】
    - 在 RewardBench 上 75% 维度打赢其他基线【6968311559756508151†L106-L110】
    - 比 External Margin / IFD-Z 的 Low-Gap 更稳，因为直击 DPO 学习潜力【6968311559756508151†L100-L105】
4.  **Results**: 10% 难例在 AlpacaEval 2.0 / RewardBench 上超 5 个强基线，接近或超全量，常在 Chat-Hard, Safety, Reasoning 上赢【6968311559756508151†L95-L98】

## 可迁移
- 对你现在 coding data 工作的 1-2 个直接可试的点：
  - DPO 做 coding 偏好时（正确vs错解），用 implicit gap 选 reward 接近的难对，踢掉一眼就能分出来的简单对
  - 把 execution 信号和 gap 融合：gap 小且一个过测一个不过的 pair 优先级最高
- Infra 视角：RLHF 洗偏好对成本最高，用 10% 难对降 90% 采样和标注，7B 级最划算

## 和之前工作的关系
是 LESS 的偏好版。LESS 挑 SFT，这个挑 DPO，都是少即是多。和 Day 5 DataInf 互补，DataInf 踢脏数据，这个留难数据。和 Day 8-10 的后训练闭环吻合：Qwen2.5/Llama3.1 的两轮 RL 前先做这个 reward-gap 筛。

## 疑问 / 下一步
- coding 上 implicit gap 会不会被 execution 颠覆？gap 小但一个能跑一个不能跑，怎么融合两个信号？

## 原文金句
> By selecting preference data examples with smaller DPO implicit reward gaps, which are indicative of more challenging cases, we improve data efficiency【6968311559756508151†L25-L28】

> achieving superior performance with only 10% of the original data【6968311559756508151†L27-L30】
