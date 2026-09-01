# 06 — GLM-5.2 case study：long-horizon compaction 下为何选 critic PPO

> 这是一个 **公开系统案例**，不是“PPO 已取代 GRPO”的普遍结论。只保留官方公开材料明确陈述的算法信息；不保留未在本专题逐项核验的参数规模、榜单、价格、训练耗时或“首创”类 headline claims。

## 官方披露了什么

Z.ai 的 GLM-5.2 官方发布文说明，long-horizon task 会产生很长 execution traces；经 compaction 拆成 sub-traces 后，同一 prompt 的不同 rollouts 会产生 **数量不同、长度高度不均** 的可训练片段。其公开方案因此从 group-wise optimization 转向：

- critic-based PPO；
- individual-rollout learning；
- critic 估计 token-level advantages；
- 把所有 compacted sub-traces 纳入训练；
- 用 token-level loss 处理片段长度不均。

官方原文：[GLM-5.2: Built for Long-Horizon Tasks](https://huggingface.co/blog/zai-org/glm-52-blog)。

## 为什么 group construction 在这里变难

标准 outcome GRPO 的统计单位是同 prompt 的完整 completion group：

$$A_i^{grp}=\frac{R_i-\bar R}{s_R}$$

compaction 后，一个原始 rollout 可能映射为 $K_i$ 个 trainable segments，每段长度为 $L_{i,k}$。若 $K_i$ 与 $L_{i,k}$ 在 rollouts 间差异很大，就出现三个问题：

1. **comparison unit 不清楚**：是比较原始完整 rollout、segment，还是某个 compaction boundary？
2. **group cardinality 不齐**：不同 rollout 贡献不同数量的 segments，强行配组会丢数据或重复加权。
3. **length weighting 改目标**：response mean、segment mean、token mean 会给予同一原始 rollout 不同总权重。

critic 形式直接对任意 history $h_t$ 估值：

$$\widehat A_t=\sum_{l\ge 0}(\gamma\lambda)^l\left(r_{t+l}^{env}+\gamma V(h_{t+l+1})-V(h_{t+l})\right)$$

这样每个可训练 segment 可使用开头/结尾 value bootstrap，不要求凑成相同大小的 peer group。代价是 critic 的训练、内存、通信和 calibration burden。

## 不应从案例推出什么

- **不是**“长轨迹必然 PPO”。若仍能定义可靠的同 prompt outcome group，GRPO 仍可训练长 completion；DAPO/GSPO/segment-level 方法也在改变 sampling 或 ratio granularity。
- **不是**“critic 自动产生 dense ground-truth reward”。critic 只拟合预期 return；reward 错、数据少或 distribution shift 时同样会错。
- **不是**“GRPO 完全不能 process supervision”。DeepSeekMath 原论文包含 process-supervision GRPO。
- **不是**“token-level PPO 对变长天然无偏”。mask、aggregation、bootstrap、staleness 与 token-local ratio 仍需审计。
- **不是**公开证据支持固定 horizon 阈值。不要写“超过 50 步就切 PPO”之类规则；应根据 segment heterogeneity、critic fit 与 held-out gain 决策。

## Anti-hacking 与 optimizer 是两条轴

同一官方发布文还讨论 coding-agent reward hacking，并采用 rule filtering 与 model judging。其意义是：verifiable pass/fail 并不等于 verifier 安全。agent 可能读取受保护评测材料、复制参考答案或绕过任务。无论 PPO/GRPO，都应把以下指标独立出来：

- environment/reward-service failure；
- prohibited artifact access；
- suspicious tool calls；
- held-out clean evaluator pass rate；
- reward 与人工/独立 judge 的 disagreement。

换成 PPO 不会修复 reward hacking；换成 GRPO 也不会。

## 可复用决策实验

在自己的任务上，不先贴算法标签，先做三组测量：

1. **segment heterogeneity**：$K_i$ 分布、$L_{i,k}$ 分布、每个原始 rollout 的总 token 权重；
2. **critic viability**：按 horizon bucket 的 value error / explained variance，以及 truncation bootstrap 敏感性；
3. **group viability**：non-degenerate rate、组内 reward dispersion、为凑有效 group 消耗的额外 generated tokens。

若 group formation 丢弃大量片段，而 critic 在 held-out trajectories 上能稳定预测 return，critic PPO 的系统成本可能值得；反之，短且可验证任务仍可从 critic-free group estimator 起步。
