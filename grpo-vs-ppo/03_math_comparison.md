# 03 — 数学对照：到底换掉了什么

PPO 与 GRPO 不是“一个有 clip、一个没 clip”。常见实现都使用 clipped surrogate；核心差异是 advantage estimator、采样组织与 loss aggregation。

## 1. Unified template

定义 token surrogate：

$$s(\rho,A)=\min(\rho A,c(\rho,1-\epsilon_l,1+\epsilon_h)A)$$

其中 $c(x,l,u)=\min(\max(x,l),u)$。两者都可写为 masked weighted sum：

$$J(\theta)=\mathbb E\left[\sum_{i,t}w_{i,t}m_{i,t}s(\rho_{i,t}(\theta),\widehat A_{i,t})\right]-\beta J_{KL}$$

差别如下：

| 轴 | critic PPO | outcome GRPO |
|---|---|---|
| rollout | 每 prompt 可一条或多条 | 每 prompt 必须成组 $G>1$ |
| advantage | $\widehat A_{i,t}^{GAE}$，依赖 $V_\phi$ | $A_i^{grp}$，同 response 广播 |
| temporal credit | token/state specific；可 bootstrap | terminal outcome 下不区分 token |
| baseline | learned state value | 同 prompt sampled rewards |
| extra trainable state | critic parameters + optimizer state | 无 critic；但需要 group rollouts |
| aggregation | 通常 masked token mean | 原始 response mean；变体常 token mean |
| failure signal | value loss / explained variance | degenerate groups / reward dispersion |

## 2. Baseline 不是免费午餐

理想 state baseline $b(h)$ 若不依赖 sampled action，有：

$$\mathbb E_{a\sim\pi}[b(h)\nabla\log\pi(a\mid h)]=b(h)\nabla\sum_a\pi(a\mid h)=0$$

critic 的问题主要是 estimation/optimization burden 与 bootstrap trade-off，而不是“只要 $V$ 不准就必然把 policy gradient 变有偏”。

GRPO 的 group mean 来自同一批 actions。未标准化时 self-inclusion 把期望梯度缩成 $(G-1)/G$；LOO 可修正该 scale。按组 standard deviation 归一化进一步改变 prompt 权重：同样 reward gap 在低 dispersion group 中权重更大。它可稳定 scale，也可造成 question-level difficulty bias。

## 3. Ratio granularity

对 terminal sequence reward，三个常被混写的量：

$$\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid h_{i,t})}{\pi_b(y_{i,t}\mid h_{i,t})}$$

$$\rho_i^{seq}=\prod_t\rho_{i,t}$$

$$\rho_i^{geom}=\exp\left(\frac{1}{T_i}\sum_t\log\rho_{i,t}\right)$$

- token ratio 是 PPO/GRPO 常见局部 surrogate 的单位。
- exact sequence ratio 是 trajectory distribution correction，但乘积在长序列上高方差。
- GSPO 的 length-normalized sequence ratio $\rho_i^{geom}$ 是 geometric mean，并非 exact sequence importance ratio；它选择 sequence-level clipping 来换稳定性。

因此“sequence-level”不自动等于“理论无偏”，而“token-level”也不自动等于错误；要看 reward/advantage 的粒度与目标是什么。

## 4. 方差与 sample efficiency：拒绝伪公式

原笔记给出 `Var(GRPO) proportional to 1/G` 和某个不完整的 GAE 大 O 式；这两句都不足以支持算法比较，现删除。更准确的拆分是：

1. 增大 $G$ 改善同一 prompt reward statistics，但在固定 rollout-token budget 下减少不同 prompts 数量。
2. group normalization 控制 advantage 尺度，不等于自动把 policy-gradient variance 降为 $1/G$；token score covariance、reward ties、length 与 clipping 都参与。
3. critic 可跨 prompts/states 泛化，因此在 critic 可靠时，一个 rollout 也能获得 baseline；但 critic 本身需要训练数据、forward/backward 与 optimizer state。
4. PPO 和 GRPO 都是 on-policy-ish 方法。对同一 rollout 做更多 optimization epochs 可提高数据复用，却加剧 $\pi_\theta$ 与 $\pi_b$ 的 drift；clipping 不能保证样本始终有效。
5. dynamic filtering 提高 **训练 batch 的有效 group 比例**，但被过滤的 rollout tokens 已经生成，不能说 generation sample efficiency 免费提升。

固定生成预算 $B_{tok}$ 下，粗略可容纳的 prompt 数为：

$$N_q\approx\frac{B_{tok}}{G\,\mathbb E[T]}$$

这是 GRPO 的关键机会成本：更可靠的 within-prompt comparison，换更少的 prompt coverage。

## 5. 目标与适用边界

| 条件 | 更自然的起点 | 原因 / 警告 |
|---|---|---|
| 短、可验证、terminal reward；同 prompt 可便宜采多次 | outcome GRPO / RLOO family | 省 critic；但需监控 group degeneracy 与 reward hacking |
| 变长多步轨迹、需要从截断片段学习 | critic PPO | $V(h)$ 可给每个 history/bootstrap；critic 质量决定收益 |
| 有可靠 process reward | 两者都可 | PPO 可 GAE；GRPO 原论文也有 process supervision，不是 PPO 专属 |
| 固定 token budget、prompt 多样性重要 | PPO/RLOO 小 $G$ 或混合 | 大 $G$ 会吞掉 prompt coverage |
| critic 占主要训练内存 | GRPO family | 省的是 critic path，不保证 rollout KV/cache 或 actor optimizer 不再主导 |
| very long sequence 且 token-ratio clipping 不稳 | sequence/segment-level variants 值得评估 | GSPO 改优化粒度；不能仅凭名称假定更优 |
| reward 不可验证或 judge 易被利用 | neither by default | 首先修 reward/eval；换 optimizer 不会修好错误目标 |

## 6. Current method boundary map

- **RLOO**：leave-one-out raw reward baseline；避免 self-inclusion scale bias，不做 per-group std normalization。
- **Dr.GRPO**：针对原始 GRPO 的 response-length 与 per-question-std weighting bias。
- **DAPO**：asymmetric clip、dynamic sampling、token-level loss aggregation、overlong reward shaping；它是一组 recipe，不是“只换 advantage”。
- **GSPO**：保留 group advantage，但改用 length-normalized sequence likelihood ratio 与 sequence-level clipping。

这些方法改变不同轴：baseline、normalization、aggregation、ratio granularity、sampling。比较实验应一次只改一轴，否则“GRPO vs PPO”标签无法解释结果。

来源：[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[RLOO for RLHF](https://arxiv.org/abs/2402.14740)、[Dr.GRPO](https://arxiv.org/abs/2503.20783)、[DAPO](https://arxiv.org/abs/2503.14476)、[GSPO](https://arxiv.org/abs/2507.18071)。
