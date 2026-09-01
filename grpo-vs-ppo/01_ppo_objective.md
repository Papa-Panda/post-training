# 01 — PPO：从 policy gradient 到 clipped surrogate

记号见 [`00_notation.md`](00_notation.md)。这里讨论的是 PPO-Clip；PPO 论文还给出了 adaptive-KL 版本。

## 1. Policy-gradient surrogate

在 behavior policy $\pi_b$ 收集的数据上，单 token 的 importance ratio 是：

$$\rho_{i,t}(\theta)=\exp\left(\log\pi_\theta(y_{i,t}\mid h_{i,t})-\log\pi_b(y_{i,t}\mid h_{i,t})\right)$$

若已有 advantage estimate $\widehat A_{i,t}$，最直接的局部 surrogate 是：

$$J_{CPI}(\theta)=\mathbb E_{(h,a)\sim\pi_b}[\rho(\theta)\widehat A]$$

在采样点 $\theta=b$，$\rho=1$，这个 surrogate 的一阶梯度与 policy-gradient estimator 对齐。问题是有限 batch 上反复更新会让 ratio 变得极端，少数 token 主导梯度。

## 2. Clipping 是怎样来的

PPO 对每个样本取 unclipped 与 clipped 两项中的较小值。用 $c(x,l,u)=\min(\max(x,l),u)$ 表示 clip：

$$J_{clip}(\theta)=\mathbb E\left[\min\left(\rho\widehat A,\;c(\rho,1-\epsilon,1+\epsilon)\widehat A\right)\right]$$

分符号看更清楚：

- 若 $\widehat A>0$，当 $\rho>1+\epsilon$ 时不再奖励继续增大该 action 概率；但 $\rho<1-\epsilon$ 的坏方向仍保留梯度。
- 若 $\widehat A<0$，当 $\rho<1-\epsilon$ 时不再奖励继续减小该 action 概率；但 $\rho>1+\epsilon$ 的坏方向仍保留梯度。

因此 clipping 不是“把所有 ratio 截进区间”，也不是严格 KL trust region。它只截断会让 surrogate 看起来更好的那一侧；更新仍可能产生较大 KL，实践中应监控 KL、clip fraction 和极端 log-ratio。

## 3. GAE 与 critic

定义 token/environment reward $r_{i,t}^{env}$、终止标记 $d_{i,t}$ 和 TD residual：

$$\delta_{i,t}=r_{i,t}^{env}+\gamma(1-d_{i,t})V_\phi(h_{i,t+1})-V_\phi(h_{i,t})$$

截断轨迹上的 GAE 为：

$$\widehat A_{i,t}^{GAE}=\sum_{l=0}^{T_i-1-t}(\gamma\lambda)^l\delta_{i,t+l}$$

递推实现是：

$$\widehat A_{i,t}^{GAE}=\delta_{i,t}+\gamma\lambda(1-d_{i,t})\widehat A_{i,t+1}^{GAE}$$

关键边界：

- $\lambda=0$ 是 one-step TD advantage，通常方差较低、对 critic 偏差更敏感。
- $\lambda=1$ 在完整 episode 且 terminal bootstrap 为 0 时退化为 Monte Carlo return 减 $V$；若轨迹被截断，应从末状态 bootstrap，不能一律填 0。
- value baseline 本身若不依赖当前 sampled action，不会因“预测不准”自动让 vanilla policy-gradient estimator 有偏；偏差来自 bootstrap、$\gamma<1$ 对未折扣目标的替代、$\lambda<1$ 与函数逼近/截断等组合。原版笔记把“critic 不准 = estimator 有偏”说得过于笼统。

常见 joint objective（写成最大化）是：

$$J_{PPO}=J_{clip}-c_v\mathbb E[(V_\phi-V^{target})^2]+c_H\mathbb E[H(\pi_\theta)]$$

LLM post-training 还常加入相对 $\pi_{ref}$ 的 KL penalty，但它不是 PPO-Clip 定义本身必须包含的项。

## 4. Token ratio 与 sequence reward

PPO 把每个生成 token 当 action，故 token-level ratio 与 token-level advantage 是自然配对。若只有 terminal outcome $R_i$，GAE/return 可把信息传播到 earlier histories；这不等于获得了真正的 process supervision，critic 只能从数据中预测预期终局结果。

若把同一个 terminal $R_i$ 广播给所有 token，却在旧数据上多 epoch 用各 token 自己的 $\rho_{i,t}$，它不是完整 trajectory importance sampling。exact correction 应使用 $\rho_i^{seq}=\prod_t\rho_{i,t}$，但长序列乘积方差极高。PPO/GRPO 的 token-local clipped objective应被理解为实用 surrogate，而不是 exact off-policy identity。

## 5. 实现检查项

- rollout 时存 $\log\pi_b$；训练时重算 $\log\pi_\theta$，用 log-space 相减再 exponentiate。
- mask prompt、padding、environment/tool observation；只在真正由 policy 采样且要训练的 action token 上求 loss。
- terminal 与 truncation 分开；truncation 可能需要 bootstrap。
- report `approx_kl`, `clip_fraction`, ratio quantiles, value loss, explained variance 与有效 token 数。
- 不复制“固定学习率 / epoch 数适用于所有模型”的经验数字；它们依模型规模、batch、optimizer、reward scale 与系统延迟而变。

来源：PPO 原论文 [Schulman et al., 2017](https://arxiv.org/abs/1707.06347v2)，GAE 原论文 [Schulman et al., 2015](https://arxiv.org/abs/1506.02438)。
