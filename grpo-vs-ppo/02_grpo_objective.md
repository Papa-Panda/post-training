# 02 — GRPO 目标 / GRPO Objective (DeepSeekMath 2024)

## 动机 Motivation

math/code 的 reward 往往是 verifiable 的标量结果：解对 1 / 错 0，测试通过率。学 $V(s)$ 不划算，方差来源改为**组内比较**。

> No critic. Group as baseline.

## 采样 Sampling

对每个 query $q$，旧策略 $\pi_{old}$ 采样 $G$ 个输出：

$$
\{o_1,\dots,o_G\} \sim \pi_{old}(\cdot|q)
$$

每个拿标量 reward $r_i = R(q,o_i)$（规则 / model reward 都行）。

## Group Relative Advantage

$$
\mu = \frac1G\sum r_i,\quad \sigma = \sqrt{\frac1G\sum (r_i-\mu)^2}
$$

$$
\hat{A}_i = \frac{r_i - \mu}{\sigma + \varepsilon_0}
$$

关键：**同一 token 都共享同一个 $\hat A_i$**，但 loss 按 token 平均，优雅对应 outcome-level reward。

EN: one outcome scalar → broadcast to all tokens in that output.

变体：DeepSeek-R1 强调 $σ$-norm 控方差；若 $\sigma=0$（全对/全错）置 0 advantage 或跳过该 group。

## GRPO Objective

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}_{q,\{o_i\}}\left[ \frac1G\sum_{i=1}^G \frac1{|o_i|}\sum_{t=1}^{|o_i|} \min\Big(r_{i,t}(\theta)\hat A_i,\; \text{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat A_i\Big) - \beta D_{KL}(\pi_\theta||\pi_{ref})\right]
$$

$$
r_{i,t}(\theta)=\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|...)}
$$

KL 常用 unbiased estimator:

$$
D_{KL}\approx \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} -1
$$

或直接 KL-penalty loss。

## 去 critic 的效果

- 内存：省 1x 7B/70B value 模型 → 70B 时省 >60GB；通信也少。
- 稳定：无 value warmup / loss balancing $c_1$ 之争。
- 代价：不能做 token-level credit assignment，密集中间奖励场景信息丢。

## vs PPO 形式对应

PPO 学 $V(s)$ 当 baseline，GRPO 用 $μ$。两者都是 $A = r - b$ 的 REINFORCE variance reduction。GRPO 的 $b$ 是 leave-one-group 均值，PPO 的 $b$ 是 learned $V$。

> ZH：一句话：GRPO = PPO把$A^{GAE}$换成组排名标准化，去掉 critic。  
> EN: GRPO = PPO with group-ranked advantage, critic removed.

## 扩展 Ext

- Dr.GRPO: 去掉 $|o_i|$ 长度平均 bias，去掉 σ-norm 让方差真正反映任务难度。
- DAPO / VAPO: dynamic filtering (skip 100% / 0% groups), token-level weighting.
