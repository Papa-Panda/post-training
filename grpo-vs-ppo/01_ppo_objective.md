# 01 — PPO 目标 / PPO Objective

## 定义 Definition

PPO (Schulman 2017) 解决 $J(\theta)=\mathbb{E}[r(\tau)]$ 的不稳定大步问题。

Policy ratio:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

Clipped surrogate:

$$
\mathcal{L}^{CLIP}(\theta)=\mathbb{E}_t\left[ \min\big(r_t(\theta)A_t,\; \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t\big)\right]
$$

Overall (with value + entropy):

$$
\mathcal{L}_{PPO}= -\mathcal{L}^{CLIP} + c_1 \underbrace{(V_\phi(s_t)-V^{targ}_t)^2}_{\text{critic}} - c_2 \mathcal{H}[\pi_\theta]
$$

## Advantage — GAE

TD residual:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE($\gamma$,$\lambda$):

$$
A^{GAE}_t = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}
$$

$\lambda=0$ → one-step bias 大 / variance 小； $\lambda=1$ → Monte Carlo unbiased / variance 大。常用 0.95。

## Critic / Value

- 必须学 $V_\phi(s)$ 去做 baseline，降方差。
- 实践：value 通常独立 head，或独立 model (RLHF 常用 4 模型分开)。
- 痛点：sparse reward / code 执行只有最终 $0/1$，critic 拟合慢、偏差传给 policy。

## 为啥在 RLHF 曾是标配

- KL 控散：加 $\beta D_{KL}(\pi||\pi_{ref})$ 或 early stop by KL。
- PPO 对超参相对稳，比 vanilla PG 能走多 epoch。

> ZH：PPO 是“带刹车的 PG”，用 ratio clip + GAE 方差控制。代价是多一个要精调的 value 网络。  
> EN：PPO is braked PG: clip ratio, smooth with GAE; price is a value net you must tune.

## 调参经验

- $\epsilon=0.2$, clip grad 1.0, 4-8 PPO epochs per batch
- lr 1e-6 ~ 3e-6 for 7B+ RLHF, critic lr 可 2x
- $γ=1.0$ for bandit / single-turn, 0.99 for multi-turn
