# 00 — 统一记号 / Notation

本专题把语言模型生成看成有限时域 MDP。先固定记号，避免把 **behavior / old policy** 和 **reference policy** 混为一谈。

| 符号 | 含义 | 形状 / 生命周期 |
|---|---|---|
| $q$ | prompt / query | 一个 group 共用 |
| $y_i=(y_{i,1},\ldots,y_{i,T_i})$ | 第 $i$ 条 completion | $i\in\{1,\ldots,G\}$ |
| $h_{i,t}=(q,y_{i,<t})$ | token $t$ 前的 history/state | prompt + 已生成 token |
| $m_{i,t}\in\{0,1\}$ | policy-loss mask | 通常只训练 assistant token |
| $R_i=R(q,y_i)$ | outcome reward | 每条 completion 一个标量 |
| $r_{i,t}^{env}$ | environment/process reward | 不要和 probability ratio 混用 |
| $\pi_b$ | behavior policy | 生成当前 rollout 的冻结快照；也常写 $\pi_{old}$ |
| $\pi_\theta$ | trainable policy | 在同一批 rollout 上做一次或多次更新 |
| $\pi_{ref}$ | reference policy | KL 锚点，通常在一个训练阶段内冻结 |
| $V_\phi(h)$ | critic/value model | 估计从 history 出发的期望 return |
| $\rho_{i,t}(\theta)$ | token importance ratio | $\pi_\theta(y_{i,t}\mid h_{i,t})/\pi_b(y_{i,t}\mid h_{i,t})$ |
| $\rho_i^{seq}(\theta)$ | exact sequence ratio | $\pi_\theta(y_i\mid q)/\pi_b(y_i\mid q)$ |
| $A_{i,t}^{GAE}$ | PPO token advantage | 每个 token 可以不同 |
| $A_i^{grp}$ | group-relative advantage | outcome GRPO 中广播给该 response 的 token |

Autoregressive factorization gives:

$$\pi_\theta(y_i\mid q)=\prod_{t=1}^{T_i}\pi_\theta(y_{i,t}\mid h_{i,t})$$

所以 exact sequence ratio 是 token ratios 的乘积：

$$\rho_i^{seq}(\theta)=\prod_{t=1}^{T_i}\rho_{i,t}(\theta)=\exp\left(\sum_{t=1}^{T_i}\log\rho_{i,t}(\theta)\right)$$

这和 GRPO/PPO 常用的 **逐 token ratio + 逐 token clipping** 不是同一个对象。前者是完整轨迹分布的 importance ratio；后者是低方差、可训练的局部 surrogate。长序列上乘积很容易爆炸或消失，不能把两者口头都叫 “ratio” 后直接互换。

## 三个 policy 的职责

1. $\pi_b$ 回答“数据从哪里来？”；ratio clipping 约束当前更新别离本批数据太远。
2. $\pi_{ref}$ 回答“长期锚在哪里？”；KL 防止策略远离初始/阶段参考模型。
3. $\pi_\theta$ 是正在求梯度的模型。

若 rollout 后只做一次、且更新前 $\theta=b$，则 $\rho_{i,t}=1$。多 epoch 重用该批数据后，$\pi_\theta\neq\pi_b$，ratio 才开始偏离 1。$\pi_b$ 可以只保存 rollout log-probabilities，不一定需要另一份常驻模型；$\pi_{ref}$ 若要精确 KL 则通常需要 reference logits/log-probabilities。

## 两种平均方式不是小细节

Response mean（原始 GRPO）先对每条 response 的 token 取平均，再对 $G$ 条 response 平均：

$$J_{resp}=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{M_i}\sum_{t=1}^{T_i}m_{i,t}\ell_{i,t},\qquad M_i=\sum_{t=1}^{T_i}m_{i,t}$$

Token mean（DAPO）对整批有效 token 一次平均：

$$J_{tok}=\frac{\sum_i\sum_t m_{i,t}\ell_{i,t}}{\sum_i M_i}$$

前者让每条 response 总权重相同，单个短 response token 权重更大；后者让每个 token 权重相同，长 response 总权重更大。它们优化的是不同的加权目标，不能只当实现细节。
