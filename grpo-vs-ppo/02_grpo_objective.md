# 02 — GRPO：group-relative advantage、KL 与边界

GRPO 的主要动机不是“组内排名”，而是用同一 prompt 的多条 completion reward 构造 baseline，省掉 learned critic。原始 DeepSeekMath 同时给出 outcome-supervision 与 process-supervision 形式；本章先聚焦最常见的 outcome GRPO。

## 1. Sampling 与 reward

对每个 $q$，behavior policy 独立采样 $G$ 条 response：

$$y_1,\ldots,y_G\sim\pi_b(\cdot\mid q),\qquad R_i=R(q,y_i)$$

组均值与 population standard deviation 为：

$$\bar R=\frac{1}{G}\sum_{j=1}^{G}R_j,\qquad s_R=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\bar R)^2}$$

原始 outcome GRPO 使用：

$$A_i^{grp}=\frac{R_i-\bar R}{s_R}$$

然后令该 response 的所有可训练 token 共享 $A_{i,t}=A_i^{grp}$。实现必须显式定义标准差是 population 还是 sample convention；二者只差 scale，但会改变实际 step size。示例代码默认 population standard deviation，与上式一致。

## 2. 原始 token-level clipped objective

令 $c(x,l,u)=\min(\max(x,l),u)$。忽略 mask 的原始 response-mean 形式为：

$$J_{GRPO}=\mathbb E\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}\left(\min(\rho_{i,t}A_i^{grp},c(\rho_{i,t},1-\epsilon,1+\epsilon)A_i^{grp})-\beta k_{i,t}\right)\right]$$

这里 $\rho_{i,t}=\pi_\theta/\pi_b$，而 KL 项比较 $\pi_\theta$ 与 $\pi_{ref}$。两个 denominator 不同：

- $\pi_b$ 是本批 rollout 的来源，解决数据复用期间的 policy drift。
- $\pi_{ref}$ 是训练锚点，限制长期行为漂移。

原始 DeepSeekMath 的多次 GRPO update 用 token-level ratio；DeepSeek-R1 论文正文把 objective 压缩成 sequence notation。不要据此把 token product ratio 与单 token ratio 混用。

## 3. KL sample estimators

设 sampled token 来自当前 policy $a\sim p=\pi_\theta(\cdot\mid h)$，参考分布为 $q=\pi_{ref}(\cdot\mid h)$，并定义 $z=\log p(a)-\log q(a)$。常见三个量是：

$$k_1=z$$

$$k_2=\frac{1}{2}z^2$$

$$k_3=\exp(-z)-1+z$$

在 **样本确实来自 $p$** 时，$k_1$ 和 $k_3$ 都是 forward KL $D_{KL}(p\Vert q)$ 的 unbiased value estimator；$k_3\ge 0$ pointwise，通常更稳定，$k_2$ 是近 $p=q$ 时的二阶近似而非一般 unbiased estimator。原始 DeepSeekMath 使用 $k_3$。

如果 token 实际来自 $\pi_b$ 且 $\pi_b\neq\pi_\theta$，直接平均 $k_3$ 不再是当前-policy KL 的 unbiased value estimate。理论上 value estimation 需要乘 $\pi_\theta/\pi_b$；而“把某个 KL estimator 当 differentiable loss”还涉及 sampling-distribution 的 score-function derivative，不能只凭 value estimator 无偏就宣称 gradient 无偏。工程上要明确自己是在做 exact estimate、近似 penalty，还是仅监控指标。

## 4. Group mean 的偏差细节

“baseline 与 sampled action 独立”是 vanilla baseline 不改变期望梯度的条件。$\bar R$ 包含 $R_i$ 自身，因此不独立。先不做 std normalization，可直接算出：

$$\mathbb E\left[\frac{1}{G}\sum_i(R_i-\bar R)\nabla\log\pi(y_i\mid q)\right]=\frac{G-1}{G}\nabla\mathbb E[R\mid q]$$

它保留方向但缩小 $\frac{G-1}{G}$。leave-one-out baseline 则排除自身：

$$b_{-i}=\frac{1}{G-1}\sum_{j\ne i}R_j,\qquad A_i^{LOO}=R_i-b_{-i}=\frac{G}{G-1}(R_i-\bar R)$$

在独立 sampling 与无额外非线性时，LOO 恢复 unbiased REINFORCE estimator。GRPO 再除以随机的组内 $s_R$，会按 prompt/reward dispersion 重加权，不能再笼统称为 unbiased expected-reward gradient；clipping 和 token-local ratios 又引入额外 surrogate bias。这些偏差可能换来更好的尺度稳定性，但必须准确命名。

## 5. Edge cases

| 情况 | 数学结果 | 安全处理 |
|---|---|---|
| $G=1$ | 无法构造相对 baseline；$s_R=0$ | 拒绝配置或改用 absolute/critic estimator |
| 全对、全错或全同分 | $s_R=0$ 且 centered reward 全 0 | 返回零 advantage；可记录后跳过，或重采样 prompt |
| $s_R$ 很小 | 除法放大 noise | threshold/epsilon、batch-level scale，或关闭 std scaling |
| 大量 ties | 有效比较信号少 | 记录 unique rewards 与 non-degenerate-group rate |
| reward 各 component 尺度不同 | z-score 掩盖绝对 scale | 分量先校准，并记录 raw reward |
| 同 prompt rollout 非独立 | 标准 group 推导失效 | 记录 sampler coupling；不要声称 i.i.d. |
| 多任务 prompt 难度不同 | 每组除 $s_R$ 改变 prompt 权重 | 与 no-std / batch-std 做消融 |

## 6. Length aggregation 与后续修正

原始 response-mean objective 的 $1/T_i$ 让每条 response 总权重相同，因此短 response 的单 token 权重更大。Dr.GRPO 指出 response length normalization 与 per-question std normalization 会造成 response-level length bias 与 question-level difficulty bias，并提出固定 normalization / 去 std 的修正。DAPO 改用全局 token mean、dynamic sampling、asymmetric clipping 和 overlong reward shaping。

注意：token mean 不是“无偏”的同义词，它改成每个 token 等权，因而长 response 总梯度权重更大。选择 aggregation 必须匹配想优化的是 response distribution 还是 token distribution。

## 7. GRPO 能否做细粒度 credit assignment？

不能简单回答“不能”。

- outcome GRPO 把一个 terminal reward 广播到整条 response，确实没有识别哪一步真正贡献了结果。
- DeepSeekMath 原论文还描述了 process-supervision GRPO，可由每一步 reward 的后缀归一化和得到 token-varying advantage。
- 但只要没有 learned value/return model，跨很长 horizon 的 bootstrap 与任意截断片段估值仍是短板。

所以准确边界是：**vanilla outcome GRPO 缺少 token-specific credit；GRPO 框架可接 process reward，但 process labels/verifier 本身有成本。**

来源：[DeepSeekMath](https://arxiv.org/abs/2402.03300)、[DeepSeek-R1](https://arxiv.org/abs/2501.12948)、[Dr.GRPO](https://arxiv.org/abs/2503.20783)、[DAPO](https://arxiv.org/abs/2503.14476)、[KL estimator pitfalls](https://arxiv.org/abs/2506.09477)。
