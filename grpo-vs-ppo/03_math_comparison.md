# 03 — 数学比分 / Math Scorecard

## 目标分解

PPO:

$$
\nabla J_{PPO} \propto \mathbb{E}[ \hat A^{GAE}_t \nabla \log \pi \cdot \mathbf{1}_{clip}]
$$

GRPO:

$$
\nabla J_{GRPO} \propto \mathbb{E}[\hat A^{group}_i \nabla \log \pi]
$$

两者都是 Policy Gradient 的 baseline 变体：降低 variance 不引入 bias 只要 $b$ 与 $a$ 独立。

## 偏差 Bias

| Method | Baseline $b$ | Bias |
|---|---|---|
| PPO | $V_\phi(s)$ learned 拟合偏 | $V$ 不准 → estimator 有偏，但 GAE $λ<1$ 主动引入 bias 换 variance |
| GRPO | $\mu_{group}$ 为 Monte Carlo 均值 | 无偏 if $G$ 独立采样；组内共享导致轻微相关但实践可忽略 |

> Math/code 0/1 reward：$V$ 偏差 > group 偏差，GRPO 占优。

## 方差 Variance

$$
\text{Var}[\hat A^{GAE}] \propto \frac{1-\lambda^{...}}{...} \approx O(\frac{1}{1-\gamma\lambda}) \text{但被 }V \text{ 平滑}
$$

$$
\text{Var}[\hat A^{group}] \propto \frac{\text{Var}[r_i]}{\sigma^2}=O(1) \text{ (标准化后)},\; \text{但 token 间同 advantage 导致协方差项 }|o_i|\text{放大}
$$

- Dr.GRPO 去掉除以 $σ$，variance 随任务难度自适应：简单题自动缩梯度。
- 经验法则：$G=16$~32 对 7B 足够降方差到和 PPO-critic 相当；$G=64$ 对 hard math 更好。

## 内存 / 计算

设 model params $P$:

- PPO: peak = $P_{actor}+P_{critic}+P_{ref}+P_{rw}+2*optim$ ≈ 3-4x (ZeRO后)
- GRPO: peak = $P_{actor}+P_{ref}+P_{rw}$ ≈ 2x；省 $P+opt$。70B 场景：A100 80G 单节点可 RLOO/GRPO，PPO 需 2 节点。

Compute token cost:

- Rollout dominates: both $O(G \cdot L)$. PPO extra forward for $V(s)$ 每token。
- GRPO 用 vLLM for $G$ 并行 sampling 吞吐高，因无 critic forward。

## 何时赢 When GRPO wins

- verifiable reward: math boxed answer, code unit test
- sparse outcome: $r$ 只在 EOS
- 模型大，显存瓶颈

何时 PPO 赢:

- dense / process reward model
- 需细粒度 credit assignment: multi-turn tool use 需 $V$ 播价值
- 小模型 + 充足 critic 预训练

## 统一视角 Unified

Both minimize KL-regularized RL:

$$
\max_\pi \mathbb{E}[R]-\beta KL(\pi||\pi_{ref})
$$

最优解 $\pi^* \propto \pi_{ref}\exp(R/\beta)$。PPO/GRPO 只是用不同 estimator 逼近这个目标。

> GRPO 是 PPO 去 critic 的 special case of RLOO with group norm.

Scoring (满分5):

- **数学严谨**: PPO 4.5 (成熟收敛证明) > GRPO 3.5 (经验为主，方差分析不完整)
- **硬件友好**: GRPO 5 > PPO 3
- **code/math 适配**: GRPO 5 > PPO 3.5
- **可扩展到多步 agentic**: PPO 4 > GRPO 3（需扩展 token weighting）
