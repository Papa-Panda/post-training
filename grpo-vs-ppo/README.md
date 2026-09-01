# GRPO vs PPO — estimator、ratio 与 systems trade-offs

PPO 和 GRPO 都不是一个固定 recipe。这个专题统一 token/group 记号，从 policy gradient 推到 clipped surrogate，拆开 behavior policy 与 reference policy，并把算法选择落到 memory、communication、rollout-token budget 与 failure diagnostics。

## 先纠正四个常见误解

1. **GRPO 不等于“按排名训练”**：原始形式用组内 reward 的 mean/std normalization；只有 reward 本身来自 ranking 时才可口头说“排名”。
2. **PPO 不等于“四个模型常驻”**：actor、critic、reference、reward 的部署方式各异；behavior policy 也可只保存 old log-probs。rule reward 不需要 reward model。
3. **去 critic 不等于固定省 30–50% 显存**：确定省掉 critic path，但总 peak 还取决于 optimizer state、sharding、activations、reference 与 rollout KV cache。
4. **critic 不等于 dense ground truth**：GAE 可给 token-specific estimate 和 bootstrap，但 value model 仍只是在拟合 return；错误 reward 不会被 optimizer 修好。

## Map

```text
00_notation.md              unified q / group / token / policy notation
01_ppo_objective.md         PPO-Clip derivation, sign behavior, GAE, truncation
02_grpo_objective.md        group advantage, KL estimators, bias and edge cases
03_math_comparison.md       baseline/ratio/aggregation axes and method boundaries
04_infra_tradeoffs.md       memory, collectives, token budget, async staleness
05_code/                    dependency-free reference code + semantic tests
06_glm52_ppo_comeback.md    sourced long-horizon compaction case study
papers.md                   primary-source ledger
```

## One-screen comparison

| 维度 | Critic PPO | Outcome GRPO |
|---|---|---|
| rollout organization | 每 prompt 可一条或多条 | 同 prompt 要 $G>1$ completions |
| advantage | token/state-specific GAE | group-normalized outcome，通常整条 response 广播 |
| learned baseline | $V_\phi(h_t)$ | none |
| extra cost | critic state、forward/backward、collectives | grouped generation、group sync、可能更多 rollout tokens |
| degenerate case | critic collapse / poor bootstrap | all-equal reward group gives zero signal |
| temporal credit | 可 bootstrap；仍依赖 critic/reward quality | vanilla outcome form 粗；process-supervision variant 可更细 |
| ratio unit | 常用 token ratio | 原始 GRPO 也常用 token ratio；不要和 sequence product 混写 |
| natural starting point | variable-length segments / critic 可可靠拟合 | short verifiable outcomes / critic 成本高 |

## Minimal equations

Token ratio uses the behavior snapshot:

$$\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid q,y_{i,<t})}{\pi_b(y_{i,t}\mid q,y_{i,<t})}$$

PPO-style clipped token surrogate, with $c(x,l,u)=\min(\max(x,l),u)$:

$$s(\rho,A)=\min(\rho A,c(\rho,1-\epsilon_l,1+\epsilon_h)A)$$

PPO usually inserts GAE:

$$A_{i,t}^{GAE}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{i,t+l}$$

Outcome GRPO inserts group-normalized reward:

$$A_i^{grp}=\frac{R_i-\bar R}{s_R}$$

KL compares the trainable policy to a separate reference anchor:

$$D_{KL}(\pi_\theta\Vert\pi_{ref})$$

## Selection workflow

不要先问“PPO 还是 GRPO”，先量：

1. reward 是 terminal、process 还是可被 hack 的 proxy？
2. 同 prompt 能否产生独立、可比较且 non-degenerate 的 group？
3. 固定 generated-token budget 下，增大 $G$ 后 prompt coverage 损失多少？
4. critic 在 held-out trajectories 上的 value error / explained variance 如何？
5. peak memory 主项是 critic、actor optimizer、activations 还是 KV cache？
6. trajectory 是自然终止还是基础设施截断？
7. loss 按 response、segment 还是 token 加权？

没有这些测量时，不给固定 `G`、KL coefficient、GPU 数或内存节省百分比。原论文里的具体设置是复现实验起点，不是跨任务推荐。

## Run the checks

```bash
cd grpo-vs-ppo/05_code
python3 ppo_vs_grpo_advantage.py
python3 -m unittest -v test_rl_objectives.py test_docs.py
python3 -m py_compile *.py
```

Primary sources and claim scope are in [`papers.md`](papers.md).
