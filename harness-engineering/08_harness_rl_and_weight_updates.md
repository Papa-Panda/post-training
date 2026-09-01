# 08 — Harness、RL 与 Weight Updates：双时间尺度协同优化

## 元信息

- 核心论文：[SIA](https://arxiv.org/abs/2605.27276v2) · [Continual Harness](https://arxiv.org/abs/2605.09998) · [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621)
- 相关目录：[`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) · [`vllm-rollout/`](../vllm-rollout/README.md) · [`ai-infra/`](../ai-infra/README.md) · [`model-aware-data-curation/`](../model-aware-data-curation/README.md)
- 本章目标：明确 harness 如何定义 rollout distribution、credit assignment 和数据 provenance；明确什么时候应该改 $h$、改 $\theta$，或两者都不改。

## 1. Harness 改变 RL 的环境

给定任务 $x$，harness-conditioned rollout：

$$\tau\sim p_{\theta,h}(\tau\mid x)=p(s_0\mid x,h)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid c_t)\,p_h(s_{t+1}\mid s_t,a_t),$$

$$c_t=C_h(s_{\le t},x).$$

这里沿用 README 的约定，把固定 control plane $q$ 从 $p_{\theta,h;q}$ 中省略；改变 permission、budget 或 verifier 也会改变实际轨迹，因此跨实验必须固定并记录 $q$。

因此 harness 同时改变：

1. policy 的 observation/context；
2. 合法动作与 tool API；
3. transition dynamics；
4. trajectory truncation、retry 和 verifier；
5. 哪些 rollout 被记录、过滤和用于训练。

Reward 写成：

$$R(\tau;h,V)=R_{\mathrm{task}}-\lambda_KK(\tau,h)-\lambda_L\mathrm{Lat}(\tau,h)-\lambda_QQ(\tau,h).$$

不记录 $h$ 的版本，就无法知道模型是在什么环境下产生行为。

## 2. Harness-aware trajectory schema

每条 rollout 至少绑定：

```text
model id + checkpoint + decoding config
harness digest + context compiler + workflow graph
component/tool/memory versions
permission + budget profile
environment / task / verifier versions
per-event observation, proposal, authorized action, result
terminal reason + reward decomposition
```

训练样本不是只有 $(x,y,r)$；至少是：

$$d_i=(x_i,\tau_i,r_i,\theta_i,h_i,q_i,V_i).$$

这使 replay filtering、off-policy correction、failure attribution 和可复现性成为可能。

## 3. Harness update 与 weight update 的边界

诊断问题：

| 失败 | 首选路径 |
|---|---|
| 信息未进入 context | $h^C$ |
| 正确能力但未激活/未坚持 | $h^C/h^W$ 或训练 |
| 流程、retry、join、tool schema 错 | $h^W/h^K$ |
| 缺少稳定的新语义/推理能力 | $\theta$ |
| evaluator/environment 故障 | $V/\mathcal E$，不要改 harness/model |

冻结模型能力上限：

$$J^\star(\theta)=\max_{h\in\mathcal H}J(\theta,h).$$

若在广泛的 $\mathcal H$ 内 $J^\star(\theta)$ 仍低，说明仅靠 harness 不够。但有限搜索失败不等于证明能力缺失；必须区分“未找到”与“不存在”。

## 4. Harness updating 与 harness benefit

令 solver model 为 $f$、evolver 为 $e$，演化后 harness 为 $H_T^{(f,e)}$。基于 [Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621)：

$$\Delta(f,e)=J_X(f,H_T^{(f,e)})-M_{\mathrm{base}}(f),$$

$$\Delta_{\mathrm{update}}(e)=\frac1{|F^\star|}\sum_{f\in F^\star}\Delta(f,e),$$

$$\Delta_{\mathrm{benefit}}(f)=\max_{e\in E^\star}\Delta(f,e).$$

$\Delta_{\mathrm{update}}(e)$ 衡量 evolver $e$ 产生的 harness updates 在 anchor agents 上带来的平均性能增益；$\Delta_{\mathrm{benefit}}(f)$ 衡量 model $f$ 从 anchor evolvers 中可获得的最大性能增益。两者不是一回事，也不能分别简化成“是否发生修改”和“是否遵循某个绝对最优 harness”。

论文中 best/worst evolver 在任何 benchmark 的最大差距只有 $3.1$ points，并且不存在一个 evolver 三项都最佳。Activation/adherence proxy：Qwen3-32B 的 SLR/HFR 为 $0.251/0.142$，Qwen3-235B 为 $0.961/0.350$，Opus 4.6 为 $0.957/0.757$。这支持：复杂 harness 是否有效，受 solver 激活和遵循能力制约。

## 5. $2\times2$ factorial：分离贡献与 interaction

比较旧/新 model $f_0,f_1$ 与旧/新 harness $h_0,h_1$：

|  | $h_0$ | $h_1$ |
|---|---:|---:|
| $f_0$ | $J_{00}$ | $J_{01}$ |
| $f_1$ | $J_{10}$ | $J_{11}$ |

Harness main effect under old model：

$$E_h^{(0)}=J_{01}-J_{00}.$$

Model main effect under old harness：

$$E_f^{(0)}=J_{10}-J_{00}.$$

Interaction：

$$I_{f,h}=J_{11}-J_{10}-J_{01}+J_{00}.$$

若 $I_{f,h}\gg0$，新模型解锁了新 harness；若 $I_{f,h}<0$，可能新模型不遵循旧 scaffolding、harness 过度约束或训练/部署接口漂移。只报 $J_{11}-J_{00}$ 无法知道收益来自哪里。

## 6. Dual-timescale optimization

模型更新昂贵、慢、影响面大；harness 更新便宜、快、局部。可用双时间尺度：

$$h_{t+1}=h_t+\alpha_t\widehat g_h(\theta_t,h_t),$$

$$\theta_{t+1}=\theta_t+\beta_t\widehat g_\theta(\theta_t,h_t),\qquad \beta_t\ll\alpha_t.$$

这里 $g_h$ 通常不是可微梯度，而是黑盒 search/evolution 的抽象方向。关键工程规则：

1. 固定 $\theta$，先做 bounded harness exploration；
2. harness 进入稳定版本后再收集训练数据；
3. weight update 后做 $2\times2$ cross-evaluation；
4. 若模型行为分布改变，重新验证 harness，而不是假定兼容；
5. 控制同时更新，避免 attribution 消失。

## 7. Off-policy mismatch

训练数据来自旧 pair $(\theta_b,h_b)$，新 policy/harness 是 $(\theta,h)$。重要性比率形式上是：

$$w(\tau)=\frac{p_{\theta,h}(\tau)}{p_{\theta_b,h_b}(\tau)}.$$

但当 tool schema、workflow 或可达状态改变时，两处分布 support 可能不同，$w$ 不可估或方差极大。常见错误是只计算 token policy ratio：

$$\prod_t\frac{\pi_\theta(a_t\mid c_t)}{\pi_{\theta_b}(a_t\mid c_t)},$$

却忽略 context compiler、authorized action、retry 和 transition 改变。实际策略：

- 只在兼容 harness family 内 replay；
- 记录 behavior harness digest；
- schema/action-space 变化后 fresh rollout；
- 对 stale rollout 降权或丢弃；
- 让 rollout serving 与 trainer 做 provenance handshake。

## 8. Credit assignment across layers

总回报变化不是模型单独贡献：

$$\Delta J\approx \nabla_\theta J\cdot\Delta\theta+\Delta_hJ+I_{\theta,h}+\Delta_{\mathcal E}J+\Delta_VJ.$$

Failure attribution 要回答：

- model proposal $a_t$ 是否已正确？
- runtime authorized action $u_t$ 是否错误改变？
- tool/environment 是否失败？
- verifier 是否错误打分？
- context 是否遗漏关键证据？

如果 $a_t$ 正确而 $u_t$ 错，训练模型可能反而学坏；如果 $a_t$ 一贯错误且信息充足，单纯加 workflow 可能只掩盖 capability gap。

## 9. SIA：feedback agent 选择 update surface

[SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276v2) 的 Meta-Agent 编排 Task Agent 和 Feedback-Agent。Feedback-Agent 分析 execution history，在 harness update 与 weight update 间选择；实验通常先多轮 scaffold iteration，reward plateau 后执行 RL update。

可抽象为 routing policy：

$$d_t^{\mathrm{meta}}\in\{\mathrm{update\_harness},\mathrm{update\_weights},\mathrm{collect\_data},\mathrm{stop}\}.$$

论文 v2 Table 3：

- LawBench：$13.5\%\to50.0\%\to70.1\%$（base→harness→joint）；joint 比 harness-only 高 $20.1$ points；
- TriMul reward：$0.105\to0.120\to1.475$；runtime $12{,}483\ \mu s\to1{,}017\ \mu s$；
- denoising $mse_{\mathrm{norm}}$：$0.048\to0.241\to0.289$。

Weight methods 依任务不同：LawBench PPO+GAE、TriMul entropic advantage weighting、denoising GRPO。论文也明确提出 coupled co-evolution 的 Goodhart 风险：harness 和 model 同时适应 evaluator，可能共同放大 proxy 漏洞。

## 10. Continual Harness：reset-free state 与 co-learning

[Continual Harness](https://arxiv.org/abs/2605.09998) 在长期 acting loop 中不 reset 环境；每 $F$ 步 Refiner 对 system prompt、subagents、skills 和 memory 做 CRUD：

$$h_{k+1}=\mathrm{Refiner}(h_k,\tau_{kF:(k+1)F}).$$

Co-learning 每轮收集 $K=256$ steps，使用 pairwise PRM、frontier teacher relabel 和 soft SFT，同时 emulator state 不 reset。该设置突出长期状态、行为数据与 harness 演化的耦合。

Pokémon Emerald / Gemini 3.1 Pro 报告 from-scratch $100\%$ milestones、median cost US$130；minimalist harness 为 $98\%$、US$215。边界：较弱模型存在 capability floor，自我精炼可能恶化；论文没有证明 convergence，也没有 matched reset-based baseline。早期 GPP 完成 Pokémon 是 human-supervised 系统，不能归因给 fully automated Continual Harness。

## 11. 与 rollout serving / trainer 的接口

### Rollout request

```json
{
  "model_digest": "...",
  "harness_digest": "...",
  "task_ref": "...",
  "seed": 17,
  "attempt": 2,
  "budget_profile": "eval-v4",
  "verifier_ref": "external:v9"
}
```

### Rollout result

```json
{
  "trajectory_ref": "artifact://...",
  "terminal_reason": "verified_success",
  "reward_terms": {"task": 1.0, "cost": -0.04, "risk": 0.0},
  "model_digest": "...",
  "harness_digest": "...",
  "environment_digest": "..."
}
```

Trainer ingestion gate：schema/version compatible、reward provenance valid、no evaluation leakage、policy/harness staleness within threshold、tool side effects reconciled。

## 12. Joint Goodhart risk

单独优化 model 或 harness 已会利用 reward 漏洞；联合优化更危险：

$$\max_{\theta,h}\mathbb E[\hat R(\tau)]\not\Rightarrow\max_{\theta,h}\mathbb E[R^\star(\tau)].$$

可能的共谋式 failure：

- harness 隐藏某些 observation，model 学会只在剩余 proxy 上表现；
- verifier 可预测，model 与 workflow 共同生成表面证据；
- context compiler 选择性丢弃失败；
- retry/filter 只保留高 reward rollout，制造 selection bias。

防线：冻结 evaluator、独立 adversarial checks、periodic human audit、不同 verifier ensembles、untouched final tasks、model/harness factorial、provenance-complete replay。

## 13. Failure modes

- harness 变了但 replay buffer 未标版本；
- 训练吃到 hidden test traces；
- 只按最终 reward 训练，不看 authorized action/side effects；
- model update 与 harness update 同轮大改，无法归因；
- token-level importance ratio 被误当成完整 trajectory correction；
- stronger model 不遵循旧 harness，却被误判为能力退化；
- weaker model 被复杂 harness 压垮，却继续增加 scaffolding；
- rollout server 为吞吐隐式改变 attempts/context 长度。

## 14. Engineering checklist

- [ ] 每条 trajectory 绑定 model、harness、tool、environment、verifier digest；
- [ ] rollout request/response 使用明确 schema；
- [ ] harness major change 后 fresh rollout；
- [ ] harness/weight 更新采用双时间尺度并做 $2\times2$ 评估；
- [ ] updater routing 包含 model、harness、data、environment、verifier 和 stop；
- [ ] replay ingestion 检查 staleness、leakage 和 action-space compatibility；
- [ ] throughput optimization 不得静默改变 evaluation semantics；
- [ ] joint loop 有独立 evaluator 和 Goodhart audit。

<!-- NAVIGATION -->
## 导航

- 上一篇：[07 Observability、Evaluation 与 Security](07_observability_evaluation_security.md)
- 下一篇：[论文地图与逐项核验](papers.md)
- 回到：[专题 README](README.md)
