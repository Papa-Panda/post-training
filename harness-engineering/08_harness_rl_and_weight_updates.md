# 08 — Harness、RL 与权重更新：双时间尺度优化

## 元信息

- 核心方向：[SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276)
- 相关专题：[`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) · [`model-aware-data-curation/`](../model-aware-data-curation/README.md) · [`vllm-rollout/`](../vllm-rollout/README.md)
- 论文状态：SIA 为 2026 预印本；联合优化证据仍属早期，不作为成熟工业结论。

## 1. 两个可学习对象

完整系统同时有模型权重 $\theta$ 和 harness $h$：

$$J(\theta,h)=\mathbb E_{x,\tau\sim p_{\theta,h}}[R(\tau,x)].$$

可以交替更新：

$$h_{t+1}=\arg\max_h\hat J(\theta_t,h;D_t),$$

$$\theta_{t+1}=\theta_t+\Delta\theta(h_{t+1},\mathcal B_t).$$

但两个更新会互相改变分布：新 harness 改变 rollout/context/tool usage，导致训练数据 $\mathcal B_t$ 变化；新权重又改变旧 harness 的最佳策略。因此不能把两者的单独增益简单相加。

## 2. 双时间尺度

实践上 harness 更新快、便宜、可回滚；权重更新慢、成本高、能内化能力。可令：

$$h_{t+1}=h_t+\alpha_t\Delta h_t,\qquad \theta_{t+1}=\theta_t+\beta_t\Delta\theta_t,\qquad \beta_t\ll\alpha_t.$$

这里不是把离散代码真的相加，而是表示 harness 在短周期迭代，权重在积累足够证据后较慢发布。

## 3. 什么时候改哪一层

| 失败根因 | 首选动作 | 原因 |
|---|---|---|
| 缺少必要文件/工具结果 | context/retrieval edit | 权重更新无法恢复未提供的信息 |
| 工具顺序、重试、timeout 错误 | workflow/middleware edit | 这是控制流问题 |
| 权限或副作用不安全 | permission boundary | 不能靠模型“自觉”替代强制策略 |
| 跨任务反复不会某种推理/协议 | SFT/RL/weight update | 值得内化并减少每次 prompt 成本 |
| 训练池缺能力覆盖 | data curation/generation | 先补训练信号 |
| rollout serving 慢/OOM | vLLM/infra optimization | 不是 harness 逻辑问题 |

## 4. SIA：由反馈 Agent 选择更新层

[SIA](https://arxiv.org/abs/2605.27276) 探索把三类角色放进同一闭环：Meta-Agent 提出 harness，Task-Specific Agent 执行任务，Feedback-Agent 根据近期 trajectories 决定下一轮更新 harness 还是模型权重。

概念上可写为控制策略：

$$u_t\in\{\mathrm{UPDATE\_HARNESS},\mathrm{UPDATE\_WEIGHTS},\mathrm{COLLECT\_MORE}\},$$

$$u_t=\pi_{\mathrm{feedback}}(T_{t-k:t},E_{t-k:t},B_t).$$

博客指出其现有实验有明显 confound，例如不同角色使用的模型能力不一致、baseline 较弱，因此本专题只把它作为研究方向，不把结果写成联合优化已经优于单独优化的定论。

## 5. Harness-aware RL

在 Agentic RL 中，policy 的环境实际上包含 harness：

$$\tau\sim p(\tau\mid\theta,h,\mathcal E).$$

如果训练时 harness 与部署时不同，得到 distribution shift。至少要版本化：

- rollout 用的 harness digest；
- tool schema / implementation；
- context policy 与 memory snapshot；
- verifier/reward version；
- permission 和 budget。

[`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) 负责 advantage、KL、critic/group baseline；本专题负责产生 trajectory 的外部 runtime 是否一致、可追踪。

## 6. 与 model-aware data curation 的闭环

三个优化轴可以写成：

$$\max_{\theta,h,\mathcal B}J(\theta,h;\mathcal B),$$

其中：

- [`model-aware-data-curation/`](../model-aware-data-curation/README.md) 选择/生成 $\mathcal B$；
- RL/SFT 用 $\mathcal B$ 更新 $\theta$；
- Harness Engineering 根据 rollout failure 更新 $h$。

推荐顺序不是盲目联合搜索，而是 root-cause routing：

```text
trace failure
  -> missing/incorrect data?       -> curate/generate data
  -> model capability failure?     -> weight update
  -> context/tool/workflow failure?-> harness edit
  -> serving/resource failure?     -> rollout infra
```

## 7. Credit assignment 与非平稳性

若一次同时改 $h$、$\theta$ 和数据，无法知道收益来源。实验矩阵至少包括：

| Arm | Harness | Weights | Data |
|---|---|---|---|
| A | fixed | fixed | fixed |
| B | updated | fixed | fixed |
| C | fixed | updated | fixed |
| D | fixed | fixed | updated |
| E | updated | updated | fixed |
| F | updated | updated | updated |

并在共同 held-out 上报告质量、成本、风险与 interaction：

$$\mathrm{Interaction}=\Delta J_{h+\theta}-\Delta J_h-\Delta J_\theta.$$

## 8. 最终系统形态

```text
traces + evaluator
      |
root-cause router
  |       |        |
harness  data    weights
  |       |        |
  └── versioned rollout runtime ──> new traces
```

人类应上移到目标、权限、evaluator、重大发布和模糊判断，而不是从所有执行步骤中完全移除。

<!-- NAVIGATION -->
## 导航

- 上一篇：[07 Observability、Evaluation 与 Security](07_observability_evaluation_security.md)
- 下一篇：[论文证据账本](papers.md)
- 回到：[专题 README](README.md)
