# Harness Engineering

> **优化 frozen model 如何被运行，而不只是优化模型权重。** 本专题把 context、workflow、可执行代码、工具、副作用、持久状态、subagents、评估、权限与发布统一成一个可测量、可回归、可审计的控制系统。

## 1. 统一问题

给定任务 $x\sim\mathcal D$、冻结参数 $\theta$ 与 harness $h$，runtime 诱导出完整轨迹分布：

$$\tau\sim p_{\theta,h;q}(\tau\mid x),\qquad J_q(\theta,h)=\mathbb E[R(\tau,x)-\lambda_KK(\tau)-\lambda_L\mathrm{Lat}(\tau)-\lambda_QQ(\tau)].$$

$R$ 是任务收益，$K$ 是 token/工具/GPU 成本，$\mathrm{Lat}$ 是延迟，$Q$ 是安全或回归风险；$q$ 是固定的外置 control plane。后文在 $q$ 固定时简写为 $p_{\theta,h}$ 和 $J(\theta,h)$。Post-training 主要优化 $\theta$；Harness Engineering 在固定模型下优化外部可执行结构：

$$h^\star(\theta;q)=\arg\max_{h\in\mathcal H}J_q(\theta,h).$$

这是一种**非参数优化**：没有反向传播到模型权重，但会通过改变 prompt、信息、动作空间、控制流、状态和 verifier feedback，改变 $p_{\theta,h;q}(\tau\mid x)$。

## 2. 全专题统一符号

后续八章都使用同一套记号：

| 符号 | 含义 |
|---|---|
| $x\sim\mathcal D$ | 用户任务及环境初态 |
| $f_\theta$ | 参数冻结的基础模型；其条件策略记为 $\pi_\theta$ |
| $h=(h^C,h^W,h^K,h^M)$ | 可编辑 harness：context、workflow、code/tool、memory 四层 |
| $q=(V,\Pi,B,L)$ | 外置 control plane：verifier、permission、budget、append-only log |
| $s_t$ | runtime 在步骤 $t$ 的显式状态 |
| $c_t=C_{h^C}(s_t)$ | context compiler 产生的模型输入 |
| $a_t\sim\pi_\theta(\cdot\mid c_t)$ | 模型建议的动作；不是已授权副作用 |
| $u_t=G_{h^W,h^K}(s_t,a_t)$ | runtime 解析、校验后的可执行动作 |
| $o_{t+1}$ | 工具/环境返回的观察 |
| $\tau=(s_0,c_0,a_0,u_0,o_1,\ldots,s_T)$ | 完整可重放轨迹 |
| $z=A(\tau,V)$ | failure attribution：根因 surface、component 与证据 |
| $\rho_\phi(\delta\mid h,z)$ | 人工、LLM 或搜索算法诱导的 proposal distribution |
| $\delta$ | 对可编辑区域的 bounded edit；$h'=h\oplus\delta$ |
| $D_{\mathrm{mine}}$ | 暴露给 proposer、用于找失败的轨迹集 |
| $D_{\mathrm{in}}$ | 验证目标 failure 是否修复的 held-in 集 |
| $D_{\mathrm{ho}}$ | 对 proposer 隐藏、可反复用于 promotion 的回归集 |
| $D_{\mathrm{test}}$ | 冻结后只使用一次的最终报告集 |

外置 $q$ 不等于“绝对不可攻击”，而是说 proposer 的正常写权限不包含它；生产系统还要靠进程/容器隔离、只读挂载、签名 digest 和独立身份实现。

## 3. 三层搜索空间

本专题把 harness 搜索空间分为三层，而不是把所有修改都叫 prompt engineering：

```text
Layer C — context
  instruction / examples / retrieval / compression / memory read-write

Layer W — workflow
  branch / loop / retry / planner / verifier placement / subagent graph

Layer K — executable code
  tool implementation / middleware / parser / sandbox / state migration
```

形式化状态中单列 $h^M$，因为长期 memory 有独立生命周期；搜索风险分层时把 memory read/write policy 并入 Layer C 的信息管理面。三层包含关系近似为 $\mathcal H_C\subset\mathcal H_{C,W}\subset\mathcal H_{C,W,K}$。空间越大，潜在收益越高，但 credit assignment、搜索成本和权限风险也更高。因此每个 proposal 必须声明 edit surface、证据、预期修复、回归风险和预算变化。

## 4. 一套完整 control-loop

```text
                   frozen model f_theta
                           |
request -> runtime state -> context compiler -> model action
    |             ^                              |
    |             |                              v
    |       event reducer <- tool/result <- permission proxy
    |             |
    |          trace + artifacts
    |             v
    |      failure attribution
    |             v
    |      bounded proposal delta
    |             v
    |  static checks + sandbox execution
    |             v
    | held-in repair + hidden regression + cost/risk gate
    |             v
    +--- reject/audit       accept/version
                                  |
                           shadow -> canary -> active
                                  |
                              monitoring

external q: verifier / permissions / budgets / audit / final test
```

对应的更新算子是：

$$z_t=A(\tau_t,V),\qquad \delta_t\sim \rho_\phi(\delta\mid h_t,z_t),\qquad \tilde h_t=h_t\oplus\delta_t,$$

$$h_{t+1}=\begin{cases}\tilde h_t,&\mathrm{Gate}(h_t,\tilde h_t;q)=1,\\h_t,&\mathrm{otherwise}.\end{cases}$$

$A$ 做 failure attribution；$\rho_\phi$ 是人工、LLM proposer 或搜索算法诱导的 proposal distribution；`Gate` 在 proposer 外部。**提案、评估、部署是三个权限域，不应由同一个可编辑进程自我声明成功。**

## 5. 不是论文堆叠：每篇工作落在哪个环节

| Control-loop 环节 | 代表工作 | 本专题抽取的机制 |
|---|---|---|
| context 更新 | ACE、MCE | 增量 playbook；连 context-management skill 也进入搜索空间 |
| workflow/code 搜索 | ADAS、AFlow、Meta-Harness、AlphaEvolve | 代码化候选、archive、MCTS/evolution、Pareto 成本 |
| recursive improver | STOP | 让 improver 优化 improver；递归结构不保证单调提升 |
| attribution + versioning | AHE | component/experience/decision observability；延迟归因与 rollback |
| split-wise promotion | Self-Harness | held-in/hidden-held-out 都不退化，至少一边改善 |
| population search | DGM | 多谱系 archive、模型/benchmark transfer、objective hacking 风险 |
| updater/beneficiary 解耦 | Harness Updating Is Not Harness Benefit | 会写 update 与会激活/遵循 update 是两个能力 |
| harness/weight 联合更新 | SIA、Continual Harness | 双时间尺度、联合 Goodhart、rollout provenance |

原论文、模型、benchmark、数字与 caveat 见 [`papers.md`](papers.md)。博客是地图，不是数字的最终证据。

## 6. 与仓库其他专题的边界

| 专题 | 优化对象 | 与本专题的接口 |
|---|---|---|
| [`model-aware-data-curation/`](../model-aware-data-curation/README.md) | 训练/评估数据 $\mathcal B$ | failure attribution 决定是否补数据；harness digest 随样本 provenance 保存 |
| [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) | policy weights $\theta$ | harness 定义 rollout 分布、工具动作和 reward observation；RL 更新权重 |
| [`vllm-rollout/`](../vllm-rollout/README.md) | token generation serving | rollout engine 负责吞吐；runtime 负责状态、工具、副作用与终止 |
| [`ICL/`](../ICL/README.md) | 给定 context 后的预测变化 | ICL 研究 $\pi_\theta(\cdot\mid c)$；本专题研究 $c=C_h(s)$ 如何被构造 |
| `harness-engineering/` | 外部执行机制 $h$ | 端到端 orchestration、evaluation、versioning 与 deployment |

## 7. 阅读路线

1. [`01_harness_vs_model.md`](01_harness_vs_model.md)：frozen model 的非参数优化、诱导轨迹分布、updating 与 benefit。
2. [`02_agent_runtime_loop.md`](02_agent_runtime_loop.md)：状态机、工具事务、event sourcing、停止与恢复。
3. [`03_context_and_persistent_memory.md`](03_context_and_persistent_memory.md)：context compiler、预算选择、memory 写入与污染控制。
4. [`04_workflow_and_subagents.md`](04_workflow_and_subagents.md)：workflow graph、调度、join/cancel、并行成本和搜索。
5. [`05_harness_optimization.md`](05_harness_optimization.md)：context/workflow/code 三层搜索、Pareto archive 与 optimizer overfitting。
6. [`06_self_improving_harness.md`](06_self_improving_harness.md)：failure attribution、proposal contract、accept/reject/rollback 的完整闭环。
7. [`07_observability_evaluation_security.md`](07_observability_evaluation_security.md)：外置 verifier/permission、统计 gate、威胁模型与发布状态机。
8. [`08_harness_rl_and_weight_updates.md`](08_harness_rl_and_weight_updates.md)：与 post-training、replay buffer、rollout infra 的双时间尺度接口。
9. [`papers.md`](papers.md)：逐篇证据账本。
10. [`code/`](code/) + [`tests/`](tests/)：标准库实现与回归测试。

## 8. 每章阅读模板

每章都回答七个问题：

1. **State**：系统状态和不变量是什么？
2. **Objective**：优化什么，成本/风险如何进入？
3. **Action**：可编辑面和动作空间是什么？
4. **Evidence**：从哪些 trace 判断失败？
5. **Algorithm**：搜索或控制循环怎样运行？
6. **Failure modes**：哪里会过拟合、误归因、越权或失控？
7. **Engineering**：落到 schema、状态机、测试和部署怎样实现？

## 9. 最小可运行示例

```bash
python3 harness-engineering/code/demo.py
python3 -m unittest discover -s harness-engineering/tests -v
```

Demo 覆盖：context/workflow/tool/memory 四类 edit surface；带 evidence、hypothesis、expected-fix 和 at-risk 字段的 proposal manifest；Self-Harness 式 held-in/held-out gate；外置 permission/cost/risk/control-plane digest；parent/content lineage；rejected candidate 不污染 active registry。它仍是教学实现：生产协议还需要进程级隔离、统计置信区间、shadow/canary 和真实 side-effect reconciliation。

## 10. 核心判断

1. Harness 是可执行控制系统，不是 prompt 合集。
2. Frozen model 不意味着 frozen behavior；改变 $h$ 会改变信息、动作空间和轨迹分布。
3. 大搜索空间必须配更强 attribution、隔离、评估与成本核算。
4. `harness-updating` 与 `harness-benefit` 必须用交叉实验拆开。
5. hidden regression set 不是最终 test；重复查询会把它变成 validation signal。
6. verifier、permission、budget、tracer 和 model identity 必须位于 proposer 的写权限之外。
7. Harness update 与 weight update 互补，但联合优化会带来分布漂移和 coupled Goodhart。

## 11. 起点来源

本专题由 Lilian Weng 的综述 [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) 启动；所有重要实验结论回到原论文核查。

<!-- NAVIGATION -->
## 导航

- 开始阅读：[01 Harness vs Model](01_harness_vs_model.md)
- 证据账本：[papers.md](papers.md)
- 运行代码：[code/demo.py](code/demo.py) · [tests/test_harness_lab.py](tests/test_harness_lab.py)
