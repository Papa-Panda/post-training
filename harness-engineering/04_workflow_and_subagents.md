# 04 — Workflow 与 Subagents：把编排写成可验证的图

## 元信息

- 核心论文：[ADAS](https://arxiv.org/abs/2408.08435v2) · [AFlow](https://arxiv.org/abs/2410.10762v4)
- 内容类型：论文综合 + 分布式 runtime 工程
- 本章目标：把 planner、tool、verifier 和 subagent 的组合写成有依赖、预算、join 和取消语义的 executable graph。

## 1. State：Workflow 是持久化图

令 workflow 为有向图：

$$W_h=(V_h,E_h,\Gamma_h),$$

- $V_h$：model、tool、verifier、human approval、fork/join 等节点；
- $E_h$：数据依赖和控制依赖；
- $\Gamma_h$：每个节点的 input/output schema、retry、timeout、permission 和 budget。

运行状态不是当前 prompt，而是：

$$w_t=(\sigma_t,\kappa_t,\omega_t),$$

其中 $\sigma_t(v)$ 是节点状态，$\kappa_t$ 是已提交 outputs/artifact refs，$\omega_t$ 是 leases、重试计数和预算预留。

节点状态机：

```text
PENDING -> READY -> LEASED -> RUNNING -> COMMITTED
              |         |          |
              |         |          +-> RETRYABLE_FAILED
              |         +-> LEASE_EXPIRED -> RECONCILE
              +-> SKIPPED
RUNNING -> CANCEL_REQUESTED -> DRAINING -> CANCELLED
```

只有 `COMMITTED` output 才能满足强依赖；日志里出现结果文本，不等于已提交。

## 2. Objective：质量、延迟、成本与风险

对任务 $x$，workflow 产生轨迹 $\tau(W,x)$。目标是向量：

$$\mathbf y(W)=(R(W),-K(W),-L(W),-Q(W)).$$

标量化版本：

$$G_{\boldsymbol\lambda}(W)=\mathbb E[R]-\lambda_K\mathbb E[K]-\lambda_L\mathbb E[L]-\lambda_Q\mathbb E[Q].$$

但生产选择最好保留 Pareto frontier，而不是只用一组随意权重。对两个 workflow：

$$W_a\succ W_b\iff R_a\ge R_b,\ K_a\le K_b,\ L_a\le L_b,\ Q_a\le Q_b,$$

且至少一项严格更好。若候选只靠增加调用数获得更高分，应明确落在不同成本点，而不是宣称无条件更优。

## 3. Workflow node contract

每个节点定义：

```text
NodeSpec:
  id, kind, version
  input_schema, output_schema
  required_artifacts
  permission_capabilities
  token/tool/time budget
  retry_policy, timeout_policy
  idempotency_policy
  verifier
```

执行前必须满足：

$$\mathrm{Ready}(v)=\mathbf1[\forall u:(u,v)\in E,\sigma(u)=\mathrm{COMMITTED}]\mathbf1[\mathrm{SchemaOK}]\mathbf1[\mathrm{Authorized}].$$

输出提交时要做 compare-and-swap 或版本检查，防止两个并发 worker 覆盖相同 artifact。

## 4. Subagent 的收益条件

把任务拆为 $q_1,\ldots,q_k$。并行 wall time 理想下界：

$$T_{\mathrm{parallel}}\ge\max_iT(q_i)+T_{\mathrm{join}}.$$

总成本：

$$K_{\mathrm{parallel}}=\sum_iK(q_i)+K_{\mathrm{coord}}+K_{\mathrm{dup}}.$$

并行有价值需要：

- 子任务依赖弱；
- 输入可以被完整封装，不靠父 agent 隐含上下文；
- 输出有 schema/provenance；
- 失败能单独重试/取消；
- join 不是把全部长文本重新塞回主 context。

若子任务共享大量未知前提，fan-out 会重复搜索并放大错误。可用粗略净收益条件：

$$\Delta T>0\quad\land\quad \Delta R-\lambda_K\Delta K-\lambda_Q\Delta Q>0.$$

## 5. Spawn、join 与 cancellation semantics

最小 process manager：

```text
spawn(task_spec, input_refs, budget, parent) -> job_id
observe(job_id) -> state, heartbeat, artifact_refs
cancel(job_id, reason) -> CANCEL_REQUESTED
join(job_ids, merge_spec) -> merged_artifact | partial | failed
```

### Join policy

- `ALL`：所有必需子任务成功；适合互补分解；
- `ANY`：首个满足 verifier 的结果；适合 speculative execution；
- `QUORUM(k)`：至少 $k$ 个独立证据；适合事实核验；
- `BEST`：外部 scorer 选择；适合候选生成；
- `REDUCE`：按 associative reducer 合并；适合统计量。

Join 必须定义失败容忍：一个 optional agent 超时是否阻塞？partial result 是否可接受？不同结论由谁裁决？

### Structured merge

若输出为 $y_{1:k}$，merge 不应简单拼接：

$$y^\star=\arg\max_{y\in\mathcal M(y_{1:k})}\left[\mathrm{Score}(y)-\alpha\mathrm{Conflict}(y)-\beta\mathrm{Unsupported}(y)-\gamma K(y)\right].$$

每个 claim 保留 source agent、evidence pointer 和 confidence。Judge 也可能错，因此高风险结论需要 deterministic verifier 或人工审批。

## 6. Workflow 搜索空间

常见 rewrite operators：

| 类别 | 操作 |
|---|---|
| Topology | insert/delete node、fork、join、loop、fallback |
| Routing | 修改 branch predicate、model/tool selector |
| Budget | 节点 token、timeout、fan-out、retry cap |
| Verification | verifier placement、early exit、self-check |
| State | checkpoint、artifact handoff、memory scope |
| Recovery | compensation、retry、rollback、human escalation |

为了保证候选可执行，搜索不直接生成任意代码，而是从 typed grammar 采样：

$$W'\sim \rho_\phi(W'\mid W,z),\qquad W'\in\mathcal G_{\mathrm{typed}}.$$

静态 validator 检查：无悬空依赖、无不可达 terminal、所有 side-effect path 都经过 permission node、循环有 budget/stop、join schema 一致。

## 7. ADAS：Agent design 作为代码候选

[ADAS](https://arxiv.org/abs/2408.08435v2) 将自动设计拆成 search space、search algorithm 和 evaluation function。Meta Agent Search 维护 archive：

```text
sample prior designs
  -> propose new agent description
  -> implement in code
  -> repair parse/runtime errors
  -> evaluate
  -> add successful design to archive
```

重要启示不是“让另一个 LLM 写代码”本身，而是：prompt、角色分工、控制流和 aggregation 可以作为一个可执行候选整体评价。风险是 archive 中 benchmark-specific tricks 被误当成一般结构。

## 8. AFlow：MCTS 风格 workflow search

[AFlow](https://arxiv.org/abs/2410.10762v4) 把 LLM actions 表示为节点、代码表示边，并用 MCTS 风格迭代。抽象的 UCB parent selection：

$$W_t=\arg\max_W\left[\hat G(W)+c\sqrt{\frac{\log N}{n_W}}\right].$$

其中 $N$ 是父节点总访问数，$n_W$ 是候选 $W$ 的访问数；具体实现的 score normalization 与 exploration schedule 依论文代码而定。

循环：

```text
select parent with exploitation/exploration
expand by LLM-generated workflow edit
execute candidate on search tasks
backpropagate score to search tree
stop on budget or top-k plateau
freeze selected workflow and evaluate held-out
```

论文数字与成本口径见 [`papers.md`](papers.md)。特别注意摘要的 4.55% cost 只支持 parity；超过比较对象的配置成本更高，不能把它写成普遍的“不到 5% 成本即可超越”。

## 9. 调度与资源隔离

Workflow scheduler 需要同时考虑 precedence 与资源：

$$\min \mathrm{makespan}(W)\quad\text{s.t.}\quad \sum_{v\in\mathrm{running}}g_v\le G,\ \sum c_v\le B.$$

其中 $g_v$ 是 GPU/worker 占用，$c_v$ 是 token/tool budget。实际策略可使用：

- critical-path priority；
- memory-aware batching；
- per-tenant quota；
- backpressure；
- bounded speculative execution；
- deadline-aware cancellation。

Rollout infra 的 throughput optimization 不能改变任务语义。若 scheduler 在高负载下截断 context 或减少 attempts，必须作为 evaluation condition 记录。

## 10. Failure attribution for workflow

失败可能来自：

- node 本身输出错误；
- router 选错分支；
- 正确节点未运行；
- join 丢失 evidence；
- retry 造成重复副作用；
- stale output 被下游使用；
- timeout 太短；
- fan-out 太大导致预算耗尽。

对节点 $v$ 的 observational blame：

$$\mathrm{Blame}(v)=\sum_{\tau}\mathbf1[v\in\tau]\,w(\tau)\,\mathbf1[V(\tau)=0],$$

只能用于排序，不是因果结论。更强方法是 replay intervention：保持其他版本和 seed 分布不变，只替换 $v$ 或其 routing rule，比较配对 outcome。

## 11. Failure modes

1. **Graph explosion**：搜索不断插入 agent/judge，成本失控；
2. **Judge loop**：多个模型相互点评但无外部 verifier；
3. **Zombie child**：父任务取消后子任务继续写外部状态；
4. **Merge hallucination**：aggregator 添加输入中不存在的 claim；
5. **Hidden serialization**：表面并行，关键锁/模型队列使延迟不降；
6. **Correlated agents**：相同模型/提示产生一致错误，quorum 不等于独立证据；
7. **Retry storm**：下游故障时所有节点同步重试；
8. **Search/test leakage**：同一 task 反复改图再当最终结果；
9. **Unbounded loop**：没有进度度量与 stop budget。

## 12. Engineering checklist

- [ ] node I/O schema、permission、budget、retry 明确；
- [ ] join policy 和 partial-result 语义明确；
- [ ] 子任务用 artifact refs 传递，不复制全部日志；
- [ ] cancel 能向下传播并 reconcile side effects；
- [ ] graph 静态检查覆盖 side-effect path 和循环上界；
- [ ] 并行收益同时报告 latency 与总成本；
- [ ] attribution 用 replay/ablation 验证；
- [ ] workflow search、validation、final test 分离。

<!-- NAVIGATION -->
## 导航

- 上一篇：[03 Context 与持久记忆](03_context_and_persistent_memory.md)
- 下一篇：[05 Harness Optimization](05_harness_optimization.md)
- 回到：[专题 README](README.md)
