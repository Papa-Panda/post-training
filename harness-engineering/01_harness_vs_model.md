# 01 — Harness vs Model：Frozen Model 的非参数优化

## 元信息

- 起点综述：[Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)
- 关键辨析：[Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621)
- 本章目标：解释为什么固定参数并不等于固定系统行为，以及 harness gain 应怎样识别而不是凭 leaderboard 猜测。

## 1. State：模型只是闭环中的一个算子

按照专题统一记号，可编辑 harness 为：

$$h=(h^C,h^W,h^K,h^M),$$

分别表示 context policy、workflow、可执行代码/工具、memory policy。外置 control plane 为：

$$q=(V,\Pi,B,L),$$

其中 $V$ 是 verifier，$\Pi$ 是权限策略，$B$ 是资源预算，$L$ 是 append-only audit log。模型参数 $\theta$ 固定，但 runtime 每一步仍由 harness 决定：

$$c_t=C_{h^C}(s_t),\qquad a_t\sim\pi_\theta(\cdot\mid c_t),$$

$$u_t=G_{h^W,h^K}(s_t,a_t;\Pi),\qquad s_{t+1}=F_{h^M}(s_t,u_t,o_{t+1}).$$

这里 $a_t$ 是模型建议，$u_t$ 才是经过 schema、permission 和 workflow 约束后的执行动作。相同的 $a_t$ 在不同 harness 下可能被拒绝、改写、拆分、重试或送入不同工具。

## 2. Objective：优化的是诱导轨迹分布

对任务 $x\sim\mathcal D$，完整轨迹记为：

$$\tau=(s_0,c_0,a_0,u_0,o_1,\ldots,s_T).$$

按 README 的统一约定，本章固定外置 control plane $q$，并把 $p_{\theta,h;q}$ 简写为 $p_{\theta,h}$。模型和 harness 共同诱导 $p_{\theta,h}(\tau\mid x)$。系统目标通常是多目标：

$$J_{\boldsymbol\lambda}(\theta,h)=\mathbb E[R(\tau,x)-\lambda_KK(\tau)-\lambda_L\mathrm{Lat}(\tau)-\lambda_QQ(\tau)].$$

- $R$：任务正确性或 verifier reward；
- $K$：tokens、模型调用、工具/GPU 资源；
- $\mathrm{Lat}$：wall-clock latency；
- $Q$：权限、不可逆副作用、维护和回归风险。

固定 harness 的 post-training 是：

$$\theta^\star(h)=\arg\max_\theta J(\theta,h).$$

固定模型的 harness optimization 是：

$$h^\star(\theta)=\arg\max_{h\in\mathcal H}J(\theta,h).$$

二者一般不交换：

$$\max_\theta\max_hJ(\theta,h)=\max_h\max_\theta J(\theta,h),$$

但按不同顺序训练得到的局部解、数据分布、成本和稳定性可以完全不同。数学上的最大值交换不意味着实际非凸优化路径相同。

## 3. 为什么 frozen model 仍能显著变化

把一次 action 的概率写开：

$$p_{\theta,h}(u_t\mid s_t)=\sum_{a_t}\pi_\theta(a_t\mid C_{h^C}(s_t))\,G_{h^W,h^K}(u_t\mid s_t,a_t;\Pi).$$

Harness 有四个杠杆：

1. **Information**：$C_{h^C}$ 改变模型看到的证据与顺序；
2. **Action**：tool schema 和 parser 改变可表示动作；
3. **Dynamics**：workflow/retry/subagents 改变未来状态转移；
4. **Selection**：verifier、stop rule 和 archive 改变哪些轨迹被保留。

所以 harness gain 不是“凭空增加参数知识”，而是减少 observation aliasing、搜索错误、接口错误和执行损耗。可将性能差距概念性拆成：

$$J^\star-J(\theta,h)=\epsilon_{\mathrm{model}}+\epsilon_{\mathrm{information}}+\epsilon_{\mathrm{planning}}+\epsilon_{\mathrm{execution}}+\epsilon_{\mathrm{verification}}.$$

这不是可唯一识别的统计分解，而是 root-cause taxonomy：只有后三四项有证据时，才优先改 harness；若错误来自模型无法表示/遵循基本策略，则应考虑权重更新。

## 4. 三层 action space

### 4.1 Context edit

$$\delta^C:\ C_h(s)\mapsto C_{h'}(s).$$

例：增加 retrieval、改变 compression、插入结构化 playbook。它不改控制图，回滚快，但容易发生 context dilution、stale memory 和 prompt overfitting。

### 4.2 Workflow edit

$$\delta^W:\ W_h=(V_h,E_h)\mapsto W_{h'}.$$

例：增加 planner/verifier 节点、重试分支、并行 subagent、join rule。它能改变搜索深度和错误恢复，但也会增加调用成本与非确定性。

### 4.3 Code/tool edit

$$\delta^K:\ (G_h,F_h,\text{middleware})\mapsto(G_{h'},F_{h'},\text{middleware}').$$

例：修改 parser、工具实现、状态迁移和 sandbox。表达力最大，也最可能越权、泄漏 evaluator 或破坏状态兼容性。

搜索空间满足近似包含关系：

$$\mathcal H_C\subset\mathcal H_{C,W}\subset\mathcal H_{C,W,K}.$$

表达力增加时，不应只扩大 proposer；还要同时加强 type checks、sandbox、test split、权限和 rollout budget。

## 5. Updating 不等于 Benefit

设 solver 为 $f$、evolver 为 $e$，harness update 为：

$$A_t=(f,H_t),\qquad D_t=\{(x,\tau_{t,x},y_{t,x}):x\in X_t\},$$

$$\Delta H_t=e(H_{t-1},D_t),\qquad H_t=\mathrm{Apply}(H_{t-1},\Delta H_t).$$

固定 pair 的演化增益：

$$\Delta(f,e)=J_X(f,H_T^{(f,e)})-J_X(f,H_0).$$

[`Harness Updating Is Not Harness Benefit`](https://arxiv.org/abs/2605.30621) 用 anchor solvers $F^\star$ 与 anchor evolvers $E^\star$ 定义：

$$\Delta_{\mathrm{update}}(e)=\frac1{|F^\star|}\sum_{f\in F^\star}\Delta(f,e),$$

$$\Delta_{\mathrm{benefit}}(f)=\max_{e\in E^\star}\Delta(f,e).$$

第一项问“谁写出的 update 对一组 solver 平均有用”；第二项问“这个 solver 能从某个强 evolver 的 update 中得到多少”。两者之间至少隔着三个机制：

- **activation**：solver 是否检索/调用了新 skill；
- **adherence**：调用后是否按协议执行；
- **execution**：外部工具和环境是否把计划正确落地。

论文在 SWE-bench Verified、MCP-Atlas、SkillsBench 上发现，best/worst evolver 的最大差距只有 $3.1$ 个百分点，且没有一个 evolver三项都最好；solver benefit 则非单调。Qwen3-235B 的 skill-loading rate 为 $0.961$，接近 Opus 4.6 的 $0.957$，但 harness-following rate 为 $0.350$，低于后者的 $0.757$。会写规则不等于会用规则。

## 6. 识别 harness 的真实因果增益

至少需要 $2\times2$ factorial：

| Arm | Solver | Harness | 用途 |
|---|---|---|---|
| A | $f_0$ | $h_0$ | baseline |
| B | $f_0$ | $h_1$ | 纯 harness effect |
| C | $f_1$ | $h_0$ | 纯 model effect |
| D | $f_1$ | $h_1$ | 联合效果 |

定义：

$$\Delta_h(f)=J(f,h_1)-J(f,h_0),\qquad \Delta_f(h)=J(f_1,h)-J(f_0,h),$$

$$I_{f,h}=J(f_1,h_1)-J(f_1,h_0)-J(f_0,h_1)+J(f_0,h_0).$$

$I_{f,h}$ 是 interaction。若 $I_{f,h}>0$，强模型更能利用新 harness；若为负，harness 可能只为旧模型补洞，或新模型与旧约束冲突。

更严格的研究还要固定 decoding、token/tool budget、模型版本、环境镜像和 verifier。否则把更长 context、更高调用次数或更强 judge 混进 $h_1$，就不能把收益归因于结构设计。

## 7. Algorithm：非参数优化循环

```text
input: frozen model theta, active harness h, external control plane q
repeat:
    traces = rollout(theta, h, D_mine)
    failures = attribute(traces, verifier=q.V)
    proposals = propose_bounded_edits(h, failures)
    for delta in proposals:
        reject if delta touches q or violates schema
        candidate = sandbox_materialize(h, delta)
        metrics = evaluate(candidate, D_in, D_ho, fixed budgets)
        if repair && no_regression && permission_ok && budget_ok:
            version(candidate, parent=h)
    h = select_from_accepted_frontier()
until budget exhausted or validation plateaus
freeze h; evaluate once on D_test
```

注意：`propose_bounded_edits` 可以由同一个 frozen model完成，也可以是另一模型、搜索算法或人工；优化的对象仍是外部 $h$，不是 $\theta$。

## 8. Failure modes

1. **Capability laundering**：换更强模型却声称是 harness gain；
2. **Budget laundering**：多十倍 tokens/tools，却只报 accuracy；
3. **Verifier leakage**：候选读取 tests/labels，学会过关而非完成任务；
4. **Search overfitting**：评估太多候选，挑中噪声最大者；
5. **Activation failure**：skill 写对了，但 solver 不检索；
6. **Adherence failure**：solver 读到规则但执行中偏离；
7. **Maintenance debt**：短期 pass rate 提升，长期代码复杂度和迁移成本上升；
8. **Negative transfer**：为某 benchmark 添加的规则破坏其他任务。

## 9. Engineering checklist

- [ ] model ID、decoding、tool set 和预算写入每条 trace；
- [ ] proposal 声明 context/workflow/code 哪一层被修改；
- [ ] update quality 与 solver benefit 分开测；
- [ ] quality、cost、latency、risk 同时报告；
- [ ] search、hidden regression、final test 严格分离；
- [ ] verifier/permission/tracer 不在 proposer 的写权限内；
- [ ] accepted version 有 parent digest、schema version 和 rollback path。

<!-- NAVIGATION -->
## 导航

- 上一篇：[专题 README](README.md)
- 下一篇：[02 Agent Runtime Loop](02_agent_runtime_loop.md)
- 证据：[papers.md](papers.md)
