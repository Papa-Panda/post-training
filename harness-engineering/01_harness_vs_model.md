# 01 — Harness vs Model：两个优化轴

## 元信息

- 内容类型：统一问题定义，不对应单篇论文
- 起点综述：[Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/)
- 关键辨析：[Harness Updating Is Not Harness Benefit](https://arxiv.org/abs/2605.30621)

## 1. Harness 是什么

基础模型给出条件分布 $p_\theta$；harness 决定模型实际看到什么、能调用什么、何时停止、怎样验证，以及哪些状态跨轮持久化。把它写成参数化系统：

$$h=(C,T,W,M,A,P,E),$$

其中：

- $C$：context construction / compression；
- $T$：tool schema 与实现；
- $W$：workflow、loop 和停止条件；
- $M$：memory / artifact lifecycle；
- $A$：subagent topology 与后台任务；
- $P$：permission / side-effect policy；
- $E$：evaluation、tracing 与 deployment gates。

这比早期的“LLM + tools + memory”多出 control plane：状态、权限、评估、版本与恢复。

## 2. 同一模型，不同轨迹分布

对输入 $x$，runtime 在时刻 $t$ 的状态为 $s_t$，harness 把状态变成上下文并约束动作：

$$c_t=C_h(s_t),\qquad a_t\sim p_\theta(a_t\mid c_t),\qquad s_{t+1}=F_h(s_t,a_t,o_{t+1}).$$

因此系统效用不是只由 $	heta$ 决定：

$$J(\theta,h)=\mathbb E_{x\sim\mathcal D,\tau\sim p_{\theta,h}}[R(\tau,x)-\lambda K(\tau)],$$

其中 $R$ 是任务收益，$K$ 是 token、工具调用、wall time 或风险成本。换 harness 会改变 context、行动空间、观察和停止规则，从而改变 $p_{\theta,h}(\tau\mid x)$。

## 3. Core intelligence 与 harness benefit

固定 harness 比模型：

$$\theta^\star=\arg\max_\theta J(\theta,h).$$

固定模型比 harness：

$$h^\star=\arg\max_h J(\theta,h).$$

两者不是替代关系。更强模型可能减少脆弱的手工规则，但外部工具、权限、持久状态、预算和验证接口不会自动消失。反过来，复杂 harness 也不能让缺乏基本推理和长程指令遵循的模型稳定使用它。

## 4. Updating 不等于 Benefit

对当前 harness $H_t$，模型可能具备两种不同能力：

- **Harness updating**：evolver $e$ 根据轨迹数据提出有效修改；
- **Harness benefit**：solver $f$ 在部署时正确激活并遵循新 harness，从而得到收益。

论文 [`Harness Updating Is Not Harness Benefit`](https://arxiv.org/abs/2605.30621) 把两种角色交叉组合：

$$A_t=(f,H_t),\qquad D_t=\{(x,\tau_{t,x},y_{t,x}):x\in X_t\},$$

$$\Delta H_t=e(H_{t-1},D_t),\qquad H_t=\mathrm{Apply}(H_{t-1},\Delta H_t).$$

先定义固定 solver/evolver pair 的增益：

$$\Delta(f,e)=J_X(f,H_T^{(f,e)})-J_X(f,H_0).$$

再用 anchor solvers $F^\star$ 与 anchor evolvers $E^\star$ 解耦两个轴：

$$\Delta_{\mathrm{update}}(e)=\frac1{|F^\star|}\sum_{f\in F^\star}\Delta(f,e),$$

$$\Delta_{\mathrm{benefit}}(f)=\max_{e\in E^\star}\Delta(f,e).$$

其实验覆盖 SWE-bench Verified、MCP-Atlas 与 SkillsBench。evolver 的最优/最差差距在任一 benchmark 上最多只有 $3.1$ 个百分点，且没有一个 evolver 在三项任务都最好；solver 的 harness benefit 则明显非单调：弱模型可能不会激活 skill，中等模型仍有提升空间，强模型遵循更稳但 ceiling 更高、余量更小。比如 Qwen3-235B 的 skill-loading rate 为 $0.961$，接近 Opus 4.6 的 $0.957$，但 harness-following rate 仅 $0.350$，低于后者的 $0.757$。

因此，会写一份看起来合理的 skill，不意味着部署模型会在正确时机调用它、遵循它并完成长任务。本专题不把“能提出修改”误写成“已经实现递归智能增长”。论文主实验是 in-situ evaluation：任务先用 $H_{t-1}$ 评分，再将其 trace 用于 $H_t$；它避免同一任务从自己的证据中获益，但不等价于独立 held-out promotion set。

## 5. 为什么近期路径更像系统优化

直接让模型改写权重需要训练数据、optimizer、算力、稳定性和模型发布；修改 harness 通常：

- 不需要反向传播；
- 可以在 sandbox 中快速评估；
- 修改是代码 diff，容易审计和回滚；
- 可以针对某个模型、任务或失败模式局部适配。

所以近期 self-improvement 更可操作的形式是：

$$h_t\xrightarrow{\text{trace failures}}\mathcal H_t\xrightarrow{\text{external eval}}h_{t+1},$$

而不是无边界地让模型修改自身运行环境。

## 6. 适用边界

Harness 收益大，通常需要：

1. 任务能通过 unit tests、execution reward 或结构化 rubric 快速评价；
2. 失败能追踪到 context/tool/workflow 等组件；
3. 修改面可隔离，verifier 和权限在循环外；
4. 基础模型能可靠遵循新协议。

如果 evaluator 模糊、反馈延迟、长期维护成本不可见，短期 benchmark 提升不一定等价于真实系统改进。

<!-- NAVIGATION -->
## 导航

- 上一篇：[专题 README](README.md)
- 下一篇：[02 Agent Runtime Loop](02_agent_runtime_loop.md)
- 证据：[papers.md](papers.md)
