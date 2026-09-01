# 03 — Context 与持久记忆：Compiler、Budget 与 Provenance

## 元信息

- 核心论文：[ACE](https://arxiv.org/abs/2510.04618v3) · [MCE](https://arxiv.org/abs/2601.21557)
- 内容边界：[`ICL/`](../ICL/README.md) 研究给定 demonstrations 后模型为何改变预测；本章研究 runtime 如何选择、编译、更新和持久化 context。
- 本章目标：把“多塞一些历史”改写成一个有预算、版本、来源与写入门控的状态管理问题。

## 1. State：四种信息寿命

在 runtime 状态 $s_t$ 中，信息按寿命和权威性分层：

```text
request facts       当前请求原文与不可漂移约束
working context     本次模型调用所需的短上下文
checkpoint state    阶段状态、决策、未完成项、artifact refs
evidence store      原始 trace、文件、diff、工具结果、verifier evidence
```

长期 memory 不是第五种“更长 prompt”，而是 evidence store 上的一个可检索、带 provenance 的视图。Context compiler：

$$c_t=C_{h^C}(s_t;B_t^{\mathrm{ctx}}).$$

它输出的是一次调用的有序 token 序列，不等于持久状态本身。

## 2. Context selection 是约束优化

候选信息项为 $I_t=\{i_1,\ldots,i_n\}$。每项有 token 成本 $\ell_i$、相关性 $r_i$、可信度 $g_i$、新鲜度 $f_i$、依赖集合 $P_i$。选择集合 $S$：

$$S_t^\star=\arg\max_{S\subseteq I_t}\left[\sum_{i\in S}u(i\mid s_t)-\lambda_DD(S)-\lambda_CC(S)\right],$$

$$\text{s.t.}\quad \sum_{i\in S}\ell_i\le B_t^{\mathrm{ctx}},\qquad i\in S\Rightarrow P_i\subseteq S.$$

$D(S)$ 惩罚重复，$C(S)$ 惩罚冲突/过期组合。这个问题类似带依赖的 knapsack；实践中常用分层 quota、MMR、greedy coverage 和 hard priority，而不是求精确最优。

一项合理的 utility 可以写成：

$$u(i\mid s_t)=\alpha r_i+\beta g_i+\gamma f_i+\eta\mathrm{CoverageGain}(i)-\lambda_R\mathrm{Risk}(i).$$

语义相似度只是 $r_i$ 的一个特征；用户约束、当前 tool schema、失败证据通常比“看起来相关的旧对话”优先级更高。

## 3. Context compilation pipeline

```text
collect references
  -> authorize reads
  -> resolve versions
  -> rank/select under budget
  -> detect contradiction
  -> compress with source pointers
  -> order sections
  -> render tool/model schema
  -> hash compiled context
```

每步都有可测试 contract：

- `collect` 只能读取允许 scope；
- `resolve` 必须固定 artifact/version，避免 mid-run 漂移；
- `select` 保留 mandatory constraints；
- `compress` 不能丢 provenance；
- `render` 必须满足模型/tool parser schema；
- `hash` 让 rollout 可重现。

推荐 context layout：

```text
[goal + success contract]
[hard constraints + permissions]
[current workflow node]
[minimum state snapshot]
[selected evidence with refs]
[tool schemas]
[requested output schema]
```

把工具结果或网页内容放在与系统约束相同层级，会放大 prompt injection；data plane 与 instruction plane 应显式分区。

## 4. Compression 的信息损失

设原始证据 $E$、摘要 $z=S(E)$、当前决策目标 $Y$。理想摘要希望保留足够信息：

$$I(z;Y)\approx I(E;Y),\qquad |z|\ll|E|.$$

但 $Y$ 会随 workflow 节点变化，固定摘要不可能对所有未来问题都充分。工程上因此采用：

1. 原始 evidence 不删除；
2. 摘要带 source pointers；
3. 针对当前 query 动态展开；
4. 重要数值/权限/版本不做模糊压缩；
5. 摘要版本和生成模型写入 metadata。

反复“摘要的摘要”会累积误差。设每轮保真率为 $1-\epsilon$，经过 $k$ 次独立压缩的粗略上界是 $(1-\epsilon)^k$；这不是现实误差模型，但直观说明为什么必须从原证据重建，而不是无限递归摘要。

## 5. ACE：增量 playbook 而非整段重写

[Agentic Context Engineering](https://arxiv.org/abs/2510.04618v3) 采用 Generator→Reflector→Curator：

$$\tau_t\sim p_{\theta,P_t}(\tau\mid x_t),\qquad \Delta_t=R(\tau_t,V),$$

$$P_{t+1}=\mathrm{Merge}(P_t,\Delta_t).$$

关键不是角色名称，而是增量、结构化、可追踪的 context update：稳定 ID、成功/失败 evidence、去重、冲突合并和条目生命周期。相对整段重写，它减少 context collapse 和 brevity bias。

工程条目：

```json
{
  "id": "retry-after-timeout",
  "claim": "Timeout 后先查询副作用状态，再决定是否重试",
  "scope": ["external-write"],
  "evidence": ["run-17:event-42"],
  "confidence": 0.93,
  "status": "candidate",
  "supersedes": [],
  "validated_on": ["eval-v5"]
}
```

ACE 报告的 agent/finance 改善与成本口径见 [`papers.md`](papers.md)；它支持“增量 context evolution 有效”，不证明所有任务都应使用相同三角色结构。

## 6. MCE：优化 context-management program

[MCE](https://arxiv.org/abs/2601.21557) 把某个 skill $s$ 的 context function 写为：

$$c_s(x)=F_s(x;\mathcal R_s),$$

其中 $\mathcal R_s$ 是 prompts、knowledge bases、code 等资源，$F_s$ 是 retrieve/filter/format/update 算子。形成双层问题：

$$c_s^\star=\arg\max_{c_s}J_{\mathrm{train}}(c_s;s),$$

$$s^\star=\arg\max_{s\in\mathcal S}J_{\mathrm{val}}(c_s^\star;s).$$

内层更新内容，外层更新“怎样管理内容”的 executable skill。它把 context engineering 从静态文本提升为 program synthesis，但同时产生更强 leakage 风险：外层若看到 validation labels，就会把评测逻辑编码进 retrieval/update policy。

## 7. Persistent memory 的读写语义

Memory item $m$ 至少包含：

$$m=(\text{claim},\text{scope},\text{source},\text{time},\text{confidence},\text{status},\text{acl}).$$

写入 gate：

$$\mathrm{Write}(m)=\mathrm{Grounded}(m)\land\mathrm{Relevant}(m)\land\mathrm{Authorized}(m)\land\neg\mathrm{Contradicted}(m).$$

这还不够。不同类型要有不同策略：

| 类型 | 例子 | 更新规则 |
|---|---|---|
| Immutable fact | 已确认接口版本 | 新 evidence 不能静默覆盖，必须 supersede |
| Preference | 用户输出格式偏好 | 保留来源和适用 scope |
| Procedure | timeout recovery | 先 candidate，回归通过后 active |
| Ephemeral state | 当前 job 进度 | TTL/完成后归档 |
| Hypothesis | 可能的 failure root cause | 明确未验证，不升级为事实 |

读取时同时检查 ACL、scope、freshness 和 conflicts；“被存过”不等于“当前可用”。

## 8. Memory lifecycle

```text
OBSERVED -> CANDIDATE -> VALIDATED -> ACTIVE
                 |           |          |
                 +-> REJECTED           +-> STALE -> SUPERSEDED
```

- `OBSERVED`：原始 trace 中出现；
- `CANDIDATE`：被提炼成结构化条目；
- `VALIDATED`：有独立证据支持；
- `ACTIVE`：context compiler 可选择；
- `STALE`：版本/时间条件改变；
- `SUPERSEDED`：被新条目替代但保留 lineage。

删除 active rule 前要检查依赖；否则 workflow 可能引用失效 memory ID。

## 9. Failure attribution for context

Context 失败不要统一写成“模型没注意”。至少区分：

1. **Missing**：必要 evidence 未被检索；
2. **Buried**：已检索但顺序/长度使其失效；
3. **Contradictory**：新旧版本同时出现；
4. **Untrusted**：外部文本被当成指令；
5. **Overcompressed**：摘要丢失数值或条件；
6. **Stale**：旧状态覆盖当前环境；
7. **Mis-scoped**：某任务经验被泛化到不适用任务；
8. **Write pollution**：错误推断进入长期 memory。

可定义 attribution score：

$$A_C(i)=\Pr(Y=1\mid i\ \mathrm{included})-\Pr(Y=1\mid i\ \mathrm{excluded}),$$

但单条 observational trace 不能识别因果。更可信的方法是 counterfactual replay：固定 model seed/预算，只改变 context item；对随机模型则做多次配对 rollout。

## 10. 与 ICL 的边界

[`ICL/`](../ICL/README.md) 研究给定 $c$ 时 $\pi_\theta(y\mid x,c)$ 的机制；本章研究 $c=C_{h^C}(s)$ 的构造与生命周期。接口是：

```text
persistent state -> context compiler -> ICL behavior -> action trace
       ^                                      |
       +----------- validated update --------+
```

不能用 ICL 的“模型会从示例学习”替代 memory correctness；错误示例也会被学习。

## 11. Failure modes

- append-all 导致 token 成本和干扰单调增长；
- 只用 embedding similarity，遗漏权限、依赖和版本；
- summary 无 source pointer，无法验证；
- held-out failure 被写入 active memory，造成 test leakage；
- 多 agent 同时写同一条目，产生 lost update；
- 规则永不过期，旧工具 schema 长期污染；
- proposer 可写 system constraints，memory 变成权限逃逸通道；
- 为单一 benchmark 写出的 heuristics 被误称为通用知识。

## 12. Engineering checklist

- [ ] working context、checkpoint、evidence store 分层；
- [ ] 每次 compiled context 有 digest 和 source refs；
- [ ] mandatory constraints 不参与普通 relevance 竞争；
- [ ] memory 有 schema、ACL、scope、TTL、status 和 supersedes；
- [ ] 原始 evidence 保留，摘要可重建；
- [ ] memory edit 与 verifier/permission 隔离；
- [ ] context ablation 使用配对 rollout，而不是凭直觉归因；
- [ ] $D_{\mathrm{ho}}$ 与 $D_{\mathrm{test}}$ 不进入可写 memory。

<!-- NAVIGATION -->
## 导航

- 上一篇：[02 Agent Runtime Loop](02_agent_runtime_loop.md)
- 下一篇：[04 Workflow 与 Subagents](04_workflow_and_subagents.md)
- 回到：[专题 README](README.md)
