# 07 — Observability、Evaluation 与 Security

## 元信息

- 内容类型：生产系统护栏与评估设计
- 论文接口：[Self-Harness](https://arxiv.org/abs/2606.09498) · [AHE](https://arxiv.org/abs/2604.25850) · [Meta-Harness](https://arxiv.org/abs/2603.28052)

## 1. 先冻结测量系统

如果被优化的 harness 能修改 verifier、模型、预算或测试数据，任何提升都无法归因。将系统拆为：

```text
editable workspace              immutable control plane
---------------------------     -----------------------------
system prompt                   held-out tasks and labels
context policy                  verifier implementation
skills                          permission policy
workflow / middleware           model identity
subagent configuration          budget accounting
long-term memory policy         tracer and audit log
```

工程上 immutable 不等于只写一句“不要修改”，而是文件权限、不同进程/容器、签名 digest、只读挂载和独立 deployment identity。

## 2. Held-in 与 held-out

- $D_{\mathrm{in}}$：验证已观察到的 failure 是否被修复；
- $D_{\mathrm{out}}$：检查未知任务和应保留行为是否退化；
- $D_{\mathrm{test}}$：搜索结束后只使用一次，报告最终泛化。

严格 gate：

$$J_{\mathrm{in}}(h')>J_{\mathrm{in}}(h),\qquad J_{\mathrm{out}}(h')\ge J_{\mathrm{out}}(h).$$

多指标下还需：

$$K(h')\le B_K,\qquad \mathrm{Violation}(h')=0,\qquad \mathrm{Risk}(h')\le B_R.$$

允许统计噪声时，应使用重复 runs、置信区间或 non-inferiority margin，而不是单次恰好高一分就发布。

## 3. 三类 observability

### Component

每个可编辑组件有 owner、schema、版本、依赖和测试。失败必须映射到具体 surface，而不是给整套 system prompt 加一句泛化规则。

### Experience

原始 trace 很长，采用层级索引：

```text
benchmark overview
  -> failure cluster
     -> per-task diagnosis
        -> raw event / artifact / tool output
```

摘要必须能回到证据；没有 source pointer 的“经验”不能驱动自动 edit。

### Decision

每次 edit 都是一个可证伪假设：

$$H_\delta:\quad \Delta J_{\mathrm{target}}>0,\quad \Delta J_{\mathrm{retain}}\ge0,\quad \Delta K\le B.$$

下一轮评估应直接验证该预测，并记录不符合预期的原因。

## 4. Reward hacking taxonomy

常见攻击面：

1. **Verifier tampering**：关闭、放宽或绕过测试；
2. **Test leakage**：把 held-out answer 写进 prompt/memory；
3. **Budget hacking**：换更强模型、增加 reasoning tokens、无限重试；
4. **Metric gaming**：只修可见测试，破坏真实行为；
5. **Trace suppression**：删除失败日志，只保留成功样本；
6. **Permission escalation**：扩大工具权限以完成任务。

对应防线：只读 verifier、split isolation、固定模型和预算、trace append-only、外部权限代理、人工审计高风险 diff。

## 5. 长期效用不等于短期任务分

代码 Agent 完成当前 issue，不代表维护了 repo 的长期健康。更完整目标：

$$J_{\mathrm{long}}=J_{\mathrm{task}}-\lambda_1C_{\mathrm{maintenance}}-\lambda_2C_{\mathrm{migration}}-\lambda_3C_{\mathrm{debug}}-\lambda_4R_{\mathrm{security}}.$$

这些成本常在 rollout 结束后才出现，sandbox verifier 很难即时测量。应使用静态分析、兼容性测试、ownership checks、延迟回归和人工 code review 补足。

## 6. Failure 是一等数据

如果 archive 只保存成功候选，搜索会重复失败路径且高估系统可靠性。对 rejected proposal 保存：

- parent/candidate digest；
- 修改 diff 和 proposal rationale；
- held-in/out 结果；
- permission/cost violation；
- 已知失败条件和可复用反例。

但 rejected candidate 绝不能成为 active state；审计历史与部署状态要分离。

## 7. 发布与回滚

推荐状态机：

```text
DRAFT -> SANDBOXED -> VALIDATED -> SHADOW -> CANARY -> ACTIVE
   \         \           \          \          \
    ---------------------------> REJECTED / ROLLED_BACK
```

每个版本绑定：harness digest、model ID、tool versions、evaluator digest、数据 split version、预算和迁移说明。Rollback 恢复的不只是 prompt，还包括 memory schema、workflow state 和 tool config。

## 8. 最小检查表

- [ ] proposer 不能写 verifier/permission/tracer；
- [ ] held-out 标签对 proposer 不可见；
- [ ] 模型、token/tool/time budget 固定；
- [ ] 每个 gain 可追到 evidence 与 diff；
- [ ] accepted/rejected/archive/active 状态分离；
- [ ] side effects 有审批、幂等和 rollback；
- [ ] 新版本先 shadow/canary，再扩大流量。

<!-- NAVIGATION -->
## 导航

- 上一篇：[06 Self-Improving Harness](06_self_improving_harness.md)
- 下一篇：[08 Harness、RL 与权重更新](08_harness_rl_and_weight_updates.md)
- 回到：[专题 README](README.md)
