# 02 — Agent Runtime Loop：状态、动作与恢复

## 元信息

- 内容类型：系统抽象与工程设计
- 相关专题：[`vllm-rollout/`](../vllm-rollout/README.md) 负责模型生成服务；本章负责生成之外的控制流、工具副作用和持久状态。

## 1. 从聊天循环到状态机

一个最小 Agent loop 不是“反复调用 LLM”，而是带类型的状态转移：

$$s_t=(g,b_t,c_t,m_t,a_t,e_t),$$

- $g$：不可随意漂移的用户目标与约束；
- $b_t$：剩余 token、时间、工具调用和风险预算；
- $c_t$：本轮送入模型的工作上下文；
- $m_t$：持久 memory / artifact references；
- $a_t$：已启动的子任务和后台 job；
- $e_t$：事件日志、工具结果与 verifier 证据。

模型提出动作 $u_t$，环境返回观察 $o_{t+1}$：

$$u_t\sim p_\theta(\cdot\mid C_h(s_t)),\qquad s_{t+1}=F_h(s_t,u_t,o_{t+1}).$$

$F_h$ 才是 harness 的核心：它处理权限、重试、超时、幂等、日志、预算和终止，而不是把所有责任交给自然语言输出。

## 2. 五阶段执行循环

```text
observe -> construct context -> decide -> execute -> persist + verify
    ^                                                     |
    └---------------- retry / replan / finish ------------┘
```

1. **Observe**：读取任务、当前版本、未完成 jobs、最近证据；
2. **Construct**：按预算选择 context，不把全部历史重新塞入；
3. **Decide**：模型输出结构化 action 或计划；
4. **Execute**：runtime 校验权限后调用工具；
5. **Persist + verify**：保存 artifact/event，运行局部 verifier，再决定继续或结束。

## 3. 工具调用是带副作用的事务

工具调用建议显式建模：

$$u_t=(\text{name},\text{args},\text{idempotency key},\text{risk class}),$$

执行状态至少有：

```text
PROPOSED -> AUTHORIZED -> RUNNING -> SUCCEEDED
                           |
                           ├-> RETRYABLE_FAILED
                           └-> TERMINAL_FAILED
```

关键约束：

- 读操作和外部写操作分级；
- 可重试调用必须有幂等键，避免重复发送或重复创建；
- timeout 不等于失败，需先检查是否已经产生副作用；
- tool output 是数据，不应成为修改系统目标或权限的指令；
- 每次执行记录输入摘要、输出摘要、时间、版本与证据位置。

## 4. Event sourcing 与 checkpoint

不只保存“当前状态”，还应追加不可变事件：

$$s_t=\mathrm{Fold}(s_0,e_1,e_2,\ldots,e_t).$$

事件可包括：`MODEL_DECISION`、`TOOL_STARTED`、`TOOL_RESULT`、`ARTIFACT_WRITTEN`、`VERIFIER_RESULT`、`BUDGET_CHANGED`、`USER_APPROVAL`。好处是：

- 崩溃后能从 checkpoint + event tail 恢复；
- 可重放失败轨迹；
- 可比较两个 harness 在同一事件前缀上的分叉；
- 评估提升能追溯到具体版本和修改。

## 5. 长任务的停止条件

Agent 不能只用“模型说完成了”作为终止条件。设任务 verifier 为 $V$，预算为 $B$：

$$\mathrm{stop}(s_t)=\mathbf 1[V(s_t)=\mathrm{pass}]\lor\mathbf 1[K(s_t)>B]\lor\mathbf 1[\mathrm{blocked}(s_t)].$$

三个终态要分开：

- **完成**：证据满足 success contract；
- **预算终止**：没有宣称成功，但保留中间 artifact；
- **阻塞**：缺权限、数据或人类决策，明确下一步需要什么。

## 6. 可观测性 schema

每次 rollout 至少记录：

```json
{
  "run_id": "...",
  "harness_digest": "...",
  "model_id": "...",
  "task_id": "...",
  "events": [],
  "artifacts": [],
  "tool_cost": 0,
  "token_cost": 0,
  "verifier": {"passed": false, "evidence": []},
  "terminal_state": "completed|budget|blocked|failed"
}
```

这份 trace 是后续 failure mining 和 harness evolution 的训练数据；没有版本和事件，任何“新 harness 提升了”都难以归因。

## 7. 与 rollout infra 的接口

[`vllm-rollout/`](../vllm-rollout/README.md) 优化 TTFT、TPOT、KV cache 和批量生成；runtime 只通过稳定接口请求 generation。二者的边界是：

- rollout engine 负责“高效生成 token”；
- harness runtime 负责“为什么现在生成、给什么 context、生成后调用什么工具、是否继续”。

<!-- NAVIGATION -->
## 导航

- 上一篇：[01 Harness vs Model](01_harness_vs_model.md)
- 下一篇：[03 Context 与持久记忆](03_context_and_persistent_memory.md)
- 回到：[专题 README](README.md)
