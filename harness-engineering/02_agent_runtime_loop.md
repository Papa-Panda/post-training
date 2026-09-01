# 02 — Agent Runtime Loop：状态、动作、副作用与恢复

## 元信息

- 内容类型：系统语义与生产实现
- 相关专题：[`vllm-rollout/`](../vllm-rollout/README.md) 负责模型生成服务；本章负责 generation 外围的控制流、工具副作用和持久状态。
- 本章目标：给出可以实现、重放和验证的 Agent transition system，而不是一句 `while not done: call_llm()`。

## 1. State：显式状态而不是隐含聊天历史

使用专题统一记号，把 runtime 状态展开为：

$$s_t=(g,w_t,b_t,r_t,m_t,j_t,e_t),$$

- $g$：用户目标、success contract 与不可漂移约束；
- $w_t$：workflow 节点、依赖和 continuation；
- $b_t$：剩余 token、时间、工具、金钱和风险预算；
- $r_t$：当前资源与外部对象的已知版本；
- $m_t$：memory/artifact references，不是把全部内容复制进 prompt；
- $j_t$：subagent、异步工具和后台 job 状态；
- $e_t$：append-only events 与 verifier evidence。

Context compiler 和模型产生建议动作：

$$c_t=C_{h^C}(s_t),\qquad a_t\sim\pi_\theta(\cdot\mid c_t).$$

Runtime 再把建议动作解析为 typed command：

$$u_t=G_{h^W,h^K}(s_t,a_t;\Pi,B).$$

环境返回 observation 后，event reducer 更新状态：

$$s_{t+1}=F_h(s_t,u_t,o_{t+1}).$$

关键语义：**模型输出不是事实，也不是已授权操作；它只是 command proposal。** 权限、参数 schema、幂等、预算和 precondition 都在执行前检查。

## 2. Runtime 的六阶段循环

```text
OBSERVE -> COMPILE -> PROPOSE -> AUTHORIZE -> EXECUTE -> COMMIT/VERIFY
    ^                                                       |
    +-------------- RETRY / REPLAN / STOP -----------------+
```

1. **Observe**：读取任务、active harness/version、未完成 jobs、资源版本；
2. **Compile**：按预算选择 context，加入 schema、约束和 evidence pointers；
3. **Propose**：模型返回结构化 action/plan；
4. **Authorize**：验证权限、参数、precondition、budget 和风险等级；
5. **Execute**：调用工具或 subagent，记录开始/结束事件；
6. **Commit/Verify**：持久化结果、更新状态、运行 verifier、选择继续或终止。

每一阶段都有失败语义。比如 context compile 失败不应伪装成模型推理失败；authorize 拒绝也不应自动重试同一动作。

## 3. Command 与 side effect 的类型系统

定义 command envelope：

$$u_t=(n_t,\xi_t,\iota_t,\chi_t,p_t,d_t),$$

- $n_t$：工具名和版本；
- $\xi_t$：通过 schema 校验的参数；
- $\iota_t$：idempotency key；
- $\chi_t$：risk class；
- $p_t$：preconditions，例如目标对象版本；
- $d_t$：deadline/lease。

工具按副作用分层：

| 类别 | 示例 | 默认策略 |
|---|---|---|
| Pure read | 查静态配置、读取文件 | 可自动执行，仍记录 provenance |
| Expensive read | 大检索、长 rollout | 预算 gate、可取消 |
| Reversible write | 建草稿、写 sandbox 文件 | 自动或低风险审批；必须可 rollback |
| External send | 发消息、提交表单 | 明确授权、去重、发送后确认 |
| Irreversible/high-risk | 删除、交易、权限扩大 | 人工审批或禁止自动执行 |

Permission 不应由模型在自然语言里自我声明。令主体 $z$ 对资源 $r$ 的能力集合为 $\mathrm{Cap}(z,r)$，执行条件是：

$$\mathrm{Allow}(u_t)=\mathbf1[\mathrm{cap}(u_t)\in\mathrm{Cap}(z,r)]\mathbf1[\mathrm{Pre}(u_t,s_t)]\mathbf1[\mathrm{Cost}(u_t)\le b_t].$$

## 4. 工具事务：timeout 不是 rollback

建议状态机：

```text
PROPOSED -> AUTHORIZED -> STARTED -> OBSERVED_SUCCEEDED -> COMMITTED
                 |           |              |
                 |           |              +-> VERIFY_FAILED
                 |           +-> UNKNOWN_OUTCOME -> RECONCILE
                 +-> DENIED
STARTED -> OBSERVED_FAILED -> RETRYABLE | TERMINAL
```

最危险的是 `UNKNOWN_OUTCOME`：客户端 timeout 时，服务端可能已经完成写入。不能直接重试；先用 idempotency key 或业务对象状态 reconcile。

对外部写操作，理想协议是：

1. 生成稳定 idempotency key；
2. 写入 `TOOL_STARTED` 事件；
3. 发起操作；
4. 查询最终状态；
5. 只有在确认未发生时才 retry；
6. 写入 `TOOL_COMMITTED` 或 `TOOL_UNKNOWN`。

Exactly-once 通常不是 runtime 单方面能保证的；更现实的是 at-least-once transport + idempotent application，或 at-most-once + 显式人工 reconciliation。

## 5. Event sourcing：状态是事件折叠

使用 immutable event log：

$$s_t=\mathrm{Fold}(s_0,e_{1:t}).$$

最小 event schema：

```json
{
  "run_id": "...",
  "sequence": 42,
  "event_type": "TOOL_RESULT",
  "harness_digest": "...",
  "model_id": "...",
  "workflow_node": "execute-tests",
  "command_id": "...",
  "input_ref": "artifact://...",
  "output_ref": "artifact://...",
  "cost_delta": {"tokens": 0, "tool_calls": 1},
  "timestamp": "..."
}
```

事件记录 references 和 digest，不必把敏感原文复制到每个 log。事件序号必须单调；state reducer 应确定性且可版本化。

### 为什么同时需要 checkpoint

完整重放成本随 horizon 增长。每 $k$ 个事件生成 checkpoint：

$$\hat s_{mk}=\mathrm{Checkpoint}(e_{1:mk}),\qquad s_t=\mathrm{Fold}(\hat s_{mk},e_{mk+1:t}).$$

Checkpoint 包含 reducer version 和输入 event digest；否则 schema 迁移后无法证明重放一致。

## 6. Crash consistency 与 recovery

崩溃恢复算法：

```text
load latest valid checkpoint
verify checkpoint digest and reducer version
replay later events in sequence order
for every STARTED command without terminal event:
    query tool using idempotency key
    mark COMMITTED, FAILED, or UNKNOWN
resume only nodes whose dependencies are satisfied
never infer success from missing logs
```

恢复的核心 invariant：

$$\forall u,\\quad \#\mathrm{CommittedSideEffect}(u)\le1,$$

前提是下游工具支持相同 idempotency key；如果不支持，runtime 只能提供 reconciliation 和人工决策，不能虚构 exactly-once。

## 7. Budget 与停止语义

预算是向量而非单标量：

$$b_t=(b_t^{\mathrm{tok}},b_t^{\mathrm{tool}},b_t^{\mathrm{wall}},b_t^{\mathrm{money}},b_t^{\mathrm{risk}}).$$

每步消耗向量为 $\Delta b_t^{\mathrm{use}}$：

$$b_{t+1}=b_t-\Delta b_t^{\mathrm{use}}.$$

其中任何受保护维度越界都必须阻止执行，而不是执行后再记录。终止谓词：

$$\mathrm{Stop}(s_t)=V_{\mathrm{done}}(s_t)\lor\mathrm{BudgetExhausted}(b_t)\lor\mathrm{Blocked}(s_t)\lor\mathrm{Unsafe}(s_t).$$

终态必须区分：

- `SUCCEEDED`：success contract 有独立证据；
- `PARTIAL`：有可复用中间 artifact，但未满足成功条件；
- `BLOCKED`：缺权限、输入或人类判断；
- `BUDGET_EXHAUSTED`：资源耗尽，不冒充成功；
- `FAILED`：已知不可恢复失败；
- `UNKNOWN`：外部副作用结果无法确认。

## 8. Concurrency：lease、join 与取消

并发 job $j$ 需要 owner、lease、heartbeat 和 terminal state。令 dependency graph 为 $W=(V,E)$，节点只有在所有强依赖提交后才能启动：

$$\mathrm{Runnable}(v)=\mathbf1[\forall u:(u,v)\in E,\ \mathrm{state}(u)=\mathrm{COMMITTED}].$$

取消不是简单删除：

```text
CANCEL_REQUESTED -> cancel children -> reconcile side effects
                 -> persist partial artifacts -> CANCELLED
```

父任务结束时不能遗留无 owner 的 side effects。对无法取消的外部调用，父任务应标为 `DRAINING` 直到结果被 reconcile。

## 9. Verifier-based completion

模型自报“完成”只是一条 observation。成功条件应由外部 $V$ 给出：

$$V(\tau,x)=(v_{\mathrm{task}},v_{\mathrm{schema}},v_{\mathrm{sideeffect}},v_{\mathrm{policy}}).$$

全部硬约束通过才能标记 success；软指标进入 score/cost，不得掩盖 hard failure。比如测试通过但向错误目标发送消息，任务仍失败。

## 10. 与 rollout serving 的接口

Runtime 向 generation service 提交：

```text
(model_id, decoding_config, context_digest, max_tokens,
 request_priority, stop_sequences, trace_parent)
```

Serving 返回 token stream、finish reason、usage 和 request ID。边界如下：

- rollout engine 负责 batching、KV cache、prefill/decode 调度和硬件吞吐；
- harness runtime 负责何时请求、context 是什么、生成后执行什么、预算与终止；
- evaluator 记录两侧版本，避免把 serving 配置变化误判为 harness gain。

高吞吐不是正确性：若 rollout 被 dynamic batching 改变随机性，需要 seed/decoding config 和重复试验。延迟优化也不能绕过 permission 或 verifier。

## 11. Failure modes

1. **Implicit state**：关键信息只在 prompt，崩溃后无法恢复；
2. **Duplicate side effect**：timeout 后盲重试；
3. **Lost update**：并发节点覆盖相同 artifact；
4. **Zombie job**：父任务结束，子任务继续产生外部写入；
5. **False completion**：模型说 done，verifier 未通过；
6. **Schema drift**：tool/reducer 版本变化，旧事件无法重放；
7. **Prompt injection**：tool output 被当作权限指令；
8. **Budget afterthought**：先执行昂贵调用，再发现超预算；
9. **Untraceable generation**：缺 model/harness/context digest，结果不可归因。

## 12. Engineering checklist

- [ ] state、command、event、terminal status 都有版本化 schema；
- [ ] 模型 action 与 authorized command 分离；
- [ ] side-effecting tools 有 idempotency/reconciliation；
- [ ] checkpoint 可由 event tail 验证和恢复；
- [ ] budget 在执行前原子扣减或预留；
- [ ] verifier 独立于模型自评；
- [ ] subagent/job 有 owner、lease、cancel 和 join；
- [ ] rollout request 保存 model、decoding、context 与 harness digest。

<!-- NAVIGATION -->
## 导航

- 上一篇：[01 Harness vs Model](01_harness_vs_model.md)
- 下一篇：[03 Context 与持久记忆](03_context_and_persistent_memory.md)
- 回到：[专题 README](README.md)
