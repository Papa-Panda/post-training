# 03 — Context 与持久记忆：选择，不是堆积

## 元信息

- 核心论文：[ACE](https://arxiv.org/abs/2510.04618v3) · [MCE](https://arxiv.org/abs/2601.21557)
- 内容边界：[`ICL/`](../ICL/README.md) 研究 demonstrations 如何改变模型预测；本章研究 runtime 如何选择、压缩、更新和持久化上下文。

## 1. Context construction 是预算优化

历史、文件、工具结果和记忆条目组成候选集合 $I_t$。在 token 预算 $B$ 下选择 context：

$$C_h(s_t)=\arg\max_{S\subseteq I_t}\left[\sum_{i\in S}u(i\mid s_t)-\lambda\mathrm{Redundancy}(S)-\mu\mathrm{Staleness}(S)\right]\quad\text{s.t.}\quad\sum_{i\in S}\mathrm{tokens}(i)\le B.$$

$u(i\mid s_t)$ 不只是语义相似度，还应包含：当前步骤相关性、来源可信度、版本、依赖关系和失败风险。

简单 append-all 会导致：

- 长上下文中的相关信息稀释；
- 旧工具结果与新状态冲突；
- 反复摘要造成细节丢失；
- token 成本随 horizon 单调增长。

## 2. 三层状态，而不是一个 prompt

```text
working context        当前决策所需，短、可替换
checkpoint summary     阶段性状态、决策与未完成项
artifact / event store 完整文件、trace、diff、原始证据
```

读取策略是由上到下按需展开；写入策略是先保存原始 evidence，再生成带来源指针的摘要。摘要不能取代原始 artifact。

## 3. ACE：上下文是增量 playbook

[Agentic Context Engineering](https://arxiv.org/abs/2510.04618v3) 使用三角色循环：

1. **Generator**：在当前 playbook 下产生任务轨迹；
2. **Reflector**：从成功和失败中提炼经验；
3. **Curator**：把经验增量合并为带 identifier 的条目。

若 playbook 为 $P_t$、反思增量为 $\Delta_t$：

$$P_{t+1}=\mathrm{Merge}(P_t,\Delta_t),$$

而不是每轮把整个 $P_t$ 重新生成。增量 merge 可以降低 context collapse 和 brevity bias，并让每条规则有稳定 ID、来源和生命周期。

工程 schema：

```json
{
  "id": "tool-timeout-recovery",
  "text": "Timeout 后先检查副作用，再决定重试",
  "evidence": ["run-17:event-42"],
  "created_by": "reflector-v2",
  "status": "candidate|active|deprecated",
  "last_validated": "eval-v5"
}
```

## 4. MCE：连 context 管理机制也优化

[Meta Context Engineering](https://arxiv.org/abs/2601.21557) 把 context function 写成静态资源与动态算子的组合：

$$c_s(x)=F_s(x;\rho_s),$$

其中 $\rho_s$ 可包含 prompts、knowledge bases、代码库，$F_s$ 包含搜索、筛选、格式化和更新算子。内层优化具体 context，外层搜索管理机制/skill：

$$c_s^\star=\arg\max_{c_s}J_{\mathrm{train}}(c_s;s),\qquad s^\star=\arg\max_{s\in\mathcal S}J_{\mathrm{val}}(c_s^\star).$$

关键升级是：不再假设“Generator/Reflector/Curator”永远是最佳结构，而把如何管理 context 也变成 executable search space。

## 5. 文件系统为什么有效

文件系统提供模型已经熟悉的通用接口：路径、层级、grep、diff、append 和版本控制。它让：

- 大 artifact 留在 context 外；
- subagent 输出可被主 Agent 稍后读取；
- 中断后从文件恢复；
- 修改以 diff 审核；
- accepted/rejected 经验并存但不混淆 active state。

但文件不是自动正确的 memory。必须有 ownership、schema、TTL、dedup、source pointer 和读写权限。

## 6. 防止 memory 自我污染

写入长期记忆前应通过：

$$\mathrm{AcceptMemory}(m)=\mathrm{Grounded}(m)\land\mathrm{Relevant}(m)\land\neg\mathrm{Contradicted}(m)\land\mathrm{Authorized}(m).$$

至少保留：事实来源、适用范围、有效期、谁创建、谁验证。失败轨迹中可能包含错误推断或外部注入，不能未经验证升级为长期规则。

## 7. 与 ICL 的边界

[`ICL/`](../ICL/README.md) 关心给定 demonstrations 后 $p_\theta(y\mid x,C)$ 为什么改变；本章关心哪个 $C$ 被 runtime 放进窗口、哪些信息留在文件、什么时候更新。两者接口是 context builder，但研究问题不同。

<!-- NAVIGATION -->
## 导航

- 上一篇：[02 Agent Runtime Loop](02_agent_runtime_loop.md)
- 下一篇：[04 Workflow 与 Subagents](04_workflow_and_subagents.md)
- 回到：[专题 README](README.md)
