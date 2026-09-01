# 04 — Workflow 与 Subagents：把编排写成可搜索的图

## 元信息

- 核心论文：[Automated Design of Agentic Systems (ADAS)](https://arxiv.org/abs/2408.08435v2) · [AFlow](https://arxiv.org/abs/2410.10762v4)
- 内容类型：论文综合 + runtime 工程推演

## 1. Workflow 图

把 workflow 表示为有向图 $W=(V,E)$：

- 节点 $v\in V$ 是模型调用、工具、verifier、人工审批或 join；
- 边 $e\in E$ 是数据依赖和条件控制流；
- 节点状态必须持久化，避免整条工作流因中断重跑。

一个 workflow 的目标不只是准确率：

$$G(W)=\mathbb E[R(W,x)]-\lambda_1\mathbb E[\mathrm{tokens}]-\lambda_2\mathbb E[\mathrm{latency}]-\lambda_3\mathbb E[\mathrm{risk}].$$

若只优化 benchmark score，搜索容易无限增加调用数、换更强模型或把 verifier 信息泄露进 context。

## 2. Subagent 何时值得并行

将任务拆成子任务 $q_1,\ldots,q_k$，并行只在以下条件有意义：

- 子任务依赖弱，可独立执行；
- 输出有明确 schema 和 merge rule；
- 主 Agent 不需要把所有中间 token 留在 context；
- 失败可以单独重试或取消。

理想 wall time 下界近似：

$$T_{\mathrm{parallel}}\ge \max_i T(q_i)+T_{\mathrm{join}},$$

但总成本是：

$$K_{\mathrm{parallel}}=\sum_i K(q_i)+K_{\mathrm{coord}}.$$

因此并行降低 latency，不必然降低 token/GPU 成本；共享依赖和重复搜索还可能使总成本上升。

## 3. Process manager contract

父 Agent 需要最小 process manager：

```text
spawn(task, input_refs, budget) -> job_id
status(job_id)                  -> pending|running|done|failed
logs(job_id, cursor)            -> evidence refs
cancel(job_id)                  -> terminal state
join(job_ids, merge_schema)     -> structured result
```

子任务输入应通过文件/reference 传递，输出持久化。主 context 只保留 job ID、状态和摘要；完整日志按需读取。

## 4. 手工 workflow 与自动搜索

专家可以手工写 plan→execute→test→repair，但 workflow design space 很大。令搜索空间为 $\mathcal W$：

$$W^\star=\arg\max_{W\in\mathcal W}G(W;D_{\mathrm{search}}).$$

必须区分：

- `search set`：生成和比较候选 workflow；
- `validation set`：选超参/停止；
- `held-out test`：最终报告 generalization。

反复在同一 benchmark 上搜索，再把最高分当成未见性能，会产生 optimizer overfitting。

## 5. ADAS：Agent 设计本身成为优化对象

[ADAS](https://arxiv.org/abs/2408.08435v2) 将自动 Agent 设计拆成 search space、search algorithm 和 evaluation function。Meta Agent Search 维护 archive，循环：

```text
archive -> propose agent description -> implement in code
        -> self-refine / repair -> evaluate -> archive successful design
```

这里 code 使新的 prompt、角色、控制流和工具组合可以一起变化，而不是只搜索一段文本。

## 6. AFlow：用 MCTS 搜索 workflow graph

[AFlow](https://arxiv.org/abs/2410.10762v4) 把 LLM 调用动作表示为节点，以代码定义边，使用 MCTS 风格循环：

1. 根据历史 score 与探索概率选择父 workflow；
2. 让 LLM 基于评估反馈扩展/修改；
3. 执行新 workflow；
4. 将结果回传到搜索树；
5. 达到预算或 top-$k$ 平均分平台后停止。

可抽象为 UCB 选择：

$$W_t=\arg\max_W\left[\hat G(W)+c\sqrt{\frac{\log N}{n_W}}\right].$$

论文报告其在多个 QA、代码和数学 benchmark 上优于手工 workflow 与 ADAS；精确实验口径与成本 caveat 见 [`papers.md`](papers.md)，不把单一设置外推为通用优势。

## 7. Merge 是多 Agent 最容易被忽略的步骤

若多个 agent 给出候选 $y_1,\ldots,y_k$，不能简单拼接。Merge 应显式处理：

- provenance：每个 claim 来自哪条 trace；
- conflict：不一致结论如何裁决；
- coverage：是否遗漏必要子问题；
- confidence：缺证据时保留不确定性；
- budget：是否值得追加验证。

一个 verifier-aware merge 可写为：

$$y^\star=\arg\max_{y\in\mathrm{Merge}(y_{1:k})}\left[V(y)-\alpha\mathrm{Conflict}(y)-\beta\mathrm{Unsupported}(y)\right].$$

## 8. 生产护栏

- 每个 node 固定输入/输出 schema；
- side-effecting node 支持幂等与人工审批；
- fan-out、深度和总调用数有上限；
- 搜索阶段与部署阶段模型/预算差异必须记录；
- workflow archive 保存失败版本，避免反复探索同一坏设计。

<!-- NAVIGATION -->
## 导航

- 上一篇：[03 Context 与持久记忆](03_context_and_persistent_memory.md)
- 下一篇：[05 Harness Optimization](05_harness_optimization.md)
- 回到：[专题 README](README.md)
