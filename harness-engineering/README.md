# Harness Engineering

> **优化模型如何运行，而不只优化模型权重。** 本专题把 context、tools、workflow、subagents、memory、evaluation、permissions 和版本化发布统一成可测量、可回归的 Agent Runtime。

## 一句话定位

给定冻结模型参数 $\theta$ 与 harness $h$，Agent 的轨迹分布和最终效用同时由二者决定：

$$\tau\sim p_{\theta,h}(\tau\mid x),\qquad J(\theta,h)=\mathbb E_{x\sim\mathcal D,\tau\sim p_{\theta,h}}[R(\tau,x)].$$

Post-training 主要优化 $\theta$；Harness Engineering 主要优化 $h$：

$$h^\star=\arg\max_{h\in\mathcal H}J(\theta,h).$$

这里的 $h$ 不只是 system prompt，而是：context policy、tool schema 与实现、middleware、workflow、subagent topology、persistent memory、permission boundary、trace 和 deployment policy。

## 系统总图

```text
                         immutable boundary
                  ┌──────────────────────────┐
                  │ verifier / permissions   │
                  │ held-out eval / budgets  │
                  └─────────────┬────────────┘
                                │ score + audit
                                ▼
request -> context builder -> model -> tool/runtime -> state/artifacts
              ▲                  │             │
              │                  └── trace ────┘
              │                        │
              └── memory/select ◄──────┘
                                       │
                        failure mining / proposal
                                       │
                        sandbox evaluation + regression
                                       │
                          accept -> versioned harness
```

## 与仓库其他专题的边界

| 专题 | 优化对象 | 主要状态 | 本专题只复用、不重复 |
|---|---|---|---|
| [`model-aware-data-curation/`](../model-aware-data-curation/README.md) | 选/造哪些训练数据 | 数据池、梯度/ICL probes | 数据价值、覆盖、生成闭环 |
| [`grpo-vs-ppo/`](../grpo-vs-ppo/README.md) | 如何更新 policy 权重 | advantage、KL、critic/group reward | PPO/GRPO 数学与训练权衡 |
| [`vllm-rollout/`](../vllm-rollout/README.md) | 如何高吞吐生成 rollout | KV cache、batch、GPU 队列 | rollout serving 与压测 |
| [`ICL/`](../ICL/README.md) | 冻结权重时 prompt 内学习 | demonstrations、induction/circuit | ICL 理论与 trajectory prompting |
| `harness-engineering/` | 如何组织模型外执行机制 | runtime state、context、tools、files、traces | 端到端 orchestration、评估和自改进 |

边界原则：本专题讨论“怎样运行、检查和改进 Agent 系统”；不重写 RL objective、不重复 vLLM 性能调优，也不把所有 context 现象重新解释一遍。

## 阅读路线：数学 → Runtime → 自改进

1. [`01_harness_vs_model.md`](01_harness_vs_model.md) — 区分 core intelligence、harness benefit 与 harness updating。
2. [`02_agent_runtime_loop.md`](02_agent_runtime_loop.md) — 把 Agent 写成可恢复、可审计的状态转移系统。
3. [`03_context_and_persistent_memory.md`](03_context_and_persistent_memory.md) — context selection、文件持久化、ACE/MCE。
4. [`04_workflow_and_subagents.md`](04_workflow_and_subagents.md) — workflow graph、并行 subagents、ADAS/AFlow。
5. [`05_harness_optimization.md`](05_harness_optimization.md) — 从 prompt 到 harness code 的搜索空间，Meta-Harness 与演化搜索。
6. [`06_self_improving_harness.md`](06_self_improving_harness.md) — STOP、Self-Harness、AHE、DGM 的 propose→evaluate→accept。
7. [`07_observability_evaluation_security.md`](07_observability_evaluation_security.md) — held-in/held-out、不可编辑边界、reward hacking 与回滚。
8. [`08_harness_rl_and_weight_updates.md`](08_harness_rl_and_weight_updates.md) — harness 优化、RL 与权重更新的双时间尺度关系。
9. [`papers.md`](papers.md) — 原论文证据账本与博客/工程推演边界。
10. [`code/`](code/) + [`tests/`](tests/) — 无重型依赖的版本化 harness 演化最小实现。

## 十天路线图

### Phase 1 — 定义与 Runtime（Day 1–3）

- Day 1：读 `01`，建立 $J(\theta,h)$ 和 model/harness 两个优化轴；
- Day 2：读 `02`，实现 observe→act→persist→verify 状态机；
- Day 3：读 `03`，区分 token context、persistent memory 与 retrieval policy。

产出：能解释“同一个模型为什么换 harness 后能力不同”，并画出状态、工具副作用、artifact 和恢复点。

### Phase 2 — Workflow 与搜索（Day 4–6）

- Day 4：读 `04`，把工作流表示成图，明确 subagent isolation 和 join semantics；
- Day 5：读 `05`，比较 random/evolution/MCTS/LLM-proposer 搜索；
- Day 6：跑最小 demo，观察候选 harness 如何被测试和版本化。

产出：一个显式 search space、固定 evaluator、预算约束下的 candidate archive。

### Phase 3 — 自改进与安全边界（Day 7–10）

- Day 7–8：读 `06`，从 failure pattern 生成 bounded edits；
- Day 9：读 `07`，实现 held-in 修复、held-out 无回归、immutable surfaces；
- Day 10：读 `08`，决定何时改 harness、何时改权重、何时只补数据。

产出：一个可审计的 `trace → failure → proposal → sandbox eval → accept/reject → rollback` 闭环。

## 最小可运行示例

```bash
python3 harness-engineering/code/demo.py
python3 -m unittest discover -s harness-engineering/tests -v
```

Demo 明确演示：

- 有用且无回归的 edit 被接受并生成新版本；
- 破坏 held-out 安全行为的 edit 被拒绝；
- 尝试修改 verifier/permission boundary 的 proposal 不会被实例化；
- 被拒候选不能污染 active harness；
- 每个 accepted version 都有 parent digest 和内容 digest。

## 核心判断

1. Harness 是可执行系统，不是提示词合集；code 是统一表达 context、tools、control flow 和 memory 的语言。
2. `harness-updating`（会提出好修改）与 `harness-benefit`（基础模型会正确利用新 harness）是不同能力。
3. 自改进不是“允许 Agent 随便改自己”，而是**限制可编辑面、保留外部 verifier、用 held-out 回归控制发布**。
4. 当前最可靠的 self-improvement 发生在可自动验证的任务；开放研究、长期维护和价值判断仍受 evaluator 限制。
5. Harness 与权重更新互补：前者快、可回滚、非参数化；后者慢、可能内化通用行为，但训练成本和风险更高。

## 起点来源

本专题由 Lilian Weng 的综述 [Harness Engineering for Self-Improvement](https://lilianweng.github.io/posts/2026-07-04-harness/) 启动，但重要结论回到原论文核查；证据和外推边界见 [`papers.md`](papers.md)。

<!-- NAVIGATION -->
## 导航

- 开始阅读：[01 Harness vs Model](01_harness_vs_model.md)
- 证据账本：[papers.md](papers.md)
- 运行代码：[code/demo.py](code/demo.py) · [tests/test_harness_lab.py](tests/test_harness_lab.py)
