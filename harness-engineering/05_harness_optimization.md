# 05 — Harness Optimization：从 Prompt 到可执行搜索空间

## 元信息

- 核心论文：[Meta-Harness](https://arxiv.org/abs/2603.28052) · [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- 前序方法：[ADAS](https://arxiv.org/abs/2408.08435v2) · [AFlow](https://arxiv.org/abs/2410.10762v4)

## 1. 优化对象逐层扩大

```text
instruction prompt
  -> structured context policy
  -> workflow graph
  -> full harness code
  -> optimizer that proposes harness code
```

设 harness 源码和配置的可编辑区域为 $\Omega$，候选 edit 为 $\delta\in\Omega$：

$$h'=\mathrm{Apply}(h,\delta),\qquad \delta^\star=\arg\max_{\delta\in\Omega}\hat J(h';D)-\lambda K(h').$$

因为 harness 中包含离散控制流、工具和文件结构，通常不能对 $h$ 直接求梯度；但候选可以执行、测试和比较，因此适合黑盒、贝叶斯、MCTS 或 evolutionary search。

## 2. Code 为什么是统一表示

代码能够同时表达：

- prompt 和 context formatting；
- if/loop/retry/timeout；
- tool schema、middleware 与 permission check；
- subagent spawn/join；
- memory read/write；
- verifier 与 rollout logging。

所以 code-level search 比 prompt search 更广，但风险也更大：候选可能修改 evaluator、提高预算、泄漏答案或破坏环境。

## 3. Meta-Harness：端到端搜索 Harness

[Meta-Harness](https://arxiv.org/abs/2603.28052) 固定基础模型，优化 executable harness：

$$H^\star=\arg\max_H\mathbb E_{x\sim\mathcal X,\tau\sim p_M(H,x)}[r(\tau,x)].$$

其 proposer 是 coding agent：读取先前候选的代码、score 和 trajectories，提出新 harness，在任务上执行，再把合格候选保留在 archive / Pareto frontier。

Pareto 不是装饰。实际目标至少包含任务质量、token、latency 和风险：

$$h_a\succ h_b\iff J_a\ge J_b,\ K_a\le K_b,\ \mathrm{Risk}_a\le\mathrm{Risk}_b,$$

并且至少一项严格更优。只报告最高任务分会隐藏通过增加调用和 context 获得的收益。

## 4. Evolutionary search

当搜索空间离散、不可微但容易评价时，可维护种群 $\mathcal P_t$：

$$h_p\sim \mathrm{Select}(\mathcal P_t),\quad h_c\sim q_M(\cdot\mid h_p,\text{feedback}),\quad \mathcal P_{t+1}=\mathrm{Keep}(\mathcal P_t\cup\{h_c\}).$$

设计点包括：

- **parent selection**：平衡高分 exploitation 与低 offspring 数的 exploration；
- **mutation**：限定 edit surface，优先小而可解释的 diff；
- **novelty**：拒绝与 archive 过近的候选，避免 diversity collapse；
- **elitism**：保留强候选，但不能让一个局部模式垄断种群；
- **budget**：比较单位 compute 的收益而非裸分。

## 5. AlphaEvolve：程序与 meta-prompt 共演化

[AlphaEvolve](https://arxiv.org/abs/2506.13131) 使用冻结的 LLM coding agents 生成程序 diff，并以自动 evaluator 保留高 fitness 子代。其可迁移的 harness 设计思想是：

- parent program、结果和 instructions 一起进入 proposer context；
- 可编辑代码块显式标记；
- solution code 与 meta-prompt 可以共同演化；
- archive 保持多个候选，而不是每轮只覆盖一个 active 版本。

它最适用于 fitness 清晰、可快速执行的程序搜索。把同样机制搬到开放研究时，evaluator 质量会成为主要瓶颈。

## 6. Search 过拟合

设候选数为 $N$，即使每个候选真实质量相同，选择 search score 最高者也会利用 evaluator 噪声。候选越多，winner's curse 越强：

$$\mathbb E\left[\max_{1\le i\le N}(J+\epsilon_i)\right]>J.$$

因此要：

1. search/validation/test 三分；
2. 多随机种子并报告方差；
3. 记录搜索使用了多少次 evaluator；
4. 在新任务或新模型上冻结 harness 后复测；
5. 不让 proposer 读取 held-out labels 或 verifier internals。

## 7. 优化 Harness 还是重新训练模型

优先 harness edit：错误来自缺 context、工具调用顺序、retry、memory、并行或验证流程。

优先 weight update：错误跨任务重复、模型连协议都不能稳定遵循，或通用行为值得内化以降低 inference 成本。

这不是二选一；第 `08` 章将二者写成双时间尺度优化。

<!-- NAVIGATION -->
## 导航

- 上一篇：[04 Workflow 与 Subagents](04_workflow_and_subagents.md)
- 下一篇：[06 Self-Improving Harness](06_self_improving_harness.md)
- 回到：[专题 README](README.md)
