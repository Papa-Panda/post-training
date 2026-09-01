# Papers and Evidence Ledger

> 原则：Lilian Weng 的博客用于建立问题地图；模型、benchmark、数字和论文结论尽量回到原论文/官方 proceedings。无法从原文稳定核实的 claim 不进入 README headline。

## A. 起点综述

| Year | Source | Role | Boundary |
|---|---|---|---|
| 2026 | [Lilian Weng, *Harness Engineering for Self-Improvement*](https://lilianweng.github.io/posts/2026-07-04-harness/) | 将 workflow、context、self-harness、evolution 与 RSI 串成地图 | 二手综述；不是这些论文实验结论的唯一证据 |

## B. Context 与 Workflow

| Year / status | Paper | Primary source | 本专题使用的 claim |
|---|---|---|---|
| ICLR 2026 | **Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models (ACE)** | [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/file/8a94ff6f922d995d7d3f4ebf4143e442-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2510.04618v3) | Generator→Reflector→Curator；以增量 delta 更新结构化 playbook，而不是反复整段重写 |
| 2026 preprint | **Meta Context Engineering via Agentic Skill Evolution (MCE)** | [arXiv](https://arxiv.org/abs/2601.21557) · [code](https://github.com/metaevo-ai/meta-context-engineering) | 外层演化 context-management skill，内层优化 context function；文件/代码成为 executable context |
| ICLR 2025 | **Automated Design of Agentic Systems (ADAS)** | [arXiv](https://arxiv.org/abs/2408.08435v2) · [code](https://github.com/ShengranHu/ADAS) | Meta Agent Search 维护 archive，以代码提出、修复和评估新 Agent design |
| ICLR 2025 | **AFlow: Automating Agentic Workflow Generation** | [arXiv](https://arxiv.org/abs/2410.10762v4) · [OpenReview](https://openreview.net/forum?id=z5uVAKwmjf) · [code](https://github.com/FoundationAgents/AFlow) | 把 workflow 表示为 LLM nodes + code edges，用 MCTS 风格循环搜索 |

### 已核实数字与口径

- ACE 论文报告在其设置中，agent tasks 提升 **10.6%**、finance tasks 提升 **8.6%**。保留论文的 `%` 写法，不自行改成 percentage points，也不外推到所有 Agent。
- MCE 报告跨五个领域和四个 LLM，相对既有 context-engineering 方法提升范围 **5.6%–53.8%**，平均 **16.9%**；还报告相对 ACE **13.6× 更快训练、4.8× 更少 rollouts**。这是论文实验口径，不是通用成本定律。
- ADAS 报告 DROP F1 增加 **13.6/100**、MGSM accuracy 增加 **14.4%**，以及迁移到 GSM8K/GSM-Hard 的 **25.9% / 13.2%** 改善；本专题保留原单位，不重新解释相对/绝对口径。
- AFlow 在论文的 GPT-4o-mini divided-test 设置、六个 benchmark、三次运行平均下，报告相对手工方法 **5.7%**、相对 ADAS **19.5%** 改善。摘要的 “4.55% cost” 与 appendix 更精确口径有差异：appendix 支持的是该成本下 parity；超过 GPT-4o 的配置成本为 **5.92% 或 8.05%**，因此正文不使用更强的摘要措辞。

## C. End-to-end Harness Optimization

| Year / status | Paper | Primary source | 本专题使用的 claim |
|---|---|---|---|
| 2026 preprint | **Meta-Harness: End-to-End Optimization of Model Harnesses** | [arXiv](https://arxiv.org/abs/2603.28052) · [project](https://yoonholee.com/meta-harness/) · [code](https://github.com/stanford-iris-lab/meta-harness) | 固定基础模型，coding-agent proposer 读取代码/score/traces，搜索 executable harness 并保留 Pareto frontier |
| COLM 2024 | **Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation** | [arXiv](https://arxiv.org/abs/2310.02304) | 优化 improver 本身；递归 scaffold 只有在基础模型足够强时才可能持续改善 |
| 2026 preprint | **Self-Harness: Harnesses That Improve Themselves** | [arXiv](https://arxiv.org/abs/2606.09498) | weakness mining → bounded proposals → held-in/held-out validation；accepted edits 才更新 active harness |
| 2026 preprint；official repo 标注 EMNLP 2026 Main | **Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents** | [arXiv](https://arxiv.org/abs/2605.30621) · [code](https://github.com/A-EVO-Lab/a-evolve/tree/release/harness-evolution) | 拆分“会提出 harness 修改”与“会利用修改后 harness”两个能力轴 |
| 2026 preprint | **Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses** | [arXiv](https://arxiv.org/abs/2604.25850v4) · [code](https://github.com/china-qijizhifeng/agentic-harness-engineering) | component / experience / decision observability；editable workspace 与只读 evaluator/tracer/model config 分离 |

### Meta-Harness 数字边界

论文报告：online classification 相对 ACE 增加 **7.7 accuracy points** 且 context tokens 少 **4×**；五模型、200 道未见 IMO-level math problem 上相对 no retrieval 平均增加 **4.7 points**；TerminalBench-2 报告 Opus 4.6 为 **76.4%**、Haiku 4.5 为 **37.6%**。

但必须同时保留：

- math problem 是 unseen；五个 base models 中四个未参与搜索，GPT-OSS-20B 用于 search；
- TerminalBench-2 使用同一批 89 tasks 搜索和最终评估，不是干净 held-out generalization；
- Opus 结果是当时 overall #2；Haiku 结果只是在所报告 Haiku 4.5 agents 中 #1。

因此 README 不使用这些数字作为跨任务 headline。

### STOP / Self-Harness / AHE / Updating-vs-Benefit 的不同验证协议

| Work | Candidate 何时生效 | Promotion / rollback | 最重要的数字边界 |
|---|---|---|---|
| STOP | 对 improver programs 做 empirical meta-utility 搜索，再递归用于自身 | utility selection；没有 split-wise non-regression gate | LPN 用 20 个 meta-training instances、50 个独立 unseen instances；递归轮次不保证单调 |
| Self-Harness | bounded candidates 与当前 harness 同轮比较 | held-in、hidden held-out 都不下降，且至少一边严格改善才合并 | MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5 × Terminal-Bench-2.0、SWE-bench Verified、AppWorld；九组都同时改善两 split，最大整体绝对增益为 40.6 points |
| AHE | edit 先 commit；下一轮 rollout 归因上一轮 manifest | 下一轮 keep/rollback，并单独保留历史最优 harness | GPT-5.4 high 在同一 89-task Terminal-Bench 2 集上十轮演化到 77.0% pass@1；不是 held-out。冻结后 SWE-bench Verified 75.6%，seed 75.2% |
| Updating vs Benefit | solver $f$ 与 evolver $e$ 交叉组合，按 task stream 顺序更新 | 研究重点是能力解耦，不是 candidate-promotion 协议 | 三 benchmark 上 best/worst evolver 最大差距 3.1 points；solver benefit 非单调 |

Self-Harness 的 held-out 对 proposer 隐藏，但 promotion gate 会反复查询，因此应称 hidden regression/validation split，而非 untouched final test。AHE 的 regression prediction precision/recall 为 **11.8% / 11.1%**，说明 manifest 和 rollback 改善审计性，但预测本身仍弱。Updating-vs-Benefit 的主协议是 in-situ evaluation：每个 task 先锁定旧 harness 下的分数，再把 trace 用于下一版；这避免从自己的 evidence 获益，但仍不同于独立 held-out promotion。

## D. Evolutionary Program / Agent Search

| Year / status | Paper | Primary source | 本专题使用的 claim |
|---|---|---|---|
| 2025 white paper | **AlphaEvolve: A coding agent for scientific and algorithmic discovery** | [arXiv](https://arxiv.org/abs/2506.13131) | LLM 生成 program diff，自动 evaluator 赋 fitness，archive 保留高质量且多样的候选 |
| 2025 preprint | **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents** | [arXiv](https://arxiv.org/abs/2505.22954) | 固定基础模型下让 coding agents 修改自己的 harness codebase，并以 archive 保留谱系 |

DGM 摘要报告在其设置中，SWE-bench 从 **20.0% → 50.0%**，Polyglot 从 **14.2% → 30.7%**。这里变化的是固定模型外部的 Agent scaffold/codebase；archive lineage、sandbox 与 rollback 有助于审计，但不构成通用安全证明。

这类方法最适合 evaluator 快速、客观的程序任务。自动验证弱的开放研究不应直接套用其性能结论。

## E. Joint Harness / Weight Updates

| Year / status | Paper | Primary source | 本专题使用的 claim |
|---|---|---|---|
| 2026 preprint | **SIA: Self Improving AI with Harness & Weight Updates** | [arXiv](https://arxiv.org/abs/2605.27276) | Feedback-Agent 根据 trajectories 在 harness update 与 weight update 之间路由 |
| 2026 preprint | **Continual Harness: Online Adaptation for Self-Improving Foundation Agents** | [arXiv](https://arxiv.org/abs/2605.09998) | 长 horizon 环境中联合 harness adaptation 与 policy co-learning 的探索 |

SIA 的方向值得关注，但现有实验存在角色模型能力不一致和 baseline 较弱等 confound；本专题只采用其控制问题，不宣称联合优化已被普遍证明。

## F. Benchmark 接口

| Benchmark | 测什么 | Primary source |
|---|---|---|
| PaperBench | 复现 AI 论文的分解 rubric | [arXiv](https://arxiv.org/abs/2504.01848) |
| RE-Bench | 真实 ML R&D 环境与人类专家对比 | [arXiv](https://arxiv.org/abs/2411.15114) |
| MLE-bench | Kaggle 风格 ML engineering | [arXiv](https://arxiv.org/abs/2410.07095) |
| CORE-Bench | 给定 code/data 的论文计算可复现性 | [arXiv](https://arxiv.org/abs/2409.11363v2) |
| KernelBench | GPU kernel correctness + speed | [arXiv](https://arxiv.org/abs/2502.10517) |

## Claim 使用规则

1. “搜索集上的最高分”不写成 held-out generalization；
2. accuracy points、relative percent、F1/100 保持原单位；
3. 固定模型的 harness gain 不等于模型 intelligence gain；
4. 能产生 edit 不等于部署模型能利用 edit；
5. benchmark 自动验证成功不外推到科学品味、长期维护或安全；
6. preprint 结论标明状态，后续版本变化需要重新核对。

<!-- NAVIGATION -->
## 导航

- 上一篇：[08 Harness、RL 与权重更新](08_harness_rl_and_weight_updates.md)
- 下一篇：[专题 README](README.md)
- 运行代码：[code/demo.py](code/demo.py) · [tests/test_harness_lab.py](tests/test_harness_lab.py)
