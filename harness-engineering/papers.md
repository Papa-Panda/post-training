# Papers and Evidence Ledger

> 目的：把“论文说了什么”“本专题如何使用”“不能推出什么”分开。数字优先回到论文正文、附录或官方 proceedings；博客只用于建立问题地图。

## 0. 读表方法

每篇论文按五个问题核验：

1. **Fixed surface**：基础模型、search algorithm、evaluator 中哪些固定？
2. **Editable surface**：context、workflow、code、memory、weights 中哪些可改？
3. **Evidence timing**：proposal 看过什么 traces/labels？candidate 何时运行？
4. **Acceptance**：同轮 gate、下一轮 rollback、archive fitness，还是只报最终 best？
5. **Boundary**：数字来自 search tasks、hidden validation、unseen transfer，还是 final test？

## A. 起点综述

| Year | Source | Role | Boundary |
|---|---|---|---|
| 2026 | [Lilian Weng, *Harness Engineering for Self-Improvement*](https://lilianweng.github.io/posts/2026-07-04-harness/) | 把 workflow、context、self-harness、evolution 与 recursive self-improvement 串成问题地图 | 二手综述；不作为模型、benchmark、数字和精确协议的唯一证据 |

## B. Context engineering

### B.1 ACE

- **Exact title**：*Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models*
- **Status/source**：ICLR 2026；[proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/8a94ff6f922d995d7d3f4ebf4143e442-Paper-Conference.pdf) · [arXiv v3](https://arxiv.org/abs/2510.04618v3)
- **Editable surface**：结构化 playbook/context；基础模型权重固定。
- **Mechanism**：Generator 产生 rollout；Reflector 提炼成功/失败 insight；Curator 将增量 delta 合并入 playbook。
- **本专题使用**：稳定 ID、incremental update、evidence-grounded merge，比每轮整段重写更可追踪。
- **Reported numbers**：论文口径下 agent tasks 提升 **10.6%**、finance tasks 提升 **8.6%**。
- **Boundary**：保留论文的百分号口径，不自行改写为 percentage points，也不外推为所有 Agent 的通用增益。

### B.2 MCE

- **Exact title**：*Meta Context Engineering via Agentic Skill Evolution*
- **Status/source**：2026 preprint；[arXiv](https://arxiv.org/abs/2601.21557) · [official code](https://github.com/metaevo-ai/meta-context-engineering)
- **Editable surface**：prompts、knowledge bases、code 等 context resources，以及 retrieve/filter/format/update 的 context-management skill。
- **Mechanism**：外层演化 skill，内层优化 context function，形成双层优化。
- **Reported numbers**：跨五个领域、四个 LLM，相对既有 context-engineering 方法提升 **5.6%–53.8%**，平均 **16.9%**；相对 ACE 报告 **13.6×** 更快训练、**4.8×** 更少 rollouts。
- **Boundary**：这些是论文设置中的比较，不是 context program 一定更快的通用规律；外层若接触 validation labels，仍会泄漏。

## C. Workflow / agent design search

### C.1 ADAS

- **Exact title**：*Automated Design of Agentic Systems*
- **Status/source**：ICLR 2025；[arXiv v2](https://arxiv.org/abs/2408.08435v2) · [official code](https://github.com/ShengranHu/ADAS)
- **Editable surface**：Agent system code/description；基础 model 固定。
- **Mechanism**：Meta Agent Search 从 archive 取 prior designs，生成新 design、实现代码、修复错误、evaluate、回写 archive。
- **Reported numbers**：DROP F1 增加 **13.6/100**、MGSM accuracy 增加 **14.4%**；迁移到 GSM8K/GSM-Hard 报告 **25.9% / 13.2%** 改善。
- **Boundary**：保留原论文单位，不混用 relative percent、accuracy point 和 F1 point；archive 中高分结构可能包含 benchmark-specific tricks。

### C.2 AFlow

- **Exact title**：*AFlow: Automating Agentic Workflow Generation*
- **Status/source**：ICLR 2025；[arXiv v4](https://arxiv.org/abs/2410.10762v4) · [OpenReview](https://openreview.net/forum?id=z5uVAKwmjf) · [official code](https://github.com/FoundationAgents/AFlow)
- **Editable surface**：LLM nodes 与 code edges 组成的 workflow；MCTS 风格搜索。
- **Reported numbers**：GPT-4o-mini divided-test 设置、六个 benchmark、三次运行平均，报告相对手工 workflows **5.7%**、相对 ADAS **19.5%** 改善。
- **Cost caveat**：摘要的 “4.55% cost” 在 appendix 的精确条件下支持 parity；超过 GPT-4o 的配置为 **5.92% 或 8.05%** cost。不能写成“以 4.55% 成本普遍超越”。
- **Boundary**：workflow search 的样本复用和 search budget 必须与最终 held-out 分开。

## D. End-to-end harness optimization

### D.1 Meta-Harness

- **Exact title**：*Meta-Harness: End-to-End Optimization of Model Harnesses*
- **Status/source**：2026 preprint；[arXiv](https://arxiv.org/abs/2603.28052) · [project](https://yoonholee.com/meta-harness/) · [official code](https://github.com/stanford-iris-lab/meta-harness)
- **Fixed surface**：被优化的基础模型固定；coding-agent proposer 另行生成 harness code edits。
- **Editable surface**：prompt、control flow、tools 等 executable harness code。
- **Search/evaluation**：proposer 读取候选代码、scores 和 trajectories；保留 quality/cost Pareto frontier。
- **Reported numbers**：online classification 相对 ACE 增加 **7.7 accuracy points**，context tokens 少 **4×**；200 道未见 IMO-level math problems、五个 base models 上相对 no retrieval 平均增加 **4.7 points**；TerminalBench-2 报告 Opus 4.6 **76.4%**、Haiku 4.5 **37.6%**。
- **Boundary**：math problems 是 unseen，且五个 base models 中四个未参与 search；GPT-OSS-20B 用于 search。TerminalBench-2 使用同一批 **89 tasks** 搜索和最终评估，不是干净 held-out generalization；Opus 是当时 overall #2，Haiku 只是在所报告 Haiku 4.5 agents 中 #1。

### D.2 STOP

- **Exact title**：*Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation*
- **Status/source**：COLM 2024；[arXiv v3](https://arxiv.org/abs/2310.02304v3)
- **Fixed surface**：语言模型权重；外层 evaluator/utility。
- **Editable surface**：调用 LM 的 improver/scaffolding code。
- **Search/evaluation**：对 improver programs 做 empirical meta-utility search，再让 improver 改进自身版本。
- **Data timing**：LPN downstream utility 对 $M=20$ 个独立 LPN instances 取平均；meta-dataset $D$ 含同一个 $(u,s)$ 的 **5** 个副本；test metautility 另在 $M_{\mathrm{test}}=50$ 个独立 LPN instances 上报告。test instances 用于报告，不是 Self-Harness 式 promotion gate。
- **Boundary**：递归轮次不保证单调改善；论文检查的 unsandboxing code-pattern proxy 为 GPT-4 **0.42%**、GPT-3.5 **0.12%**，只能按此狭窄 proxy 陈述，不等于完整 sandbox escape rate 或安全证明。

### D.3 Self-Harness

- **Exact title**：*Self-Harness: Harnesses That Improve Themselves*
- **Status/source**：2026 preprint；[arXiv](https://arxiv.org/abs/2606.09498)
- **Editable surface**：结构化 harness components；基础 solver models 固定。
- **Protocol**：Weakness Mining→Harness Proposal→Proposal Validation；held-in traces 对 proposer 可见，held-out 隐藏。
- **Exact gate**：

$$\Delta_{\mathrm{in}}\ge0,\qquad \Delta_{\mathrm{ho}}\ge0,\qquad \max(\Delta_{\mathrm{in}},\Delta_{\mathrm{ho}})>0.$$

- **Meaning**：两个 split 都不得下降，至少一个严格提升；accepted compatible edits 才合并。
- **Models/benchmarks**：MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5；Terminal-Bench-2.0 固定 **64-case subset**，SWE-bench Verified **100 cases**（67/33），AppWorld **180 cases**（90/90）。
- **Reported outcome**：九组最终 harness 均同时改善两个 split；最大整体绝对增益为 GLM-5/AppWorld **+40.6 points**。
- **Boundary**：held-out 会被 promotion 反复查询，因此是 hidden regression/validation，不是 untouched final test；九组结果不能外推成工业安全证明。

### D.4 Agentic Harness Engineering

- **Exact title**：*Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses*
- **Status/source**：2026 preprint；[arXiv v4](https://arxiv.org/abs/2604.25850v4) · [official code](https://github.com/china-qijizhifeng/agentic-harness-engineering)
- **Observability**：component、experience、decision。
- **Timing**：edit 先 commit；下一轮 rollout 归因上一轮 manifest；不合格则 rollback；另保留 $H_{\mathrm{best}}$。这不是 Self-Harness 的同轮 candidate gate。
- **Main setup**：GPT-5.4 high；同一批 **89** 个 Terminal-Bench 2 tasks；$k=2$；**10 iterations**；约 **32 小时**。
- **Reported numbers**：main score **77.0% pass@1**，但不是 held-out。冻结后 SWE-bench transfer **75.6%**，seed **75.2%**；tokens/trial **461k vs 526k**。部分 repository 退化。
- **Attribution caveat**：regression prediction precision/recall **11.8% / 11.1%**，说明 attribution 仍弱；manifest/rollback 增强审计性，不等于预测可靠。

### D.5 Harness Updating Is Not Harness Benefit

- **Exact title**：*Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents*
- **Status/source**：2026 preprint；[arXiv](https://arxiv.org/abs/2605.30621) · [official code](https://github.com/A-EVO-Lab/a-evolve/tree/release/harness-evolution)。官方 repo 标注 “EMNLP 2026 Main”，arXiv metadata 本身未显示 venue，二者应区分陈述。
- **Design**：solver $f$ 与 evolver $e$ 交叉，区分 update ability 和 benefit ability。
- **Exact metrics**：

$$\Delta(f,e)=J_X(f,H_T^{(f,e)})-M_{\mathrm{base}}(f),$$

$$\Delta_{\mathrm{update}}(e)=\frac1{|F^\star|}\sum_{f\in F^\star}\Delta(f,e),\qquad \Delta_{\mathrm{benefit}}(f)=\max_{e\in E^\star}\Delta(f,e).$$

- **Reported numbers**：best/worst evolver 在任一 benchmark 的最大差距只有 **3.1 points**；无 evolver 三项都最佳。SLR/HFR：Qwen3-32B **0.251/0.142**，Qwen3-235B **0.961/0.350**，Opus 4.6 **0.957/0.757**。
- **Boundary**：主实验为 in-situ：task 先以 $H_{t-1}$ 评分，再把 trace 用于更新 $H_t$；这避免当前 task 从自己的 evidence 获益，但不等于独立 held-out promotion。

## E. Evolutionary program / agent search

### E.1 AlphaEvolve

- **Exact title**：*AlphaEvolve: A coding agent for scientific and algorithmic discovery*
- **Status/source**：2025 white paper；[arXiv PDF](https://arxiv.org/pdf/2506.13131)
- **Fixed surface**：outer evolution/database/evaluator machinery；自动 evaluator。
- **Editable surface**：显式 mutable code blocks。
- **Mechanism**：Gemini 2.0 Flash/Pro ensemble 生成 diff，evaluator cascade 评分，MAP-Elites/island-inspired archive 保留质量与多样性。
- **Reported examples**：在 **14** 个 matrix-multiplication targets 上改进 state of the art；complex $4\times4$ rank **48**，超过此前 rank 49；50+ math problems 中约 **75% match**、约 **20% surpass** 已知 best；scheduling 平均回收 **0.7%** fleet compute；TPU tiling heuristic 相对 expert-designed heuristic 在全部 kernels 上平均提速 **23%**，对应 Gemini overall training time 降低 **1%**；FlashAttention 在指定配置下 kernel 提速 **32%**，kernel 输入/输出的预处理与后处理部分提速 **15%**。
- **Boundary**：这是 white paper 与内部 production claims；适合 evaluator 清晰的任务。生产结果还有额外验证，不能外推为开放式 harness 搜索自动安全有效。

### E.2 Darwin Gödel Machine

- **Exact title**：*Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*
- **Status/source**：ICLR 2026；[arXiv PDF](https://arxiv.org/pdf/2505.22954)
- **Fixed surface**：外层 archive/search procedure 不自修改；基础模型权重固定。
- **Editable surface**：coding-agent harness/codebase；child 从 archive parent 产生。
- **Reported numbers after 80 iterations**：SWE experimental subset **20.0%→50.0%**；Polyglot 50-task subset **14.0%→38.0%**；full Polyglot **14.2%→30.7%**。
- **Cost**：单次 SWE run 约两周、约 **US$22,000**。
- **Boundary**：50.0% 不是全 500-task SWE-bench Verified。论文记录 objective hacking：node 114 绕过 detector 获得最高 evaluator score。Sandbox/archive/rollback 不等于完整安全证明。

## F. Joint harness / weight updates

### F.1 SIA

- **Title used by paper**：*SIA: Self Improving AI with Harness & Weight Updates*
- **Status/source**：2026 preprint v2；[arXiv HTML](https://arxiv.org/html/2605.27276v2)
- **Architecture**：Meta-Agent、Task Agent、Feedback-Agent；Feedback-Agent 在 harness update 和 weight update 之间选择。
- **Timing**：报告实验通常先 scaffold iteration，再在 reward plateau 后 weight update。
- **Weight methods**：LawBench PPO+GAE；TriMul entropic advantage weighting；denoising GRPO。
- **v2 Table 3**：LawBench **13.5%→50.0%→70.1%**，joint 相对 harness-only **+20.1 points**；TriMul reward **0.105→0.120→1.475**，runtime **12,483 μs→1,017 μs**；denoising $mse_{\mathrm{norm}}$ **0.048→0.241→0.289**。
- **Boundary**：不同任务的 reward/metric 不可横向比较；论文明确提出 coupled co-evolutionary Goodhart 风险。结果不证明所有任务都应联合更新。

### F.2 Continual Harness

- **Exact title**：*Continual Harness: Online Adaptation for Self-Improving Foundation Agents*
- **Status/source**：2026 preprint；[arXiv HTML](https://arxiv.org/html/2605.09998)
- **Harness update**：reset-free acting loop；每 $F$ 步 Refiner 对 system prompt、subagents、skills、memory 做 CRUD。
- **Co-learning**：每轮 $K=256$ steps；pairwise PRM；frontier teacher relabel；soft SFT；emulator state 不 reset。
- **Pokémon Emerald / Gemini 3.1 Pro**：from-scratch **100% milestones，US$130 median**；minimalist harness **98%，US$215**。
- **Boundary**：弱模型存在 capability floor，self-refinement 可能恶化；未证明 convergence，也没有 matched reset-based baseline。早期 GPP Pokémon completion 是 human-supervised 系统，不能归给 fully automated Continual Harness。论文没有实证给出一个一般 reward-hacking 发现，不能替它添加。

## G. Benchmark 接口

| Benchmark | 主要测量 | Primary source |
|---|---|---|
| PaperBench | 复现 AI 论文的分解 rubric | [arXiv](https://arxiv.org/abs/2504.01848) |
| RE-Bench | 真实 ML R&D 环境与人类专家对比 | [arXiv](https://arxiv.org/abs/2411.15114) |
| MLE-bench | Kaggle 风格 ML engineering | [arXiv](https://arxiv.org/abs/2410.07095) |
| CORE-Bench | 给定 code/data 的论文计算可复现性 | [arXiv v2](https://arxiv.org/abs/2409.11363v2) |
| KernelBench | GPU kernel correctness 与 speed | [arXiv](https://arxiv.org/abs/2502.10517) |

Benchmark 接口只告诉我们任务和 evaluator；不能自动保证 search/test split、sandbox、权限和长期维护指标正确。

## H. 跨论文比较

| Work | 主要 edit surface | Search style | Acceptance timing | 最关键 boundary |
|---|---|---|---|---|
| ACE | context/playbook | incremental reflection/curation | playbook update | 非通用增益定律 |
| MCE | context-management skill | bilevel evolution | skill evaluation | outer-loop leakage |
| ADAS | agent code/design | archive search | evaluated design 入 archive | benchmark-specific design |
| AFlow | workflow graph/code | MCTS-style | search score | cost headline 需分条件 |
| Meta-Harness | executable harness | coding-agent search + Pareto | candidate evaluation | TerminalBench 同 89 tasks search/eval |
| STOP | improver code | recursive meta-optimization | utility selection | 无 split-wise non-regression |
| Self-Harness | harness components | bounded proposals | same-round hidden regression gate | held-out 非 final test |
| AHE | coding-agent harness files | observability-driven edit | next-round attribution/rollback | main 77.0% 非 held-out |
| DGM | agent codebase | open-ended archive evolution | archive fitness | 50% 非 full SWE Verified；有 objective hack |
| SIA | harness + weights | feedback-routed alternation | task-specific | coupled Goodhart |
| Continual Harness | prompt/subagents/skills/memory + weights | reset-free continual adaptation | periodic refinement | 无 convergence/matched reset proof |

## I. Claim 使用规则

1. “搜索集最高分”不写成 held-out generalization；
2. accuracy points、relative percent、F1/100、reward 和 latency 保持原单位；
3. 固定模型上的 harness gain 不等于模型 intelligence gain；
4. 会提出 update 不等于 solver 能利用 update；
5. hidden held-out 被反复查询就是 validation，不是 untouched test；
6. benchmark 自动验证成功不外推到科学品味、长期维护或安全；
7. preprint/white paper/official repo 的 publication status 分开标注；
8. 任何数字进入 README 前，都回到本 ledger 的 source 和 boundary。

<!-- NAVIGATION -->
## 导航

- 上一篇：[08 Harness、RL 与 Weight Updates](08_harness_rl_and_weight_updates.md)
- 下一篇：[专题 README](README.md)
- 运行代码：[code/demo.py](code/demo.py) · [tests/test_harness_lab.py](tests/test_harness_lab.py)
