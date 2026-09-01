# 05 — Harness Optimization：Context、Workflow、Code 三层搜索

## 元信息

- 核心论文：[Meta-Harness](https://arxiv.org/abs/2603.28052) · [AlphaEvolve](https://arxiv.org/abs/2506.13131)
- 前序方法：[ADAS](https://arxiv.org/abs/2408.08435v2) · [AFlow](https://arxiv.org/abs/2410.10762v4)
- 本章目标：定义可搜索空间、proposal semantics、评价预算与 archive，而不是把“LLM 改 prompt”泛化成任意自改进。

## 1. Search object：结构化 harness

使用统一分解：

$$h=(h^C,h^W,h^K,h^M).$$

候选 edit：

$$\delta=(\delta^C,\delta^W,\delta^K,\delta^M),\qquad h'=h\oplus\delta.$$

不同层的可搜索性不同：

| 层 | 参数化方式 | 典型优化 | 验证难度 | 主要风险 |
|---|---|---|---|---|
| Context $h^C$ | 文本、模板、检索/压缩配置 | prompt optimizer、bandit、ACE | 低到中 | leakage、冗余、staleness |
| Workflow $h^W$ | typed graph / DSL | MCTS、graph rewrite、AFlow | 中 | 成本膨胀、循环、join error |
| Code $h^K$ | source diff、tool/middleware | coding agent、evolution | 高 | 越权、破坏状态、evaluator tampering |
| Memory $h^M$ | schema、rules、read/write policy | incremental curation、MCE | 中到高 | 自我污染、错误持久化 |

标题中的三层把 $h^M$ 视为 Layer C 的 stateful information plane；这里单独列出，是因为 memory write 会跨 rollout 持久化，验证和回滚语义强于普通 prompt edit。Prompt search 是 $\mathcal H_C$ 的一个子集；完整 Harness Engineering 不是它的同义词。

## 2. Objective：黑盒、约束、多目标

模型参数冻结时，candidate 仍要通过执行估计效用：

$$\hat J_D(h)=\frac1{|D|}\sum_{x\in D}\frac1m\sum_{j=1}^mR(\tau_{x,j},x),\qquad \tau_{x,j}\sim p_{\theta,h}.$$

搜索问题：

$$\max_{h\in\mathcal H_{\mathrm{editable}}}\ \hat J_D(h)-\lambda_KK(h)-\lambda_L\mathrm{Lat}(h)-\lambda_QQ(h),$$

$$\text{s.t.}\quad h\models\mathrm{Schema},\quad \mathrm{Cap}(h)\subseteq\Pi,\quad K(h)\le B.$$

由于 $h$ 包含字符串、图和代码，目标离散、随机、昂贵且不可微。优化器可能是随机搜索、Bayesian optimization、MCTS、evolutionary search 或 LLM proposer；无论哪种，真正瓶颈常是 evaluator sample complexity。

若一次 candidate evaluation 的方差为 $\sigma^2$，$m$ 次独立 rollout 的均值标准误约为：

$$\mathrm{SE}(\hat J)=\frac{\sigma}{\sqrt m}.$$

只跑一次就比较小差距，会让搜索偏好 lucky candidate。

## 3. Typed edit grammar

任意自由文本 diff 很难保护不变量。可以定义 edit grammar：

```text
ContextEdit:
  add_rule | remove_rule | reorder_section | change_retrieval | change_compressor
WorkflowEdit:
  insert_node | delete_node | replace_node | change_edge | change_join | change_budget
CodeEdit:
  patch_tool | patch_parser | patch_middleware | add_test | migrate_state
MemoryEdit:
  add_entry | supersede_entry | change_schema | change_read_policy
```

每个 edit 带 manifest：

```json
{
  "parent_digest": "...",
  "layer": "workflow",
  "evidence": ["run-31:event-77"],
  "failure_cluster": "duplicate-write-after-timeout",
  "hypothesis": "reconcile before retry removes duplicate side effects",
  "operations": [{"op": "insert_node", "after": "timeout", "node": "reconcile"}],
  "expected_fix": ["duplicate-write"],
  "at_risk": ["latency", "tool-budget"],
  "required_capabilities": ["read-operation-status"]
}
```

Static validation 在执行前检查 schema、可达性、循环上界、permission path、import allowlist 和 state migration。

## 4. 三种搜索粒度

### 4.1 Local edit

每次修改一个组件，便于归因：

$$h_{t+1}=h_t\oplus\delta_t.$$

优点是 audit 和 rollback 简单；缺点是无法跨越需要协同修改的 valley。

### 4.2 Bundle edit

同时改 context、workflow 和 code：

$$\delta_t=(\delta_t^C,\delta_t^W,\delta_t^K).$$

能实现完整功能，但若性能变化，难判断是哪一部分贡献。应要求 bundle 内部 ablation 或依赖说明。

### 4.3 Population / archive

维护多个候选谱系：

$$\mathcal P_{t+1}=\mathrm{Keep}(\mathcal P_t\cup\{h_c\}),\qquad h_c=h_p\oplus\delta.$$

## 5. Search algorithms

### Random / coordinate search

逐层采样 edit，适合小空间和建立 baseline。没有简单 baseline，就无法判断复杂 LLM proposer 是否真的有效。

### Bandit / Bayesian optimization

适合低维连续超参，如 retrieval top-$k$、timeout、fan-out；对任意代码 diff 的 kernel/距离定义困难。

### MCTS

把 workflow rewrite 视为 tree action。UCB：

$$h_t=\arg\max_h\left[\hat J(h)+c\sqrt{\frac{\log N}{n_h}}\right].$$

适合有局部可组合 edit 的结构搜索，但 rollout 昂贵、树统计易受非平稳 evaluator 影响。

### Evolutionary search

$$h_p\sim\mathrm{Select}(\mathcal P_t),\qquad h_c\sim \rho_\phi(\cdot\mid h_p,z_t),$$

$$\mathcal P_{t+1}=\mathrm{Archive}(\mathcal P_t,h_c,\mathbf y_c).$$

需要 parent selection、novelty、elitism、lineage、budget 和 sandbox。LLM 负责 mutation 不等于 LLM 负责 acceptance。

## 6. Meta-Harness：端到端 executable harness search

[Meta-Harness](https://arxiv.org/abs/2603.28052) 固定基础模型，用 coding-agent proposer 读取候选代码、scores 和 trajectories，优化 executable harness：

$$H^\star=\arg\max_H\mathbb E_{x,\tau\sim p_M(H,x)}[r(\tau,x)].$$

它的重要贡献是把 prompt、control flow、tools 等放进同一 code search space，并保留 quality/cost Pareto frontier。论文报告：online classification 相对 ACE 增加 $7.7$ accuracy points 且 context tokens 少 $4\times$；200 道未见 IMO-level math problems 上五个模型相对 no retrieval 平均增加 $4.7$ points。

边界同样重要：TerminalBench-2 的 89 tasks 同时用于搜索和最终评估，因此 Opus 4.6 的 $76.4\%$、Haiku 4.5 的 $37.6\%$ 不是干净 held-out generalization。详见 [`papers.md`](papers.md)。

## 7. AlphaEvolve：程序搜索的可迁移机制

[AlphaEvolve](https://arxiv.org/abs/2506.13131) 的核心：

```text
initial program + mutable blocks + evaluator
  -> sample parents/inspirations from database
  -> model ensemble proposes code diff
  -> execute evaluator cascade
  -> store fit and diverse children
  -> repeat
```

其 archive 类似 MAP-Elites/island 思路，支持多指标和分布式 evaluator。可迁移到 harness 的原则：

- 显式标记 mutable region；
- 先跑便宜 validity checks，再跑昂贵 benchmark；
- 保存 lineage 与失败；
- 选择质量和多样性，而非单一最高分；
- evaluator 必须足够自动化。

论文的数学/系统成果很强，但主要适用于可自动评分问题；“能跑并得高分”不能防止 specification gaming。生产例子还经过额外部署或人工验证，不能只照搬 evolution loop。

## 8. Pareto archive 与选择

候选指标：

$$\mathbf y(h)=(J_{\mathrm{task}},-K_{\mathrm{token}},-K_{\mathrm{tool}},-L,Q_{\mathrm{safety}},-C_{\mathrm{maint}}).$$

非支配关系：

$$h_a\succ h_b\iff \forall j,\ y_j(h_a)\ge y_j(h_b)\ \land\ \exists j,\ y_j(h_a)>y_j(h_b).$$

Archive 只解决“保留哪些候选”，不解决“上线哪一个”。部署仍需要业务权重、hard constraints 和 approval。维护成本可用代码复杂度、组件数、迁移数量、failure recovery time 等 proxy，但不能假装一个 proxy 等于真实长期成本。

## 9. Optimizer overfitting 与 winner's curse

若 $N$ 个真实质量相同的候选分数为 $J+\epsilon_i$：

$$\mathbb E\left[\max_{1\le i\le N}(J+\epsilon_i)\right]>J.$$

搜索次数越多，最高分越可能只是噪声。缓解：

1. 记录 evaluator query count；
2. candidate 使用 $D_{\mathrm{search}}$；
3. promotion 使用隐藏 $D_{\mathrm{ho}}$；
4. 最终冻结后只在 $D_{\mathrm{test}}$ 一次评估；
5. 多 seed、paired comparison、置信区间；
6. 在新模型/新任务上做 transfer；
7. 对 benchmark-specific rules 加 complexity penalty。

若 $D_{\mathrm{ho}}$ 被每轮反复查询，它实际上是 validation set，不应再叫 untouched test。

## 10. Failure attribution 决定搜索效率

没有 attribution，proposal distribution 接近盲搜。给 trace $\tau$ 和 failure label $y$，attributor 产生 component posterior：

$$p_A(z\mid\tau,y),\qquad z\in\{C,W,K,M,\theta,\mathcal E,V\}.$$

其中 $\mathcal E$ 是外部环境，$V$ 是 verifier。只有 $z\in\{C,W,K,M\}$ 才进入 harness edit；若最大概率是模型能力、环境故障或 verifier bug，应转给相应 owner。

Proposal acquisition 可综合收益、不确定性和成本：

$$\mathrm{Acq}(\delta)=\mathbb E[\Delta J\mid z,\delta]+\beta\mathrm{Uncertainty}(\delta)-\lambda_E\mathrm{EvalCost}(\delta)-\lambda_R\mathrm{Risk}(\delta).$$

高不确定但低风险的 edit 值得探索；高风险 edit 不能仅因潜在收益大而自动测试在真实环境。

## 11. Failure modes

- search space 太宽，候选多数不可执行；
- proposer 同时读 verifier internals 和 hidden labels；
- 只报 best-of-$N$，不报 $N$、成本和方差；
- 增加模型能力/预算却记作 harness gain；
- archive diversity 只按文本差异，不按行为差异；
- bundle edit 无 ablation，credit assignment 失真；
- candidate 修改 state schema 却无 migration；
- objective hacking：修 detector 而非真实 failure；
- 在 benchmark search set 上长期演化，再宣称泛化。

## 12. Engineering checklist

- [ ] 三层 editable surface 和 control plane 明确分离；
- [ ] edit grammar 有 schema、type 和 static validation；
- [ ] candidate 在 sandbox 内、最小权限运行；
- [ ] archive 保存 lineage、metrics、cost、failures 和 evaluator version；
- [ ] quality/cost/latency/risk 保留 Pareto 信息；
- [ ] evaluator query count、seed、attempts 和选择规则可复现；
- [ ] hidden regression 与 final test 分开；
- [ ] attribution 错误能路由到 model/data/infra/verifier，而不是强行改 harness。

<!-- NAVIGATION -->
## 导航

- 上一篇：[04 Workflow 与 Subagents](04_workflow_and_subagents.md)
- 下一篇：[06 Self-Improving Harness](06_self_improving_harness.md)
- 回到：[专题 README](README.md)
