# 06 — Self-Improving Harness：提案、验证、接受

## 元信息

- 核心论文：[STOP](https://arxiv.org/abs/2310.02304) · [Self-Harness](https://arxiv.org/abs/2606.09498) · [Agentic Harness Engineering](https://arxiv.org/abs/2604.25850) · [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)
- 内容类型：跨论文闭环综合；论文结果与本专题工程规范分开表述。

## 1. 自改进不是自覆盖

安全的最小闭环是：

$$T_t=\mathrm{Rollout}(h_t,D),$$

$$F_t=\mathrm{MineFailures}(T_t),\qquad \mathcal C_t=\mathrm{Propose}(h_t,F_t),$$

$$h_{t+1}=\mathrm{Gate}(h_t,\mathcal C_t,D_{\mathrm{in}},D_{\mathrm{out}}).$$

Active harness 不能被 proposer 直接覆盖。候选先进入 sandbox；只有满足修复、无回归、预算和权限条件才创建新版本。

## 2. STOP：改进 improver

[Self-Taught Optimizer (STOP)](https://arxiv.org/abs/2310.02304) 不直接优化单个解 $s$，而是优化改进函数 $I$。给定黑盒模型 $M$ 与 utility $u$：

$$s'=I(u,s;M).$$

在任务分布 $\mathcal D$ 上定义 improver 的 meta-utility：

$$\hat u(I)=\frac1{|\mathcal D|}\sum_{(u,s)\in\mathcal D}u(I(u,s;M)).$$

然后让当前 improver 递归改写自己：

$$I_t=I_{t-1}(\hat u,I_{t-1};M).$$

STOP 的重要负面边界是：递归结构本身不保证提升；论文实验中更强模型可发现有用策略，而较弱模型的平均表现可能下降。基础模型仍必须有足够能力理解并改进机制。

## 3. Self-Harness：从失败模式产生 bounded edits

[Self-Harness](https://arxiv.org/abs/2606.09498) 的结构可分三步：

1. **Weakness mining**：从 execution traces 中提取 verifier-grounded failure patterns；
2. **Harness proposal**：给 proposer 可编辑面、失败模式、应保留行为和历史尝试，生成 bounded edits；
3. **Proposal validation**：在 held-in 与 held-out 上回归，只合并合格候选。

表面相同的 timeout 可能来自错误重试、上下文丢失或工具 schema；failure record 应同时保存终端结果、因果行为和暴露出的 Agent 机制，避免把 symptom 当 root cause。

令 $\Delta_{\mathrm{in}}=J_{\mathrm{in}}(h')-J_{\mathrm{in}}(h)$、$\Delta_{\mathrm{ho}}=J_{\mathrm{ho}}(h')-J_{\mathrm{ho}}(h)$。论文的精确接受规则是：

$$\Delta_{\mathrm{in}}\ge0,\qquad \Delta_{\mathrm{ho}}\ge0,\qquad \max(\Delta_{\mathrm{in}},\Delta_{\mathrm{ho}})>0.$$

即两个 split 都不回退，且至少一个严格改善。held-in trace 对 proposer 可见，hidden held-out 只供 promotion gate 使用；它会被反复查询，因此更准确地说是隐藏 regression/validation split，而不是只用一次的最终 test set。论文在 MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5 上评测 Terminal-Bench-2.0、SWE-bench Verified、AppWorld，九个组合的最终 harness 都同时改善两个 split；最大整体绝对增益是 GLM-5/AppWorld 的 $40.6$ 个百分点。

本专题的最小代码采用更保守、也更容易解释的规则：held-in 必须严格改善、held-out 不下降，再叠加 permission 与 cost gate。

## 4. AHE：三层可观测性

[Agentic Harness Engineering](https://arxiv.org/abs/2604.25850) 把瓶颈定位为 observability：如果不知道哪一层导致失败，自动 edit 只是盲搜。

- **Component observability**：system prompt、tool description、tool implementation、middleware、skill、subagent config、long-term memory 都有显式文件表示；
- **Experience observability**：raw traces → per-task root-cause reports → benchmark overview，按需逐层展开；
- **Decision observability**：每个 edit 是可证伪预测，记录 evidence、root cause、fix、expected gain 和 at-risk regression。

每个 proposal manifest：

```json
{
  "evidence": ["run-31/tool-7"],
  "root_cause": "retry loses idempotency key",
  "surface": "middleware/retry.py",
  "edit": "propagate key across retry",
  "expected_fix": ["duplicate-write"],
  "at_risk": ["latency", "retry-budget"]
}
```

AHE 与 Self-Harness 的验证时序不同：edit 先 commit，下一轮 rollout 才归因上一轮 manifest，失败则 rollback；系统同时保留历史最优 $H_{\mathrm{best}}$。主实验固定 GPT-5.4 high，在同一组 89 个 Terminal-Bench 2 tasks 上做十轮演化并报告 $77.0\%$ pass@1；这不是独立 held-out 泛化。冻结后转到 SWE-bench Verified 得到 $75.6\%$，seed 为 $75.2\%$，但部分 repository 仍有退化。其 regression prediction precision/recall 仅 $11.8\%/11.1\%$，说明 observability 与 rollback 有价值，却不能替代可靠 regression gate。

## 5. DGM：开放式 Harness 种群

[Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) 维护可自修改 coding agents 的 archive。每轮选择 parent，Agent 检查 benchmark log 并修改自己的 codebase，评估后把足够好的 child 加回种群。

与单链 `h_t\to h_{t+1}` 相比，archive 保留多个谱系：

$$\mathcal P_{t+1}=\mathrm{Archive}(\mathcal P_t\cup\{\mathrm{Mutate}(h_p)\}).$$

这有利于探索，但也增加计算量、benchmark overfitting 和安全审计难度。论文报告的 benchmark 改善见 [`papers.md`](papers.md)；它们是在固定模型与特定 coding benchmark 下的实验，不是通用 RSI 证明。

## 6. 最小实现对应

[`code/harness_lab.py`](code/harness_lab.py) 把上述原则缩成：

```text
Proposal
  -> reject immutable edits
  -> materialize candidate version
  -> held-in evaluation
  -> held-out evaluation
  -> permission + cost checks
  -> accept and version OR reject without mutation
```

它刻意不让 candidate 接触 evaluator 的修改接口；active version 只有一个，但 accepted snapshots 全部保留，rejected decisions 留审计记录。

## 7. 成功条件

自改进闭环至少需要：

- 可验证任务与稳定 evaluator；
- rich trace 足以定位 root cause；
- bounded、版本化 edit；
- held-out regression 和成本约束；
- 可恢复 archive 和停止标准；
- 人类在权限扩大、外部副作用和重大部署处审批。

<!-- NAVIGATION -->
## 导航

- 上一篇：[05 Harness Optimization](05_harness_optimization.md)
- 下一篇：[07 Observability、Evaluation 与 Security](07_observability_evaluation_security.md)
- 回到：[专题 README](README.md)
