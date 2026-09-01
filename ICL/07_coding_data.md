# 07 Coding Data：从失败轨迹到训练样本

[← 06 Trajectory error](06_trajectory_error_prompt.md) · [下一章：系统与评测 →](08_systems_and_evaluation.md)

## 1. 先定义训练单元

一条可训练的 repository-level coding record 至少包含：

```text
repository snapshot + environment lock
issue/task specification
agent trajectory (actions, tool calls, observations)
candidate patch
executable verifier and exit status
failure taxonomy / derived rule
provenance and leakage metadata
```

只保留自然语言“反思”无法复现；只保留最终 patch 无法做 credit assignment；只保留 pass/fail 又无法判断 verifier 是否覆盖需求。

## 2. 闭环而非线性流水线

$$\mathcal D_t\to\mathrm{Solver}_t\to\mathrm{traces}_t\to\mathrm{skills}_t\to\mathrm{Generator}_t\to\mathrm{validated\ tasks}_{t+1}$$

Socratic-SWE 的 2026 预印本给出一个具体实例：从历史 solver traces 蒸馏 recurring failures / effective repair patterns，指导真实仓库中的 targeted repair-task generation，再经 execution validation 与 solver-gradient alignment 筛选。其摘要报告三轮后 SWE-bench Verified 50.40%；这是论文自身设置下的结果，不应写成通用 SOTA 或直接等同于 SFT。

## 3. 四种数据产物

### A. 成功 imitation

保留最短、可复现、测试通过的 trajectory。删除无效探索时要保留必要 observation，否则训练样本会产生“凭空知道答案”的捷径。

### B. 局部 repair pair

共享同一 prefix，构造失败动作 $a^-$ 与纠正动作 $a^+$：

$$q=(s,a^+,a^-),\qquad s=(\text{task},\text{history},\text{tool observations})$$

只有当 $a^+$ 在同一环境中经 verifier 通过、且差异可归因时，才适合作 preference data。

### C. 条件式 skill

采用 [06](06_trajectory_error_prompt.md) 的 `WHEN / DO / VERIFY` 结构，供检索或 instruction tuning。skill 必须带 evidence ids、反例和版本范围。

### D. verifier / adversarial task

把稳定、可执行的错误原则转成测试或静态检查；再生成恰好触发该 verifier 的新任务。这样规则成为环境反馈，而非模型自评。

## 4. 过滤闸门

按顺序执行，任何一步失败即隔离：

1. **snapshot reproducibility**：锁定 revision、依赖和测试命令；
2. **fail-to-pass**：原始代码失败、patch 后目标测试通过；
3. **pass-to-pass**：非目标回归测试仍通过；
4. **patch locality**：无测试删除、无跳过、无无关大改；
5. **trajectory faithfulness**：每个关键决定可由此前 observation 支持；
6. **deduplication**：按 issue、patch AST/语义和测试行为去重；
7. **contamination split**：同仓库近邻、回移 patch、模板变体不能跨 train/eval；
8. **difficulty replay**：用目标 solver 重跑，排除已过易题与不可解题。

## 5. 把 ICL 三条线映射到数据实验

| 视角 | 数据变量 | 干预 | 读数 |
|---|---|---|---|
| Bayesian | bug class / API contract coverage | 替换同类或异类示例 | calibration、pass rate |
| GD/kernel | query-demo 相似度与 residual 方向 | 反标签、复制、置换 | logit/prediction delta |
| Circuit | 标识符、格式、局部 key-value pattern | token-preserving rename、head patch | induction score、causal effect |
| Trajectory rule | condition/action/verifier | no-rule、random-rule、oracle-rule | pass、cost、negative transfer |

不要用 induction score 代替 repository correctness；它最多是局部机制 probe。

## 6. 相关工作如何准确使用

- [SWE-Gym](https://arxiv.org/abs/2412.21139)：面向 real-world software engineering agent learning 的训练环境与数据；具体规模和结果应引用对应版本表格，不在此泛化为“几万条”或“当前 SOTA”。
- [CYCLE](https://arxiv.org/abs/2403.18746)：训练模型利用 execution feedback 自我修正；原目录曾误链到 CodeGen2 (`2305.02309`)。
- [Self-Debugging](https://arxiv.org/abs/2304.05128)：让模型对生成程序进行解释、执行与反馈驱动的调试；注意原目录曾误链到 `2304.12243`。
- [Socratic-SWE](https://arxiv.org/abs/2606.07412)：trace-derived skills 驱动 task generation 的闭环；提交于 2026，不是 2025。

## 7. 最小上线顺序

1. 先固定 30–50 个 held-out tasks 与容器化 verifier。
2. 跑 baseline，人工审核首个可归因分叉。
3. 只抽取 3–5 条高频条件式 rules。
4. 先做 retrieval-time A/B；稳定后再生成 preference/SFT 数据。
5. 同时报质量和成本，遵循 [08](08_systems_and_evaluation.md)。

这比先堆大量自然语言原则更容易判断收益来自哪里，也更容易回滚。
