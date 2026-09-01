# 05 三条解释线：统一坐标系，而非单一路径

[← 04 Circuit](04_line_III_circuit.md) · [下一章：Trajectory error →](06_trajectory_error_prompt.md)

## 1. 比较表

| 问题 | Bayesian | GD / estimator | Circuit / vector |
|---|---|---|---|
| 解释对象 | 输出分布与潜任务不确定性 | 激活中可能执行的学习算法 | 信息由哪些组件搬运与表示 |
| 典型形式 | $p(c\mid D_k)$ | $W_{t+1}=W_t-\eta\nabla L(W_t)$ | $[A][B]\ldots[A]\to[B]$ |
| 最强证据 | mixture-of-HMM 条件定理 | 线性 attention 显式构造 | 小 attention-only 模型的因果干预 |
| 主要外推风险 | 真实 prompt 非 IID、模型错设 | softmax/语言任务不满足线性构造 | 复制电路不等于抽象任务学习 |
| coding 用途 | demo 覆盖、冲突与校准 | 构造 query-demo 影响实验 | 定位复制、格式、局部关联机制 |

## 2. 可以统一的部分

在线性 Gaussian regression 中，三种语言确实可能描述同一个 predictor：Gaussian prior 的 posterior mean 对应 ridge；Transformer 可构造近似 ridge 或迭代优化；激活中必须有组件编码充分统计量并把它作用到 query。

例如，若 $w\sim\mathcal N(0,\lambda^{-1}I)$ 且 $y_i=w^\top x_i+\varepsilon_i$，$\varepsilon_i\sim\mathcal N(0,\sigma^2)$，posterior mean 为：

$$\widehat w=(X^\top X+\lambda\sigma^2 I)^{-1}X^\top y$$

它既是 Bayesian posterior mean，也是 ridge 解；GD 可迭代逼近它。这个交集是真实的，但只在假设明确时成立。

## 3. 不能直接画等号的部分

- latent concept $c$ 不必等于某层的线性 task vector；
- attention value 不普遍等于 loss gradient；
- induction head 做 match-and-copy，不自动实现矩阵逆或 posterior marginalization；
- activation injection 有因果效应，不证明该方向是唯一、完整或自然运行时必需的表示；
- CoT、many-shot、in-context RL 各自需要独立定义和证据，不能仅凭类比归入三线。

更安全的统一方式是把三条线当成**三个可相互约束的模型族**：Bayes 给校准与证据累积预测，GD 给扰动方向与迭代结构预测，circuit 给可定位、可干预的实现预测。

## 4. 一组能区分机制的实验

给固定任务与查询，构造以下干预：

| 干预 | Bayesian 重点 | GD 重点 | induction 重点 |
|---|---|---|---|
| 重复同一 demo | 证据非独立时不应盲目增信 | mean loss 下更新不变 | 可能增强重复 pattern |
| 置换顺序 | IID 后验不变 | full-batch sum 不变 | positional/recency 可变 |
| 翻转标签 | 更新 likelihood | residual 符号改变 | 复制错误 successor |
| 换表面格式、保留映射 | 概念不变但观测模型变 | feature/kernel 改变 | token match 大幅改变 |
| 加无关长上下文 | 理想后验可忽略 | 理想估计器可忽略 | attention dilution 与系统成本上升 |

不要只看最终 accuracy。同步记录预测分布、置信度、每个顺序 seed、token 数、prefill/decode 时间和显存，见 [08](08_systems_and_evaluation.md)。

## 5. 面向 coding data 的决策规则

- 若目标是**选示例**：优先 Bayesian 的覆盖/区分视角。
- 若目标是**诊断 query-demo 影响**：用 GD/kernel 式扰动作为可证伪基线。
- 若目标是**理解局部复制或格式遵循**：做 induction probe 与 causal intervention。
- 若目标是**改进 agent repair**：不要停在机制类比；进入 [06](06_trajectory_error_prompt.md) 和 [07](07_coding_data.md) 的可执行数据闭环。

来源与证据等级统一列在 [references.md](references.md)。
