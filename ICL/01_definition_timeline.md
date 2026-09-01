# 01 定义、问题设置与证据层级

[← README](README.md) · [下一章：Bayesian 视角 →](02_line_I_bayesian.md)

## 1. 最小定义

令模型参数为固定的 $\theta$，演示集合按顺序写成 $D_k=((x_1,y_1),\ldots,(x_k,y_k))$，查询为 $x_q$。ICL 研究的是：

$$p_\theta(y_q\mid D_k,x_q)$$

模型在推理时不改变 $\theta$，但其激活与输出分布随 $D_k$ 改变。需要区分：

- **zero-shot instruction following**：没有输入输出示例；
- **few-shot ICL**：上下文中有少量示例；
- **retrieval augmentation**：上下文提供事实，不一定提供映射规则；
- **test-time training**：推理阶段真的更新参数，不属于上述最小定义；
- **agent memory**：过去轨迹被检索进上下文时可产生 ICL，但 memory 写入本身是外部状态更新。

## 2. 三个不同问题

同一个行为“看过示例后答得更好”至少包含三个层次：

1. **统计目标**：模型是否在推断潜在任务 $c$，并近似后验预测？见 [02](02_line_I_bayesian.md)。
2. **算法实现**：模型是否在激活中实现 GD、ridge、least squares 或别的估计器？见 [03](03_line_II_gd.md)。
3. **物理机制**：哪些 heads / MLP / residual-stream directions 搬运任务信息，干预它们是否改变输出？见 [04](04_line_III_circuit.md)。

回答一个层次不自动回答另外两个。例如，“线性 self-attention 存在一组参数等价于一步 GD”是可构造性结论，不足以证明自然语言模型对任意任务都执行 GD。

## 3. 可检验的 ICL 增益

给定同一查询集，至少比较：

$$\Delta_k=M(D_k)-M(D_0)$$

其中 $M$ 是预先冻结的任务指标。只报 $\Delta_k$ 仍不够；还需做四组对照：

- **label shuffle**：保留输入、标签空间和格式，破坏映射；
- **input shuffle / irrelevant demos**：测试语义相关性；
- **order permutation**：测试 recency 与位置敏感性；
- **format-only**：只保留模板，测试输出空间和格式贡献。

Min et al. 在一组分类与多选任务上发现，随机替换 demonstration labels 往往只造成很小损失；其正面结论是 label space、输入分布和格式很重要，而不是“错误标签普遍造成 10–20 点下降”。这也说明 ICL 增益必须拆因。

## 4. 时间线（只保留可核验节点）

- **2020**：[GPT-3](https://arxiv.org/abs/2005.14165) 系统比较 zero/one/few-shot；任务与 demonstrations 仅通过文本给出，不进行任务特定的梯度更新或微调。
- **2021–2022**：Xie et al. 在 mixture-of-HMM 理论设置中解释潜概念推断，并在 GINC 合成数据上展示顺序敏感等现象。
- **2022–2023**：Garg et al. 建立函数类 ICL testbed；Akyürek et al. 与 von Oswald et al. 给出线性模型中学习算法/梯度步骤的构造和实验。
- **2022**：Olsson et al. 研究 $[A][B]\ldots[A]\to[B]$ induction heads；小 attention-only 模型证据较强，大模型证据主要相关性。
- **2023–2024**：task vectors 与 function vectors 用 activation extraction、patching 和 causal mediation 检验任务信息的紧凑表示。

“CoT 已证明等于多步 GD”或“ICL 已被统一证明为前向元学习”都超出这些论文的证据，本文不采用。

## 5. 证据强度标签

后续章节统一使用：

- **[T-construct]**：存在性/构造性证明；
- **[T-bound]**：在明确假设下的理论界；
- **[E-synthetic]**：合成任务实验；
- **[E-model]**：真实预训练模型实验；
- **[C-causal]**：ablation、patching 或 mediation 的因果干预；
- **[H]**：待验证假说。

完整出处见 [References](references.md)。
