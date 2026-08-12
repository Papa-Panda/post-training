# 01 定义 + 时间线

## 定义

ICL 最紧定义：

$$
p_\theta(y \mid \text{prompt}) = p_\theta(y \mid (x_1,y_1),..., (x_k,y_k), x_q)
$$

权重 $\theta$ 不动，靠 prompt 里的 $k$ 个演示现场学会任务 $f: x \to y$。广义的 context 还包括 instruction、检索文档、CoT 中间步、工具返回。

典型 prompt 形态：

```
[Instruction: Translate English to Chinese]
x1: dog -> y1: 狗
x2: cat -> y2: 猫
x_q: bird -> ?
```

模型要做的就是接着写出 `y_q`。不改权重也能做，是因为预训练见过足够多的 $(x,y)$ 分布模式。

## 时间线

- **2019 GPT-2**：zero-shot 已现，未命名。
- **2020.05 GPT-3**：*Language Models are Few-Shot Learners*，ICL 正式入词，成为大模型标志能力。
- **2021.11 Xie et al.**：*An Explanation of In-Context Learning as Implicit Bayesian Inference*，把预训练看成主题模型混合，ICL = 对潜概念 $c$ 做后验积分。
- **2022.06 von Oswald / Akyürek** 并行：*Transformers learn in-context by GD* / *What learning algorithm is in-context*，证明线性注意力前向就是一步 GD。
- **2022.08 Olsson et al.**：*In-context Learning and Induction Heads*，mechanistic 视角找到真正搬运 token 的两层复合电路。
- **2023**：Dai Dual Form / Garg 理论 / Hendel Task Vectors / Todd Function Vectors，把三线串起来。
- **2024-25**：many-shot 1000+ demo、in-context RL、CoT 证明：ICL = 元学习的前向近似。

## 数学深度排序

线 II 最可算（可构造性证明）> 线 I 次之（需 HMM 假设）> 线 III 最可视但最难泛化。做 coding data 评测时，你每天改的就是这三条线的边界条件。
