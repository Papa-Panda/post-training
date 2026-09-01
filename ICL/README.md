# ICL — In-Context Learning：理论、机制与工程

ICL 指冻结参数的自回归模型，仅通过上下文中的任务描述与示例改变当前预测：

$$p_\theta(y_q\mid D_k,x_q),\qquad D_k=((x_1,y_1),\ldots,(x_k,y_k))$$

这里的“学习”发生在前向计算与激活中，不等于优化器更新了模型参数。这个目录把三条常被混写的解释分开，并要求每条结论标明适用范围：

1. **Bayesian**：描述应该推断什么——潜在任务或概念的后验。
2. **Gradient descent / estimator**：描述某些模型问题中可实现或学到的算法。
3. **Circuit**：定位具体信息搬运与因果组件。

它们可以相容，但目前没有证据证明真实语言模型总是按“Bayes 目标 → GD 算法 → induction-head 硬件”这一条唯一链路工作。

## 学习路径

| 顺序 | 内容 | 读完应能回答 |
|---|---|---|
| [01](01_definition_timeline.md) | 定义与证据层级 | 哪些现象算 ICL，哪些只是检索或提示遵循？ |
| [02](02_line_I_bayesian.md) | 隐式 Bayesian 推断 | 后验赔率为何随证据累积？原论文究竟证明到哪里？ |
| [03](03_line_II_gd.md) | 线性注意力与 GD | 一步 GD 如何严格改写成 attention-like 求和？ |
| [04](04_line_III_circuit.md) | induction heads 与 task/function vectors | 关联、干预与充分性有什么区别？ |
| [05](05_comparison.md) | 统一坐标系 | 三条线如何互补，哪里不能直接等同？ |
| [06](06_trajectory_error_prompt.md) | 轨迹错误到规则 | 怎样避免把一次反思当成可泛化规则？ |
| [07](07_coding_data.md) | coding-data 闭环 | 如何把失败轨迹变成可执行、可归因的数据？ |
| [08](08_systems_and_evaluation.md) | 系统成本与评测 | shot 数增加时，质量、延迟、显存如何一起测？ |
| [References](references.md) | 主来源证据账本 | 每项核心主张由哪篇原始论文支持？ |

## 可运行最小模型

[`icl_mechanisms.py`](icl_mechanisms.py) 不模拟完整语言模型，而是把三条线最容易混淆的代数做成可执行规范：有限概念后验、一步全批量 GD 与线性注意力等价、exact-match induction、KV-cache/attention-score 计数。

```bash
python3 ICL/icl_mechanisms.py
python3 -m unittest discover -s ICL/tests -v
```

## 阅读纪律

- “存在一组权重可实现”不等于“预训练模型实际学到了同一算法”。
- synthetic regression / associative recall 的结论不能无条件外推到自然语言或代码。
- 相关性、activation patching、ablation、端到端收益是不同证据强度。
- 示例更多不保证单调变好；顺序、格式、标签语义、噪声和上下文预算都会改变结果。
- 所有系统结论都应同时报告质量、总 token、prefill 延迟、decode 延迟和峰值显存。
