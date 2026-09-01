# 03 线 II：线性注意力与一步梯度下降

[← 02 Bayesian](02_line_I_bayesian.md) · [下一章：Circuit →](04_line_III_circuit.md)

## 1. 先统一形状和归一化

给 $k$ 个演示，$x_i\in\mathbb R^d$、$y_i\in\mathbb R^m$，参数 $W\in\mathbb R^{m\times d}$。采用 mean squared loss：

$$L(W)=\frac{1}{2k}\sum_{i=1}^{k}\lVert Wx_i-y_i\rVert_2^2$$

在 $W_0$ 处定义 residual $e_i=y_i-W_0x_i$，一步 full-batch GD 为：

$$W_1=W_0-\eta\nabla L(W_0)=W_0+\frac{\eta}{k}\sum_{i=1}^{k}e_ix_i^\top$$

对 query $x_q$：

$$W_1x_q=W_0x_q+\frac{\eta}{k}\sum_{i=1}^{k}e_i(x_i^\top x_q)$$

最后一项已经是 attention-like 聚合：key $k_i=x_i$，query $q=x_q$，value $v_i=e_i$，score $k_i^\top q$。因此无 softmax 的线性注意力修正为：

$$\Delta y_q=\frac{\eta}{k}V K^\top q=\frac{\eta}{k}\sum_{i=1}^{k}v_i(k_i^\top q)$$

代入 $v_i=e_i$ 即得到 $W_1x_q-W_0x_q$。这里的 $1/k$ 很重要：使用 mean loss 时，复制同一批示例不应让更新幅度线性爆炸。

## 2. “等价”究竟有多强

von Oswald et al. 给出线性 self-attention 的显式权重构造，使其数据变换等价于 regression loss 的一个 GD step。[T-construct] 他们还在简单回归任务上发现训练出的 attention-only 模型与该构造或 GD predictor 相近。[E-synthetic]

Akyürek et al. 进一步证明 Transformer 可构造实现 GD 与 closed-form ridge regression，并观察训练后的 ICL predictor 会随深度和噪声在 GD、ridge、least squares 等算法之间变化。[T-construct, E-synthetic]

所以正确表述是：

- **可以实现**：特定编码、projection、mask 与线性 attention 下可严格等价；
- **有时学到**：合成线性任务的训练模型可接近这些估计器；
- **尚未普遍证明**：标准 softmax、MLP、位置编码、自然语言 tokenization 下，每层都等于一次 GD。

## 3. softmax 为什么破坏精确恒等式

标准 attention 使用归一化正权重：

$$a_i=\frac{\exp(k_i^\top q/\sqrt{d_h})}{\sum_j\exp(k_j^\top q/\sqrt{d_h})},\qquad \Delta y_q=\sum_i a_iv_i$$

而 GD 需要可正可负、未必和为一的系数 $x_i^\top x_q$。额外 heads、feature lifting 或 value encoding 可以逼近更广的运算，但这已经不是上节的一行等式。

## 4. 深度、preconditioning 与其他估计器

“$L$ 层等于 $L$ 步 GD”只对明确的迭代构造成立。真实训练模型可能：

- 在不同层编码 moment matrices 或隐式参数；
- 学到 curvature correction / preconditioning，快于 plain GD；
- 近似 ridge 或 least squares，而不是固定学习率 GD；
- 使用 retrieval、pattern matching 或已有参数知识绕开拟合。

Garg et al. 证明的是标准 Transformer 能从头训练为多种函数类的 in-context learner，并与任务特定算法比较；它不是“层间轨迹与 GD 相关系数固定大于某数”的证据。

## 5. 可运行等价测试

[`icl_mechanisms.py`](icl_mechanisms.py) 同时计算：

1. 显式更新 $W_1$ 后的 $W_1x_q$；
2. residual-as-value、feature-as-key 的线性 attention 输出。

```bash
python3 -m unittest ICL.tests.test_icl_mechanisms.GradientDescentTests -v
```

测试覆盖多输出维度、mean-loss 的 $1/k$、样本顺序不变性与 shape failure。它没有声称 softmax attention 也满足等式。

## 6. coding-data 可证伪预测

若某任务主要近似 kernel/GD 式聚合，则：

- 顺序置换的影响应小于删除高相似度示例的影响；
- 复制所有示例在正确归一化后不应大幅改变预测；
- 反标签或 residual 符号翻转应产生方向一致的输出变化；
- 改变 query 与 demo 的表面相似度会系统改变权重。

若这些均不成立，不应继续用“隐式 GD”解释该任务。评测设计见 [08](08_systems_and_evaluation.md)。

来源：[von Oswald et al.](https://arxiv.org/abs/2212.07677) · [Akyürek et al.](https://arxiv.org/abs/2211.15661) · [Garg et al.](https://arxiv.org/abs/2208.01066) · [证据账本](references.md)
