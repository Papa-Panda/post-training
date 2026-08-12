# 05 三线比分 + 统一视角

## 比分表

| 维度 | 线 I Bayes | 线 II GD | 线 III Circuit |
|---|---|---|---|
| 数学硬度 | KL 可区分数学，需 HMM 假设 | **可构造证明，线性等同** | 实证电路追踪，难推广 |
| 预测力 | $k$ 对数/幂律、many-shot 单调 | 层数≈步数、需 Norm 防溢出 | 2-5B token 相变、CoT 依赖搬运 |
| 失效模式 | 顺序、噪声标签 | softmax 非线性、超长 k 发散 | 对抗性 $A \neq A$、suppression 失灵 |
| 对 coding data 最有用 | 设计 demo：选代表性 $c$，控先验 | 调格式：让 KV 对齐梯度 | 评：关注 induction 分数 |

## 合并读法（给你做 infra 时用）

三线是统一现象在不同抽象层的投影：

- **Bayes 是目标**（ICL 想推断什么）：$p(c \mid \text{prompt})$
- **GD 是算法**（前向如何用一步加法逼近推断）：$\Delta W = \eta \sum e_i x_i^\top$
- **Circuit 是硬件**（梯度加法由哪两个头搬运）：$[A][B]...[A]\to B$

所以 Xie 的 $p(c\mid\text{prompt})$，在 von Oswald 视角是梯度隐式编码的 $c$，在 Olsson 视角是 task vector $= \sum attention\cdot value$。

做 post-training 时：

- 数据侧：按潜概念 $c$ 平衡（Xie）
- 格式侧：让 demo 结构利于 $V K^\top Q$ 累加（GD）
- 评测侧：单独测 induction 精度（Circuit）

## 延伸

- **CoT**：每步思维都是对前一步的 induction，模长越长 = 多步 GD
- **Many-shot** 1024 例：性能 $\propto \log k$，符合 Bayes 后验 $1/k$ 收缩，但受限 GD Norm 溢出
- **In-Context RL**：prompt 放 reward，Transformer 前向做 policy improvement，本质同一回路对 `[state][action][reward]` 的 $A\to B$

给 daily 的可落地假设：设 coding 任务潜概念 $c$ = {lang, pattern, edge-case}，若想 2-shot 稳超 0-shot，需 $k > H(p(c))/I(demo;c)$。实测 Python 纠错高熵需 $k\ge5$，固定模板 API $k=2$ 已够。你在 `ai-data/` 里给不同类型数据设 shot 预算会更省。

参考：
- Xie 2021 https://arxiv.org/abs/2111.02080
- von Oswald 2022 https://arxiv.org/abs/2212.07677
- Olsson 2022 https://arxiv.org/abs/2210.12285
