# 04 线 III：回路实现 / Induction Head

Papers：Olsson et al. 2022, Elhage et al., Lieberum et al., Hendel / Todd 2023

## 什么在搬砖

Induction head 是两头复合：

- **Head 1 — previous-token head**：位置 $p$ 把信息搬到 $p+1$，形成 “[A] → 在 [A] 之后”
- **Head 2 — induction head**：查询“我的前面出现过 A 吗？”，attend 回之前那个 $A$ 的后一位 $B$

合成效果：

```
[A][B] ... [A] → 预测 B
```

对任意符号、词、函数名对都成立。

## 相变

小模型 1-2B token 时 induction 分数 ≈0，2.5-5B 时突然 0→0.8，对应 loss 一个小 bump（Olsson Fig.6）。这跟 few-shot 从 0 冒头完全一致，非渐进而是涌现。

## Function Vectors / Task Vectors

把 induction 抽象化：

```python
r̄_task = E[ r_L^{(demo)} ] - E[ r_L^{(random)} ]   # 层 L 残差流均值差
test时: r_L ← r_L + λ·r̄_task
# 零样本 + task vector ≈ 10-shot
```

说明 ICL 已把例子编译成一个向量程序。

其他要点：

- CoT = 长链上连续触发 induction
- copy-suppression head 负责过滤错误诱导

## 评测启示

- 单测 induction 精度，早于 overall loss 起飞，可作 emergent 早期信号
- 对抗失败：$A \neq A$（近似匹配）、copy-suppression 失灵
