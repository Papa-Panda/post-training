# 02 线 I：隐式贝叶斯 — ICL 是主题后验

Paper：Xie et al. 2021 *An Explanation of In-Context Learning as Implicit Bayesian Inference*

## 生成假设

预训练语料由潜概念 $c \sim p(c)$ 生成文档 $D \sim p(o \mid c)$。$c$ 可以想成“任务/主题/编程语言+风格”。

## 后验推导

给定 prompt（含 $k$ 个 demo + query $x_q$）：

$$
p(c \mid \text{prompt}) \propto p(c) \prod_{i=1}^k p(x_i,y_i \mid c) \cdot p(x_q \mid c)
$$

$$
p(y \mid \text{prompt}) = \int p(y \mid x_q, c) \, p(c \mid \text{prompt}) \, dc \approx p(y \mid x_q, c_{MAP})
$$

prompt 越长，后验越尖，$p(c_{MAP} \mid \text{prompt}) \to 1$ 指数快。

## 定理形态（可区分性）

若不同 $c$ 的分布 KL 可区分：$KL(p(\cdot \mid c) \parallel p(\cdot \mid c')) \ge \delta$，则需要

$$
k = \Omega\!\left(\frac{1}{\delta} \log \frac{1}{p(c^*)}\right)
$$

才能把正确概念 $c^*$ 认出来。直白：**先验越小、概念越像，越需要更多 demo**。解释了小众代码规范要 5-8 例才稳。

## 预测 vs 实测

| 预言 | 是否成立 | 比分 |
|---|---|---|
| demo 越多越好，单调 | 大致成立，many-shot 1024 例继续涨 | 3/3 |
| demo 顺序无关 | 大失败，长尾任务末位偏置严重 | 0/3 |
| 标错标签应被忽略 | Min et al. 2022 证实不忽略，掉 10-20pt | 1/3 |

## 修正

引入翻转概率 $\epsilon$ 的 noisy channel：

$$
p(y_i \mid x_i, c) = (1-\epsilon) \delta_{f_c(x_i)} + \epsilon \cdot \text{Uniform}
$$

这才解释“错误示范会学坏”。

## 对 coding data 的启示

- 设计 demo 时按潜在概念 $c$ = {lang, pattern, edge-case} 平衡，控制先验 $p(c)$
- 高熵任务（Python 纠错）互信息 $I(demo;c)$ 小 → 需 $k \ge 5$；低熵任务（固定模板 API 调用）$k=2$ 已够
