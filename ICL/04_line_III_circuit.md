# 04 线 III：Induction Circuit 与任务向量

[← 03 GD](03_line_II_gd.md) · [下一章：比较 →](05_comparison.md)

## 1. 两层 match-and-copy

对重复片段：

```text
... [A] [B] ... [A] -> [B]
```

经典 induction circuit 可分成：

1. **previous-token head**：在位置 $j$ 的 residual stream 写入前一 token $t_{j-1}$ 的信息；
2. **induction head**：当前位置 $i$ 的 query 与旧位置 $j$ 携带的 $t_{j-1}$ 匹配；若 $t_{j-1}=t_i$，则从位置 $j$ 的 value 复制 $t_j$ 的信息。

一个理想化 prefix-match score 可写为：

$$S_{\mathrm{prefix}}=\frac{1}{|Q|}\sum_{i\in Q}\sum_{j<i}{\bf 1}[t_{j-1}=t_i]a_{ij}$$

其中 $a_{ij}$ 是从当前位置 $i$ 指向历史位置 $j$ 的 attention weight。高分表示 attention 落到“之前相同前缀之后的 token”，但单独高分不是任务因果性证明。

## 2. 证据边界

Olsson et al. 报告 induction heads 的形成与其定义的 in-context loss 改善同时出现。[E-synthetic] 对小型 attention-only 模型，他们提供较强因果证据；对含 MLP 的较大模型，论文明确称证据主要是相关性的。[C-causal / E-model]

因此不应写成：

- 任意 token 对都必然由 induction head 完成；
- 某个固定训练-token 区间会从 0 跳到固定分数；不同模型曲线不同；
- CoT 的每一步都是 induction；
- induction head 足以解释需要生成未在上下文出现的新答案的 ICL。

Induction 最直接解释的是 pattern completion / associative retrieval。算法组合、抽象规则与长程规划通常还需要其他组件。

## 3. 从 attention pattern 到因果检验

证据从弱到强：

1. **attention visualization**：看到了预期 pattern；
2. **activation correlation**：head output 与任务标签相关；
3. **ablation**：移除 head 后目标能力选择性下降；
4. **activation patching**：从 clean run 注入 corrupt run，恢复目标输出；
5. **sufficiency intervention**：把提取的表示注入 zero-shot / 新格式，诱发目标函数。

必须同时做随机 head、相同范数方向、无关任务向量等 controls，否则“中间层有效”可能只是通用扰动敏感。

## 4. Task vectors 与 function vectors 不应混成一个公式

Hendel et al. 把 ICL 看作将示例集 $D_k$ 压缩为 query-agnostic task vector $\tau(D_k)$，再与 query 一起驱动预测：[E-model]

$$\widehat y=f_\theta(x_q;\tau(D_k))$$

Todd et al. 先用 causal mediation 找出对 ICL 任务重要的少量 attention heads，再聚合这些 head 的平均输出形成 function vector，并做跨 context 注入。[C-causal] 这不同于简单的“demo residual 均值减 random residual 均值”。

可靠结论是：某些任务和模型中，可提取紧凑、具有因果效应的 activation direction。它不等于通用程序字节码，也不保证与任意格式、组合任务或模型之间可移植。

## 5. coding 场景的合适测试

### 适合 induction probe

- 重复标识符后的局部 token；
- paired delimiters 与格式模板；
- 先前定义的短 API pattern；
- 明确复制型 key-value mapping。

### 不足以只测 induction

- patch 是否通过隐藏测试；
- 跨文件 control/data flow；
- API 语义与版本兼容；
- 从失败证据归纳新的 repair strategy。

[`induction_distribution`](icl_mechanisms.py) 给出 exact-match 的经验 successor 分布，只作为电路功能的离散基线。真实模型 probe 还应记录 tokenization、layer/head、attention mask、干预位置和 effect size。

来源：[Olsson et al.](https://arxiv.org/abs/2209.11895) · [Hendel et al.](https://arxiv.org/abs/2310.15916) · [Todd et al.](https://arxiv.org/abs/2310.15213) · [证据账本](references.md)
