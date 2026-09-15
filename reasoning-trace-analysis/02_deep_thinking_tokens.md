# 02 — Deep-Thinking Tokens: 不看长度，看"想了几层"

> 长度是最容易得到的 trace 统计量，也是最不可靠的效率信号。"Think Deep, Not Just Long"（Wei-Lin Chen 等，2026）提出 deep-thinking token：不数 token 有多少，而看每个 token 的预测分布在模型的多少层里被持续修正。
>
> Length is the cheapest trace statistic and the least reliable efficiency signal. "Think Deep, Not Just Long" (Wei-Lin Chen et al., 2026) proposes deep-thinking tokens: instead of counting tokens, measure for each token how many layers keep revising its prediction distribution.

## 1. 数学：settling depth 的定义

记模型有 $L$ 层。对生成位置 $t$ 、层 $l$ ，把该层的 hidden state 用**同一个** unembedding 矩阵投影到词表，得到分布 $p_{t,l}$ 。记最终层分布为 $p_{t,L}$ 。层 $l$ 的预测与最终预测的差异用 Jensen-Shannon 散度度量：

$$D_{t,l}=\mathrm{JSD}(p_{t,L}\,\|\,p_{t,l}).$$

取单调包络（防止抖动）：

$$\bar{D}_{t,l}=\min_{j\le l} D_{t,j}.$$

**settling depth** 是包络首次低于阈值 $g$ 的层：

$$c_{t}=\min\{l:\bar{D}_{t,l}\le g\}.$$

直觉： $c_{t}$ 小 = 浅层就已经"想定"，后面各层只是照抄； $c_{t}$ 大 = 直到深层还在大幅修正预测——这个 token 经历了真正的 depth-wise revision。

一个 token 是 **deep-thinking token**，当且仅当 $c_{t}\ge\lceil\rho L\rceil$ （ $\rho$ 为深层比例超参）。整条 trace 的 **deep-thinking ratio (DTR)**：

$$\mathrm{DTR}(S)=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[c_{t}\ge\lceil\rho L\rceil].$$

## 2. 实证：DTR 比长度更能预测准确率

论文在 AIME 2024、AIME 2025、HMMT 2025、GPQA-Diamond 四个 benchmark 上，对 GPT-OSS、DeepSeek-R1、Qwen3 三个模型族比较了 DTR 与长度、confidence 等基线。核心发现：DTR 与准确率稳健正相关，显著优于长度。

Figure 1 的具体数字（GPT-OSS-120B-medium，四 benchmark 平均）：输出长度与准确率的相关系数 $r=-0.544$ （越长反而越差——典型的 overthinking 信号），DTR 与准确率的相关系数 $r=0.828$ 。注意这是**该图示的特定数值**，不是普适常数；可迁移的结论是"DTR 优于长度"这个序关系。

## 3. 系统：Think@n——把 DTR 变成 inference 策略

DTR 不只是分析指标，论文把它做成了采样策略 **Think@n**：

1. **选高 DTR 样本**：并行采样 $n$ 条 trace，选 DTR 最高者，而不是选多数投票（self-consistency）或最短者。
2. **短 prefix 估计 + 早停**：用 trace 开头的一小段估计整条的 DTR，不 promising 的 generation 提前终止，省掉后面的 token。

效果：达到或超过 standard self-consistency 的准确率，只用约一半的 inference cost。工程含义：DTR 是一个**可在线估计**的 trace 质量信号——不需要等整条 trace 生成完，也不需要知道正确答案。

代价是白盒需求：要拿到每一层的 hidden state 并做 unembedding 投影。黑盒 API 场景用不了，这是 DTR 相对于纯文本信号（04、05 章）的根本限制。

## 4. 与熵信号的关系（预告 03）

DTR 和 token entropy 都是"不确定性"的度量，但量的对象不同：

- **熵** $H_{t}$ ：固定一层（通常是最终层），看词表分布有多平——"这一刻模型有多犹豫"。
- **JSD 轨迹** $D_{t,l}$ ：固定一个位置，看各层预测如何收敛——"这个决定在网络深度上被修改了多少次"。

高熵 token 不一定是 deep-thinking token（可能浅层就定了，只是词表本来就平）；deep-thinking token 也不一定高熵。两个信号互补，不是谁替代谁。

## 5. 代码对应

`code/trace_lab.py` 的 (a) 部分：

- `jsd_trajectory(tau, n_layers)` ：用解析衰减曲线 $d_{0}e^{-l/\tau}$ 代替真实的 unembedding 投影。 $\tau$ 大 = 收敛慢 = deep-thinking token 的玩具类比。
- `settling_depth` ：与论文定义逐字对应（含 min-envelope）。
- `deep_thinking_ratio` ： $g=0.15$ 、 $\rho=0.8$ 默认值。
- `run_correlation_experiment` ：模拟"准确率由 DTR 驱动、长度掺入 filler 噪声"的生成过程，复现论文 Figure 1 的定性结论：DTR 强正相关、长度弱相关。

`tests/test_trace_lab.py` 断言了语义不变量：轨迹单调递减、慢收敛 token 的 settling depth 更大、DTR 随 thinking 比例单调增、DTR 对准确率的预测力优于长度。
