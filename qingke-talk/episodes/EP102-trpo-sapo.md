# EP102 — 从 TRPO 到 SAPO：大模型 RL 算法演进

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP102-trpo-sapo.html

> "While all methods may ultimately exhibit signs of instability, SAPO sustains coherent learning for a longer duration and reaches higher Pass@1 accuracy before divergence." —— SAPO 论文 §1 对受控实验的总结

## 元信息

- 期号：102（官网期，无 B站视频）
- 标题：从 TRPO 到 SAPO：大模型 RL 算法演进
- BV：无——B站合集无对应视频（注意编号冲突：B站合集第 102 期是《Megrez2：更轻更快，但更强的推理VLM》，与本期无关）
- 直播时间：2026-01-10 10:00–11:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：高畅（SAPO 作者、通义千问算法工程师，香港中文大学博士，Qwen3、Qwen3-VL 系列模型核心贡献者；官网预告嘉宾介绍）
- 相关论文：Chang Gao, Chujie Zheng, Xiong-Hui Chen 等（Qwen Team, Alibaba），*Soft Adaptive Policy Optimization*，https://arxiv.org/abs/2511.20347（v2, 2025-12-01）
- 相关代码：论文未附公开代码链接；SAPO 已用于 Qwen3-VL 系列模型的实际训练（论文 §5.2）
- 官网预告：https://qingkeai.online/blog/SAPO

> ⚠️ 提炼方式说明：本期为官网期，B站合集无对应视频，字幕无从获取。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ SAPO 论文原文（arXiv:2511.20347）。演进主线（TRPO→PPO→GRPO→GSPO）以官网预告的讲授提纲为准，SAPO 的公式与实验结论以论文为准（预告文中的个别公式与论文不一致时从论文）。AMA 环节与现场发挥未覆盖。

## 一句话总结

这期是一条算法演进史：从 TRPO 的信任域思想出发，经 PPO 的硬裁剪、GRPO 的去 critic、GSPO 的序列级比率，走到 SAPO——用温度可控的 sigmoid 软门控替代"非黑即白"的硬裁剪，并对正/负优势 token 用非对称温度，让 off-policy 程度不同的 token 得到连续衰减的梯度权重，从而同时保住序列级一致性与 token 级自适应性，在 Qwen3-30B-A3B 与 Qwen3-VL 训练中比 GSPO/GRPO 更稳、终点更高。

## 核心

### 背景/问题：一切围绕"信任域怎么落地"

按官网预告的提纲，讲授主线是四个阶段：

1. **TRPO（理论基石）**：用 KL 散度约束把策略更新限制在信任域内，保证单调改进；但需要二阶信息，无法直接用于大模型。
2. **PPO（实用近似）**：用裁剪（clipping）把重要性比率 $r_t(\theta)$ 限制在 $1-\epsilon$ 到 $1+\epsilon$ 之间，一阶近似信任域；代价是需要同时训练价值模型。
3. **GRPO（去 critic）**：同一 prompt 采样一组响应，用组内奖励的均值与标准差构造优势 $\hat{A}_i$ ，省掉价值模型；但 token 级重要性比率方差大（论文 §1：MoE 模型中路由异质性与长响应会进一步放大），硬裁剪陷入两难——裁紧了有效样本太少，裁松了 off-policy 噪声梯度进来。
4. **GSPO（序列级比率）**：把比率换成长度归一化的序列似然比 $s_i(\theta)$ ，在序列粒度上裁剪，MoE 上更稳；但一条序列里只要有少数高度 off-policy 的 token，整条序列的梯度都被压掉，样本效率受损（官网预告把 GSPO 的局限概括为"硬裁剪浪费有效样本"）。

SAPO 要解决的正是这个两难：**硬裁剪对 off-policy 样本的处理是 0/1 式的，有效学习信号和噪声一起被扔掉**。

### 方法/设计：软门控 + 非对称温度

SAPO（Soft Adaptive Policy Optimization）保留 GRPO 的分组优势估计，只替换梯度加权方式（论文 §3，Eq.5–6）。目标为：

$$\mathcal{J}(\theta)=\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}f_{i,t}\big(r_{i,t}(\theta)\big)\,\hat{A}_{i,t}\Big]$$

其中软门控函数为：

$$f_{i,t}(x)=\sigma\big(\tau_{i,t}(x-1)\big)\cdot\frac{4}{\tau_{i,t}}$$

这里 $\sigma$ 是 sigmoid 函数， $r_{i,t}(\theta)$ 是 token 级重要性比率，温度 $\tau_{i,t}$ 按优势符号取 $\tau_{pos}$ 或 $\tau_{neg}$ 。求导后每个 token 的梯度权重是 $w_{i,t}(\theta)=4p_{i,t}(\theta)\big(1-p_{i,t}(\theta)\big)$ ，其中 $p_{i,t}(\theta)=\sigma\big(\tau_{i,t}(r_{i,t}(\theta)-1)\big)$ （Eq.7–8）：权重在 $r_{i,t}(\theta)=1$ （on-policy 点）取峰值 1，向两侧近似指数衰减——这是一个**连续信任域**：偏离越远权重越小，但不像硬裁剪那样直接归零。

三个关键设计判断：

- **非对称温度是必需的**：论文 §3 从 logit 梯度传播论证——负优势更新会同时抬高大量未采样 token 的 logit，比正优势更新更容易引入不稳定，因此令 $\tau_{neg}>\tau_{pos}$ ，让负 token 的梯度衰减更快。
- **序列一致性自动涌现**：在小步更新、序列内 token log-ratio 离散度低的常见条件下，token 门控的平均会收敛为一个光滑的序列级门控（ $\mathrm{sech}^2$ 形式），退化为 GSPO 式的序列优化但信任域连续；条件不满足时则逐 token 只压"出格"的 token，保住同序列里 near-on-policy token 的信号。
- **不需要 routing replay**：对照的 GRPO-R2 是"GRPO + routing replay"（为稳定 MoE 训练引入的工程手段），SAPO 不依赖它即可稳定，论文明确这降低了 RL 系统的工程开销。

### 实验/实战（论文 §5）

**受控实验（§5.1）**：从 Qwen3-30B-A3B-Base 冷启动、在数学推理数据上微调，评测 AIME25、HMMT25、BeyondAIME（均为 16 采样平均 Pass@1）。SAPO 取 $\tau_{pos}=1.0$ 、 $\tau_{neg}=1.05$ 。结果（论文以训练曲线呈现，Fig.4）：GSPO 与 GRPO-R2 都出现**早期训练崩溃**，SAPO 全程稳定且最终性能更高。温度消融（Fig.5）： $\tau_{neg}=1.05>\tau_{pos}=1.0$ 最稳， $\tau_{neg}=\tau_{pos}=1.0$ 次之， $\tau_{neg}=0.95<\tau_{pos}$ 显著不稳定——非对称方向反了直接翻车。

**Qwen3-VL 实际训练（§5.2）**：三种算法从同一个 Qwen3-VL-30B-A3B 冷启动 checkpoint 出发，在文本+多模态混合任务（数学、代码、逻辑推理）上训练，评测 AIME25（Pass@1，32 采样）、LiveCodeBench v6（8 采样）、ZebraLogic、MathVision。SAPO 在同等算力预算下训练奖励与四个验证基准全面优于 GSPO 与 GRPO-R2（Fig.6）；论文称 SAPO 在不同规模、MoE 与 dense 架构上均有一致收益，并已用于 Qwen3-VL 系列的实际训练。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 温度设置 $\tau_{pos}$ / $\tau_{neg}$ | — | 1.0 / 1.05 | 论文 §5.1 |
| 温度方向消融 $\tau_{neg}=0.95<\tau_{pos}$ | $\tau_{neg}=1.05$ 最稳 | 显著不稳定 | 论文 §5.1、Fig.5 |
| 早期训练崩溃 | GSPO、GRPO-R2 均出现 | SAPO 未出现、全程稳定 | 论文 §5.1、Fig.4 |
| Routing replay 依赖 | GRPO-R2 需要 | SAPO 不需要 | 论文 §5.1 |
| 受控实验底座与评测 | Qwen3-30B-A3B-Base 冷启动 | AIME25 / HMMT25 / BeyondAIME，16 采样平均 Pass@1 | 论文 §5.1 |
| Qwen3-VL-30B-A3B 对比 | GSPO、GRPO-R2 | SAPO 同等算力下四基准全面更优 | 论文 §5.2、Fig.6 |

注：论文实验以训练曲线形式报告终点对比，未给出单一终点数值表，上表只写文中可确证的设置与定性结论，不抄图估数。

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **把硬裁剪换成软门控是低侵入改动**：GRPO/GSPO 框架里只需把 clip 换成 sigmoid 门控加权（峰值归一化到 on-policy 点为 1），即可回收被整条裁掉的样本信号；在 MoE 或长响应场景下尤其值得试。
  2. **负优势 token 单独更强的衰减**： $\tau_{neg}>\tau_{pos}$ 这个非对称设计有明确的 logit 梯度论证，且方向反了会显著不稳定——调参时先固定方向再调幅度。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. SAPO 不依赖 routing replay 就能稳定 MoE 训练，直接减少 RL 系统里一类工程补丁；评估新算法时"少一个稳定化 trick"本身就是infra 成本收益。
  2. 连续信任域让"有效样本数"随训练平滑变化，监控侧可以盯门控权重的分布（均值/尾部）作为 off-policy 程度的实时探针，比只看 clip fraction 更细。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：官网预告还提到 BAPO（动态裁剪边界）与 DeepSeek-V3.2 的序列掩码等同期改进，论文未做对比——软门控与"动态边界/掩码过滤"在同一训练设置下谁更稳、能否叠加，值得找复现实验核对。
- 现场内容不可还原：TRPO 部分的讲授深度、AMA 问答无法从论文推知；若 B站后续补录视频，应以视频为准修订本纪要。
- 论文终点数值只在曲线图里：如需精确终点 Pass@1，需读 Fig.4/Fig.6 原图估读，本纪要未做。

## 原文金句（1-2句）

> "SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency." —— 论文 Abstract，软门控相对 GSPO 整条压制的核心区别

> "negative updates tend to increase the logits of many inappropriate tokens and are therefore more prone to introduce instability than positive updates." —— 论文 §1，非对称温度设计的第一性理由
