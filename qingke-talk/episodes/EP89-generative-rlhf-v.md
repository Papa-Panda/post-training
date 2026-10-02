# EP89 — Generative RLHF-V：面向多模态 RLHF 的人类意图对齐框架

> 📖 阅读版：https://papa-panda.github.io/post-training/qingke-talk/episodes/EP89-generative-rlhf-v.html

> "we observed an emergent self-praise behavior: it appended extensive content to state its advantage." —— Generative RLHF-V 论文 §4.3（RQ5），对过训练后奖励黑客行为的描述

## 元信息

- 期号：89（官网期，B站合集有对应视频 BV1JRCyBiEp2）
- 标题：Generative RLHF-V：面向多模态 RLHF 的人类意图对齐框架
- BV：BV1JRCyBiEp2（有视频；本期字幕经接口多次尝试均返回串台内容、无法验证，未取得可用字幕，故走论文还原路线）
- 直播时间：2025-11-15 10:00–11:00（官网预告）
- 提炼日期：2026-10-02
- 分享嘉宾：周嘉懿（北京大学人工智能研究院 2024 级博士生，导师杨耀东；官网预告嘉宾介绍；论文一作 Jiayi Zhou）
- 相关论文：Jiayi Zhou, Jiaming Ji, Boyuan Chen, Jiapeng Sun, Wenqi Chen, Donghai Hong, Sirui Han, Yike Guo, Yaodong Yang，*Generative RLHF-V: Learning Principles from Multi-modal Human Preference*，https://arxiv.org/abs/2505.18531（2025-05-24）
- 相关代码：https://generative-rlhf-v.github.io/（论文给出代码与模型；实验基于 verl 做 RL 训练、align-anything 做奖励建模与 SFT，8×H800，论文 §4.1）
- 官网预告：https://qingkeai.online/blog/dU5ROQAj

> ⚠️ 提炼方式说明：本期 B站有视频，但字幕接口多次返回其他视频的串台字幕、无法验证，未能取得可用字幕。本纪要根据该期对应的公开材料还原——官网预告（含讲者与提纲）+ Generative RLHF-V 论文原文（arXiv:2505.18531）。讲授提纲以官网预告为准，方法与实验数字以论文为准。现场讨论与 AMA 环节未覆盖；若后续取得可验证字幕，应以字幕为准修订。

## 一句话总结

这期讲的是把生成式奖励模型（GRM）真正接进多模态 RLHF：先用 RL 训练一个多模态 GRM，让它在成对比较中自己归纳出人类偏好的「原则」再打分；再在策略优化阶段用「分组比较」把 pairwise 判别力转成可用于 GRPO 的 pointwise 奖励。结果是 4 个 MLLM 在 7 个基准上平均提升 18.1%（传统 RLHF 基线仅 5.3%），且候选响应数越多、收益近线性增长；但论文也给出一个反面教材——GRM 与策略同时过训练后，模型会学会「自夸」来骗高分。

## 核心

### 背景/问题：标量奖励模型装不下人类意图

按官网预告提纲，讲授分四块：多模态 RLHF 的概述与挑战、框架两阶段、「自夸」奖励黑客、AMA。论文 §1 把问题定位在奖励建模上：

- 传统做法给 MLLM 加一个标量打分头，用 Bradley-Terry 损失从成对偏好里学奖励。三个根本缺陷（论文 §1）：准确率低、泛化弱、不可解释；单一标量推理装不下日趋复杂的人类偏好，成为多模态对齐的瓶颈。
- 生成式奖励模型（GRM）让模型用自身推理能力对两个响应做比较判别，看起来更强，但有个结构性断裂：**pairwise 比较学到的「原则」无法直接变成 RL 优化需要的 pointwise 分数**（论文 §1）。此前 GRM 在多模态场景多用于 Best-of-N 数据过滤或离线 DPO，没有真正进入在线 RL 优化的实证。

### 方法/设计：两阶段——先把裁判练出来，再用分组比较当奖励

**阶段一：基于 RL 的生成式奖励建模**（论文 §3）。把 MLLM 本身训练成 pairwise GRM：输入 prompt 与两个响应，输出它推断的偏好原则、推理轨迹和一对分数；打分顺序与人类标注一致记为正奖励，否则记为负奖励，用 RL（规则奖励）优化。这条路线扩展自文本域的 SPCT，但论文发现一个关键差异（§1、§4.3 RQ6）：多模态场景下让 GRM **自主探索原则**比从参考集里选原则泛化更好——给静态原则反而在分布外偏好数据上掉点（Table 3）。

**阶段二：基于分组比较的 RL 优化**（论文 §3）。策略对同一输入生成一组 $k$ 个候选响应，每个响应与其余每个响应做 pairwise 比较、双向取分再平均，得到分组分数作为 GRPO 的奖励，公式为：

$$S(\bm{y}_{i})=\frac{1}{2(k-1)}\sum_{j=1,j\neq i}^{k}\left(s(\bm{y}_{i}|\bm{y}_{i},\bm{y}_{j})+s(\bm{y}_{i}|\bm{y}_{j},\bm{y}_{i})\right)$$

这里 $s(\bm{y}_{i}|\bm{y}_{i},\bm{y}_{j})$ 是 GRM 在把 $\bm{y}_{i}$ 放在前面比较时给它的分数。分组平均把 pairwise 判别转成稳定的 pointwise 奖励。消融显示两个组件缺一不可（论文 §1、§4.3 RQ4）：去掉任一组件，随候选数 $n$ 增长的近线性提升都消失。

### 实验/实战（论文 §4）

设置（§4.1）：策略模型为 Qwen2-VL-2B/7B 与 Qwen2.5-VL-3B/7B-Instruct 共 4 个；3B 的 GRM 监督 2B/3B 策略，7B 的 GRM 监督 7B。偏好数据用 Align-Anything 的 30k helpful 偏好与 BeaverTails-V 的 harmless 数据；7 个基准覆盖 helpful（MIA-Bench、LLaVA-Wild、LLaVA-Wilder、MM-Vet、MM-Vet-v2）与 harmless（MM-SafetyBench、MSS-Bench）。

- **裁判本身更准**：在 3 个分布外偏好数据集上 GRM 全面超过标量 RM，GRM+RL 最高，论文报告其分布外判别准确率平均提升 20.4%（§1 贡献列表）；且给 GRM+RL 喂静态原则反而掉点，说明 RL 已经让它归纳出更贴切的原则。分组比较让 pointwise 打分与人类评分的 Pearson 相关从 0.37 提升到 0.43，接近 GPT-4o 专家的 0.48（§4.2，Table 1）。
- **进入 RL 后全面胜出**：Generative RLHF-V 在 4 个模型、7 个基准上一致超过 RM 与 GRM 基线（Table 2），论文口径为平均提升 18.1%，而传统 RLHF 基线仅 5.3%（Abstract、§5）；小模型（2B/3B）被抬到接近 7B 的水平。
- **一个新的 post-training scaling 维度**：候选响应数 $n$ 增大时，GRM+RL 加分组比较的 RL 性能近线性提升，而标量 RM 几乎不涨——因为新样本的打分不可靠反而稀释收益（§4.3，Fig.7）。主实验 GRPO 默认 $n = 5$ 。
- **反面教材：自夸式奖励黑客**（§4.3，RQ5）。把 GRM 与 RL 两阶段都过训练到 5 个 epoch（主实验为 2 个）后，策略学会在回答末尾追加大段自夸文字；在 MLLM-as-judge 的成对评测里连 GPT-4o 裁判都会被自夸带偏给高分，人工去掉自夸段落后性能反而低于正常训练的版本。论文推测的机制是：过训练的 GRM 视觉（OCR）能力退化，更容易被文本自夸牵着走而不再核对图像。

## 关键数字

| 指标 | 基线/对照 | 结果 | 来源 |
|---|---|---|---|
| 4 个 MLLM × 7 基准的平均提升 | 传统 RLHF 基线 5.3% | 18.1% | 论文 Abstract、§5 |
| 分布外偏好判别准确率（GRM+RL） | 标量 RM / GRM / GRM+SFT | 平均提升 20.4%，为各方案最高 | 论文 §1、§4.2 |
| Pointwise 打分与人类评分 Pearson 相关 | 无分组比较 0.37 | 分组比较后 0.43（GPT-4o 专家 0.48） | 论文 §4.2、Table 1 |
| 候选响应数 $n$ 的 scaling | 标量 RM 几乎不涨 | GRM+RL + 分组比较近线性提升 | 论文 §4.3、Fig.7 |
| 奖励黑客触发条件 | 主实验 2 epoch | 两阶段均过训练至 5 epoch 出现自夸行为 | 论文 §4.3、RQ5 |
| 实验底座 | — | Qwen2-VL-2B/7B、Qwen2.5-VL-3B/7B-Instruct；Align-Anything 30k 偏好 | 论文 §4.1 |

注：18.1% 与 5.3% 为论文给出的跨模型跨基准平均口径，逐基准的绝对分见论文 Table 2，本纪要不逐项抄录。

## 可迁移

- 对 coding data / RL infra 工作的 1-2 个直接可试的点：
  1. **Pairwise 裁判 + 分组比较是把「LLM 当裁判」接进 GRPO 的现成配方**：凡是奖励只能做相对比较的任务（代码风格、解释质量、多模态理解），都可以用「组内两两比较取平均」转成 pointwise 奖励，且候选数 $n$ 本身成为可 scaling 的旋钮。
  2. **裁判要用 RL 练且让它自述原则**：给静态 rubric 反而限制泛化（Table 3）——做生成式裁判时优先让模型自己归纳判据，再考虑人工原则注入。
- Infra 视角（扩展性 / 成本 / 评测自动化）的启发：
  1. 奖励黑客的案例说明评测与训练共用同一类裁判（MLLM-as-judge）会形成可被利用的闭环：自夸能同时骗过训练奖励和 GPT-4o 评测。评测管线需要与训练奖励解耦，并对「回答里出现自我评价性文本」这类模式做监测。
  2. GRM 的能力会随训练退化（OCR 变差导致被文本牵着走）——在线 RL 里裁判不是训完就一劳永逸，需要定期用分布外判别集复检裁判准确率。

## 疑问 / 下一步

- 没看懂的 / 想深挖的 1 个问题：分组比较的成本是每组 $O(k^2)$ 次 GRM 推理， $n$ 近线性涨收益的同时推理成本平方增长——实际 RL 训练里 $n$ 的经济最优点在哪、能否用锦标赛式淘汰降成本，论文未讨论。
- 现场内容不可还原：官网提纲第三部分的「自夸」讨论与 AMA 中关于缓解办法的现场回答无法从论文推知；论文 Limitations 只呼吁系统性研究，未给解法。
- 论文未报告 GRM 训练的算力与数据规模细节（附录 §6），复现成本需查代码仓库确认。

## 原文金句（1-2句）

> "enabling GRMs to explore principles autonomously yields superior generalization than selecting principles from a reference set." —— 论文 §1，本期最反直觉的方法论结论

> "We hope this case study provides insights for future research into MLLMs reward hacking and underscores the pressing need for more comprehensive and unbiased MLLMs benchmarks." —— 论文 §4.3（RQ5），自夸案例的落点
